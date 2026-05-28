"""
core/tracker.py
─────────────────────────────────────────────────────────────────────────────
ByteTrack multi-person tracker wrapper.

Assigns consistent track IDs across frames so that:
  - Worker A always has person_id=1 throughout the video
  - Worker B always has person_id=2, etc.
  - Re-identification works after occlusions (up to track_buffer frames)

This is critical for multi-worker industrial scenes.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

from core.pose_extractor import PersonPose

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    ByteTrack-based multi-person tracker using supervision library.

    Assigns stable person_id values to PersonPose objects across frames.

    Usage:
        tracker = PersonTracker(config)
        tracker.initialize()
        for frame_poses in all_frame_poses:
            tracked = tracker.update(frame_poses, frame)
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.track_thresh = config.get("track_thresh", 0.5)
        self.track_buffer = config.get("track_buffer", 30)
        self.match_thresh = config.get("match_thresh", 0.8)
        self.frame_rate = config.get("frame_rate", 30)
        self.tracker = None
        self._id_history: Dict[int, List[int]] = {}  # track_id → [frame_ids]

    def initialize(self) -> None:
        """Initialize the ByteTrack tracker — compatible with all supervision versions."""
        if not self.enabled:
            logger.info("Tracking disabled — person IDs will be index-based")
            return

        try:
            import supervision as sv

            # ── Version-safe instantiation ────────────────────────────────────
            # supervision >= 0.22: ByteTracker was renamed to ByteTracker (same)
            # but constructor args changed across versions — try both signatures.
            tracker_cls = None

            if hasattr(sv, "ByteTracker"):
                tracker_cls = sv.ByteTracker
            elif hasattr(sv, "byte_tracker") and hasattr(sv.byte_tracker, "ByteTracker"):
                tracker_cls = sv.byte_tracker.ByteTracker
            else:
                raise AttributeError("ByteTracker not found in supervision library")

            # Try new-style constructor first (supervision >= 0.21)
            try:
                self.tracker = tracker_cls(
                    track_activation_threshold=self.track_thresh,
                    lost_track_buffer=self.track_buffer,
                    minimum_matching_threshold=self.match_thresh,
                    frame_rate=self.frame_rate,
                )
            except TypeError:
                # Fallback: old-style constructor (supervision <= 0.20)
                self.tracker = tracker_cls(
                    track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh,
                    frame_rate=self.frame_rate,
                )

            import supervision as sv_check
            logger.info(f"ByteTrack initialized (supervision=={sv_check.__version__})")

        except ImportError:
            logger.warning(
                "supervision not installed. Falling back to index-based IDs.\n"
                "Install: pip install supervision>=0.20.0"
            )
            self.enabled = False
        except AttributeError as e:
            logger.warning(f"Tracker init failed: {e} — falling back to index IDs")
            self.enabled = False

    def update(
        self,
        poses: List[PersonPose],
        frame: np.ndarray,
    ) -> List[PersonPose]:
        """
        Assign track IDs to detected poses.

        Args:
            poses: List of PersonPose from PoseExtractor (person_id=-1)
            frame: Current BGR frame (needed by supervision tracker)

        Returns:
            Same poses with person_id filled in
        """
        if not poses:
            return poses

        if not self.enabled or self.tracker is None:
            # Fallback: assign sequential IDs by detection order
            for i, pose in enumerate(poses):
                pose.person_id = i
            return poses

        try:
            import supervision as sv

            # Build detection array for supervision
            # Format: [x1, y1, x2, y2, confidence, class_id]
            h, w = frame.shape[:2]
            xyxy = np.array([
                [
                    pose.bbox[0] * w, pose.bbox[1] * h,
                    pose.bbox[2] * w, pose.bbox[3] * h,
                ]
                for pose in poses
            ], dtype=np.float32)

            confidence = np.array([pose.det_confidence for pose in poses])
            class_ids = np.zeros(len(poses), dtype=int)

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_ids,
            )

            tracked = self.tracker.update_with_detections(detections)

            # Map tracked detections back to poses by IOU matching
            if len(tracked) > 0:
                tracked_ids = tracked.tracker_id
                tracked_xyxy = tracked.xyxy

                for i, pose in enumerate(poses):
                    # Find best matching tracked detection
                    pose_box = np.array([
                        pose.bbox[0] * w, pose.bbox[1] * h,
                        pose.bbox[2] * w, pose.bbox[3] * h,
                    ])
                    ious = self._batch_iou(pose_box, tracked_xyxy)
                    if len(ious) > 0 and ious.max() > 0.3:
                        best_idx = ious.argmax()
                        track_id = int(tracked_ids[best_idx])
                        pose.person_id = track_id
                        if track_id not in self._id_history:
                            self._id_history[track_id] = []
                        self._id_history[track_id].append(pose.frame_id)
                    else:
                        pose.person_id = i  # Unmatched: use index

        except Exception as e:
            logger.warning(f"Tracker update failed: {e} — using index IDs")
            for i, pose in enumerate(poses):
                pose.person_id = i

        return poses

    @staticmethod
    def _batch_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IOU between one box and array of boxes."""
        if len(boxes) == 0:
            return np.array([])

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        return inter / (union + 1e-6)

    def get_active_tracks(self) -> List[int]:
        """Returns list of all track IDs seen so far."""
        return list(self._id_history.keys())

    def get_track_duration(self, track_id: int) -> int:
        """Returns number of frames a track has been active."""
        return len(self._id_history.get(track_id, []))
