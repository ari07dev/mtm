"""
utils/visualizer.py
─────────────────────────────────────────────────────────────────────────────
Debug visualization: draws skeleton overlay + MTM labels on video frames.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.pose_extractor import PersonPose, SKELETON_CONNECTIONS
from core.action_classifier import ClassificationResult


# Color palette (BGR)
COLOR_HEAD       = (11, 158, 245)   # Amber
COLOR_HANDS      = (94, 197,  34)   # Green
COLOR_FEET       = (246, 130,  59)  # Blue
COLOR_BODY       = (200, 200, 200)  # White-ish
COLOR_BONE       = (80,  80,  80)   # Dark gray
COLOR_MTM_BG     = (20,  20,  20)
COLOR_MTM_TEXT   = (11, 158, 245)   # Amber
COLOR_TRACK_ID   = (255, 255,   0)  # Yellow
COLOR_BBOX       = (50,  50,  50)

HAND_JOINT_IDS   = {9, 10}
FOOT_JOINT_IDS   = {15, 16}
HEAD_JOINT_IDS   = {0, 1, 2, 3, 4}


def get_joint_color(joint_idx: int) -> Tuple[int, int, int]:
    if joint_idx in HEAD_JOINT_IDS:
        return COLOR_HEAD
    if joint_idx in HAND_JOINT_IDS:
        return COLOR_HANDS
    if joint_idx in FOOT_JOINT_IDS:
        return COLOR_FEET
    return COLOR_BODY


class Visualizer:
    """
    Draws skeleton + MTM overlays on frames.

    Usage:
        vis = Visualizer(config)
        annotated = vis.draw(frame, poses, current_mtm_code)
    """

    def __init__(self, config: dict):
        self.show_skeleton    = config.get("show_skeleton", True)
        self.show_bbox        = config.get("show_bbox", True)
        self.show_kp_conf     = config.get("show_keypoint_conf", False)
        self.show_mtm_label   = config.get("show_mtm_label", True)
        self.show_track_id    = config.get("show_track_id", True)
        self.font_scale       = config.get("font_scale", 0.6)
        self.line_thickness   = config.get("line_thickness", 2)

    def draw(
        self,
        frame: np.ndarray,
        poses: List[PersonPose],
        current_mtm: Optional[str] = None,
        frame_id: int = 0,
        timestamp: float = 0.0,
        confidence: float = 0.0,
    ) -> np.ndarray:
        """
        Annotate a frame with all pose and MTM information.

        Returns:
            Annotated BGR frame (does not modify original)
        """
        out = frame.copy()
        h, w = out.shape[:2]

        for pose in poses:
            self._draw_person(out, pose, w, h)

        if self.show_mtm_label and current_mtm:
            self._draw_mtm_banner(out, current_mtm, confidence, frame_id, timestamp, w, h)

        self._draw_pipeline_indicator(out, w, h)

        return out

    def _draw_person(
        self,
        frame: np.ndarray,
        pose: PersonPose,
        w: int,
        h: int,
    ) -> None:
        kps = pose.keypoints

        # Bounding box
        if self.show_bbox:
            x1 = int(pose.bbox[0] * w)
            y1 = int(pose.bbox[1] * h)
            x2 = int(pose.bbox[2] * w)
            y2 = int(pose.bbox[3] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 1)

        # Track ID
        if self.show_track_id and pose.person_id >= 0:
            label = f"ID:{pose.person_id}"
            x1 = int(pose.bbox[0] * w)
            y1 = int(pose.bbox[1] * h)
            cv2.putText(
                frame, label, (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7,
                COLOR_TRACK_ID, 1, cv2.LINE_AA
            )

        # Skeleton bones
        if self.show_skeleton:
            for a, b in SKELETON_CONNECTIONS:
                ka, kb = kps[a], kps[b]
                if ka.valid and kb.valid:
                    pt_a = (int(ka.x * w), int(ka.y * h))
                    pt_b = (int(kb.x * w), int(kb.y * h))
                    cv2.line(frame, pt_a, pt_b, COLOR_BONE, self.line_thickness, cv2.LINE_AA)

            # Keypoint circles
            for i, kp in enumerate(kps):
                if not kp.valid:
                    continue
                px = int(kp.x * w)
                py = int(kp.y * h)
                color = get_joint_color(i)
                radius = 6 if i in HAND_JOINT_IDS | FOOT_JOINT_IDS else 4
                cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

                if self.show_kp_conf:
                    cv2.putText(
                        frame, f"{kp.confidence:.1f}",
                        (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (180, 180, 180), 1, cv2.LINE_AA
                    )

    def _draw_mtm_banner(
        self,
        frame: np.ndarray,
        mtm_code: str,
        confidence: float,
        frame_id: int,
        timestamp: float,
        w: int,
        h: int,
    ) -> None:
        """Draw the current MTM code banner at top of frame."""
        banner_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # MTM Code (large)
        cv2.putText(
            frame, mtm_code, (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 1.1,
            COLOR_MTM_TEXT, 2, cv2.LINE_AA
        )

        # Confidence bar
        bar_w = int(180 * confidence)
        cv2.rectangle(frame, (w - 200, 8), (w - 20, 22), (40, 40, 40), -1)
        bar_color = (34, 197, 94) if confidence > 0.8 else (245, 158, 11) if confidence > 0.6 else (68, 68, 239)
        cv2.rectangle(frame, (w - 200, 8), (w - 200 + bar_w, 22), bar_color, -1)
        cv2.putText(
            frame, f"{confidence*100:.0f}%",
            (w - 200, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (180, 180, 180), 1, cv2.LINE_AA
        )

        # Frame + time info
        info = f"F:{frame_id:04d}  T:{timestamp:.2f}s"
        cv2.putText(
            frame, info, (w - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (100, 100, 100), 1, cv2.LINE_AA
        )

    def _draw_pipeline_indicator(self, frame: np.ndarray, w: int, h: int) -> None:
        """Draw small pipeline label at bottom."""
        label = "YOLOv8-POSE | BYTETRACK | SKELETON-BUILDER | MTM-PIPELINE v1"
        cv2.putText(
            frame, label, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (60, 60, 60), 1, cv2.LINE_AA
        )
