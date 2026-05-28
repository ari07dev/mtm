"""
core/pose_extractor.py
─────────────────────────────────────────────────────────────────────────────
YOLOv8-Pose wrapper for per-frame keypoint extraction.

ST-GCN Compatible Version
- keypoint coordinates stored separately from confidence
- avoids 34 vs 51 dimensional mismatch
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-keypoint names
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton visualization connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class Keypoint:
    """Single keypoint."""

    x: float
    y: float
    confidence: float
    name: str = ""
    valid: bool = True

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "conf": round(self.confidence, 4),
            "valid": self.valid,
        }

    def to_xy(self) -> np.ndarray:
        """Return only x,y."""
        return np.array([self.x, self.y], dtype=np.float32)

    def to_array(self) -> np.ndarray:
        """Return x,y,conf."""
        return np.array(
            [self.x, self.y, self.confidence],
            dtype=np.float32
        )


@dataclass
class PersonPose:
    """Pose of one person in one frame."""

    person_id: int
    frame_id: int
    timestamp: float

    bbox: Tuple[float, float, float, float, float]

    keypoints: List[Keypoint]

    det_confidence: float = 0.0
    frame_width: int = 0
    frame_height: int = 0

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None

    # ------------------------------------------------------------------
    # ST-GCN COMPATIBLE METHODS
    # ------------------------------------------------------------------

    def get_xy_array(self) -> np.ndarray:
        """
        Returns:
            (17,2) → x,y only
        """
        return np.array(
            [[kp.x, kp.y] for kp in self.keypoints],
            dtype=np.float32
        )

    def get_conf_array(self) -> np.ndarray:
        """
        Returns:
            (17,) confidence values
        """
        return np.array(
            [kp.confidence for kp in self.keypoints],
            dtype=np.float32
        )

    def get_keypoints_array(self) -> np.ndarray:
        """
        Returns:
            (17,3) → x,y,conf
        """
        return np.array(
            [kp.to_array() for kp in self.keypoints],
            dtype=np.float32
        )

    # ------------------------------------------------------------------

    def body_center(self) -> Tuple[float, float]:
        """Midpoint between hips."""

        lh = self.keypoints[11]
        rh = self.keypoints[12]

        if lh.valid and rh.valid:
            return (
                (lh.x + rh.x) / 2,
                (lh.y + rh.y) / 2
            )

        ls = self.keypoints[5]
        rs = self.keypoints[6]

        return (
            (ls.x + rs.x) / 2,
            (ls.y + rs.y) / 2
        )

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 4),
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
                "conf": self.bbox[4],
            },
            "det_confidence": round(self.det_confidence, 4),
            "keypoints": {
                kp.name: kp.to_dict()
                for kp in self.keypoints
            },
        }


class PoseExtractor:
    """
    YOLOv8 Pose extractor.
    """

    def __init__(self, config: dict):

        self.config = config

        self.model = None

        self.model_name = config.get(
            "model",
            "yolov8m-pose.pt"
        )

        self.conf_threshold = config.get(
            "confidence",
            0.5
        )

        self.iou_threshold = config.get(
            "iou_threshold",
            0.7
        )

        self.kp_conf_threshold = config.get(
            "keypoint_conf_threshold",
            0.5
        )

        self.max_persons = config.get(
            "max_persons",
            5
        )

        self.device = config.get(
            "device",
            "auto"
        )

    # ------------------------------------------------------------------

    def load(self) -> None:

        try:
            from ultralytics import YOLO

            logger.info(
                f"Loading model: {self.model_name}"
            )

            self.model = YOLO(self.model_name)

            if self.device == "auto":

                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"

                elif (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    self.device = "mps"

                else:
                    self.device = "cpu"

            logger.info(
                f"Using device: {self.device}"
            )

        except ImportError:
            raise ImportError(
                "Install ultralytics:\n"
                "pip install ultralytics"
            )

    # ------------------------------------------------------------------

    def extract(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> List[PersonPose]:

        if self.model is None:
            raise RuntimeError(
                "Model not loaded."
            )

        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        poses: List[PersonPose] = []

        for result in results:

            if (
                result.keypoints is None
                or result.boxes is None
            ):
                continue

            # ----------------------------------------------------------
            # IMPORTANT FIX
            #
            # xy      -> (N,17,2)
            # conf    -> (N,17)
            
            
            
            # because that returns:
            # (x,y,conf) -> causes 51-dim issue
            # ----------------------------------------------------------

            kps_xy = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            kps_conf = (
                result.keypoints.conf
                .cpu()
                .numpy()
            )

            boxes_data = (
                result.boxes.data
                .cpu()
                .numpy()
            )

            n_persons = min(
                len(kps_xy),
                self.max_persons
            )

            for i in range(n_persons):

                kps = kps_xy[i]          # (17,2)
                kp_scores = kps_conf[i]  # (17,)
                box = boxes_data[i]

                # Normalize bbox
                x1, y1, x2, y2, det_conf = (
                    box[0] / w,
                    box[1] / h,
                    box[2] / w,
                    box[3] / h,
                    float(box[4]),
                )

                keypoints: List[Keypoint] = []

                for j, ((kx, ky), kc) in enumerate(
                    zip(kps, kp_scores)
                ):

                    kp = Keypoint(
                        x=float(kx) / w,
                        y=float(ky) / h,
                        confidence=float(kc),
                        name=KEYPOINT_NAMES[j],
                        valid=float(kc)
                        >= self.kp_conf_threshold,
                    )

                    keypoints.append(kp)

                pose = PersonPose(
                    person_id=-1,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=(
                        x1,
                        y1,
                        x2,
                        y2,
                        det_conf,
                    ),
                    keypoints=keypoints,
                    det_confidence=det_conf,
                    frame_width=w,
                    frame_height=h,
                )

                poses.append(pose)

        logger.debug(
            f"Frame {frame_id}: "
            f"{len(poses)} persons"
        )

        # --------------------------------------------------------------
        # Primary worker selection
        # --------------------------------------------------------------

        if (
            self.max_persons == 1
            and len(poses) > 1
        ):

            def bbox_area(p):
                return (
                    (p.bbox[2] - p.bbox[0]) *
                    (p.bbox[3] - p.bbox[1])
                )

            poses = [
                max(poses, key=bbox_area)
            ]

            logger.debug(
                "Primary worker selected"
            )

        return poses

    # ------------------------------------------------------------------

    def extract_batch(
        self,
        frames: List[np.ndarray],
        start_frame_id: int,
        fps: float,
    ) -> List[List[PersonPose]]:

        results_batch = []

        for i, frame in enumerate(frames):

            frame_id = start_frame_id + i

            timestamp = frame_id / fps

            poses = self.extract(
                frame,
                frame_id,
                timestamp,
            )

            results_batch.append(poses)

        return results_batch