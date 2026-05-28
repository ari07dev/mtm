"""
core/skeleton_builder.py
─────────────────────────────────────────────────────────────────────────────
Skeleton Sequence Builder — converts frame-by-frame PersonPose objects
into temporal skeleton sequences ready for ST-GCN input.

Key operations:
1. Accumulate keypoints across frames (sliding window)
2. Normalize coordinates relative to body center
3. Interpolate missing/low-conf keypoints
4. Apply Gaussian smoothing on trajectories
5. Output (T, V, C) tensors — T=time, V=vertices/joints, C=channels(x,y,conf)

This is the "Skeleton Sequence (time series data)" block in your pipeline diagram.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from core.pose_extractor import PersonPose, KEYPOINT_NAMES

logger = logging.getLogger(__name__)

# ST-GCN graph edges (COCO 17-point skeleton)
ST_GCN_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NUM_JOINTS = 17
CHANNELS = 3  # x, y, confidence


@dataclass
class SkeletonWindow:
    """
    A single sliding window of skeleton data for one person.
    Shape: (window_size, NUM_JOINTS, CHANNELS)

    This is the direct input format for ST-GCN.
    """
    person_id: int
    window_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    data: np.ndarray      # (T, 17, 3) — normalized keypoints
    valid_frames: int     # How many frames had valid detection
    completeness: float   # valid_frames / window_size

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "window_id": self.window_id,
            "frames": [self.start_frame, self.end_frame],
            "time": [round(self.start_time, 3), round(self.end_time, 3)],
            "shape": list(self.data.shape),
            "completeness": round(self.completeness, 3),
            "data": self.data.tolist(),   # For JSON export
        }

    def to_tensor(self) -> np.ndarray:
        """Returns (T, V, C) array ready for ST-GCN."""
        return self.data

    def get_joint_trajectory(self, joint_idx: int) -> np.ndarray:
        """
        Returns (T, 3) trajectory of a single joint over the window.
        Useful for rule-based analysis (e.g., ankle trajectory for walk detection).
        """
        return self.data[:, joint_idx, :]  # (T, 3)

    def get_joint_velocity(self, joint_idx: int) -> np.ndarray:
        """Returns (T-1, 2) velocity [dx, dy] of a joint."""
        traj = self.data[:, joint_idx, :2]  # (T, 2)
        return np.diff(traj, axis=0)        # (T-1, 2)


class SkeletonBuilder:
    """
    Accumulates PersonPose frames and emits SkeletonWindows.

    Usage:
        builder = SkeletonBuilder(config)
        builder.initialize()

        for frame_poses in all_poses:
            windows = builder.update(frame_poses)
            for window in windows:
                # Feed window to classifier or ST-GCN

        # Flush remaining frames at end of video
        final_windows = builder.flush()
    """

    def __init__(self, config: dict):
        self.window_size = config.get("window_size", 30)
        self.stride = config.get("window_stride", 15)
        self.normalize = config.get("normalize", True)
        self.interpolate = config.get("interpolate_missing", True)
        self.smooth_sigma = config.get("smoothing_sigma", 1.5)

        # Per-person frame buffer: person_id → deque of (frame_id, kps_array)
        self._buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size * 2)
        )
        self._window_counters: Dict[int, int] = defaultdict(int)
        self._frame_counter = 0
        self._ready_windows: List[SkeletonWindow] = []

    def initialize(self) -> None:
        logger.info(
            f"SkeletonBuilder initialized: window={self.window_size} "
            f"stride={self.stride} normalize={self.normalize}"
        )

    def update(self, frame_poses: List[PersonPose]) -> List[SkeletonWindow]:
        """
        Process one frame of poses. Returns any completed windows.

        Args:
            frame_poses: Tracked PersonPose list for current frame

        Returns:
            List of completed SkeletonWindow objects (may be empty)
        """
        self._frame_counter += 1
        completed = []

        for pose in frame_poses:
            pid = pose.person_id
            kps = pose.get_keypoints_array()  # (17, 3)

            self._buffers[pid].append({
                "frame_id": pose.frame_id,
                "timestamp": pose.timestamp,
                "keypoints": kps,
            })

            # Emit window when we have enough frames (every stride frames)
            buf = self._buffers[pid]
            if (len(buf) >= self.window_size and
                    self._frame_counter % self.stride == 0):
                window = self._build_window(pid, list(buf)[-self.window_size:])
                if window is not None:
                    completed.append(window)

        return completed

    def flush(self) -> List[SkeletonWindow]:
        """
        Emit windows for all remaining buffered data (call at end of video).
        Uses partial windows if buffer has fewer than window_size frames.
        """
        completed = []
        for pid, buf in self._buffers.items():
            if len(buf) >= max(8, self.window_size // 4):  # Min viable window
                frames = list(buf)[-self.window_size:]
                window = self._build_window(pid, frames, is_final=True)
                if window is not None:
                    completed.append(window)
        return completed

    def _build_window(
        self,
        person_id: int,
        frames: List[dict],
        is_final: bool = False,
    ) -> Optional[SkeletonWindow]:
        """
        Build a SkeletonWindow from a list of frame dicts.

        Steps:
            1. Stack keypoints into (T, 17, 3) array
            2. Pad if window is shorter than window_size
            3. Normalize relative to body center
            4. Interpolate missing keypoints
            5. Apply Gaussian smoothing
        """
        if not frames:
            return None

        T_actual = len(frames)
        T = self.window_size

        # Stack into array (T_actual, 17, 3)
        kps_stack = np.array([f["keypoints"] for f in frames])  # (T, 17, 3)

        # Count valid frames (where detection exists)
        valid_mask = kps_stack[:, 0, 2] > 0.1  # Nose confidence as proxy
        valid_frames = int(valid_mask.sum())

        # Pad to window_size if needed
        if T_actual < T:
            pad = np.zeros((T - T_actual, NUM_JOINTS, CHANNELS))
            kps_stack = np.concatenate([kps_stack, pad], axis=0)

        # ── Step 1: Normalize ────────────────────────────────────────────────
        if self.normalize:
            kps_stack = self._normalize(kps_stack)

        # ── Step 2: Interpolate missing keypoints ────────────────────────────
        if self.interpolate:
            kps_stack = self._interpolate(kps_stack)

        # ── Step 3: Gaussian smoothing ───────────────────────────────────────
        if self.smooth_sigma > 0:
            # Smooth x and y channels only, not confidence
            kps_stack[:, :, 0] = gaussian_filter1d(
                kps_stack[:, :, 0], sigma=self.smooth_sigma, axis=0
            )
            kps_stack[:, :, 1] = gaussian_filter1d(
                kps_stack[:, :, 1], sigma=self.smooth_sigma, axis=0
            )

        wid = self._window_counters[person_id]
        self._window_counters[person_id] += 1

        return SkeletonWindow(
            person_id=person_id,
            window_id=wid,
            start_frame=frames[0]["frame_id"],
            end_frame=frames[-1]["frame_id"],
            start_time=frames[0]["timestamp"],
            end_time=frames[-1]["timestamp"],
            data=kps_stack.astype(np.float32),
            valid_frames=valid_frames,
            completeness=valid_frames / T,
        )

    def _normalize(self, kps: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints relative to body center (hip midpoint).
        Also scales by torso height for scale invariance.

        Args:
            kps: (T, 17, 3) array

        Returns:
            (T, 17, 3) normalized array
        """
        kps = kps.copy()

        for t in range(len(kps)):
            lh = kps[t, 11, :2]  # left hip
            rh = kps[t, 12, :2]  # right hip
            ls = kps[t, 5, :2]   # left shoulder
            rs = kps[t, 6, :2]   # right shoulder

            lh_conf = kps[t, 11, 2]
            rh_conf = kps[t, 12, 2]

            # Body center (hip midpoint)
            if lh_conf > 0.1 and rh_conf > 0.1:
                center = (lh + rh) / 2
            else:
                center = np.array([0.5, 0.5])

            # Torso height for scale normalization
            shoulder_center = (ls + rs) / 2
            torso_height = np.linalg.norm(shoulder_center - center)
            scale = torso_height if torso_height > 0.01 else 0.2

            # Apply normalization to xy only
            kps[t, :, :2] = (kps[t, :, :2] - center) / scale

        return kps

    def _interpolate(self, kps: np.ndarray) -> np.ndarray:
        """
        Linearly interpolate low-confidence keypoints from neighbors.
        Helps when a joint is occluded for a few frames.

        Args:
            kps: (T, 17, 3) array

        Returns:
            (T, 17, 3) with interpolated values where conf was low
        """
        kps = kps.copy()
        T = len(kps)

        for j in range(NUM_JOINTS):
            conf = kps[:, j, 2]
            low_conf_mask = conf < 0.3

            if low_conf_mask.all():
                continue  # All frames invalid, can't interpolate

            # Find valid frame indices
            valid_indices = np.where(~low_conf_mask)[0]

            for t in np.where(low_conf_mask)[0]:
                # Find nearest valid frames before and after
                before = valid_indices[valid_indices < t]
                after = valid_indices[valid_indices > t]

                if len(before) == 0 and len(after) == 0:
                    continue
                elif len(before) == 0:
                    kps[t, j, :2] = kps[after[0], j, :2]
                elif len(after) == 0:
                    kps[t, j, :2] = kps[before[-1], j, :2]
                else:
                    t0, t1 = before[-1], after[0]
                    alpha = (t - t0) / (t1 - t0)
                    kps[t, j, :2] = (
                        (1 - alpha) * kps[t0, j, :2] +
                        alpha * kps[t1, j, :2]
                    )
                # Mark as low confidence still (don't fake high conf)
                kps[t, j, 2] = 0.3

        return kps

    def get_buffer_status(self) -> dict:
        """Returns current buffer sizes per person."""
        return {pid: len(buf) for pid, buf in self._buffers.items()}
