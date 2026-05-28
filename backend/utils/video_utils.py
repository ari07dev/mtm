"""
utils/video_utils.py
─────────────────────────────────────────────────────────────────────────────
Video I/O utilities: frame extraction, metadata, resizing, progress tracking.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    file_size_mb: float

    def __str__(self) -> str:
        return (
            f"Video: {Path(self.path).name}\n"
            f"  Resolution: {self.width}x{self.height}\n"
            f"  FPS: {self.fps:.2f}  |  Frames: {self.total_frames}\n"
            f"  Duration: {self.duration_seconds:.1f}s\n"
            f"  Codec: {self.codec}  |  Size: {self.file_size_mb:.1f}MB"
        )


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from video file without reading frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec  = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        size   = Path(video_path).stat().st_size / (1024 * 1024)

        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps if fps > 0 else 30.0,
            total_frames=total,
            duration_seconds=total / fps if fps > 0 else 0,
            codec=codec.strip(),
            file_size_mb=round(size, 2),
        )
    finally:
        cap.release()


def frame_generator(
    video_path: str,
    config: dict,
    max_frames: int = 0,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Generator that yields (frame_id, timestamp, frame) tuples.

    Args:
        video_path: Path to video file
        config: Video config dict (target_fps, resize_width, skip_frames)
        max_frames: Max frames to yield (0 = all)

    Yields:
        (frame_id, timestamp_seconds, bgr_frame)
    """
    meta = get_video_metadata(video_path)
    logger.info(str(meta))

    target_fps     = config.get("target_fps", 30)
    resize_w       = config.get("resize_width", 0)
    resize_h       = config.get("resize_height", 0)
    skip_frames    = config.get("skip_frames", 0)

    # Calculate frame sampling ratio
    sample_ratio = max(1, round(meta.fps / target_fps)) if target_fps > 0 else 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_id = 0
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for FPS downsampling
            if frame_id % sample_ratio != 0:
                frame_id += 1
                continue

            # Skip additional frames if configured
            if skip_frames > 0 and frame_id % (skip_frames + 1) != 0:
                frame_id += 1
                continue

            # Resize if configured
            if resize_w > 0 and resize_h > 0:
                frame = cv2.resize(frame, (resize_w, resize_h))

            timestamp = frame_id / meta.fps
            yield frame_id, timestamp, frame

            yielded += 1
            frame_id += 1

            if max_frames > 0 and yielded >= max_frames:
                logger.info(f"Reached max_frames limit: {max_frames}")
                break

    finally:
        cap.release()
        logger.info(f"Video processing complete: {yielded} frames processed")


class VideoWriter:
    """Context manager for writing debug output video."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.writer: Optional[cv2.VideoWriter] = None

    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video: {self.output_path}")
        return self

    def write(self, frame: np.ndarray) -> None:
        if self.writer:
            self.writer.write(frame)

    def __exit__(self, *args):
        if self.writer:
            self.writer.release()
        logger.info(f"Debug video saved: {self.output_path}")
