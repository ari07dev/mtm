"""
phase2/core/stgcn_inference.py
─────────────────────────────────────────────────────────────────────────────
ST-GCN inference engine using MMAction2 pre-trained model.

Takes SkeletonWindow (T, 17, 3) from Phase 1 skeleton builder
→ runs ST-GCN inference
→ returns NTU-60 class probabilities
→ maps to MTM codes via config mapping

CPU-optimized: async inference so video processing isn't blocked.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# NTU-RGB+D 60 class names (index → label)
NTU60_CLASSES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop",          # 0-4
    "pick up", "throw", "sit down", "stand up", "applaud",                   # 5-9 (9=walking in some splits)
    "reading", "writing", "tear up paper", "put on jacket", "take off jacket", # 10-14
    "put on shoe", "take off shoe", "put on glasses", "take off glasses",    # 15-18
    "put on hat/cap", "take off hat/cap", "cheer up", "hand waving",        # 19-22 (22=hand wave)
    "kicking something", "reach into pocket", "hopping", "jump up",         # 23-26
    "phone call", "play with phone", "type on keyboard", "point to something", # 27-30 (28=pick up)
    "taking a selfie", "check time", "rub two hands", "nod head/bow",       # 31-34
    "shake head", "wipe face", "salute", "put palms together",              # 35-38
    "cross hands in front", "sneeze/cough", "staggering", "falling down",   # 39-42
    "headache", "chest pain", "back pain", "neck pain",                     # 43-46
    "nausea/vomiting", "fan self", "punch/slap", "kicking",                 # 47-50
    "pushing", "pat on back", "point finger", "hugging",                    # 51-54
    "giving object", "touch pocket", "shaking hands", "walking towards",    # 55-58
    "walking apart",                                                          # 59
]


@dataclass
class STGCNResult:
    """Output from ST-GCN inference for one skeleton window."""
    window_id: int
    person_id: int
    start_time: float
    end_time: float
    class_scores: np.ndarray      # (60,) softmax probabilities
    top_k_classes: List[int]      # Top-K class indices
    top_k_scores: List[float]     # Top-K scores
    mtm_code: str                 # Mapped MTM code
    mtm_confidence: float         # Confidence in MTM mapping
    inference_time_ms: float

    @property
    def top_class_name(self) -> str:
        if self.top_k_classes:
            idx = self.top_k_classes[0]
            return NTU60_CLASSES[idx] if idx < len(NTU60_CLASSES) else f"class_{idx}"
        return "unknown"

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "person_id": self.person_id,
            "time": [round(self.start_time, 3), round(self.end_time, 3)],
            "top_prediction": self.top_class_name,
            "top_score": round(float(self.top_k_scores[0]), 3) if self.top_k_scores else 0,
            "mtm_code": self.mtm_code,
            "mtm_confidence": round(self.mtm_confidence, 3),
            "inference_ms": round(self.inference_time_ms, 1),
        }


class MTMMappingEngine:
    """
    Maps ST-GCN NTU-60 class predictions → MTM codes.

    Uses priority-ordered mapping from stgcn_config.yaml.
    Considers top-K predictions, not just top-1.
    """

    def __init__(self, mapping_config: dict):
        self.mapping = mapping_config
        self.top_k = 3
        # Build reverse lookup: ntu_class_idx → list of (mtm_code, min_conf)
        self._reverse: Dict[int, List[Tuple[str, float]]] = {}
        for mtm_code, cfg in self.mapping.items():
            for cls_idx in cfg.get("ntu_classes", []):
                if cls_idx not in self._reverse:
                    self._reverse[cls_idx] = []
                self._reverse[cls_idx].append((mtm_code, cfg.get("min_confidence", 0.2)))

    def map(
        self,
        class_scores: np.ndarray,
        top_k: int = 3,
    ) -> Tuple[str, float]:
        """
        Map class score vector to MTM code.

        Args:
            class_scores: (N,) softmax probabilities from ST-GCN
            top_k: Consider top-K predictions

        Returns:
            (mtm_code, confidence)
        """
        top_indices = np.argsort(class_scores)[::-1][:top_k]
        top_scores = class_scores[top_indices]

        # Walk through top predictions in order
        for idx, score in zip(top_indices, top_scores):
            idx = int(idx)
            if idx in self._reverse:
                for mtm_code, min_conf in self._reverse[idx]:
                    if score >= min_conf:
                        # Scale MTM confidence based on class score
                        mtm_conf = min(score * 1.2, 0.95)
                        return mtm_code, mtm_conf

        # No mapping found — return highest scoring mapped class or fallback
        return "GRASP + HOLD OBJECT", 0.30  # Most common automobile action

    def get_walk_mtm(self, step_count: int) -> str:
        """Get WALK MTM code from step count (from skeleton builder)."""
        if step_count <= 4:
            return "WALK 1-4 STEPS"
        elif step_count <= 7:
            return "WALK 5-7 STEPS"
        elif step_count <= 10:
            return "WALK 8-10 STEPS"
        elif step_count <= 15:
            return "WALK 11-15 STEPS"
        else:
            return "WALK 16-30 STEPS"

    def is_walk_prediction(self, top_indices: List[int], top_scores: List[float]) -> bool:
        """Check if top predictions indicate walking."""
        walk_classes = {9, 10, 57, 58, 59}  # NTU walk-related classes
        for idx, score in zip(top_indices, top_scores):
            if int(idx) in walk_classes and score > 0.20:
                return True
        return False


class STGCNInferenceEngine:
    """
    MMAction2 ST-GCN inference wrapper.

    Supports:
    - Synchronous inference (simple, blocks video processing)
    - Async inference (background thread, non-blocking for CPU)

    Usage:
        engine = STGCNInferenceEngine(config)
        engine.load()

        # Synchronous
        result = engine.infer(skeleton_window)

        # Async (recommended for CPU)
        engine.submit(skeleton_window)
        results = engine.collect_results()
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.get("stgcn", {})
        self.fusion_config = config.get("fusion", {})
        self.cpu_config = config.get("cpu", {})

        self.cfg_name = self.model_config.get("config")
        self.checkpoint_url = self.model_config.get("checkpoint")
        self.device = self.model_config.get("device", "cpu")
        self.score_threshold = self.model_config.get("score_threshold", 0.25)
        self.top_k = self.model_config.get("top_k", 3)

        # Async inference
        self.async_mode = self.cpu_config.get("async_inference", True)
        self.async_queue: queue.Queue = queue.Queue(
            maxsize=self.cpu_config.get("async_queue_size", 8)
        )
        self.result_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        self.model = None
        self.mapping_engine: Optional[MTMMappingEngine] = None
        self._ready = False

    def load(self) -> bool:
        """
        Load pre-trained ST-GCN model from MMAction2.
        Downloads checkpoint automatically on first run (~100MB).

        Returns:
            True if loaded successfully, False if MMAction2 not available
        """
        try:
            from mmaction.apis import init_recognizer, inference_skeleton
            from mmengine import Config
            import torch

            logger.info(f"Loading ST-GCN: {self.cfg_name}")
            logger.info("First run: downloading checkpoint (~100MB)...")

            # Use mim to get the config file path
            try:
                from mim import get_model_info
                config_path = self._get_mmaction_config()
            except Exception:
                config_path = self.cfg_name

            self.model = init_recognizer(
                config_path,
                self.checkpoint_url,
                device=self.device,
            )
            self.model.eval()
            logger.info(f"ST-GCN loaded on {self.device}")
            self._ready = True

        except ImportError as e:
            logger.error(
                f"MMAction2 not available: {e}\n"
                "Install: pip install openmim && mim install mmaction2"
            )
            return False
        except Exception as e:
            logger.error(f"ST-GCN load failed: {e}")
            return False

        # Initialize mapping engine
        mapping_cfg = self.config.get("mtm_mapping", {})
        self.mapping_engine = MTMMappingEngine(mapping_cfg)

        # Start async worker if needed
        if self.async_mode and self._ready:
            self._start_async_worker()

        return True

    def _get_mmaction_config(self) -> str:
        """Get MMAction2 config file path via mim."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "mim", "download", "mmaction2",
             "--config", self.cfg_name, "--dest", "phase2/checkpoints/"],
            capture_output=True, text=True
        )
        config_path = f"phase2/checkpoints/{self.cfg_name}.py"
        return config_path

    def infer(self, window, step_count: int = 0) -> Optional[STGCNResult]:
        """
        Run ST-GCN inference using MMAction2's inference_recognizer API.
        This is the correct way — lets MMAction2 handle all preprocessing.
        """
        if not self._ready or self.model is None:
            return None

        t_start = time.time()

        try:
            import torch
            import numpy as np
            from mmaction.apis import inference_recognizer
            from mmaction.structures import ActionDataSample

            kps = window.data  # (T_actual, 17, 3)
            T_actual = kps.shape[0]

            # ── Build fake_anno dict — same format MMAction2 PoseDataset uses ─
            # This is what the model was trained on — use exact same structure
            clip_len = 100

            # xy coords only — shape (T, 17, 2)
            xy = kps[:, :, :2].astype(np.float32)

            # Resample to clip_len frames
            if T_actual >= clip_len:
                idx = np.linspace(0, T_actual - 1, clip_len, dtype=int)
            else:
                idx = np.linspace(0, T_actual - 1, clip_len, dtype=int)
            xy = xy[idx]  # (100, 17, 2)

            # confidence scores — shape (T, 17)
            scores_kp = kps[idx, :, 2].astype(np.float32)  # (100, 17)

            # Build annotation in PoseDataset format
            # keypoints: (M, T, V, 2), keypoint_scores: (M, T, V)
            fake_anno = {
                'keypoint': xy[np.newaxis],          # (1, 100, 17, 2)
                # 'keypoint_scores': scores_kp[np.newaxis],  # (1, 100, 17)
                'total_frames': clip_len,
                'frame_inds': np.arange(clip_len),
                'img_shape': (1080, 1920),
                'label': -1,
                'num_clips': 1,
                'clip_len': clip_len,
            }

            # ── Run inference via MMAction2 pipeline ──────────────────────────
            with torch.no_grad():
                result = inference_recognizer(self.model, fake_anno)

            # Extract scores from result
            if hasattr(result, 'pred_score'):
                class_scores = result.pred_score.cpu().numpy()
            elif hasattr(result, 'pred_scores'):
                class_scores = result.pred_scores.item.cpu().numpy()
            else:
                class_scores = np.array(result)

            # Softmax if raw logits
            if class_scores.max() > 1.5 or class_scores.min() < -0.1:
                class_scores = self._softmax(class_scores)

            # ── Map to MTM ────────────────────────────────────────────────────
            top_indices = np.argsort(class_scores)[::-1][:self.top_k].tolist()
            top_scores_list = class_scores[top_indices].tolist()

            is_walk = self.mapping_engine.is_walk_prediction(
                top_indices, top_scores_list
            )
            if is_walk:
                mtm_code = self.mapping_engine.get_walk_mtm(step_count)
                mtm_conf = float(top_scores_list[0]) * 1.1
            else:
                mtm_code, mtm_conf = self.mapping_engine.map(
                    class_scores, self.top_k
                )

            elapsed_ms = (time.time() - t_start) * 1000
            top_name = NTU60_CLASSES[top_indices[0]] if top_indices[0] < len(NTU60_CLASSES) else "unknown"
            logger.debug(
                f"ST-GCN OK: {top_name} ({top_scores_list[0]:.2f}) "
                f"→ {mtm_code} [{elapsed_ms:.0f}ms]"
            )

            return STGCNResult(
                window_id=window.window_id,
                person_id=window.person_id,
                start_time=window.start_time,
                end_time=window.end_time,
                class_scores=class_scores,
                top_k_classes=top_indices,
                top_k_scores=top_scores_list,
                mtm_code=mtm_code,
                mtm_confidence=min(float(mtm_conf), 0.95),
                inference_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"ST-GCN inference error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _make_data_sample(self, num_frames: int):
        """Create MMAction2 data sample metadata."""
        try:
            from mmaction.structures import ActionDataSample
            sample = ActionDataSample()
            sample.set_metainfo({"num_clips": 1, "clip_len": num_frames})
            return sample
        except Exception:
            return {}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - x.max())
        return e_x / e_x.sum()

    def submit(self, window, step_count: int = 0) -> None:
        """Submit window for async inference (non-blocking)."""
        if self.async_mode and self._running:
            try:
                self.async_queue.put_nowait((window, step_count))
            except queue.Full:
                logger.debug("Async queue full — dropping oldest window")
                try:
                    self.async_queue.get_nowait()
                    self.async_queue.put_nowait((window, step_count))
                except queue.Empty:
                    pass
        else:
            result = self.infer(window, step_count)
            if result:
                self.result_queue.put(result)

    def collect_results(self) -> List[STGCNResult]:
        """Collect all available async results (non-blocking)."""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _start_async_worker(self) -> None:
        """Start background inference thread."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
            name="stgcn-inference-worker"
        )
        self._worker_thread.start()
        logger.info("ST-GCN async inference worker started")

    def _inference_worker(self) -> None:
        """Background thread: pull from queue, infer, push results."""
        while self._running:
            try:
                window, step_count = self.async_queue.get(timeout=1.0)
                result = self.infer(window, step_count)
                if result:
                    self.result_queue.put(result)
                    logger.debug(
                        f"Async result: {result.mtm_code} "
                        f"({result.inference_time_ms:.0f}ms)"
                    )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference worker error: {e}")

    def stop(self) -> None:
        """Stop async inference worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def get_pending_count(self) -> int:
        """How many windows are still queued waiting for inference."""
        return self.async_queue.qsize()

    @property
    def is_ready(self) -> bool:
        return self._ready
