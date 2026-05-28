"""
core/action_classifier.py
─────────────────────────────────────────────────────────────────────────────
Rule-based MTM action classifier — operates on SkeletonWindows.

This is the bridge between Phase 1 (pose extraction) and Phase 2 (ST-GCN).
It uses biomechanical rules to classify MTM codes without needing training data.

Replaces ST-GCN during the prototype phase. Once ST-GCN is trained, this
module becomes a fallback / ensemble member.

Classification logic:
  - WALK codes: ankle displacement + stride count
  - GET/HOLD/GRASP: wrist velocity + hand-to-object-zone proximity
  - PUT/PLACE: downward wrist movement + stillness
  - SLIDE: lateral wrist movement with maintained height
  - IDLE: no significant movement in any joint
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.skeleton_builder import SkeletonWindow

logger = logging.getLogger(__name__)

# ── MTM Code Constants ────────────────────────────────────────────────────────
MTM_CODES = {
    "walk_1_4":    "WALK 1-4 STEPS",
    "walk_5_7":    "WALK 5-7 STEPS",
    "walk_8_10":   "WALK 8-10 STEPS",
    "walk_11_15":  "WALK 11-15 STEPS",
    "walk_16_30":  "WALK 16-30 STEPS",
    "reach_get":   "GET + HOLD OBJECT",
    "grasp_hold":  "GRASP + HOLD OBJECT",
    "grasp_place": "GRASP + PLACE OBJECT",
    "hold_put":    "HOLD + PUT OBJECT",
    "hold_slide":  "HOLD + SLIDE OBJECT (M3)",
    "position":    "PT",
    "idle":        "IDLE",
}

# Keypoint indices
IDX_LEFT_WRIST   = 9
IDX_RIGHT_WRIST  = 10
IDX_LEFT_ANKLE   = 15
IDX_RIGHT_ANKLE  = 16
IDX_LEFT_HIP     = 11
IDX_RIGHT_HIP    = 12
IDX_LEFT_KNEE    = 13
IDX_RIGHT_KNEE   = 14
IDX_LEFT_SHOULDER  = 5
IDX_RIGHT_SHOULDER = 6


@dataclass
class ClassificationResult:
    """Output of the action classifier for one SkeletonWindow."""
    person_id: int
    window_id: int
    start_time: float
    end_time: float
    mtm_code: str           # Final MTM label
    raw_action: str         # Internal action key
    confidence: float       # [0, 1]
    step_count: int         # For WALK actions, estimated steps
    features: dict          # Debug features used for classification

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "window_id": self.window_id,
            "time_start": round(self.start_time, 3),
            "time_end": round(self.end_time, 3),
            "mtm_code": self.mtm_code,
            "raw_action": self.raw_action,
            "confidence": round(self.confidence, 3),
            "step_count": self.step_count,
        }


class ActionClassifier:
    """
    Rule-based MTM action classifier operating on SkeletonWindows.

    Usage:
        classifier = ActionClassifier(config)
        result = classifier.classify(skeleton_window)
    """

    def __init__(self, config: dict):
        cfg_walk = config.get("walk", {})
        cfg_hands = config.get("hands", {})
        cfg_smooth = config.get("smoothing", {})

        # Walk thresholds
        self.ankle_displacement_thresh = cfg_walk.get("min_ankle_displacement", 0.03)
        self.stride_window = cfg_walk.get("stride_window", 15)
        self.step_thresholds = [
            (1, 4,   "walk_1_4"),
            (5, 7,   "walk_5_7"),
            (8, 10,  "walk_8_10"),
            (11, 15, "walk_11_15"),
            (16, 30, "walk_16_30"),
        ]

        # Hand thresholds
        self.reach_velocity_thresh = cfg_hands.get("reach_velocity_threshold", 0.05)
        self.grasp_proximity_thresh = cfg_hands.get("grasp_proximity_threshold", 0.08)
        self.hold_stillness_thresh = cfg_hands.get("hold_stillness_threshold", 0.02)
        self.slide_axis_thresh = cfg_hands.get("slide_axis_threshold", 0.03)

        # Smoothing
        self.min_action_duration = cfg_smooth.get("min_action_duration_frames", 8)

        self._result_history: Dict[int, List[ClassificationResult]] = {}

    def classify(self, window: SkeletonWindow) -> ClassificationResult:
        """
        Classify a single SkeletonWindow into an MTM code.

        Args:
            window: SkeletonWindow with (T, 17, 3) normalized keypoints

        Returns:
            ClassificationResult with MTM code + metadata
        """
        data = window.data  # (T, 17, 3)
        T = data.shape[0]

        # ── Extract key joint trajectories ────────────────────────────────────
        l_wrist  = data[:, IDX_LEFT_WRIST,   :]  # (T, 3)
        r_wrist  = data[:, IDX_RIGHT_WRIST,  :]
        l_ankle  = data[:, IDX_LEFT_ANKLE,   :]
        r_ankle  = data[:, IDX_RIGHT_ANKLE,  :]
        l_hip    = data[:, IDX_LEFT_HIP,     :]
        r_hip    = data[:, IDX_RIGHT_HIP,    :]

        # ── Compute motion features ───────────────────────────────────────────
        features = self._compute_features(
            l_wrist, r_wrist, l_ankle, r_ankle, l_hip, r_hip
        )

        # ── Classify ─────────────────────────────────────────────────────────
        raw_action, confidence, step_count = self._apply_rules(features)

        result = ClassificationResult(
            person_id=window.person_id,
            window_id=window.window_id,
            start_time=window.start_time,
            end_time=window.end_time,
            mtm_code=MTM_CODES.get(raw_action, "IDLE"),
            raw_action=raw_action,
            confidence=confidence,
            step_count=step_count,
            features=features,
        )

        # Track history
        if window.person_id not in self._result_history:
            self._result_history[window.person_id] = []
        self._result_history[window.person_id].append(result)

        logger.debug(
            f"Person {window.person_id} | W{window.window_id} | "
            f"{result.mtm_code} ({confidence:.2f})"
        )

        return result

    def _compute_features(
        self,
        l_wrist: np.ndarray,
        r_wrist: np.ndarray,
        l_ankle: np.ndarray,
        r_ankle: np.ndarray,
        l_hip: np.ndarray,
        r_hip: np.ndarray,
    ) -> dict:
        """Extract biomechanical features from joint trajectories."""

        def velocity(traj: np.ndarray) -> np.ndarray:
            """(T-1, 2) velocity vector."""
            return np.diff(traj[:, :2], axis=0)

        def speed(traj: np.ndarray) -> np.ndarray:
            """(T-1,) scalar speed."""
            v = velocity(traj)
            return np.sqrt((v ** 2).sum(axis=1))

        def total_displacement(traj: np.ndarray) -> float:
            """Total path length traveled."""
            return float(speed(traj).sum())

        def net_displacement(traj: np.ndarray) -> float:
            """Straight-line displacement from start to end."""
            valid = traj[traj[:, 2] > 0.1]
            if len(valid) < 2:
                return 0.0
            return float(np.linalg.norm(valid[-1, :2] - valid[0, :2]))

        def mean_speed(traj: np.ndarray) -> float:
            s = speed(traj)
            return float(s.mean()) if len(s) > 0 else 0.0

        def count_ankle_oscillations(
            l_traj: np.ndarray,
            r_traj: np.ndarray,
            l_hip_ref: np.ndarray,
            r_hip_ref: np.ndarray,
        ) -> int:
            """
            Count stride cycles using BODY-RELATIVE ankle motion.

            MOVING CAMERA FIX:
            Subtracts hip center Y from ankle Y — this removes camera pan/tilt
            motion, leaving only true leg swing. Works regardless of camera angle.
            Each peak in relative ankle Y = one foot lift = one step.
            """
            # Hip midpoint = body reference (moves with camera)
            hip_center_y = (l_hip_ref[:, 1] + r_hip_ref[:, 1]) / 2

            # Body-relative ankle Y — true leg motion only
            l_rel_y = l_traj[:, 1] - hip_center_y
            r_rel_y = r_traj[:, 1] - hip_center_y

            def count_peaks(signal: np.ndarray, min_prominence: float = 0.015) -> int:
                peaks = 0
                for i in range(1, len(signal) - 1):
                    if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                        prominence = signal[i] - min(signal[i-1], signal[i+1])
                        if prominence > min_prominence:
                            peaks += 1
                return peaks

            return count_peaks(l_rel_y) + count_peaks(r_rel_y)

        def lateral_dominance(traj: np.ndarray) -> float:
            """Ratio of x-movement to y-movement (>1 = horizontal/slide)."""
            v = velocity(traj)
            x_range = np.abs(v[:, 0]).mean()
            y_range = np.abs(v[:, 1]).mean()
            return float(x_range / (y_range + 1e-6))

        # ── Walk features ────────────────────────────────────────────────────
        ankle_total_disp = (
            total_displacement(l_ankle) + total_displacement(r_ankle)
        ) / 2
        ankle_net_disp = (
            net_displacement(l_ankle) + net_displacement(r_ankle)
        ) / 2
        step_count = count_ankle_oscillations(l_ankle, r_ankle, l_hip, r_hip)
        hip_speed = mean_speed(l_hip)

        # ── Hand features ─────────────────────────────────────────────────────
        l_wrist_speed = mean_speed(l_wrist)
        r_wrist_speed = mean_speed(r_wrist)
        max_wrist_speed = max(l_wrist_speed, r_wrist_speed)

        l_wrist_disp = total_displacement(l_wrist)
        r_wrist_disp = total_displacement(r_wrist)

        # Wrist height (y) at end vs start — negative = moving up, positive = down
        l_wrist_y_delta = float(l_wrist[-1, 1] - l_wrist[0, 1]) if l_wrist[0, 2] > 0.1 else 0
        r_wrist_y_delta = float(r_wrist[-1, 1] - r_wrist[0, 1]) if r_wrist[0, 2] > 0.1 else 0
        mean_wrist_y_delta = (l_wrist_y_delta + r_wrist_y_delta) / 2

        # Wrist stillness (low = holding object)
        wrist_stillness = 1.0 - min(max_wrist_speed / 0.1, 1.0)

        # Lateral dominance (slide detection)
        l_lateral = lateral_dominance(l_wrist)
        r_lateral = lateral_dominance(r_wrist)
        max_lateral = max(l_lateral, r_lateral)

        return {
            "ankle_total_disp": ankle_total_disp,
            "ankle_net_disp": ankle_net_disp,
            "step_count": step_count,
            "hip_speed": hip_speed,
            "l_wrist_speed": l_wrist_speed,
            "r_wrist_speed": r_wrist_speed,
            "max_wrist_speed": max_wrist_speed,
            "l_wrist_disp": l_wrist_disp,
            "r_wrist_disp": r_wrist_disp,
            "mean_wrist_y_delta": mean_wrist_y_delta,
            "wrist_stillness": wrist_stillness,
            "max_lateral_dominance": max_lateral,
        }

    def _apply_rules(self, f: dict) -> Tuple[str, float, int]:
        """
        Apply biomechanical rules to feature dict.

        Returns:
            (raw_action, confidence, step_count)
        """
        step_count = f["step_count"]

        # ── WALK detection ────────────────────────────────────────────────────
        is_walking = (
            f["ankle_total_disp"] > self.ankle_displacement_thresh and
            f["hip_speed"] > 0.01
        )

        if is_walking:
            for lo, hi, key in self.step_thresholds:
                if lo <= step_count <= hi:
                    conf = 0.7 + 0.1 * min(f["ankle_total_disp"] / 0.2, 1.0)
                    return key, min(conf, 0.95), step_count

            # Out of range: nearest bucket
            if step_count > 30:
                return "walk_16_30", 0.65, step_count
            elif step_count == 0:
                pass  # Fall through to hand analysis
            else:
                # Snap to nearest bucket
                return "walk_1_4", 0.55, step_count

        # ── SLIDE detection ───────────────────────────────────────────────────
        if (f["max_lateral_dominance"] > 2.0 and
                f["max_wrist_speed"] > self.reach_velocity_thresh and
                f["wrist_stillness"] < 0.6):
            return "hold_slide", 0.72, 0

        # ── REACH / GET detection ─────────────────────────────────────────────
        reaching = (
            f["max_wrist_speed"] > self.reach_velocity_thresh and
            f["mean_wrist_y_delta"] < 0      # Hand moving upward (reaching)
        )
        if reaching and f["wrist_stillness"] < 0.5:
            conf = 0.65 + 0.1 * min(f["max_wrist_speed"] / 0.1, 1.0)
            return "reach_get", min(conf, 0.88), 0

        # ── PUT / PLACE detection ─────────────────────────────────────────────
        placing = (
            f["max_wrist_speed"] > self.reach_velocity_thresh * 0.5 and
            f["mean_wrist_y_delta"] > 0.02   # Hand moving downward
        )
        if placing:
            if f["max_wrist_speed"] > self.reach_velocity_thresh:
                return "hold_put", 0.70, 0
            else:
                return "grasp_place", 0.68, 0

        # ── HOLD / GRASP detection ────────────────────────────────────────────
        if f["wrist_stillness"] > 0.7:
            if f["max_wrist_speed"] < self.hold_stillness_thresh:
                return "grasp_hold", 0.72, 0
            else:
                return "reach_get", 0.58, 0

        # ── POSITION (PT) ─────────────────────────────────────────────────────
        if (f["max_wrist_speed"] < self.reach_velocity_thresh * 0.3 and
                f["ankle_total_disp"] < self.ankle_displacement_thresh * 0.5):
            return "position", 0.60, 0

        # ── IDLE fallback ─────────────────────────────────────────────────────
        return "idle", 0.50, 0

    def smooth_sequence(
        self,
        results: List[ClassificationResult],
        min_duration_frames: int = 8,
    ) -> List[ClassificationResult]:
        """
        Post-process a sequence of results to remove flickering.
        Merges very short actions and smooths rapid oscillations.
        (Light version of MS-TCN, which replaces this in Phase 3)
        """
        if len(results) < 2:
            return results

        smoothed = [results[0]]
        for r in results[1:]:
            last = smoothed[-1]
            # Merge if same code and small gap
            if r.mtm_code == last.mtm_code:
                # Extend the last result's end time
                smoothed[-1] = ClassificationResult(
                    person_id=last.person_id,
                    window_id=last.window_id,
                    start_time=last.start_time,
                    end_time=r.end_time,
                    mtm_code=last.mtm_code,
                    raw_action=last.raw_action,
                    confidence=max(last.confidence, r.confidence),
                    step_count=last.step_count + r.step_count,
                    features=last.features,
                )
            else:
                smoothed.append(r)

        return smoothed

    def to_mtm_text(self, results: List[ClassificationResult]) -> str:
        """
        Convert result sequence to final MTM text format.

        Output:
            TITLE
            WALK 11-15 STEPS
            GET + HOLD OBJECT
            ...
        """
        lines = ["TITLE"]
        seen = []
        for r in results:
            if r.mtm_code == "IDLE":
                continue
            if not seen or seen[-1] != r.mtm_code:
                lines.append(r.mtm_code)
                seen.append(r.mtm_code)
        return "\n".join(lines)
