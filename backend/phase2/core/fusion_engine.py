"""
phase2/core/fusion_engine.py
─────────────────────────────────────────────────────────────────────────────
Fusion Engine — combines Phase 1 rule-based classifier with Phase 2 ST-GCN.

Three modes:
  stgcn_only    — pure ST-GCN (best accuracy when confident)
  rules_only    — pure Phase 1 rules (fast, no GPU needed)
  weighted_vote — weighted combination (default, most robust)

On CPU, ST-GCN runs ~0.5-2s per window. The fusion engine handles the
timing mismatch: Phase 1 gives immediate results, ST-GCN results arrive
later and retroactively update the sequence.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.action_classifier import ClassificationResult
from phase2.core.stgcn_inference import STGCNResult

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """Final fused MTM classification from Phase 1 + Phase 2."""
    window_id: int
    person_id: int
    start_time: float
    end_time: float
    mtm_code: str
    confidence: float
    source: str           # "stgcn", "rules", "fused", "stgcn_fallback"
    phase1_code: Optional[str] = None
    phase2_code: Optional[str] = None
    phase1_conf: float = 0.0
    phase2_conf: float = 0.0

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "person_id": self.person_id,
            "time_start": round(self.start_time, 3),
            "time_end": round(self.end_time, 3),
            "mtm_code": self.mtm_code,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "phase1": self.phase1_code,
            "phase2": self.phase2_code,
        }


class FusionEngine:
    """
    Combines Phase 1 (rules) + Phase 2 (ST-GCN) predictions.

    Usage:
        fusion = FusionEngine(config)

        # For each window:
        # Phase 1 result is immediate
        p1_result = classifier.classify(window)

        # Phase 2 result arrives later (async)
        fusion.add_phase1(p1_result)
        ...
        stgcn_result = engine.collect_results()
        fusion.add_phase2(stgcn_result)

        # Get final sequence
        final = fusion.get_fused_sequence()
    """

    def __init__(self, config: dict):
        self.method = config.get("method", "weighted_vote")
        self.stgcn_weight = config.get("stgcn_weight", 0.65)
        self.rules_weight = config.get("rules_weight", 0.35)
        self.fallback_threshold = config.get("fallback_threshold", 0.25)

        # Buffers: window_id → result
        self._phase1: Dict[int, ClassificationResult] = {}
        self._phase2: Dict[int, STGCNResult] = {}
        self._fused: Dict[int, FusedResult] = {}
        self._pending_window_ids: List[int] = []

    def add_phase1(self, result: ClassificationResult) -> FusedResult:
        """
        Register a Phase 1 result. Returns immediate fused result
        using rules only (ST-GCN not yet available for this window).
        """
        wid = result.window_id
        self._phase1[wid] = result
        self._pending_window_ids.append(wid)

        # Immediate result = Phase 1 only
        fused = FusedResult(
            window_id=wid,
            person_id=result.person_id,
            start_time=result.start_time,
            end_time=result.end_time,
            mtm_code=result.mtm_code,
            confidence=result.confidence,
            source="rules",
            phase1_code=result.mtm_code,
            phase1_conf=result.confidence,
        )
        self._fused[wid] = fused
        return fused

    def add_phase2(self, stgcn_result: STGCNResult) -> Optional[FusedResult]:
        """
        Register a Phase 2 ST-GCN result and retroactively update
        the fused result for that window.

        Returns updated FusedResult, or None if window not found.
        """
        wid = stgcn_result.window_id
        self._phase2[wid] = stgcn_result

        if wid not in self._phase1:
            logger.debug(f"Phase 2 result for unknown window {wid} — storing for later")
            return None

        p1 = self._phase1[wid]
        p2 = stgcn_result

        fused = self._fuse(p1, p2)
        self._fused[wid] = fused

        logger.debug(
            f"Window {wid} updated: {p1.mtm_code}({p1.confidence:.2f}) + "
            f"{p2.mtm_code}({p2.mtm_confidence:.2f}) → {fused.mtm_code}({fused.confidence:.2f})"
        )
        return fused

    def _fuse(
        self,
        p1: ClassificationResult,
        p2: STGCNResult,
    ) -> FusedResult:
        """Core fusion logic."""

        if self.method == "stgcn_only":
            # Trust ST-GCN completely; fallback to rules if low confidence
            if p2.mtm_confidence >= self.fallback_threshold:
                return FusedResult(
                    window_id=p1.window_id,
                    person_id=p1.person_id,
                    start_time=p1.start_time,
                    end_time=p1.end_time,
                    mtm_code=p2.mtm_code,
                    confidence=p2.mtm_confidence,
                    source="stgcn",
                    phase1_code=p1.mtm_code,
                    phase2_code=p2.mtm_code,
                    phase1_conf=p1.confidence,
                    phase2_conf=p2.mtm_confidence,
                )
            else:
                return FusedResult(
                    window_id=p1.window_id,
                    person_id=p1.person_id,
                    start_time=p1.start_time,
                    end_time=p1.end_time,
                    mtm_code=p1.mtm_code,
                    confidence=p1.confidence,
                    source="stgcn_fallback",
                    phase1_code=p1.mtm_code,
                    phase2_code=p2.mtm_code,
                    phase1_conf=p1.confidence,
                    phase2_conf=p2.mtm_confidence,
                )

        elif self.method == "rules_only":
            return FusedResult(
                window_id=p1.window_id,
                person_id=p1.person_id,
                start_time=p1.start_time,
                end_time=p1.end_time,
                mtm_code=p1.mtm_code,
                confidence=p1.confidence,
                source="rules",
                phase1_code=p1.mtm_code,
                phase2_code=p2.mtm_code,
                phase1_conf=p1.confidence,
                phase2_conf=p2.mtm_confidence,
            )

        else:  # weighted_vote (default)
            # If both agree → high confidence fused result
            if p1.mtm_code == p2.mtm_code:
                fused_conf = min(
                    self.stgcn_weight * p2.mtm_confidence +
                    self.rules_weight * p1.confidence,
                    0.97
                )
                return FusedResult(
                    window_id=p1.window_id,
                    person_id=p1.person_id,
                    start_time=p1.start_time,
                    end_time=p1.end_time,
                    mtm_code=p1.mtm_code,
                    confidence=fused_conf,
                    source="fused",
                    phase1_code=p1.mtm_code,
                    phase2_code=p2.mtm_code,
                    phase1_conf=p1.confidence,
                    phase2_conf=p2.mtm_confidence,
                )
            else:
                # Disagreement: pick higher weighted score
                p1_weighted = self.rules_weight * p1.confidence
                p2_weighted = self.stgcn_weight * p2.mtm_confidence

                if p2_weighted >= p1_weighted:
                    return FusedResult(
                        window_id=p1.window_id,
                        person_id=p1.person_id,
                        start_time=p1.start_time,
                        end_time=p1.end_time,
                        mtm_code=p2.mtm_code,
                        confidence=p2_weighted,
                        source="fused",
                        phase1_code=p1.mtm_code,
                        phase2_code=p2.mtm_code,
                        phase1_conf=p1.confidence,
                        phase2_conf=p2.mtm_confidence,
                    )
                else:
                    return FusedResult(
                        window_id=p1.window_id,
                        person_id=p1.person_id,
                        start_time=p1.start_time,
                        end_time=p1.end_time,
                        mtm_code=p1.mtm_code,
                        confidence=p1_weighted,
                        source="fused",
                        phase1_code=p1.mtm_code,
                        phase2_code=p2.mtm_code,
                        phase1_conf=p1.confidence,
                        phase2_conf=p2.mtm_confidence,
                    )

    def get_fused_sequence(self) -> List[FusedResult]:
        """Return all fused results sorted by time."""
        results = list(self._fused.values())
        return sorted(results, key=lambda r: r.start_time)

    def get_pending_count(self) -> int:
        """How many Phase 1 windows are still waiting for ST-GCN."""
        return len(set(self._pending_window_ids) - set(self._phase2.keys()))

    def to_mtm_text(self, results: List[FusedResult]) -> str:
        """Convert fused sequence to final MTM text output."""
        lines = ["TITLE"]
        last = ""
        for r in results:
            if r.mtm_code == "IDLE":
                continue
            if r.mtm_code != last:
                lines.append(r.mtm_code)
                last = r.mtm_code
        return "\n".join(lines)

    def get_source_stats(self) -> dict:
        """Returns breakdown of how many results came from each source."""
        from collections import Counter
        sources = Counter(r.source for r in self._fused.values())
        return dict(sources)
