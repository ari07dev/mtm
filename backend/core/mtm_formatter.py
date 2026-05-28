"""
core/mtm_formatter.py
─────────────────────────────────────────────────────────────────────────────
Claude API MTM Code Formatter — Phase 4

Takes raw classifier output and uses Claude to:
1. Clean up ambiguous action sequences
2. Map to correct MTM vocabulary
3. Format final output in standard industrial layout
4. Handle edge cases the rule-based classifier misses
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging
import os
from typing import List, Optional

from core.action_classifier import ClassificationResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Methods-Time Measurement (MTM) industrial analyst.
You will receive a raw sequence of motion labels extracted from a pose estimation pipeline analyzing an industrial worker video.

Your job is to:
1. Clean and standardize the sequence into proper MTM codes
2. Remove noise and impossible transitions
3. Infer missing codes where the sequence logic demands them
4. Output ONLY the final MTM sequence, one code per line

Valid MTM codes you must use:
- WALK 1-4 STEPS
- WALK 5-7 STEPS  
- WALK 8-10 STEPS
- WALK 11-15 STEPS
- WALK 16-30 STEPS
- GET + HOLD OBJECT
- GRASP + HOLD OBJECT
- GRASP + PLACE OBJECT
- HOLD + PUT OBJECT
- HOLD + SLIDE OBJECT (M3)
- PT

Rules:
- Worker must WALK before they can GET or GRASP (if workstation is >4 steps away)
- After HOLD + PUT or GRASP + PLACE, worker typically walks away
- HOLD + SLIDE implies horizontal surface work (e.g., assembly line)
- PT (position) occurs when precise placement is needed
- Do not output IDLE
- First line must be TITLE
- No explanations, no numbering, just codes"""


class MTMFormatter:
    """
    Claude API wrapper for MTM sequence refinement.

    Usage:
        formatter = MTMFormatter(config)
        raw_text = classifier.to_mtm_text(results)
        refined_text = formatter.format(raw_text)
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = config.get("max_tokens", 2048)
        self.system_prompt = config.get("system_prompt", SYSTEM_PROMPT)
        self.client = None

    def initialize(self) -> bool:
        """Initialize Anthropic client. Returns True if successful."""
        if not self.enabled:
            logger.info("Claude API formatter disabled (set claude.enabled=true in config)")
            return False

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error(
                "ANTHROPIC_API_KEY not set. Export it:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
            return False

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Claude API initialized: {self.model}")
            return True
        except ImportError:
            logger.error("anthropic package not installed: pip install anthropic")
            return False

    def format(self, raw_mtm_text: str, context: Optional[str] = None) -> str:
        """
        Send raw MTM sequence to Claude for refinement.

        Args:
            raw_mtm_text: Raw output from ActionClassifier.to_mtm_text()
            context: Optional context (industry type, workstation description)

        Returns:
            Refined MTM sequence string, or raw_mtm_text if Claude unavailable
        """
        if not self.enabled or self.client is None:
            logger.debug("Returning raw MTM text (Claude not enabled)")
            return raw_mtm_text

        user_content = f"Raw motion sequence:\n{raw_mtm_text}"
        if context:
            user_content = f"Context: {context}\n\n{user_content}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )

            refined = response.content[0].text.strip()
            logger.info(f"Claude refined {len(raw_mtm_text.splitlines())} → "
                       f"{len(refined.splitlines())} MTM lines")
            return refined

        except Exception as e:
            logger.error(f"Claude API error: {e} — returning raw output")
            return raw_mtm_text

    def format_batch(
        self,
        segments: List[str],
        context: Optional[str] = None,
    ) -> str:
        """
        Format multiple segments (e.g., per-worker) and combine.
        """
        all_results = []
        for i, segment in enumerate(segments):
            logger.info(f"Formatting segment {i+1}/{len(segments)}")
            refined = self.format(segment, context)
            all_results.append(refined)
        return "\n\n".join(all_results)
