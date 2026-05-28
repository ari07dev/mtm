"""
utils/exporter.py
─────────────────────────────────────────────────────────────────────────────
Export pipeline results to JSON, CSV, and MTM text formats.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import List

from core.action_classifier import ClassificationResult
from core.skeleton_builder import SkeletonWindow

logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Exports pipeline results to multiple formats.

    Usage:
        exporter = ResultExporter(config, output_dir="outputs/run_001")
        exporter.save_all(results, windows, mtm_text)
    """

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_json    = config.get("save_json", True)
        self.save_csv     = config.get("save_csv", True)
        self.save_mtm_txt = config.get("save_mtm_txt", True)
        self.json_pretty  = config.get("json_pretty", True)

    def save_all(
        self,
        results: List[ClassificationResult],
        windows: List[SkeletonWindow],
        mtm_text: str,
        video_name: str = "output",
    ) -> dict:
        """
        Save all output formats. Returns dict of saved file paths.
        """
        saved = {}
        base = video_name.replace(" ", "_")

        if self.save_mtm_txt:
            path = self._save_mtm_txt(mtm_text, f"{base}_mtm_codes.txt")
            saved["mtm_txt"] = str(path)

        if self.save_json:
            path = self._save_json(results, windows, f"{base}_results.json")
            saved["json"] = str(path)

        if self.save_csv:
            path = self._save_csv(results, f"{base}_results.csv")
            saved["csv"] = str(path)

        logger.info(f"Saved outputs: {list(saved.values())}")
        return saved

    def _save_mtm_txt(self, mtm_text: str, filename: str) -> Path:
        path = self.output_dir / filename
        path.write_text(mtm_text, encoding="utf-8")
        logger.info(f"MTM codes saved: {path}")
        return path

    def _save_json(
        self,
        results: List[ClassificationResult],
        windows: List[SkeletonWindow],
        filename: str,
    ) -> Path:
        data = {
            "summary": {
                "total_windows": len(windows),
                "total_actions": len(results),
                "unique_codes": list({r.mtm_code for r in results}),
                "duration_seconds": results[-1].end_time if results else 0,
            },
            "actions": [r.to_dict() for r in results],
            "skeleton_windows": [
                {k: v for k, v in w.to_dict().items() if k != "data"}
                for w in windows
            ],
        }

        path = self.output_dir / filename
        indent = 2 if self.json_pretty else None
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
        logger.info(f"JSON saved: {path}")
        return path

    def _save_csv(
        self,
        results: List[ClassificationResult],
        filename: str,
    ) -> Path:
        path = self.output_dir / filename
        if not results:
            path.write_text("no data\n")
            return path

        fieldnames = [
            "person_id", "window_id", "time_start", "time_end",
            "mtm_code", "raw_action", "confidence", "step_count"
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

        logger.info(f"CSV saved: {path}")
        return path
