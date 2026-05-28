"""
scripts/run_pipeline_phase2.py
─────────────────────────────────────────────────────────────────────────────
Full Phase 1 + Phase 2 pipeline:

    Video → YOLOv8-Pose → Skeleton → [Rules + ST-GCN] → Fusion → MTM Output

ST-GCN runs asynchronously on CPU — video processing continues
while ST-GCN works in background on each window.
─────────────────────────────────────────────────────────────────────────────

Usage:
    python scripts/run_pipeline_phase2.py --video your_video.mp4
    python scripts/run_pipeline_phase2.py --video video.mp4 --stgcn-only
    python scripts/run_pipeline_phase2.py --video video.mp4 --no-stgcn  (Phase 1 only)
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from core.pose_extractor import PoseExtractor
from core.tracker import PersonTracker
from core.skeleton_builder import SkeletonBuilder
from core.action_classifier import ActionClassifier
from core.mtm_formatter import MTMFormatter
from phase2.core.stgcn_inference import STGCNInferenceEngine
from phase2.core.fusion_engine import FusionEngine
from utils.video_utils import frame_generator, get_video_metadata
from utils.visualizer import Visualizer
from utils.exporter import ResultExporter

console = Console()
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--video",        required=True,  help="Input video path")
@click.option("--config",       default="configs/pipeline_config.yaml")
@click.option("--stgcn-config", default="phase2/configs/stgcn_config.yaml")
@click.option("--output-dir",   default="outputs")
@click.option("--no-stgcn",     is_flag=True,  help="Skip ST-GCN, use Phase 1 rules only")
@click.option("--stgcn-only",   is_flag=True,  help="Use ST-GCN result only (ignore rules)")
@click.option("--debug",        is_flag=True,  help="Show live skeleton window")
@click.option("--max-frames",   default=0, type=int)
@click.option("--verbose",      is_flag=True)
def main(video, config, stgcn_config, output_dir, no_stgcn,
         stgcn_only, debug, max_frames, verbose):
    """MTM Pipeline Phase 1 + 2: YOLOv8-Pose + ST-GCN Fusion"""


    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(Panel.fit(
        "[bold cyan]MTM Industrial Motion Analysis — Phase 1 + 2[/bold cyan]\n"
        "[dim]YOLOv8-Pose + ST-GCN (MMAction2) + Fusion Engine[/dim]",
        border_style="dim"
    ))

    # ── Load configs ───────────────────────────────────────────────────────────
    cfg = load_config(config)
    stgcn_cfg = load_config(stgcn_config)

    if stgcn_only:
        stgcn_cfg["fusion"]["method"] = "stgcn_only"
    if no_stgcn:
        stgcn_cfg["fusion"]["method"] = "rules_only"

    # ── Video metadata ─────────────────────────────────────────────────────────
    meta = get_video_metadata(video)
    console.print(f"\n[cyan]Video:[/cyan] {Path(video).name}  "
                  f"[cyan]Duration:[/cyan] {meta.duration_seconds:.1f}s  "
                  f"[cyan]FPS:[/cyan] {meta.fps:.1f}\n")

    # ── Initialize all components ──────────────────────────────────────────────
    console.print("[bold]Initializing pipeline...[/bold]")

    # Phase 1
    pose_extractor = PoseExtractor(cfg["pose"])
    pose_extractor.load()
    console.print("  [green]✓[/green] YOLOv8-Pose")

    tracker = PersonTracker(cfg["tracking"])
    tracker.initialize()
    console.print("  [green]✓[/green] ByteTrack")

    builder = SkeletonBuilder(cfg["skeleton"])
    builder.initialize()
    console.print("  [green]✓[/green] Skeleton Builder")

    classifier = ActionClassifier(cfg["classifier"])
    console.print("  [green]✓[/green] Phase 1 Rule Classifier")

    # Phase 2
    stgcn_engine = None
    if not no_stgcn:
        stgcn_engine = STGCNInferenceEngine(stgcn_cfg)
        stgcn_ok = stgcn_engine.load()
        if stgcn_ok:
            console.print("  [green]✓[/green] ST-GCN (MMAction2) — async CPU mode")
        else:
            console.print("  [yellow]⚠[/yellow] ST-GCN unavailable — using Phase 1 only")
            stgcn_engine = None

    # Fusion
    fusion = FusionEngine(stgcn_cfg.get("fusion", {}))
    console.print("  [green]✓[/green] Fusion Engine "
                  f"({stgcn_cfg.get('fusion', {}).get('method', 'weighted_vote')})")

    # Claude formatter + exporter
    formatter = MTMFormatter(cfg["claude"])
    formatter.initialize()

    video_name = Path(video).stem
    run_dir = str(Path(output_dir) / f"{video_name}_phase2")
    exporter = ResultExporter(cfg["output"], run_dir)
    console.print(f"  [green]✓[/green] Output → {run_dir}\n")

    visualizer = Visualizer(cfg.get("visualization", {}))

    # ── Process video ──────────────────────────────────────────────────────────
    all_fused: dict = {}
    current_mtm = "—"
    current_conf = 0.0
    frames_processed = 0
    start_time = time.time()
    total = max_frames if max_frames > 0 else meta.total_frames

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing video...", total=total)

        for frame_id, timestamp, frame in frame_generator(video, cfg["video"], max_frames):

            # Phase 1: Pose → Skeleton → Rules
            poses = pose_extractor.extract(frame, frame_id, timestamp)
            poses = tracker.update(poses, frame)
            windows = builder.update(poses)

            for window in windows:
                # Phase 1 classification (immediate)
                p1_result = classifier.classify(window)
                fused = fusion.add_phase1(p1_result)
                all_fused[window.window_id] = fused
                current_mtm = fused.mtm_code
                current_conf = fused.confidence

                # Phase 2: Submit to ST-GCN async (non-blocking)
                if stgcn_engine and stgcn_engine.is_ready:
                    stgcn_engine.submit(window, p1_result.step_count)

            # Collect any ST-GCN results that finished in background
            if stgcn_engine:
                for stgcn_result in stgcn_engine.collect_results():
                    updated = fusion.add_phase2(stgcn_result)
                    if updated:
                        all_fused[updated.window_id] = updated
                        # Update current display if this is the latest window
                        if updated.window_id == max(all_fused.keys(), default=0):
                            current_mtm = updated.mtm_code
                            current_conf = updated.confidence

            # Debug visualization
            if debug:
                import cv2
                annotated = visualizer.draw(
                    frame, poses, current_mtm, frame_id, timestamp, current_conf
                )
                cv2.imshow("MTM Phase 1+2", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames_processed += 1
            progress.advance(task)

    # Flush remaining windows
    for window in builder.flush():
        p1 = classifier.classify(window)
        fused = fusion.add_phase1(p1)
        all_fused[window.window_id] = fused

    # Wait for remaining ST-GCN results (up to 30s on CPU)
    if stgcn_engine:
        console.print("[dim]Waiting for remaining ST-GCN results...[/dim]")
        deadline = time.time() + 30
        while stgcn_engine.get_pending_count() > 0 and time.time() < deadline:
            for r in stgcn_engine.collect_results():
                updated = fusion.add_phase2(r)
                if updated:
                    all_fused[updated.window_id] = updated
            time.sleep(0.5)
        stgcn_engine.stop()

    if debug:
        import cv2
        cv2.destroyAllWindows()

    # ── Build final output ─────────────────────────────────────────────────────
    fused_sequence = fusion.get_fused_sequence()
    raw_mtm = fusion.to_mtm_text(fused_sequence)
    final_mtm = formatter.format(raw_mtm)

    # Save outputs (convert FusedResult to ClassificationResult-like for exporter)
    from core.action_classifier import ClassificationResult
    compat_results = [
        ClassificationResult(
            person_id=r.person_id,
            window_id=r.window_id,
            start_time=r.start_time,
            end_time=r.end_time,
            mtm_code=r.mtm_code,
            raw_action=r.source,
            confidence=r.confidence,
            step_count=0,
            features={},
        )
        for r in fused_sequence
    ]
    saved = exporter.save_all(compat_results, [], final_mtm, video_name)

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    source_stats = fusion.get_source_stats()

    console.print(f"\n[bold green]✓ Complete[/bold green] — "
                  f"{frames_processed} frames in {elapsed:.1f}s")

    # Source breakdown
    table = Table(title="Fusion Source Breakdown")
    table.add_column("Source", style="cyan")
    table.add_column("Windows", justify="right")
    for src, count in source_stats.items():
        table.add_row(src, str(count))
    console.print(table)

    # MTM output
    console.print("\n[bold]Final MTM Output:[/bold]")
    console.print(Panel(final_mtm, border_style="cyan"))

    console.print("\n[bold]Saved:[/bold]")
    for k, v in saved.items():
        console.print(f"  [green]{k}:[/green] {v}")


if __name__ == "__main__":
    main()
