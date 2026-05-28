"""
scripts/run_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Main CLI entry point for the MTM Industrial Motion Analysis Pipeline.

Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4
    python scripts/run_pipeline.py --video video.mp4 --debug --save-video
    python scripts/run_pipeline.py --video video.mp4 --config configs/custom.yaml
    python scripts/run_pipeline.py --video video.mp4 --no-track --max-frames 300
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from core.pose_extractor import PoseExtractor
from core.tracker import PersonTracker
from core.skeleton_builder import SkeletonBuilder
from core.action_classifier import ActionClassifier
from core.mtm_formatter import MTMFormatter
from utils.video_utils import frame_generator, get_video_metadata, VideoWriter
from utils.visualizer import Visualizer
from utils.exporter import ResultExporter

console = Console()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_pipeline")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--video",       required=True,  help="Path to input video file")
@click.option("--config",      default="configs/pipeline_config.yaml", help="Config YAML path")
@click.option("--output-dir",  default="outputs", help="Output directory")
@click.option("--debug",       is_flag=True, help="Show debug visualization window")
@click.option("--save-video",  is_flag=True, help="Save annotated debug video")
@click.option("--no-track",    is_flag=True, help="Disable ByteTrack person tracking")
@click.option("--max-frames",  default=0, type=int, help="Max frames to process (0=all)")
@click.option("--device",      default=None, help="Device override: cpu/cuda/mps")
@click.option("--verbose",     is_flag=True, help="Verbose logging")
def main(video, config, output_dir, debug, save_video, no_track, max_frames, device, verbose):
    """
    MTM Industrial Motion Analysis Pipeline — Phase 1
    YOLOv8-Pose → ByteTrack → Skeleton Builder → MTM Classifier
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(Panel.fit(
        "[bold yellow]MTM Industrial Motion Analysis Pipeline[/bold yellow]\n"
        "[dim]Phase 1: YOLOv8-Pose + Skeleton Sequence Builder[/dim]",
        border_style="dim"
    ))

    # ── Load config ────────────────────────────────────────────────────────────
    cfg = load_config(config)
    if device:
        cfg["pose"]["device"] = device
    if no_track:
        cfg["tracking"]["enabled"] = False
    if max_frames:
        cfg["video"]["max_frames"] = max_frames

    # ── Video metadata ─────────────────────────────────────────────────────────
    try:
        meta = get_video_metadata(video)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"\n[cyan]Video:[/cyan] {Path(video).name}")
    console.print(f"[cyan]Resolution:[/cyan] {meta.width}×{meta.height} @ {meta.fps:.1f}fps")
    console.print(f"[cyan]Duration:[/cyan] {meta.duration_seconds:.1f}s ({meta.total_frames} frames)")
    console.print(f"[cyan]Device:[/cyan] {cfg['pose'].get('device', 'auto')}\n")

    # ── Initialize pipeline components ────────────────────────────────────────
    console.print("[bold]Initializing pipeline...[/bold]")

    pose_extractor = PoseExtractor(cfg["pose"])
    pose_extractor.load()
    console.print("  [green]✓[/green] YOLOv8-Pose loaded")

    tracker = PersonTracker(cfg["tracking"])
    tracker.initialize()
    console.print("  [green]✓[/green] ByteTrack initialized")

    skeleton_builder = SkeletonBuilder(cfg["skeleton"])
    skeleton_builder.initialize()
    console.print("  [green]✓[/green] Skeleton Builder ready")

    classifier = ActionClassifier(cfg["classifier"])
    console.print("  [green]✓[/green] Action Classifier ready")

    formatter = MTMFormatter(cfg["claude"])
    claude_ready = formatter.initialize()
    status = "[green]✓[/green]" if claude_ready else "[yellow]⚠[/yellow] (disabled)"
    console.print(f"  {status} Claude API formatter")

    video_name = Path(video).stem
    run_dir = str(Path(output_dir) / video_name)
    exporter = ResultExporter(cfg["output"], run_dir)
    console.print(f"  [green]✓[/green] Output → {run_dir}\n")

    visualizer = Visualizer(cfg.get("visualization", {}))

    # ── Setup debug video writer ───────────────────────────────────────────────
    video_writer = None
    if save_video:
        debug_video_path = str(Path(run_dir) / f"{video_name}_debug.mp4")
        resize_w = cfg["video"].get("resize_width", meta.width) or meta.width
        resize_h = cfg["video"].get("resize_height", meta.height) or meta.height
        video_writer = VideoWriter(debug_video_path, meta.fps, resize_w, resize_h)
        video_writer.__enter__()

    # ── Process video ──────────────────────────────────────────────────────────
    all_results = []
    all_windows = []
    current_mtm = "—"
    current_conf = 0.0
    frames_processed = 0
    start_time = time.time()

    total_frames = max_frames if max_frames > 0 else meta.total_frames

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing video...", total=total_frames)

        for frame_id, timestamp, frame in frame_generator(video, cfg["video"], max_frames):

            # 1. Pose Extraction
            poses = pose_extractor.extract(frame, frame_id, timestamp)

            # 2. Tracking
            poses = tracker.update(poses, frame)

            # 3. Skeleton Building
            new_windows = skeleton_builder.update(poses)
            all_windows.extend(new_windows)

            # 4. Classification
            for window in new_windows:
                result = classifier.classify(window)
                all_results.append(result)
                current_mtm = result.mtm_code
                current_conf = result.confidence

            # 5. Debug visualization
            if debug or save_video:
                annotated = visualizer.draw(
                    frame, poses, current_mtm,
                    frame_id, timestamp, current_conf
                )
                if debug:
                    cv2.imshow("MTM Pipeline — Phase 1", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        console.print("[yellow]Stopped by user[/yellow]")
                        break
                if video_writer:
                    video_writer.write(annotated)

            frames_processed += 1
            progress.advance(task)

    # Flush remaining skeleton windows
    final_windows = skeleton_builder.flush()
    all_windows.extend(final_windows)
    for window in final_windows:
        result = classifier.classify(window)
        all_results.append(result)

    elapsed = time.time() - start_time

    if debug:
        import cv2
        cv2.destroyAllWindows()
    if video_writer:
        video_writer.__exit__(None, None, None)

    # ── Post-process + export ──────────────────────────────────────────────────
    smoothed = classifier.smooth_sequence(all_results)
    raw_mtm_text = classifier.to_mtm_text(smoothed)

    # Claude refinement (if enabled)
    final_mtm_text = formatter.format(raw_mtm_text)

    saved_files = exporter.save_all(smoothed, all_windows, final_mtm_text, video_name)

    # ── Summary ────────────────────────────────────────────────────────────────
    fps_achieved = frames_processed / elapsed if elapsed > 0 else 0

    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Pipeline Complete[/bold green]\n"
        f"Frames: {frames_processed}  |  Time: {elapsed:.1f}s  |  Speed: {fps_achieved:.1f} fps",
        border_style="green"
    ))

    # MTM Code summary table
    table = Table(title="Detected MTM Codes", show_header=True, header_style="bold yellow")
    table.add_column("MTM Code", style="cyan")
    table.add_column("Occurrences", justify="right")
    table.add_column("Avg Confidence", justify="right")

    from collections import Counter
    code_counts = Counter(r.mtm_code for r in smoothed if r.mtm_code != "IDLE")
    code_conf = {}
    for r in smoothed:
        if r.mtm_code not in code_conf:
            code_conf[r.mtm_code] = []
        code_conf[r.mtm_code].append(r.confidence)

    for code, count in code_counts.most_common():
        avg_conf = sum(code_conf[code]) / len(code_conf[code])
        table.add_row(code, str(count), f"{avg_conf*100:.0f}%")

    console.print(table)

    # Print final MTM output
    console.print("\n[bold]Final MTM Output:[/bold]")
    console.print(Panel(final_mtm_text, border_style="yellow"))

    # Print saved files
    console.print("\n[bold]Saved files:[/bold]")
    for key, path in saved_files.items():
        console.print(f"  [green]{key}:[/green] {path}")


if __name__ == "__main__":
    main()
