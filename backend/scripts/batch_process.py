"""
scripts/batch_process.py
─────────────────────────────────────────────────────────────────────────────
Batch process an entire folder of videos.

Usage:
    python scripts/batch_process.py --folder videos/ --output outputs/batch_run
    python scripts/batch_process.py --folder videos/ --pattern "*.avi" --workers 2
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

console = Console()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


@click.command()
@click.option("--folder",   required=True, help="Folder containing video files")
@click.option("--output",   default="outputs/batch", help="Output directory")
@click.option("--config",   default="configs/pipeline_config.yaml", help="Config path")
@click.option("--pattern",  default="*", help="File glob pattern (e.g. '*.mp4')")
@click.option("--max-frames", default=0, type=int, help="Max frames per video (0=all)")
def main(folder, output, config, pattern, max_frames):
    """Batch process multiple industrial videos."""

    folder_path = Path(folder)
    if not folder_path.exists():
        console.print(f"[red]Folder not found:[/red] {folder}")
        sys.exit(1)

    # Find all videos
    video_files = [
        f for f in folder_path.glob(pattern)
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        console.print(f"[yellow]No videos found in {folder} matching '{pattern}'[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold]Found {len(video_files)} videos to process[/bold]\n")

    results = []
    for i, video_path in enumerate(sorted(video_files), 1):
        console.print(f"[cyan]── [{i}/{len(video_files)}] {video_path.name}[/cyan]")
        start = time.time()

        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            "--video", str(video_path),
            "--config", config,
            "--output-dir", output,
        ]
        if max_frames:
            cmd += ["--max-frames", str(max_frames)]

        result = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - start
        success = result.returncode == 0

        results.append({
            "video": video_path.name,
            "status": "✓ OK" if success else "✗ FAILED",
            "time": f"{elapsed:.1f}s",
        })
        console.print()

    # Summary table
    table = Table(title="Batch Processing Summary")
    table.add_column("Video", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right")

    for r in results:
        color = "green" if "OK" in r["status"] else "red"
        table.add_row(r["video"], f"[{color}]{r['status']}[/{color}]", r["time"])

    console.print(table)
    console.print(f"\n[bold]Output directory:[/bold] {output}")


if __name__ == "__main__":
    main()
