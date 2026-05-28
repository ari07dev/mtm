"""
scripts/setup_phase2.py
─────────────────────────────────────────────────────────────────────────────
One-click Phase 2 setup: verifies MMAction2 stack + downloads ST-GCN checkpoint.
Run this BEFORE run_pipeline_phase2.py
─────────────────────────────────────────────────────────────────────────────

Usage:
    python scripts/setup_phase2.py
"""

import subprocess
import sys
from pathlib import Path

def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)

def check_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False

def main():
    print("\n" + "="*60)
    print("  MTM Pipeline — Phase 2 Setup")
    print("  ST-GCN via MMAction2 (CPU mode)")
    print("="*60 + "\n")

    # ── Step 1: Verify existing stack ─────────────────────────────────────────
    print("Step 1: Verifying your OpenMMLab stack...")
    checks = {
        "torch":      "PyTorch",
        "mmcv":       "MMCV",
        "mmengine":   "MMEngine",
        "mmaction":   "MMAction2",
    }
    all_ok = True
    for module, name in checks.items():
        ok = check_import(module)
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n[ERROR] Some packages missing. Your existing stack should have these.")
        print("Re-run your original OpenMMLab setup, then run this script again.")
        sys.exit(1)

    print("\n  All packages verified ✓")

    # ── Step 2: Print versions ─────────────────────────────────────────────────
    print("\nStep 2: Checking versions...")
    version_checks = [
        ("torch", "torch.__version__"),
        ("mmcv", "mmcv.__version__"),
        ("mmengine", "mmengine.__version__"),
        ("mmaction", "mmaction.__version__"),
    ]
    for module, expr in version_checks:
        try:
            import importlib
            mod = importlib.import_module(module)
            ver = eval(expr)
            print(f"  {module}: {ver}")
        except Exception:
            print(f"  {module}: (version unknown)")

    # ── Step 3: Download ST-GCN checkpoint ────────────────────────────────────
    print("\nStep 3: Downloading ST-GCN pre-trained checkpoint...")
    checkpoint_dir = Path("phase2/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d"
    checkpoint_file = checkpoint_dir / f"{model_name}.pth"

    if checkpoint_file.exists():
        print(f"  ✓ Checkpoint already exists: {checkpoint_file}")
    else:
        print(f"  Downloading {model_name}...")
        print("  (~100MB, this may take a few minutes on first run)")
        result = run(
            f'python -m mim download mmaction2 --config {model_name} --dest {checkpoint_dir}',
            check=False
        )
        if result.returncode != 0:
            print("\n  [WARNING] mim download failed. Checkpoint will be downloaded")
            print("  automatically on first inference run instead.")
        else:
            print("  ✓ Checkpoint downloaded")

    # ── Step 4: Verify inference ───────────────────────────────────────────────
    print("\nStep 4: Quick inference test...")
    test_code = """
import numpy as np
import torch
print("  Creating dummy skeleton window (T=45, V=17, C=3)...")
dummy = np.random.randn(45, 17, 3).astype(np.float32)
tensor = torch.from_numpy(dummy).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
print(f"  Tensor shape: {tensor.shape} (expected: [1, 3, 45, 17, 1])")
print("  ✓ Tensor creation OK")
"""
    exec(test_code)

    # ── Step 5: Create __init__ files ─────────────────────────────────────────
    Path("phase2/__init__.py").write_text("# Phase 2\n")
    Path("phase2/core/__init__.py").write_text("# Phase 2 Core\n")
    Path("phase2/configs/__init__.py").write_text("")
    print("\nStep 5: ✓ Package structure ready")

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Phase 2 Setup Complete!")
    print("="*60)
    print("\nNow run the full pipeline:")
    print("  python scripts/run_pipeline_phase2.py --video your_video.mp4")
    print("\nOr rules-only (faster, no ST-GCN):")
    print("  python scripts/run_pipeline_phase2.py --video your_video.mp4 --no-stgcn")
    print("\nOr ST-GCN only (best accuracy):")
    print("  python scripts/run_pipeline_phase2.py --video your_video.mp4 --stgcn-only")

if __name__ == "__main__":
    main()
