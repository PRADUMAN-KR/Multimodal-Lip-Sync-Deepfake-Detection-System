#!/usr/bin/env python3
"""
Thin wrapper: runs validate_pipeline.py in preprocessed (Zarr/NPY/LMDB) mode.

Use this for backward compatibility, or call validate_pipeline.py directly:

    python scripts/validate_pipeline.py --preprocessed_dir ./data/precomputed --storage_format zarr --model weights/best_model.pth --output_dir ./results
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_SCRIPT = ROOT / "scripts" / "validate_pipeline.py"

if __name__ == "__main__":
    args = sys.argv[1:]
    # If no --preprocessed_dir, treat first positional as preprocessed_dir (legacy)
    if args and not any(s.startswith("--preprocessed_dir") for s in args):
        for i, a in enumerate(args):
            if not a.startswith("-") and "=" not in a:
                args = [f"--preprocessed_dir={a}"] + args[:i] + args[i + 1:]
                break
    argv = [sys.executable, str(PIPELINE_SCRIPT)] + args
    sys.exit(subprocess.run(argv, cwd=ROOT).returncode)
