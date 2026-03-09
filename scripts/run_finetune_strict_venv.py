#!/usr/bin/env python3
"""
Strict venv finetune runner.

Behavior:
- Requires repo-local venv at ./venv
- Fails immediately if venv is missing
- Re-executes itself with venv Python if not already running inside it
- Runs the exact finetune command configuration requested
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    venv_python = repo_root / "venv" / "bin" / "python"
    internal_flag = "--__inside-venv"

    if not venv_python.is_file():
        print("ERROR: venv Python not found.")
        print(f"Expected: {venv_python}")
        print("Create it first, then install dependencies:")
        print(f"  cd {repo_root}")
        print("  python3 -m venv venv")
        print("  ./venv/bin/pip install -r requirements.txt")
        return 1

    # Re-launch this script with venv Python if needed.
    if internal_flag not in sys.argv:
        current_python = Path(sys.executable).resolve()
        if current_python != venv_python.resolve():
            os.execv(
                str(venv_python),
                [str(venv_python), str(script_path), internal_flag],
            )

    data_dir = repo_root / "data" / "AVLips12"
    pretrained = repo_root / "weights" / "best_model.pth"

    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}")
        return 1
    if not pretrained.is_file():
        print(f"ERROR: pretrained checkpoint not found: {pretrained}")
        return 1

    os.chdir(repo_root)

    cmd = [
        str(venv_python),
        "-m",
        "app.training.finetune",
        "--data-dir",
        "data/AVLips12",
        "--pretrained",
        "weights/best_model.pth",
        "--epochs",
        "36",
        "--freeze-epochs",
        "8",
        "--batch-size",
        "8",
        "--lr",
        "2e-4",
        "--lr-encoder",
        "2e-5",
        "--contrastive-weight",
        "0.1",
        "--use-augmentation",
        "--early-stopping-patience",
        "8",
        "--log-every",
        "5",
    ]

    print("=" * 70)
    print("Running strict-venv finetune command")
    print(f"Repo root: {repo_root}")
    print(f"Python: {venv_python}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 70)

    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

