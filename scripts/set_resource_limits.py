#!/usr/bin/env python3
"""
Set environment variables to prevent resource exhaustion during training.

This script sets limits on threads created by OpenCV, MediaPipe, and other libraries
to prevent the 'pthread_create failed' error on macOS.

Usage:
    # Run training with resource limits
    python scripts/set_resource_limits.py python -m app.training.train --data-dir "data/AVLips1 2" --epochs 20 --batch-size 4
"""

import os
import sys
import subprocess

# Set environment variables to limit threading
os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS threads
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # NumExpr threads

# Limit OpenCV threads
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '1'  # Use FFmpeg backend
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'threads;1'  # Limit FFmpeg threads

# MediaPipe threading (experimental)
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU acceleration (reduces threads)

# PyTorch threading
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'  # Don't spin threads waiting

print("=" * 80)
print("🔧 Resource Limits Set")
print("=" * 80)
print(f"  OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
print(f"  MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")
print(f"  OPENCV threads: Limited")
print(f"  MediaPipe GPU: Disabled")
print("=" * 80)
print()

if len(sys.argv) > 1:
    # Run the command with these environment variables
    cmd = sys.argv[1:]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)
else:
    print("Usage:")
    print(f"  python {sys.argv[0]} <command>")
    print()
    print("Example:")
    print(f'  python {sys.argv[0]} python -m app.training.train --data-dir "data/AVLips1 2" --epochs 20 --batch-size 4')
