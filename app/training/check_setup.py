#!/usr/bin/env python3
"""
Quick setup checker - verifies your environment is ready for training.

Run this before training to catch issues early.
"""

import sys
from pathlib import Path

print("🔍 Checking setup for lip-sync training...\n")

# Check Python version
python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major != 3 or python_version.minor not in (10, 11, 12):
    print("❌ WARNING: MediaPipe typically requires Python 3.10, 3.11, or 3.12")
    print("   You're using Python 3.14 which may not work!")
else:
    print("✅ Python version OK")

# Check MediaPipe
print("\n📦 Checking MediaPipe...")
try:
    import mediapipe as mp
    print(f"   MediaPipe version: {getattr(mp, '__version__', 'unknown')}")
    
    if hasattr(mp, "solutions"):
        print("✅ MediaPipe has 'solutions' module")
        try:
            face_mesh = mp.solutions.face_mesh
            print("✅ Face mesh module accessible")
        except Exception as e:
            print(f"❌ Face mesh module failed: {e}")
    else:
        print("❌ MediaPipe does NOT have 'solutions' module")
        print("   This means face detection will fail!")
        print("   Fix: Use Python 3.11 venv and reinstall mediapipe")
except ImportError as e:
    print(f"❌ MediaPipe not installed: {e}")
    print("   Fix: pip install mediapipe>=0.10")

# Check data directory
print("\n📁 Checking data directory...")
data_dir = Path("data/AVLips1 2")
if data_dir.exists():
    real_dir = data_dir / "0_real"
    fake_dir = data_dir / "1_fake"
    
    real_count = len(list(real_dir.glob("*.mp4"))) if real_dir.exists() else 0
    fake_count = len(list(fake_dir.glob("*.mp4"))) if fake_dir.exists() else 0
    
    print(f"   Data directory: {data_dir}")
    print(f"   Real videos: {real_count}")
    print(f"   Fake videos: {fake_count}")
    
    if real_count > 0 and fake_count > 0:
        print(f"✅ Data looks good ({real_count + fake_count} total videos)")
    else:
        print("❌ No videos found in 0_real/ or 1_fake/")
else:
    print(f"❌ Data directory not found: {data_dir}")

# Check PyTorch
print("\n🔥 Checking PyTorch...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple GPU) available")
    else:
        print("⚠️  MPS not available (will use CPU)")
except ImportError:
    print("❌ PyTorch not installed")

print("\n" + "="*50)
print("Setup check complete!")
print("="*50)
