# Fix Training Crash: Corrupt Videos & MediaPipe Threading

## Problem

Training crashes with:
- `Could not open codec h264, error: -35` (OpenCV can't decode video)
- `pthread_create failed` (MediaPipe threading crash)
- Process aborts

## Root Causes

1. **Corrupt/unsupported video files** in your dataset
2. **MediaPipe threading conflicts** when processing certain videos
3. **OpenCV codec issues** (missing H.264 support)

## What I Fixed

✅ **Better error handling** - Videos that can't be decoded are now **skipped** instead of crashing  
✅ **MediaPipe crash protection** - Threading errors are caught and handled gracefully  
✅ **Video validation** - Checks video properties before processing  
✅ **Fallback crops** - If face detection fails on a frame, uses center crop instead of crashing

## What You Should Do

### Option 1: Clean Your Dataset (Recommended)

Remove corrupt videos:

```bash
cd /Users/macsolution/Desktop/lip_sync_service
source venv/bin/activate

# Test a few videos to find corrupt ones
python -c "
from pathlib import Path
import cv2

data_dir = Path('data/AVLips1 2')
corrupt = []

for vid in list(data_dir.rglob('*.mp4'))[:100]:  # Test first 100
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) <= 0:
        corrupt.append(vid)
        print(f'Corrupt: {vid}')
    cap.release()

print(f'Found {len(corrupt)} potentially corrupt videos')
"
```

### Option 2: Reduce Batch Size

Smaller batches = less memory pressure = fewer threading issues:

```bash
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 50 \
  --batch-size 2 \
  --device mps
```

### Option 3: Install FFmpeg Codec Support

```bash
# Install FFmpeg with H.264 support
brew install ffmpeg

# Or install opencv with better codec support
pip uninstall opencv-python -y
pip install opencv-python-headless
```

## After Fix: Training Should Continue

With the fixes I added:
- Corrupt videos are **skipped** (logged as warnings)
- MediaPipe crashes are **caught** (fallback to center crop)
- Training **continues** instead of aborting

You'll see warnings like:
```
⚠️ Skipping sample: corrupt_video.mp4 :: ValueError: Video decoding failed
```

But training will **continue** with the remaining valid videos.

## Monitor Progress

Watch for:
- **"Skipping sample"** warnings = corrupt videos being filtered out
- **Training continues** = Good! The fixes are working
- **Still crashing** = May need to clean dataset or reduce batch size further

---

## Summary

The crash was caused by:
1. A corrupt video that OpenCV couldn't decode
2. MediaPipe hitting a threading error when trying to process it

**Fixed:** Videos are now validated and MediaPipe errors are caught gracefully. Training should continue even if some videos are bad.
