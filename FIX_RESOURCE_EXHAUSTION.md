# Fix: Resource Exhaustion / pthread_create Failed

## The Problem

You're seeing this error during training:

```
Could not open codec h264, error: -35
pthread_create failed
Check failed: res == 0 (35 vs. 0) pthread_create failed
zsh: abort
```

**Root Cause**: MediaPipe and OpenCV create many threads for processing videos. After processing many videos (~133 batches), your system hits macOS thread limits and runs out of resources.

Error code `-35` = `EAGAIN` = "Resource temporarily unavailable" = Too many threads/processes

## The Solution

### Option 1: Use Resource Limits Script (Recommended)

Train with explicit resource limits:

```bash
cd /Users/macsolution/Desktop/lip_sync_service

python scripts/set_resource_limits.py \
  python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 20 \
  --batch-size 4 \
  --val-split 0.2 \
  --lr 0.0001
```

This script automatically:
- ✅ Limits OpenMP/MKL threads to 1
- ✅ Limits OpenCV threads
- ✅ Disables MediaPipe GPU (reduces thread usage)
- ✅ Sets passive thread waiting

### Option 2: Set Environment Variables Manually

```bash
cd /Users/macsolution/Desktop/lip_sync_service

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_VIDEOIO_PRIORITY_FFMPEG=1
export MEDIAPIPE_DISABLE_GPU=1

python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 20 \
  --batch-size 4 \
  --val-split 0.2 \
  --lr 0.0001
```

### Option 3: Reduce Batch Size

Smaller batches = fewer parallel MediaPipe instances = fewer threads:

```bash
# Try batch size 2 or 4 instead of 8
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 20 \
  --batch-size 2 \
  --val-split 0.2 \
  --lr 0.0001
```

## Recommended Configuration

For M1 MacBook Pro with 8GB-16GB RAM:

```bash
# Best configuration to avoid crashes
python scripts/set_resource_limits.py \
  python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 20 \
  --batch-size 4 \
  --val-split 0.2 \
  --lr 0.0001
```

**Why batch-size 4?**
- Batch size 8 = 8 videos processed in parallel = 8 MediaPipe instances = many threads
- Batch size 4 = 4 videos = 4 MediaPipe instances = fewer threads = less likely to crash
- Batch size 2 = safest but slower training

## What Changed

I've updated the code to:
1. ✅ **Lazy-initialize MediaPipe** - Only create FaceMesh when needed
2. ✅ **Proper cleanup** - Add `__del__` and context manager support to FaceDetector
3. ✅ **Resource limit script** - Easy way to set all environment variables
4. ✅ **Better error handling** - Clearer messages when resources run out

## Monitoring During Training

Watch for these signs of resource issues:

**Good** ✅:
```
Epoch 0 [Train]:  50%|████████| 714/1428 [30:12<28:45, 2.41s/it]
```
Smooth progress, no crashes

**Warning** ⚠️:
```
[ERROR:0@404.645] global cap_ffmpeg_impl.hpp:1969 retrieveFrame Cannot initialize...
```
OpenCV struggling but still working - consider reducing batch size

**Critical** ❌:
```
pthread_create failed
zsh: abort
```
Out of resources - MUST reduce batch size or set resource limits

## Troubleshooting

### Still Crashing After Setting Limits?

1. **Reduce batch size further**:
   ```bash
   --batch-size 2
   ```

2. **Restart your terminal** (clear any leftover processes):
   ```bash
   # Close terminal and open a new one
   cd /Users/macsolution/Desktop/lip_sync_service
   ```

3. **Check running processes**:
   ```bash
   # Kill any stuck Python processes
   pkill -9 Python
   ```

4. **Increase system limits** (advanced):
   ```bash
   # Check current limits
   ulimit -a
   
   # Increase max processes (temporary)
   ulimit -u 2048
   ```

### Training is Too Slow with batch-size 2?

Use **gradient accumulation** to simulate larger batches:

```bash
# Simulate batch-size 8 by accumulating 4 batches of size 2
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 20 \
  --batch-size 2 \
  --gradient-accumulation-steps 4
```

*(Note: You'd need to add this feature to the training script)*

### Memory Usage Still High?

Monitor with:

```bash
# Check memory usage during training
top -pid $(pgrep -f "python.*train")

# Or use Activity Monitor app
```

If memory keeps growing:
- Videos may have memory leaks
- Reduce `--test-frames` in dataset preprocessing
- Process videos in smaller chunks

## Performance Comparison

| Batch Size | Speed | Stability | Recommendation |
|-----------|-------|-----------|----------------|
| 8 | Fast | ❌ Crashes | Don't use |
| 4 | Good | ✅ Stable with limits | **Recommended** |
| 2 | Slower | ✅ Very stable | Use if batch-4 crashes |
| 1 | Slowest | ✅ Most stable | Last resort |

## Expected Training Time

With cleaned dataset (7,140 videos) and batch-size 4:

- **Training set**: ~5,712 videos (80%)
- **Validation set**: ~1,428 videos (20%)
- **Batches per epoch**: ~1,428 batches
- **Time per batch**: ~3-4 seconds
- **Time per epoch**: ~1.5-2 hours
- **Total (20 epochs)**: **30-40 hours**

**Tips to speed up**:
- Use batch-size 4 with resource limits (optimal)
- Close other apps to free RAM
- Let it run overnight
- Use checkpointing (saves progress every epoch)

## Verification

After setting resource limits, you should see:

```
🔧 Resource Limits Set
================================================================================
  OMP_NUM_THREADS: 1
  MKL_NUM_THREADS: 1
  OPENCV threads: Limited
  MediaPipe GPU: Disabled
================================================================================

[2026-02-16 18:00:00] [INFO] Starting training...
Epoch 0 [Train]:   0%|          | 0/1428 [00:00<?, ?it/s]
```

Training should now complete without crashes!

## Summary

**Quick Fix (Copy-Paste)**:

```bash
cd /Users/macsolution/Desktop/lip_sync_service

python scripts/set_resource_limits.py \
  python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 20 \
  --batch-size 4 \
  --val-split 0.2 \
  --lr 0.0001
```

This should prevent the `pthread_create failed` error and let training complete successfully!
