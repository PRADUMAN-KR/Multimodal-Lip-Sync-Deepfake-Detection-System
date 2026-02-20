# How to Check if Training is Using GPU or CPU

## Quick Check

When training starts, you'll see:

```
================================================================================
🚀 Training Configuration:
  Device: mps
  ✅ Using Apple Silicon GPU (MPS)
  Batch size: 4
  Learning rate: 0.0001
================================================================================
```

**What to look for:**
- ✅ `Device: mps` = **Apple GPU** (M1/M2/M3) - **FASTEST**
- ✅ `Device: cuda` = **NVIDIA GPU** - **FAST**
- ⚠️ `Device: cpu` = **CPU only** - **SLOW**

---

## About Those Messages

### "TensorFlow Lite XNNPACK delegate for CPU"
This is **NOT** about PyTorch training! This is MediaPipe (face detection) using CPU, which is **normal and fine**. MediaPipe preprocessing is fast enough on CPU.

### "Class CVSlider is implemented in both..."
This is just a **warning** about duplicate OpenCV libraries (one from `opencv-python`, one bundled with MediaPipe). It's harmless and won't affect training.

---

## What Actually Matters

**PyTorch model training** is what uses GPU/CPU. Look for:

```
Model moved to device: mps  ← This means GPU!
```

Or check your training speed:
- **GPU (MPS)**: ~100-500 batches/minute
- **CPU**: ~10-50 batches/minute

---

## Force GPU Usage

If you see CPU but want GPU:

```bash
python -m app.training.train --device mps ...
```

Or check if MPS is available:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## Summary

- **MediaPipe using CPU** = Normal (preprocessing)
- **PyTorch using MPS** = Your model training is on GPU ✅
- **PyTorch using CPU** = Slow training ⚠️

The verbose output will now clearly show which device PyTorch is using!
