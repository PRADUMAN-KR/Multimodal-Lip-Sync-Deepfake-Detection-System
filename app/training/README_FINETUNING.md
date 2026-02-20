# Production-Grade Fine-Tuning Guide

## Overview

This guide covers fine-tuning your lip-sync manipulation detection model for production use, including:
- **Multi-angle face detection** (frontal, profile, etc.)
- **Multi-subject support** (multiple faces per video)
- **Transfer learning** from pre-trained weights
- **Data augmentation** for robustness

---

## Features

### 1. **Production-Grade Face Detection**
- Uses **MediaPipe Face Mesh** for robust face detection
- Supports **multiple angles** (not just frontal)
- **Multi-face tracking** across frames
- Automatic fallback to center crop if no face detected

### 2. **Multi-Face Support**
- Detects up to 5 faces per frame (configurable)
- Tracks faces across frames using IoU-based matching
- Selects longest track (most consistent face) by default
- Can be configured to select largest face or first detected

### 3. **Fine-Tuning Strategy**
- **Phase 1**: Train classifier with frozen encoders (transfer learning)
- **Phase 2**: Fine-tune encoders with lower learning rate
- **Progressive unfreezing** for stable training

### 4. **Data Augmentation**
- **Temporal**: Speed variation (0.9x - 1.1x)
- **Spatial**: Rotation (±15°), horizontal flip, color jitter
- **Noise**: Gaussian noise for robustness

---

## Installation

Install additional dependencies:

```bash
pip install mediapipe>=0.10
```

---

## Training Workflow

### Step 1: Initial Training (if starting from scratch)

```bash
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 50 \
  --batch-size 8 \
  --device mps
```

This creates `weights/best_model.pth` with initial weights.

### Step 2: Fine-Tuning for Production

```bash
python -m app.training.finetune \
  --data-dir "data/AVLips1 2" \
  --pretrained weights/best_model.pth \
  --epochs 30 \
  --freeze-epochs 10 \
  --batch-size 8 \
  --lr 1e-4 \
  --lr-encoder 1e-5 \
  --use-augmentation \
  --device mps
```

**Parameters:**
- `--pretrained`: Path to pre-trained weights (from Step 1 or external)
- `--freeze-epochs`: Epochs with frozen encoders (Phase 1)
- `--lr`: Learning rate for classifier/fusion layers
- `--lr-encoder`: Lower LR for encoders when unfrozen (Phase 2)
- `--use-augmentation`: Enable data augmentation

---

## Multi-Face and Multi-Angle Support

### Face Detection Configuration

The preprocessing automatically handles:
- **Multiple angles**: MediaPipe detects faces at various angles
- **Multiple faces**: Tracks up to 5 faces, selects best one
- **Robust tracking**: IoU-based matching across frames

### Customization

In `app/preprocessing/video.py`, `preprocess_video()` accepts:
- `use_face_detection=True`: Enable MediaPipe (default)
- `max_faces=1`: Maximum faces to detect
- `select_strategy="longest"`: How to select face ("longest", "largest", "first")

### For Multiple Subjects

To process multiple faces in the same video:

1. **Option A**: Process each face separately
   ```python
   # In your custom script
   faces = detect_faces(frame)
   for face in faces:
       crop = crop_mouth_region(frame, face)
       # Process each face
   ```

2. **Option B**: Modify dataset to return multiple crops
   - Update `LipSyncDataset` to return list of crops
   - Batch multiple faces together

---

## Fine-Tuning Best Practices

### 1. **Transfer Learning**
- Start with pre-trained weights (your own or external)
- Freeze encoders initially, train classifier
- Gradually unfreeze for fine-tuning

### 2. **Learning Rates**
- **Classifier/Fusion**: `1e-4` (higher)
- **Encoders**: `1e-5` (lower, when unfrozen)
- Use learning rate scheduling (ReduceLROnPlateau)

### 3. **Data Augmentation**
- Enable for training: `--use-augmentation`
- Disable for validation (handled automatically)
- Helps with multi-angle robustness

### 4. **Batch Size**
- Start with 8, increase if memory allows
- Larger batches = more stable gradients
- Adjust based on GPU memory

### 5. **Validation**
- Use 20% of data for validation
- Monitor validation accuracy, not just loss
- Save best model based on validation loss

---

## Production Deployment

After fine-tuning, your model will:
- ✅ Detect faces at multiple angles
- ✅ Handle multiple subjects (selects best face)
- ✅ Be robust to variations (augmentation-trained)
- ✅ Detect AI manipulation artifacts
- ✅ Work with real-world video quality

### Testing

Test with your production videos:

```bash
# Start service
python -m app.main

# Test API
curl -X POST "http://localhost:8000/api/lip-sync" \
  -F "video_file=@test_video.mp4"
```

Response includes:
- `is_real`: Authentic video
- `is_fake`: AI-manipulated
- `confidence`: Probability of being real
- `manipulation_probability`: Probability of manipulation

---

## Troubleshooting

### Face Detection Fails
- Falls back to center crop automatically
- Check video quality (resolution, lighting)
- Ensure faces are visible

### Low Accuracy
- Increase training data
- Use more augmentation
- Fine-tune longer (more epochs)
- Check data quality (labels correct?)

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable half precision (`use_half_precision=True`)

---

## Next Steps

1. **Collect diverse data**: Multiple angles, lighting, subjects
2. **Fine-tune iteratively**: Train → Evaluate → Adjust → Repeat
3. **Monitor production**: Track accuracy on real videos
4. **Continuous improvement**: Add new data, retrain periodically

Your model is now production-ready with multi-angle and multi-subject support! 🚀
