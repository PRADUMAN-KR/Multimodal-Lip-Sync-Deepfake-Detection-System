# Fine-Tuning Guide for Beginners

## What is Fine-Tuning?

**Fine-tuning** is like teaching an already-trained model to do a specific task better. Think of it like this:
- **Initial training**: Teaching a model from scratch (slow, needs lots of data)
- **Fine-tuning**: Taking a pre-trained model and adjusting it for your specific needs (faster, needs less data)

For your lip-sync detection model:
1. **First**, train the model on your data (creates `weights/best_model.pth`)
2. **Then**, fine-tune it to make it even better (updates `weights/best_model.pth`)

---

## Step-by-Step Guide

### Prerequisites

1. **Install MediaPipe** (for face detection):
   ```bash
   pip install mediapipe>=0.10
   ```

2. **Make sure you have training data**:
   ```
   data/AVLips1 2/
      0_real/    (authentic videos)
      1_fake/    (AI-manipulated videos)
   ```

---

## Option 1: Simple Fine-Tuning (Recommended for Beginners)

This is the easiest way - just run one command:

```bash
cd /Users/macsolution/Desktop/lip_sync_service

python -m app.training.finetune \
  --data-dir "data/AVLips1 2" \
  --pretrained weights/best_model.pth \
  --epochs 30 \
  --freeze-epochs 10 \
  --batch-size 4 \
  --use-augmentation \
  --device mps
```

**What this does:**
- Uses your existing `weights/best_model.pth` as starting point
- Trains for 30 epochs total
- First 10 epochs: Only trains classifier (encoders frozen)
- Next 20 epochs: Fine-tunes everything
- Uses data augmentation for better results
- Saves improved model to `weights/best_model.pth`

**If you don't have `weights/best_model.pth` yet**, skip to "Option 2" below.

---

## Option 2: Start from Scratch, Then Fine-Tune

If you're starting fresh, do this in two steps:

### Step 1: Initial Training

```bash
cd /Users/macsolution/Desktop/lip_sync_service

python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 50 \
  --batch-size 4 \
  --device mps
```

**What happens:**
- Trains model from scratch
- Saves checkpoints to `weights/` folder
- Creates `weights/best_model.pth` (best model)
- Creates `weights/latest.pth` (latest epoch)

**Wait for this to finish** (might take 30 minutes to several hours depending on your data size).

### Step 2: Fine-Tune the Model

After Step 1 completes, run:

```bash
python -m app.training.finetune \
  --data-dir "data/AVLips1 2" \
  --pretrained weights/best_model.pth \
  --epochs 30 \
  --freeze-epochs 10 \
  --batch-size 4 \
  --use-augmentation \
  --device mps
```

**What happens:**
- Takes your trained model from Step 1
- Fine-tunes it further
- Improves accuracy and robustness
- Saves improved model back to `weights/best_model.pth`

---

## Understanding the Parameters

### Basic Parameters

| Parameter | What it does | Recommended Value |
|-----------|--------------|-------------------|
| `--data-dir` | Path to your training data | `"data/AVLips1 2"` |
| `--pretrained` | Path to pre-trained weights | `weights/best_model.pth` |
| `--epochs` | Total training epochs | `30` (fine-tuning) or `50` (initial) |
| `--batch-size` | How many videos per batch | `4` (start small), increase if you have more GPU memory |
| `--device` | Which device to use | `mps` (M1 Mac), `cuda` (NVIDIA GPU), or `cpu` |

### Fine-Tuning Specific Parameters

| Parameter | What it does | Recommended Value |
|-----------|--------------|-------------------|
| `--freeze-epochs` | How many epochs to keep encoders frozen | `10` |
| `--lr` | Learning rate for classifier | `1e-4` (0.0001) |
| `--lr-encoder` | Learning rate for encoders (when unfrozen) | `1e-5` (0.00001) |
| `--use-augmentation` | Enable data augmentation | Always use this! |

---

## What You'll See During Training

### Phase 1: Frozen Encoders (First 10 epochs)

```
Epoch 0 [Train]: loss=0.6234
Epoch 0: Train Loss=0.6234, Val Loss=0.5891, Val Acc=0.7123
Epoch 1 [Train]: loss=0.5892
Epoch 1: Train Loss=0.5892, Val Loss=0.5543, Val Acc=0.7456
...
```

**What's happening:**
- Only the classifier is learning
- Encoders are frozen (not changing)
- Loss should decrease slowly
- Validation accuracy should improve

### Phase 2: Unfrozen Encoders (Epochs 11-30)

```
Epoch 10 [Train]: loss=0.4123
Epoch 10: Train Loss=0.4123, Val Loss=0.3891, Val Acc=0.8234
Unfrozen visual encoder
Unfrozen audio encoder
Epoch 11 [Train]: loss=0.4012
Epoch 11: Train Loss=0.4012, Val Loss=0.3789, Val Acc=0.8345
...
```

**What's happening:**
- Now everything is learning
- Loss might jump a bit, then decrease
- Validation accuracy should continue improving
- Model is getting better at detecting manipulation

---

## How to Know It's Working

### Good Signs ✅

1. **Loss decreases**: Training loss goes down over epochs
2. **Validation accuracy increases**: Should reach 80%+ eventually
3. **No crashes**: Training completes without errors
4. **Checkpoints saved**: You see `weights/best_model.pth` updated

### Warning Signs ⚠️

1. **Loss increases**: Model might be overfitting (reduce learning rate)
2. **Accuracy stuck**: Model not learning (check data, increase epochs)
3. **Out of memory**: Reduce `--batch-size` (try 2 or 4)
4. **Very slow**: Normal for first training, fine-tuning should be faster

---

## Common Issues and Solutions

### Issue 1: "Model weights not found"

**Problem**: `weights/best_model.pth` doesn't exist

**Solution**: Run initial training first (Option 2, Step 1)

### Issue 2: "Out of memory" or "CUDA out of memory"

**Problem**: Batch size too large

**Solution**: Reduce batch size:
```bash
--batch-size 2  # Instead of 4 or 8
```

### Issue 3: "No module named 'mediapipe'"

**Problem**: MediaPipe not installed

**Solution**: Install it:
```bash
pip install mediapipe>=0.10
```

### Issue 4: Training is very slow

**Problem**: Normal for first training, but might be too slow

**Solution**: 
- Use smaller dataset for testing
- Reduce `--epochs` for testing (e.g., `--epochs 5`)
- Make sure you're using GPU (`--device mps`)

### Issue 5: Accuracy not improving

**Problem**: Model stuck or data issues

**Solution**:
- Check your data labels are correct
- Increase `--epochs`
- Try different learning rates (`--lr 5e-5` or `--lr 2e-4`)
- Make sure you have enough data (at least 100 videos per class)

---

## Quick Start Checklist

- [ ] Install MediaPipe: `pip install mediapipe>=0.10`
- [ ] Check data exists: `ls data/AVLips1\ 2/0_real/` and `ls data/AVLips1\ 2/1_fake/`
- [ ] If no weights exist, run initial training first
- [ ] Run fine-tuning command
- [ ] Wait for training to complete
- [ ] Check `weights/best_model.pth` was updated
- [ ] Test your model: Restart service and test with a video

---

## Example: Complete Workflow

```bash
# 1. Navigate to project
cd /Users/macsolution/Desktop/lip_sync_service

# 2. Check if you have weights
ls weights/best_model.pth

# 3a. If weights exist, fine-tune directly:
python -m app.training.finetune \
  --data-dir "data/AVLips1 2" \
  --pretrained weights/best_model.pth \
  --epochs 30 \
  --freeze-epochs 10 \
  --batch-size 4 \
  --use-augmentation \
  --device mps

# 3b. If no weights exist, train first:
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 50 \
  --batch-size 4 \
  --device mps

# Then fine-tune:
python -m app.training.finetune \
  --data-dir "data/AVLips1 2" \
  --pretrained weights/best_model.pth \
  --epochs 30 \
  --freeze-epochs 10 \
  --batch-size 4 \
  --use-augmentation \
  --device mps

# 4. Test your improved model
python -m app.main
# Then test API with a video
```

---

## Tips for Beginners

1. **Start small**: Use `--epochs 5` for testing, then increase
2. **Monitor progress**: Watch the loss and accuracy values
3. **Save often**: Checkpoints are saved automatically
4. **Be patient**: Training takes time, especially first time
5. **Use augmentation**: Always include `--use-augmentation`
6. **Check your data**: Make sure labels are correct (0_real = authentic, 1_fake = manipulated)

---

## What Happens After Fine-Tuning?

Once fine-tuning completes:

1. **Improved model**: `weights/best_model.pth` is updated with better weights
2. **Better accuracy**: Model should detect manipulation more accurately
3. **More robust**: Works better with different angles, lighting, etc.
4. **Ready for production**: Can be used in your API service

To use the fine-tuned model:
```bash
# Restart your service (it will load the updated weights)
python -m app.main
```

The service automatically loads `weights/best_model.pth` when it starts!

---

## Need Help?

If you get stuck:
1. Check error messages carefully
2. Make sure all dependencies are installed
3. Verify your data structure is correct
4. Try reducing batch size if memory issues
5. Start with fewer epochs to test

Good luck with your fine-tuning! 🚀
