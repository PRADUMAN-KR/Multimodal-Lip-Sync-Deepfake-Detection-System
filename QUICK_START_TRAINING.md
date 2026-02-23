# Quick Start: Training Commands

## Step 1: Activate Python 3.11 Virtual Environment

```bash
cd /Users/macsolution/Desktop/lip_sync_service

# Activate venv (if it exists)
source venv/bin/activate

# Verify Python version (MUST show 3.11.x)
python --version
```

**If you see Python 3.11.x**, you're good!  
**If you see Python 3.14 or other**, you need to create/use the venv:

```bash
# Create venv with Python 3.11 (if not exists)
/opt/homebrew/bin/python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 2: First-Time Training (Initial Training)

```bash
# Make sure venv is activated (you should see (venv) in your prompt)
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 50 \
  --batch-size 4 \
  --device mps
```

**What this does:**
- Trains your model from scratch
- Saves checkpoints to `weights/` folder
- Creates `weights/best_model.pth` (best model)
- Creates `weights/latest.pth` (latest epoch)

**Expected output:**
```
[INFO] Using device: mps
[INFO] Train samples: 6082, Val samples: 1520
[INFO] Model created with 5,208,321 parameters
Epoch 0 [Train]: loss=0.6234
Epoch 0: Train Loss=0.6234, Val Loss=0.5891, Val Acc=0.7123
...
```

**Wait for this to complete** (may take 30 minutes to several hours depending on your data size).

---

## Step 3: Fine-Tuning (After Initial Training)

Once Step 2 completes and you have `weights/best_model.pth`, run:

```bash
# Still in venv
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
- Takes your trained model from Step 2
- Fine-tunes it further (improves accuracy)
- Updates `weights/best_model.pth` with improved weights

---

## Complete Workflow (Copy-Paste Ready)

```bash
# 1. Navigate to project
cd /Users/macsolution/Desktop/lip_sync_service

# 2. Activate venv
source venv/bin/activate

# 3. Verify Python 3.11
python --version

# 4. First-time training
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 50 \
  --batch-size 4 \
  --device mps

# 5. After training completes, fine-tune
python -m app.training.finetune \
  --data-dir "data/AVLips1 2" \
  --pretrained weights/best_model.pth \
  --epochs 30 \
  --freeze-epochs 10 \
  --batch-size 4 \
  --use-augmentation \
  --device mps
```

---

## Troubleshooting

### "No module named 'app'"
**Fix:** Make sure you're in `/Users/macsolution/Desktop/lip_sync_service` directory.

### "Python 3.14" instead of 3.11
**Fix:** Activate venv: `source venv/bin/activate`

### "MediaPipe has no attribute 'solutions'"
**Fix:** You're using Python 3.14. Activate venv with Python 3.11.

### "Out of memory"
**Fix:** Reduce batch size: `--batch-size 2`

### Training is slow
**Normal** for first training. Fine-tuning will be faster.

---

## Quick Reference

| Command | When to Use |
|---------|-------------|
| `source venv/bin/activate` | Always before training |
| `python -m app.training.train` | First time training |
| `python -m app.training.finetune` | After you have weights |

---

**Remember:** Always activate venv first, then use `python` (not `python3`)!
