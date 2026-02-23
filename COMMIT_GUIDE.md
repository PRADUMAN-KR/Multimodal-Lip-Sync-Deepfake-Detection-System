# Git Commit Guide

This document lists what files should be committed to your GitHub repository.

## ✅ Files to COMMIT (Source Code & Documentation)

### Core Application Code
- `app/` directory (all Python source files)
  - `app/__init__.py`
  - `app/config.py`
  - `app/lifecycle.py`
  - `app/main.py`
  - `app/core/` (logger.py, device.py)
  - `app/models/` (all model files)
  - `app/preprocessing/` (face_detection.py, audio.py, video.py)
  - `app/inference/` (predictor.py)
  - `app/training/` (all training scripts)
  - `app/utils/` (file_manager.py)
  - `app/api/` (schemas.py, etc.)

### Scripts & Tools
- `scripts/` directory (all utility scripts)
- `tools/` directory
- `check_epoch.py`

### Configuration & Dependencies
- `requirements.txt` ✅
- `.gitignore` ✅ (just created)

### Documentation
- All `.md` files in root:
  - `CHECK_DEVICE.md`
  - `CLEAN_DATASET_GUIDE.md`
  - `FINETUNING_GUIDE.md`
  - `FIX_CRASH.md`
  - `FIX_RESOURCE_EXHAUSTION.md`
  - `QUICK_START_TRAINING.md`
  - `SETUP_PYTHON.md`
  - `VIDEO_COMPATIBILITY.md`
- `data/README*.md` files
- `weights/README.md`
- `app/training/README_FINETUNING.md`
- `tools/README.md`

## ❌ Files to IGNORE (Already in .gitignore)

### Virtual Environment
- `venv/` - Never commit virtual environments

### Python Cache
- `__pycache__/` directories
- `*.pyc` files

### Model Weights (Large Files)
- `weights/*.pth` (~135MB each)
- `weights/*.pt`
- Note: If you need to share models, use Git LFS or external storage

### Generated Results
- `results/` directory
- `*.png`, `*.json` in results/

### Data Files (Large Datasets)
- `data/*/` subdirectories (video clips, datasets)
- `avspeech_test.csv` (9MB - consider Git LFS if needed)

### System Files
- `.DS_Store` (macOS)

## 📝 Recommended Git Commands

```bash
# Initialize repository (if not already done)
git init

# Add all files that should be committed
git add .

# Review what will be committed
git status

# Create initial commit
git commit -m "Initial commit: Lip Sync Detection Service"

# Add remote repository
git remote add origin <your-github-repo-url>

# Push to GitHub
git push -u origin main
```

## ⚠️ Important Notes

1. **Model Files**: The `.pth` files in `weights/` are ~135MB each. GitHub has a 100MB file size limit. Consider:
   - Using Git LFS for large files
   - Hosting models on cloud storage (S3, Google Drive, etc.)
   - Adding download instructions in README

2. **Data Files**: Large datasets in `data/` should not be committed. Consider:
   - Using Git LFS for smaller datasets
   - External storage links in README
   - Dataset download scripts

3. **Results**: The `results/` directory contains generated outputs that can be recreated, so they're excluded.

4. **Virtual Environment**: Always exclude `venv/` - users should create their own with `requirements.txt`

## 🔍 Verify Before Committing

Before your first commit, run:
```bash
git status
```

This will show you exactly what will be committed. Make sure:
- ✅ All Python source files are included
- ✅ `requirements.txt` is included
- ✅ Documentation files are included
- ❌ No `venv/` files
- ❌ No `__pycache__/` directories
- ❌ No `.pth` model files
- ❌ No large data files
