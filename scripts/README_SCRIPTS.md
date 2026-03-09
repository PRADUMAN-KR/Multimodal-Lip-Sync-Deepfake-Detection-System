# Scripts Overview – What to Keep When Sharing the Pipeline

## Required for inference & validation (keep these)

| Script | Purpose |
|--------|--------|
| **validate_pipeline.py** | Run full pipeline on a dataset (real/ + fake/ folders); outputs predictions.csv, metrics.json, confusion matrix, ROC curve. **Main validation script.** |
| **run_grid_eval.py** | Run inference on a directory or CSV of videos; compute Precision, Recall, F1. Good for GRID-style or any path+label list. |
| **fit_calibrator.py** | Fit calibration (temperature / Platt / isotonic) on a labelled validation set; reduces overconfidence. Optional but recommended for better metrics. |

---

## Optional – training (only if you share training)

| Script | Purpose |
|--------|--------|
| run_finetune.sh | Bash wrapper to run fine-tuning from best_model.pth. |
| quick_finetune.sh | Quick fine-tuning for beginners. |
| set_resource_limits.py | Sets thread limits for training (avoids pthread exhaustion on macOS). Used by the finetune shells. |

---

## Optional – data & debugging (not needed for “run the pipeline”)

| Script | Purpose |
|--------|--------|
| download_grid_corpus.py | Download GRID corpus from Zenodo (~16 GB). |
| download_avspeech_clips.py | Download AVSpeech test clips from YouTube. |
| convert_videos.py | Convert videos to OpenCV-friendly format (e.g. H.264). |
| diagnose_videos.py | Diagnose why OpenCV fails on certain videos. |
| filter_corrupt_videos.py | Find and move corrupt videos; generate corrupt report. |

---

## Minimal set to mail “the pipeline”

To share only what’s needed to **run inference and validate** (no training, no data downloads):

- **Keep:** `validate_pipeline.py`, `run_grid_eval.py`, `fit_calibrator.py`
- **Can drop:** everything else in `scripts/` if the recipient only needs to run the model and evaluate it.

If the recipient will also **train or fine-tune**, add: `run_finetune.sh`, `quick_finetune.sh`, `set_resource_limits.py`.
