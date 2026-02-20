# Dataset Cleaning Guide

## Why Pre-filter Corrupt Videos?

Pre-filtering corrupt videos before training offers several advantages:
- ✅ **Faster training** - No time wasted trying to load corrupt files
- ✅ **Cleaner dataset** - Know exactly which videos are good
- ✅ **Better tracking** - Keep corrupt videos for inspection/repair
- ✅ **Reproducible** - Same dataset across multiple training runs

## Quick Start

### Step 1: Dry Run (Preview)

First, see what would be moved **without actually moving anything**:

```bash
cd /Users/macsolution/Desktop/lip_sync_service

python scripts/filter_corrupt_videos.py "data/AVLips1 2/" --dry-run
```

This will:
- Test all videos with OpenCV
- Show which videos would be moved
- Display why each video is considered corrupt
- **NOT modify any files**

### Step 2: Actually Clean the Dataset

If the dry run looks good, run it for real:

```bash
python scripts/filter_corrupt_videos.py "data/AVLips1 2/"
```

This will:
- Move corrupt videos to `data/AVLips1 2/corruptedclips/`
- Preserve directory structure (0_real, 1_fake)
- Generate a detailed report at `data/AVLips1 2/corrupt_videos_report.txt`

### Step 3: Review the Report

Check what was moved:

```bash
cat "data/AVLips1 2/corrupt_videos_report.txt"
```

The report includes:
- Total videos scanned
- Number of valid vs corrupt
- Full list of corrupt videos with reasons
- Original and new locations

### Step 4: Start Training with Clean Data

```bash
python -m app.training.train \
  --data-dir "data/AVLips1 2" \
  --epochs 50 \
  --batch-size 8 \
  --val-split 0.2 \
  --lr 0.0001
```

## Advanced Options

### Custom Corrupt Directory

Move corrupt videos to a different location:

```bash
python scripts/filter_corrupt_videos.py "data/AVLips1 2/" \
  --corrupt-dir "/path/to/corrupt_archive"
```

### Test More Frames

By default, the script tests the first 10 frames. To be more thorough:

```bash
python scripts/filter_corrupt_videos.py "data/AVLips1 2/" \
  --test-frames 30
```

**Note**: More frames = slower scanning, but more accurate

### Custom Report Location

```bash
python scripts/filter_corrupt_videos.py "data/AVLips1 2/" \
  --report "reports/corruption_analysis.txt"
```

## What Gets Flagged as Corrupt?

The script identifies videos as corrupt if:

1. **Cannot open** - `cv2.VideoCapture` fails
2. **Cannot read first frame** - File opens but no frames readable
3. **Consecutive read failures** - 3+ consecutive frames fail to decode
4. **Too few frames readable** - Less than 50% of test frames succeed
5. **Exceptions during processing** - Any crash/error during OpenCV operations

## Directory Structure After Cleaning

**Before:**
```
data/AVLips1 2/
├── 0_real/
│   ├── 0.mp4
│   ├── 1.mp4
│   ├── 568.mp4  (corrupt)
│   └── ...
└── 1_fake/
    ├── 0.mp4
    ├── 910.mp4  (corrupt)
    └── ...
```

**After:**
```
data/AVLips1 2/
├── 0_real/
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...  (only valid videos)
├── 1_fake/
│   ├── 0.mp4
│   └── ...  (only valid videos)
├── corruptedclips/  (NEW)
│   ├── 0_real/
│   │   └── 568.mp4
│   └── 1_fake/
│       └── 910.mp4
└── corrupt_videos_report.txt  (NEW)
```

## Interpreting the Report

### Example Report Section:
```
File: 0_real/568.mp4
Reason: Cannot initialize the conversion context!
Original Path: /Users/.../data/AVLips1 2/0_real/568.mp4
Moved To: /Users/.../data/AVLips1 2/corruptedclips/0_real/568.mp4
```

This tells you:
- **File**: Relative path in your dataset
- **Reason**: Why OpenCV couldn't read it
- **Original Path**: Where it was
- **Moved To**: Where it is now

## Common Reasons for Corruption

| Reason | Explanation | Can Be Fixed? |
|--------|-------------|---------------|
| `Cannot open video file` | File header is damaged | ❌ Usually not |
| `Cannot read first frame` | Missing codec data | ❌ Usually not |
| `Failed after reading N frames` | Partial corruption mid-file | ⚠️ Maybe (can extract good frames) |
| `Only read N/M frames` | Incomplete download | ❌ Need to re-download |
| `Exception: ...` | Severe file corruption | ❌ Usually not |

## Restoring Corrupt Videos

If you want to try fixing or restoring corrupt videos:

### Option 1: Re-download
If you still have access to the original source, re-download these specific files.

### Option 2: Re-encode with FFmpeg
Sometimes FFmpeg can repair slightly corrupt videos:

```bash
# Install FFmpeg if not already installed
brew install ffmpeg

# Try to re-encode a corrupt video
ffmpeg -i "corruptedclips/0_real/568.mp4" \
  -c:v libx264 \
  -c:a aac \
  "repaired/568.mp4"
```

### Option 3: Extract Valid Frames
For partially corrupt videos, extract the valid frames:

```bash
ffmpeg -i "corruptedclips/0_real/568.mp4" \
  -vsync 0 \
  -f image2 \
  "frames/568_%04d.jpg"
```

Then recreate the video from valid frames.

## Statistics Tracking

After cleaning multiple datasets, you might want to track corruption rates:

```bash
# Analyze multiple datasets
for dataset in data/*/; do
  echo "Analyzing: $dataset"
  python scripts/filter_corrupt_videos.py "$dataset" --dry-run | grep "Corrupt videos"
done
```

## Best Practices

1. **Always run --dry-run first** - Preview before moving files
2. **Keep the report** - Document what was removed
3. **Archive corrupt videos** - Don't delete immediately (might be fixable)
4. **Check dataset balance** - Ensure 0_real and 1_fake have similar corruption rates
5. **Validate after cleaning** - Run training on small subset to verify

## Troubleshooting

### Script runs slowly
- Reduce `--test-frames` (try 5 instead of 10)
- You're checking 7602 videos, this will take time (~5-10 minutes)

### Too many videos flagged as corrupt
- Increase `--test-frames` to be more lenient
- Check if OpenCV is properly installed
- Verify videos play in VLC/QuickTime

### Script crashes
- Ensure you have write permissions to the data directory
- Check disk space for the corruptedclips folder
- Try on a smaller subset first

## Integration with Training

After cleaning, your training script will:
- ✅ Load faster (no corrupt video attempts)
- ✅ Have more stable batches (fewer skipped samples)
- ✅ Show accurate progress (no hidden failures)
- ✅ Use all available data efficiently

The training code still has safeguards for edge cases, but with a clean dataset, you should rarely see batch skip warnings.
