# Video Compatibility Issues with OpenCV

## Why "Good" Videos Fail in OpenCV

Videos that play perfectly in VLC, QuickTime, or browsers can fail in OpenCV due to various technical reasons. This is because **video players are much more forgiving** than OpenCV's FFmpeg backend.

## Common Edge Cases

### 1. **Codec Issues** ⚠️ MOST COMMON

**Problem**: Modern codecs that OpenCV doesn't support well

- **H.265/HEVC**: Modern, efficient codec but requires additional FFmpeg compilation flags
- **VP9/AV1**: Web codecs (WebM) that may not be in your OpenCV build
- **Unusual codecs**: Anything other than H.264/MPEG-4

**Why it happens**:
- OpenCV is compiled without patent-encumbered codec support
- Your FFmpeg version may lack certain codec libraries
- License restrictions on certain codecs

**Solution**: Convert to H.264
```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a copy output.mp4
```

---

### 2. **Variable Frame Rate (VFR)** 🎥

**Problem**: Frame rate changes throughout the video

**Where it comes from**:
- Screen recordings (OBS, QuickTime screen capture)
- Phone videos (some Android devices)
- Game recordings
- Concatenated videos from different sources

**Why OpenCV fails**:
- OpenCV expects constant frame rate (CFR)
- `cv2.VideoCapture` can't handle dynamic frame timing
- Seeking becomes unpredictable

**How to detect**:
- `r_frame_rate` ≠ `avg_frame_rate` in ffprobe
- Videos recorded from phones often have this

**Solution**: Convert to constant frame rate
```bash
ffmpeg -i input.mp4 -vsync cfr -r 30 output.mp4
```

---

### 3. **Pixel Format Issues** 🎨

**Problem**: Unusual color encoding

**Common problematic formats**:
- `yuv422p` - 4:2:2 chroma subsampling (professional cameras)
- `yuv444p` - 4:4:4 no chroma subsampling (high-end cameras)
- `yuvj420p` - Full range YUV (JPEG color range)
- `yuv420p10le` - 10-bit color depth (HDR, professional)

**Why OpenCV fails**:
- OpenCV's conversion code is optimized for `yuv420p` (8-bit, 4:2:0)
- 10-bit and 12-bit formats require special handling
- Some pixel formats don't have conversion routines

**Solution**: Convert to yuv420p
```bash
ffmpeg -i input.mp4 -pix_fmt yuv420p output.mp4
```

---

### 4. **Complex GOP Structure** 🎬

**Problem**: Too many B-frames or unusual GOP patterns

**What are B-frames?**
- B-frames (bidirectional prediction) reference both past and future frames
- Require more buffering and complex decoding
- Common in high-compression videos

**Why OpenCV struggles**:
- Random access becomes expensive
- Decoding a single frame requires decoding multiple frames
- Memory management issues with deep B-frame stacks

**Solution**: Re-encode with simpler GOP
```bash
ffmpeg -i input.mp4 -c:v libx264 -bf 0 -g 30 output.mp4
```
(`-bf 0` = no B-frames, `-g 30` = keyframe every 30 frames)

---

### 5. **Audio Stream Problems** 🔊

**Problem**: Audio issues preventing video decode

**Common issues**:
- Multiple audio tracks (multilingual, commentary)
- Unusual audio codecs (Opus, FLAC, AC3)
- Corrupted audio streams
- Audio/video sync metadata issues

**Why it affects video**:
- FFmpeg demuxer processes both streams
- Audio decode errors can abort the entire process
- Some containers have audio-first ordering

**Solution**: Strip/replace audio
```bash
# Remove audio entirely
ffmpeg -i input.mp4 -an -c:v copy output.mp4

# Or convert audio to AAC
ffmpeg -i input.mp4 -c:v copy -c:a aac output.mp4
```

---

### 6. **Resolution/Aspect Ratio Issues** 📐

**Problem**: Non-standard dimensions or display aspect ratio metadata

**Examples**:
- Odd dimensions (e.g., 1919×1079 instead of 1920×1080)
- Non-divisible-by-2 dimensions (required for yuv420p)
- Display aspect ratio (DAR) vs storage aspect ratio (SAR) mismatch
- Rotation metadata (90°, 180°, 270°)

**Why OpenCV fails**:
- Some codecs require even dimensions
- Rotation metadata is often ignored
- Pixel aspect ratio calculations fail

**Solution**: Resize to standard resolution
```bash
ffmpeg -i input.mp4 -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2" output.mp4
```

---

### 7. **Container Format Issues** 📦

**Problem**: Container metadata or structure issues

**Common issues**:
- Missing moov atom (MP4 header) at file start
- Fragmented MP4 (fMP4) used for streaming
- Unusual MP4 variants (3GP, F4V)
- Container corruption that players can work around

**Why OpenCV fails**:
- OpenCV requires complete metadata upfront
- Can't handle streaming-optimized containers
- Less error recovery than video players

**Solution**: Remux with faststart
```bash
ffmpeg -i input.mp4 -c copy -movflags +faststart output.mp4
```

---

## Diagnostic Workflow

### Step 1: Run the diagnostic script

```bash
cd /Users/macsolution/Desktop/lip_sync_service
python scripts/diagnose_videos.py "data/AVLips1 2/" --limit 50
```

This will:
- Test each video with OpenCV
- Extract codec/format information with ffprobe
- Identify specific issues
- Show frequency of each problem

### Step 2: Review the results

Look for patterns:
- If most videos have the same issue → batch convert all
- If only a few videos fail → consider removing them
- If all videos fail → OpenCV installation issue

### Step 3: Convert problematic videos

```bash
# Convert all videos to OpenCV-friendly format
python scripts/convert_videos.py "data/AVLips1 2/0_real/" --output data/converted/0_real/ --workers 2

python scripts/convert_videos.py "data/AVLips1 2/1_fake/" --output data/converted/1_fake/ --workers 2
```

Or manually convert specific videos:

```bash
# Universal conversion (fixes most issues)
ffmpeg -i input.mp4 \
  -c:v libx264 \
  -preset fast \
  -crf 23 \
  -pix_fmt yuv420p \
  -vsync cfr \
  -r 30 \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  output.mp4
```

---

## Quick Reference

| Issue | Detection | Fix Command |
|-------|-----------|-------------|
| H.265 codec | `codec_name: hevc` | `ffmpeg -i in.mp4 -c:v libx264 out.mp4` |
| Variable FPS | `r_frame_rate ≠ avg_frame_rate` | `ffmpeg -i in.mp4 -vsync cfr -r 30 out.mp4` |
| Unusual pixel format | `pix_fmt: yuv422p` | `ffmpeg -i in.mp4 -pix_fmt yuv420p out.mp4` |
| Multiple audio tracks | `Audio #0:1, #0:2` | `ffmpeg -i in.mp4 -map 0:v -map 0:a:0 out.mp4` |
| Odd dimensions | `width: 1919` | `ffmpeg -i in.mp4 -vf scale=1920:1080 out.mp4` |
| Missing moov atom | OpenCV can't open | `ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4` |

---

## Prevention

When creating/acquiring training videos:

1. **Use standard settings**:
   - H.264 codec (`-c:v libx264`)
   - 30 fps constant frame rate
   - yuv420p pixel format
   - 1280×720 or 1920×1080 resolution

2. **Test before training**:
   ```python
   import cv2
   cap = cv2.VideoCapture("video.mp4")
   print(f"Opens: {cap.isOpened()}")
   ret, frame = cap.read()
   print(f"Reads: {ret}")
   ```

3. **Validate with ffprobe**:
   ```bash
   ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,r_frame_rate video.mp4
   ```

---

## When Nothing Works

If videos still fail after conversion:

1. **Check OpenCV build**:
   ```python
   import cv2
   print(cv2.getBuildInformation())
   # Look for FFmpeg: YES
   ```

2. **Try different OpenCV version**:
   ```bash
   pip uninstall opencv-python opencv-python-headless
   pip install opencv-python==4.8.0.76
   ```

3. **Use system FFmpeg**:
   Some OpenCV builds use system FFmpeg instead of bundled version.
   ```bash
   brew install ffmpeg
   # Then reinstall opencv-python
   ```

4. **Last resort - extract frames manually**:
   ```bash
   # Extract all frames as images
   ffmpeg -i video.mp4 -vf fps=30 frames/%06d.jpg
   
   # Load images in Python instead
   ```
