# AVSpeech test data for lip-sync

## Your CSV: `avspeech_test.csv`

**Format (no header):**
```text
youtube_id, start_sec, end_sec, face_center_x, face_center_y
```

- **youtube_id** – YouTube video ID (e.g. `u5MPyrRJPmc`)
- **start_sec**, **end_sec** – segment start/end in seconds
- **face_center_x**, **face_center_y** – face center in frame, normalized 0.0–1.0 (0,0 = top-left)

Your file has **~183k rows**. It is correct; AVSpeech does not ship video/audio, only these annotations.

## Getting video/audio for testing

Clips live on YouTube. To get a few for testing:

1. Install **yt-dlp** and **ffmpeg**:
   ```bash
   pip install yt-dlp
   # ffmpeg: https://ffmpeg.org or: brew install ffmpeg
   ```

2. Run the download script (downloads 10 clips by default):
   ```bash
   python scripts/download_avspeech_clips.py
   ```

3. Clips are saved under `data/avspeech_test_clips/` as MP4 (video + audio). Use these paths with your `load_video_frames()` and audio preprocessing. If your pipeline expects separate WAVs, extract with:
   ```bash
   ffmpeg -i data/avspeech_test_clips/clip_0000_xxx.mp4 -vn -acodec pcm_s16le -ar 16000 clip_0000.wav
   ```

**Note:** Some videos may be unavailable (deleted, region-locked, age-restricted). The script skips failures and continues.
