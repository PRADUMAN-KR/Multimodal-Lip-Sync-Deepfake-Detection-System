# Optional local tools

## ffmpeg (no Homebrew)

If you don't have Homebrew, you can use a static ffmpeg binary here:

1. Download a static build for macOS:
   - **Intel Mac:** https://evermeet.cx/ffmpeg/ (get the **ffmpeg** zip, not ffprobe)
   - Or search for "ffmpeg static build macOS" and pick a trusted source

2. Unzip and move the `ffmpeg` executable into this folder:
   ```bash
   mv ~/Downloads/ffmpeg lip_sync_service/tools/ffmpeg
   chmod +x lip_sync_service/tools/ffmpeg
   ```

3. Run `python3 scripts/download_avspeech_clips.py` again; it will use `tools/ffmpeg` automatically.
