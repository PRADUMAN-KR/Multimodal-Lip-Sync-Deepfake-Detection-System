#!/usr/bin/env python3
"""
Download a small subset of AVSpeech test clips from YouTube for testing lip-sync pipelines.

Your avspeech_test.csv has columns (no header):
  youtube_id, start_sec, end_sec, face_center_x, face_center_y

Requires: yt-dlp, ffmpeg
  pip install yt-dlp
  ffmpeg: Put a binary in tools/ffmpeg (or install via Homebrew / evermeet.cx static build)
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Prefer project-local ffmpeg, then Homebrew paths (Cursor/IDE terminals often don't have PATH set)
def _ffmpeg_path() -> str:
    root = Path(__file__).resolve().parent.parent
    local = root / "tools" / "ffmpeg"
    if local.exists():
        return str(local)
    for path in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(path).exists():
            return path
    return "ffmpeg"


def run(cmd: list[str], timeout: int = 300, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        timeout=timeout,
        text=True,
    )


def download_clip(
    youtube_id: str,
    start_sec: float,
    end_sec: float,
    out_path: Path,
    cookies_from_browser: str | None = None,
) -> bool:
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    duration_sec = end_sec - start_sec
    tmp_dir = out_path.parent / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_file = tmp_dir / f"tmp_{youtube_id}"

    # Try multiple format strategies to maximize compatibility
    format_strategies = [
        # Strategy 1: Best video+audio with mp4 preference
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
        # Strategy 2: Just best available (simpler fallback)
        "best[ext=mp4]/best",
        # Strategy 3: Any video format with audio
        "bestvideo*+bestaudio/best",
        # Strategy 4: Absolute fallback - whatever is available
        "best",
    ]

    downloaded = None
    last_error = None
    
    for strategy_idx, format_str in enumerate(format_strategies):
        # 1) Download full video with current format strategy
        dl_cmd = [
            "yt-dlp",
            "-f", format_str,
            "--no-playlist",
            "--merge-output-format", "mp4",
            "-o", str(tmp_file) + ".%(ext)s",
            "--no-check-certificate",
        ]
        # Only suppress output on retry attempts
        if strategy_idx > 0:
            dl_cmd.extend(["--quiet", "--no-warnings"])
            
        if cookies_from_browser:
            dl_cmd.extend(["--cookies-from-browser", cookies_from_browser])
        dl_cmd.append(url)
        
        try:
            r = run(dl_cmd, timeout=180)
            if r.returncode == 0:
                # Success! Find the downloaded file
                for f in tmp_dir.iterdir():
                    if f.name.startswith(f"tmp_{youtube_id}."):
                        downloaded = f
                        break
                if downloaded and downloaded.exists():
                    break  # Successfully downloaded, exit retry loop
            else:
                # Capture error message
                if r.stderr:
                    last_error = r.stderr.strip()
                    stderr_lower = last_error.lower()
                    
                    # Check if it's a permanent failure (private, removed, etc.)
                    if any(term in stderr_lower for term in ["private", "removed", "unavailable", "deleted", "members-only"]):
                        # Permanent failure, don't retry
                        print(f"\n    Error: {last_error[:200]}", file=sys.stderr)
                        return False
                    
                    # Only print format errors on first attempt
                    if strategy_idx == 0:
                        print(f"\n    Trying fallback formats... ({last_error[:100]})", file=sys.stderr, end="")
                        
                # Otherwise, try next format strategy
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            last_error = str(e)
            if strategy_idx == 0:  # Only print on first attempt
                print(f"\n    Error: {e}", file=sys.stderr)
            if isinstance(e, FileNotFoundError):
                return False  # yt-dlp not found, no point retrying
            continue
    
    if not downloaded or not downloaded.exists():
        if last_error:
            print(f"\n    Final error: {last_error[:200]}", file=sys.stderr)
        return False

    # 2) Trim to segment with ffmpeg (-ss before -i for fast seek)
    ffmpeg = _ffmpeg_path()
    try:
        r = run([
            ffmpeg,
            "-y",
            "-ss", str(start_sec),
            "-i", str(downloaded),
            "-t", str(duration_sec),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            str(out_path),
        ], timeout=60, capture=True)
        if r.returncode != 0:
            if r.stderr:
                # Extract meaningful error from ffmpeg output
                error_lines = [line for line in r.stderr.strip().split('\n') if 'error' in line.lower() or 'invalid' in line.lower()]
                if error_lines:
                    print(f"\n    ffmpeg error: {error_lines[-1][:200]}", file=sys.stderr)
                else:
                    print(f"\n    ffmpeg failed with code {r.returncode}", file=sys.stderr)
            downloaded.unlink(missing_ok=True)
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"\n    Error: {e}", file=sys.stderr)
        downloaded.unlink(missing_ok=True)
        return False
    finally:
        downloaded.unlink(missing_ok=True)

    success = out_path.exists() and out_path.stat().st_size > 0
    if not success:
        print(f"\n    Error: Output file is empty or wasn't created", file=sys.stderr)
    return success


def check_deps() -> bool:
    """Ensure yt-dlp and ffmpeg are available; print install hint if not."""
    if not shutil.which("yt-dlp"):
        print("Missing dependency: yt-dlp", file=sys.stderr)
        print("  Install with: pip install yt-dlp", file=sys.stderr)
        return False
    ffmpeg = _ffmpeg_path()
    # Absolute path = we found it in tools/ or Homebrew; else rely on PATH
    if Path(ffmpeg).is_absolute():
        if not os.access(ffmpeg, os.X_OK):
            print(f"ffmpeg at {ffmpeg} is not executable. Run: chmod +x {ffmpeg}", file=sys.stderr)
            return False
    elif not shutil.which("ffmpeg"):
        print("Missing dependency: ffmpeg", file=sys.stderr)
        print("  Option A: Install Homebrew (https://brew.sh), then: brew install ffmpeg", file=sys.stderr)
        print("  Option B: Download static build from https://evermeet.cx/ffmpeg/", file=sys.stderr)
        print("            Unzip and put the binary in: tools/ffmpeg (chmod +x tools/ffmpeg)", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download AVSpeech test clips from YouTube.")
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        choices=["chrome", "chromium", "edge", "firefox", "safari"],
        help="Use cookies from BROWSER (e.g. safari, chrome) to download age-restricted videos. You must be logged into YouTube in that browser.",
    )
    parser.add_argument("-n", "--num-clips", type=int, default=10, help="Number of clips to download (default: 10)")
    args = parser.parse_args()

    if not check_deps():
        sys.exit(1)
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "avspeech_test.csv"
    out_dir = project_root / "data" / "avspeech_test_clips"
    num_clips = args.num_clips

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cookies_from_browser:
        print(f"Using cookies from browser: {args.cookies_from_browser}")
    print(f"Downloading up to {num_clips} clips to {out_dir}")

    success_count = 0
    fail_count = 0
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= num_clips:
                break
            if len(row) < 5:
                continue
            yt_id, start_s, end_s, x, y = row[0], float(row[1]), float(row[2]), row[3], row[4]
            out_path = out_dir / f"clip_{i:04d}_{yt_id}.mp4"
            if out_path.exists():
                print(f"  [{i+1}/{num_clips}] exists: {out_path.name}")
                success_count += 1
                continue
            print(f"  [{i+1}/{num_clips}] {yt_id} [{start_s:.1f}-{end_s:.1f}s] ... ", end="", flush=True)
            if download_clip(yt_id, start_s, end_s, out_path, cookies_from_browser=args.cookies_from_browser):
                print("ok")
                success_count += 1
            else:
                print("failed")
                fail_count += 1

    print(f"\nDone. Downloaded/existing: {success_count}, Failed: {fail_count}")
    print(f"Use files in {out_dir} to test your lip-sync pipeline (video + audio).")


if __name__ == "__main__":
    main()