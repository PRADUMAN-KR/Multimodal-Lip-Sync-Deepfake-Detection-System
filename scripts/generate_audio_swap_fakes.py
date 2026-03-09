#!/usr/bin/env python3
"""
Generate fake videos using Audio Swap (multi-speaker confusion).

Takes: Video A (visual) + Audio from Video B → output fake video.
Output is saved under custom_fake/ with names like: video_0001_audio_0002.mp4

Usage:
    python scripts/generate_audio_swap_fakes.py
    python scripts/generate_audio_swap_fakes.py --real_dir ./real --out_dir ./custom_fake
    python scripts/generate_audio_swap_fakes.py --max_videos 20 --pairs_per_video 3
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

# Run from project root
ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXT = {".mp4", ".mov", ".mpg", ".mpeg", ".avi", ".mkv", ".webm"}


def _ffmpeg_path() -> str:
    """Prefer project tools/ or Homebrew ffmpeg."""
    try:
        local = ROOT / "tools" / "ffmpeg"
        if local.exists():
            return str(local)
    except Exception:
        pass
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return "ffmpeg"


def collect_videos(real_dir: Path):
    """Collect video paths from real_dir, sorted for stable ordering."""
    paths = []
    for p in sorted(real_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            paths.append(p)
    return paths


def audio_swap(
    video_a: Path,
    video_b: Path,
    out_path: Path,
    ffmpeg: str,
    overwrite: bool = False,
    timeout: int = 300,
) -> tuple[bool, str]:
    """
    Create one fake: video from A, audio from B.
    Uses -shortest so length = min(len_A, len_B).
    """
    if out_path.exists() and not overwrite:
        return True, "exists"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i", str(video_a),
        "-i", str(video_b),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(out_path),
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode == 0:
            return True, "ok"
        err = (r.stderr or "").strip().split("\n")[-3:]
        return False, " ".join(err)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "ffmpeg not found"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Audio Swap fake videos (video A + audio B) into custom_fake/"
    )
    parser.add_argument(
        "--real_dir",
        type=Path,
        default=ROOT / "real",
        help="Directory of real videos",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "custom_fake",
        help="Output directory for fake videos",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=10,
        help="Use at most this many source videos (default 10)",
    )
    parser.add_argument(
        "--pairs_per_video",
        type=int,
        default=2,
        help="For each video, create this many fakes with different audio sources (default 2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all videos and all pairs (can create very many files)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for choosing pairs")
    args = parser.parse_args()

    real_dir = args.real_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not real_dir.is_dir():
        print(f"Error: real_dir not found: {real_dir}", file=sys.stderr)
        sys.exit(1)

    videos = collect_videos(real_dir)
    # Skip files with space-only or awkward stems (e.g. " .mp4")
    videos = [v for v in videos if v.stem.strip() and v.stem.strip()[0].isalnum()]
    if len(videos) < 2:
        print("Need at least 2 videos in real_dir.", file=sys.stderr)
        sys.exit(1)

    ffmpeg = _ffmpeg_path()
    if ffmpeg == "ffmpeg" and not Path(ffmpeg).is_absolute():
        try:
            subprocess.run([ffmpeg, "-version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ffmpeg not found. Install with: brew install ffmpeg", file=sys.stderr)
            sys.exit(1)

    # Limit set and build pairs
    if args.all:
        use = videos
        pairs = []
        for i, va in enumerate(use):
            for j, vb in enumerate(use):
                if i != j:
                    pairs.append((va, vb))
    else:
        use = videos[: args.max_videos]
        rng = random.Random(args.seed)
        others = list(use)
        pairs = []
        for va in use:
            rest = [v for v in others if v != va]
            k = min(args.pairs_per_video, len(rest))
            if k == 0:
                continue
            chosen = rng.sample(rest, k)
            for vb in chosen:
                pairs.append((va, vb))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Real dir: {real_dir}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Videos in set: {len(use)}, pairs to generate: {len(pairs)}", flush=True)

    ok = 0
    fail = 0
    for va, vb in pairs:
        name_a = va.stem
        name_b = vb.stem
        out_name = f"video_{name_a}_audio_{name_b}.mp4"
        out_path = out_dir / out_name
        success, msg = audio_swap(
            va, vb, out_path, ffmpeg, overwrite=args.overwrite
        )
        if success:
            ok += 1
            if msg != "exists":
                print(f"  OK: {out_name}", flush=True)
            else:
                print(f"  skip (exists): {out_name}", flush=True)
        else:
            fail += 1
            print(f"  FAIL: {out_name} — {msg}", file=sys.stderr, flush=True)

    print(f"\nDone. Success: {ok}, Failed: {fail}", flush=True)


if __name__ == "__main__":
    main()
