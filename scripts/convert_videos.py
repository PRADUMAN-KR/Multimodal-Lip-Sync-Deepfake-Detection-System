#!/usr/bin/env python3
"""
Convert problematic videos to OpenCV-friendly format.

Usage:
    python scripts/convert_videos.py data/AVLips1\ 2/0_real/ --output data/converted/
"""

import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple


def convert_video(input_path: Path, output_path: Path, overwrite: bool = False) -> Tuple[Path, bool, str]:
    """
    Convert a video to OpenCV-friendly format:
    - H.264 codec
    - yuv420p pixel format
    - Constant frame rate (30 fps)
    - AAC audio
    """
    if output_path.exists() and not overwrite:
        return input_path, False, "Already exists (use --overwrite)"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-c:v", "libx264",           # H.264 codec
            "-preset", "fast",            # Fast encoding
            "-crf", "23",                 # Quality (lower = better, 18-28 is good range)
            "-pix_fmt", "yuv420p",        # Standard pixel format
            "-vsync", "cfr",              # Constant frame rate
            "-r", "30",                   # 30 fps
            "-c:a", "aac",                # AAC audio
            "-b:a", "128k",               # Audio bitrate
            "-movflags", "+faststart",    # Enable streaming
            "-y" if overwrite else "-n",  # Overwrite or skip if exists
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per video
        )
        
        if result.returncode == 0:
            return input_path, True, "Success"
        else:
            error_msg = result.stderr.split('\n')[-5:]  # Last 5 lines
            return input_path, False, f"FFmpeg error: {' '.join(error_msg)}"
    
    except subprocess.TimeoutExpired:
        return input_path, False, "Timeout (>60s)"
    except FileNotFoundError:
        return input_path, False, "FFmpeg not found (install with: brew install ffmpeg)"
    except Exception as e:
        return input_path, False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Convert videos to OpenCV-friendly format")
    parser.add_argument("input_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--output", type=Path, help="Output directory (default: input_dir/converted)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel conversions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without converting")
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    output_dir = args.output if args.output else args.input_dir / "converted"
    
    # Find all video files
    video_files = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
        video_files.extend(args.input_dir.rglob(ext))
    
    if not video_files:
        print(f"❌ No video files found in {args.input_dir}")
        return
    
    print(f"📹 Found {len(video_files)} videos")
    print(f"📂 Output directory: {output_dir}")
    print(f"🔧 Using {args.workers} parallel workers")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - no files will be converted\n")
        for video in video_files[:10]:
            relative = video.relative_to(args.input_dir)
            output_path = output_dir / relative
            print(f"  {video.name} -> {output_path}")
        if len(video_files) > 10:
            print(f"  ... and {len(video_files) - 10} more")
        return
    
    print(f"\n🚀 Starting conversion...\n")
    
    # Convert videos in parallel
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for video in video_files:
            # Preserve directory structure
            relative = video.relative_to(args.input_dir)
            output_path = output_dir / relative.parent / f"{relative.stem}_converted.mp4"
            
            future = executor.submit(convert_video, video, output_path, args.overwrite)
            futures.append(future)
        
        # Process results as they complete
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, future in enumerate(as_completed(futures), 1):
            input_path, success, message = future.result()
            
            if success:
                print(f"✅ [{i}/{len(video_files)}] {input_path.name}")
                success_count += 1
            elif "Already exists" in message:
                print(f"⏭️  [{i}/{len(video_files)}] {input_path.name} - {message}")
                skipped_count += 1
            else:
                print(f"❌ [{i}/{len(video_files)}] {input_path.name} - {message}")
                failed_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Success: {success_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📂 Output: {output_dir}")
    
    if success_count > 0:
        print(f"\n💡 To use converted videos for training:")
        print(f"   Update your data path to: {output_dir}")


if __name__ == "__main__":
    main()
