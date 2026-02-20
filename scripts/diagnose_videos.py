#!/usr/bin/env python3
"""
Diagnostic script to analyze videos and identify why OpenCV might fail.

Usage:
    python scripts/diagnose_videos.py data/AVLips1\ 2/
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2


def get_ffprobe_info(video_path: Path) -> Optional[Dict]:
    """Get detailed video information using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return None


def diagnose_video(video_path: Path) -> Dict:
    """Diagnose a single video file."""
    issues = []
    info = {
        "path": str(video_path),
        "opencv_can_open": False,
        "opencv_can_read": False,
        "opencv_frame_count": 0,
        "issues": [],
        "codec": "unknown",
        "pixel_format": "unknown",
        "fps": "unknown",
        "resolution": "unknown",
    }
    
    # Test OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path))
        info["opencv_can_open"] = cap.isOpened()
        
        if cap.isOpened():
            # Try to read first frame
            ret, frame = cap.read()
            info["opencv_can_read"] = ret
            
            # Try to get frame count
            frame_count = 0
            while frame_count < 10:  # Test first 10 frames
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
            info["opencv_frame_count"] = frame_count
            
            if frame_count == 0:
                issues.append("OpenCV can't read any frames")
            elif frame_count < 10:
                issues.append(f"OpenCV only read {frame_count} frames (may fail later)")
        else:
            issues.append("OpenCV can't open video file")
        
        cap.release()
    except Exception as e:
        issues.append(f"OpenCV error: {str(e)}")
    
    # Get detailed info with ffprobe
    ffprobe_data = get_ffprobe_info(video_path)
    if ffprobe_data:
        # Find video stream
        video_stream = None
        audio_streams = []
        
        for stream in ffprobe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
            elif stream.get("codec_type") == "audio":
                audio_streams.append(stream)
        
        if video_stream:
            codec = video_stream.get("codec_name", "unknown")
            info["codec"] = codec
            info["pixel_format"] = video_stream.get("pix_fmt", "unknown")
            info["resolution"] = f"{video_stream.get('width', '?')}x{video_stream.get('height', '?')}"
            
            # Check FPS
            fps_str = video_stream.get("r_frame_rate", "0/0")
            try:
                num, den = map(int, fps_str.split("/"))
                fps = num / den if den != 0 else 0
                info["fps"] = f"{fps:.2f}"
                
                # Check for VFR
                avg_fps_str = video_stream.get("avg_frame_rate", fps_str)
                avg_num, avg_den = map(int, avg_fps_str.split("/"))
                avg_fps = avg_num / avg_den if avg_den != 0 else 0
                
                if abs(fps - avg_fps) > 1.0:
                    issues.append(f"Variable frame rate detected (r_frame_rate={fps:.2f}, avg={avg_fps:.2f})")
            except (ValueError, ZeroDivisionError):
                issues.append(f"Invalid frame rate: {fps_str}")
            
            # Check codec
            if codec == "hevc" or codec == "h265":
                issues.append("H.265/HEVC codec - may not be supported by OpenCV")
            elif codec == "vp9":
                issues.append("VP9 codec - may not be supported by OpenCV")
            elif codec == "av1":
                issues.append("AV1 codec - may not be supported by OpenCV")
            elif codec not in ["h264", "mpeg4"]:
                issues.append(f"Unusual codec: {codec}")
            
            # Check pixel format
            pix_fmt = video_stream.get("pix_fmt", "")
            if pix_fmt not in ["yuv420p", "yuvj420p"]:
                issues.append(f"Unusual pixel format: {pix_fmt} (OpenCV prefers yuv420p)")
            
            # Check bit depth
            if "10" in pix_fmt or "12" in pix_fmt:
                issues.append(f"High bit depth detected: {pix_fmt} (OpenCV prefers 8-bit)")
            
            # Check profile
            profile = video_stream.get("profile", "")
            if "High 10" in profile or "High 4:2:2" in profile:
                issues.append(f"High profile detected: {profile}")
        
        # Check audio streams
        if len(audio_streams) > 1:
            issues.append(f"Multiple audio streams ({len(audio_streams)}) - may confuse OpenCV")
        
        for audio in audio_streams:
            audio_codec = audio.get("codec_name", "")
            if audio_codec not in ["aac", "mp3"]:
                issues.append(f"Unusual audio codec: {audio_codec}")
    else:
        issues.append("ffprobe failed - file may be severely corrupt")
    
    info["issues"] = issues
    return info


def main():
    parser = argparse.ArgumentParser(description="Diagnose video compatibility issues")
    parser.add_argument("data_dir", type=Path, help="Directory containing videos")
    parser.add_argument("--limit", type=int, default=50, help="Max number of videos to check")
    parser.add_argument("--show-good", action="store_true", help="Show videos without issues too")
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ Directory not found: {args.data_dir}")
        return
    
    # Find all video files
    video_files = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
        video_files.extend(args.data_dir.rglob(ext))
    
    if not video_files:
        print(f"❌ No video files found in {args.data_dir}")
        return
    
    print(f"📹 Found {len(video_files)} videos")
    print(f"🔍 Analyzing (limit: {args.limit})...\n")
    
    # Analyze videos
    problem_videos = []
    good_videos = []
    
    for i, video_path in enumerate(video_files[:args.limit]):
        if i > 0 and i % 10 == 0:
            print(f"   ... processed {i}/{min(len(video_files), args.limit)}")
        
        result = diagnose_video(video_path)
        
        if result["issues"]:
            problem_videos.append(result)
        else:
            good_videos.append(result)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 RESULTS")
    print(f"{'='*80}\n")
    print(f"✅ Good videos: {len(good_videos)}/{len(good_videos) + len(problem_videos)}")
    print(f"⚠️  Problem videos: {len(problem_videos)}/{len(good_videos) + len(problem_videos)}\n")
    
    if problem_videos:
        print(f"{'='*80}")
        print(f"⚠️  PROBLEM VIDEOS")
        print(f"{'='*80}\n")
        
        # Group by issue type
        issue_counts = {}
        for video in problem_videos:
            for issue in video["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        print("Issue frequency:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  • {issue}: {count} videos")
        
        print(f"\nDetailed information for problem videos:\n")
        for video in problem_videos[:20]:  # Show first 20
            print(f"📄 {Path(video['path']).name}")
            print(f"   Codec: {video['codec']}, Pixel: {video['pixel_format']}, FPS: {video['fps']}, Resolution: {video['resolution']}")
            print(f"   OpenCV: open={video['opencv_can_open']}, read={video['opencv_can_read']}, frames={video['opencv_frame_count']}")
            print(f"   Issues:")
            for issue in video["issues"]:
                print(f"      • {issue}")
            print()
        
        if len(problem_videos) > 20:
            print(f"   ... and {len(problem_videos) - 20} more problem videos\n")
    
    if args.show_good and good_videos:
        print(f"{'='*80}")
        print(f"✅ GOOD VIDEOS (sample)")
        print(f"{'='*80}\n")
        for video in good_videos[:5]:
            print(f"📄 {Path(video['path']).name}")
            print(f"   Codec: {video['codec']}, Pixel: {video['pixel_format']}, FPS: {video['fps']}, Resolution: {video['resolution']}")
            print()
    
    # Recommendations
    print(f"{'='*80}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if not problem_videos:
        print("✅ All videos look compatible with OpenCV!")
    else:
        print("Based on the issues found, here are some solutions:\n")
        
        if any("H.265" in str(v["issues"]) or "HEVC" in str(v["issues"]) for v in problem_videos):
            print("• H.265/HEVC codec detected:")
            print("  - Convert to H.264: ffmpeg -i input.mp4 -c:v libx264 -c:a copy output.mp4")
            print()
        
        if any("Variable frame rate" in str(v["issues"]) for v in problem_videos):
            print("• Variable frame rate detected:")
            print("  - Convert to CFR: ffmpeg -i input.mp4 -vsync cfr -r 30 output.mp4")
            print()
        
        if any("pixel format" in str(v["issues"]) for v in problem_videos):
            print("• Unusual pixel format detected:")
            print("  - Convert to yuv420p: ffmpeg -i input.mp4 -pix_fmt yuv420p output.mp4")
            print()
        
        if any("OpenCV can't" in str(v["issues"]) for v in problem_videos):
            print("• OpenCV compatibility issues:")
            print("  - Re-encode entire video: ffmpeg -i input.mp4 -c:v libx264 -preset fast -pix_fmt yuv420p -c:a aac output.mp4")
            print()


if __name__ == "__main__":
    main()
