#!/usr/bin/env python3
"""
Filter out corrupt videos by moving them to a separate folder.

This script tests videos the SAME WAY as the training pipeline:
1. Tests video frame reading with OpenCV
2. Tests audio extraction with librosa
3. Optionally tests face detection with MediaPipe
4. Moves corrupt/unreadable videos to corruptedclips/ folder
5. Preserves directory structure (0_real, 1_fake)
6. Generates a report of all corrupt files

Usage:
    python scripts/filter_corrupt_videos.py "data/AVLips1 2/" --dry-run
    python scripts/filter_corrupt_videos.py "data/AVLips1 2/" --test-audio --test-faces
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm

# Optional imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, 'solutions')
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False


def test_video_frames(video_path: Path, test_frames: int = 30, test_all: bool = False) -> Tuple[bool, str]:
    """
    Test if a video can be read by OpenCV.
    
    Args:
        video_path: Path to video file
        test_frames: Number of frames to test (if not test_all)
        test_all: If True, test ALL frames in the video
    
    Returns:
        (is_valid, error_message)
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # Get total frame count if testing all
        if test_all:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                test_all = False
                frames_to_test = test_frames
            else:
                frames_to_test = total_frames
        else:
            frames_to_test = test_frames
        
        # Try to read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, "Cannot read first frame"
        
        # Try to read more frames to ensure video is not just partially corrupt
        frames_read = 1
        consecutive_failures = 0
        
        for i in range(frames_to_test - 1):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_read += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                # If we get 3 consecutive failures, consider it corrupt
                if consecutive_failures >= 3:
                    cap.release()
                    return False, f"Failed after reading {frames_read} frames (3 consecutive failures at frame {i+1})"
        
        cap.release()
        
        # If we couldn't read at least half the test frames, consider it suspect
        if frames_read < frames_to_test // 2:
            return False, f"Only read {frames_read}/{frames_to_test} frames"
        
        return True, "OK"
    
    except Exception as e:
        return False, f"Video read exception: {str(e)}"


def test_audio_extraction(video_path: Path, sr: int = 16000) -> Tuple[bool, str]:
    """
    Test if audio can be extracted from the video using librosa.
    This matches the training pipeline's audio extraction.
    
    Returns:
        (is_valid, error_message)
    """
    if not LIBROSA_AVAILABLE:
        return True, "Skipped (librosa not available)"
    
    try:
        # This is exactly how the training pipeline loads audio
        audio, _ = librosa.load(str(video_path), sr=sr, mono=True)
        
        if audio is None or len(audio) == 0:
            return False, "Audio extraction returned empty array"
        
        # Check for NaN or Inf values
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return False, "Audio contains NaN or Inf values"
        
        # Check if audio is all zeros (silent/corrupted)
        if np.all(audio == 0):
            return False, "Audio is all zeros (no audio track or corrupted)"
        
        return True, "OK"
    
    except Exception as e:
        return False, f"Audio extraction failed: {str(e)}"


def test_face_detection(video_path: Path, min_confidence: float = 0.5) -> Tuple[bool, str]:
    """
    Test if at least one face can be detected in the video using MediaPipe.
    This matches the training pipeline's face detection.
    
    Returns:
        (is_valid, error_message)
    """
    if not MEDIAPIPE_AVAILABLE:
        return True, "Skipped (MediaPipe not available)"
    
    try:
        # Initialize MediaPipe Face Mesh (same as training)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            face_mesh.close()
            return False, "Cannot open video for face detection"
        
        # Test first 10 frames for face detection
        faces_detected = 0
        frames_tested = 0
        
        for i in range(10):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frames_tested += 1
            
            # Convert BGR to RGB (MediaPipe requirement)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                faces_detected += 1
        
        cap.release()
        face_mesh.close()
        
        if frames_tested == 0:
            return False, "Could not read any frames for face detection"
        
        if faces_detected == 0:
            return False, f"No faces detected in first {frames_tested} frames"
        
        # If less than 30% of frames have faces, flag as problematic
        face_ratio = faces_detected / frames_tested
        if face_ratio < 0.3:
            return False, f"Low face detection rate: {face_ratio:.1%} ({faces_detected}/{frames_tested} frames)"
        
        return True, "OK"
    
    except Exception as e:
        return False, f"Face detection failed: {str(e)}"


def test_video_comprehensive(
    video_path: Path,
    test_frames: int = 30,
    test_all_frames: bool = False,
    test_audio: bool = False,
    test_faces: bool = False,
) -> Tuple[bool, str]:
    """
    Comprehensive test that matches the training pipeline.
    
    Returns:
        (is_valid, error_message)
    """
    # Test 1: Video frame reading
    valid, error = test_video_frames(video_path, test_frames, test_all_frames)
    if not valid:
        return False, f"[VIDEO] {error}"
    
    # Test 2: Audio extraction (if requested)
    if test_audio:
        valid, error = test_audio_extraction(video_path)
        if not valid and error != "Skipped (librosa not available)":
            return False, f"[AUDIO] {error}"
    
    # Test 3: Face detection (if requested)
    if test_faces:
        valid, error = test_face_detection(video_path)
        if not valid and error != "Skipped (MediaPipe not available)":
            return False, f"[FACE] {error}"
    
    return True, "OK"


def move_corrupt_video(
    video_path: Path,
    data_dir: Path,
    corrupt_dir: Path,
    error_msg: str,
    dry_run: bool = False
) -> bool:
    """
    Move a corrupt video to the corrupt directory, preserving structure.
    
    Returns:
        True if successful (or would be successful in dry-run)
    """
    try:
        # Calculate relative path to preserve directory structure
        relative_path = video_path.relative_to(data_dir)
        
        # Create destination path
        dest_path = corrupt_dir / relative_path
        
        if dry_run:
            print(f"  [DRY RUN] Would move: {video_path} -> {dest_path}")
            print(f"            Reason: {error_msg}")
            return True
        
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        shutil.move(str(video_path), str(dest_path))
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error moving {video_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter corrupt videos and move them to a separate folder"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing videos to check"
    )
    parser.add_argument(
        "--corrupt-dir",
        type=Path,
        default=None,
        help="Directory to move corrupt videos to (default: data_dir/corruptedclips)"
    )
    parser.add_argument(
        "--test-frames",
        type=int,
        default=30,
        help="Number of frames to test per video (default: 30, use more to match training)"
    )
    parser.add_argument(
        "--test-all-frames",
        action="store_true",
        help="Test ALL frames in each video (slower but most thorough)"
    )
    parser.add_argument(
        "--test-audio",
        action="store_true",
        help="Test audio extraction with librosa (recommended - matches training)"
    )
    parser.add_argument(
        "--test-faces",
        action="store_true",
        help="Test face detection with MediaPipe (recommended - matches training)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Save report to file (default: data_dir/corrupt_videos_report.txt)"
    )
    args = parser.parse_args()
    
    # Validate input directory
    if not args.data_dir.exists():
        print(f"❌ Directory not found: {args.data_dir}")
        return
    
    # Set up corrupt directory
    corrupt_dir = args.corrupt_dir if args.corrupt_dir else args.data_dir / "corruptedclips"
    
    # Set up report file
    report_file = args.report if args.report else args.data_dir / "corrupt_videos_report.txt"
    
    print(f"{'='*80}")
    print(f"🔍 Corrupt Video Filter (Training Pipeline Match)")
    print(f"{'='*80}")
    print(f"📂 Data directory: {args.data_dir}")
    print(f"🗑️  Corrupt directory: {corrupt_dir}")
    print(f"📄 Report file: {report_file}")
    print(f"\n🧪 Test Configuration:")
    
    if args.test_all_frames:
        print(f"   🎬 Video: Testing ALL frames (thorough)")
    else:
        print(f"   🎬 Video: Testing first {args.test_frames} frames")
    
    if args.test_audio:
        if LIBROSA_AVAILABLE:
            print(f"   🔊 Audio: Testing extraction with librosa ✅")
        else:
            print(f"   🔊 Audio: SKIPPED (librosa not installed) ⚠️")
            print(f"      Install with: pip install librosa")
    else:
        print(f"   🔊 Audio: Not testing (use --test-audio)")
    
    if args.test_faces:
        if MEDIAPIPE_AVAILABLE:
            print(f"   👤 Faces: Testing detection with MediaPipe ✅")
        else:
            print(f"   👤 Faces: SKIPPED (MediaPipe not available) ⚠️")
            print(f"      Install with: pip install mediapipe-silicon")
    else:
        print(f"   👤 Faces: Not testing (use --test-faces)")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No files will be moved")
    
    print(f"{'='*80}\n")
    
    # Find all video files
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    video_files: List[Path] = []
    
    print("📹 Scanning for video files...")
    for ext in video_extensions:
        video_files.extend(args.data_dir.rglob(ext))
    
    # Exclude videos already in corrupt directory
    video_files = [v for v in video_files if "corruptedclips" not in str(v)]
    
    if not video_files:
        print(f"❌ No video files found in {args.data_dir}")
        return
    
    print(f"   Found {len(video_files)} videos to check\n")
    
    # Test all videos
    print("🧪 Testing videos...\n")
    
    corrupt_videos = []
    valid_videos = []
    
    for video_path in tqdm(video_files, desc="Testing videos", unit="video"):
        is_valid, error_msg = test_video_comprehensive(
            video_path,
            test_frames=args.test_frames,
            test_all_frames=args.test_all_frames,
            test_audio=args.test_audio,
            test_faces=args.test_faces,
        )
        
        if is_valid:
            valid_videos.append(video_path)
        else:
            corrupt_videos.append((video_path, error_msg))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 RESULTS")
    print(f"{'='*80}")
    print(f"✅ Valid videos: {len(valid_videos)}/{len(video_files)} ({len(valid_videos)/len(video_files)*100:.1f}%)")
    print(f"❌ Corrupt videos: {len(corrupt_videos)}/{len(video_files)} ({len(corrupt_videos)/len(video_files)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    if corrupt_videos:
        print(f"🗑️  Moving {len(corrupt_videos)} corrupt videos to {corrupt_dir}...\n")
        
        moved_count = 0
        failed_count = 0
        
        for video_path, error_msg in tqdm(corrupt_videos, desc="Moving files", unit="file"):
            success = move_corrupt_video(
                video_path,
                args.data_dir,
                corrupt_dir,
                error_msg,
                args.dry_run
            )
            
            if success:
                moved_count += 1
            else:
                failed_count += 1
        
        print(f"\n{'='*80}")
        print(f"📦 MOVE SUMMARY")
        print(f"{'='*80}")
        
        if args.dry_run:
            print(f"🔍 DRY RUN: {moved_count} files would be moved")
        else:
            print(f"✅ Moved: {moved_count}")
            print(f"❌ Failed: {failed_count}")
        
        print(f"{'='*80}\n")
        
        # Generate report
        if not args.dry_run:
            print(f"📄 Generating report: {report_file}")
            
            with open(report_file, "w") as f:
                f.write("Corrupt Videos Report\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Scan Date: {__import__('datetime').datetime.now()}\n")
                f.write(f"Data Directory: {args.data_dir}\n")
                f.write(f"Corrupt Directory: {corrupt_dir}\n")
                f.write(f"Total Videos Scanned: {len(video_files)}\n")
                f.write(f"Valid Videos: {len(valid_videos)}\n")
                f.write(f"Corrupt Videos: {len(corrupt_videos)}\n")
                f.write(f"Success Rate: {len(valid_videos)/len(video_files)*100:.2f}%\n")
                f.write("\n" + "=" * 80 + "\n\n")
                f.write("Corrupt Videos List:\n")
                f.write("-" * 80 + "\n\n")
                
                for video_path, error_msg in corrupt_videos:
                    relative = video_path.relative_to(args.data_dir)
                    f.write(f"File: {relative}\n")
                    f.write(f"Reason: {error_msg}\n")
                    f.write(f"Original Path: {video_path}\n")
                    f.write(f"Moved To: {corrupt_dir / relative}\n")
                    f.write("\n")
            
            print(f"✅ Report saved to: {report_file}\n")
        
        # Show sample of corrupt videos
        print(f"📋 Sample of corrupt videos (first 10):\n")
        for video_path, error_msg in corrupt_videos[:10]:
            print(f"  ❌ {video_path.name}")
            print(f"     Reason: {error_msg}")
        
        if len(corrupt_videos) > 10:
            print(f"\n  ... and {len(corrupt_videos) - 10} more (see report for full list)")
    
    else:
        print(f"🎉 All videos are valid! No corrupt files found.\n")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print(f"💡 NEXT STEPS")
    print(f"{'='*80}")
    
    if args.dry_run:
        print(f"✅ Review the files that would be moved above")
        print(f"✅ Run without --dry-run to actually move the files:")
        print(f'   python scripts/filter_corrupt_videos.py "{args.data_dir}"')
    else:
        if corrupt_videos:
            print(f"✅ Dataset cleaned! {len(valid_videos)} valid videos remain")
            print(f"✅ You can now train with the cleaned dataset:")
            print(f'   python -m app.training.train --data-dir "{args.data_dir}" --epochs 50 --batch-size 8')
            print(f"\n💡 If you want to review or fix corrupt videos:")
            print(f'   Check: {corrupt_dir}')
            print(f'   Report: {report_file}')
        else:
            print(f"✅ Your dataset is clean! Start training:")
            print(f'   python -m app.training.train --data-dir "{args.data_dir}" --epochs 50 --batch-size 8')
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
