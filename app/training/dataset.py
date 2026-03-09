from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.logger import get_logger
from ..preprocessing.audio import preprocess_audio
from ..preprocessing.video import preprocess_video

logger = get_logger(__name__)

# Video extensions supported (case-insensitive): .mp4, .mov, .avi
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


class LipSyncDataset(Dataset):
    """
    Dataset for lip-sync manipulation detection training.

    Expected directory structure:
        data/
            AVLips12/
                0_real/ or real/    # REAL videos (label=1)
                    *.mp4, *.mov, *.avi (case-insensitive)
                1_fake/ or fake/   # FAKE videos (label=0)
                    *.mp4, *.mov, *.avi (case-insensitive)

    Label meaning:
        - 1 = REAL: Natural, unmodified video with authentic lip-sync
        - 0 = FAKE: AI-manipulated video (modified with Wav2Lip, DeepFaceLab, etc.)

    Or provide a list of (video_path, label) pairs.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str | None = None,
        video_frames: int = 32,
        audio_frames: int = 128,
        require_face_detection: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.video_frames = int(video_frames)
        self.audio_frames = int(audio_frames)
        self.require_face_detection = bool(require_face_detection)
        
        # Track failures for diagnostics
        self._failure_count = 0
        self._first_error = None

        # Build list of (video_path, label) pairs
        self.samples: list[Tuple[Path, int]] = []

        # Prefer named subdirs: 0_real/1_fake (AVLips) or real/fake (case-insensitive)
        def _find_subdir(*names: str):
            for n in names:
                p = self.data_dir / n
                if p.is_dir():
                    return p
            # Case-insensitive fallback (e.g. Real, Fake on Linux)
            for c in self.data_dir.iterdir():
                if c.is_dir() and c.name.lower() in {n.lower() for n in names}:
                    return c
            return None
        real_dir = _find_subdir("0_real", "real")
        fake_dir = _find_subdir("1_fake", "fake")
        if real_dir is not None and fake_dir is not None:
            for vid_path in real_dir.iterdir():
                if vid_path.is_file() and _is_video(vid_path):
                    self.samples.append((vid_path, 1))  # REAL = 1
            for vid_path in fake_dir.iterdir():
                if vid_path.is_file() and _is_video(vid_path):
                    self.samples.append((vid_path, 0))  # FAKE = 0
        else:
            # Assume flat or custom structure: find any video under data_dir
            for vid_path in self.data_dir.rglob("*"):
                if vid_path.is_file() and _is_video(vid_path):
                    parent = vid_path.parent.name.lower()
                    if "real" in parent or "authentic" in parent or "natural" in parent:
                        label = 1
                    elif (
                        "fake" in parent
                        or "manipulated" in parent
                        or "ai" in parent
                        or "wav2lip" in parent
                        or "deepfake" in parent
                    ):
                        label = 0
                    else:
                        label = 1
                    self.samples.append((vid_path, label))

        if not self.samples:
            data_dir_resolved = self.data_dir.resolve()
            msg = (
                f"No video files found in {data_dir}\n"
                f"  Resolved path: {data_dir_resolved}\n"
                f"  Directory exists: {self.data_dir.is_dir()}\n"
            )
            if self.data_dir.is_dir():
                subdirs = [p.name for p in self.data_dir.iterdir() if p.is_dir()]
                files = [p.name for p in self.data_dir.iterdir() if p.is_file()]
                msg += f"  Subdirectories: {subdirs or '(none)'}\n"
                msg += f"  Files in root: {len(files)} (extensions: {list(set(p.suffix for p in self.data_dir.iterdir() if p.is_file()))})\n"
                for name in ("0_real", "1_fake", "real", "fake"):
                    d = self.data_dir / name
                    if d.is_dir():
                        n = sum(1 for p in d.iterdir() if p.is_file() and _is_video(p))
                        msg += f"  {name}/: {n} video(s)\n"
            else:
                msg += f"  Current working directory: {Path.cwd()}\n"
                msg += "  Tip: Use an absolute path for --data-dir, or run from the repo root so 'data/AVLips12' exists.\n"
            raise ValueError(msg)

        # Test MediaPipe if face detection is required
        if self.require_face_detection:
            try:
                from ..preprocessing.face_detection import FaceDetector
                # Try to create a detector to catch MediaPipe issues early
                test_detector = FaceDetector(max_num_faces=1)
                logger.info("✅ Face detection (MediaPipe) is working")
            except Exception as e:
                error_msg = str(e)
                logger.error("=" * 80)
                logger.error("❌ CRITICAL: Face detection (MediaPipe) is NOT working!")
                logger.error(f"   Error: {error_msg}")
                logger.error("")
                logger.error("This will cause ALL samples to fail preprocessing.")
                logger.error("")
                logger.error("Fix MediaPipe BEFORE training:")
                logger.error("  1. pip uninstall mediapipe -y")
                logger.error("  2. pip install mediapipe-silicon  # For Apple Silicon")
                logger.error("  3. Or: pip install --upgrade --force-reinstall mediapipe")
                logger.error("")
                logger.error("Test: python -c \"import mediapipe as mp; print(hasattr(mp, 'solutions'))\"")
                logger.error("=" * 80)
                raise RuntimeError(
                    "MediaPipe face detection is required but not working. "
                    "Fix MediaPipe installation before training. See error above."
                ) from e

        # Optionally filter by split if provided (for train/val/test)
        if split:
            # Simple split: use modulo on sorted paths
            sorted_samples = sorted(self.samples, key=lambda x: str(x[0]))
            if split == "train":
                self.samples = sorted_samples[: int(len(sorted_samples) * 0.8)]
            elif split == "val":
                self.samples = sorted_samples[
                    int(len(sorted_samples) * 0.8) : int(len(sorted_samples) * 0.9)
                ]
            elif split == "test":
                self.samples = sorted_samples[int(len(sorted_samples) * 0.9) :]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns:
            visual: (3, T, H, W) - preprocessed video frames
            audio:  (1, F, T) - mel-spectrogram
            label:  (1,) - binary label (1=REAL/authentic, 0=FAKE/AI-manipulated)
        """
        video_path, label = self.samples[idx]

        try:
            # Preprocess video with face detection (production-grade).
            # This enforces fixed `T = self.video_frames` for batching.
            visual = preprocess_video(
                video_path,
                use_face_detection=True,
                max_faces=1,
                crop_size=(96, 96),
                max_frames=self.video_frames,
                strict_face_detection=self.require_face_detection,
            )  # (C, T, H, W)

            # Preprocess audio mel and enforce fixed mel time for batching.
            # NOTE: mel time steps are NOT the same as video frames; we keep a
            # stable fixed mel length and let fusion interpolate as needed.
            audio = preprocess_audio(video_path, target_frames=self.audio_frames)  # (1, F, T_mel)

        except Exception as e:
            # Production behavior: reject bad samples rather than feeding garbage.
            if self.require_face_detection:
                error_msg = str(e)
                error_type = type(e).__name__
                self._failure_count += 1
                
                # Store first error for summary
                if self._first_error is None:
                    self._first_error = (error_type, error_msg, video_path.name)
                
                # Log first few errors in detail, then summarize
                if self._failure_count <= 3:
                    if "mediapipe" in error_msg.lower() or "solutions" in error_msg.lower() or "FaceDetector" in error_type:
                        logger.error(
                            f"❌ Face detection failed for {video_path.name}:\n"
                            f"   Error: {error_type}: {error_msg}\n"
                            f"   This means MediaPipe is not working properly.\n"
                            f"   Fix: Install mediapipe-silicon or reinstall mediapipe in Python 3.11 venv."
                        )
                    else:
                        logger.warning(
                            f"⚠️  Skipping sample: {video_path.name} :: {error_type}: {error_msg}"
                        )
                elif self._failure_count == 4:
                    logger.error(
                        f"⚠️  Multiple samples failing. First error was: {self._first_error[0]}: {self._first_error[1]}"
                    )
                    logger.error(
                        f"   This suggests a systematic issue. Check MediaPipe installation or video files."
                    )
                return None
            raise

        visual_tensor = torch.from_numpy(visual).float()
        audio_tensor = torch.from_numpy(audio).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return visual_tensor, audio_tensor, label_tensor
