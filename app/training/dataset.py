import io
import json
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


def discover_video_samples(data_dir: Path) -> list[Tuple[Path, int]]:
    """
    Discover raw video samples and infer labels.

    Label meaning:
      - 1 = REAL
      - 0 = FAKE
    """
    samples: list[Tuple[Path, int]] = []

    def _find_subdir(*names: str):
        lowered = {n.lower() for n in names}
        for n in names:
            p = data_dir / n
            if p.is_dir():
                return p
        for c in data_dir.iterdir():
            if c.is_dir() and c.name.lower() in lowered:
                return c
        return None

    real_dir = _find_subdir("0_real", "real")
    fake_dir = _find_subdir("1_fake", "fake")
    if real_dir is not None and fake_dir is not None:
        for vid_path in real_dir.iterdir():
            if vid_path.is_file() and _is_video(vid_path):
                samples.append((vid_path, 1))
        for vid_path in fake_dir.iterdir():
            if vid_path.is_file() and _is_video(vid_path):
                samples.append((vid_path, 0))
        return samples

    for vid_path in data_dir.rglob("*"):
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
            samples.append((vid_path, label))
    return samples


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
        preprocessed_dir: Path | str | None = None,
        storage_format: str = "npy",
    ) -> None:
        super().__init__()
        self.split = split
        self.data_dir = Path(data_dir)
        self.video_frames = int(video_frames)
        self.audio_frames = int(audio_frames)
        self.require_face_detection = bool(require_face_detection)
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.storage_format = storage_format.lower().strip()
        self._lmdb_env = None
        self._zarr_root = None
        self._manifest: list[dict] = []
        
        # Track failures for diagnostics
        self._failure_count = 0
        self._first_error = None

        self.use_preprocessed = self.preprocessed_dir is not None
        if self.use_preprocessed:
            self._load_preprocessed_manifest()
        else:
            self.samples: list[Tuple[Path, int]] = discover_video_samples(self.data_dir)

        if (self.use_preprocessed and not self._manifest) or (
            not self.use_preprocessed and not self.samples
        ):
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

        # Test MediaPipe only for on-the-fly preprocessing mode.
        if self.require_face_detection and not self.use_preprocessed:
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
            if self.use_preprocessed:
                sorted_samples = sorted(
                    self._manifest, key=lambda x: str(x.get("source_path", ""))
                )
            else:
                sorted_samples = sorted(self.samples, key=lambda x: str(x[0]))
            if split == "train":
                sliced = sorted_samples[: int(len(sorted_samples) * 0.8)]
            elif split == "val":
                sliced = sorted_samples[
                    int(len(sorted_samples) * 0.8) : int(len(sorted_samples) * 0.9)
                ]
            elif split == "test":
                sliced = sorted_samples[int(len(sorted_samples) * 0.9) :]
            else:
                sliced = sorted_samples

            if self.use_preprocessed:
                self._manifest = sliced
                self.samples = [
                    (
                        Path(rec.get("source_path", rec.get("key", f"sample_{i}"))),
                        int(rec["label"]),
                    )
                    for i, rec in enumerate(self._manifest)
                ]
            else:
                self.samples = sliced

    def __len__(self) -> int:
        return len(self._manifest) if self.use_preprocessed else len(self.samples)

    def _load_preprocessed_manifest(self) -> None:
        if self.preprocessed_dir is None:
            raise ValueError("preprocessed_dir is required in preprocessed mode")
        manifest_path = self.preprocessed_dir / "manifest.jsonl"
        if not manifest_path.is_file():
            raise ValueError(
                f"Missing preprocessed manifest: {manifest_path}. "
                "Run precompute script first."
            )
        self._manifest = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                self._manifest.append(json.loads(s))
        if not self._manifest:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        # Keep `samples` shape-compatible with existing training utilities.
        self.samples = [
            (Path(rec.get("source_path", rec.get("key", f"sample_{i}"))), int(rec["label"]))
            for i, rec in enumerate(self._manifest)
        ]

        if self.storage_format not in {"npy", "lmdb", "zarr"}:
            raise ValueError(
                f"Unsupported storage_format='{self.storage_format}'. "
                "Expected one of: npy, lmdb, zarr."
            )
        if self.storage_format == "lmdb":
            self._init_lmdb()
        elif self.storage_format == "zarr":
            self._init_zarr()

    def _init_lmdb(self) -> None:
        if self.preprocessed_dir is None:
            raise ValueError("preprocessed_dir is required for lmdb mode")
        if self._lmdb_env is not None:
            return
        try:
            import lmdb  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "LMDB storage selected but `lmdb` is not installed. "
                "Install with: pip install lmdb"
            ) from e
        lmdb_path = self.preprocessed_dir / "samples.lmdb"
        if not lmdb_path.exists():
            raise ValueError(f"LMDB path not found: {lmdb_path}")
        self._lmdb_env = lmdb.open(
            str(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            subdir=False,
        )

    def _init_zarr(self) -> None:
        if self.preprocessed_dir is None:
            raise ValueError("preprocessed_dir is required for zarr mode")
        if self._zarr_root is not None:
            return
        try:
            import zarr  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Zarr storage selected but `zarr` is not installed. "
                "Install with: pip install zarr"
            ) from e
        zarr_path = self.preprocessed_dir / "samples.zarr"
        if not zarr_path.exists():
            raise ValueError(f"Zarr path not found: {zarr_path}")
        self._zarr_root = zarr.open_group(str(zarr_path), mode="r")

    def _load_preprocessed_sample(
        self, idx: int, train_mode_override: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.preprocessed_dir is None:
            raise ValueError("preprocessed_dir is required in preprocessed mode")
        rec = self._manifest[idx]
        label = int(rec["label"])
        mode = str(rec.get("precompute_mode", "fixed_clip"))

        if self.storage_format == "npy":
            visual_path = self.preprocessed_dir / rec["visual_relpath"]
            audio_path = self.preprocessed_dir / rec["audio_relpath"]
            visual = np.load(visual_path).astype(np.float32, copy=False)
            audio = np.load(audio_path).astype(np.float32, copy=False)
        elif self.storage_format == "lmdb":
            self._init_lmdb()
            key = str(rec["key"]).encode("utf-8")
            with self._lmdb_env.begin(write=False) as txn:
                blob = txn.get(key)
            if blob is None:
                raise KeyError(f"LMDB key not found: {rec['key']}")
            with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                visual = data["visual"].astype(np.float32, copy=False)
                audio = data["audio"].astype(np.float32, copy=False)
        else:
            self._init_zarr()
            key = str(rec["key"])
            if key not in self._zarr_root:
                raise KeyError(f"Zarr key not found: {key}")
            grp = self._zarr_root[key]
            visual = grp["visual"][:].astype(np.float32, copy=False)
            audio = grp["audio"][:].astype(np.float32, copy=False)

        if mode == "full_sequence":
            target_fps = float(rec.get("target_fps", 15.0))
            mel_hz = float(rec.get("mel_hz", 100.0))
            # Default behavior:
            # - split='train' -> random contiguous window
            # - split='val'/'test' -> center window
            # - split=None -> random (used by random_split + explicit override wrappers)
            train_mode_default = True if self.split is None else (self.split == "train")
            train_mode = (
                train_mode_default
                if train_mode_override is None
                else bool(train_mode_override)
            )
            visual, audio = self._sample_aligned_contiguous_clip(
                visual_seq=visual,
                audio_seq=audio,
                video_frames=self.video_frames,
                audio_frames=self.audio_frames,
                target_fps=target_fps,
                mel_hz=mel_hz,
                train_mode=train_mode,
            )

        visual_tensor = torch.from_numpy(visual).float()
        audio_tensor = torch.from_numpy(audio).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return visual_tensor, audio_tensor, label_tensor

    def _sample_aligned_contiguous_clip(
        self,
        visual_seq: np.ndarray,
        audio_seq: np.ndarray,
        video_frames: int,
        audio_frames: int,
        target_fps: float,
        mel_hz: float,
        train_mode: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample contiguous visual window and aligned audio window.

        Input full-sequence shapes:
          - visual_seq: (T, H, W, C)
          - audio_seq:  (80, T_audio) or (1, 80, T_audio)

        Returns model-ready shapes:
          - visual: (3, video_frames, 96, 96), float32 in [0, 1]
          - audio:  (1, 80, audio_frames), float32
        """
        if visual_seq.ndim != 4:
            raise ValueError(
                f"Expected visual_seq ndim=4 (T,H,W,C), got {visual_seq.shape}"
            )
        if audio_seq.ndim == 3:
            if audio_seq.shape[0] != 1:
                raise ValueError(f"Expected audio_seq shape (1,80,T), got {audio_seq.shape}")
            audio_seq_2d = audio_seq[0]
        elif audio_seq.ndim == 2:
            audio_seq_2d = audio_seq
        else:
            raise ValueError(
                f"Expected audio_seq ndim=2 or 3, got {audio_seq.shape}"
            )
        if audio_seq_2d.shape[0] != 80:
            raise ValueError(
                f"Expected mel bins=80 in full audio, got {audio_seq_2d.shape}"
            )

        t_total = int(visual_seq.shape[0])
        if t_total <= 0:
            raise ValueError("Empty visual sequence")

        # Reject clips that are too short to yield a useful training sample.
        # A clip padded beyond 50% of its window has mostly frozen-mouth frames
        # that introduce a systematic audio-visual mismatch in the padded region.
        min_frames = max(4, video_frames // 2)
        if t_total < min_frames:
            raise ValueError(
                f"Visual sequence too short ({t_total} frames < minimum {min_frames}). "
                "Sample skipped."
            )

        if t_total > video_frames:
            if train_mode:
                start = int(np.random.randint(0, t_total - video_frames + 1))
            else:
                start = int((t_total - video_frames) // 2)
            visual_win = visual_seq[start : start + video_frames]
        else:
            # t_total is in [min_frames, video_frames]: pad tail with last frame.
            # start stays 0 so mel_start = 0 (aligned to beginning of clip).
            start = 0
            pad_n = video_frames - t_total
            if pad_n > 0:
                pad = np.repeat(visual_seq[-1:], pad_n, axis=0)
                visual_win = np.concatenate([visual_seq, pad], axis=0)
            else:
                visual_win = visual_seq

        visual_win = visual_win.astype(np.float32) / 255.0
        visual = np.transpose(visual_win, (3, 0, 1, 2))  # (C, T, H, W)

        # ── Audio–video time alignment ────────────────────────────────────────
        # mel_hz = sr / hop_length = 16000 / 160 = 100 mel-frames / second
        # time_start_sec = start / fps
        # mel_start      = time_start_sec * mel_hz  →  frame-index to mel-index
        # mel_len        = (video_frames / fps) * mel_hz  →  exact window length
        # This is the standard formula; rounding error < 1 mel frame at 100 fps.
        a_total = int(audio_seq_2d.shape[1])
        mel_start = int(round((start / max(target_fps, 1e-6)) * mel_hz))
        mel_len = max(1, int(round((video_frames / max(target_fps, 1e-6)) * mel_hz)))

        mel_start = max(0, min(mel_start, max(0, a_total - 1)))
        mel_end = min(a_total, mel_start + mel_len)
        mel_win = audio_seq_2d[:, mel_start:mel_end]
        if mel_win.shape[1] == 0:
            mel_win = np.repeat(audio_seq_2d[:, -1:], 1, axis=1)

        if mel_win.shape[1] < mel_len:
            pad = np.repeat(mel_win[:, -1:], mel_len - mel_win.shape[1], axis=1)
            mel_win = np.concatenate([mel_win, pad], axis=1)

        # Resample aligned mel window to model's fixed audio_frames.
        if mel_win.shape[1] != audio_frames:
            idx = np.linspace(0, mel_win.shape[1] - 1, audio_frames).astype(np.int64)
            mel_win = mel_win[:, idx]

        audio = np.expand_dims(mel_win.astype(np.float32, copy=False), axis=0)  # (1, 80, T)
        return visual, audio

    def get_item(
        self, idx: int, train_mode_override: Optional[bool] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns:
            visual: (3, T, H, W) - preprocessed video frames
            audio:  (1, F, T) - mel-spectrogram
            label:  (1,) - binary label (1=REAL/authentic, 0=FAKE/AI-manipulated)
        """
        if self.use_preprocessed:
            try:
                return self._load_preprocessed_sample(
                    idx, train_mode_override=train_mode_override
                )
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                self._failure_count += 1
                if self._first_error is None:
                    self._first_error = (error_type, error_msg, f"sample_idx={idx}")
                if self._failure_count <= 3:
                    logger.warning(
                        "Skipping preprocessed sample %d: %s: %s",
                        idx,
                        error_type,
                        error_msg,
                    )
                return None

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

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.get_item(idx, train_mode_override=None)
