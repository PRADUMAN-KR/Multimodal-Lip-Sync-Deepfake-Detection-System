"""
Data augmentation for robust lip-sync manipulation detection.

Supports:
- Temporal augmentation (speed variation)
- Spatial augmentation (rotation, flip, color jitter)
- Multi-angle robustness
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocessing.audio import preprocess_audio
from ..preprocessing.video import preprocess_video
from .dataset import LipSyncDataset


class AugmentedLipSyncDataset(Dataset):
    """
    Wrapper around LipSyncDataset that applies data augmentation.

    Augmentations:
    - Temporal: Speed variation (0.8x - 1.2x)
    - Spatial: Rotation (±15°), horizontal flip (50%), color jitter
    - Noise: Gaussian noise
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str | None = None,
        video_frames: int = 32,
        audio_frames: int = 128,
        require_face_detection: bool = True,
        apply_augmentation: bool = True,
    ) -> None:
        self.base_dataset = LipSyncDataset(
            data_dir,
            split=split,
            video_frames=video_frames,
            audio_frames=audio_frames,
            require_face_detection=require_face_detection,
        )
        self.apply_augmentation = apply_augmentation

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _temporal_augment(
        self, visual: np.ndarray, audio: np.ndarray, speed_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Temporal augmentation: speed variation **without changing tensor shapes**.

        Important: 3D CNN training requires fixed T across the batch, so this
        function always returns the same `(C, T, H, W)` and `(1, F, T_a)` lengths
        as its inputs.
        """
        if speed_factor == 1.0:
            return visual, audio

        T = visual.shape[1]
        T_audio = audio.shape[2]

        # Build new indices but keep output length fixed.
        # speed_factor > 1.0 => "faster" => sample further ahead in time.
        base_idx = np.linspace(0, T - 1, T)
        idx = np.clip(base_idx * speed_factor, 0, T - 1).astype(np.int64)
        visual_aug = visual[:, idx, :, :]

        base_idx_a = np.linspace(0, T_audio - 1, T_audio)
        idx_a = np.clip(base_idx_a * speed_factor, 0, T_audio - 1).astype(np.int64)
        audio_aug = audio[:, :, idx_a]

        return visual_aug, audio_aug

    def _spatial_augment(self, visual: np.ndarray) -> np.ndarray:
        """Spatial augmentation: rotation, flip, color jitter."""
        C, T, H, W = visual.shape

        # Random horizontal flip (50%)
        if np.random.rand() > 0.5:
            visual = np.flip(visual, axis=-1)

        # Random rotation (±15 degrees)
        angle = np.random.uniform(-15, 15)
        center = (W / 2, H / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        visual_aug = []
        for t in range(T):
            frame = visual[:, t, :, :].transpose(1, 2, 0)  # (H, W, C)
            frame = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            visual_aug.append(frame.transpose(2, 0, 1))  # (C, H, W)

        visual = np.stack(visual_aug, axis=1)  # (C, T, H, W)

        # Color jitter (brightness, contrast)
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            visual = np.clip(visual * brightness, 0, 1)

        if np.random.rand() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean = visual.mean()
            visual = np.clip((visual - mean) * contrast + mean, 0, 1)

        return visual

    def _add_noise(self, visual: np.ndarray, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise."""
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, visual.shape).astype(visual.dtype)
            visual = np.clip(visual + noise, 0, 1)

        if np.random.rand() > 0.5:
            noise_audio = np.random.normal(0, 0.01, audio.shape).astype(audio.dtype)
            audio = np.clip(audio + noise_audio, -1, 1)

        return visual, audio

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual, audio, label = self.base_dataset[idx]

        if not self.apply_augmentation:
            return visual, audio, label

        # Convert to numpy for augmentation
        visual_np = visual.numpy()
        audio_np = audio.numpy()

        # Temporal augmentation (speed variation)
        if np.random.rand() > 0.5:
            speed_factor = np.random.uniform(0.9, 1.1)
            visual_np, audio_np = self._temporal_augment(visual_np, audio_np, speed_factor)

        # Spatial augmentation
        visual_np = self._spatial_augment(visual_np)

        # Noise
        visual_np, audio_np = self._add_noise(visual_np, audio_np)

        # Convert back to tensors
        visual = torch.from_numpy(visual_np).float()
        audio = torch.from_numpy(audio_np).float()

        return visual, audio, label
