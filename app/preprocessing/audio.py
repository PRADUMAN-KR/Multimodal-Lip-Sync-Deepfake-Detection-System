import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio.functional as F

from ..core.logger import get_logger


logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mpg", ".mpeg", ".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _extract_audio_from_video(video_path: Path, sr: int = 16000) -> Path:
    """Extract audio from video to a temporary WAV using ffmpeg. Caller should unlink the result."""
    out = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    ffmpeg = _ffmpeg_path()
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return out


def _ffmpeg_path() -> str:
    """Reuse same logic as download script: project tools/ or Homebrew."""
    try:
        root = Path(__file__).resolve().parent.parent.parent
        local = root / "tools" / "ffmpeg"
        if local.exists():
            return str(local)
    except Exception:
        pass
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if Path(p).exists():
            return p
    return "ffmpeg"


def preprocess_audio(
    path: Path,
    sr: int = 16000,
    n_mels: int = 80,
    hop_length: int = 160,
    win_length: int = 400,
    target_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Load audio and compute Mel-spectrogram.
    If path is a video file (.mpg, .mp4, etc.), extracts audio with ffmpeg first.
    If target_frames is set, pad or truncate the time dimension to that length.

    Output: (1, F, T) float32 suitable for 2D CNN encoder.
    """
    load_path = path
    temp_wav = None
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        logger.debug("Extracting audio from video %s", path)
        temp_wav = _extract_audio_from_video(path, sr=sr)
        load_path = temp_wav
    try:
        logger.debug("Loading audio from %s (sr=%d)", load_path, sr)
        y, _ = librosa.load(load_path, sr=sr)
    finally:
        if temp_wav is not None and temp_wav.exists():
            try:
                temp_wav.unlink()
            except Exception:
                pass
    if y.size == 0:
        raise ValueError(f"Empty audio signal for {path}")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.astype("float32")
    mel_db = np.expand_dims(mel_db, axis=0)  # (1, F, T)

    if target_frames is not None:
        T_current = mel_db.shape[2]
        if T_current < target_frames:
            padding = np.repeat(mel_db[:, :, -1:], target_frames - T_current, axis=2)
            mel_db = np.concatenate([mel_db, padding], axis=2)
        elif T_current > target_frames:
            mel_db = mel_db[:, :, :target_frames]

    logger.debug("Audio preprocessed to mel-spectrogram with shape %s", mel_db.shape)
    return mel_db


def detect_voice_activity(
    path: Path,
    sr: int = 16000,
    trigger_level: float = 7.0,
    trigger_time: float = 0.25,
    search_time: float = 1.0,
    allowed_gap: float = 0.25,
) -> Tuple[np.ndarray, float]:
    """
    Detect voice activity using torchaudio VAD.
    
    Args:
        path: Audio or video file path
        sr: Sample rate
        trigger_level: VAD trigger level (lower = more sensitive)
        trigger_time: Time constant for ignoring short bursts
        search_time: Amount of audio to search before trigger
        allowed_gap: Allowed gap between bursts
        
    Returns:
        Tuple of (voice_activity_mask, duration_sec) where:
        - voice_activity_mask: boolean array (T,) indicating speech presence per mel-frame
        - duration_sec: total audio duration in seconds
        
    The mask is aligned with mel-spectrogram time frames (hop_length=160 @ 16kHz).
    """
    load_path = path
    temp_wav = None
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        logger.debug("Extracting audio from video %s for VAD", path)
        temp_wav = _extract_audio_from_video(path, sr=sr)
        load_path = temp_wav
    
    try:
        logger.debug("Loading audio for VAD from %s (sr=%d)", load_path, sr)
        y, _ = librosa.load(load_path, sr=sr)
    finally:
        if temp_wav is not None and temp_wav.exists():
            try:
                temp_wav.unlink()
            except Exception:
                pass
    
    if y.size == 0:
        logger.warning("Empty audio signal for VAD, returning all-True mask")
        # Return a default mask (assume all speech) if audio is empty
        duration_sec = 0.0
        # Estimate frames based on typical mel hop_length
        hop_length = 160
        n_frames = max(1, int(len(y) / hop_length))
        return np.ones(n_frames, dtype=bool), duration_sec
    
    duration_sec = len(y) / sr
    
    # Convert to torch tensor for VAD
    waveform = torch.from_numpy(y).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension: (1, T)
    
    try:
        # Apply VAD - this trims silence from ends
        vad_waveform = F.vad(
            waveform=waveform,
            sample_rate=sr,
            trigger_level=trigger_level,
            trigger_time=trigger_time,
            search_time=search_time,
            allowed_gap=allowed_gap,
        )
        
        # VAD returns trimmed audio - we need to map back to original timeline
        # So we'll use energy-based detection on the original waveform instead
        # This gives us per-frame speech presence
        hop_length = 160  # Match mel-spectrogram hop_length
        frame_length = 400  # Match mel win_length
        
        # Compute frame-level energy
        n_frames = int(np.ceil(len(y) / hop_length))
        frame_energies = []
        
        for i in range(n_frames):
            start_idx = i * hop_length
            end_idx = min(start_idx + frame_length, len(y))
            if start_idx >= len(y):
                break
            frame_energy = np.mean(y[start_idx:end_idx] ** 2)
            frame_energies.append(frame_energy)
        
        frame_energies = np.array(frame_energies)
        
        # Adaptive threshold: use percentile-based approach for better sensitivity
        # This adapts to the actual audio content rather than relying on VAD-trimmed energy
        if len(frame_energies) > 0:
            # Use 20th percentile as threshold - catches most speech including quieter segments
            # But still filters out pure silence/noise
            energy_median = np.median(frame_energies)
            energy_p20 = np.percentile(frame_energies, 20)
            # Use the lower of: 20th percentile or 5% of median (whichever is more sensitive)
            threshold = min(energy_p20, energy_median * 0.05)
            threshold = max(1e-8, threshold)  # Ensure non-zero
            
            # If VAD found significant speech, validate threshold isn't too high
            if vad_waveform.numel() > 0:
                vad_energy = float(torch.mean(vad_waveform ** 2))
                # Ensure threshold is at most 5% of VAD energy (very sensitive)
                threshold = min(threshold, max(1e-8, vad_energy * 0.05))
        else:
            threshold = 1e-8
        
        voice_mask = frame_energies >= threshold
        
        # Smooth the mask to avoid flickering, but be more lenient
        # Require at least 1 frame in a 3-frame window (more permissive)
        smoothed_mask = np.zeros_like(voice_mask)
        for i in range(len(voice_mask)):
            window = voice_mask[max(0, i-1):min(len(voice_mask), i+2)]
            smoothed_mask[i] = np.sum(window) >= 1
        
        logger.debug(
            "VAD detected speech in %.1f%% of frames (%d/%d)",
            np.mean(smoothed_mask) * 100,
            np.sum(smoothed_mask),
            len(smoothed_mask)
        )
        
        return smoothed_mask, duration_sec
        
    except Exception as e:
        logger.warning("VAD detection failed: %s, falling back to all-True mask", e)
        # Fallback: assume all frames contain speech
        hop_length = 160
        n_frames = max(1, int(len(y) / hop_length))
        return np.ones(n_frames, dtype=bool), duration_sec

