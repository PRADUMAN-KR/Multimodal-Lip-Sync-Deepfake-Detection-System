from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Settings(BaseModel):
    project_name: str = "Lip Sync Detection Service"
    model_path: Path = Path("weights") / "best_model.pth"
    device: str = "cuda"  # or "cpu"
    confidence_threshold: float = 0.5
    use_torchscript: bool = False
    use_half_precision: bool = True
    uncertainty_margin: float = 0.05
    confidence_smoothing: str = "median"  # one of: none, median, trimmed_mean
    trim_ratio: float = 0.1
    max_tracks: int = 3
    refine_margin: float = 0.08
    refine_top_k: int = 2
    chunk_size: int = 32
    chunk_stride: int = 16
    long_video_threshold_sec: float = 2.0
    max_total_frames: int = 900

    # Confidence Margin Rule: flag uncertain when top-2 track confidence scores
    # are within this margin of each other (cuts FP in multi-speaker videos).
    confidence_margin: float = 0.10

    # Output Calibration: correct model overconfidence without retraining.
    # Options: "none" | "temperature" | "platt" | "isotonic"
    #   temperature: softens logits by dividing by T (T > 1 reduces overconfidence)
    #   platt:       affine transform of logit: sigmoid(a * logit + b)
    #   isotonic:    monotone regression on probabilities (requires a fitted .pkl)
    calibration_method: str = "none"
    calibration_temperature: float = 1.0
    calibration_platt_a: float = 1.0
    calibration_platt_b: float = 0.0
    calibration_isotonic_path: Optional[str] = None  # path to fitted .pkl

    # Mouth Motion Energy Check: guards against silent-mouth deepfakes.
    #   likely_fake  – audio loud   + mouth nearly still → reduce P(REAL)
    #   uncertain    – audio silent + mouth nearly still → don't predict fake
    mouth_motion_check: bool = True
    # Mean abs frame-diff in lower face region below this → "near zero motion"
    mouth_motion_low_threshold: float = 0.015
    # P(REAL) penalty applied when check result is "likely_fake"
    mouth_motion_fake_penalty: float = 0.10
    # Mean mel-dB threshold: above this = "loud audio"  (mel-dB ≤ 0, ref=max)
    audio_energy_high_threshold: float = -25.0
    # Mean mel-dB threshold: below this = "quiet/silent audio"
    # Set conservatively at -50 dB: a person speaking quietly (e.g. -53 dB)
    # with near-zero mouth motion should be treated as "uncertain", not fake.
    audio_energy_low_threshold: float = -50.0

    # Sparse-real-signal guard (long video only):
    # If the model shows very low overall confidence YET at least one window
    # shows a notable real signal, it is safer to return "uncertain" than
    # commit to a fake verdict.  Applies when ALL of these hold:
    #   - final_confidence  < weak_real_gate              (overall very fake-leaning)
    #   - max(window_confs) >= weak_real_window_threshold  (at least one real-ish window)
    weak_real_gate: float = 0.08
    weak_real_window_threshold: float = 0.30


def get_settings() -> Settings:
    """
    Return application settings.

    In production, extend this to read from environment variables,
    secrets managers, or configuration files rather than hardcoding.
    """
    return Settings()
