from pathlib import Path

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


def get_settings() -> Settings:
    """
    Return application settings.

    In production, extend this to read from environment variables,
    secrets managers, or configuration files rather than hardcoding.
    """
    return Settings()

