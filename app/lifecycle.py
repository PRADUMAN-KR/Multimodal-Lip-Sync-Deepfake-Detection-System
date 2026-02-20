from fastapi import FastAPI

from .config import get_settings
from .core.device import get_device
from .core.logger import get_logger
from .inference.predictor import Predictor


logger = get_logger(__name__)


def create_start_app_handler(app: FastAPI):
    async def start_app() -> None:
        settings = get_settings()
        device = get_device(settings.device)
        logger.info(
            "Starting Lip Sync Detection Service on device=%s, model_path=%s, "
            "confidence_threshold=%.3f, torchscript=%s, half_precision=%s, "
            "uncertainty_margin=%.3f, confidence_smoothing=%s, trim_ratio=%.2f, "
            "max_tracks=%d, refine_margin=%.3f, refine_top_k=%d",
            device,
            settings.model_path,
            settings.confidence_threshold,
            settings.use_torchscript,
            settings.use_half_precision,
            settings.uncertainty_margin,
            settings.confidence_smoothing,
            settings.trim_ratio,
            settings.max_tracks,
            settings.refine_margin,
            settings.refine_top_k,
        )
        app.state.settings = settings
        app.state.device = device

        if not settings.model_path.is_file():
            logger.warning(
                "Model weights not found at %s; prediction endpoints will return 503. "
                "Train a model and save weights to this path, or set MODEL_PATH to a valid file.",
                settings.model_path,
            )
            app.state.predictor = None
        else:
            app.state.predictor = Predictor(
                model_path=settings.model_path,
                device=device,
                confidence_threshold=settings.confidence_threshold,
                use_torchscript=settings.use_torchscript,
                use_half_precision=settings.use_half_precision,
                uncertainty_margin=settings.uncertainty_margin,
                confidence_smoothing=settings.confidence_smoothing,
                trim_ratio=settings.trim_ratio,
                max_tracks=settings.max_tracks,
                refine_margin=settings.refine_margin,
                refine_top_k=settings.refine_top_k,
                chunk_size=settings.chunk_size,
                chunk_stride=settings.chunk_stride,
                long_video_threshold_sec=settings.long_video_threshold_sec,
                max_total_frames=settings.max_total_frames,
            )

    return start_app


def create_stop_app_handler(app: FastAPI):
    async def stop_app() -> None:
        # Add any resource cleanup here (e.g., closing pools)
        predictor: Predictor | None = getattr(app.state, "predictor", None)
        if predictor is not None:
            predictor.close()
        logger.info("Lip Sync Detection Service shutdown complete")

    return stop_app

