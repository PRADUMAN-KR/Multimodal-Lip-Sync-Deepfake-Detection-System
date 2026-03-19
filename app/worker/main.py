import asyncio

from ..config import get_settings
from ..core.device import get_device
from ..core.logger import get_logger
from ..db.database import init_db, init_engine
from ..inference.predictor import Predictor
from .worker import JobWorker

logger = get_logger(__name__)


async def run_worker() -> None:
    settings = get_settings()
    device = get_device(settings.device)
    init_engine(settings.sqlite_db_url)
    init_db()

    predictor = Predictor(
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
        confidence_margin=settings.confidence_margin,
        calibration_method=settings.calibration_method,
        calibration_temperature=settings.calibration_temperature,
        calibration_platt_a=settings.calibration_platt_a,
        calibration_platt_b=settings.calibration_platt_b,
        calibration_isotonic_path=settings.calibration_isotonic_path,
        mouth_motion_check=settings.mouth_motion_check,
        mouth_motion_low_threshold=settings.mouth_motion_low_threshold,
        mouth_motion_fake_penalty=settings.mouth_motion_fake_penalty,
        audio_energy_high_threshold=settings.audio_energy_high_threshold,
        audio_energy_low_threshold=settings.audio_energy_low_threshold,
        weak_real_gate=settings.weak_real_gate,
        weak_real_window_threshold=settings.weak_real_window_threshold,
        fake_vote_gate=settings.fake_vote_gate,
        fake_vote_min_windows=settings.fake_vote_min_windows,
    )
    worker = JobWorker(
        predictor=predictor,
        poll_interval_sec=settings.worker_poll_interval_sec,
        processing_timeout_sec=settings.worker_processing_timeout_sec,
    )
    worker.start()
    logger.info("Worker process running")
    try:
        while True:
            await asyncio.sleep(60)
    finally:
        await worker.stop()
        predictor.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
