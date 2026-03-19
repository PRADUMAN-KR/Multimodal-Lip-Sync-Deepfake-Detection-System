import asyncio
from pathlib import Path

from ..core.logger import get_logger
from ..db import database
from ..services.job_service import JobService
from ..services.pipeline_runner import run_pipeline_job

logger = get_logger(__name__)


class JobWorker:
    def __init__(
        self,
        predictor,
        poll_interval_sec: float = 1.0,
        processing_timeout_sec: int = 900,
    ):
        self.predictor = predictor
        self.poll_interval_sec = max(0.2, poll_interval_sec)
        self.processing_timeout_sec = max(30, processing_timeout_sec)
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Job worker started")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
        logger.info("Job worker stopped")

    async def _run_loop(self) -> None:
        if database.SessionLocal is None:
            raise RuntimeError("Database SessionLocal is not initialized")

        while not self._stop_event.is_set():
            session = database.SessionLocal()
            try:
                service = JobService(session)
                job = service.get_next_claimable_job(self.processing_timeout_sec)
                if job is None:
                    await asyncio.sleep(self.poll_interval_sec)
                    continue

                logger.info("Job picked: job_id=%s", job.job_id)
                try:
                    result = await run_pipeline_job(self.predictor, Path(job.input_path))
                    service.mark_completed(job.job_id, result)
                    logger.info("Job completed: job_id=%s", job.job_id)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Job failed: job_id=%s", job.job_id)
                    service.mark_failed(job.job_id, str(exc))
            finally:
                session.close()
