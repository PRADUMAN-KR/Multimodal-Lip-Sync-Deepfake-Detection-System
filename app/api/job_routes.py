import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from ..core.logger import get_logger
from ..db import database
from ..services.job_service import JobService
from ..utils.file_manager import save_upload_to_temp
from .job_schemas import JobResultResponse, PredictJobCreateResponse

router = APIRouter(tags=["Asynchronous"])
logger = get_logger(__name__)

MINIMAL_RESULT_KEYS = {
    "verdict",
    "is_real",
    "is_fake",
    "confidence",
    "manipulation_probability",
    "detail",
}


def _to_minimal_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {k: v for k, v in result.items() if k in MINIMAL_RESULT_KEYS}


@router.post("/jobs", response_model=PredictJobCreateResponse)
async def create_predict_job(
    request: Request,
    video_file: UploadFile = File(..., description="Video file containing face and audio"),
) -> PredictJobCreateResponse:
    if database.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    if getattr(request.app.state, "predictor", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_path: Path = save_upload_to_temp(video_file, suffix=".mp4")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to persist upload: {exc}") from exc

    session = database.SessionLocal()
    try:
        service = JobService(session)
        job = service.create_job(input_path=input_path, payload={"filename": video_file.filename})
        logger.info("Job created: job_id=%s input=%s", job.job_id, input_path)
        return PredictJobCreateResponse(
            job_id=job.job_id,
            status=job.status.value,
            created_at=job.created_at,
        )
    finally:
        session.close()


@router.get("/result/{job_id}", response_model=JobResultResponse)
async def get_job_result(
    job_id: str,
    include_debug: bool = Query(
        default=False,
        description="When true, returns full internal debug payload; otherwise returns minimal result fields.",
    ),
) -> JobResultResponse:
    if database.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    session = database.SessionLocal()
    try:
        service = JobService(session)
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status.value != "COMPLETED":
            if job.status.value == "FAILED":
                return JobResultResponse(job_id=job.job_id, status=job.status.value, error=job.error)
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Job not completed yet. Current status={job.status.value}",
            )
        parsed = json.loads(job.result) if job.result else None
        response_result = parsed if include_debug else _to_minimal_result(parsed)
        return JobResultResponse(job_id=job.job_id, status=job.status.value, result=response_result, error=job.error)
    finally:
        session.close()
