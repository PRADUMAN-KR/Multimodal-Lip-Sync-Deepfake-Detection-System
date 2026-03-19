import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from ..core.logger import get_logger
from ..db import database
from ..services.job_service import JobService
from ..utils.file_manager import save_upload_to_temp
from .job_schemas import JobResultResponse, JobStatusResponse, PredictJobCreateResponse

router = APIRouter(tags=["jobs"])
logger = get_logger(__name__)


@router.post("/predict", response_model=PredictJobCreateResponse)
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


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    if database.SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not ready")

    session = database.SessionLocal()
    try:
        service = JobService(session)
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status.value,
            input_path=job.input_path,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
        )
    finally:
        session.close()


@router.get("/result/{job_id}", response_model=JobResultResponse)
async def get_job_result(job_id: str) -> JobResultResponse:
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
        return JobResultResponse(job_id=job.job_id, status=job.status.value, result=parsed, error=job.error)
    finally:
        session.close()
