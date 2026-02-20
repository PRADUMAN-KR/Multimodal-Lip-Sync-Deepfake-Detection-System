from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi import Request

from ..core.logger import get_logger
from .schemas import LipSyncResponse

router = APIRouter(tags=["lip-sync"])
logger = get_logger(__name__)


def get_predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        settings = getattr(request.app.state, "settings", None)
        path = settings.model_path if settings else "weights/best_model.pth"
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Place trained weights at {path} and restart the service.",
        )
    return predictor


@router.post("/lip-sync", response_model=LipSyncResponse)
async def check_lip_sync(
    request: Request,
    video_file: UploadFile = File(..., description="Video file containing face and audio"),
    predictor=Depends(get_predictor),
) -> LipSyncResponse:
    """Run AI manipulation detection on an uploaded video clip (≈2–3 seconds).
    
    Detects if the video has been modified by AI tools (Wav2Lip, DeepFaceLab, etc.)
    or if it's authentic/natural.
    """
    logger.info(
        "Received lip-sync manipulation detection request from %s, filename=%s, content_type=%s",
        request.client.host if request.client else "unknown",
        video_file.filename,
        video_file.content_type,
    )
    try:
        result = await predictor.predict_from_upload(video_file)
    except ValueError as e:
        logger.warning("Lip-sync request failed with client error: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Lip-sync request failed with server error")
        raise HTTPException(status_code=500, detail="Lip-sync inference failed") from e

    logger.info(
        "Manipulation detection completed: is_real=%s, is_fake=%s, confidence=%.4f, manipulation_prob=%.4f",
        result.get("is_real"),
        result.get("is_fake"),
        result.get("confidence"),
        result.get("manipulation_probability"),
    )
    return LipSyncResponse(**result)

