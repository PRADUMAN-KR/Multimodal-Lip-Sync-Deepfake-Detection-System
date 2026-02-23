from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ..core.logger import get_logger
from ..utils.metrics import compute_metrics
from .schemas import (
    BatchEvaluateRequest,
    BatchEvaluateResponse,
    LipSyncResponse,
)

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
        "Manipulation detection completed: verdict=%s, is_real=%s, is_fake=%s, confidence=%.4f, manipulation_prob=%.4f",
        result.get("verdict"),
        result.get("is_real"),
        result.get("is_fake"),
        result.get("confidence"),
        result.get("manipulation_probability"),
    )
    return LipSyncResponse(**result)


@router.post("/metrics/evaluate", response_model=BatchEvaluateResponse)
async def evaluate_batch(body: BatchEvaluateRequest) -> BatchEvaluateResponse:
    """Compute Precision, Recall, F1 and Accuracy over a labelled batch.

    Supply a list of ``{predicted_is_fake, true_is_fake}`` pairs collected from
    previous ``/lip-sync`` calls (plus any ground-truth labels you hold).
    The positive class is **fake** (label 1), matching the standard deepfake-
    detection convention.

    Example request body::

        {
          "evaluations": [
            {"predicted_is_fake": true,  "true_is_fake": true},
            {"predicted_is_fake": false, "true_is_fake": false},
            {"predicted_is_fake": true,  "true_is_fake": false}
          ]
        }
    """
    if not body.evaluations:
        raise HTTPException(status_code=422, detail="evaluations list must not be empty.")

    y_true = [int(e.true_is_fake) for e in body.evaluations]
    y_pred = [int(e.predicted_is_fake) for e in body.evaluations]

    metrics = compute_metrics(y_true, y_pred, positive_label=1)

    logger.info(
        "Batch evaluation: total=%d, precision=%.4f, recall=%.4f, f1=%.4f, accuracy=%.4f",
        metrics["total"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["accuracy"],
    )
    return BatchEvaluateResponse(**metrics)
