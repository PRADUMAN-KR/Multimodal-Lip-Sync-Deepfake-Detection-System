import asyncio
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from ..core.logger import get_logger

logger = get_logger(__name__)


def _run_predictor_from_path_sync(predictor: Any, input_path: Path) -> dict[str, Any]:
    # Reuse existing upload-based predictor path without changing core logic.
    with input_path.open("rb") as file_obj:
        upload = UploadFile(filename=input_path.name, file=file_obj)
        return asyncio.run(predictor.predict_from_upload(upload))


async def run_pipeline_job(predictor: Any, input_path: Path) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Job input file not found: {input_path}")
    return await asyncio.to_thread(_run_predictor_from_path_sync, predictor, input_path)
