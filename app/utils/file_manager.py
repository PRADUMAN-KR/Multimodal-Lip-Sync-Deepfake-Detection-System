from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

from fastapi import UploadFile

from ..core.logger import get_logger


logger = get_logger(__name__)


def save_upload_to_temp(upload: UploadFile, suffix: str = "") -> Path:
    """
    Persist an UploadFile to a temporary file on disk.
    Caller is responsible for cleanup.
    """
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload.file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    path = Path(tmp.name)
    logger.debug(
        "Saved upload to temporary file %s (filename=%s, content_type=%s, size=%d bytes)",
        path,
        getattr(upload, "filename", None),
        getattr(upload, "content_type", None),
        len(content),
    )
    return path


def split_av_paths(video_path: Path) -> Tuple[Path, Path]:
    """
    In a more advanced setup you might separate audio and video via ffmpeg.
    Here we assume the uploaded file is a standard video container with audio
    and we use the same path for both stages, allowing preprocessing to
    decide how to handle it.
    """
    logger.debug("Using same path for video and audio: %s", video_path)
    return video_path, video_path

