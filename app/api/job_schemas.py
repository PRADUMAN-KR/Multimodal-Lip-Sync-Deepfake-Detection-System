from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PredictJobCreateResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    input_path: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
