import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.orm import Session

from ..core.logger import get_logger
from ..db.models import Job, JobStatus

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobService:
    def __init__(self, session: Session):
        self.session = session

    def create_job(self, input_path: Path, payload: dict[str, Any] | None = None) -> Job:
        job = Job(
            job_id=str(uuid4()),
            status=JobStatus.PENDING,
            input_path=str(input_path),
            payload=json.dumps(payload) if payload else None,
        )
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.session.get(Job, job_id)

    def get_next_claimable_job(self, processing_timeout_sec: int = 900) -> Job | None:
        stale_cutoff = _utc_now() - timedelta(seconds=processing_timeout_sec)
        stmt: Select[tuple[Job]] = (
            select(Job)
            .where(
                or_(
                    Job.status == JobStatus.PENDING,
                    and_(Job.status == JobStatus.PROCESSING, Job.updated_at < stale_cutoff),
                )
            )
            .order_by(Job.created_at.asc())
            .limit(1)
        )
        job = self.session.execute(stmt).scalar_one_or_none()
        if job is None:
            return None

        claimed = (
            self.session.query(Job)
            .filter(
                Job.job_id == job.job_id,
                Job.status == job.status,
                Job.updated_at == job.updated_at,
            )
            .update(
                {
                    Job.status: JobStatus.PROCESSING,
                    Job.updated_at: _utc_now(),
                    Job.error: None,
                },
                synchronize_session=False,
            )
        )
        if claimed != 1:
            self.session.rollback()
            return None

        self.session.commit()
        return self.session.get(Job, job.job_id)

    def mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        self.session.query(Job).filter(Job.job_id == job_id).update(
            {
                Job.status: JobStatus.COMPLETED,
                Job.result: json.dumps(result),
                Job.error: None,
                Job.updated_at: _utc_now(),
            },
            synchronize_session=False,
        )
        self.session.commit()

    def mark_failed(self, job_id: str, error: str) -> None:
        self.session.query(Job).filter(Job.job_id == job_id).update(
            {
                Job.status: JobStatus.FAILED,
                Job.error: error[:4000],
                Job.updated_at: _utc_now(),
            },
            synchronize_session=False,
        )
        self.session.commit()
