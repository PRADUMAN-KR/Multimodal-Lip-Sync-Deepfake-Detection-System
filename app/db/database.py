from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.logger import get_logger
from .models import Base

logger = get_logger(__name__)

engine = None
SessionLocal = None


def init_engine(db_url: str) -> None:
    global engine, SessionLocal
    if engine is not None and SessionLocal is not None:
        return

    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    logger.info("Database engine initialized: %s", db_url)


def init_db() -> None:
    if engine is None:
        raise RuntimeError("Database engine is not initialized. Call init_engine first.")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured")


def get_session() -> Generator[Session, None, None]:
    if SessionLocal is None:
        raise RuntimeError("SessionLocal is not initialized. Call init_engine first.")
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
