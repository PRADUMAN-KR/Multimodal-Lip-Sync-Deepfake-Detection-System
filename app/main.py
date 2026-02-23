from fastapi import FastAPI

from .api.routes import router as api_router
from .core.logger import get_logger
from .lifecycle import create_start_app_handler, create_stop_app_handler


logger = get_logger(__name__)


def get_application() -> FastAPI:
    logger.info("Creating FastAPI application instance for Lip Sync Detection Service")
    app = FastAPI(
        title="Lip Sync Detection Service",
        description="CNN-based audio–visual lip-sync detection microservice.",
        version="0.1.0",
    )

    app.include_router(api_router, prefix="/api")

    @app.get("/", tags=["root"])
    def root():
        return {
            "service": "Lip Sync Detection Service",
            "version": "0.1.0",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "predict": "POST /api/lip-sync with multipart video file",
        }

    app.add_event_handler("startup", create_start_app_handler(app))
    app.add_event_handler("shutdown", create_stop_app_handler(app))

    return app


app = get_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
