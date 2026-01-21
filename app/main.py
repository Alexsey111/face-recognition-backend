"""
app/main.py
Точка входа для Face Recognition Service API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
import uvicorn
import os
import json
from datetime import datetime, date, timezone
from typing import Any

from . import __version__
from .config import settings
from .routes import (
    health,
    upload,
    verify,
    liveness,
    reference,
    admin,
    auth,
    webhook,
    metrics,
)
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging_middleware import LoggingMiddleware
from .middleware.metrics import MetricsMiddleware, initialize_metrics
from .utils.logger import setup_logger


# -----------------------------
# JSON utils
# -----------------------------


def json_serializer(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_custom_json_response(
    content: Any, status_code: int = 200, **kwargs
) -> JSONResponse:
    return JSONResponse(
        content=json.loads(json.dumps(content, default=json_serializer)),
        status_code=status_code,
        **kwargs,
    )


# -----------------------------
# Exception handlers
# -----------------------------


async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": "HTTP_EXCEPTION",
            "error_details": {"error": exc.detail},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def general_exception_handler(request, exc: Exception):
    from .utils.exceptions import ValidationError, ProcessingError, NotFoundError

    if isinstance(exc, ValidationError):
        code, status = "VALIDATION_ERROR", 400
    elif isinstance(exc, ProcessingError):
        code, status = "PROCESSING_ERROR", 422
    elif isinstance(exc, NotFoundError):
        code, status = "NOT_FOUND", 404
    else:
        code, status = "INTERNAL_ERROR", 500

    return JSONResponse(
        status_code=status,
        content={
            "success": False,
            "error_code": code,
            "error_details": {"error": str(exc)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# -----------------------------
# Lifespan
# -----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = setup_logger()
    app.state.logger = logger
    logger.info("Service starting up")

    initialize_metrics()
    logger.info("Metrics initialized")

    from .services.auth_service import AuthService

    AuthService.init_redis()
    logger.info("Redis initialized")

    # Test environment DB bootstrap
    is_test_env = getattr(settings, "ENVIRONMENT", "") == "test" or bool(
        os.getenv("PYTEST_CURRENT_TEST")
    )

    if is_test_env:
        try:
            from .db import database as db_mod

            test_db_path = os.path.join(os.getcwd(), "test_sqlite.db")
            test_url = f"sqlite+aiosqlite:///{test_db_path}"

            if not str(db_mod.db_manager.database_url).startswith("sqlite"):
                db_mod.db_manager = db_mod.DatabaseManager(database_url=test_url)

            await db_mod.db_manager.create_tables()
        except Exception:
            logger.exception("Test DB initialization failed")

    # Schedulers
    try:
        from .tasks.scheduler import start_schedulers

        start_schedulers()
        logger.info("Schedulers started")
    except Exception:
        logger.warning("Schedulers not started", exc_info=True)

    yield

    logger.info("Service shutting down")

    try:
        from .tasks.scheduler import stop_schedulers

        await stop_schedulers()
        logger.info("Schedulers stopped")
    except Exception:
        logger.warning("Schedulers stop failed", exc_info=True)

    # Close cache service connections
    try:
        from .dependencies import shutdown_cache_service

        await shutdown_cache_service()
        logger.info("CacheService shutdown complete")
    except Exception:
        logger.warning("CacheService shutdown failed", exc_info=True)

    AuthService.close_redis()
    logger.info("Redis closed")


# -----------------------------
# App factory
# -----------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Face Recognition Service",
        description="API для распознавания лиц, верификации и проверки живости",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-New-Access-Token"],
    )

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)

    @app.get("/")
    async def root():
        return {
            "message": "Face Recognition Service API",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(verify.router, prefix="/api/v1")
    app.include_router(liveness.router, prefix="/api/v1")
    app.include_router(reference.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(webhook.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")

    app.include_router(metrics.router)

    app.add_api_route("/health", health.health_check, methods=["GET"])
    app.add_api_route("/status", health.detailed_status_check, methods=["GET"])
    app.add_api_route("/ready", health.readiness_check, methods=["GET"])
    app.add_api_route("/live", health.liveness_check, methods=["GET"])

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
        if os.path.isfile(path):
            return FileResponse(path, media_type="image/x-icon")
        return Response(status_code=204)

    return app


def create_test_app() -> FastAPI:
    app = FastAPI(
        title="Face Recognition Service (Test)",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(verify.router, prefix="/api/v1")
    app.include_router(liveness.router, prefix="/api/v1")
    app.include_router(reference.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(webhook.router, prefix="/api/v1")

    app.include_router(metrics.router)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
