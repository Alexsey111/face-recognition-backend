"""Точка входа для Face Recognition Service API."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from . import __version__
from .config import settings
from .routes import health

# TODO Phase 3+: Добавить остальные роуты
# from .routes import upload, verify, liveness, reference, admin
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware
from .utils.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager для приложения."""
    # Startup
    logger = setup_logger()
    app.state.logger = logger
    logger.info("🚀 Face Recognition Service starting up...")

    # TODO Phase 3: Инициализация подключений
    # await init_database()
    # await init_redis()

    logger.info("✅ Service started successfully")
    yield

    # Shutdown
    logger.info("🛑 Service shutting down...")
    # TODO Phase 3: Закрытие подключений
    # await close_database()
    # await close_redis()
    logger.info("✅ Shutdown completed")


def create_app() -> FastAPI:
    """Создание и настройка FastAPI приложения."""
    app = FastAPI(
        title="Face Recognition Service",
        description="API для распознавания лиц, верификации и проверки живости",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-New-Access-Token"],
    )

    # Custom middleware (порядок важен: снизу вверх)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    # Root endpoint
    @app.get("/")
    async def root():
        """Корневой endpoint."""
        return {
            "message": "Face Recognition Service API",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "status": "/status",
        }

    # Регистрация роутов
    app.include_router(health.router, prefix="/api/v1")

    # Алиасы для совместимости
    app.add_api_route("/status", health.detailed_status_check, methods=["GET"])
    app.add_api_route("/health", health.health_check, methods=["GET"])
    app.add_api_route("/ready", health.readiness_check, methods=["GET"])
    app.add_api_route("/live", health.liveness_check, methods=["GET"])
    app.add_api_route("/metrics", health.get_metrics, methods=["GET"])

    # TODO Phase 3+: Добавить остальные роуты
    # app.include_router(upload.router, prefix="/api/v1")
    # app.include_router(verify.router, prefix="/api/v1")
    # app.include_router(liveness.router, prefix="/api/v1")
    # app.include_router(reference.router, prefix="/api/v1")
    # app.include_router(admin.router, prefix="/api/v1")

    return app


# Создание экземпляра приложения
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
