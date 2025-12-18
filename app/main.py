"""Точка входа для Face Recognition Service API."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from . import __version__
from .config import settings
from .routes import health, upload, verify, liveness, reference, admin, auth
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .utils.logger import setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager для приложения."""
    # Startup
    logger = setup_logger()
    app.state.logger = logger
    logger.info("🚀 Face Recognition Service starting up...")

    # Инициализация подключений (если нужно)
    # await init_database()
    # await init_redis()

    # Phase 5: Запуск cleanup scheduler для автоматической очистки
    try:
        from .tasks.scheduler import start_cleanup_scheduler
        start_cleanup_scheduler()
        logger.info("✅ Cleanup scheduler started")
    except Exception as e:
        logger.warning(f"⚠️ Failed to start cleanup scheduler: {e}")

    logger.info("✅ Service started successfully")
    yield

    # Shutdown
    logger.info("🛑 Service shutting down...")
    
    # Phase 5: Остановка cleanup scheduler
    try:
        from .tasks.scheduler import stop_cleanup_scheduler
        stop_cleanup_scheduler()
        logger.info("✅ Cleanup scheduler stopped")
    except Exception as e:
        logger.warning(f"⚠️ Failed to stop cleanup scheduler: {e}")
    
    # Закрытие подключений (если нужно)
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
    # ✅ Только существующие middleware
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)

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

    # Основные роуты
    # Роутеры уже имеют свои префиксы, поэтому не добавляем дополнительные
    app.include_router(upload.router)
    app.include_router(verify.router)
    app.include_router(liveness.router)
    app.include_router(reference.router)
    app.include_router(admin.router)
    app.include_router(auth.router)  # Добавляем роутер auth для тестов
    app.include_router(auth.router)  # Включаем только один раз

    return app


def create_test_app() -> FastAPI:
    """Создание тестового FastAPI приложения без middleware авторизации."""
    app = FastAPI(
        title="Face Recognition Service (Test)",
        description="Тестовая версия API для распознавания лиц",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware (оставляем для тестов)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-New-Access-Token"],
    )

    # НЕ добавляем AuthMiddleware и RateLimitMiddleware для тестов
    # Это позволяет тестировать endpoints без необходимости в JWT токенах

    # Root endpoint
    @app.get("/")
    async def root():
        """Корневой endpoint."""
        return {
            "message": "Face Recognition Service API (Test)",
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

    # Основные роуты
    # Роутеры уже имеют свои префиксы, поэтому не добавляем дополнительные
    app.include_router(upload.router)
    app.include_router(verify.router)
    app.include_router(liveness.router)
    app.include_router(reference.router)
    app.include_router(admin.router)

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
