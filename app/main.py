"""Точка входа для Face Recognition Service API."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
import uvicorn
import os
import json
from datetime import datetime
from typing import Any


def json_serializer(obj: Any) -> Any:
    """
    Кастомный JSON сериализатор для datetime и других специальных типов.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_custom_json_response(content: Any, status_code: int = 200, **kwargs) -> JSONResponse:
    """
    Создает JSONResponse с кастомным сериализатором для datetime.
    """
    return JSONResponse(
        content=json.loads(json.dumps(content, default=json_serializer)),
        status_code=status_code,
        **kwargs
    )


async def http_exception_handler(request, exc):
    """Кастомный обработчик HTTP исключений с правильной сериализацией datetime."""
    error_content = {
        "success": False,
        "error_code": getattr(exc, "status_code", 500),
        "error_details": {"error": str(exc.detail)} if hasattr(exc, "detail") else {"error": str(exc)},
        "timestamp": datetime.now(),
    }
    return create_custom_json_response(error_content, status_code=exc.status_code)


async def general_exception_handler(request, exc):
    """Кастомный обработчик общих исключений с правильной сериализацией datetime."""
    error_content = {
        "success": False,
        "error_code": "INTERNAL_ERROR",
        "error_details": {"error": str(exc)},
        "timestamp": datetime.now(),
    }
    return create_custom_json_response(error_content, status_code=500)

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
        from .tasks.scheduler import start_global_scheduler
        start_global_scheduler()
        logger.info("✅ Cleanup scheduler started")
    except Exception as e:
        logger.warning(f"⚠️ Failed to start cleanup scheduler: {e}")

    logger.info("✅ Service started successfully")
    yield

    # Shutdown
    logger.info("🛑 Service shutting down...")
    
    # Phase 5: Остановка cleanup scheduler
    try:
        from .tasks.scheduler import stop_global_scheduler
        stop_global_scheduler()
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

    # Регистрация кастомных обработчиков исключений
    from fastapi.exceptions import HTTPException
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # Регистрация кастомных обработчиков исключений
    from fastapi.exceptions import HTTPException
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # Регистрация роутов
    app.include_router(health.router, prefix="/api/v1")

    # Алиасы для совместимости (только если health роутер не имеет префикса)
    # Если health роутер уже включен с префиксом, алиасы создают дубли
    # app.add_api_route("/status", health.detailed_status_check, methods=["GET"])
    # app.add_api_route("/health", health.health_check, methods=["GET"])
    # app.add_api_route("/ready", health.readiness_check, methods=["GET"])
    # app.add_api_route("/live", health.liveness_check, methods=["GET"])
    # app.add_api_route("/metrics", health.get_metrics, methods=["GET"])

    # Основные роуты
    # Роутеры уже имеют свои префиксы, поэтому не добавляем дополнительные
    app.include_router(upload.router)
    app.include_router(verify.router)
    app.include_router(liveness.router)
    app.include_router(reference.router)
    app.include_router(admin.router)
    app.include_router(auth.router)  # Включаем только один раз

    # Обработчик для favicon.ico (решает проблему 401 для браузерных запросов)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Отдача favicon с fallback на пустую иконку."""
        # Путь к файлу (опционально)
        favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
        
        if os.path.isfile(favicon_path):
            return FileResponse(favicon_path, media_type="image/x-icon")
        else:
            # Возвращаем пустую прозрачную иконку (1x1 PNG)
            empty_icon = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
                b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return Response(content=empty_icon, media_type="image/png")

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

    # Алиасы для совместимости (только если health роутер не имеет префикса)
    # Если health роутер уже включен с префиксом, алиасы создают дубли
    # app.add_api_route("/status", health.detailed_status_check, methods=["GET"])
    # app.add_api_route("/health", health.health_check, methods=["GET"])
    # app.add_api_route("/ready", health.readiness_check, methods=["GET"])
    # app.add_api_route("/live", health.liveness_check, methods=["GET"])
    # app.add_api_route("/metrics", health.get_metrics, methods=["GET"])

    # Основные роуты
    # Роутеры уже имеют свои префиксы, поэтому не добавляем дополнительные
    app.include_router(upload.router)
    app.include_router(verify.router)
    app.include_router(liveness.router)
    app.include_router(reference.router)
    app.include_router(admin.router)

    # Обработчик для favicon.ico (решает проблему 401 для браузерных запросов)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Отдача favicon с fallback на пустую иконку."""
        # Путь к файлу (опционально)
        favicon_path = os.path.join(os.path.dirname(__file__), "static", "favicon.ico")
        
        if os.path.isfile(favicon_path):
            return FileResponse(favicon_path, media_type="image/x-icon")
        else:
            # Возвращаем пустую прозрачную иконку (1x1 PNG)
            empty_icon = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
                b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return Response(content=empty_icon, media_type="image/png")

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
