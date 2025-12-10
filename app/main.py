"""
Точка входа для Face Recognition Service API.
Создание FastAPI приложения и настройка всех компонентов.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .routes import (
    health, 
    upload, 
    verify, 
    liveness, 
    reference, 
    admin
)
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware
from .utils.logger import setup_logger


def create_app() -> FastAPI:
    """
    Создание и настройка FastAPI приложения.
    
    Returns:
        FastAPI: Настроенное приложение
    """
    # Создание приложения
    app = FastAPI(
        title="Face Recognition Service",
        description="API для распознавания лиц, верификации и проверки живости",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Настройка логгера
    logger = setup_logger()
    app.logger = logger

    # Добавление middleware
    setup_middleware(app)

    # Регистрация роутов
    register_routes(app)

    # Настройка обработчиков
    setup_handlers(app)

    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Настройка middleware для приложения.
    
    Args:
        app: FastAPI приложение
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)


def register_routes(app: FastAPI) -> None:
    """
    Регистрация всех роутов приложения.
    
    Args:
        app: FastAPI приложение
    """
    # Health check endpoints
    app.include_router(health.router, prefix="/api/v1")
    # Аллиасы без префикса для совместимости с внешними требованиями (/status, /health)
    app.add_api_route("/status", health.detailed_status_check, methods=["GET"])
    app.add_api_route("/health", health.health_check, methods=["GET"])
    
    # Core functionality
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(verify.router, prefix="/api/v1")
    app.include_router(liveness.router, prefix="/api/v1")
    app.include_router(reference.router, prefix="/api/v1")
    
    # Admin endpoints
    app.include_router(admin.router, prefix="/api/v1")


def setup_handlers(app: FastAPI) -> None:
    """
    Настройка обработчиков событий приложения.
    
    Args:
        app: FastAPI приложение
    """
    
    @app.on_event("startup")
    async def startup_event():
        """Действия при запуске приложения."""
        app.logger.info("Face Recognition Service starting up...")
        
        # Инициализация подключений к внешним сервисам
        # (БД, Redis, S3, etc.)
        
        app.logger.info("Face Recognition Service started successfully")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Действия при остановке приложения."""
        app.logger.info("Face Recognition Service shutting down...")
        
        # Закрытие подключений к внешним сервисам
        
        app.logger.info("Face Recognition Service shutdown completed")


# Создание экземпляра приложения
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )