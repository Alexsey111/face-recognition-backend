"""
Middleware для настройки CORS (Cross-Origin Resource Sharing).
Конфигурация политики CORS для API.
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CORSMiddleware:
    """
    Middleware для настройки CORS.
    """

    def __init__(self):
        # Преобразуем строку CORS_ORIGINS в список
        if isinstance(settings.CORS_ORIGINS, str):
            self.allowed_origins = [
                origin.strip() for origin in settings.CORS_ORIGINS.split(",")
            ]
        else:
            self.allowed_origins = settings.CORS_ORIGINS
        self.allowed_methods = [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
            "PATCH",
            "HEAD",
        ]
        self.allowed_headers = [
            "Accept",
            "Accept-Language",
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-API-Key",
            "X-Request-ID",
            "User-Agent",
            "Referer",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
        ]
        self.allowed_credentials = True
        self.max_age = 86400  # 24 часа

        # Специальные настройки для разных окружений
        if settings.DEBUG:
            # В debug режиме разрешаем больше источников
            self.allowed_origins.extend(
                [
                    "http://localhost:3000",
                    "http://localhost:8080",
                    "http://localhost:8081",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:8080",
                    "http://127.0.0.1:8081",
                ]
            )

    def add_cors_middleware(self, app: FastAPI) -> None:
        """
        Добавление CORS middleware к FastAPI приложению.

        Args:
            app: FastAPI приложение
        """
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.allowed_origins,
            allow_credentials=self.allowed_credentials,
            allow_methods=self.allowed_methods,
            allow_headers=self.allowed_headers,
            max_age=self.max_age,
            expose_headers=[
                "X-Request-ID",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
                "X-New-Access-Token",
            ],
        )

        logger.info(f"CORS middleware added with origins: {self.allowed_origins}")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware для добавления security headers.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Добавление security headers к ответу.

        Args:
            request: HTTP запрос
            call_next: Следующий обработчик

        Returns:
            Response: HTTP ответ с security headers
        """
        response = await call_next(request)

        # Добавляем security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        # HSTS только для HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # CSP для API (более мягкие правила)
        csp = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp

        return response


class PreFlightHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware для обработки CORS preflight запросов.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Обработка preflight запросов.

        Args:
            request: HTTP запрос
            call_next: Следующий обработчик

        Returns:
            Response: HTTP ответ
        """
        # Обрабатываем OPTIONS запросы
        if request.method == "OPTIONS":
            return await self._handle_preflight(request)

        return await call_next(request)

    async def _handle_preflight(self, request: Request) -> Response:
        """
        Обработка CORS preflight запроса.

        Args:
            request: HTTP запрос

        Returns:
            Response: Ответ на preflight запрос
        """
        origin = request.headers.get("Origin")
        method = request.headers.get("Access-Control-Request-Method")
        headers = request.headers.get("Access-Control-Request-Headers")

        # Проверяем источник
        if not self._is_origin_allowed(origin):
            logger.warning(f"CORS preflight rejected for origin: {origin}")
            return JSONResponse(
                status_code=403, content={"error": "Origin not allowed"}
            )

        # Создаем ответ
        response = JSONResponse(
            status_code=200, content={"message": "CORS preflight successful"}
        )

        # Добавляем CORS заголовки
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = ", ".join(
            self.allowed_methods
        )
        response.headers["Access-Control-Allow-Headers"] = ", ".join(
            self.allowed_headers
        )
        response.headers["Access-Control-Max-Age"] = str(self.max_age)

        # Добавляем Vary header
        response.headers["Vary"] = "Origin"

        logger.debug(f"CORS preflight allowed for origin: {origin}")
        return response

    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """
        Проверка, разрешен ли источник.

        Args:
            origin: Источник запроса

        Returns:
            bool: True если источник разрешен
        """
        if not origin:
            return False

        # Проверяем точное совпадение
        if origin in self.allowed_origins:
            return True

        # Проверяем wildcard совпадения
        for allowed_origin in self.allowed_origins:
            if allowed_origin.endswith("*"):
                prefix = allowed_origin[:-1]  # Убираем *
                if origin.startswith(prefix):
                    return True

        # В debug режиме разрешаем localhost и 127.0.0.1
        if settings.DEBUG and (
            origin.startswith("http://localhost")
            or origin.startswith("http://127.0.0.1")
        ):
            return True

        return False


def setup_cors(app: FastAPI) -> None:
    """
    Настройка CORS для приложения.

    Args:
        app: FastAPI приложение
    """
    cors_config = CORSMiddleware()
    cors_config.add_cors_middleware(app)

    # Добавляем дополнительные security middleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(PreFlightHandlerMiddleware)

    logger.info("CORS configuration applied successfully")


# Утилиты для CORS


def is_cors_request(request: Request) -> bool:
    """
    Проверка, является ли запрос CORS запросом.

    Args:
        request: HTTP запрос

    Returns:
        bool: True если это CORS запрос
    """
    origin = request.headers.get("Origin")
    return origin is not None


def get_allowed_origins() -> List[str]:
    """
    Получение списка разрешенных источников.

    Returns:
        List[str]: Список разрешенных источников
    """
    return CORSMiddleware().allowed_origins


def add_allowed_origin(origin: str) -> bool:
    """
    Добавление нового разрешенного источника.

    Args:
        origin: Новый источник

    Returns:
        bool: True если источник добавлен
    """
    try:
        cors_config = CORSMiddleware()
        if origin not in cors_config.allowed_origins:
            cors_config.allowed_origins.append(origin)
            logger.info(f"Added new CORS origin: {origin}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to add CORS origin {origin}: {str(e)}")
        return False


def remove_allowed_origin(origin: str) -> bool:
    """
    Удаление разрешенного источника.

    Args:
        origin: Источник для удаления

    Returns:
        bool: True если источник удален
    """
    try:
        cors_config = CORSMiddleware()
        if origin in cors_config.allowed_origins:
            cors_config.allowed_origins.remove(origin)
            logger.info(f"Removed CORS origin: {origin}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to remove CORS origin {origin}: {str(e)}")
        return False


def validate_cors_config() -> Dict[str, Any]:
    """
    Валидация конфигурации CORS.

    Returns:
        Dict[str, Any]: Результат валидации
    """
    try:
        cors_config = CORSMiddleware()

        return {
            "valid": True,
            "allowed_origins": cors_config.allowed_origins,
            "allowed_methods": cors_config.allowed_methods,
            "allowed_headers": cors_config.allowed_headers,
            "max_age": cors_config.max_age,
            "debug_mode": settings.DEBUG,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
