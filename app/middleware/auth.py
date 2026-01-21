"""
Middleware для аутентификации и авторизации.
JWT токены, проверка прав доступа и управление сессиями.
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timezone, timedelta
import hashlib

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Схема безопасности для Bearer токенов
security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware для обработки аутентификации и авторизации.
    Обрабатывает JWT токены и добавляет информацию о пользователе в request state.
    """

    def __init__(self, app):
        super().__init__(app)
        self.jwt_secret_key = settings.JWT_SECRET_KEY
        self.jwt_algorithm = settings.JWT_ALGORITHM
        self.jwt_access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES

    async def dispatch(self, request: Request, call_next):
        """Обработка запроса с проверкой аутентификации."""
        # Проверяем, нужна ли аутентификация
        if self._should_skip_auth(request):
            return await call_next(request)

        # Получаем токен
        token = await self._extract_token(request)
        if not token:
            return JSONResponse(
                status_code=401, content={"error": "Authentication required"}
            )

        # Валидируем токен
        payload = await self._validate_token(token)
        if not payload:
            return JSONResponse(status_code=401, content={"error": "Invalid token"})

        # Добавляем данные в request.state
        request.state.user_id = payload.get("user_id")
        request.state.user_role = payload.get("role")

        response = await call_next(request)
        return response

    def _should_skip_auth(self, request: Request) -> bool:
        """
        Проверка, нужно ли пропустить аутентификацию для endpoint.

        Args:
            request: HTTP запрос

        Returns:
            bool: True если аутентификация не нужна
        """
        # Список endpoint-ов, которые не требуют аутентификации
        public_paths = [
            "/",  # Root endpoint
            "/health",
            "/status",
            "/ready",
            "/live",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",  # Favicon для браузеров
            # ✅ API v1 health endpoints
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/ready",
            "/api/v1/live",
            "/api/v1/metrics",
            "/api/v1/ml/details",
            "/api/v1/system/metrics",
            # ✅ Auth endpoints - только с префиксом /api/v1 (после исправления в main.py)
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/api/v1/auth/verify",
            "/api/v1/auth/change-password",
            # ✅ Reference endpoints (только GET для публичного доступа)
            "/api/v1/reference/",
        ]

        path = request.url.path

        # Проверяем точные совпадения и префиксы
        for public_path in public_paths:
            if path == public_path or path.startswith(public_path + "/"):
                return True

        # Проверяем методы, которые могут не требовать аутентификации
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            # GET запросы к публичным ресурсам
            if path.startswith("/api/v1/reference/") and request.method == "GET":
                return True
            # GET запросы к health endpoints без префикса /api/v1
            if path.startswith("/") and path.split("/")[1] in [
                "health",
                "status",
                "ready",
                "live",
                "metrics",
            ]:
                return True

        return False

    async def _extract_token(self, request: Request) -> Optional[str]:
        """
        Извлечение токена из запроса.

        Args:
            request: HTTP запрос

        Returns:
            Optional[str]: Токен или None
        """
        # Проверяем Authorization заголовок
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ", 1)[1]

        # Проверяем cookies
        token_cookie = request.cookies.get("access_token")
        if token_cookie:
            return token_cookie

        return None

    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Валидация JWT токена.

        Args:
            token: JWT токен

        Returns:
            Optional[Dict[str, Any]]: Payload токена или None
        """
        try:
            # Декодируем токен
            payload = jwt.decode(
                token, self.jwt_secret_key, algorithms=[self.jwt_algorithm]
            )

            # Проверяем срок действия
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(
                timezone.utc
            ):
                logger.warning("Expired token used")
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Expired token")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            return None


class RequireRole:
    """
    Декоратор для проверки роли пользователя.
    """

    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Получаем request из аргументов
            request = None
            for arg in args:
                if hasattr(arg, "state") and hasattr(arg, "url"):
                    request = arg
                    break

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found",
                )

            user_role = getattr(request.state, "user_role", None)

            if not user_role:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User role not found",
                )

            if user_role not in self.allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {self.allowed_roles}",
                )

            return await func(*args, **kwargs)

        return wrapper


class RequireAuth:
    """
    Декоратор для проверки аутентификации.
    """

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Получаем request из аргументов
            request = None
            for arg in args:
                if hasattr(arg, "state") and hasattr(arg, "url"):
                    request = arg
                    break

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found",
                )

            user_id = getattr(request.state, "user_id", None)

            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            return await func(*args, **kwargs)

        return wrapper
