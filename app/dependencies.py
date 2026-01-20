"""
Зависимости для FastAPI endpoints.
Аутентификация и авторизация через JWT.
"""

from typing import Dict, Any, AsyncGenerator
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.config import settings
from app.services.cache_service import CacheService
from app.utils.logger import get_logger, log_with_context

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)

# ==================== Cache Service Singleton ====================

_cache_service_instance: CacheService = None


async def get_cache_service() -> AsyncGenerator[CacheService, None]:
    """
    Dependency injection for CacheService
    Implements singleton pattern:
    - Creates ONE instance per application lifecycle
    - Reuses connection pool across all requests
    - Properly closes connection on shutdown

    Usage in endpoints:
        cache: CacheService = Depends(get_cache_service)
    """
    global _cache_service_instance

    # Create instance once (singleton)
    if _cache_service_instance is None:
        _cache_service_instance = CacheService()
        logger.info("CacheService instance created (singleton)")

    try:
        # Verify Redis connection is alive
        redis_client = await _cache_service_instance._get_redis()
        await redis_client.ping()
    except Exception as e:
        logger.error(f"Cache service error: {e}")
        # Don't raise exception - allow request to proceed without cache
        # (graceful degradation)

    yield _cache_service_instance


async def shutdown_cache_service():
    """
    Cleanup function to close Redis connections on app shutdown
    Call this in main.py lifespan or shutdown event:
        @app.on_event("shutdown")
        async def shutdown():
            await shutdown_cache_service()
    """
    global _cache_service_instance
    if _cache_service_instance:
        await _cache_service_instance.close()
        _cache_service_instance = None
        logger.info("CacheService closed")



# =========================
# Internal helpers
# =========================

def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def _decode_jwt(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token.

    Raises HTTPException on any validation error.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={
                "require": ["exp", "iat"],
                "verify_aud": False,
            },
        )
        return payload

    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired")

    except jwt.InvalidTokenError:
        # Никогда не прокидываем детали ошибки клиенту
        raise _unauthorized("Invalid authentication token")


# =========================
# Dependencies
# =========================

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Получение текущего пользователя.

    Приоритет:
    1. request.state.user (если установлен middleware)
    2. JWT из Authorization header
    """

    # 1. Данные из middleware (если уже аутентифицирован)
    user = getattr(request.state, "user", None)
    if isinstance(user, dict) and user.get("user_id"):
        return {
            "user_id": user["user_id"],
            "role": user.get("role", "user"),
        }

    # 2. JWT из заголовка
    if not credentials:
        raise _unauthorized("Authentication required")

    payload = _decode_jwt(credentials.credentials)

    user_id = payload.get("user_id")
    role = payload.get("role", "user")

    if not user_id:
        raise _unauthorized("Invalid authentication token")

    # Сохраняем в request.state для повторного использования
    request.state.user = {
        "user_id": user_id,
        "role": role,
    }

    log_with_context(
        logger.debug,
        "User authenticated",
        user_id=user_id,
        role=role,
        auth_source="jwt",
    )

    return {
        "user_id": user_id,
        "role": role,
    }


async def get_current_admin(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Проверка прав администратора.
    """

    role = user.get("role")
    if role not in {"admin", "superuser"}:
        log_with_context(
            logger.warning,
            "Forbidden admin access attempt",
            user_id=user.get("user_id"),
            role=role,
        )
        raise _forbidden("Admin privileges required")

    return user
