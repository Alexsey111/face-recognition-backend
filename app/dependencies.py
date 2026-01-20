"""
Зависимости для FastAPI endpoints.
Аутентификация и авторизация через JWT.
"""

from typing import Dict, Any
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from app.config import settings
from app.utils.logger import get_logger, log_with_context

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)



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
