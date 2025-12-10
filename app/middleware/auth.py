"""
Middleware для аутентификации и авторизации.
JWT токены, проверка прав доступа и управление сессиями.
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timezone, timedelta
import hashlib

from ..config import settings
from ..utils.logger import get_logger
from ..services.cache_service import CacheService

logger = get_logger(__name__)

# Схема безопасности для Bearer токенов
security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """
    Middleware для обработки аутентификации.
    """
    
    def __init__(self):
        self.jwt_secret_key = settings.JWT_SECRET_KEY
        self.jwt_algorithm = settings.JWT_ALGORITHM
        self.jwt_access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.cache_service = CacheService()
    
    async def __call__(self, request: Request, call_next):
        """
        Обработка запроса с проверкой аутентификации.
        
        Args:
            request: HTTP запрос
            call_next: Следующий обработчик
            
        Returns:
            Response: HTTP ответ
        """
        try:
            # Проверяем, нужна ли аутентификация для этого endpoint
            if self._should_skip_auth(request):
                return await call_next(request)
            
            # Получаем токен из заголовка или cookies
            token = await self._extract_token(request)
            
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication token missing",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Валидируем токен
            payload = await self._validate_token(token)
            
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Добавляем информацию о пользователе в request state
            request.state.user = payload.get("user")
            request.state.user_id = payload.get("user_id")
            request.state.user_role = payload.get("role")
            request.state.token_payload = payload
            
            # Продлеваем токен если нужно
            if self._should_refresh_token(payload):
                new_token = await self._refresh_token(payload)
                request.state.new_token = new_token
            
            response = await call_next(request)
            
            # Добавляем новый токен в ответ если он был обновлен
            if hasattr(request.state, "new_token"):
                response.headers["X-New-Access-Token"] = request.state.new_token
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Auth middleware error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication system error"
            )
    
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
            "/health",
            "/status", 
            "/ready",
            "/live",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/ready",
            "/api/v1/live",
            "/api/v1/admin/test-webhook"
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
                token,
                self.jwt_secret_key,
                algorithms=[self.jwt_algorithm]
            )
            
            # Проверяем, что токен не отозван
            token_id = payload.get("jti")
            if token_id:
                is_revoked = await self.cache_service.get(f"revoked_token:{token_id}")
                if is_revoked:
                    logger.warning(f"Revoked token used: {token_id}")
                    return None
            
            # Проверяем срок действия
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
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
    
    def _should_refresh_token(self, payload: Dict[str, Any]) -> bool:
        """
        Проверка, нужно ли обновить токен.
        
        Args:
            payload: Payload токена
            
        Returns:
            bool: True если токен нужно обновить
        """
        exp = payload.get("exp")
        if not exp:
            return False
        
        # Если токен истекает меньше чем через 5 минут, обновляем
        exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
        refresh_threshold = datetime.now(timezone.utc) + timedelta(minutes=5)
        
        return exp_datetime <= refresh_threshold
    
    async def _refresh_token(self, payload: Dict[str, Any]) -> str:
        """
        Обновление токена.
        
        Args:
            payload: Старый payload
            
        Returns:
            str: Новый токен
        """
        user_id = payload.get("user_id")
        role = payload.get("role")
        
        # Создаем новый токен
        new_payload = {
            "user_id": user_id,
            "role": role,
            "type": "access",
            "jti": self._generate_jti()
        }
        
        # Устанавливаем новое время истечения
        expiration = datetime.now(timezone.utc) + timedelta(minutes=self.jwt_access_token_expire_minutes)
        new_payload["exp"] = expiration
        new_payload["iat"] = datetime.now(timezone.utc)
        
        # Создаем токен
        new_token = jwt.encode(
            new_payload,
            self.jwt_secret_key,
            algorithm=self.jwt_algorithm
        )
        
        logger.info(f"Token refreshed for user {user_id}")
        return new_token
    
    def _generate_jti(self) -> str:
        """
        Генерация уникального ID токена.
        
        Returns:
            str: UUID в виде строки
        """
        import uuid
        return str(uuid.uuid4())


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
                if hasattr(arg, 'state') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            user_role = getattr(request.state, 'user_role', None)
            
            if not user_role:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User role not found"
                )
            
            if user_role not in self.allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {self.allowed_roles}"
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
                if hasattr(arg, 'state') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found"
                )
            
            user_id = getattr(request.state, 'user_id', None)
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper


async def create_access_token(
    user_id: str,
    role: str = "user",
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Создание access токена.
    
    Args:
        user_id: ID пользователя
        role: Роль пользователя
        expires_delta: Время жизни токена
        
    Returns:
        str: JWT токен
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "user_id": user_id,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(hashlib.md5(f"{user_id}{expire}".encode()).hexdigest())
    }
    
    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return token


async def create_refresh_token(user_id: str) -> str:
    """
    Создание refresh токена.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        str: JWT refresh токен
    """
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "user_id": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(hashlib.md5(f"refresh{user_id}{expire}".encode()).hexdigest())
    }
    
    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return token


async def revoke_token(token: str) -> bool:
    """
    Отзыв токена.
    
    Args:
        token: Токен для отзыва
        
    Returns:
        bool: True если токен отозван
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        jti = payload.get("jti")
        if jti:
            cache_service = CacheService()
            # Сохраняем отозванный токен на 24 часа
            await cache_service.set(f"revoked_token:{jti}", True, expire_seconds=86400)
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        return False