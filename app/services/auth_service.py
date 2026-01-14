"""
Сервис аутентификации и авторизации.
JWT токены, refresh tokens, управление сессиями и ролями.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4
import jwt
import hashlib
import secrets
from collections import defaultdict
from passlib.context import CryptContext
from prometheus_client import Histogram

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import (
    UnauthorizedError, 
    ForbiddenError, 
    ValidationError,
    AuthenticationError
)
from ..services.encryption_service import EncryptionService
from ..services.database_service import BiometricService

# Redis integration for token revocation
try:
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Prometheus metrics
auth_service_duration = Histogram(
    'auth_service_duration_seconds',
    'Time spent in AuthService methods'
)

logger = get_logger(__name__)

class AuthService:
    """
    Authentication service with hybrid sync/async approach.

    Sync methods (fast CPU operations):
    - create_access_token()      # JWT encoding (<1ms)
    - create_refresh_token()     # JWT encoding (<1ms)
    - validate_user_permissions() # Memory operations
    - generate_secure_token()    # Fast random generation
    - get_token_info()           # JWT decode without verification
    - needs_password_rehash()    # Simple string check

    Async methods (I/O or heavy CPU):
    - verify_token()             # Needs Redis I/O for revocation check
    - check_rate_limit()         # Redis I/O
    - hash_password()            # Heavy CPU (pbkdf2 ~100ms)
    - verify_password()          # Heavy CPU (pbkdf2)
    - refresh_access_token()     # Calls async methods
    - create_user_session()      # Calls async methods
    - revoke_token()             # Redis I/O
    - is_token_revoked()         # Redis I/O
    """

    # ✅ Shared resources (class-level)
    _redis_pool = None
    _pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

    def __init__(self, db: AsyncSession = None):
        # ✅ Per-request resources
        self.db = db
        self.db_service = BiometricService(db) if db else None

        # ✅ Use shared pwd_context
        self.pwd_context = AuthService._pwd_context
        
        # JWT settings
        self.jwt_secret_key = settings.JWT_SECRET_KEY
        self.jwt_algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        self.encryption_service = EncryptionService()
        
        # ✅ Shared Redis connection
        self.redis = AuthService._redis_pool

        # Rate limiting for login attempts
        self.login_attempts = defaultdict(list)
        
        # Rate limit failure policy
        self.rate_limit_failure_policy = settings.rate_limit_on_redis_failure.lower()
        
        policy_messages = {
            "block": "BLOCK — all logins will be blocked if Redis is unavailable (highest security)",
            "allow": "ALLOW — logins permitted if Redis is unavailable (highest availability)",
            "error": "ERROR — return 503 Service Unavailable if Redis is unavailable",
        }
        
        if self.rate_limit_failure_policy not in policy_messages:
            logger.warning(f"Invalid rate_limit_on_redis_failure value: {settings.rate_limit_on_redis_failure}...")
            self.rate_limit_failure_policy = "block"
        
        message = policy_messages[self.rate_limit_failure_policy]
        log_level = logger.warning if self.rate_limit_failure_policy == "allow" else logger.info
        log_level(f"Rate limit on Redis failure policy: {message}")
        
        # In-memory fallback storage for revoked tokens
        self._initialize_memory_storage()

    @classmethod
    def init_redis(cls):
        """Initialize shared Redis connection."""
        if REDIS_AVAILABLE and cls._redis_pool is None:
            try:
                cls._redis_pool = aioredis.from_url(settings.REDIS_URL)
                logger.info("Redis connection pool initialized")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")

    @classmethod
    async def close_redis(cls):
        """Close shared Redis connection."""
        if cls._redis_pool:
            await cls._redis_pool.close()
            cls._redis_pool = None
            logger.info("Redis connection closed")

    def create_access_token(
        self, 
        user_id: str, 
        role: str = "user",
        permissions: List[str] = None,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """
        Создание access токена.

        Args:
            user_id: ID пользователя
            role: Роль пользователя
            permissions: Список разрешений
            additional_claims: Дополнительные claims

        Returns:
            str: JWT access токен
        """
        try:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.access_token_expire_minutes
            )
            
            # Уникальный идентификатор токена
            jti = str(uuid4())
            
            payload = {
                "user_id": user_id,
                "role": role,
                "type": "access",
                "permissions": permissions or [],
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "jti": jti,
            }
            
            # Добавляем дополнительные claims
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(
                payload, 
                self.jwt_secret_key, 
                algorithm=self.jwt_algorithm
            )
            
            logger.info(f"Access token created for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error creating access token: {str(e)}")
            raise AuthenticationError(f"Failed to create access token: {str(e)}")

    def create_refresh_token(self, user_id: str) -> str:
        """
        Создание refresh токена.

        Args:
            user_id: ID пользователя

        Returns:
            str: JWT refresh токен
        """
        try:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.refresh_token_expire_days
            )
            
            jti = str(uuid4())
            
            payload = {
                "user_id": user_id,
                "type": "refresh",
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "jti": jti,
            }
            
            token = jwt.encode(
                payload,
                self.jwt_secret_key,
                algorithm=self.jwt_algorithm
            )
            
            logger.info(f"Refresh token created for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error creating refresh token: {str(e)}")
            raise AuthenticationError(f"Failed to create refresh token: {str(e)}")

    async def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Верификация токена (async: needs Redis I/O for revocation check).

        Args:
            token: JWT токен
            token_type: Тип токена (access или refresh)

        Returns:
            Dict[str, Any]: Payload токена

        Raises:
            UnauthorizedError: Если токен невалиден
        """
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret_key, 
                algorithms=[self.jwt_algorithm]
            )
            
            # Проверяем тип токена
            if payload.get("type") != token_type:
                raise UnauthorizedError(f"Invalid token type. Expected {token_type}")
            
            # Проверяем срок действия
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
                raise UnauthorizedError("Token has expired")
            
            # Проверяем, не отозван ли токен (Redis I/O)
            jti = payload.get("jti")
            if jti and await self.is_token_revoked(jti):
                raise UnauthorizedError("Token has been revoked")

            logger.debug(f"Token verified successfully for user {payload.get('user_id')}")
            return payload

        except jwt.ExpiredSignatureError:
            raise UnauthorizedError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise UnauthorizedError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise UnauthorizedError(f"Token verification failed: {str(e)}")

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Обновление access токена с помощью refresh токена (с rotation).
        
        Возвращает НОВЫЕ access и refresh токены, а старый refresh токен отзывает.
        Это критически важно для безопасности.

        Args:
            refresh_token: JWT refresh токен

        Returns:
            Dict[str, str]: Новые access и refresh токены

        Raises:
            UnauthorizedError: Если refresh токен невалиден
        """
        try:
            # Верифицируем refresh токен (async: needs Redis check)
            payload = await self.verify_token(refresh_token, "refresh")
            
            user_id = payload.get("user_id")
            if not user_id:
                raise UnauthorizedError("Invalid refresh token payload")
            
            # Создаём новые токены (sync: fast CPU)
            new_access_token = self.create_access_token(user_id)
            new_refresh_token = self.create_refresh_token(user_id)

            # Отзываем старый refresh токен (async: Redis I/O)
            await self.revoke_token(refresh_token)
            
            logger.info(f"Token rotation completed for user {user_id}")
            
            return {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }
            
        except Exception as e:
            logger.error(f"Error refreshing access token: {str(e)}")
            raise

    async def hash_password(self, password: str) -> str:
        """
        Хеширование пароля с использованием pbkdf2_sha256 через passlib.
        Тяжёлая операция выполняется в отдельном потоке, чтобы не блокировать event loop.
        """
        try:
            # Переносим тяжёлое хэширование в отдельный поток
            hashed_password = await asyncio.to_thread(self.pwd_context.hash, password)
            
            logger.debug("Password hashed successfully using pbkdf2_sha256")
            return hashed_password
        
        except Exception as e:
            logger.error(f"Error hashing password with pbkdf2_sha256: {str(e)}")
            raise AuthenticationError(f"Failed to hash password: {str(e)}")

    @auth_service_duration.time()
    async def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Проверка пароля против хеша с использованием passlib.
        Тяжёлые операции выполняются в отдельном потоке.
        """
        start = time.time()
        try:
            # Основная проверка pbkdf2_sha256 — в отдельном потоке
            try:
                is_valid = await asyncio.to_thread(self.pwd_context.verify, password, hashed_password)
                if is_valid:
                    logger.debug(f"Password verified successfully using pbkdf2_sha256 (took {time.time() - start:.3f}s)")
                    return True
            except Exception:
                pass  # Если не удалось — пробуем legacy

            # Legacy PBKDF2 — тоже в отдельном потоке (редко, но на всякий случай)
            is_legacy_valid = await asyncio.to_thread(self._verify_legacy_pbkdf2, password, hashed_password)
            if is_legacy_valid:
                logger.debug(f"Password verified using legacy PBKDF2 hash (took {time.time() - start:.3f}s)")
                return True

            logger.debug("Password verification failed")
            return False

        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False

    def _verify_legacy_pbkdf2(self, password: str, hashed_password: str) -> bool:
        """
        Проверка пароля против старого PBKDF2 хеша для обратной совместимости.

        Args:
            password: Пароль для проверки
            hashed_password: Старый PBKDF2 хеш

        Returns:
            bool: True если пароль корректен
        """
        try:
            # Декодируем хеш из hex
            combined = bytes.fromhex(hashed_password)
            
            # Извлекаем соль и хеш
            salt = combined[:32]
            stored_hash = combined[32:]
            
            # Хешируем введенный пароль с той же солью
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            
            # Сравниваем хеши
            return password_hash == stored_hash
            
        except Exception as e:
            logger.debug(f"Legacy PBKDF2 verification failed: {str(e)}")
            return False

    def needs_password_rehash(self, hashed_password: str) -> bool:
        """
        Проверяет, нужно ли перехешировать пароль с новым алгоритмом.

        Args:
            hashed_password: Хешированный пароль для проверки

        Returns:
            bool: True если пароль нужно перехешировать
        """
        try:
            # Проверяем, используется ли устаревший алгоритм
            if not self.pwd_context.identify(hashed_password):
                # Если passlib не может определить алгоритм, это может быть старый PBKDF2
                return len(hashed_password) == 128  # Старый PBKDF2 hex формат
            else:
                # Если алгоритм определен, проверяем, не устарел ли он
                return self.pwd_context.needs_update(hashed_password)
                
        except Exception as e:
            logger.debug(f"Error checking if password needs rehash: {str(e)}")
            return False

    # 🟢 Добавь миграцию старых паролей
    async def migrate_password_if_needed(self, user_id: str, password: str, hashed: str, db_service: BiometricService = None):
        """
        Re-hash password with new algorithm after successful login
        
        Args:
            user_id: ID пользователя
            password: Пароль в открытом виде
            hashed: Текущий хеш пароля
            db_service: Database service instance (required)
        """
        if db_service is None:
            logger.warning("No db_service provided for password migration")
            return
            
        if await self.needs_password_rehash(hashed):
            new_hash = await self.hash_password(password)
            await db_service.update_user(user_id, {"password_hash": new_hash})
            logger.info(f"Password rehashed for user {user_id}")

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Генерация криптографически безопасного токена.

        Args:
            length: Длина токена в байтах

        Returns:
            str: Токен в hex формате
        """
        try:
            token_bytes = secrets.token_bytes(length)
            return token_bytes.hex()
            
        except Exception as e:
            logger.error(f"Error generating secure token: {str(e)}")
            raise AuthenticationError(f"Failed to generate secure token: {str(e)}")

    def create_user_session(
        self, 
        user_id: str, 
        user_agent: str = None, 
        ip_address: str = None,
        device_fingerprint: str = None
    ) -> Dict[str, str]:
        """
        Создание пользовательской сессии с поддержкой device tracking (sync: fast CPU).

        Args:
            user_id: ID пользователя
            user_agent: User agent браузера
            ip_address: IP адрес пользователя
            device_fingerprint: Отпечаток устройства (browser fingerprint)

        Returns:
            Dict[str, str]: Токены сессии (access и refresh)
        """
        try:
            # Подготавливаем device info для токенов
            device_id = None
            if device_fingerprint:
                device_id = hashlib.sha256(device_fingerprint.encode()).hexdigest()
            
            # Создаем access и refresh токены с device info (sync: fast CPU)
            access_token = self.create_access_token(
                user_id, 
                additional_claims={"device_id": device_id} if device_id else None
            )
            refresh_token = self.create_refresh_token(user_id)

            session_data = {
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "user_agent": user_agent,
                "ip_address": ip_address,
                "device_id": device_id,
                "access_token": access_token,
                "refresh_token": refresh_token
            }
            
            # В production здесь бы сохраняли сессию в Redis или БД
            logger.info(f"User session created for user {user_id}")
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }
            
        except Exception as e:
            logger.error(f"Error creating user session: {str(e)}")
            raise

    async def check_rate_limit(self, user_id: str) -> bool:
        """
        Проверка rate limit для попыток входа — только через Redis.
        Redis обязателен для работы rate limiting.
        """
        try:
            key = f"login_attempts:{user_id}"
            attempts = await self.redis.get(key)
            attempts_count = int(attempts) if attempts is not None else 0
            
            if attempts_count >= 5:
                logger.warning(f"Rate limit exceeded for user {user_id} (attempts: {attempts_count})")
                raise UnauthorizedError("Too many login attempts. Try again later.")
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking rate limit in Redis: {str(e)}")
            # Здесь твой выбор: блокировка или ошибка
            raise UnauthorizedError("Login temporarily unavailable due to technical issues. Try again later.")

    async def record_failed_login(self, user_id: str) -> None:
        """
        Записывает неудачную попытку входа в Redis.
        """
        try:
            key = f"login_attempts:{user_id}"
            if self.redis:
                # Увеличиваем счётчик на 1
                attempts = await self.redis.incr(key)
                # Если это первая попытка — устанавливаем TTL 15 минут
                if attempts == 1:
                    await self.redis.expire(key, 900)  # 15 * 60 = 900 секунд
                logger.debug(f"Failed login attempt recorded for user {user_id} (attempts: {attempts})")
            else:
                # Fallback на in-memory, если Redis недоступен
                self.login_attempts[user_id].append(datetime.now(timezone.utc))
                logger.debug(f"Failed login attempt recorded in memory for user {user_id}")
        except Exception as e:
            logger.error(f"Error recording failed login: {str(e)}")
            # Даже если Redis упал — fallback на память
            self.login_attempts[user_id].append(datetime.now(timezone.utc))

    async def reset_login_attempts(self, user_id: str) -> None:
        """
        Сбрасывает счётчик попыток входа при успешной аутентификации.
        Очищает как in-memory, так и Redis.
        """
        try:
            key = f"login_attempts:{user_id}"
            if self.redis:
                await self.redis.delete(key)
                logger.debug(f"Login attempts reset in Redis for user {user_id}")
            # Очищаем in-memory на всякий случай
            if user_id in self.login_attempts:
                self.login_attempts[user_id].clear()
                logger.debug(f"Login attempts reset in memory for user {user_id}")
        except Exception as e:
            logger.error(f"Error resetting login attempts: {str(e)}")
            # Fallback: очищаем хотя бы in-memory
            if user_id in self.login_attempts:
                self.login_attempts[user_id].clear()

    def validate_user_permissions(
        self, 
        user_role: str, 
        required_permissions: List[str],
        user_permissions: List[str] = None
    ) -> bool:
        """
        Валидация разрешений пользователя.

        Args:
            user_role: Роль пользователя
            required_permissions: Требуемые разрешения
            user_permissions: Разрешения пользователя

        Returns:
            bool: True если у пользователя есть все требуемые разрешения

        Raises:
            ForbiddenError: Если разрешений недостаточно
        """
        # Роли с полными правами
        admin_roles = ["admin", "superuser"]
        
        if user_role in admin_roles:
            return True
        
        # Проверяем конкретные разрешения
        if user_permissions:
            missing_permissions = set(required_permissions) - set(user_permissions)
            if missing_permissions:
                raise ForbiddenError(
                    f"Insufficient permissions. Missing: {', '.join(missing_permissions)}"
                )
            return True
        
        # Если нет конкретных разрешений, проверяем роль
        role_permissions = {
            "user": ["read_own_data"],
            "premium": ["read_own_data", "advanced_features"],
            "admin": ["read_own_data", "manage_users", "system_admin"]
        }
        
        available_permissions = role_permissions.get(user_role, [])
        missing_permissions = set(required_permissions) - set(available_permissions)
        
        if missing_permissions:
            raise ForbiddenError(
                f"Insufficient permissions for role '{user_role}'. "
                f"Missing: {', '.join(missing_permissions)}"
            )
            
        return True

    async def get_user_info_from_token(self, token: str) -> Dict[str, Any]:
        """
        Извлечение информации о пользователе из токена (async: needs Redis check).

        Args:
            token: JWT токен

        Returns:
            Dict[str, Any]: Информация о пользователе
        """
        try:
            payload = await self.verify_token(token)
            
            return {
                "user_id": payload.get("user_id"),
                "role": payload.get("role"),
                "permissions": payload.get("permissions", []),
                "token_type": payload.get("type"),
                "issued_at": payload.get("iat"),
                "expires_at": payload.get("exp"),
                "jti": payload.get("jti")
            }
            
        except Exception as e:
            logger.error(f"Error extracting user info from token: {str(e)}")
            raise

    async def revoke_token(self, token: str) -> bool:
        """
        Отзыв токена с сохранением в Redis.
        
        Args:
            token: Токен для отзыва

        Returns:
            bool: True если токен отозван успешно
        """
        try:
            payload = await self.verify_token(token)
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if jti and exp:
                # Сохраняем в Redis с TTL до истечения токена
                if self.redis:
                    try:
                        ttl = int(exp - datetime.now(timezone.utc).timestamp())
                        if ttl > 0:
                            await self.redis.setex(f"revoked:{jti}", ttl, "1")
                            logger.info(f"Token revoked and stored in Redis: {jti}")
                        else:
                            logger.warning(f"Token already expired: {jti}")
                    except Exception as e:
                        logger.error(f"Redis error during token revocation: {e}")
                        # Fallback to in-memory storage if Redis fails
                        self._revoked_tokens_memory[jti] = exp
                        logger.info(f"Token revoked in memory: {jti}")
                else:
                    # Fallback to in-memory storage if Redis not available
                    self._revoked_tokens_memory[jti] = exp
                    logger.info(f"Token revoked in memory: {jti}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking token: {str(e)}")
            return False

    async def is_token_revoked(self, jti: str) -> bool:
        """
        Проверяет, был ли токен отозван.
        
        Args:
            jti: JWT ID токена
            
        Returns:
            bool: True если токен отозван
        """
        try:
            if self.redis:
                # Проверяем в Redis
                return await self.redis.exists(f"revoked:{jti}")
            else:
                # Fallback to in-memory storage
                return jti in self._revoked_tokens_memory
        except Exception as e:
            logger.error(f"Error checking token revocation status: {str(e)}")
            return False

    def _initialize_memory_storage(self) -> None:
        """Инициализация in-memory хранилища для отозванных токенов (fallback)."""
        if not hasattr(self, '_revoked_tokens_memory'):
            self._revoked_tokens_memory = {}

    def get_token_info(self, token: str) -> Dict[str, Any]:
        """
        Получение информации о токене без верификации.

        Args:
            token: JWT токен

        Returns:
            Dict[str, Any]: Информация о токене
        """
        try:
            # Декодируем без проверки подписи для получения информации
            payload = jwt.decode(
                token, 
                options={"verify_signature": False}
            )
            
            return {
                "type": payload.get("type"),
                "user_id": payload.get("user_id"),
                "role": payload.get("role"),
                "issued_at": payload.get("iat"),
                "expires_at": payload.get("exp"),
                "is_expired": datetime.fromtimestamp(
                    payload.get("exp", 0), tz=timezone.utc
                ) < datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return {"error": str(e)}


class TokenExpiredError(AuthenticationError):
    """Исключение для истекших токенов."""
    pass


class InvalidTokenError(AuthenticationError):
    """Исключение для невалидных токенов."""
    pass
