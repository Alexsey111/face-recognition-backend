"""
Сервис кэширования.
Работа с Redis для кэширования данных, сессий и токенов.
"""

import json
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import CacheError

logger = get_logger(__name__)


class CacheService:
    """
    Сервис для работы с Redis кэшем.
    """

    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.redis_password = settings.REDIS_PASSWORD
        self.socket_timeout = settings.REDIS_SOCKET_TIMEOUT
        self.connection_pool_size = settings.REDIS_CONNECTION_POOL_SIZE
        self._redis = None

    async def _get_redis_connection(self) -> redis.Redis:
        """
        Получение подключения к Redis.

        Returns:
            redis.Redis: Подключение к Redis
        """
        if self._redis is None or self._redis.closed:
            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    password=self.redis_password,
                    socket_timeout=self.socket_timeout,
                    max_connections=self.connection_pool_size,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                # Проверяем соединение
                await self._redis.ping()
            except (ConnectionError, RedisError) as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise CacheError(f"Redis connection failed: {str(e)}")

        return self._redis

    async def health_check(self) -> bool:
        """
        Проверка состояния Redis.

        Returns:
            bool: True если Redis доступен
        """
        try:
            redis_client = await self._get_redis_connection()
            await redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False

    async def set(
        self,
        key: str,
        value: Any,
        expire_seconds: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Установка значения в кэш.

        Args:
            key: Ключ для кэша
            value: Значение для сохранения
            expire_seconds: Время жизни в секундах
            nx: Установить только если ключ не существует
            xx: Установить только если ключ существует

        Returns:
            bool: True если значение установлено
        """
        try:
            redis_client = await self._get_redis_connection()

            # Сериализация значения
            serialized_value = self._serialize_value(value)

            # Установка с дополнительными параметрами
            if nx:
                result = await redis_client.setnx(key, serialized_value)
            elif xx:
                result = await redis_client.set(key, serialized_value, keepttl=True)
            else:
                if expire_seconds:
                    result = await redis_client.setex(
                        key, expire_seconds, serialized_value
                    )
                else:
                    result = await redis_client.set(key, serialized_value)

            logger.debug(f"Cache set: {key} (TTL: {expire_seconds})")
            return bool(result)

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            raise CacheError(f"Failed to set cache key {key}: {str(e)}")

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения из кэша.

        Args:
            key: Ключ для кэша
            default: Значение по умолчанию

        Returns:
            Any: Значение из кэша или default
        """
        try:
            redis_client = await self._get_redis_connection()
            value = await redis_client.get(key)

            if value is None:
                return default

            # Десериализация значения
            return self._deserialize_value(value)

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return default

    async def delete(self, key: str) -> bool:
        """
        Удаление ключа из кэша.

        Args:
            key: Ключ для удаления

        Returns:
            bool: True если ключ был удален
        """
        try:
            redis_client = await self._get_redis_connection()
            result = await redis_client.delete(key)
            return bool(result)

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Проверка существования ключа в кэше.

        Args:
            key: Ключ для проверки

        Returns:
            bool: True если ключ существует
        """
        try:
            redis_client = await self._get_redis_connection()
            result = await redis_client.exists(key)
            return bool(result)

        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {str(e)}")
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Установка времени жизни для ключа.

        Args:
            key: Ключ
            seconds: Время жизни в секундах

        Returns:
            bool: True если время жизни установлено
        """
        try:
            redis_client = await self._get_redis_connection()
            result = await redis_client.expire(key, seconds)
            return bool(result)

        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {str(e)}")
            return False

    async def ttl(self, key: str) -> int:
        """
        Получение времени жизни ключа.

        Args:
            key: Ключ

        Returns:
            int: Время жизни в секундах (-1 если без TTL, -2 если ключ не существует)
        """
        try:
            redis_client = await self._get_redis_connection()
            return await redis_client.ttl(key)

        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {str(e)}")
            return -2

    # Специализированные методы для работы с сессиями

    async def set_verification_session(
        self, session_id: str, session_data: Dict[str, Any], expire_seconds: int = 1800
    ) -> bool:
        """
        Сохранение сессии верификации.

        Args:
            session_id: ID сессии
            session_data: Данные сессии
            expire_seconds: Время жизни в секундах

        Returns:
            bool: True если сессия сохранена
        """
        key = f"verification_session:{session_id}"
        return await self.set(key, session_data, expire_seconds)

    async def get_verification_session(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Получение сессии верификации.

        Args:
            session_id: ID сессии

        Returns:
            Optional[Dict[str, Any]]: Данные сессии или None
        """
        key = f"verification_session:{session_id}"
        return await self.get(key)

    async def delete_verification_session(self, session_id: str) -> bool:
        """
        Удаление сессии верификации.

        Args:
            session_id: ID сессии

        Returns:
            bool: True если сессия удалена
        """
        key = f"verification_session:{session_id}"
        return await self.delete(key)

    # Методы для работы с токенами

    async def set_access_token(
        self, user_id: str, token: str, expire_seconds: int = 1800
    ) -> bool:
        """
        Сохранение access токена.

        Args:
            user_id: ID пользователя
            token: Access токен
            expire_seconds: Время жизни в секундах

        Returns:
            bool: True если токен сохранен
        """
        key = f"access_token:{user_id}:{token}"
        return await self.set(key, {"user_id": user_id}, expire_seconds)

    async def validate_access_token(self, user_id: str, token: str) -> bool:
        """
        Валидация access токена.

        Args:
            user_id: ID пользователя
            token: Access токен

        Returns:
            bool: True если токен валиден
        """
        key = f"access_token:{user_id}:{token}"
        return await self.exists(key)

    async def revoke_access_token(self, user_id: str, token: str) -> bool:
        """
        Отзыв access токена.

        Args:
            user_id: ID пользователя
            token: Access токен

        Returns:
            bool: True если токен отозван
        """
        key = f"access_token:{user_id}:{token}"
        return await self.delete(key)

    # Методы для работы с эмбеддингами

    async def cache_embedding(
        self, image_hash: str, embedding: bytes, expire_seconds: int = 3600
    ) -> bool:
        """
        Кэширование эмбеддинга лица.

        Args:
            image_hash: Хеш изображения
            embedding: Зашифрованный эмбеддинг
            expire_seconds: Время жизни в секундах

        Returns:
            bool: True если эмбеддинг кэширован
        """
        key = f"embedding:{image_hash}"
        return await self.set(key, embedding, expire_seconds)

    async def get_cached_embedding(self, image_hash: str) -> Optional[bytes]:
        """
        Получение кэшированного эмбеддинга.

        Args:
            image_hash: Хеш изображения

        Returns:
            Optional[bytes]: Эмбеддинг или None
        """
        key = f"embedding:{image_hash}"
        result = await self.get(key)
        return result if isinstance(result, bytes) else None

    # Методы для работы с rate limiting

    async def check_rate_limit(
        self, identifier: str, limit: int, window_seconds: int
    ) -> Dict[str, Any]:
        """
        Проверка rate limiting.

        Args:
            identifier: Идентификатор (IP, user_id, etc.)
            limit: Максимальное количество запросов
            window_seconds: Временное окно в секундах

        Returns:
            Dict[str, Any]: Информация о rate limit
        """
        try:
            redis_client = await self._get_redis_connection()
            key = f"rate_limit:{identifier}"
            current_time = datetime.now().timestamp()

            # Используем sliding window с Redis sorted sets
            window_start = current_time - window_seconds

            # Удаляем старые записи
            await redis_client.zremrangebyscore(key, 0, window_start)

            # Получаем текущее количество запросов
            current_requests = await redis_client.zcard(key)

            # Проверяем лимит
            if current_requests >= limit:
                # Получаем время последнего запроса
                last_request = await redis_client.zrevrange(key, 0, 0, withscores=True)
                reset_time = (
                    last_request[0][1] + window_seconds
                    if last_request
                    else current_time
                )

                return {
                    "allowed": False,
                    "current": current_requests,
                    "limit": limit,
                    "reset_time": reset_time,
                    "retry_after": int(reset_time - current_time),
                }

            # Добавляем текущий запрос
            await redis_client.zadd(key, {str(current_time): current_time})
            await redis_client.expire(key, window_seconds)

            return {
                "allowed": True,
                "current": current_requests + 1,
                "limit": limit,
                "reset_time": current_time + window_seconds,
            }

        except Exception as e:
            logger.error(f"Rate limit check error for {identifier}: {str(e)}")
            # В случае ошибки разрешаем запрос
            return {
                "allowed": True,
                "current": 0,
                "limit": limit,
                "reset_time": current_time + window_seconds,
                "error": str(e),
            }

    # Методы для работы с режимом обслуживания

    async def set_maintenance_mode(self, maintenance_data: Dict[str, Any]) -> bool:
        """
        Установка режима обслуживания.

        Args:
            maintenance_data: Данные о режиме обслуживания

        Returns:
            bool: True если режим установлен
        """
        key = "system:maintenance_mode"
        return await self.set(key, maintenance_data, expire_seconds=86400)  # 24 часа

    async def get_maintenance_mode(self) -> Optional[Dict[str, Any]]:
        """
        Получение информации о режиме обслуживания.

        Returns:
            Optional[Dict[str, Any]]: Данные о режиме обслуживания
        """
        key = "system:maintenance_mode"
        return await self.get(key)

    # Утилитные методы

    async def clear_pattern(self, pattern: str) -> int:
        """
        Очистка ключей по паттерну.

        Args:
            pattern: Паттерн ключей (например, "user:*")

        Returns:
            int: Количество удаленных ключей
        """
        try:
            redis_client = await self._get_redis_connection()
            keys = await redis_client.keys(pattern)
            if keys:
                return await redis_client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Clear pattern error for {pattern}: {str(e)}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики Redis.

        Returns:
            Dict[str, Any]: Статистика Redis
        """
        try:
            redis_client = await self._get_redis_connection()
            info = await redis_client.info()

            return {
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "expired_keys": info.get("expired_keys"),
                "evicted_keys": info.get("evicted_keys"),
            }

        except Exception as e:
            logger.error(f"Redis stats error: {str(e)}")
            return {}

    def _serialize_value(self, value: Any) -> bytes:
        """
        Сериализация значения для Redis.

        Args:
            value: Значение для сериализации

        Returns:
            bytes: Сериализованное значение
        """
        try:
            # Простые типы сериализуем как JSON
            if isinstance(value, (str, int, float, bool, list, dict)):
                return json.dumps(value).encode("utf-8")
            # Сложные объекты сериализуем как pickle
            else:
                return pickle.dumps(value)

        except Exception as e:
            logger.error(f"Serialization error: {str(e)}")
            # Fallback к pickle
            return pickle.dumps(value)

    def _deserialize_value(self, value: bytes) -> Any:
        """
        Десериализация значения из Redis.

        Args:
            value: Сериализованное значение

        Returns:
            Any: Десериализованное значение
        """
        try:
            # Пробуем JSON сначала
            try:
                return json.loads(value.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Если JSON не подходит, используем pickle
                return pickle.loads(value)

        except Exception as e:
            logger.error(f"Deserialization error: {str(e)}")
            return None

    async def close(self):
        """
        Закрытие соединения с Redis.
        """
        if self._redis and not self._redis.closed:
            await self._redis.close()
            self._redis = None
