"""
Сервис кэширования.
Redis cache для данных, сессий, токенов, rate limit и embedding.
Production-safe версия.
"""

import json
from typing import Any, Optional, Dict
from datetime import datetime
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError
try:
    from prometheus_client import Counter
except Exception:
    class _NoopCounter:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

    Counter = _NoopCounter

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import CacheError

logger = get_logger(__name__)


# Reuse application-level cache metrics to avoid duplicate registration
try:
    from ..middleware.metrics import cache_hits_total as CACHE_HITS, cache_misses_total as CACHE_MISSES
except Exception:
    # Fallback to local Counter definitions if metrics module isn't importable
    CACHE_HITS = Counter("cache_hits_total", "Total cache hits", ["cache_name"]) 
    CACHE_MISSES = Counter("cache_misses_total", "Total cache misses", ["cache_name"]) 


RATE_LIMIT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

redis.call("ZREMRANGEBYSCORE", key, 0, now - window)
local count = redis.call("ZCARD", key)

if count >= limit then
    local last = redis.call("ZREVRANGE", key, 0, 0, "WITHSCORES")
    return {0, count, last[2]}
end

redis.call("ZADD", key, now, now)
redis.call("EXPIRE", key, window)
return {1, count + 1, now + window}
"""


class CacheService:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        if self._redis is not None:
            try:
                await self._redis.ping()
                return self._redis
            except (ConnectionError, RedisError, TimeoutError) as e:
                logger.warning(f"Redis connection lost: {e}. Attempting to reconnect...")
                await self._redis.close()  # Закрываем старый клиент
                self._redis = None  # Сбрасываем, чтобы пересоздать

        # Если клиента нет или ping упал — создаём новый
        try:
            self._redis = redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                max_connections=settings.REDIS_CONNECTION_POOL_SIZE,
                retry_on_timeout=True,
                health_check_interval=30,  # Автоматическая проверка каждые 30 сек
            )
            await self._redis.ping()  # Проверяем сразу
            logger.info("Redis connection established (or reconnected)")
            return self._redis
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis connection failed: {e}")
            raise CacheError(f"Redis connection failed: {e}")

    # -------------------- Base ops --------------------

    async def set(
        self,
        key: str,
        value: Any,
        expire_seconds: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        redis_client = await self._get_redis()
        data = self._serialize(value)

        result = await redis_client.set(
            key,
            data,
            ex=expire_seconds,
            nx=nx,
            xx=xx,
        )
        return bool(result)

    async def get(self, key: str, default: Any = None) -> Any:
        redis_client = await self._get_redis()
        raw = await redis_client.get(key)
        if raw is None:
            try:
                CACHE_MISSES.labels(cache_name="redis").inc()
            except Exception:
                pass
            return default
        try:
            CACHE_HITS.labels(cache_name="redis").inc()
        except Exception:
            pass
        return self._deserialize(raw)

    async def delete(self, key: str) -> bool:
        redis_client = await self._get_redis()
        return bool(await redis_client.delete(key))

    async def exists(self, key: str) -> bool:
        redis_client = await self._get_redis()
        return bool(await redis_client.exists(key))

    async def expire(self, key: str, seconds: int) -> bool:
        redis_client = await self._get_redis()
        return bool(await redis_client.expire(key, seconds))

    async def ttl(self, key: str) -> int:
        redis_client = await self._get_redis()
        return await redis_client.ttl(key)

    # -------------------- Sessions --------------------

    async def set_verification_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        expire_seconds: int = 1800,
    ) -> bool:
        return await self.set(
        f"{settings.CACHE_KEY_PREFIX}verification_session:{session_id}",
        data,
        expire_seconds,
    )
        

    async def get_verification_session(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await self.get(
    f"{settings.CACHE_KEY_PREFIX}verification_session:{session_id}")

    async def delete_verification_session(self, session_id: str) -> bool:
        return await self.delete(
    f"{settings.CACHE_KEY_PREFIX}verification_session:{session_id}")

    # -------------------- Tokens --------------------

    async def store_access_token(
        self,
        user_id: str,
        token: str,
        expire_seconds: int = 1800,
    ) -> bool:
        redis_client = await self._get_redis()

        pipe = redis_client.pipeline()
        pipe.set(f"{settings.CACHE_KEY_PREFIX}access_token:{token}", user_id, ex=expire_seconds)
        pipe.sadd(f"{settings.CACHE_KEY_PREFIX}user_tokens:{user_id}", token)
        pipe.expire(f"{settings.CACHE_KEY_PREFIX}user_tokens:{user_id}", expire_seconds)
        await pipe.execute()

        return True

    async def validate_access_token(self, token: str) -> Optional[str]:
        redis_client = await self._get_redis()
        return await redis_client.get(f"{settings.CACHE_KEY_PREFIX}access_token:{token}")

    async def revoke_access_token(self, user_id: str, token: str) -> bool:
        redis_client = await self._get_redis()

        pipe = redis_client.pipeline()
        pipe.delete(f"{settings.CACHE_KEY_PREFIX}access_token:{token}")
        pipe.srem(f"{settings.CACHE_KEY_PREFIX}user_tokens:{user_id}", token)
        await pipe.execute()

        return True

    async def revoke_all_user_tokens(self, user_id: str) -> int:
        redis_client = await self._get_redis()
        tokens = await redis_client.smembers(f"{settings.CACHE_KEY_PREFIX}user_tokens:{user_id}")

        if not tokens:
            return 0

        pipe = redis_client.pipeline()
        for token in tokens:
            pipe.delete(f"{settings.CACHE_KEY_PREFIX}access_token:{token}")
        pipe.delete(f"{settings.CACHE_KEY_PREFIX}user_tokens:{user_id}")
        await pipe.execute()

        return len(tokens)

    # -------------------- Embeddings --------------------

    async def cache_embedding(
        self,
        image_hash: str,
        embedding: bytes,
        expire_seconds: int = 3600,
    ) -> bool:
        redis_client = await self._get_redis()
        return bool(
            await redis_client.set(
                f"{settings.CACHE_KEY_PREFIX}embedding:{image_hash}",
                embedding,
                ex=expire_seconds,
            )
        )

    async def get_cached_embedding(self, image_hash: str) -> Optional[bytes]:
        redis_client = await self._get_redis()
        return await redis_client.get(f"{settings.CACHE_KEY_PREFIX}embedding:{image_hash}")

    # -------------------- Rate limit (Lua) --------------------

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> Dict[str, Any]:
        redis_client = await self._get_redis()
        now = int(datetime.utcnow().timestamp())

        allowed, current, reset_time = await redis_client.eval(
            RATE_LIMIT_LUA,
            1,
            f"{settings.CACHE_KEY_PREFIX}rate_limit:{identifier}",
            limit,
            window_seconds,
            now,
        )

        return {
            "allowed": bool(allowed),
            "current": current,
            "limit": limit,
            "reset_time": reset_time,
            "retry_after": max(0, reset_time - now) if not allowed else 0,
        }

    # -------------------- Maintenance --------------------

    async def set_maintenance_mode(self, data: Dict[str, Any]) -> bool:
        return await self.set(
            f"{settings.CACHE_KEY_PREFIX}system:maintenance",
            data,
            expire_seconds=86400,
        )

    async def get_maintenance_mode(self) -> Optional[Dict[str, Any]]:
        return await self.get(f"{settings.CACHE_KEY_PREFIX}system:maintenance")

    # -------------------- Utils --------------------

    async def clear_pattern(self, pattern: str) -> int:
        redis_client = await self._get_redis()
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = await redis_client.scan(
                cursor=cursor,
                match=f"{settings.CACHE_KEY_PREFIX}{pattern}",
                count=500,
            )
            if keys:
                deleted += await redis_client.delete(*keys)
            if cursor == 0:
                break

        return deleted

    async def get_stats(self) -> Dict[str, Any]:
        redis_client = await self._get_redis()
        info = await redis_client.info()
        return {
            "clients": info.get("connected_clients"),
            "used_memory": info.get("used_memory_human"),
            "hits": info.get("keyspace_hits"),
            "misses": info.get("keyspace_misses"),
            "evicted": info.get("evicted_keys"),
        }

    async def close(self):
        if self._redis:
            await self._redis.close()
            self._redis = None

    # -------------------- Serialization --------------------

    def _serialize(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        return json.dumps(value, default=str).encode("utf-8")

    def _deserialize(self, raw: bytes) -> Any:
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return raw
