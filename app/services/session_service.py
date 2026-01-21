from __future__ import annotations

import json
import uuid
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from ..config import settings
from ..utils.logger import get_logger
from .cache_service import CacheService

logger = get_logger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def utcnow() -> datetime:
    """Единая точка получения UTC-времени"""
    return datetime.now(timezone.utc)


# =============================================================================
# Domain Model
# =============================================================================
@dataclass
class UploadSession:
    session_id: str
    user_id: str
    created_at: datetime
    expiration_at: datetime
    file_key: Optional[str] = None
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<UploadSession id={self.session_id} user={self.user_id}>"

    def is_expired(self) -> bool:
        """Проверка истечения сессии"""
        return utcnow() > self.expiration_at

    @classmethod
    def from_redis_data(cls, session_id: str, data: Dict[str, str]) -> "UploadSession":
        # ✅ Безопасная загрузка metadata с дефолтным значением
        metadata_str = data.get("metadata", "{}")
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Invalid metadata JSON for session {session_id}")
            metadata = {}
        return cls(
            session_id=session_id,
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expiration_at=datetime.fromisoformat(data["expiration_at"]),
            file_key=data.get("file_key"),
            file_size=int(data["file_size"]) if data.get("file_size") else None,
            file_hash=data.get("file_hash"),
            metadata=metadata,
        )


# =============================================================================
# Session Service
# =============================================================================


class SessionService:
    """
    Production-ready сервис сессий на Redis.
    """

    """
    Создаёт новую сессию загрузки для пользователя.

    Args:
        user_id (str): Идентификатор пользователя

    Returns:
        UploadSession: Объект сессии с сгенерированным session_id

    Raises:
        ValueError: Если user_id пустой

    Note:
        Сессия сохраняется в Redis как HASH с TTL = UPLOAD_EXPIRATION_DAYS дней.
        Возвращается локальный объект для удобства работы в коде.
    """

    @staticmethod
    async def create_session(user_id: str) -> UploadSession:
        if not user_id:
            raise ValueError("user_id is required")

        session_id = str(uuid.uuid4())
        created_at = utcnow()
        created_at_str = created_at.isoformat()

        expire_seconds = settings.UPLOAD_EXPIRATION_DAYS * 24 * 60 * 60
        expiration_at = created_at + timedelta(days=settings.UPLOAD_EXPIRATION_DAYS)

        key = f"upload_session:{session_id}"

        cache = CacheService()

        session_data = {
            "user_id": user_id,
            "created_at": created_at_str,
            "expiration_at": expiration_at.isoformat(),
        }

        await cache.set(key, session_data, expire_seconds=expire_seconds)

        logger.info("Upload session created", extra={"session_id": session_id})

        expiration_at = created_at + timedelta(days=settings.UPLOAD_EXPIRATION_DAYS)

        return UploadSession(
            session_id=session_id,
            user_id=user_id,
            created_at=created_at,
            expiration_at=expiration_at,
        )

    @staticmethod
    async def get_session(session_id: str) -> Optional[UploadSession]:
        if not session_id:
            return None

        key = f"upload_session:{session_id}"

        cache = CacheService()
        data = await cache.get(key)

        if data is None:
            return None

        return UploadSession.from_redis_data(session_id, data)

    @staticmethod
    async def attach_file_to_session(
        session_id: str,
        user_id: str,
        file_key: str,
        file_size: int,
        file_hash: str,
    ) -> UploadSession:
        """
        Безопасное обновление сессии с использованием Redis HSET.
        ✅ Атомарное обновление полей файла с сохранением оригинального TTL.
        """
        session = await SessionService.get_session(session_id)
        if not session:
            raise ValueError("Session not found or expired")

        if session.user_id != user_id:
            raise PermissionError("Session does not belong to user")

        key = f"upload_session:{session_id}"
        cache = CacheService()

        try:
            # ✅ Получаем текущий TTL перед обновлением
            ttl = await cache.redis.ttl(key)
            if ttl <= 0:
                raise ValueError("Session expired")

            # ✅ Атомарное обновление только нужных полей через HSET
            await cache.redis.hset(
                key,
                mapping={
                    "file_key": file_key,
                    "file_size": str(file_size),
                    "file_hash": file_hash,
                },
            )

            # ✅ Сохраняем оригинальный TTL
            await cache.redis.expire(key, ttl)

            logger.info("File attached to session", extra={"session_id": session_id})

            # Возвращаем обновлённую сессию
            return await SessionService.get_session(session_id)

        except Exception as e:
            logger.error(f"Failed to attach file to session: {e}")
            raise

    @staticmethod
    async def delete_session(session_id: str) -> bool:
        if not session_id:
            return False

        key = f"upload_session:{session_id}"

        cache = CacheService()
        deleted = await cache.delete(key)

        if deleted:
            logger.info("Session deleted", extra={"session_id": session_id})
            return True

        return False

    @staticmethod
    async def validate_session(session_id: str, user_id: str) -> bool:
        if not session_id or not user_id:
            return False

        key = f"upload_session:{session_id}"

        cache = CacheService()
        data = await cache.get(key)

        if data is None:
            return False  # не существует или истекла

        stored_user_id = data.get("user_id")
        if stored_user_id != user_id:
            logger.warning(
                "Session user mismatch",
                extra={
                    "session_id": session_id,
                    "expected_user_id": user_id,
                    "stored_user_id": stored_user_id,
                },
            )
            return False

        return True

    @staticmethod
    async def get_user_sessions(user_id: str) -> List[UploadSession]:
        """
        Получение всех активных сессий пользователя
        """
        cache = CacheService()
        pattern = "upload_session:*"

        # Scan all session keys
        all_keys = []
        cursor = 0
        while True:
            cursor, keys = await cache.redis.scan(cursor, match=pattern, count=100)
            all_keys.extend(keys)
            if cursor == 0:
                break

        # Filter by user_id
        user_sessions = []
        for key in all_keys:
            session_id = key.decode().split(":")[-1]
            session = await SessionService.get_session(session_id)
            if session and session.user_id == user_id:
                user_sessions.append(session)

        return user_sessions
