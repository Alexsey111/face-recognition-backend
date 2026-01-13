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
    # ---------------------------------------------------------------------
  
@classmethod
def from_redis_data(cls, session_id: str, data: Dict[str, str]) -> "UploadSession":
    return cls(
        session_id=session_id,
        user_id=data["user_id"],  # обязательное, предполагаем, что есть
        created_at=datetime.fromisoformat(data["created_at"]),
        expiration_at=datetime.fromisoformat(data["expiration_at"]),
        file_key=data.get("file_key"),
        file_size=int(data["file_size"]) if data.get("file_size") else None,
        file_hash=data.get("file_hash"),
        metadata=json.loads(data.get("metadata", "{}")),
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
        
        key = f"upload_session:{session_id}"
        
        cache = CacheService()
        
        session_data = {
            "user_id": user_id,
            "created_at": created_at_str,
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
        Безопасное обновление сессии (ТОЛЬКО файл)
        """
        session = await SessionService.get_session(session_id)
        if not session:
            raise ValueError("Session not found or expired")
        
        if session.user_id != user_id:
            raise PermissionError("Session does not belong to user")
        
        key = f"upload_session:{session_id}"
        
        cache = CacheService()
        
        updated_data = {
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "file_key": file_key,
            "file_size": file_size,
            "file_hash": file_hash,
        }
        
        await cache.set(key, updated_data)  # TTL сохранится от создания
        
        logger.info("File attached to session", extra={"session_id": session_id})
        
        # Возвращаем обновлённую сессию
        return await SessionService.get_session(session_id)
    
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
                extra={"session_id": session_id, "expected_user_id": user_id, "stored_user_id": stored_user_id},
            )
            return False
        
        return True
