"""
Сервис управления сессиями загрузки.
Реализация временных сессий для загрузки файлов с TTL.

WARNING:
- In-memory хранилище (_sessions) НЕ подходит для production с multiple workers
- Для production используйте Redis или другую distributed storage
- Thread safety не гарантирована для concurrent access
"""

import uuid
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UploadSession:
    """Модель сессии загрузки"""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        created_at: datetime,
        expiration_at: datetime
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = created_at
        self.expiration_at = expiration_at
        self.file_key: Optional[str] = None
        self.file_size: Optional[float] = None
        self.file_hash: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def is_expired(self) -> bool:
        """Проверка истечения сессии"""
        return datetime.utcnow() > self.expiration_at
    
    def update_file_info(self, file_key: str, file_size: float, file_hash: str):
        """Обновление информации о файле"""
        self.file_key = file_key
        self.file_size = file_size
        self.file_hash = file_hash
    
    def __repr__(self):
        return f"<UploadSession({self.session_id}, user={self.user_id})>"


class SessionService:
    """
    Сервис для управления сессиями загрузки
    
    ⚠️  PRODUCTION LIMITATIONS:
    - Использует in-memory хранилище (_sessions)
    - НЕ подходит для multiple workers (каждый worker имеет свою копию)
    - НЕ подходит для distributed deployment
    - Thread safety не гарантирована
    
    TODO: Для production заменить на:
    - Redis с TTL для sessions
    - Или database storage с TTL
    - Или distributed cache (memcached, Hazelcast)
    """
    
    # In-memory хранилище сессий
    # ⚠️  WARNING: Это не thread-safe и не работает с multiple workers!
    _sessions: Dict[str, UploadSession] = {}
    
    # Thread lock для basic thread safety
    _lock = threading.RLock()
    
    @classmethod
    def _get_redis_session_store(cls):
        """
        Получение Redis хранилища для сессий (для production)
        
        TODO: Реализовать для production
        """
        # import redis
        # return redis.Redis.from_url(settings.REDIS_URL)
        return None
    
    @classmethod
    def create_session(cls, user_id: str) -> UploadSession:
        """
        Создание новой сессии загрузки
        
        Args:
            user_id: ID пользователя
            
        Returns:
            UploadSession: Объект сессии
            
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        """
        try:
            session_id = str(uuid.uuid4())
            created_at = datetime.utcnow()
            expiration_at = created_at + timedelta(days=settings.UPLOAD_EXPIRATION_DAYS)
            
            session = UploadSession(
                session_id=session_id,
                user_id=user_id,
                created_at=created_at,
                expiration_at=expiration_at
            )
            
            # Thread-safe добавление сессии
            with cls._lock:
                cls._sessions[session_id] = session
            
            logger.info(f"Создана сессия загрузки: {session_id} для пользователя {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Ошибка создания сессии загрузки: {e}")
            raise
    
    @classmethod
    def get_session(cls, session_id: str) -> Optional[UploadSession]:
        """
        Получение сессии загрузки
        
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        """
        try:
            with cls._lock:
                session = cls._sessions.get(session_id)
                
            if not session:
                logger.warning(f"Сессия не найдена: {session_id}")
                return None
                
            if session.is_expired():
                logger.warning(f"Сессия истекла: {session_id}")
                with cls._lock:
                    cls._sessions.pop(session_id, None)
                return None
                
            return session
            
        except Exception as e:
            logger.error(f"Ошибка получения сессии: {e}")
            raise
    
    @classmethod
    def validate_session(cls, session_id: str, user_id: str) -> bool:
        """
        Валидация сессии пользователя
        
        Args:
            session_id: ID сессии
            user_id: ID пользователя
            
        Returns:
            bool: True если валидна
            
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        """
        try:
            session = cls.get_session(session_id)
            if not session:
                return False
                
            if session.user_id != user_id:
                logger.warning(
                    f"Несоответствие пользователя в сессии: {session_id} "
                    f"(ожидался {user_id}, получен {session.user_id})"
                )
                return False
                
            logger.debug(f"Сессия валидна: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации сессии: {e}")
            return False
    
    @classmethod
    def update_session(cls, session_id: str, **kwargs) -> Optional[UploadSession]:
        """
        Обновление метаданных сессии
        
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        """
        try:
            with cls._lock:
                session = cls._sessions.get(session_id)
                if not session:
                    return None
                    
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                        
            logger.info(f"Сессия обновлена: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Ошибка обновления сессии: {e}")
            raise
    
    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """
        Удаление сессии
        
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        """
        try:
            with cls._lock:
                if session_id in cls._sessions:
                    cls._sessions.pop(session_id)
                    logger.info(f"Сессия удалена: {session_id}")
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Ошибка удаления сессии: {e}")
            raise
    
    @classmethod
    def cleanup_expired_sessions(cls) -> int:
        """
        Удаление всех истекших сессий
        
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        - Возможны race conditions при concurrent cleanup
        """
        try:
            with cls._lock:
                expired_sessions = [
                    sid for sid, session in cls._sessions.items()
                    if session.is_expired()
                ]
                
                for session_id in expired_sessions:
                    cls._sessions.pop(session_id)
                    
            logger.info(f"Очищено {len(expired_sessions)} истекших сессий")
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Ошибка очистки сессий: {e}")
            raise
    
    @classmethod
    def get_active_sessions_count(cls) -> int:
        """
        Получение количества активных сессий
        
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        """
        try:
            with cls._lock:
                active_sessions = [
                    session for session in cls._sessions.values()
                    if not session.is_expired()
                ]
            return len(active_sessions)
        except Exception as e:
            logger.error(f"Ошибка подсчета активных сессий: {e}")
            return 0
    
    @classmethod
    def get_user_sessions(cls, user_id: str) -> list:
        """
        Получение всех сессий пользователя
        
        ⚠️  PRODUCTION WARNING:
        - Не работает с multiple workers
        - Используйте Redis для production
        """
        try:
            with cls._lock:
                user_sessions = [
                    session for session in cls._sessions.values()
                    if session.user_id == user_id and not session.is_expired()
                ]
            return user_sessions
        except Exception as e:
            logger.error(f"Ошибка получения сессий пользователя: {e}")
            return []

    @classmethod
    def get_session_stats(cls) -> Dict[str, Any]:
        """
        Получение статистики о текущем состоянии сессий
        
        Returns:
            Dict[str, Any]: Статистика сессий
        """
        try:
            with cls._lock:
                total_sessions = len(cls._sessions)
                expired_sessions = len([s for s in cls._sessions.values() if s.is_expired()])
                active_sessions = total_sessions - expired_sessions
                
                # Статистика по пользователям
                user_stats = {}
                for session in cls._sessions.values():
                    if not session.is_expired():
                        user_id = session.user_id
                        if user_id not in user_stats:
                            user_stats[user_id] = 0
                        user_stats[user_id] += 1
                
                return {
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "expired_sessions": expired_sessions,
                    "users_with_sessions": len(user_stats),
                    "top_users": sorted(user_stats.items(), key=lambda x: x[1], reverse=True)[:5],
                    "storage_type": "in_memory",
                    "thread_safe": True,
                    "production_ready": False
                }
        except Exception as e:
            logger.error(f"Ошибка получения статистики сессий: {e}")
            return {"error": str(e)}

    @classmethod
    def get_production_recommendations(cls) -> Dict[str, str]:
        """
        Получение рекомендаций для production deployment
        
        Returns:
            Dict[str, str]: Рекомендации и TODO items
        """
        return {
            "storage": "Заменить in-memory storage на Redis с TTL",
            "deployment": "Для multiple workers обязательно использовать distributed storage",
            "thread_safety": "Current RLock обеспечивает basic thread safety, но не distributed safety",
            "scaling": "Не подходит для горизонтального масштабирования",
            "todo_redis": "TODO: Реализовать RedisSessionService с командами SETEX, GET, DEL",
            "todo_database": "TODO: Альтернативно - использовать database с TTL индексами",
            "todo_monitoring": "TODO: Добавить метрики количества сессий и времени жизни",
            "warning": "WARNING: НЕ ИСПОЛЬЗУЙТЕ в production без замены storage!"
        }


# =============================================================================
# PRODUCTION IMPLEMENTATION EXAMPLES
# =============================================================================

class RedisSessionService:
    """
    Пример Redis-based SessionService для production
    
    TODO: Реализовать для production deployment
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl_seconds = settings.UPLOAD_EXPIRATION_DAYS * 24 * 60 * 60
    
    async def create_session(self, user_id: str) -> UploadSession:
        """Создание сессии в Redis"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expiration_at": (datetime.utcnow() + timedelta(days=settings.UPLOAD_EXPIRATION_DAYS)).isoformat()
        }
        
        # Сохраняем в Redis с TTL
        await self.redis.setex(
            f"session:{session_id}",
            self.ttl_seconds,
            json.dumps(session_data)
        )
        
        return UploadSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.fromisoformat(session_data["created_at"]),
            expiration_at=datetime.fromisoformat(session_data["expiration_at"])
        )
    
    async def get_session(self, session_id: str) -> Optional[UploadSession]:
        """Получение сессии из Redis"""
        data = await self.redis.get(f"session:{session_id}")
        if not data:
            return None
            
        session_data = json.loads(data)
        return UploadSession(
            session_id=session_data["session_id"],
            user_id=session_data["user_id"],
            created_at=datetime.fromisoformat(session_data["created_at"]),
            expiration_at=datetime.fromisoformat(session_data["expiration_at"])
        )
    
    async def delete_session(self, session_id: str) -> bool:
        """Удаление сессии из Redis"""
        result = await self.redis.delete(f"session:{session_id}")
        return result > 0