"""
CRUD операции для работы с базой данных.
Базовые операции создания, чтения, обновления и удаления.
"""

from typing import Optional, List, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone

from .database import get_db_session
from .models import User, Reference, VerificationSession, AuditLog, SystemConfig, ApiKey
from ..utils.logger import get_logger
from ..utils.exceptions import DatabaseError, NotFoundError

logger = get_logger(__name__)


class BaseCRUD:
    """
    Базовый класс для CRUD операций.
    """
    
    def __init__(self, model):
        self.model = model
    
    async def create(self, db: AsyncSession, obj_data: Dict[str, Any]) -> Any:
        """
        Создание новой записи.
        
        Args:
            db: Сессия базы данных
            obj_data: Данные для создания
            
        Returns:
            Созданный объект
        """
        try:
            db_obj = self.model(**obj_data)
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to create {self.model.__name__}: {str(e)}")
    
    async def get(self, db: AsyncSession, id: str) -> Optional[Any]:
        """
        Получение записи по ID.
        
        Args:
            db: Сессия базы данных
            id: ID записи
            
        Returns:
            Найденный объект или None
        """
        try:
            result = await db.execute(select(self.model).where(self.model.id == id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} by id {id}: {str(e)}")
            raise DatabaseError(f"Failed to get {self.model.__name__}: {str(e)}")
    
    async def get_multi(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        return_total: bool = False
    ) -> Union[List[Any], tuple[List[Any], int]]:
        """
        Получение списка записей с пагинацией.
        
        Args:
            db: Сессия базы данных
            skip: Количество записей для пропуска
            limit: Максимальное количество записей
            filters: Фильтры для поиска
            return_total: Возвращать ли общее количество записей
            
        Returns:
            Если return_total=True: (список, total_count)
            Иначе: список
        """
        try:
            query = select(self.model)
            if filters:
                query = self._apply_filters(query, filters)
            if return_total:
                count_query = select(func.count(self.model.id))
                if filters:
                    count_query = self._apply_filters(count_query, filters)
                total = (await db.execute(count_query)).scalar()
            query = query.offset(skip).limit(limit)
            result = await db.execute(query)
            items = result.scalars().all()
            return (items, total) if return_total else items
        except Exception as e:
            logger.error(f"Error getting {self.model.__name__} list: {str(e)}")
            raise DatabaseError(f"Failed to get {self.model.__name__} list: {str(e)}")
    
    async def update(self, db: AsyncSession, id: str, obj_data: Dict[str, Any]) -> Optional[Any]:
        """
        Обновление записи.
        
        Args:
            db: Сессия базы данных
            id: ID записи
            obj_data: Данные для обновления
            
        Returns:
            Обновленный объект или None
        """
        try:
            # Добавляем updated_at если поле существует
            if hasattr(self.model, 'updated_at'):
                obj_data['updated_at'] = datetime.now(timezone.utc)
            
            result = await db.execute(
                update(self.model)
                .where(self.model.id == id)
                .values(**obj_data)
                .returning(self.model)
            )
            
            updated_obj = result.scalar_one_or_none()
            
            if not updated_obj:
                raise NotFoundError(f"{self.model.__name__} with id {id} not found")
            
            await db.commit()
            await db.refresh(updated_obj)
            return updated_obj
        except NotFoundError:
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating {self.model.__name__} {id}: {str(e)}")
            raise DatabaseError(f"Failed to update {self.model.__name__}: {str(e)}")
    
    async def delete(self, db: AsyncSession, id: str) -> bool:
        """
        Удаление записи.
        
        Args:
            db: Сессия базы данных
            id: ID записи
            
        Returns:
            True если запись удалена
        """
        try:
            result = await db.execute(delete(self.model).where(self.model.id == id))
            deleted_count = result.rowcount
            await db.commit()
            
            if deleted_count == 0:
                raise NotFoundError(f"{self.model.__name__} with id {id} not found")
            
            return True
        except NotFoundError:
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting {self.model.__name__} {id}: {str(e)}")
            raise DatabaseError(f"Failed to delete {self.model.__name__}: {str(e)}")
    
    async def count(self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Подсчет записей.
        
        Args:
            db: Сессия базы данных
            filters: Фильтры для поиска
            
        Returns:
            Количество записей
        """
        try:
            query = select(func.count(self.model.id))
            
            if filters:
                query = self._apply_filters(query, filters)
            
            result = await db.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error counting {self.model.__name__}: {str(e)}")
            raise DatabaseError(f"Failed to count {self.model.__name__}: {str(e)}")
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """
        Применение фильтров к запросу.
        Поддерживает:
        - Точное совпадение: {"field": value}
        - IN оператор: {"field": [value1, value2]}
        - Операторы: {"field__gt": value, "field__lt": value, "field__like": "%value%"}
        """
        for field, value in filters.items():
            # Проверка на операторы
            if "__" in field:
                field_name, operator = field.split("__", 1)
                if not hasattr(self.model, field_name):
                    continue
                field_obj = getattr(self.model, field_name)
                if operator == "gt":
                    query = query.where(field_obj > value)
                elif operator == "lt":
                    query = query.where(field_obj < value)
                elif operator == "gte":
                    query = query.where(field_obj >= value)
                elif operator == "lte":
                    query = query.where(field_obj <= value)
                elif operator == "like":
                    query = query.where(field_obj.like(value))
                elif operator == "ilike":
                    query = query.where(field_obj.ilike(value))
                elif operator == "ne":
                    query = query.where(field_obj != value)
                else:
                    continue
            else:
                if not hasattr(self.model, field):
                    continue
                field_obj = getattr(self.model, field)
                if isinstance(value, list):
                    query = query.where(field_obj.in_(value))
                elif value is None:
                    query = query.where(field_obj.is_(None))
                else:
                    query = query.where(field_obj == value)
        
        return query


# CRUD классы для каждой модели

class UserCRUD(BaseCRUD):
    """
    CRUD операции для пользователей.
    """
    
    def __init__(self):
        super().__init__(User)
    
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Получение пользователя по email."""
        try:
            result = await db.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            raise DatabaseError(f"Failed to get user by email: {str(e)}")
    
    async def get_by_username(self, db: AsyncSession, username: str) -> Optional[User]:
        """Получение пользователя по имени."""
        try:
            result = await db.execute(select(User).where(User.username == username))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by username {username}: {str(e)}")
            raise DatabaseError(f"Failed to get user by username: {str(e)}")
    
    async def get_active_users(self, db: AsyncSession) -> List[User]:
        """Получение активных пользователей."""
        try:
            result = await db.execute(
                select(User).where(User.is_active == True).order_by(desc(User.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting active users: {str(e)}")
            raise DatabaseError(f"Failed to get active users: {str(e)}")
    
    async def update_last_login(self, db: AsyncSession, user_id: str) -> bool:
        """Обновление времени последнего входа."""
        try:
            result = await db.execute(
                update(User)
                .where(User.id == user_id)
                .values(last_login=datetime.now(timezone.utc))
            )
            await db.commit()
            return result.rowcount > 0
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating last login for user {user_id}: {str(e)}")
            return False

    async def increment_stats(self, db: AsyncSession, user_id: str, stats_type: str, increment: int = 1) -> bool:
        """Увеличение статистики пользователя."""
        try:
            if stats_type == "uploads":
                field = User.total_uploads
            elif stats_type == "verifications":
                field = User.total_verifications
            elif stats_type == "successful_verifications":
                field = User.successful_verifications
            else:
                raise ValueError(f"Unknown stats type: {stats_type}")

            result = await db.execute(
                update(User)
                .where(User.id == user_id)
                .values({field.name: field + increment})
            )
            await db.commit()
            return result.rowcount > 0
        except Exception as e:
            await db.rollback()
            logger.error(f"Error incrementing {stats_type} for user {user_id}: {str(e)}")
            return False


class ReferenceCRUD(BaseCRUD):
    """
    CRUD операции для эталонов.
    """
    
    def __init__(self):
        super().__init__(Reference)
    
    async def get_by_user(self, db: AsyncSession, user_id: str, active_only: bool = True) -> List[Reference]:
        """Получение эталонов пользователя."""
        try:
            query = select(Reference).where(Reference.user_id == user_id)
            
            if active_only:
                query = query.where(Reference.is_active == True)
            
            query = query.order_by(desc(Reference.created_at))
            
            result = await db.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting references for user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to get references: {str(e)}")
    
    async def get_by_label(self, db: AsyncSession, user_id: str, label: str) -> Optional[Reference]:
        """Получение эталона по метке."""
        try:
            result = await db.execute(
                select(Reference).where(
                    and_(
                        Reference.user_id == user_id,
                        Reference.label == label
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting reference by label {label} for user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to get reference by label: {str(e)}")
    
    async def increment_usage(self, db: AsyncSession, reference_id: str) -> bool:
        """Увеличение счетчика использования."""
        try:
            result = await db.execute(
                update(Reference)
                .where(Reference.id == reference_id)
                .values(
                    usage_count=Reference.usage_count + 1,
                    last_used=datetime.now(timezone.utc)
                )
            )
            await db.commit()
            return result.rowcount > 0
        except Exception as e:
            await db.rollback()
            logger.error(f"Error incrementing usage for reference {reference_id}: {str(e)}")
            return False


class VerificationSessionCRUD(BaseCRUD):
    """
    CRUD операции для сессий верификации.
    """
    
    def __init__(self):
        super().__init__(VerificationSession)
    
    async def get_by_user(self, db: AsyncSession, user_id: str, session_type: Optional[str] = None) -> List[VerificationSession]:
        """Получение сессий пользователя."""
        try:
            query = select(VerificationSession).where(VerificationSession.user_id == user_id)
            
            if session_type:
                query = query.where(VerificationSession.session_type == session_type)
            
            query = query.order_by(desc(VerificationSession.created_at))
            
            result = await db.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting sessions for user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to get sessions: {str(e)}")
    
    async def get_active_sessions(self, db: AsyncSession) -> List[VerificationSession]:
        """Получение активных сессий."""
        try:
            result = await db.execute(
                select(VerificationSession)
                .where(VerificationSession.status.in_(["pending", "processing"]))
                .order_by(desc(VerificationSession.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting active sessions: {str(e)}")
            raise DatabaseError(f"Failed to get active sessions: {str(e)}")
    
    async def cleanup_expired_sessions(self, db: AsyncSession) -> int:
        """Очистка истекших сессий."""
        try:
            now = datetime.now(timezone.utc)
            result = await db.execute(
                update(VerificationSession)
                .where(
                    and_(
                        VerificationSession.expires_at < now,
                        VerificationSession.status.in_(["pending", "processing"])
                    )
                )
                .values(status="expired")
            )
            await db.commit()
            return result.rowcount
        except Exception as e:
            await db.rollback()
            logger.error(f"Error cleaning up expired sessions: {str(e)}")
            return 0


class AuditLogCRUD(BaseCRUD):
    """
    CRUD операции для журнала аудита.
    """
    
    def __init__(self):
        super().__init__(AuditLog)
    
    async def create_log(
        self,
        db: AsyncSession,
        user_id: Optional[str],
        action: str,
        resource_type: Optional[str],
        resource_id: Optional[str],
        description: Optional[str] = None,
        old_values: Optional[Dict] = None,
        new_values: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """Создание записи в журнале аудита."""
        try:
            log_data = {
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "description": description,
                "old_values": old_values,
                "new_values": new_values,
                "ip_address": ip_address,
                "user_agent": user_agent
            }
            
            return await self.create(db, log_data)
        except Exception as e:
            logger.error(f"Error creating audit log: {str(e)}")
            raise DatabaseError(f"Failed to create audit log: {str(e)}")
    
    async def get_by_user(self, db: AsyncSession, user_id: str, limit: int = 100) -> List[AuditLog]:
        """Получение логов пользователя."""
        try:
            result = await db.execute(
                select(AuditLog)
                .where(AuditLog.user_id == user_id)
                .order_by(desc(AuditLog.created_at))
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting audit logs for user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to get audit logs: {str(e)}")


class SystemConfigCRUD(BaseCRUD):
    """
    CRUD операции для системных настроек.
    """
    
    def __init__(self):
        super().__init__(SystemConfig)
    
    async def get_by_key(self, db: AsyncSession, key: str) -> Optional[SystemConfig]:
        """Получение настройки по ключу."""
        try:
            result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting config by key {key}: {str(e)}")
            raise DatabaseError(f"Failed to get config: {str(e)}")
    
    async def get_by_category(self, db: AsyncSession, category: str) -> List[SystemConfig]:
        """Получение настроек по категории."""
        try:
            result = await db.execute(
                select(SystemConfig)
                .where(SystemConfig.category == category)
                .where(SystemConfig.is_active == True)
                .order_by(SystemConfig.key)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting configs by category {category}: {str(e)}")
            raise DatabaseError(f"Failed to get configs: {str(e)}")


class ApiKeyCRUD(BaseCRUD):
    """
    CRUD операции для API ключей.
    """
    
    def __init__(self):
        super().__init__(ApiKey)
    
    async def get_by_hash(self, db: AsyncSession, key_hash: str) -> Optional[ApiKey]:
        """Получение API ключа по хешу."""
        try:
            result = await db.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting API key by hash {key_hash}: {str(e)}")
            raise DatabaseError(f"Failed to get API key: {str(e)}")
    
    async def get_by_user(self, db: AsyncSession, user_id: str) -> List[ApiKey]:
        """Получение API ключей пользователя."""
        try:
            result = await db.execute(
                select(ApiKey)
                .where(ApiKey.user_id == user_id)
                .order_by(desc(ApiKey.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting API keys for user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to get API keys: {str(e)}")
    
    async def increment_requests(self, db: AsyncSession, api_key_id: str, successful: bool = True) -> bool:
        """Увеличение счетчика запросов."""
        try:
            field = ApiKey.successful_requests if successful else ApiKey.failed_requests
            
            result = await db.execute(
                update(ApiKey)
                .where(ApiKey.id == api_key_id)
                .values(
                    {field.name: field + 1},
                    total_requests=ApiKey.total_requests + 1,
                    last_used=datetime.now(timezone.utc)
                )
            )
            await db.commit()
            return result.rowcount > 0
        except Exception as e:
            await db.rollback()
            logger.error(f"Error incrementing requests for API key {api_key_id}: {str(e)}")
            return False


# Создание экземпляров CRUD классов
user_crud = UserCRUD()
reference_crud = ReferenceCRUD()
verification_session_crud = VerificationSessionCRUD()
audit_log_crud = AuditLogCRUD()
system_config_crud = SystemConfigCRUD()
api_key_crud = ApiKeyCRUD()