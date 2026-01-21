# app/services/audit_service.py
"""
Сервис для записи audit trail событий в БД.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import AuditLog
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AuditService:
    """
    Сервис для работы с audit logs.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def log_event(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditLog:
        """
        Записывает audit событие в БД.

        Args:
            action: Тип действия (e.g., "user_login", "reference_deleted")
            resource_type: Тип ресурса (e.g., "user", "reference")
            resource_id: ID ресурса
            user_id: ID пользователя
            old_values: Старые значения (для update/delete)
            new_values: Новые значения (для create/update)
            details: Дополнительные детали
            ip_address: IP адрес
            user_agent: User agent
            success: Успешно ли выполнено
            error_message: Сообщение об ошибке

        Returns:
            AuditLog: Созданная запись
        """
        try:
            audit_log = AuditLog(
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                old_values=old_values,
                new_values=new_values,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                error_message=error_message,
            )

            self.db.add(audit_log)
            await self.db.commit()
            await self.db.refresh(audit_log)

            logger.debug(
                f"Audit event logged: {action} on {resource_type}:{resource_id}"
            )
            return audit_log

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            await self.db.rollback()
            raise

    async def get_logs(
        self,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """
        Получает audit logs с фильтрацией.

        Args:
            action: Фильтр по действию
            resource_type: Фильтр по типу ресурса
            user_id: Фильтр по пользователю
            date_from: Начальная дата
            date_to: Конечная дата
            limit: Максимальное количество записей
            offset: Смещение для пагинации

        Returns:
            List[AuditLog]: Список audit logs
        """
        try:
            query = select(AuditLog)

            # Применяем фильтры
            conditions = []
            if action:
                conditions.append(AuditLog.action == action)
            if resource_type:
                conditions.append(AuditLog.resource_type == resource_type)
            if user_id:
                conditions.append(AuditLog.user_id == user_id)
            if date_from:
                conditions.append(AuditLog.created_at >= date_from)
            if date_to:
                conditions.append(AuditLog.created_at <= date_to)

            if conditions:
                query = query.where(and_(*conditions))

            # Сортировка и пагинация
            query = (
                query.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
            )

            result = await self.db.execute(query)
            logs = result.scalars().all()

            logger.debug(f"Retrieved {len(logs)} audit logs")
            return logs

        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            raise

    async def get_logs_count(
        self,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> int:
        """
        Получает количество audit logs с фильтрацией.
        """
        try:
            query = select(func.count(AuditLog.id))

            conditions = []
            if action:
                conditions.append(AuditLog.action == action)
            if resource_type:
                conditions.append(AuditLog.resource_type == resource_type)
            if user_id:
                conditions.append(AuditLog.user_id == user_id)
            if date_from:
                conditions.append(AuditLog.created_at >= date_from)
            if date_to:
                conditions.append(AuditLog.created_at <= date_to)

            if conditions:
                query = query.where(and_(*conditions))

            result = await self.db.execute(query)
            count = result.scalar()

            return count or 0

        except Exception as e:
            logger.error(f"Failed to count audit logs: {e}")
            return 0
