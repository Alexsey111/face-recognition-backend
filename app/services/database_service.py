"""
Сервис базы данных.
Обёртка над CRUD операциями для работы с базой данных.
"""

from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from ..db.database import get_db_session
from ..db.models import User, Reference, VerificationSession, AuditLog
from ..utils.logger import get_logger
from ..utils.exceptions import DatabaseError, NotFoundError

logger = get_logger(__name__)


class DatabaseService:
    """
    Сервис для работы с базой данных.
    """

    def __init__(self):
        self.db_session = get_db_session

    # User operations

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение пользователя по ID.

        Args:
            user_id: ID пользователя

        Returns:
            Optional[Dict[str, Any]]: Данные пользователя или None
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()

                if user:
                    return self._user_to_dict(user)
                return None

        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to get user: {str(e)}")

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Получение пользователя по имени.

        Args:
            username: Имя пользователя

        Returns:
            Optional[Dict[str, Any]]: Данные пользователя или None
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(User).where(User.username == username.lower())
                )
                user = result.scalar_one_or_none()

                if user:
                    return self._user_to_dict(user)
                return None

        except Exception as e:
            logger.error(f"Error getting user by username {username}: {str(e)}")
            raise DatabaseError(f"Failed to get user: {str(e)}")

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Получение пользователя по email.

        Args:
            email: Email пользователя

        Returns:
            Optional[Dict[str, Any]]: Данные пользователя или None
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(User).where(User.email == email.lower())
                )
                user = result.scalar_one_or_none()

                if user:
                    return self._user_to_dict(user)
                return None

        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            raise DatabaseError(f"Failed to get user: {str(e)}")

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание нового пользователя.

        Args:
            user_data: Данные пользователя

        Returns:
            Dict[str, Any]: Созданный пользователь
        """
        try:
            async with self.db_session() as session:
                user = User(**user_data)
                session.add(user)
                await session.commit()
                await session.refresh(user)

                logger.info(f"User created successfully: {user.id}")
                return self._user_to_dict(user)

        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise DatabaseError(f"Failed to create user: {str(e)}")

    async def update_user(
        self, user_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Обновление пользователя.

        Args:
            user_id: ID пользователя
            update_data: Данные для обновления

        Returns:
            Dict[str, Any]: Обновленный пользователь
        """
        try:
            async with self.db_session() as session:
                # Обновляем поле updated_at
                update_data["updated_at"] = datetime.now(timezone.utc)

                result = await session.execute(
                    update(User)
                    .where(User.id == user_id)
                    .values(**update_data)
                    .returning(User)
                )

                user = result.scalar_one_or_none()

                if not user:
                    raise NotFoundError(f"User {user_id} not found")

                await session.commit()

                logger.info(f"User updated successfully: {user_id}")
                return self._user_to_dict(user)

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to update user: {str(e)}")

    async def delete_user(self, user_id: str) -> bool:
        """
        Удаление пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            bool: True если пользователь удален
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(delete(User).where(User.id == user_id))

                deleted_count = result.rowcount
                await session.commit()

                if deleted_count > 0:
                    logger.info(f"User deleted successfully: {user_id}")
                    return True
                else:
                    logger.warning(f"User not found for deletion: {user_id}")
                    return False

        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to delete user: {str(e)}")

    async def get_all_users(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Получение списка всех пользователей.

        Args:
            filters: Фильтры для поиска
            page: Номер страницы
            per_page: Количество пользователей на странице
            sort_by: Поле для сортировки
            sort_order: Порядок сортировки

        Returns:
            Dict[str, Any]: Список пользователей с пагинацией
        """
        try:
            async with self.db_session() as session:
                # Построение базового запроса
                query = select(User)

                # Применение фильтров
                if filters:
                    query = self._apply_user_filters(query, filters)

                # Сортировка
                sort_column = getattr(User, sort_by, User.created_at)
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))

                # Подсчет общего количества
                count_query = select(func.count(User.id))
                if filters:
                    count_query = self._apply_user_filters(count_query, filters)

                total_count = await session.scalar(count_query)

                # Пагинация
                offset = (page - 1) * per_page
                query = query.offset(offset).limit(per_page)

                # Выполнение запроса
                result = await session.execute(query)
                users = result.scalars().all()

                # Преобразование в словари
                users_data = [self._user_to_dict(user) for user in users]

                return {
                    "users": users_data,
                    "total_count": total_count,
                    "page": page,
                    "per_page": per_page,
                    "has_next": offset + per_page < total_count,
                    "has_prev": page > 1,
                }

        except Exception as e:
            logger.error(f"Error getting all users: {str(e)}")
            raise DatabaseError(f"Failed to get users: {str(e)}")

    # Reference operations

    async def get_reference_by_id(self, reference_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение эталона по ID.

        Args:
            reference_id: ID эталона

        Returns:
            Optional[Dict[str, Any]]: Данные эталона или None
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(Reference).where(Reference.id == reference_id)
                )
                reference = result.scalar_one_or_none()

                if reference:
                    return self._reference_to_dict(reference)
                return None

        except Exception as e:
            logger.error(f"Error getting reference by ID {reference_id}: {str(e)}")
            raise DatabaseError(f"Failed to get reference: {str(e)}")

    async def get_active_references_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Получение активных эталонов пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            List[Dict[str, Any]]: Список активных эталонов
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(Reference)
                    .where(
                        and_(Reference.user_id == user_id, Reference.is_active == True)
                    )
                    .order_by(desc(Reference.created_at))
                )
                references = result.scalars().all()

                return [self._reference_to_dict(ref) for ref in references]

        except Exception as e:
            logger.error(
                f"Error getting active references for user {user_id}: {str(e)}"
            )
            raise DatabaseError(f"Failed to get references: {str(e)}")

    async def create_reference(self, reference_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создание нового эталона.

        Args:
            reference_data: Данные эталона

        Returns:
            Dict[str, Any]: Созданный эталон
        """
        try:
            async with self.db_session() as session:
                reference = Reference(**reference_data)
                session.add(reference)
                await session.commit()
                await session.refresh(reference)

                logger.info(f"Reference created successfully: {reference.id}")
                return self._reference_to_dict(reference)

        except Exception as e:
            logger.error(f"Error creating reference: {str(e)}")
            raise DatabaseError(f"Failed to create reference: {str(e)}")

    async def update_reference(
        self, reference_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Обновление эталона.

        Args:
            reference_id: ID эталона
            update_data: Данные для обновления

        Returns:
            Dict[str, Any]: Обновленный эталон
        """
        try:
            async with self.db_session() as session:
                update_data["updated_at"] = datetime.now(timezone.utc)

                result = await session.execute(
                    update(Reference)
                    .where(Reference.id == reference_id)
                    .values(**update_data)
                    .returning(Reference)
                )

                reference = result.scalar_one_or_none()

                if not reference:
                    raise NotFoundError(f"Reference {reference_id} not found")

                await session.commit()

                logger.info(f"Reference updated successfully: {reference_id}")
                return self._reference_to_dict(reference)

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating reference {reference_id}: {str(e)}")
            raise DatabaseError(f"Failed to update reference: {str(e)}")

    async def delete_reference(self, reference_id: str) -> bool:
        """
        Удаление эталона.

        Args:
            reference_id: ID эталона

        Returns:
            bool: True если эталон удален
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    delete(Reference).where(Reference.id == reference_id)
                )

                deleted_count = result.rowcount
                await session.commit()

                if deleted_count > 0:
                    logger.info(f"Reference deleted successfully: {reference_id}")
                    return True
                else:
                    logger.warning(f"Reference not found for deletion: {reference_id}")
                    return False

        except Exception as e:
            logger.error(f"Error deleting reference {reference_id}: {str(e)}")
            raise DatabaseError(f"Failed to delete reference: {str(e)}")

    async def get_references(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """
        Получение списка эталонов.

        Args:
            filters: Фильтры для поиска
            page: Номер страницы
            per_page: Количество эталонов на странице
            sort_by: Поле для сортировки
            sort_order: Порядок сортировки

        Returns:
            Dict[str, Any]: Список эталонов с пагинацией
        """
        try:
            async with self.db_session() as session:
                query = select(Reference)

                if filters:
                    query = self._apply_reference_filters(query, filters)

                sort_column = getattr(Reference, sort_by, Reference.created_at)
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))

                # Подсчет общего количества
                count_query = select(func.count(Reference.id))
                if filters:
                    count_query = self._apply_reference_filters(count_query, filters)

                total_count = await session.scalar(count_query)

                # Пагинация
                offset = (page - 1) * per_page
                query = query.offset(offset).limit(per_page)

                result = await session.execute(query)
                references = result.scalars().all()

                references_data = [self._reference_to_dict(ref) for ref in references]

                return {
                    "items": references_data,
                    "total_count": total_count,
                    "page": page,
                    "per_page": per_page,
                    "has_next": offset + per_page < total_count,
                    "has_prev": page > 1,
                }

        except Exception as e:
            logger.error(f"Error getting references: {str(e)}")
            raise DatabaseError(f"Failed to get references: {str(e)}")

    # Verification session operations

    async def create_verification_session(
        self, session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Создание сессии верификации.

        Args:
            session_data: Данные сессии

        Returns:
            Dict[str, Any]: Созданная сессия
        """
        try:
            async with self.db_session() as session:
                session_obj = VerificationSession(**session_data)
                session.add(session_obj)
                await session.commit()
                await session.refresh(session_obj)

                logger.info(f"Verification session created: {session_obj.id}")
                return self._session_to_dict(session_obj)

        except Exception as e:
            logger.error(f"Error creating verification session: {str(e)}")
            raise DatabaseError(f"Failed to create session: {str(e)}")

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
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(VerificationSession).where(
                        VerificationSession.id == session_id
                    )
                )
                session_obj = result.scalar_one_or_none()

                if session_obj:
                    return self._session_to_dict(session_obj)
                return None

        except Exception as e:
            logger.error(f"Error getting verification session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to get session: {str(e)}")

    # Statistics and analytics

    async def get_total_requests_count(
        self, date_from: datetime, date_to: datetime
    ) -> int:
        """
        Подсчет общего количества запросов.

        Args:
            date_from: Начальная дата
            date_to: Конечная дата

        Returns:
            int: Количество запросов
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(func.count(VerificationSession.id)).where(
                        and_(
                            VerificationSession.created_at >= date_from,
                            VerificationSession.created_at <= date_to,
                        )
                    )
                )
                return result.scalar() or 0

        except Exception as e:
            logger.error(f"Error getting total requests count: {str(e)}")
            return 0

    async def get_successful_requests_count(
        self, date_from: datetime, date_to: datetime
    ) -> int:
        """
        Подсчет успешных запросов.

        Args:
            date_from: Начальная дата
            date_to: Конечная дата

        Returns:
            int: Количество успешных запросов
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(func.count(VerificationSession.id)).where(
                        and_(
                            VerificationSession.created_at >= date_from,
                            VerificationSession.created_at <= date_to,
                            VerificationSession.status == "completed",
                        )
                    )
                )
                return result.scalar() or 0

        except Exception as e:
            logger.error(f"Error getting successful requests count: {str(e)}")
            return 0

    async def get_failed_requests_count(
        self, date_from: datetime, date_to: datetime
    ) -> int:
        """
        Подсчет неуспешных запросов.

        Args:
            date_from: Начальная дата
            date_to: Конечная дата

        Returns:
            int: Количество неуспешных запросов
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(func.count(VerificationSession.id)).where(
                        and_(
                            VerificationSession.created_at >= date_from,
                            VerificationSession.created_at <= date_to,
                            VerificationSession.status.in_(["failed", "expired"]),
                        )
                    )
                )
                return result.scalar() or 0

        except Exception as e:
            logger.error(f"Error getting failed requests count: {str(e)}")
            return 0

    async def get_average_response_time(
        self, date_from: datetime, date_to: datetime
    ) -> float:
        """
        Подсчет среднего времени ответа.

        Args:
            date_from: Начальная дата
            date_to: Конечная дата

        Returns:
            float: Среднее время ответа в секундах
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(func.avg(VerificationSession.processing_time)).where(
                        and_(
                            VerificationSession.created_at >= date_from,
                            VerificationSession.created_at <= date_to,
                            VerificationSession.processing_time.is_not(None),
                        )
                    )
                )
                avg_time = result.scalar()
                return float(avg_time) if avg_time else 0.0

        except Exception as e:
            logger.error(f"Error getting average response time: {str(e)}")
            return 0.0

    async def get_verification_stats(
        self, date_from: datetime, date_to: datetime
    ) -> Dict[str, int]:
        """
        Получение статистики верификации.

        Args:
            date_from: Начальная дата
            date_to: Конечная дата

        Returns:
            Dict[str, int]: Статистика верификации
        """
        try:
            async with self.db_session() as session:
                # Подсчет по типам сессий
                result = await session.execute(
                    select(
                        VerificationSession.session_type,
                        func.count(VerificationSession.id),
                    )
                    .where(
                        and_(
                            VerificationSession.created_at >= date_from,
                            VerificationSession.created_at <= date_to,
                            VerificationSession.session_type.in_(
                                ["verification", "liveness"]
                            ),
                        )
                    )
                    .group_by(VerificationSession.session_type)
                )

                stats = {}
                for session_type, count in result.all():
                    stats[session_type] = count

                return stats

        except Exception as e:
            logger.error(f"Error getting verification stats: {str(e)}")
            return {}

    async def get_liveness_stats(
        self, date_from: datetime, date_to: datetime
    ) -> Dict[str, int]:
        """
        Получение статистики проверки живости.

        Args:
            date_from: Начальная дата
            date_to: Конечная дата

        Returns:
            Dict[str, int]: Статистика проверки живости
        """
        try:
            # Используем общую статистику верификации
            verification_stats = await self.get_verification_stats(date_from, date_to)
            return verification_stats.get("liveness", {})

        except Exception as e:
            logger.error(f"Error getting liveness stats: {str(e)}")
            return {}

    async def get_user_stats(
        self, date_from: datetime, date_to: datetime
    ) -> Dict[str, Any]:
        """
        Получение статистики по пользователям.

        Args:
            date_from: Начальная дата
            date_to: Конечная дата

        Returns:
            Dict[str, Any]: Статистика по пользователям
        """
        try:
            async with self.db_session() as session:
                # Общее количество пользователей
                total_users = await session.scalar(select(func.count(User.id)))

                # Новые пользователи за период
                new_users = await session.scalar(
                    select(func.count(User.id)).where(
                        and_(User.created_at >= date_from, User.created_at <= date_to)
                    )
                )

                # Активные пользователи (с запросами в период)
                active_users = await session.scalar(
                    select(
                        func.count(func.distinct(VerificationSession.user_id))
                    ).where(
                        and_(
                            VerificationSession.created_at >= date_from,
                            VerificationSession.created_at <= date_to,
                            VerificationSession.user_id.is_not(None),
                        )
                    )
                )

                return {
                    "total_users": total_users,
                    "new_users": new_users,
                    "active_users": active_users,
                }

        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return {}

    async def get_references_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по эталонам.

        Returns:
            Dict[str, Any]: Статистика по эталонам
        """
        try:
            async with self.db_session() as session:
                # Общее количество эталонов
                total_references = await session.scalar(
                    select(func.count(Reference.id))
                )

                # Активные эталоны
                active_references = await session.scalar(
                    select(func.count(Reference.id)).where(Reference.is_active == True)
                )

                # Неактивные эталоны
                inactive_references = total_references - active_references

                # Среднее качество
                avg_quality = await session.scalar(
                    select(func.avg(Reference.quality_score)).where(
                        Reference.quality_score.is_not(None)
                    )
                )

                # Общее количество использований
                total_usage = await session.scalar(
                    select(func.sum(Reference.usage_count))
                )

                return {
                    "total_references": total_references,
                    "active_references": active_references,
                    "inactive_references": inactive_references,
                    "average_quality": float(avg_quality) if avg_quality else 0.0,
                    "total_usage_count": total_usage or 0,
                    "quality_distribution": {
                        "excellent": await self._get_references_by_quality_range(
                            0.9, 1.0
                        ),
                        "good": await self._get_references_by_quality_range(0.7, 0.9),
                        "fair": await self._get_references_by_quality_range(0.5, 0.7),
                        "poor": await self._get_references_by_quality_range(0.0, 0.5),
                    },
                }

        except Exception as e:
            logger.error(f"Error getting references statistics: {str(e)}")
            return {}

    async def get_sessions_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по сессиям.

        Returns:
            Dict[str, Any]: Статистика по сессиям
        """
        try:
            async with self.db_session() as session:
                # Общее количество сессий
                total_sessions = await session.scalar(
                    select(func.count(VerificationSession.id))
                )

                # Сессии по статусам
                status_counts = {}
                for status in [
                    "pending",
                    "processing",
                    "completed",
                    "failed",
                    "expired",
                    "cancelled",
                ]:
                    count = await session.scalar(
                        select(func.count(VerificationSession.id)).where(
                            VerificationSession.status == status
                        )
                    )
                    status_counts[status] = count

                # Сессии по типам
                type_counts = {}
                for session_type in [
                    "verification",
                    "liveness",
                    "enrollment",
                    "identification",
                ]:
                    count = await session.scalar(
                        select(func.count(VerificationSession.id)).where(
                            VerificationSession.session_type == session_type
                        )
                    )
                    type_counts[session_type] = count

                # Среднее время обработки
                avg_processing_time = await session.scalar(
                    select(func.avg(VerificationSession.processing_time)).where(
                        VerificationSession.processing_time.is_not(None)
                    )
                )

                # Процент успешных верификаций
                successful_verifications = await session.scalar(
                    select(func.count(VerificationSession.id)).where(
                        and_(
                            VerificationSession.session_type == "verification",
                            VerificationSession.status == "completed",
                        )
                    )
                )

                total_verifications = type_counts.get("verification", 0)
                success_rate = (
                    (successful_verifications / total_verifications * 100)
                    if total_verifications > 0
                    else 0
                )

                return {
                    "total_sessions": total_sessions,
                    "active_sessions": status_counts.get("pending", 0)
                    + status_counts.get("processing", 0),
                    "completed_sessions": status_counts.get("completed", 0),
                    "failed_sessions": status_counts.get("failed", 0),
                    "expired_sessions": status_counts.get("expired", 0),
                    "average_processing_time": (
                        float(avg_processing_time) if avg_processing_time else 0.0
                    ),
                    "success_rate": success_rate,
                    "liveness_success_rate": 0.0,  # TODO: реализовать
                    "sessions_by_type": type_counts,
                    "sessions_by_status": status_counts,
                    "hourly_distribution": await self._get_hourly_distribution(),
                }

        except Exception as e:
            logger.error(f"Error getting sessions statistics: {str(e)}")
            return {}

    # Health check

    async def health_check(self) -> bool:
        """
        Проверка состояния базы данных.

        Returns:
            bool: True если БД доступна
        """
        try:
            async for session in self.db_session():
                await session.execute(select(1))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    # Helper methods

    def _user_to_dict(self, user: User) -> Dict[str, Any]:
        """
        Преобразование User модели в словарь.

        Args:
            user: User объект

        Returns:
            Dict[str, Any]: Данные пользователя
        """
        return {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login": user.last_login,
            "metadata": user.metadata,
            "total_uploads": user.total_uploads,
            "total_verifications": user.total_verifications,
            "successful_verifications": user.successful_verifications,
            "settings": user.settings,
        }

    def _reference_to_dict(self, reference: Reference) -> Dict[str, Any]:
        """
        Преобразование Reference модели в словарь.

        Args:
            reference: Reference объект

        Returns:
            Dict[str, Any]: Данные эталона
        """
        return {
            "id": str(reference.id),
            "user_id": str(reference.user_id) if reference.user_id else None,
            "label": reference.label,
            "file_url": reference.file_url,
            "file_size": reference.file_size,
            "image_format": reference.image_format,
            "image_dimensions": reference.image_dimensions,
            "embedding": reference.embedding,  # Зашифрованный эмбеддинг
            "embedding_version": reference.embedding_version,
            "quality_score": reference.quality_score,
            "is_active": reference.is_active,
            "created_at": reference.created_at,
            "updated_at": reference.updated_at,
            "last_used": reference.last_used,
            "usage_count": reference.usage_count,
            "metadata": reference.metadata,
            "original_filename": reference.original_filename,
            "checksum": reference.checksum,
            "processing_time": reference.processing_time,
        }

    def _session_to_dict(self, session_obj: VerificationSession) -> Dict[str, Any]:
        """
        Преобразование VerificationSession модели в словарь.

        Args:
            session_obj: VerificationSession объект

        Returns:
            Dict[str, Any]: Данные сессии
        """
        return {
            "id": str(session_obj.id),
            "user_id": str(session_obj.user_id) if session_obj.user_id else None,
            "session_type": session_obj.session_type,
            "status": session_obj.status,
            "reference_id": (
                str(session_obj.reference_id) if session_obj.reference_id else None
            ),
            "request_data": session_obj.request_data,
            "response_data": session_obj.response_data,
            "created_at": session_obj.created_at,
            "started_at": session_obj.started_at,
            "completed_at": session_obj.completed_at,
            "expires_at": session_obj.expires_at,
            "metadata": session_obj.metadata,
            "ip_address": session_obj.ip_address,
            "user_agent": session_obj.user_agent,
            "processing_time": session_obj.processing_time,
            "error_message": session_obj.error_message,
        }

    def _apply_user_filters(self, query, filters: Dict[str, Any]):
        """
        Применение фильтров к запросу пользователей.

        Args:
            query: SQLAlchemy запрос
            filters: Фильтры

        Returns:
            query: Отфильтрованный запрос
        """
        if "role" in filters:
            query = query.where(User.role == filters["role"])

        if "is_active" in filters:
            query = query.where(User.is_active == filters["is_active"])

        if "search" in filters:
            search_term = f"%{filters['search']}%"
            query = query.where(
                or_(
                    User.username.ilike(search_term),
                    User.email.ilike(search_term),
                    User.first_name.ilike(search_term),
                    User.last_name.ilike(search_term),
                )
            )

        return query

    def _apply_reference_filters(self, query, filters: Dict[str, Any]):
        """
        Применение фильтров к запросу эталонов.

        Args:
            query: SQLAlchemy запрос
            filters: Фильтры

        Returns:
            query: Отфильтрованный запрос
        """
        if "user_id" in filters:
            query = query.where(Reference.user_id == filters["user_id"])

        if "label" in filters:
            query = query.where(Reference.label.ilike(f"%{filters['label']}%"))

        if "is_active" in filters:
            query = query.where(Reference.is_active == filters["is_active"])

        if "quality_min" in filters:
            query = query.where(Reference.quality_score >= filters["quality_min"])

        if "quality_max" in filters:
            query = query.where(Reference.quality_score <= filters["quality_max"])

        return query

    async def _get_references_by_quality_range(
        self, min_quality: float, max_quality: float
    ) -> int:
        """
        Подсчет эталонов в диапазоне качества.

        Args:
            min_quality: Минимальное качество
            max_quality: Максимальное качество

        Returns:
            int: Количество эталонов
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(func.count(Reference.id)).where(
                        and_(
                            Reference.quality_score >= min_quality,
                            Reference.quality_score < max_quality,
                        )
                    )
                )
                return result.scalar() or 0
        except Exception:
            return 0

    async def _get_hourly_distribution(self) -> Dict[str, int]:
        """
        Получение распределения сессий по часам.

        Returns:
            Dict[str, int]: Распределение по часам
        """
        try:
            async with self.db_session() as session:
                result = await session.execute(
                    select(
                        func.extract("hour", VerificationSession.created_at),
                        func.count(VerificationSession.id),
                    ).group_by(func.extract("hour", VerificationSession.created_at))
                )

                distribution = {}
                for hour, count in result.all():
                    distribution[f"{int(hour):02d}"] = count

                return distribution
        except Exception:
            return {}

    async def create_audit_log(self, audit_data: Dict[str, Any]) -> bool:
        """
        Создание записи в журнале аудита.

        Args:
            audit_data: Данные для записи аудита

        Returns:
            bool: True если запись создана успешно
        """
        try:
            async for session in self.db_session():
                audit_log = AuditLog(**audit_data)
                session.add(audit_log)
                await session.commit()
                logger.info(f"Audit log created successfully: {audit_log.id}")
                return True
        except Exception as e:
            logger.error(f"Error creating audit log: {str(e)}")
            return False
