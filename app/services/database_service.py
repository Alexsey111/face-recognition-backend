from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import logging

from app.config import settings

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.crud import (
    UserCRUD,
    ReferenceCRUD,
    VerificationSessionCRUD,
    AuditLogCRUD,
)
from app.db.models import (
    User,
    Reference,
    VerificationSession,
    VerificationStatus,
    AuditLog,
)
from app.models.user import UserCreate, UserUpdate

logger = logging.getLogger(__name__)
from app.services.cache_service import CacheService


class BiometricService:
    """
    High-level business service for biometric system.
    All operations are transactional and async-safe.
    """

    def __init__(self, db: Optional[AsyncSession] = None):
        self._db = db

    @property
    def db(self) -> AsyncSession:
        """Lazy initialization of database session."""
        if self._db is None:
            raise RuntimeError(
                "Database session is required. Use context manager: "
                "async with get_db() as db: ..."
            )
        return self._db

    def set_session(self, db: AsyncSession):
        """Set database session."""
        self._db = db

    # =========================================================================
    # User operations
    # =========================================================================

    async def create_user_with_audit(
        self,
        user: UserCreate,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> User:
        # Используем существующую транзакцию, не создаём новую
        new_user = await UserCRUD.create_user(self.db, user)

        await AuditLogCRUD.log_action(
            self.db,
            action="user_created",
            resource_type="user",
            resource_id=new_user.id,
            user_id=new_user.id,
            description=f"User created: {new_user.email}",
            new_values=user.model_dump(exclude_none=True),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return new_user

    async def update_user_with_audit(
        self,
        user_id: str,
        user_update: UserUpdate,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[User]:
        # Используем существующую транзакцию, не создаём новую
        user = await UserCRUD.get_user(self.db, user_id)
        if not user:
            return None

        old_values = {
            "email": user.email,
            "phone": user.phone,
            "full_name": user.full_name,
        }

        updated_user = await UserCRUD.update_user(
            self.db,
            user_id,
            user_update,
        )

        await AuditLogCRUD.log_action(
            self.db,
            action="user_updated",
            resource_type="user",
            resource_id=user_id,
            user_id=user_id,
            description="User updated",
            old_values=old_values,
            new_values=user_update.model_dump(exclude_unset=True),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return updated_user

    async def delete_user_with_audit(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> bool:
        # Используем существующую транзакцию, не создаём новую
        user = await UserCRUD.get_user(self.db, user_id)
        if not user:
            return False

        old_values = {
            "email": user.email,
            "is_active": user.is_active,
        }

        success = await UserCRUD.delete_user(self.db, user_id)

        if success:
            await AuditLogCRUD.log_action(
                self.db,
                action="user_deleted",
                resource_type="user",
                resource_id=user_id,
                description="User deleted",
                old_values=old_values,
                ip_address=ip_address,
                user_agent=user_agent,
            )

        return success

    # =========================================================================
    # Reference operations
    # =========================================================================

    async def create_reference_with_audit(
        self,
        user_id: str,
        embedding_encrypted: bytes,
        embedding_hash: str,
        quality_score: float,
        image_filename: str,
        image_size_mb: float,
        image_format: str,
        face_landmarks: Optional[dict] = None,
    ) -> Reference:
        # Используем существующую транзакцию, не создаём новую
        ref = await ReferenceCRUD.create_reference(
            self.db,
            user_id=user_id,
            embedding_encrypted=embedding_encrypted,
            embedding_hash=embedding_hash,
            quality_score=quality_score,
            image_filename=image_filename,
            image_size_mb=image_size_mb,
            image_format=image_format,
            face_landmarks=face_landmarks,
        )

        await AuditLogCRUD.log_action(
            self.db,
            action="reference_created",
            resource_type="reference",
            resource_id=ref.id,
            user_id=user_id,
            description="Reference created",
            new_values={
                "version": ref.version,
                "quality_score": ref.quality_score,
                "image_format": image_format,
            },
        )

        # Cache latest reference metadata for faster retrieval
        try:
            cache = CacheService()
            await cache.set(
                f"{settings.CACHE_KEY_PREFIX}user:{user_id}:reference:latest",
                {
                    "id": ref.id,
                    "version": ref.version,
                    "quality_score": ref.quality_score,
                    "image_filename": ref.image_filename,
                    "image_format": ref.image_format,
                },
                expire_seconds=3600,
            )
        except Exception:
            logger.warning("Failed to cache latest reference")

        return ref

    async def get_latest_reference_cached(self, user_id: str) -> Optional[Reference]:
        """Try to get latest reference from Redis cache, fallback to DB and populate cache."""
        try:
            cache = CacheService()
            cached = await cache.get(f"{settings.CACHE_KEY_PREFIX}user:{user_id}:reference:latest")
            if cached:
                # ✅ Кешируем только ID, загружаем полный объект из DB
                ref_id = cached.get("id")
                if ref_id:
                    return await ReferenceCRUD.get_reference_by_id(self.db, ref_id)
        except Exception:
            logger.debug("Cache unavailable or error while reading latest reference")

        # Fallback to DB
        ref = await ReferenceCRUD.get_latest_reference(self.db, user_id)
        if ref:
            try:
                cache = CacheService()
                # ✅ Кешируем только ID для быстрого доступа
                await cache.set(
                    f"{settings.CACHE_KEY_PREFIX}user:{user_id}:reference:latest",
                    {"id": ref.id},
                    expire_seconds=3600,
                )
            except Exception:
                logger.debug("Failed to cache latest reference after DB read")

        return ref

    # =========================================================================
    # Verification session operations
    # =========================================================================

    async def create_verification_session_with_audit(
        self,
        user_id: str,
        session_id: str,
        image_filename: str,
        image_size_mb: float,
    ) -> VerificationSession:
        # Используем существующую транзакцию, не создаём новую
        session = await VerificationSessionCRUD.create_session(
            self.db,
            user_id=user_id,
            session_id=session_id,
            image_filename=image_filename,
            image_size_mb=image_size_mb,
        )

        await AuditLogCRUD.log_action(
            self.db,
            action="verification_started",
            resource_type="verification_session",
            resource_id=session.id,
            user_id=user_id,
            description="Verification started",
        )

        return session

    async def complete_verification_with_audit(
        self,
        session_id: str,
        is_match: bool,
        similarity_score: float,
        confidence: float,
        is_liveness_passed: Optional[bool] = None,
        liveness_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
    ) -> Optional[VerificationSession]:
        # Используем существующую транзакцию, не создаём новую
        session = await VerificationSessionCRUD.complete_session(
            self.db,
            session_id=session_id,
            is_match=is_match,
            similarity_score=similarity_score,
            confidence=confidence,
            is_liveness_passed=is_liveness_passed,
            liveness_score=liveness_score,
            processing_time_ms=processing_time_ms,
        )

        if not session:
            return None

        await AuditLogCRUD.log_action(
            self.db,
            action="verification_completed",
            resource_type="verification_session",
            resource_id=session.id,
            user_id=session.user_id,
            description="Verification completed",
            new_values={
                "is_match": is_match,
                "similarity_score": similarity_score,
                "confidence": confidence,
                "is_liveness_passed": is_liveness_passed,
            },
        )

        user = await UserCRUD.get_user(self.db, session.user_id)
        if user:
            user.last_verified_at = datetime.now(timezone.utc)
            self.db.add(user)

        return session

    async def create_verification(
        self,
        user_id: str,
        similarity: float,
        confidence: float,
        liveness: float,
        threshold: float,
        verified: bool,
        request_id: str,
    ) -> VerificationSession:
        """Создание записи о верификации с сохранением threshold_used."""
        # Создаём сессию верификации
        import uuid
        session_id = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        
        session = await VerificationSessionCRUD.create_session(
            self.db,
            user_id=user_id,
            session_id=session_id,
            image_filename=f"verification_{request_id}.jpg",
            image_size_mb=0.0,
            expires_at=expires_at,
        )

        # Обновляем результаты верификации
        session = await VerificationSessionCRUD.complete_session(
            self.db,
            session_id=session.session_id,
            is_match=verified,
            similarity_score=similarity,
            confidence=confidence,
            is_liveness_passed=liveness >= 0.5 if liveness > 0 else None,
            liveness_score=liveness,
            processing_time_ms=None,
            threshold=threshold,  # Сохраняем threshold_used
        )

        return session

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = await UserCRUD.get_user(self.db, user_id)
        if not user:
            return None

        total_sessions = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.user_id == user_id
            )
        )

        successful_sessions = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.user_id == user_id,
                VerificationSession.status == VerificationStatus.SUCCESS,
                VerificationSession.is_match.is_(True),
            )
        )

        total_references = await self.db.scalar(
            select(func.count()).where(Reference.user_id == user_id)
        )

        return {
            "user_id": user.id,
            "email": user.email,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "last_verified_at": user.last_verified_at,
            "total_references": total_references,
            "total_sessions": total_sessions,
            "successful_verifications": successful_sessions,
            "success_rate": (
                successful_sessions / total_sessions
                if total_sessions else 0
            ),
        }

    async def get_system_stats(self) -> Dict[str, Any]:
        total_users = await self.db.scalar(select(func.count()).select_from(User))
        active_users = await self.db.scalar(
            select(func.count()).where(User.is_active.is_(True))
        )
        total_references = await self.db.scalar(
            select(func.count()).select_from(Reference)
        )
        total_sessions = await self.db.scalar(
            select(func.count()).select_from(VerificationSession)
        )
        successful_sessions = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status == VerificationStatus.SUCCESS
            )
        )

        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_references": total_references,
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": (
                successful_sessions / total_sessions
                if total_sessions else 0
            ),
            "timestamp": datetime.now(timezone.utc),
        }

    # =========================================================================
    # Auth compatibility (explicitly isolated)
    # =========================================================================

    async def get_user_by_email(self, email: str) -> Optional[User]:
        return await UserCRUD.get_user_by_email(self.db, email)

    async def get_user(self, user_id: str) -> Optional[User]:
        return await UserCRUD.get_user(self.db, user_id)


# =========================================================================
# Admin Statistics Methods (for admin routes)
# =========================================================================

    async def get_total_requests_count(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> int:
        """Get total requests count for period."""
        result = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        )
        return result or 0

    async def get_successful_requests_count(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> int:
        """Get successful requests count for period."""
        result = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status == VerificationStatus.SUCCESS,
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        )
        return result or 0

    async def get_failed_requests_count(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> int:
        """Get failed requests count for period."""
        result = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status.in_([
                    VerificationStatus.FAILED,
                    VerificationStatus.ERROR,
                ]),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        )
        return result or 0

    async def get_average_response_time(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> float:
        """Get average response time in ms for period."""
        result = await self.db.scalar(
            select(func.avg(VerificationSession.processing_time_ms)).where(
                VerificationSession.status == VerificationStatus.SUCCESS,
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
                VerificationSession.processing_time_ms.isnot(None),
            )
        )
        return float(result) if result else 0.0

    async def get_verification_stats(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> Dict[str, Any]:
        """Get verification statistics for period."""
        total = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        ) or 0

        successful = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status == VerificationStatus.SUCCESS,
                VerificationSession.is_match.is_(True),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        ) or 0

        failed = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status.in_([
                    VerificationStatus.FAILED,
                    VerificationStatus.ERROR,
                ]),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        ) or 0

        avg_similarity = await self.db.scalar(
            select(func.avg(VerificationSession.similarity_score)).where(
                VerificationSession.similarity_score.isnot(None),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        )

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total else 0,
            "average_similarity": float(avg_similarity) if avg_similarity else None,
        }

    async def get_liveness_stats(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> Dict[str, Any]:
        """Get liveness check statistics for period."""
        total = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.is_liveness_passed.isnot(None),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        ) or 0

        passed = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.is_liveness_passed.is_(True),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        ) or 0

        failed = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.is_liveness_passed.is_(False),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        ) or 0

        avg_score = await self.db.scalar(
            select(func.avg(VerificationSession.liveness_score)).where(
                VerificationSession.liveness_score.isnot(None),
                VerificationSession.created_at >= date_from,
                VerificationSession.created_at <= date_to,
            )
        )

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total else 0,
            "average_score": float(avg_score) if avg_score else None,
        }

    async def get_user_stats(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> Dict[str, Any]:
        """Get user statistics for period."""
        total = await self.db.scalar(select(func.count()).select_from(User))

        new_users = await self.db.scalar(
            select(func.count()).where(
                User.created_at >= date_from,
                User.created_at <= date_to,
            )
        ) or 0

        active_users = await self.db.scalar(
            select(func.count()).where(User.is_active.is_(True))
        ) or 0

        verified_users = await self.db.scalar(
            select(func.count()).where(User.is_verified.is_(True))
        ) or 0

        return {
            "total_users": total,
            "new_users": new_users,
            "active_users": active_users,
            "verified_users": verified_users,
        }

    async def get_references_statistics(self) -> Dict[str, Any]:
        """Get reference statistics."""
        total = await self.db.scalar(select(func.count()).select_from(Reference))
        active = await self.db.scalar(
            select(func.count()).where(Reference.is_active.is_(True))
        )
        avg_quality = await self.db.scalar(
            select(func.avg(Reference.quality_score)).where(
                Reference.quality_score.isnot(None)
            )
        )

        return {
            "total_references": total,
            "active_references": active,
            "average_quality_score": float(avg_quality) if avg_quality else None,
        }

    async def get_sessions_statistics(self) -> Dict[str, Any]:
        """Get verification sessions statistics."""
        total = await self.db.scalar(
            select(func.count()).select_from(VerificationSession)
        )
        pending = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status == VerificationStatus.PENDING
            )
        )
        processing = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status == VerificationStatus.PROCESSING
            )
        )
        success = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status == VerificationStatus.SUCCESS
            )
        )
        failed = await self.db.scalar(
            select(func.count()).where(
                VerificationSession.status == VerificationStatus.FAILED
            )
        )

        return {
            "total_sessions": total,
            "pending_sessions": pending,
            "processing_sessions": processing,
            "successful_sessions": success,
            "failed_sessions": failed,
        }

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID (alias for admin routes)."""
        return await UserCRUD.get_user(self.db, user_id)

    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Optional[User]:
        """Update user (for admin routes)."""
        from app.models.user import UserUpdate
        user_update = UserUpdate(**user_data)
        return await UserCRUD.update_user(self.db, user_id, user_update)

    async def get_all_users(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Dict[str, Any]:
        """Get all users with pagination and filters (for admin routes)."""
        query = select(User)

        # Apply filters
        if filters:
            if filters.get("role"):
                # Filter by role if exists
                pass
            if filters.get("is_active") is not None:
                query = query.where(User.is_active == filters["is_active"])
            if filters.get("search"):
                search = f"%{filters['search']}%"
                query = query.where(
                    (User.email.ilike(search)) | (User.full_name.ilike(search))
                )

        # Sorting
        if sort_order.lower() == "desc":
            query = query.order_by(getattr(User, sort_by, User.created_at).desc())
        else:
            query = query.order_by(getattr(User, sort_by, User.created_at).asc())

        # Pagination
        skip = (page - 1) * per_page
        query = query.offset(skip).limit(per_page)

        result = await self.db.execute(query)
        users = list(result.scalars().all())

        # Get total count
        count_query = select(func.count()).select_from(User)
        if filters:
            if filters.get("is_active") is not None:
                count_query = count_query.where(User.is_active == filters["is_active"])
        total_count = await self.db.scalar(count_query) or 0

        return {
            "users": users,
            "total_count": total_count,
            "page": page,
            "per_page": per_page,
            "has_next": (page * per_page) < total_count,
            "has_prev": page > 1,
        }

    async def get_logs(
        self,
        level: Optional[str] = None,
        service: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get audit logs with filters for admin panel.

        Note: AuditLog model doesn't have 'level' or 'service' fields,
        this is a compatibility wrapper that maps actions to log levels.
        """
        query = select(AuditLog)

        if date_from:
            query = query.where(AuditLog.created_at >= date_from)
        if date_to:
            query = query.where(AuditLog.created_at <= date_to)

        query = query.order_by(AuditLog.created_at.desc()).limit(limit)

        result = await self.db.execute(query)
        logs = result.scalars().all()

        # Map action to level for compatibility
        level_map = {
            "error": "ERROR",
            "auth_failed": "WARNING",
            "user_deleted": "WARNING",
            "verification_failed": "ERROR",
        }

        return [
            {
                "id": log.id,
                "level": level_map.get(log.action, level or "INFO"),
                "service": service or "app",
                "message": log.description or log.action,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "user_id": log.user_id,
                "old_values": log.old_values,
                "new_values": log.new_values,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "success": log.success,
                "error_message": log.error_message,
                "timestamp": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ]

    async def get_error_logs(
        self,
        filters: Dict[str, Any] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Получение error logs из audit_logs.
        Args:
            filters: Фильтры (error_type, severity)
            date_from: Начальная дата
            date_to: Конечная дата
            limit: Максимальное количество записей
        Returns:
            Список error logs
        """
        from sqlalchemy import select, and_

        query = select(AuditLog).where(AuditLog.success == False)

        # Применяем фильтры
        conditions = []
        if filters:
            if "error_type" in filters:
                conditions.append(
                    AuditLog.details["error_type"].astext == filters["error_type"]
                )
            if "severity" in filters:
                conditions.append(
                    AuditLog.details["severity"].astext == filters["severity"]
                )
        if date_from:
            conditions.append(AuditLog.created_at >= date_from)
        if date_to:
            conditions.append(AuditLog.created_at <= date_to)
        if conditions:
            query = query.where(and_(*conditions))
        
        # Сортировка и лимит
        query = query.order_by(AuditLog.created_at.desc()).limit(limit)
        
        result = await self.db.execute(query)
        logs = result.scalars().all()

        return [
            {
                "id": log.id,
                "action": log.action,
                "target_type": log.target_type,
                "target_id": log.target_id,
                "user_id": log.user_id,
                "error_message": log.error_message,
                "error_type": log.details.get("error_type") if log.details else None,
                "severity": log.details.get("severity") if log.details else None,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ]

    async def get_error_logs_count(
        self,
        filters: Dict[str, Any] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> int:
        """Подсчёт error logs."""
        from sqlalchemy import select, func, and_

        query = select(func.count(AuditLog.id)).where(AuditLog.success == False)

        # Применяем фильтры
        conditions = []
        if filters:
            if "error_type" in filters:
                conditions.append(
                    AuditLog.details["error_type"].astext == filters["error_type"]
                )
            if "severity" in filters:
                conditions.append(
                    AuditLog.details["severity"].astext == filters["severity"]
                )
        if date_from:
            conditions.append(AuditLog.created_at >= date_from)
        if date_to:
            conditions.append(AuditLog.created_at <= date_to)
        if conditions:
            query = query.where(and_(*conditions))

        result = await self.db.execute(query)
        return result.scalar() or 0


# Backward compatibility alias
DatabaseService = BiometricService
