from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import logging
from ..config import settings

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
)
from app.models.user import UserCreate, UserUpdate

logger = logging.getLogger(__name__)
from .cache_service import CacheService


class BiometricService:
    """
    High-level business service for biometric system.
    All operations are transactional and async-safe.
    """

    def __init__(self, db: Optional[AsyncSession] = None):
        # Support optional db for backwards compatibility during imports/tests.
        # If db is None, methods that require a session should obtain one
        # from the application's DatabaseManager or be mocked in tests.
        self.db = db

    # =========================================================================
    # User operations
    # =========================================================================

    async def create_user_with_audit(
        self,
        user: UserCreate,
        operator_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> User:
        async with self.db.begin():
            new_user = await UserCRUD.create_user(self.db, user)

            await AuditLogCRUD.log_action(
                self.db,
                action="user_created",
                resource_type="user",
                resource_id=new_user.id,
                user_id=new_user.id,
                operator_id=operator_id,
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
        operator_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[User]:
        async with self.db.begin():
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
                operator_id=operator_id,
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
        operator_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> bool:
        async with self.db.begin():
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
                    operator_id=operator_id,
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
        operator_id: Optional[str] = None,
    ) -> Reference:
        async with self.db.begin():
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
                operator_id=operator_id,
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
                # minimal object-like dict
                return cached
        except Exception:
            logger.debug("Cache unavailable or error while reading latest reference")

        # Fallback to DB
        ref = await ReferenceCRUD.get_latest_reference(self.db, user_id)
        if ref:
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
        operator_id: Optional[str] = None,
    ) -> VerificationSession:
        async with self.db.begin():
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
                operator_id=operator_id,
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
        async with self.db.begin():
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

    async def get_user_by_username(self, username: str) -> Optional[User]:
        return await UserCRUD.get_user_by_username(self.db, username)

    async def get_user(self, user_id: str) -> Optional[User]:
        return await UserCRUD.get_user(self.db, user_id)


# Backward compatibility alias
DatabaseService = BiometricService

