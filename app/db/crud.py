from sqlalchemy import select, update, delete, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import logging

# Импортируем модели из правильного места (как мы определили в прошлом шаге)
from app.db.models import (
    User, Reference, VerificationSession, AuditLog,
    VerificationStatus
)
# Импортируем Pydantic схемы из правильного места
from app.models.user import UserCreate, UserUpdate

logger = logging.getLogger(__name__)

# ============================================================================
# User CRUD (Async)
# ============================================================================

class UserCRUD:
    @staticmethod
    async def get_user(db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID asynchronously"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_phone(db: AsyncSession, phone: str) -> Optional[User]:
        """Get user by phone number"""
        result = await db.execute(select(User).where(User.phone == phone))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
        result = await db.execute(select(User).offset(skip).limit(limit))
        return list(result.scalars().all())

    @staticmethod
    async def count_users(db: AsyncSession) -> int:
        """Count total users"""
        result = await db.execute(select(func.count()).select_from(User))
        return result.scalar_one()

    @staticmethod
    async def create_user(db: AsyncSession, user: UserCreate) -> User:
        try:
            new_user = User(
                email=user.email,
                phone=user.phone,
                full_name=user.full_name
            )
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            logger.info(f"✅ User created: {new_user.id}")
            return new_user
        except IntegrityError:
            await db.rollback()
            logger.error(f"❌ User creation failed: Email/Phone already exists")
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error creating user: {e}")
            raise
    
    @staticmethod
    async def update_user(db: AsyncSession, user_id: str, user_update: UserUpdate) -> Optional[User]:
        try:
            # Получаем пользователя
            user = await UserCRUD.get_user(db, user_id)
            if not user:
                return None
            
            # Pydantic v2 syntax: model_dump вместо dict
            update_data = user_update.model_dump(exclude_unset=True)
            
            for field, value in update_data.items():
                setattr(user, field, value)
            
            await db.commit()
            await db.refresh(user)
            return user
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error updating user: {e}")
            raise

    @staticmethod
    async def deactivate_user(db: AsyncSession, user_id: str) -> Optional[User]:
        """Deactivate user (soft delete)"""
        try:
            user = await UserCRUD.get_user(db, user_id)
            if not user:
                return None
            
            user.is_active = False
            await db.commit()
            await db.refresh(user)
            logger.info(f"✅ User deactivated: {user_id}")
            return user
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error deactivating user: {e}")
            raise

    @staticmethod
    async def delete_user(db: AsyncSession, user_id: str) -> bool:
        try:
            # SQLAlchemy 2.0 delete syntax
            stmt = delete(User).where(User.id == user_id)
            result = await db.execute(stmt)
            await db.commit()
            
            if result.rowcount > 0:
                logger.info(f"✅ User deleted: {user_id}")
                return True
            return False
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error deleting user: {e}")
            raise

# ============================================================================
# Reference CRUD (Async)
# ============================================================================

class ReferenceCRUD:
    @staticmethod
    async def get_reference_by_id(db: AsyncSession, reference_id: str) -> Optional[Reference]:
        """Get reference by ID"""
        result = await db.execute(select(Reference).where(Reference.id == reference_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_latest_reference(db: AsyncSession, user_id: str) -> Optional[Reference]:
        stmt = (
            select(Reference)
            .where(Reference.user_id == user_id)
            .order_by(desc(Reference.version))
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_reference_by_version(db: AsyncSession, user_id: str, version: int) -> Optional[Reference]:
        """Get reference by specific version"""
        stmt = (
            select(Reference)
            .where(
                and_(
                    Reference.user_id == user_id,
                    Reference.version == version
                )
            )
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_references(db: AsyncSession, user_id: str) -> List[Reference]:
        """Get all references for user"""
        stmt = (
            select(Reference)
            .where(Reference.user_id == user_id)
            .order_by(desc(Reference.version))
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def create_reference(
        db: AsyncSession,
        user_id: str,
        embedding: str,
        embedding_encrypted: bytes,
        embedding_hash: str,
        quality_score: float,
        image_filename: str,
        image_size_mb: float,
        image_format: str,
        file_url: str = None,
        face_landmarks: dict = None
    ) -> Reference:
        try:
            # Определяем версию
            latest = await ReferenceCRUD.get_latest_reference(db, user_id)
            version = (latest.version + 1) if latest else 1
            
            new_ref = Reference(
                user_id=user_id,
                embedding=embedding,
                embedding_encrypted=embedding_encrypted,
                embedding_hash=embedding_hash,
                quality_score=quality_score,
                image_filename=image_filename,
                image_size_mb=image_size_mb,
                image_format=image_format,
                file_url=file_url or f"file://{image_filename}",
                face_landmarks=face_landmarks,
                version=version,
                previous_reference_id=latest.id if latest else None
            )
            
            db.add(new_ref)
            await db.commit()
            await db.refresh(new_ref)
            logger.info(f"✅ Reference created: {new_ref.id}, version: {version}")
            return new_ref
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error creating reference: {e}")
            raise

    @staticmethod
    async def update_reference(db: AsyncSession, reference_id: str, **kwargs) -> Optional[Reference]:
        """Update reference"""
        try:
            stmt = select(Reference).where(Reference.id == reference_id)
            result = await db.execute(stmt)
            ref = result.scalar_one_or_none()
            
            if not ref:
                return None
            
            for field, value in kwargs.items():
                if hasattr(ref, field):
                    setattr(ref, field, value)
            
            await db.commit()
            await db.refresh(ref)
            logger.info(f"✅ Reference updated: {reference_id}")
            return ref
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error updating reference: {e}")
            raise

    @staticmethod
    async def delete_reference(db: AsyncSession, reference_id: str) -> bool:
        """Delete reference"""
        try:
            stmt = delete(Reference).where(Reference.id == reference_id)
            result = await db.execute(stmt)
            await db.commit()
            
            if result.rowcount > 0:
                logger.info(f"✅ Reference deleted: {reference_id}")
                return True
            return False
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error deleting reference: {e}")
            raise

# ============================================================================
# Verification Session CRUD (Async)
# ============================================================================

class VerificationSessionCRUD:
    @staticmethod
    async def create_session(
        db: AsyncSession,
        user_id: str,
        session_id: str,
        image_filename: str,
        image_size_mb: float,
        expires_at: datetime = None
    ) -> VerificationSession:
        try:
            from datetime import datetime, timedelta
            # Устанавливаем expires_at на 30 минут вперед, если не указан
            if expires_at is None:
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
                
            session = VerificationSession(
                user_id=user_id,
                session_id=session_id,
                image_filename=image_filename,
                image_size_mb=image_size_mb,
                expires_at=expires_at,
                status=VerificationStatus.PENDING
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            logger.info(f"✅ Verification session created: {session.id}")
            return session
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error creating session: {e}")
            raise

    @staticmethod
    async def get_session_by_sid(db: AsyncSession, session_id: str) -> Optional[VerificationSession]:
        result = await db.execute(select(VerificationSession).where(VerificationSession.session_id == session_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_session(db: AsyncSession, session_id: str) -> Optional[VerificationSession]:
        """Alias for get_session_by_sid"""
        return await VerificationSessionCRUD.get_session_by_sid(db, session_id)

    @staticmethod
    async def get_user_sessions(db: AsyncSession, user_id: str, limit: int = 100) -> List[VerificationSession]:
        """Get all sessions for user"""
        stmt = (
            select(VerificationSession)
            .where(VerificationSession.user_id == user_id)
            .order_by(desc(VerificationSession.created_at))
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def update_session(db: AsyncSession, session_id: str, **kwargs) -> Optional[VerificationSession]:
        """Update session"""
        try:
            stmt = (
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(**kwargs)
                .execution_options(synchronize_session="fetch")
            )
            await db.execute(stmt)
            await db.commit()
            
            # Return updated object
            return await VerificationSessionCRUD.get_session_by_sid(db, session_id)
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error updating session: {e}")
            raise

    @staticmethod
    async def complete_session(
        db: AsyncSession,
        session_id: str,
        is_match: bool,
        similarity_score: float,
        confidence: float,
        **kwargs
    ) -> Optional[VerificationSession]:
        """Complete session with results"""
        try:
            stmt = (
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(
                    status=VerificationStatus.SUCCESS,
                    is_match=is_match,
                    similarity_score=similarity_score,
                    confidence=confidence,
                    completed_at=datetime.now(timezone.utc),
                    **kwargs
                )
                .execution_options(synchronize_session="fetch")
            )
            await db.execute(stmt)
            await db.commit()
            
            # Возвращаем обновленный объект
            return await VerificationSessionCRUD.get_session_by_sid(db, session_id)
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error completing session: {e}")
            raise

    @staticmethod
    async def fail_session(
        db: AsyncSession,
        session_id: str,
        error_code: str = None,
        error_message: str = None
    ) -> Optional[VerificationSession]:
        """Mark session as failed"""
        try:
            stmt = (
                update(VerificationSession)
                .where(VerificationSession.session_id == session_id)
                .values(
                    status=VerificationStatus.FAILED,
                    error_code=error_code,
                    error_message=error_message,
                    completed_at=datetime.now(timezone.utc)
                )
                .execution_options(synchronize_session="fetch")
            )
            await db.execute(stmt)
            await db.commit()
            return await VerificationSessionCRUD.get_session_by_sid(db, session_id)
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error failing session: {e}")
            raise

    @staticmethod
    async def cleanup_old_sessions(db: AsyncSession, days: int = 30) -> int:
        """Delete sessions older than N days"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = delete(VerificationSession).where(
                VerificationSession.created_at < cutoff_date
            )
            result = await db.execute(stmt)
            await db.commit()
            
            deleted_count = result.rowcount
            logger.info(f"✅ Cleaned up {deleted_count} old sessions")
            return deleted_count
        except Exception as e:
            await db.rollback()
            logger.error(f"❌ Error cleaning up sessions: {e}")
            raise

# ============================================================================
# Audit Log CRUD (Async)
# ============================================================================

class AuditLogCRUD:
    @staticmethod
    async def log_action(
        db: AsyncSession,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: Optional[str] = None,
        description: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Optional[AuditLog]:
        """Fire-and-forget style logging (safe)"""
        try:
            log = AuditLog(
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                description=description,
                old_values=old_values,
                new_values=new_values,
                ip_address=ip_address,
                user_agent=user_agent,
                success=success,
                error_message=error_message
            )
            db.add(log)
            await db.commit()
            logger.info(f"✅ Audit log created: {action} on {resource_type}")
            return log
        except Exception as e:
            # Не роняем приложение, если не записался лог
            logger.error(f"❌ Failed to write audit log: {e}")
            await db.rollback()
            return None

    @staticmethod
    async def get_logs(db: AsyncSession, user_id: Optional[str] = None, action: Optional[str] = None, limit: int = 100) -> List[AuditLog]:
        """Get audit logs with filters"""
        stmt = select(AuditLog)
        
        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        
        stmt = stmt.order_by(desc(AuditLog.created_at)).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_resource_logs(db: AsyncSession, resource_type: str, resource_id: str) -> List[AuditLog]:
        """Get all logs for specific resource"""
        stmt = (
            select(AuditLog)
            .where(
                and_(
                    AuditLog.resource_type == resource_type,
                    AuditLog.resource_id == resource_id
                )
            )
            .order_by(desc(AuditLog.created_at))
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())