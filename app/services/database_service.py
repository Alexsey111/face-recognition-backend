from sqlalchemy.ext.asyncio import AsyncSession
from app.db.crud import (
    UserCRUD, ReferenceCRUD, VerificationSessionCRUD, AuditLogCRUD
)
from app.db.models import User, Reference, VerificationSession, VerificationStatus
# Импортируем Pydantic схемы из правильного места
from app.models.user import UserCreate, UserUpdate
from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class BiometricService:
    """
    High-level database operations with business logic,
    designed for ASYNCHRONOUS operation (SQLAlchemy 2.0).
    """
    
    def __init__(self, db: Optional[AsyncSession] = None):
        # Опциональный параметр для тестирования и инициализации
        self.db = db 
    
    # ============================================================================
    # User Operations
    # ============================================================================
    
    async def create_user_with_audit(
        self,
        user: UserCreate,
        operator_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> User:
        """Create user and log the action"""
        if not self.db:
            raise RuntimeError("Database session is required for this operation")
            
        try:
            # 1. Используем await для асинхронного CRUD
            new_user = await UserCRUD.create_user(self.db, user)
            
            # 2. Логирование (тоже await)
            await AuditLogCRUD.log_action(
                self.db,
                action="user_created",
                resource_type="user",
                resource_id=new_user.id,
                user_id=new_user.id,
                operator_id=operator_id,
                description=f"User created: {user.email}",
                new_values=user.model_dump(exclude_none=True), # Pydantic v2
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return new_user
        except Exception as e:
            logger.error(f"❌ Error creating user: {e}")
            raise
    
    async def update_user_with_audit(
        self,
        user_id: str,
        user_update: UserUpdate,
        operator_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[User]:
        """Update user and log the change"""
        if not self.db:
            raise RuntimeError("Database session is required for this operation")
            
        try:
            # 1. Получаем старые значения асинхронно
            old_user = await UserCRUD.get_user(self.db, user_id)
            if not old_user:
                return None
            
            old_values = old_user.__dict__.copy() 
            old_values_for_log = {
                "email": old_values.get("email"),
                "phone": old_values.get("phone"),
                "full_name": old_values.get("full_name")
            }
            
            # 2. Обновляем пользователя асинхронно
            updated_user = await UserCRUD.update_user(self.db, user_id, user_update)
            
            if not updated_user:
                return None
            
            # 3. Логирование с Pydantic v2 .model_dump()
            await AuditLogCRUD.log_action(
                self.db,
                action="user_updated",
                resource_type="user",
                resource_id=user_id,
                user_id=user_id,
                operator_id=operator_id,
                description=f"User updated: {user_id}",
                old_values=old_values_for_log,
                new_values=user_update.model_dump(exclude_unset=True), # Pydantic v2
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return updated_user
        except Exception as e:
            logger.error(f"❌ Error updating user: {e}")
            raise
    
    async def delete_user_with_audit(
        self,
        user_id: str,
        operator_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Delete user and log the action"""
        try:
            # Получаем пользователя перед удалением для логирования
            user = await UserCRUD.get_user(self.db, user_id)
            if not user:
                return False
            
            old_values = {
                "email": user.email,
                "is_active": user.is_active
            }
            
            # Удаляем пользователя асинхронно
            success = await UserCRUD.delete_user(self.db, user_id)
            
            if success:
                await AuditLogCRUD.log_action(
                    self.db,
                    action="user_deleted",
                    resource_type="user",
                    resource_id=user_id,
                    operator_id=operator_id,
                    description=f"User deleted: {user_id}",
                    old_values=old_values,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            
            return success
        except Exception as e:
            logger.error(f"❌ Error deleting user: {e}")
            raise
    
    # ============================================================================
    # Reference Operations
    # ============================================================================
    
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
        operator_id: Optional[str] = None
    ) -> Reference:
        """Create reference and log"""
        try:
            # Создаем Reference асинхронно
            ref = await ReferenceCRUD.create_reference(
                self.db,
                user_id=user_id,
                embedding_encrypted=embedding_encrypted,
                embedding_hash=embedding_hash,
                quality_score=quality_score,
                image_filename=image_filename,
                image_size_mb=image_size_mb,
                image_format=image_format,
                face_landmarks=face_landmarks
            )
            
            # Логирование
            await AuditLogCRUD.log_action(
                self.db,
                action="reference_created",
                resource_type="reference",
                resource_id=ref.id,
                user_id=user_id,
                operator_id=operator_id,
                description=f"Reference created for user: {user_id}",
                new_values={
                    "version": ref.version,
                    "quality_score": ref.quality_score,
                    "image_format": image_format
                }
            )
            
            return ref
        except Exception as e:
            logger.error(f"❌ Error creating reference: {e}")
            raise
    
    # ============================================================================
    # Verification Session Operations
    # ============================================================================
    
    async def create_verification_session_with_audit(
        self,
        user_id: str,
        session_id: str,
        image_filename: str,
        image_size_mb: float,
        operator_id: Optional[str] = None
    ) -> VerificationSession:
        """Create verification session and log start"""
        try:
            # Создаем сессию асинхронно
            session = await VerificationSessionCRUD.create_session(
                self.db,
                user_id=user_id,
                session_id=session_id,
                image_filename=image_filename,
                image_size_mb=image_size_mb
            )
            
            # Логируем старт
            await AuditLogCRUD.log_action(
                self.db,
                action="verification_started",
                resource_type="verification_session",
                resource_id=session.id,
                user_id=user_id,
                operator_id=operator_id,
                description=f"Verification started for user: {user_id}"
            )
            
            return session
        except Exception as e:
            logger.error(f"❌ Error creating verification session: {e}")
            raise
    
    async def complete_verification_with_audit(
        self,
        session_id: str,
        is_match: bool,
        similarity_score: float,
        confidence: float,
        is_liveness_passed: Optional[bool] = None,
        liveness_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None
    ) -> Optional[VerificationSession]:
        """Complete verification session with results and update user's last_verified_at"""
        try:
            # 1. Завершаем сессию асинхронно
            session = await VerificationSessionCRUD.complete_session(
                self.db,
                session_id=session_id,
                is_match=is_match,
                similarity_score=similarity_score,
                confidence=confidence,
                is_liveness_passed=is_liveness_passed,
                liveness_score=liveness_score,
                processing_time_ms=processing_time_ms
            )
            
            if not session:
                return None
            
            # 2. Логируем завершение
            await AuditLogCRUD.log_action(
                self.db,
                action="verification_completed",
                resource_type="verification_session",
                resource_id=session.id,
                user_id=session.user_id,
                description=f"Verification completed: is_match={is_match}",
                new_values={
                    "is_match": is_match,
                    "similarity_score": similarity_score,
                    "confidence": confidence,
                    "is_liveness_passed": is_liveness_passed
                }
            )
            
            # 3. Обновляем user.last_verified_at (бизнес-логика)
            # Так как метод complete_session обновляет сессию, мы можем обновить пользователя отдельно.
            user = await UserCRUD.get_user(self.db, session.user_id)
            if user:
                user.last_verified_at = datetime.now(timezone.utc) # Использование aware-datetime
                self.db.add(user)
                await self.db.commit() # Отдельный коммит для обновления поля
            
            return session
        except Exception as e:
            logger.error(f"❌ Error completing verification: {e}")
            raise
    
    # ============================================================================
    # Analytics & Stats (Async)
    # ============================================================================
    
    async def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for user"""
        user = await UserCRUD.get_user(self.db, user_id)
        if not user:
            return None
        
        # Асинхронные вызовы к CRUD
        references = await ReferenceCRUD.get_all_references(self.db, user_id)
        sessions = await VerificationSessionCRUD.get_user_sessions(self.db, user_id, limit=1000)
        
        successful_verifications = [
            s for s in sessions if s.status == "success" and s.is_match
        ]
        
        total_sessions = len(sessions)
        successful_count = len(successful_verifications)
        
        return {
            "user_id": user_id,
            "email": user.email,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "last_verified_at": user.last_verified_at,
            "total_references": len(references),
            "latest_reference_version": references[0].version if references else None,
            "total_sessions": total_sessions,
            "successful_verifications": successful_count,
            "success_rate": (
                successful_count / total_sessions
                if total_sessions > 0 else 0
            )
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics (using async count methods if available)"""
        
        # Используем асинхронные методы (нужно добавить их в UserCRUD/ReferenceCRUD)
        # Если их нет, нужно использовать db.execute(select(func.count())...)
        
        # ПРИМЕЧАНИЕ: Предполагаем, что в UserCRUD добавлен метод count_users_async
        # Если нет, это нужно исправить в CRUD слое, как показано в предыдущем шаге
        
        # Заглушка, использующая async count (если бы он был реализован)
        try:
            total_users = await UserCRUD.count_users(self.db)
        except AttributeError:
             # Если count_users не async, используем прямой SQLAlchemy 2.0 async call
            from sqlalchemy import select, func
            total_users_result = await self.db.execute(select(func.count()).select_from(User))
            total_users = total_users_result.scalar_one()

        # Для примера: используем прямые асинхронные запросы для остальных
        from sqlalchemy import select
        
        active_users_result = await self.db.execute(
            select(func.count()).select_from(User).where(User.is_active == True)
        )
        active_users = active_users_result.scalar_one()

        total_references_result = await self.db.execute(select(func.count()).select_from(Reference))
        total_references = total_references_result.scalar_one()

        total_sessions_result = await self.db.execute(select(func.count()).select_from(VerificationSession))
        total_sessions = total_sessions_result.scalar_one()
        
        successful_sessions_result = await self.db.execute(
            select(func.count()).select_from(VerificationSession).where(
                VerificationSession.status == "success"
            )
        )
        successful_sessions = successful_sessions_result.scalar_one()
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_references": total_references,
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": (
                successful_sessions / total_sessions if total_sessions > 0 else 0
            ),
            "timestamp": datetime.now(timezone.utc)
        }
    
    @staticmethod
    async def count_users(db: AsyncSession) -> int:
        result = await db.execute(select(func.count()).select_from(User))
        return result.scalar_one()
    
    @staticmethod
    async def get_all_references(db: AsyncSession, user_id: str) -> List[Reference]:
        result = await db.execute(select(Reference).where(Reference.user_id == user_id))
        return list(result.scalars().all())

    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """
        Simple user update for backward compatibility
        """
        if not self.db:
            raise RuntimeError("Database session is required for this operation")

        from app.models.user import UserUpdate
        user_update = UserUpdate(**update_data)
        return await self.update_user_with_audit(user_id, user_update)


# DatabaseService alias for backward compatibility
DatabaseService = BiometricService
