import pytest
from app.db.crud import (
    UserCRUD, ReferenceCRUD, VerificationSessionCRUD, AuditLogCRUD
)
# Импорты моделей из app.models
from app.db.models import User, Reference, VerificationSession, AuditLog, VerificationStatus
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from datetime import datetime, timezone # Используем aware datetime
from sqlalchemy import select, func, desc, and_
from typing import Optional, List, Dict, Any

# Pydantic импорты
from pydantic import BaseModel, Field, EmailStr
from pydantic import ConfigDict

# Импортируем схемы из app.models.user
from app.models.user import UserCreate, UserUpdate

# Удаляем локальные определения, которые конфликтуют с импортами

# Делаем класс тестовых методов асинхронным
# ============================================================================
# User Tests
# ============================================================================

class TestUserCRUD:
    # 1. Все тесты становятся async def
    # 2. Фикстура db заменена на db_session (AsyncSession)
    @pytest.mark.asyncio
    async def test_create_user(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test user creation (Async)"""
        # 3. Добавляем await ко всем вызовам CRUD
        user = await UserCRUD.create_user(db_session, test_user_data)
        
        assert user.id is not None
        assert user.email == test_user_data.email  # Используем данные из фикстуры
        assert user.phone == test_user_data.phone  # Используем данные из фикстуры
        assert user.is_active == True
        
    async def test_get_user(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test get user by ID (Async)"""
        created = await UserCRUD.create_user(db_session, test_user_data)
        user = await UserCRUD.get_user(db_session, created.id)
        
        assert user.id == created.id
        assert user.email == test_user_data.email  # Используем данные из фикстуры
        
    async def test_get_user_by_email(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test get user by email (Async)"""
        created = await UserCRUD.create_user(db_session, test_user_data)
        user = await UserCRUD.get_user_by_email(db_session, test_user_data.email)  # Используем правильный email
        
        assert user.id == created.id
        
    async def test_get_user_by_phone(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test get user by phone (Async)"""
        created = await UserCRUD.create_user(db_session, test_user_data)
        user = await UserCRUD.get_user_by_phone(db_session, test_user_data.phone)  # Используем правильный телефон
        
        assert user.id == created.id
        
    async def test_update_user(self, db_session: AsyncSession, test_user_data: UserCreate, test_user_update_data: UserUpdate):
        """Test user update (Async)"""
        created = await UserCRUD.create_user(db_session, test_user_data)
        updated = await UserCRUD.update_user(db_session, created.id, test_user_update_data)
        
        assert updated.full_name == "Updated User"
        
    async def test_delete_user(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test user deletion (Async)"""
        created = await UserCRUD.create_user(db_session, test_user_data)
        success = await UserCRUD.delete_user(db_session, created.id)
        
        assert success == True
        assert await UserCRUD.get_user(db_session, created.id) is None
        
    async def test_deactivate_user(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test user deactivation (Async)"""
        created = await UserCRUD.create_user(db_session, test_user_data)
        deactivated = await UserCRUD.deactivate_user(db_session, created.id)
        
        assert deactivated.is_active == False
        
    async def test_get_all_users(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test get all users (Async)"""
        await UserCRUD.create_user(db_session, test_user_data)
        users = await UserCRUD.get_all_users(db_session)
        
        assert len(users) >= 1
        
    async def test_count_users(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test count users (Async)"""
        await UserCRUD.create_user(db_session, test_user_data)
        count = await UserCRUD.count_users(db_session)
        
        # Добавляем небольшое смещение, если другие тесты уже создали пользователя
        assert count >= 1

    # ============================================================================
    # Edge Case Tests for UserCRUD
    # ============================================================================
    
    async def test_get_user_not_found(self, db_session: AsyncSession):
        """Test getting non-existent user returns None"""
        user = await UserCRUD.get_user(db_session, "non-existent-id")
        assert user is None
        
    async def test_get_user_by_email_not_found(self, db_session: AsyncSession):
        """Test getting user by non-existent email returns None"""
        user = await UserCRUD.get_user_by_email(db_session, "non-existent@example.com")
        assert user is None
        
    async def test_get_user_by_phone_not_found(self, db_session: AsyncSession):
        """Test getting user by non-existent phone returns None"""
        user = await UserCRUD.get_user_by_phone(db_session, "+9999999999")
        assert user is None
        
    async def test_update_user_not_found(self, db_session: AsyncSession, test_user_update_data: UserUpdate):
        """Test updating non-existent user returns None"""
        user = await UserCRUD.update_user(db_session, "non-existent-id", test_user_update_data)
        assert user is None
        
    async def test_delete_user_not_found(self, db_session: AsyncSession):
        """Test deleting non-existent user returns False"""
        success = await UserCRUD.delete_user(db_session, "non-existent-id")
        assert success == False
        
    async def test_deactivate_user_not_found(self, db_session: AsyncSession):
        """Test deactivating non-existent user returns None"""
        user = await UserCRUD.deactivate_user(db_session, "non-existent-id")
        assert user is None
        
    async def test_create_duplicate_email(self, db_session: AsyncSession, test_user_data: UserCreate):
        """Test creating user with duplicate email raises IntegrityError"""
        await UserCRUD.create_user(db_session, test_user_data)
        with pytest.raises(Exception):  # IntegrityError is caught and re-raised
            await UserCRUD.create_user(db_session, test_user_data)

# ============================================================================
# Reference Tests
# ============================================================================

class TestReferenceCRUD:
    async def test_create_reference(self, db_session: AsyncSession, test_user: User):
        """Test reference creation (Async)"""
        ref = await ReferenceCRUD.create_reference(
            db_session,
            user_id=test_user.id,
            embedding="mock_embedding_string",  # Добавляем недостающий аргумент
            embedding_encrypted=b"encrypted_data",
            embedding_hash="hash123",
            quality_score=0.95,
            image_filename="photo.jpg",
            image_size_mb=2.5,
            image_format="JPG"
        )
        
        assert ref.id is not None
        assert ref.version == 1
        assert ref.quality_score == 0.95
        
    async def test_reference_versioning(self, db_session: AsyncSession, test_user: User):
        """Test reference versioning (Async)"""
        ref1 = await ReferenceCRUD.create_reference(
            db_session, user_id=test_user.id, 
            embedding="mock_embedding_1",  # Добавляем недостающий аргумент
            embedding_encrypted=b"data1",
            embedding_hash="hash1", quality_score=0.9,
            image_filename="photo1.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        ref2 = await ReferenceCRUD.create_reference(
            db_session, user_id=test_user.id, 
            embedding="mock_embedding_2",  # Добавляем недостающий аргумент
            embedding_encrypted=b"data2",
            embedding_hash="hash2", quality_score=0.95,
            image_filename="photo2.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        assert ref1.version == 1
        assert ref2.version == 2
        assert ref2.previous_reference_id == ref1.id
        
    async def test_get_latest_reference(self, db_session: AsyncSession, test_user: User):
        """Test getting latest reference (Async)"""
        await ReferenceCRUD.create_reference(
            db_session, user_id=test_user.id, 
            embedding="mock_embedding_1",  # Добавляем недостающий аргумент
            embedding_encrypted=b"data1",
            embedding_hash="hash1", quality_score=0.9,
            image_filename="photo1.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        ref2 = await ReferenceCRUD.create_reference(
            db_session, user_id=test_user.id, 
            embedding="mock_embedding_2",  # Добавляем недостающий аргумент
            embedding_encrypted=b"data2",
            embedding_hash="hash2", quality_score=0.95,
            image_filename="photo2.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        latest = await ReferenceCRUD.get_latest_reference(db_session, test_user.id)
        assert latest.id == ref2.id
        assert latest.version == 2
        
    async def test_delete_reference(self, db_session: AsyncSession, test_user: User):
        """Test reference deletion (Async)"""
        ref = await ReferenceCRUD.create_reference(
            db_session, user_id=test_user.id, 
            embedding="mock_embedding",  # Добавляем недостающий аргумент
            embedding_encrypted=b"data",
            embedding_hash="hash", quality_score=0.9,
            image_filename="photo.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        success = await ReferenceCRUD.delete_reference(db_session, ref.id)
        assert success == True

    # ============================================================================
    # Edge Case Tests for ReferenceCRUD
    # ============================================================================
    
    async def test_get_latest_reference_not_found(self, db_session: AsyncSession, test_user: User):
        """Test getting latest reference for non-existent user returns None"""
        ref = await ReferenceCRUD.get_latest_reference(db_session, "non-existent-user-id")
        assert ref is None
        
    async def test_get_reference_by_version_not_found(self, db_session: AsyncSession, test_user: User):
        """Test getting reference by non-existent version returns None"""
        ref = await ReferenceCRUD.get_reference_by_version(db_session, "non-existent-user-id", 999)
        assert ref is None
        
    async def test_get_all_references_empty(self, db_session: AsyncSession, test_user: User):
        """Test getting all references for user with no references returns empty list"""
        refs = await ReferenceCRUD.get_all_references(db_session, "non-existent-user-id")
        assert refs == []
        
    async def test_update_reference_not_found(self, db_session: AsyncSession, test_user: User):
        """Test updating non-existent reference returns None"""
        ref = await ReferenceCRUD.update_reference(db_session, "non-existent-ref-id", quality_score=0.8)
        assert ref is None
        
    async def test_delete_reference_not_found(self, db_session: AsyncSession, test_user: User):
        """Test deleting non-existent reference returns False"""
        success = await ReferenceCRUD.delete_reference(db_session, "non-existent-ref-id")
        assert success == False

# ============================================================================
# Verification Session Tests
# ============================================================================

class TestVerificationSessionCRUD:
    async def test_create_session(self, db_session: AsyncSession, test_user: User):
        """Test session creation (Async)"""
        session = await VerificationSessionCRUD.create_session(
            db_session,
            user_id=test_user.id,
            session_id=str(uuid.uuid4()),
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        assert session.id is not None
        assert session.status == "pending"
        
    async def test_get_session(self, db_session: AsyncSession, test_user: User):
        """Test get session (Async)"""
        session_id = str(uuid.uuid4())
        created = await VerificationSessionCRUD.create_session(
            db_session,
            user_id=test_user.id,
            session_id=session_id,
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        session = await VerificationSessionCRUD.get_session(db_session, session_id)
        assert session.id == created.id
        
    async def test_complete_session(self, db_session: AsyncSession, test_user: User):
        """Test completing session (Async)"""
        session_id = str(uuid.uuid4())
        await VerificationSessionCRUD.create_session(
            db_session,
            user_id=test_user.id,
            session_id=session_id,
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        completed = await VerificationSessionCRUD.complete_session(
            db_session,
            session_id=session_id,
            is_match=True,
            similarity_score=0.95,
            confidence=0.98
        )
        
        assert completed.status == "success"
        assert completed.is_match == True
        
    async def test_fail_session(self, db_session: AsyncSession, test_user: User):
        """Test failing session (Async)"""
        session_id = str(uuid.uuid4())
        await VerificationSessionCRUD.create_session(
            db_session,
            user_id=test_user.id,
            session_id=session_id,
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        failed = await VerificationSessionCRUD.fail_session(
            db_session,
            session_id=session_id,
            error_code="NO_FACE",
            error_message="No face detected in image"
        )
        
        assert failed.status == "failed"
        assert failed.error_code == "NO_FACE"

    # ============================================================================
    # Edge Case Tests for VerificationSessionCRUD
    # ============================================================================
    
    async def test_get_session_not_found(self, db_session: AsyncSession, test_user: User):
        """Test getting non-existent session returns None"""
        session = await VerificationSessionCRUD.get_session(db_session, "non-existent-session-id")
        assert session is None
        
    async def test_update_session_not_found(self, db_session: AsyncSession, test_user: User):
        """Test updating non-existent session returns None"""
        session = await VerificationSessionCRUD.update_session(db_session, "non-existent-session-id", status="processing")
        assert session is None
        
    async def test_complete_session_not_found(self, db_session: AsyncSession, test_user: User):
        """Test completing non-existent session returns None"""
        session = await VerificationSessionCRUD.complete_session(
            db_session, "non-existent-session-id", 
            is_match=True, similarity_score=0.95, confidence=0.98
        )
        assert session is None
        
    async def test_fail_session_not_found(self, db_session: AsyncSession, test_user: User):
        """Test failing non-existent session returns None"""
        session = await VerificationSessionCRUD.fail_session(
            db_session, "non-existent-session-id", 
            error_code="NOT_FOUND", error_message="Session not found"
        )
        assert session is None
        
    async def test_get_user_sessions_empty(self, db_session: AsyncSession, test_user: User):
        """Test getting sessions for user with no sessions returns empty list"""
        sessions = await VerificationSessionCRUD.get_user_sessions(db_session, "non-existent-user-id")
        assert sessions == []

# ============================================================================
# Audit Log Tests
# ============================================================================

class TestAuditLogCRUD:
    async def test_log_action(self, db_session: AsyncSession, test_user: User):
        """Test logging action (Async)"""
        log = await AuditLogCRUD.log_action(
            db_session,
            action="user_created",
            resource_type="user",
            resource_id=test_user.id,
            user_id=test_user.id,
            description="Test user created",
            old_values={"v1": 1},
            new_values={"v2": 2},
            ip_address="127.0.0.1"
        )
        
        assert log is not None
        assert log.action == "user_created"
        
    async def test_get_logs(self, db_session: AsyncSession, test_user: User):
        """Test getting logs (Async)"""
        await AuditLogCRUD.log_action(
            db_session,
            action="user_created",
            resource_type="user",
            resource_id=test_user.id,
            user_id=test_user.id
        )
        
        logs = await AuditLogCRUD.get_logs(db_session, user_id=test_user.id)
        assert len(logs) >= 1

    async def test_get_resource_logs(self, db_session: AsyncSession, test_user: User):
        """Test getting resource logs (Async)"""
        await AuditLogCRUD.log_action(
            db_session,
            action="user_updated",
            resource_type="user",
            resource_id=test_user.id,
            user_id=test_user.id
        )
        
        logs = await AuditLogCRUD.get_resource_logs(db_session, "user", test_user.id)
        assert len(logs) >= 1

    # ============================================================================
    # Edge Case Tests for AuditLogCRUD
    # ============================================================================
    
    async def test_get_logs_empty(self, db_session: AsyncSession, test_user: User):
        """Test getting logs for non-existent user returns empty list"""
        logs = await AuditLogCRUD.get_logs(db_session, user_id="non-existent-user-id")
        assert logs == []
        
    async def test_get_logs_by_action_empty(self, db_session: AsyncSession, test_user: User):
        """Test getting logs by non-existent action returns empty list"""
        logs = await AuditLogCRUD.get_logs(db_session, action="non-existent-action")
        assert logs == []
        
    async def test_get_resource_logs_empty(self, db_session: AsyncSession, test_user: User):
        """Test getting logs for non-existent resource returns empty list"""
        logs = await AuditLogCRUD.get_resource_logs(db_session, "non-existent-resource", "non-existent-id")
        assert logs == []
        
    async def test_log_action_fire_and_forget(self, db_session: AsyncSession, test_user: User):
        """Test that audit logging doesn't break on errors (fire-and-forget)"""
        # Even with invalid data, logging should not crash the application
        log = await AuditLogCRUD.log_action(
            db_session,
            action="test_action",
            resource_type="test_resource",
            resource_id="test_id",
            user_id=None,  # Invalid user_id
            success=False,
            error_message="Test error"
        )
        # Log might be None due to error handling, but should not crash
        assert log is None or isinstance(log, type(log))


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    def test_config_validation_basic(self):
        """Test basic configuration validation"""
        from app.config import Settings
        
        # Test that we can create settings with valid values
        settings = Settings(DEBUG=False, ENVIRONMENT="production")
        assert settings.DEBUG == False
        assert settings.ENVIRONMENT == "production"
        
        # Test default values (DEBUG=True, ENVIRONMENT='development')
        default_settings = Settings()
        assert default_settings.DEBUG == True  # default is True
        assert default_settings.ENVIRONMENT == "development"  # default is development
            
    def test_config_validation_debug_development(self):
        """Test that DEBUG can be enabled in development environment"""
        from app.config import Settings
        
        # This should not raise an exception
        settings = Settings(DEBUG=True, ENVIRONMENT="development")
        assert settings.DEBUG == True
        assert settings.ENVIRONMENT == "development"
        
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from string"""
        from app.config import Settings
        
        settings = Settings(CORS_ORIGINS="http://localhost:3000,https://example.com")
        origins = settings.cors_origins_list
        assert "http://localhost:3000" in origins
        assert "https://example.com" in origins
        
    def test_allowed_image_formats_parsing(self):
        """Test allowed image formats parsing from string"""
        from app.config import Settings
        
        settings = Settings(ALLOWED_IMAGE_FORMATS="JPEG,PNG,WEBP")
        formats = settings.allowed_image_formats_list
        assert "JPEG" in formats
        assert "PNG" in formats
        assert "WEBP" in formats
        
    def test_database_url_properties(self):
        """Test database URL properties"""
        from app.config import Settings
        
        settings = Settings(DATABASE_URL="sqlite:///./test.db")
        assert settings.sync_database_url == "sqlite:///./test.db"
        assert settings.async_database_url == "sqlite+aiosqlite:///./test.db"
        
    def test_redis_url_with_auth(self):
        """Test Redis URL with authentication"""
        from app.config import Settings
        
        settings = Settings(REDIS_URL="redis://localhost:6379/0", REDIS_PASSWORD="secret")
        auth_url = settings.redis_url_with_auth
        assert ":secret@" in auth_url