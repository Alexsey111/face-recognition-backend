#!/usr/bin/env python3
"""
Асинхронные тесты для CRUD операций - исправлены проблемы с coroutine objects
"""

import pytest
import sys
import os
import asyncio

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.db.crud import UserCRUD, ReferenceCRUD, VerificationSessionCRUD, AuditLogCRUD
from app.models.user import UserCreate, UserUpdate
from app.db.models import User, Reference, VerificationSession, AuditLog, VerificationStatus
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Асинхронная настройка SQLite для тестов
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./test_async.db"

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

@pytest.fixture(scope="function")
async def db_session():
    """Provides an async session for tests."""
    # Создаем таблицы
    async with engine.begin() as conn:
        from app.db.database import Base
        await conn.run_sync(Base.metadata.create_all)
    
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

class TestUserCRUDSync:
    """Асинхронные тесты для UserCRUD (исправлены coroutine проблемы)"""
    
    @pytest.mark.asyncio
    async def test_create_user(self, db_session):
        """Test user creation (async)"""
        user_data = UserCreate(
            email="test@example.com",
            phone="+1234567890",
            full_name="Test User"
        )
        
        user = await UserCRUD.create_user(db_session, user_data)  # ✅ await
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.phone == "+1234567890"
        assert user.is_active == True
        
    @pytest.mark.asyncio
    async def test_get_user(self, db_session):
        """Test get user by ID (async)"""
        user_data = UserCreate(
            email="test2@example.com",
            phone="+1234567891",
            full_name="Test User 2"
        )
        
        created = await UserCRUD.create_user(db_session, user_data)  # ✅ await
        user = await UserCRUD.get_user(db_session, created.id)  # ✅ await
        
        assert user.id == created.id
        assert user.email == "test2@example.com"
        
    @pytest.mark.asyncio
    async def test_get_user_by_email(self, db_session):
        """Test get user by email (async)"""
        user_data = UserCreate(
            email="test3@example.com",
            phone="+1234567892",
            full_name="Test User 3"
        )
        
        created = await UserCRUD.create_user(db_session, user_data)  # ✅ await
        user = await UserCRUD.get_user_by_email(db_session, "test3@example.com")  # ✅ await
        
        assert user.id == created.id
        
    @pytest.mark.asyncio
    async def test_count_users(self, db_session):
        """Test count users (async)"""
        # Создаем двух пользователей
        await UserCRUD.create_user(db_session, UserCreate(  # ✅ await
            email="count1@example.com", phone="+1111111111", full_name="Count 1"
        ))
        await UserCRUD.create_user(db_session, UserCreate(  # ✅ await
            email="count2@example.com", phone="+2222222222", full_name="Count 2"
        ))
        
        count = await UserCRUD.count_users(db_session)  # ✅ await
        assert count == 2
        
    @pytest.mark.asyncio
    async def test_update_user(self, db_session):
        """Test user update (async)"""
        user_data = UserCreate(
            email="update@example.com",
            phone="+3333333333",
            full_name="Original Name"
        )
        
        created = await UserCRUD.create_user(db_session, user_data)  # ✅ await
        updated_data = UserUpdate(full_name="Updated Name")
        updated = await UserCRUD.update_user(db_session, created.id, updated_data)  # ✅ await
        
        assert updated.full_name == "Updated Name"
        
    @pytest.mark.asyncio
    async def test_delete_user(self, db_session):
        """Test user deletion (async)"""
        user_data = UserCreate(
            email="delete@example.com",
            phone="+4444444444",
            full_name="Delete Me"
        )
        
        created = await UserCRUD.create_user(db_session, user_data)  # ✅ await
        success = await UserCRUD.delete_user(db_session, created.id)  # ✅ await
        
        assert success == True
        user = await UserCRUD.get_user(db_session, created.id)  # ✅ await
        assert user is None

class TestReferenceCRUDSync:
    """Асинхронные тесты для ReferenceCRUD (исправлены coroutine проблемы)"""
    
    @pytest.mark.asyncio
    async def test_create_reference(self, db_session):
        """Test reference creation (async)"""
        # Сначала создаем пользователя
        user_data = UserCreate(
            email="ref@example.com",
            phone="+5555555555",
            full_name="Reference User"
        )
        user = await UserCRUD.create_user(db_session, user_data)  # ✅ await
        
        ref = await ReferenceCRUD.create_reference(  # ✅ await
            db_session,
            user_id=user.id,
            embedding="test_embedding",
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
        
    @pytest.mark.asyncio
    async def test_get_latest_reference(self, db_session):
        """Test getting latest reference (async)"""
        # Создаем пользователя
        user = await UserCRUD.create_user(db_session, UserCreate(  # ✅ await
            email="latest@example.com", phone="+6666666666", full_name="Latest User"
        ))
        
        # Создаем два reference
        ref1 = await ReferenceCRUD.create_reference(  # ✅ await
            db_session, user_id=user.id, embedding="test_embedding1",
            embedding_encrypted=b"data1",
            embedding_hash="hash1", quality_score=0.9,
            image_filename="photo1.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        ref2 = await ReferenceCRUD.create_reference(  # ✅ await
            db_session, user_id=user.id, embedding="test_embedding2",
            embedding_encrypted=b"data2",
            embedding_hash="hash2", quality_score=0.95,
            image_filename="photo2.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        latest = await ReferenceCRUD.get_latest_reference(db_session, user.id)  # ✅ await
        assert latest.id == ref2.id
        assert latest.version == 2

class TestVerificationSessionCRUDSync:
    """Асинхронные тесты для VerificationSessionCRUD (исправлены coroutine проблемы)"""
    
    @pytest.mark.asyncio
    async def test_create_session(self, db_session):
        """Test session creation (async)"""
        # Создаем пользователя
        user = await UserCRUD.create_user(db_session, UserCreate(  # ✅ await
            email="session@example.com", phone="+7777777777", full_name="Session User"
        ))
        
        session = await VerificationSessionCRUD.create_session(  # ✅ await
            db_session,
            user_id=user.id,
            session_id="test_session_123",
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        assert session.id is not None
        assert session.status == VerificationStatus.PENDING
        
    @pytest.mark.asyncio
    async def test_complete_session(self, db_session):
        """Test completing session (async)"""
        # Создаем пользователя
        user = await UserCRUD.create_user(db_session, UserCreate(  # ✅ await
            email="complete@example.com", phone="+8888888888", full_name="Complete User"
        ))
        
        session = await VerificationSessionCRUD.create_session(  # ✅ await
            db_session,
            user_id=user.id,
            session_id="complete_session_123",
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        completed = await VerificationSessionCRUD.complete_session(  # ✅ await
            db_session,
            session_id="complete_session_123",
            is_match=True,
            similarity_score=0.95,
            confidence=0.98
        )
        
        assert completed.status == VerificationStatus.SUCCESS
        assert completed.is_match == True

class TestAuditLogCRUDSync:
    """Асинхронные тесты для AuditLogCRUD (исправлены coroutine проблемы)"""
    
    @pytest.mark.asyncio
    async def test_log_action(self, db_session):
        """Test logging action (async)"""
        # Создаем пользователя
        user = await UserCRUD.create_user(db_session, UserCreate(  # ✅ await
            email="audit@example.com", phone="+9999999999", full_name="Audit User"
        ))
        
        log = await AuditLogCRUD.log_action(  # ✅ await
            db_session,
            action="user_created",
            resource_type="user",
            resource_id=user.id,
            user_id=user.id,
            description="Test user created"
        )
        
        assert log is not None
        assert log.action == "user_created"
        
    @pytest.mark.asyncio
    async def test_get_logs(self, db_session):
        """Test getting logs (async)"""
        # Создаем пользователя
        user = await UserCRUD.create_user(db_session, UserCreate(  # ✅ await
            email="log@example.com", phone="+1010101010", full_name="Log User"
        ))
        
        # Создаем два лога
        await AuditLogCRUD.log_action(  # ✅ await
            db_session,
            action="user_created",
            resource_type="user",
            resource_id=user.id,
            user_id=user.id
        )
        
        await AuditLogCRUD.log_action(  # ✅ await
            db_session,
            action="user_updated",
            resource_type="user",
            resource_id=user.id,
            user_id=user.id
        )
        
        logs = await AuditLogCRUD.get_logs(db_session, user_id=user.id)  # ✅ await
        assert len(logs) >= 2