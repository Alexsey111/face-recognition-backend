#!/usr/bin/env python3
"""
Синхронные тесты для CRUD операций - без проблем с pytest-asyncio
"""

import pytest
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.db.crud import UserCRUD, ReferenceCRUD, VerificationSessionCRUD, AuditLogCRUD
from app.models.user import UserCreate, UserUpdate
from app.db.models import User, Reference, VerificationSession, AuditLog, VerificationStatus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Синхронная настройка SQLite для тестов
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_sync.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

@pytest.fixture(scope="function")
def db_session():
    """Provides a sync session for tests."""
    # Создаем таблицы
    from app.db.database import Base
    Base.metadata.create_all(bind=engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

class TestUserCRUDSync:
    """Синхронные тесты для UserCRUD"""
    
    def test_create_user(self, db_session):
        """Test user creation (sync)"""
        user_data = UserCreate(
            email="test@example.com",
            phone="+1234567890",
            full_name="Test User"
        )
        
        user = UserCRUD.create_user(db_session, user_data)
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.phone == "+1234567890"
        assert user.is_active == True
        
    def test_get_user(self, db_session):
        """Test get user by ID (sync)"""
        user_data = UserCreate(
            email="test2@example.com",
            phone="+1234567891",
            full_name="Test User 2"
        )
        
        created = UserCRUD.create_user(db_session, user_data)
        user = UserCRUD.get_user(db_session, created.id)
        
        assert user.id == created.id
        assert user.email == "test2@example.com"
        
    def test_get_user_by_email(self, db_session):
        """Test get user by email (sync)"""
        user_data = UserCreate(
            email="test3@example.com",
            phone="+1234567892",
            full_name="Test User 3"
        )
        
        created = UserCRUD.create_user(db_session, user_data)
        user = UserCRUD.get_user_by_email(db_session, "test3@example.com")
        
        assert user.id == created.id
        
    def test_count_users(self, db_session):
        """Test count users (sync)"""
        # Создаем двух пользователей
        UserCRUD.create_user(db_session, UserCreate(
            email="count1@example.com", phone="+1111111111", full_name="Count 1"
        ))
        UserCRUD.create_user(db_session, UserCreate(
            email="count2@example.com", phone="+2222222222", full_name="Count 2"
        ))
        
        count = UserCRUD.count_users(db_session)
        assert count == 2
        
    def test_update_user(self, db_session):
        """Test user update (sync)"""
        user_data = UserCreate(
            email="update@example.com",
            phone="+3333333333",
            full_name="Original Name"
        )
        
        created = UserCRUD.create_user(db_session, user_data)
        updated_data = UserUpdate(full_name="Updated Name")
        updated = UserCRUD.update_user(db_session, created.id, updated_data)
        
        assert updated.full_name == "Updated Name"
        
    def test_delete_user(self, db_session):
        """Test user deletion (sync)"""
        user_data = UserCreate(
            email="delete@example.com",
            phone="+4444444444",
            full_name="Delete Me"
        )
        
        created = UserCRUD.create_user(db_session, user_data)
        success = UserCRUD.delete_user(db_session, created.id)
        
        assert success == True
        assert UserCRUD.get_user(db_session, created.id) is None

class TestReferenceCRUDSync:
    """Синхронные тесты для ReferenceCRUD"""
    
    def test_create_reference(self, db_session):
        """Test reference creation (sync)"""
        # Сначала создаем пользователя
        user_data = UserCreate(
            email="ref@example.com",
            phone="+5555555555",
            full_name="Reference User"
        )
        user = UserCRUD.create_user(db_session, user_data)
        
        ref = ReferenceCRUD.create_reference(
            db_session,
            user_id=user.id,
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
        
    def test_get_latest_reference(self, db_session):
        """Test getting latest reference (sync)"""
        # Создаем пользователя
        user = UserCRUD.create_user(db_session, UserCreate(
            email="latest@example.com", phone="+6666666666", full_name="Latest User"
        ))
        
        # Создаем два reference
        ref1 = ReferenceCRUD.create_reference(
            db_session, user_id=user.id, embedding_encrypted=b"data1",
            embedding_hash="hash1", quality_score=0.9,
            image_filename="photo1.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        ref2 = ReferenceCRUD.create_reference(
            db_session, user_id=user.id, embedding_encrypted=b"data2",
            embedding_hash="hash2", quality_score=0.95,
            image_filename="photo2.jpg", image_size_mb=2.5,
            image_format="JPG"
        )
        
        latest = ReferenceCRUD.get_latest_reference(db_session, user.id)
        assert latest.id == ref2.id
        assert latest.version == 2

class TestVerificationSessionCRUDSync:
    """Синхронные тесты для VerificationSessionCRUD"""
    
    def test_create_session(self, db_session):
        """Test session creation (sync)"""
        # Создаем пользователя
        user = UserCRUD.create_user(db_session, UserCreate(
            email="session@example.com", phone="+7777777777", full_name="Session User"
        ))
        
        session = VerificationSessionCRUD.create_session(
            db_session,
            user_id=user.id,
            session_id="test_session_123",
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        assert session.id is not None
        assert session.status == "pending"
        
    def test_complete_session(self, db_session):
        """Test completing session (sync)"""
        # Создаем пользователя
        user = UserCRUD.create_user(db_session, UserCreate(
            email="complete@example.com", phone="+8888888888", full_name="Complete User"
        ))
        
        session = VerificationSessionCRUD.create_session(
            db_session,
            user_id=user.id,
            session_id="complete_session_123",
            image_filename="verify.jpg",
            image_size_mb=1.5
        )
        
        completed = VerificationSessionCRUD.complete_session(
            db_session,
            session_id="complete_session_123",
            is_match=True,
            similarity_score=0.95,
            confidence=0.98
        )
        
        assert completed.status == "success"
        assert completed.is_match == True

class TestAuditLogCRUDSync:
    """Синхронные тесты для AuditLogCRUD"""
    
    def test_log_action(self, db_session):
        """Test logging action (sync)"""
        # Создаем пользователя
        user = UserCRUD.create_user(db_session, UserCreate(
            email="audit@example.com", phone="+9999999999", full_name="Audit User"
        ))
        
        log = AuditLogCRUD.log_action(
            db_session,
            action="user_created",
            resource_type="user",
            resource_id=user.id,
            user_id=user.id,
            description="Test user created"
        )
        
        assert log is not None
        assert log.action == "user_created"
        
    def test_get_logs(self, db_session):
        """Test getting logs (sync)"""
        # Создаем пользователя
        user = UserCRUD.create_user(db_session, UserCreate(
            email="log@example.com", phone="+1010101010", full_name="Log User"
        ))
        
        # Создаем два лога
        AuditLogCRUD.log_action(
            db_session,
            action="user_created",
            resource_type="user",
            resource_id=user.id,
            user_id=user.id
        )
        
        AuditLogCRUD.log_action(
            db_session,
            action="user_updated",
            resource_type="user",
            resource_id=user.id,
            user_id=user.id
        )
        
        logs = AuditLogCRUD.get_logs(db_session, user_id=user.id)
        assert len(logs) >= 2