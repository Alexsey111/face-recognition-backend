"""Pytest configuration and fixtures for database testing."""
import pytest
import asyncio
import os
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# =============================================================================
# Async Database Fixtures (Primary - for async CRUD tests) - PostgreSQL Only
# =============================================================================

# Get PostgreSQL URL from environment or use default test database
ASYNC_DATABASE_URL = os.getenv(
    "ASYNC_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/face_recognition"
)


@pytest.fixture(scope="function")
async def async_engine():
    """Create async engine for testing with PostgreSQL."""
    from sqlalchemy.orm import configure_mappers
    from app.db.models import Base, User, Reference, VerificationSession, AuditLog
    
    engine = create_async_engine(
        ASYNC_DATABASE_URL,
        echo=False,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Explicitly compile mappers to resolve relationships
    configure_mappers()
    
    yield engine
    
    # Dispose
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        # Clean up test data before each test
        await session.execute(text("DELETE FROM audit_logs"))
        await session.execute(text("DELETE FROM verification_sessions"))
        await session.execute(text('DELETE FROM "references"'))
        await session.execute(text("DELETE FROM users WHERE email LIKE 'test%@example.com'"))
        await session.commit()
        
        yield session
        await session.rollback()


# =============================================================================
# Sync Database Fixtures (Optional - for sync tests) - PostgreSQL Only
# =============================================================================

SYNC_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/face_recognition"
)


@pytest.fixture
def sync_engine():
    """Create synchronous engine for testing."""
    pytest.skip("Use async fixtures for CRUD tests")
    return None


@pytest.fixture
def sync_session():
    """Create synchronous session for testing."""
    pytest.skip("Use async fixtures for CRUD tests")
    return None


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def test_user_data():
    """Test user data for creation."""
    from app.models.user import UserCreate
    return UserCreate(
        email="test@example.com",
        phone="+1234567890",
        full_name="Test User"
    )


@pytest.fixture
def test_user_update_data():
    """Test user update data."""
    from app.models.user import UserUpdate
    return UserUpdate(full_name="Updated User")


@pytest.fixture
async def test_user(db_session: AsyncSession):
    """Create and return a test user."""
    from app.db.crud import UserCRUD
    from app.models.user import UserCreate

    user = await UserCRUD.create_user(
        db_session,
        UserCreate(
            email="testuser@example.com",
            phone="+1234567890",
            full_name="Test User"
        )
    )
    return user
        

@pytest.fixture
async def test_reference(db_session: AsyncSession, test_user):
    """Create and return a test reference."""
    from app.db.crud import ReferenceCRUD

    ref = await ReferenceCRUD.create_reference(
        db_session,
        user_id=test_user.id,
        embedding_encrypted=b"encrypted_data",
        embedding_hash="hash123",
        image_filename="photo.jpg",
        image_size_mb=2.5,
        image_format="JPG"
    )
    return ref


# =============================================================================
# Seed Test Reference (Backward Compatibility)
# =============================================================================

@pytest.fixture(autouse=True)
async def seed_test_reference(async_engine):
    """Seed minimal test data for backward compatibility tests.
    
    This fixture ensures tests that expect specific IDs will work.
    """
    from sqlalchemy import text
    
    async with async_engine.begin() as conn:
        # Clean up existing test data
        await conn.execute(text('DELETE FROM "references" WHERE id = :id'), {"id": "test-reference-123"})
        await conn.execute(text('DELETE FROM users WHERE id = :id'), {"id": "test-user-123"})
        
        # Insert test user
        await conn.execute(
            text(
                "INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications) "
                "VALUES (:id, :email, TRUE, 0, 0, 0)"
            ),
            {"id": "test-user-123", "email": "test-user@example.com"},
        )
        
        # Insert test reference (only using fields that exist in the model)
        await conn.execute(
            text(
                'INSERT INTO "references" (id, user_id, embedding_encrypted, embedding) '
                'VALUES (:id, :user_id, :embedding_encrypted, :embedding)'
            ),
            {
                "id": "test-reference-123",
                "user_id": "test-user-123",
                "embedding_encrypted": b"",
                "embedding": None,
            },
        )