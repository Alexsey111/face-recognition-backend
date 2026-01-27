"""
Pytest configuration and fixtures for testing.
Объединённая версия с mock services и реальной БД.
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

'''# Mock cv2 to avoid import issues
sys.modules['cv2'] = type(sys)('cv2_mock')
sys.modules['cv2.dnn'] = type(sys)('cv2_dnn_mock')

# Mock torch to avoid import issues
sys.modules['torch'] = type(sys)('torch_mock')
sys.modules['torch.nn'] = type(sys)('torch_nn_mock')
sys.modules['torchvision'] = type(sys)('torchvision_mock')

# Mock facenet_pytorch
sys.modules['facenet_pytorch'] = type(sys)('facenet_pytorch_mock')
#sys.modules['PIL'] = type(sys)('PIL_mock')'''

import asyncio
import uuid
import gc
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.config import settings
from app.main import app
from app.db.database import get_db


# =============================================================================
# Event Loop Configuration
# =============================================================================
# Измени блок в conftest.py на этот:

@pytest_asyncio.fixture(scope="session")
async def initialized_service():
    """
    Инициализирует AntiSpoofingService один раз для всех тестов.
    Это значительно ускоряет прогон за счет однократной загрузки весов.
    """
    from app.services.anti_spoofing_service import AntiSpoofingService
    import torch
    import gc

    # Создаем и инициализируем сервис
    svc = AntiSpoofingService()
    
    try:
        await svc.initialize()
    except Exception as e:
        pytest.skip(f"Критическая ошибка инициализации ML-моделей: {e}")

    yield svc

    # Очистка ресурсов только в самом конце сессии
    try:
        if hasattr(svc, 'model') and svc.model is not None:
            svc.model.cpu()
            del svc.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
    except Exception as e:
        print(f"Ошибки при очистке ML-ресурсов: {e}")

'''@pytest.fixture(scope="function")
def event_loop():
    """
    Create event loop for async tests.
    CRITICAL: Must be function-scoped, not session-scoped!
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop

    # Cleanup pending tasks
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        if pending:
            loop.run_until_complete(
                asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=5.0
                )
            )
    except (asyncio.TimeoutError, RuntimeError, Exception) as e:
        print(f"Warning: Error during loop cleanup: {e}")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        except:
            pass
        loop.close()
        gc.collect()'''


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create async engine for tests using real PostgreSQL."""
    db_url = settings.DATABASE_URL

    # Ensure asyncpg driver
    if "postgresql://" in db_url and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if "postgres://" in db_url and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(
        db_url,
        poolclass=NullPool,
        echo=False,
        future=True,
    )

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create database session for tests with automatic rollback.
    Изолированная транзакция для каждого теста.
    """
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        # Automatic rollback after test
        await session.rollback()


# =============================================================================
# User Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def test_user(db_session: AsyncSession) -> AsyncGenerator[str, None]:
    """Create test user and return user_id."""
    test_user_id = f"test-user-{uuid.uuid4().hex[:8]}"

    try:
        await db_session.execute(
            text(
                "INSERT INTO users (id, email, password_hash, is_active, "
                "total_uploads, total_verifications, successful_verifications) "
                "VALUES (:id, :email, :password_hash, TRUE, 0, 0, 0) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": test_user_id,
                "email": f"{test_user_id}@example.com",
                "password_hash": "test_hash_placeholder",
            },
        )
        await db_session.commit()

        yield test_user_id

    finally:
        # Cleanup
        try:
            await db_session.execute(
                text("DELETE FROM users WHERE id = :id"), {"id": test_user_id}
            )
            await db_session.commit()
        except Exception as e:
            print(f"Warning: Failed to cleanup test user: {e}")
            await db_session.rollback()


@pytest_asyncio.fixture(scope="function")
async def test_user_with_reference(
    db_session: AsyncSession, test_user: str
) -> AsyncGenerator[dict, None]:
    """Create test user with reference image."""
    from app.db.crud import ReferenceCRUD

    ref = await ReferenceCRUD.create_reference(
        db_session,
        user_id=test_user,
        embedding_encrypted=b"encrypted_test_data",
        embedding_hash=f"hash_{uuid.uuid4().hex[:16]}",
        quality_score=0.85,
        image_filename="test_photo.jpg",
        image_size_mb=1.5,
        image_format="JPG",
    )
    await db_session.commit()

    yield {"user_id": test_user, "reference_id": ref.id}


@pytest.fixture
def test_user_data():
    """Create test user data for UserCreate model."""
    from app.models.user import UserCreate

    unique_id = uuid.uuid4().hex[:8]
    return UserCreate(
        email=f"test-{unique_id}@example.com",
        password="testpassword123",
        phone=f"+1234567890{unique_id[:4]}",
        full_name=f"Test User {unique_id}",
    )


@pytest_asyncio.fixture
async def test_user_123(db_session: AsyncSession) -> str:
    """Return a fixed user ID for testing."""
    user_id = "user-123"

    try:
        await db_session.execute(
            text(
                "INSERT INTO users (id, email, password_hash, is_active, "
                "total_uploads, total_verifications, successful_verifications) "
                "VALUES (:id, :email, :password_hash, TRUE, 0, 0, 0) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": user_id,
                "email": "user-123@example.com",
                "password_hash": "test_hash",
            },
        )
        await db_session.commit()
    except:
        await db_session.rollback()

    return user_id


# =============================================================================
# Mock Data Fixtures (для unit-тестов)
# =============================================================================


@pytest.fixture
def mock_user_id():
    """Mock user ID."""
    return "test-user-123"


@pytest.fixture
def mock_reference_id():
    """Mock reference ID."""
    return "test-reference-456"


@pytest.fixture
def mock_image_data():
    """Mock image data (base64 encoded 1x1 pixel PNG)."""
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.fixture
def mock_embedding():
    """Mock face embedding (512-dimensional vector)."""
    import numpy as np

    return np.random.rand(512).astype(np.float32).tobytes()


@pytest.fixture
def mock_quality_score():
    """Mock quality score."""
    return 0.85


# =============================================================================
# Redis Fixtures
# =============================================================================

import redis.asyncio as aioredis


@pytest_asyncio.fixture(scope="function")
async def redis_client():
    """Create async Redis client for tests."""
    client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )

    yield client

    await client.aclose()


@pytest.fixture(scope="function")
def redis_sync_client():
    """Create sync Redis client for sync tests."""
    import redis as redis_sync

    client = redis_sync.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )

    yield client

    client.close()


@pytest_asyncio.fixture(scope="function")
async def cache():
    """Create cache service for tests."""
    from app.services.cache_service import CacheService

    cache_service = CacheService()

    yield cache_service

    await cache_service.close()


# =============================================================================
# Test Client Fixtures
# =============================================================================

from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient


@pytest.fixture(scope="function")
def app_instance():
    """Create FastAPI app instance for tests."""
    return app


@pytest_asyncio.fixture(scope="function")
async def db_session_override(app_instance, async_engine):
    """Override database dependency with test session."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def get_test_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_maker() as session:
            yield session

    app_instance.dependency_overrides[get_db] = get_test_db

    yield

    app_instance.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(db_session_override, app_instance):
    """Create synchronous test client."""
    with TestClient(app_instance) as test_client:
        yield test_client


@pytest_asyncio.fixture(scope="function")
async def async_client(db_session_override, app_instance):
    """Create asynchronous test client."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# =============================================================================
# Authentication Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def auth_headers(test_user: str) -> dict:
    """Create authorization headers with test user JWT."""
    from app.services.auth_service import AuthService

    auth_service = AuthService()
    tokens = await auth_service.create_user_session(
        user_id=test_user, user_agent="test-agent", ip_address="127.0.0.1"
    )

    return {"Authorization": f"Bearer {tokens['access_token']}"}


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for upload tests."""
    import base64

    # Minimal 1x1 PNG image
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    return base64.b64decode(png_base64)


@pytest.fixture
def sample_image_file(tmp_path, sample_image_bytes):
    """Create temporary image file for upload tests."""
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(sample_image_bytes)
    return str(image_path)


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    for item in items:
        # Add timeout marker for all async tests
        if asyncio.iscoroutinefunction(item.function):
            if not any(mark.name == "timeout" for mark in item.iter_markers()):
                item.add_marker(pytest.mark.timeout(60))


# =============================================================================
# Session-level Setup/Teardown
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests."""
    # Set test environment variables
    os.environ["TESTING"] = "1"

    yield

    # Cleanup after all tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture(scope="session", autouse=True)
def configure_asyncio():
    """Configure asyncio for tests."""
    if sys.platform == "win32":
        # Use WindowsSelectorEventLoopPolicy on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    yield


# =============================================================================
# Resource Cleanup
# =============================================================================


@pytest.fixture(scope="function", autouse=True)
def cleanup_resources():
    """Automatic resource cleanup after each test."""
    yield

    # Force garbage collection
    gc.collect()


@pytest.fixture(scope="function", autouse=True)
def reset_torch_state():
    """Reset PyTorch state between tests."""
    yield

    try:
        import torch

        # Disable gradients
        torch.set_grad_enabled(False)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# =============================================================================
# Anti-Spoofing Service Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function", autouse=True)
async def reset_antispoofing_between_tests():
    """Сброс AntiSpoofing сервиса между тестами."""
    from app.services.anti_spoofing_service import reset_anti_spoofing_service

    yield

    # Cleanup после каждого теста
    try:
        await reset_anti_spoofing_service()
    except Exception as e:
        print(f"Warning: Failed to reset anti-spoofing service: {e}")


@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_test_users(db_session: AsyncSession):
    """
    Очистка тестовых пользователей перед каждым тестом.
    Удаляет пользователей с тестовыми email'ами.
    """
    yield

    # Cleanup после теста
    try:
        # Удаляем тестовых пользователей по email паттерну
        test_emails = [
            "newuser@example.com",
            "livenesstest@example.com",
            "security@example.com",
            "perf@example.com",
        ]

        for email in test_emails:
            try:
                await db_session.execute(
                    text("DELETE FROM users WHERE email = :email"), {"email": email}
                )
            except Exception:
                pass

        await db_session.commit()
    except Exception as e:
        print(f"Warning: Failed to cleanup test users: {e}")
        await db_session.rollback()


@pytest_asyncio.fixture(scope="function")
async def initialized_service():
    """
    Fixture для инициализированного AntiSpoofingService.
    """
    # Полная очистка перед созданием
    from app.services.anti_spoofing_service import reset_anti_spoofing_service

    await reset_anti_spoofing_service()

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Создание нового сервиса
    from app.services.anti_spoofing_service import AntiSpoofingService

    svc = AntiSpoofingService()

    try:
        await svc.initialize()
    except Exception as e:
        pytest.skip(f"Failed to initialize service: {e}")

    yield svc

    # Cleanup
    try:
        if svc.model is not None:
            svc.model.cpu()
            del svc.model
        svc.model = None
        svc.is_initialized = False

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        gc.collect()
    finally:
        await reset_anti_spoofing_service()
