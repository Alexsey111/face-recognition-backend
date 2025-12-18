"""
Улучшенная конфигурация тестирования с изоляцией от внешних сервисов.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.db.database import Base
from app.db.database import get_db
from app.main import app
from fastapi.testclient import TestClient
from datetime import datetime

# =============================================================================
# НАСТРОЙКА ТЕСТОВОЙ БАЗЫ ДАННЫХ
# =============================================================================

# Используем SQLite для тестирования (не требует внешних сервисов)
SQLALCHEMY_TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_async.db"

# Создаем асинхронный движок для тестов
async_engine = create_async_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Фабрика сессий для тестов
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# =============================================================================
# МОКИ ВНЕШНИХ СЕРВИСОВ
# =============================================================================

def mock_redis_client():
    """Создает мок Redis клиента"""
    mock_redis = MagicMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.exists = AsyncMock(return_value=False)
    mock_redis.expire = AsyncMock(return_value=True)
    return mock_redis

def mock_storage_service():
    """Создает мок StorageService"""
    mock_storage = MagicMock()
    mock_storage.list_images = AsyncMock(return_value=[])
    mock_storage.get_image_info = AsyncMock(return_value=None)
    mock_storage.delete_image = AsyncMock(return_value=True)
    mock_storage.upload_image = AsyncMock(return_value="test_url")
    mock_storage.get_storage_stats = MagicMock(return_value={
        "bucket_name": "test-bucket",
        "total_objects": 0,
        "status": "accessible"
    })
    return mock_storage

def mock_ml_service():
    """Создает мок MLService"""
    mock_ml = MagicMock()
    mock_ml.detect_faces = AsyncMock(return_value=[])
    mock_ml.extract_embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_ml.compare_faces = AsyncMock(return_value=0.85)
    mock_ml.validate_image_quality = AsyncMock(return_value=True)
    return mock_ml

def mock_auth_service():
    """Создает мок AuthService"""
    mock_auth = MagicMock()
    mock_auth.create_access_token = MagicMock(return_value="test_token")
    mock_auth.verify_password = MagicMock(return_value=True)
    mock_auth.get_password_hash = MagicMock(return_value="hashed_password")
    mock_auth.create_user_session = AsyncMock(return_value={
        "access_token": "test_token",
        "token_type": "bearer"
    })
    return mock_auth

def mock_database_service():
    """Создает мок DatabaseService"""
    mock_db = MagicMock()
    
    # Настраиваем CRUD объекты с правильными методами
    mock_verification_crud = MagicMock()
    mock_verification_crud.cleanup_old_sessions = MagicMock(return_value=5)
    
    mock_reference_crud = MagicMock()
    mock_reference_crud.get_reference_by_file_key = MagicMock(return_value=None)
    
    mock_user_crud = MagicMock()
    
    mock_audit_crud = MagicMock()
    mock_audit_crud.cleanup_old_logs = MagicMock(return_value=10)
    
    mock_db.verification_crud = mock_verification_crud
    mock_db.reference_crud = mock_reference_crud
    mock_db.user_crud = mock_user_crud
    mock_db.audit_crud = mock_audit_crud
    
    return mock_db

# =============================================================================
# АСИНХРОННЫЕ ФИКСТУРЫ
# =============================================================================

@pytest.fixture(scope="session")
async def setup_test_database():
    """Создает и удаляет таблицы для всей тестовой сессии"""
    # Создаем таблицы
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Удаляем таблицы
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# =============================================================================
# ПЕРЕОПРЕДЕЛЕНИЕ ЗАВИСИМОСТЕЙ
# =============================================================================

async def override_get_db():
    """Переопределение get_db для тестирования"""
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

# Создаем синхронную версию для тестов, которые используют sync код
def override_get_db_sync():
    """Синхронная версия get_db для тестирования"""
    from contextlib import contextmanager
    
    @contextmanager
    def get_db_context():
        """Контекстный менеджер для БД"""
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    return get_db_context()

# Применяем переопределение зависимостей
app.dependency_overrides[get_db] = override_get_db

# =============================================================================
# ОСНОВНЫЕ ФИКСТУРЫ
# =============================================================================

@pytest.fixture
def test_settings():
    """Переопределяет настройки для тестирования"""
    from app.config import settings
    
    # Сохраняем оригинальные настройки
    original_debug = settings.DEBUG
    original_db_url = settings.DATABASE_URL
    original_redis_url = settings.REDIS_URL
    
    # Устанавливаем тестовые настройки
    settings.DEBUG = True
    settings.DATABASE_URL = SQLALCHEMY_TEST_DATABASE_URL
    settings.REDIS_URL = "redis://localhost:6379/15"  # Используем другую БД для тестов
    
    yield settings
    
    # Восстанавливаем оригинальные настройки
    settings.DEBUG = original_debug
    settings.DATABASE_URL = original_db_url
    settings.REDIS_URL = original_redis_url

@pytest.fixture
async def db_session(setup_test_database):
    """Предоставляет асинхронную сессию БД для тестов"""
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

@pytest.fixture
def client():
    """FastAPI тестовый клиент с переопределенными зависимостями"""
    with TestClient(app) as c:
        yield c

@pytest.fixture
async def client_with_mocks():
    """Клиент с моками внешних сервисов"""
    with patch('app.services.cache_service.redis.Redis', mock_redis_client):
        with patch('app.services.storage_service.StorageService', mock_storage_service):
            with patch('app.services.ml_service.MLService', mock_ml_service):
                with patch('app.services.auth_service.AuthService', mock_auth_service):
                    with patch('app.services.database_service.DatabaseService', mock_database_service):
                        with TestClient(app) as client:
                            yield client

# =============================================================================
# ФИКСТУРЫ ДЛЯ ТЕСТОВЫХ ДАННЫХ
# =============================================================================

@pytest.fixture
def test_user_data():
    """Тестовые данные пользователя"""
    from app.models.user import UserCreate
    import uuid
    
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(datetime.now().timestamp())
    
    return UserCreate(
        email=f"test_{unique_id}_{timestamp}@example.com",
        phone=f"+1234567890{unique_id}",
        full_name="Test User"
    )

@pytest.fixture
def test_reference_data():
    """Тестовые данные референса"""
    from app.models.reference import ReferenceCreate
    import uuid
    
    return ReferenceCreate(
        user_id=str(uuid.uuid4()),
        embedding="test_embedding",
        quality_score=0.9,
        image_filename="test.jpg"
    )

@pytest.fixture
def test_verification_data():
    """Тестовые данные верификации"""
    from app.models.verification import VerificationCreate
    
    return VerificationCreate(
        session_id="test_session",
        reference_id="test_reference"
    )

# =============================================================================
# ФИКСТУРЫ ДЛЯ СОЗДАНИЯ ТЕСТОВЫХ ОБЪЕКТОВ
# =============================================================================

@pytest.fixture
async def test_user(db_session, test_user_data):
    """Создает тестового пользователя"""
    from app.db.crud import UserCRUD
    
    user = await UserCRUD.create_user(db_session, test_user_data)
    await db_session.commit()
    return user

@pytest.fixture
async def test_reference(db_session, test_user, test_reference_data):
    """Создает тестовый референс"""
    from app.db.crud import ReferenceCRUD
    
    reference = await ReferenceCRUD.create_reference(
        db_session,
        user_id=test_user.id,
        embedding=test_reference_data.embedding,
        embedding_encrypted=b"encrypted_embedding",
        embedding_hash="test_hash",
        quality_score=test_reference_data.quality_score,
        image_filename=test_reference_data.image_filename,
        image_size_mb=1.0,
        image_format="JPEG",
        file_url="test_url"
    )
    await db_session.commit()
    return reference

@pytest.fixture
async def test_session(db_session, test_user):
    """Создает тестовую сессию верификации"""
    from app.db.crud import VerificationSessionCRUD
    from datetime import datetime, timedelta
    
    session_data = await VerificationSessionCRUD.create_session(
        db_session,
        user_id=test_user.id,
        session_id="test_session_123",
        image_filename="test.jpg",
        image_size_mb=1.0,
        expires_at=datetime.utcnow() + timedelta(hours=1)
    )
    await db_session.commit()
    return session_data

# =============================================================================
# ФИКСТУРЫ ДЛЯ АВТОРИЗАЦИИ
# =============================================================================

@pytest.fixture
async def authenticated_client(client_with_mocks, test_user):
    """Клиент с авторизацией"""
    from app.services.auth_service import AuthService
    
    auth_service = AuthService()
    tokens = await auth_service.create_user_session(test_user.id)
    
    client_with_mocks.headers = {
        "Authorization": f"Bearer {tokens['access_token']}"
    }
    return client_with_mocks

# =============================================================================
# ФИКСТУРЫ ДЛЯ РАЗНЫХ ТИПОВ ТЕСТОВ
# =============================================================================

@pytest.fixture
def unit_test_mocks():
    """Моки для unit тестов (без внешних сервисов)"""
    with patch('app.services.cache_service.redis.Redis', mock_redis_client):
        with patch('app.services.storage_service.StorageService', mock_storage_service):
            with patch('app.services.ml_service.MLService', mock_ml_service):
                with patch('app.services.auth_service.AuthService', mock_auth_service):
                    yield {
                        'redis': mock_redis_client(),
                        'storage': mock_storage_service(),
                        'ml': mock_ml_service(),
                        'auth': mock_auth_service()
                    }

@pytest.fixture
async def integration_test_setup():
    """Настройка для integration тестов"""
    # Можно добавить настройку внешних сервисов здесь
    # Например, запуск testcontainers для Redis, MinIO и т.д.
    yield {
        'database_url': SQLALCHEMY_TEST_DATABASE_URL,
        'redis_url': "redis://localhost:6379/15",
        'storage_endpoint': "http://localhost:9000"
    }

# =============================================================================
# АВТООЧИСТКА
# =============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Автоматическая очистка после каждого теста"""
    yield
    # Очистка после теста
    await asyncio.sleep(0.01)  # Даем время для завершения операций

# =============================================================================
# НАСТРОЙКИ ДЛЯ РАЗНЫХ МАРКЕРОВ
# =============================================================================

def pytest_configure(config):
    """Настройка pytest маркеров"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (быстрые тесты без внешних зависимостей)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (тесты с внешними сервисами)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (долгие тесты)"
    )
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "database: Database-related tests"
    )
    config.addinivalue_line(
        "markers", "ml: Machine Learning model tests"
    )
    config.addinivalue_line(
        "markers", "security: Security-related tests"
    )

# =============================================================================
# ХУКИ ДЛЯ УЛУЧШЕНИЯ ТЕСТИРОВАНИЯ
# =============================================================================

@pytest.fixture(autouse=True)
def set_test_environment():
    """Устанавливает тестовое окружение"""
    os.environ["TESTING"] = "True"
    yield
    # Восстанавливаем окружение
    if "TESTING" in os.environ:
        del os.environ["TESTING"]