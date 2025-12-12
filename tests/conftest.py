"""
Общие фикстуры для тестов Face Recognition Service
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch, patch
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from app.config import Settings, settings
from app.db.database import DatabaseManager


@pytest.fixture(scope="session")
def event_loop():
    """Создает экземпляр цикла событий для каждого тестового сеанса"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Фикстура для мок-объекта конфигурации"""
    config = Mock(spec=Settings)
    config.database_url = "postgresql://test:test@localhost:5432/test_db"
    config.redis_url = "redis://localhost:6379/0"
    config.jwt_secret_key = "test-jwt-secret-key"
    config.encryption_key = "test-encryption-key-256-bits"
    config.debug = True
    config.environment = "test"
    config.max_upload_size = 10485760  # 10MB
    config.allowed_image_formats = ["JPEG", "JPG", "PNG", "WEBP"]
    return config


@pytest.fixture
def mock_settings():
    """Фикстура для мок-объекта настроек"""
    settings = Mock(spec=Settings)
    settings.database_url = "postgresql://test:test@localhost:5432/test_db"
    settings.redis_url = "redis://localhost:6379/0"
    settings.jwt_secret_key = "test-jwt-secret-key"
    settings.encryption_key = "test-encryption-key-256-bits"
    settings.debug = True
    return settings


@pytest.fixture
def mock_database_manager():
    """Фикстура для мок-объекта менеджера базы данных"""
    manager = Mock(spec=DatabaseManager)
    manager.get_session = AsyncMock()
    manager.close_connections = AsyncMock()
    return manager


@pytest.fixture
def mock_async_session():
    """Фикстура для мок-объекта асинхронной сессии базы данных"""
    session = Mock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_redis_client():
    """Фикстура для мок-объекта Redis клиента"""
    redis_client = AsyncMock()
    redis_client.get = AsyncMock()
    redis_client.set = AsyncMock(return_value=True)
    redis_client.delete = AsyncMock(return_value=1)
    redis_client.exists = AsyncMock(return_value=1)
    redis_client.incr = AsyncMock(return_value=5)
    redis_client.decr = AsyncMock(return_value=3)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.ping = AsyncMock(return_value=True)
    return redis_client


@pytest.fixture
def mock_ml_service():
    """Фикстура для мок-объекта ML сервиса"""
    ml_service = Mock()
    ml_service.detect_faces = AsyncMock(return_value=[{"bbox": [100, 100, 200, 200]}])
    ml_service.extract_embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    ml_service.compare_faces = AsyncMock(return_value=0.85)
    ml_service.detect_liveness = AsyncMock(return_value=True)
    return ml_service


@pytest.fixture
def mock_storage_service():
    """Фикстура для мок-объекта сервиса хранения"""
    storage_service = Mock()
    storage_service.upload_image = AsyncMock(return_value="http://minio/test_image.jpg")
    storage_service.download_image = AsyncMock(return_value=b"image_data")
    storage_service.delete_image = AsyncMock(return_value=True)
    storage_service.get_image_url = Mock(return_value="http://minio/test_image.jpg")
    return storage_service


@pytest.fixture
def sample_user_data():
    """Фикстура с примерами данных пользователя"""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": None
    }


@pytest.fixture
def sample_reference_data():
    """Фикстура с примерами данных эталонного изображения"""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "label": "John Doe",
        "image_data": "base64_encoded_image_data",
        "created_at": "2024-01-01T00:00:00Z",
        "is_active": True,
        "metadata": {"source": "upload", "format": "JPEG"}
    }


@pytest.fixture
def sample_verification_data():
    """Фикстура с примерами данных верификации"""
    return {
        "session_id": "550e8400-e29b-41d4-a716-446655440002",
        "reference_id": "550e8400-e29b-41d4-a716-446655440001",
        "image_data": "base64_encoded_test_image",
        "threshold": 0.8,
        "confidence": 0.95,
        "is_match": True,
        "processing_time": 0.123,
        "created_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_image_data():
    """Фикстура с примерами данных изображения"""
    return {
        "format": "JPEG",
        "size": 1024 * 1024,  # 1MB
        "width": 1920,
        "height": 1080,
        "base64_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
        "file_name": "test_image.jpg"
    }


@pytest.fixture
def mock_http_response():
    """Фикстура для мок-объекта HTTP ответа"""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {"status": "success", "data": {}}
    response.text = '{"status": "success"}'
    response.content = b'{"status": "success"}'
    return response


@pytest.fixture
def test_api_key():
    """Фикстура с тестовым API ключом"""
    return "sk_test_1234567890abcdef1234567890abcdef"


@pytest.fixture
def test_uuid():
    """Фикстура с тестовым UUID"""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def async_engine():
    """Фикстура для асинхронного движка базы данных (для интеграционных тестов)"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    engine.sync_engine.dispose()


@pytest.fixture
def async_session_factory(async_engine):
    """Фикстура для фабрики асинхронных сессий"""
    return sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )


@pytest.fixture
async def async_session(async_session_factory):
    """Фикстура для асинхронной сессии базы данных"""
    async with async_session_factory() as session:
        yield session
        await session.close()


@pytest.fixture
def mock_logger():
    """Фикстура для мок-объекта логгера"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def capture_logs():
    """Фикстура для перехвата логов во время тестирования"""
    import logging
    import io
    
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger('test_logger')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


@pytest.fixture
def temp_file():
    """Фикстура для создания временного файла"""
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "test_file.txt")
    
    with open(temp_file_path, "w") as f:
        f.write("test content")
    
    yield temp_file_path
    
    # Очистка
    os.unlink(temp_file_path)
    os.rmdir(temp_dir)


@pytest.fixture
def mock_time():
    """Фикстура для мок-объекта времени"""
    with patch('time.time') as mock_time:
        mock_time.return_value = 1640995200.0  # 2022-01-01 00:00:00 UTC
        yield mock_time


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Автоматическая настройка тестового окружения"""
    # Устанавливаем переменные окружения для тестов
    import os
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DEBUG"] = "true"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_db"
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key"
    os.environ["ENCRYPTION_KEY"] = "test-encryption-key-256-bits"
    
    yield
    
    # Очистка после тестов
    test_env_vars = [
        "ENVIRONMENT", "DEBUG", "DATABASE_URL", "REDIS_URL",
        "JWT_SECRET_KEY", "ENCRYPTION_KEY"
    ]
    for var in test_env_vars:
        os.environ.pop(var, None)