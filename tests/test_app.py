"""
Тестовое приложение с правильными настройками для тестирования.
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Устанавливаем тестовые переменные окружения до импорта модулей приложения
os.environ["TESTING"] = "True"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_async.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["S3_ACCESS_KEY"] = "minioadmin"
os.environ["S3_SECRET_KEY"] = "minioadmin"
os.environ["S3_BUCKET_NAME"] = "test-bucket"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only-long-enough"
os.environ["ENCRYPTION_KEY"] = "test-encryption-key-for-testing-only"

# Импортируем модули приложения с правильными настройками
from app.main import create_test_app
from app.db import get_db
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

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
TestingSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# =============================================================================
# ПЕРЕОПРЕДЕЛЕНИЕ ЗАВИСИМОСТЕЙ
# =============================================================================

async def override_get_async_db():
    """Переопределение get_async_db для тестирования"""
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

# =============================================================================
# СОЗДАНИЕ ТЕСТОВОГО ПРИЛОЖЕНИЯ
# =============================================================================

# Создаем тестовое приложение без AuthMiddleware
test_app = create_test_app()

# Переопределяем зависимости для тестирования
test_app.dependency_overrides[get_db] = override_get_async_db

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ ТЕСТОВОЙ БАЗЫ ДАННЫХ
# =============================================================================

async def setup_test_database():
    """Создает таблицы для тестовой базы данных"""
    from app.db.database import Base
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Примечание: Таблицы создаются через фикстуру setup_test_database в conftest.py

# Экспортируем для использования в тестах
__all__ = ['test_app', 'TestingSessionLocal', 'override_get_async_db', 'SQLALCHEMY_TEST_DATABASE_URL', 'async_engine']