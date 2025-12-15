import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.db.database import Base
from app.db.database import get_db # Предполагаем, что это наша асинхронная зависимость
from app.main import app
from fastapi.testclient import TestClient

# 1. Асинхронный драйвер SQLite
SQLALCHEMY_TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_async.db"

# 2. Асинхронный движок
async_engine = create_async_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    echo=False # Можно поставить True для отладки
)

# 3. Асинхронный класс сессии
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 4. Асинхронное переопределение зависимости get_db
async def override_get_db_async():
    """Provides a transactional session for testing"""
    async with TestingSessionLocal() as session:
        # Для изоляции тестов используем begin/rollback
        # Это позволяет откатить все изменения после теста
        async with session.begin():
            yield session
            await session.rollback() # Откат изменений после yield

# Переопределяем асинхронную зависимость
app.dependency_overrides[get_db] = override_get_db_async

@pytest.fixture(scope="session")
def event_loop(request):
    """
    Создание и настройка асинхронного цикла событий для pytest-asyncio
    (Обычно не требуется, если используется async-compatible runner, 
    но полезно для явной настройки.)
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# 5. Асинхронная фикстура для настройки базы данных
@pytest.fixture(scope="session")
async def setup_db():
    """Create and drop all tables once per test session."""
    # Создание таблиц
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Удаление таблиц
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# 6. Асинхронная фикстура клиента
@pytest.fixture(scope="function")
async def client():
    """FastAPI test client"""
    # Теперь TestClient использует переопределенную async зависимость
    yield TestClient(app)

# 7. Асинхронная фикстура сессии (для прямого доступа к CRUD)
@pytest.fixture(scope="function")
async def db_session(setup_db):  # Добавляем setup_db как зависимость
    """Provides an async session instance tied to a transaction for functional tests."""
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
        
# 8. Асинхронная фикстура тестового пользователя
from app.models.user import UserCreate, UserUpdate
import uuid
from datetime import datetime

@pytest.fixture
def test_user_data():
    """Test user data as UserCreate object"""
    unique_id = str(uuid.uuid4())[:8]  # Берем первые 8 символов UUID
    timestamp = int(datetime.now().timestamp())  # Добавляем timestamp для уникальности
    return UserCreate(
        email=f"test_{unique_id}_{timestamp}@example.com",  # Уникальный email
        phone=f"+1234567890{unique_id}",  # Уникальный телефон
        full_name="Test User"
    )
    
@pytest.fixture
def test_user_update_data():
    """Test user update data as UserUpdate object"""
    return UserUpdate(
        full_name="Updated User"
    )

# 9. Создание тестового пользователя асинхронно
@pytest.fixture
async def test_user(db_session: AsyncSession, test_user_data: UserCreate):
    """Create test user asynchronously"""
    from app.db.crud import UserCRUD
    
    # Используем CRUD для создания пользователя
    user = await UserCRUD.create_user(db_session, test_user_data)
    return user