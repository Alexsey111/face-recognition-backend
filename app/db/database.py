from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import Generator, AsyncGenerator, Optional
from contextlib import asynccontextmanager, contextmanager
from ..config import settings
from ..utils.logger import get_logger
logger = get_logger(__name__)

# --- 1. Modern SQLAlchemy 2.0 Base ---
class Base(DeclarativeBase):
    """Базовый класс для всех моделей (SQLAlchemy 2.0 style)"""
    pass

# ============================================================================
# Database Configuration
# ============================================================================

def get_database_url(database_type: str = "sync") -> str:
    """Получение URL базы данных в зависимости от типа."""
    base_url = settings.DATABASE_URL
    
    if database_type == "async":
        # Более надежная замена драйвера
        if base_url.startswith("postgresql://") or base_url.startswith("postgresql+psycopg2://"):
            return base_url.replace("postgresql://", "postgresql+asyncpg://") \
                   .replace("postgresql+psycopg2://", "postgresql+asyncpg://")
        elif base_url.startswith("sqlite://"):
            return base_url.replace("sqlite://", "sqlite+aiosqlite://")
    
    return base_url

# ============================================================================
# Database Manager Classes
# ============================================================================

class DatabaseManager:
    """Менеджер синхронной базы данных."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_database_url("sync")
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=settings.DATABASE_POOL_SIZE or 10,
            max_overflow=settings.DATABASE_MAX_OVERFLOW or 20,
            pool_pre_ping=True,
            echo=settings.DEBUG,
            connect_args={"connect_timeout": 30} if "postgresql" in self.database_url else {}
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False
        )
        
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Получение сессии базы данных через контекстный менеджер."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Sync DB Session Error: {e}")
            raise
        finally:
            session.close()

    def create_tables(self):
        """Создание всех таблиц в базе данных."""
        Base.metadata.create_all(bind=self.engine)

    def health_check(self) -> bool:
        """Проверка подключения к базе данных."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def close_connections(self):
        """Закрытие всех подключений к базе данных."""
        try:
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Failed to close database connections: {e}")


class AsyncDatabaseManager:
    """Менеджер асинхронной базы данных."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_database_url("async")
        self.engine = create_async_engine(
            self.database_url,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=settings.DEBUG,
        )
        # Используем async_sessionmaker (нововведение в новых версиях)
        self.AsyncSessionLocal = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Получение асинхронной сессии базы данных."""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Async DB Session Error: {e}")
                raise
            # session.close() вызывается автоматически контекстным менеджером async with

    async def create_tables(self):
        """Создание всех таблиц в асинхронной базе данных."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def health_check(self) -> bool:
        """Асинхронная проверка подключения к базе данных."""
        try:
            async with self.AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False

    async def close_connections(self):
        """Закрытие всех асинхронных подключений к базе данных."""
        try:
            await self.engine.dispose()
            logger.info("Async database connections closed")
        except Exception as e:
            logger.error(f"Failed to close async database connections: {e}")


# ============================================================================
# Global Instances
# ============================================================================

# Создаем глобальные экземпляры менеджеров
db_manager = DatabaseManager()
async_db_manager = AsyncDatabaseManager()

# Алиас для обратной совместимости
Base_metadata = Base.metadata

# ============================================================================
# Dependency Injection for FastAPI
# ============================================================================

def get_db() -> Generator[Session, None, None]:
    """Dependency injection для FastAPI - синхронная сессия."""
    with db_manager.get_session() as session:
        yield session

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection для FastAPI - асинхронная сессия."""
    async with async_db_manager.get_session() as session:
        yield session

# ============================================================================
# SQLite Configuration
# ============================================================================

@event.listens_for(db_manager.engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Настройка SQLite для production."""
    if "sqlite" in str(db_manager.database_url):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# ============================================================================
# Auto-Init Logic (Refined)
# ============================================================================

def initialize_database():
    """Функция для явного вызова при старте приложения"""
    logger.info("Initializing database...")
    db_manager.create_tables()

# Убрали автоматический вызов при импорте!
# Лучше вызывать initialize_database() в main.py в событии startup