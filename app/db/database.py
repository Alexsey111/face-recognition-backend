"""
Инициализация базы данных.
Создание подключения, engine и session для работы с БД.
"""

from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from typing import Generator, AsyncGenerator
import asyncio
from contextlib import asynccontextmanager

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Создание базового класса для моделей
Base = declarative_base()

# Настройки подключения к БД
DATABASE_URL = settings.DATABASE_URL

# Создание sync engine для общих операций (временно отключено для Phase 2)
sync_database_url = settings.DATABASE_URL
# sync_engine = create_engine(
#     sync_database_url,
#     pool_pre_ping=True,
#     pool_recycle=300,
#     echo=settings.DEBUG,
# )

# Заглушка для Phase 2
sync_engine = None

# Создание async engine ОДИН раз при импорте
async_database_url = sync_database_url.replace("postgresql://", "postgresql+asyncpg://")
async_database_url = async_database_url.replace("sqlite:///", "sqlite+aiosqlite:///")

# Временно отключено для Phase 2
# async_engine = create_async_engine(
#     async_database_url,
#     pool_pre_ping=True,
#     pool_recycle=300,
#     echo=settings.DEBUG,
#     pool_size=10,
#     max_overflow=20,
# )

# Заглушка для Phase 2
async_engine = None

AsyncSessionMaker = None
# AsyncSessionMaker = sessionmaker(
#     async_engine,
#     class_=AsyncSession,
#     expire_on_commit=False
# )

# Создание SessionLocal (временно отключено)
SessionLocal = None
# SessionLocal = sessionmaker(
#     autocommit=False,
#     autoflush=False,
#     bind=sync_engine
# )


def get_db() -> Generator[Session, None, None]:
    """
    Получение сессии базы данных.

    Yields:
        Session: Сессия базы данных
    """
    # Временно отключено для Phase 2
    # db = SessionLocal()
    # try:
    #     yield db
    # finally:
    #     db.close()

    # Заглушка для Phase 2
    yield None


# Асинхронная сессия для работы с async/await
class AsyncSessionLocal:
    """
    Асинхронная сессия базы данных.
    """

    def __init__(self):
        self._session = None
        self._engine = None

    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход."""
        # Создаем асинхронный engine
        async_database_url = DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
        self._engine = create_async_engine(
            async_database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=settings.DEBUG,
        )

        # Создаем асинхронную сессию
        async_session = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        self._session = async_session()
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход."""
        if self._session:
            await self._session.close()
        if self._engine:
            await self._engine.dispose()


# Функция для получения асинхронной сессии
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Получение асинхронной сессии базы данных.

    Yields:
        AsyncSession: Асинхронная сессия базы данных
    """
    async with AsyncSessionLocal() as session:
        yield session


# Создание таблиц при импорте (для development) - временно отключено для Phase 2
# @event.listens_for(sync_engine, "connect")
# def set_sqlite_pragma(dbapi_connection, connection_record):
#     """Настройка SQLite для production."""
#     if "sqlite" in str(sync_database_url):
#         cursor = dbapi_connection.cursor()
#         cursor.execute("PRAGMA foreign_keys=ON")
#         cursor.close()


def create_tables():
    """
    Создание всех таблиц в базе данных.
    """
    try:
        Base.metadata.create_all(bind=sync_engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


def drop_tables():
    """
    Удаление всех таблиц из базы данных.
    """
    try:
        Base.metadata.drop_all(bind=sync_engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {str(e)}")
        raise


async def check_database_connection() -> bool:
    """Проверка подключения к базе данных."""
    try:
        async with AsyncSessionMaker() as session:
            await session.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False


def get_database_info() -> dict:
    """
    Получение информации о базе данных.

    Returns:
        dict: Информация о БД
    """
    try:
        # Получаем информацию из engine
        url = sync_engine.url
        driver = url.get_backend_name()

        return {
            "url": str(url),
            "driver": driver,
            "database": url.database if hasattr(url, "database") else None,
            "host": url.host if hasattr(url, "host") else None,
            "port": url.port if hasattr(url, "port") else None,
            "username": url.username if hasattr(url, "username") else None,
            "pool_size": sync_engine.pool.size(),
            "checked_in": sync_engine.pool.checkedin(),
            "checked_out": sync_engine.pool.checkedout(),
            "overflow": sync_engine.pool.overflow(),
            "invalid": sync_engine.pool.invalid(),
        }
    except Exception as e:
        logger.error(f"Failed to get database info: {str(e)}")
        return {}


class DatabaseManager:
    """
    Менеджер базы данных для управления подключениями.
    """

    def __init__(self):
        # Временно отключено для Phase 2
        # self.engine = sync_engine
        # self.SessionLocal = SessionLocal

        # Заглушки для Phase 2
        self.engine = None
        self.SessionLocal = None

    async def health_check(self) -> bool:
        """
        Проверка здоровья базы данных.

        Returns:
            bool: True если БД доступна
        """
        return await check_database_connection()

    def get_session(self) -> Session:
        """
        Получение сессии БД.

        Returns:
            Session: Сессия базы данных
        """
        return self.SessionLocal()

    async def get_async_session(self):
        """
        Получение асинхронной сессии БД.

        Yields:
            AsyncSession: Асинхронная сессия
        """
        async with AsyncSessionLocal() as session:
            yield session

    def execute_raw_sql(self, sql: str, params: dict = None):
        """
        Выполнение сырого SQL запроса.

        Args:
            sql: SQL запрос
            params: Параметры запроса
        """
        with self.engine.connect() as connection:
            if params:
                return connection.execute(sql, params)
            else:
                return connection.execute(sql)

    async def execute_raw_sql_async(self, sql: str, params: dict = None):
        """
        Выполнение сырого SQL запроса асинхронно.

        Args:
            sql: SQL запрос
            params: Параметры запроса
        """
        async with AsyncSessionLocal() as session:
            if params:
                result = await session.execute(sql, params)
            else:
                result = await session.execute(sql)
            await session.commit()
            return result

    def close_connections(self):
        """
        Закрытие всех подключений к базе данных.
        """
        try:
            self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Failed to close database connections: {str(e)}")


# Глобальный экземпляр менеджера БД
db_manager = DatabaseManager()
