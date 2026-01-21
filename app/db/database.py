# app/db/database.py
"""
PostgreSQL Database Manager (asyncpg + psycopg2 для Alembic).
✅ Удален весь код для SQLite.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase

try:
    from ..middleware.metrics import (
        database_connections_active as DATABASE_CONNECTIONS_ACTIVE,
    )
except Exception:
    DATABASE_CONNECTIONS_ACTIVE = None
from typing import AsyncGenerator
from contextlib import asynccontextmanager
import os

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


# Базовый класс для всех моделей
class Base(DeclarativeBase):
    pass


# ============================================================================
# PostgreSQL Database Manager (только asyncpg)
# ============================================================================
class DatabaseManager:
    """Менеджер для работы с PostgreSQL через asyncpg."""

    def __init__(self, database_url: str | None = None):
        self._provided_database_url = database_url
        self.database_url = None
        self.engine = None
        self.SessionLocal = None
        self._engine_initialized = False

    def _ensure_engine(self):
        """Ленивая инициализация async engine для PostgreSQL."""
        if self._engine_initialized:
            return

        # Используем предоставленный URL или из settings
        base_url = self._provided_database_url or settings.DATABASE_URL

        # ✅ Проверка, что это PostgreSQL
        if not base_url.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                f"❌ Only PostgreSQL is supported!\n"
                f"Got: {base_url[:50]}...\n"
                f"Please configure DATABASE_URL with postgresql:// or postgres://"
            )

        # Преобразуем в async URL (asyncpg)
        async_url = base_url
        if async_url.startswith("postgresql://"):
            async_url = async_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif async_url.startswith("postgres://"):
            async_url = async_url.replace("postgres://", "postgresql+asyncpg://", 1)

        self.database_url = async_url

        # Настройка async engine для PostgreSQL
        self.engine = create_async_engine(
            self.database_url,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_timeout=settings.DATABASE_POOL_TIMEOUT,
            pool_recycle=settings.DATABASE_POOL_RECYCLE,
            pool_pre_ping=True,
            echo=settings.DEBUG,
            # PostgreSQL specific optimizations
            connect_args={
                "server_settings": {
                    "application_name": "face_recognition_service",
                    "jit": "off",  # Отключаем JIT для стабильности
                },
                "command_timeout": 60,
                "timeout": 10,
            },
        )

        self.SessionLocal = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        self._engine_initialized = True
        logger.info(
            f"✅ PostgreSQL engine initialized: {self.database_url.split('@')[-1]}"
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Безопасное получение сессии с автоматическим commit/rollback."""
        self._ensure_engine()

        # Metrics tracking - определяем доступность метрик один раз
        metrics_available = DATABASE_CONNECTIONS_ACTIVE is not None
        if metrics_available:
            try:
                DATABASE_CONNECTIONS_ACTIVE.inc()
            except Exception:
                metrics_available = False

        async with self.SessionLocal() as session:
            try:
                yield session
                # ✅ Commit только если были изменения
                if session.dirty or session.new or session.deleted:
                    await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"❌ Database session error: {e}")
                raise
            finally:
                # ✅ Декремент только если инкремент был успешен
                if metrics_available:
                    try:
                        DATABASE_CONNECTIONS_ACTIVE.dec()
                    except Exception:
                        pass

    async def create_tables(self):
        """Создание таблиц (для тестов или первого запуска)."""
        self._ensure_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ PostgreSQL tables created")

    async def drop_tables(self):
        """Удаление всех таблиц (для тестов)."""
        self._ensure_engine()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("⚠️  PostgreSQL tables dropped")

    async def health_check(self) -> bool:
        """Проверка подключения к PostgreSQL."""
        try:
            self._ensure_engine()
            async with self.SessionLocal() as session:
                result = await session.execute(text("SELECT 1"))
                result.scalar_one()
            logger.debug("✅ PostgreSQL health check passed")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL health check failed: {e}")
            return False

    async def close(self):
        """Закрытие всех подключений."""
        if self.engine is not None:
            await self.engine.dispose()
            logger.info("✅ PostgreSQL connections closed")


# ============================================================================
# Глобальный экземпляр менеджера
# ============================================================================
db_manager = DatabaseManager()

# Backwards compatibility - expose engine directly
engine = None


def _get_engine():
    """Ленивое получение engine для обратной совместимости."""
    global engine
    if engine is None:
        db_manager._ensure_engine()
        engine = db_manager.engine
    return engine


# Expose engine for backwards compatibility
engine = _get_engine()


# ============================================================================
# Dependency для FastAPI
# ============================================================================
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Зависимость для маршрутов FastAPI."""
    async with db_manager.get_session() as session:
        yield session


# Backwards compatibility
get_async_db = get_db


def get_async_db_manager() -> DatabaseManager:
    """Возврат глобального DatabaseManager."""
    return db_manager
