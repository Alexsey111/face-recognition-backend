"""
Alembic Environment Configuration
Настройка окружения для миграций базы данных.
"""

import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

# Добавляем путь к корню проекта для импорта моделей
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Импортируем настройки и модели
from app.config import settings
from app.db.models import Base

# Конфигурация Alembic
config = context.config

# Настройка логирования
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Метаданные моделей для автогенерации
target_metadata = Base.metadata


def get_database_url():
    """Получение URL базы данных из настроек."""
    # НЕ преобразовываем в async для Alembic (он работает синхронно)
    db_url = settings.DATABASE_URL
    
    # Для PostgreSQL не добавляем asyncpg для Alembic
    # Для SQLite оставляем как есть (синхронный драйвер)
    return db_url


def run_migrations_offline() -> None:
    """Запуск миграций в 'оффлайн' режиме."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Запуск миграций в 'онлайн' режиме."""
    # Для синхронных миграций используем стандартный подход
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_database_url()  # Sync URL
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=True,  # Важно для SQLite
        )

        with context.begin_transaction():
            context.run_migrations()


# Синхронная версия для Alembic
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()