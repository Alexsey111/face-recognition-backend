# alembic/env.py
"""
Alembic Environment Configuration (PostgreSQL Only).
✅ Удален весь код для SQLite.
"""

import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Добавляем путь к корню проекта
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
    """✅ Получение sync PostgreSQL URL для Alembic."""
    db_url = settings.sync_database_url
    
    # Проверка, что это PostgreSQL
    if not db_url.startswith(("postgresql://", "postgres://")):
        raise ValueError(
            f"❌ Alembic requires PostgreSQL!\n"
            f"Got: {db_url[:50]}...\n"
            f"Configure DATABASE_URL with postgresql:// URL"
        )
    
    return db_url


def run_migrations_offline() -> None:
    """Запуск миграций в offline режиме."""
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
    """Запуск миграций в online режиме (PostgreSQL)."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_database_url()
    
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
            # ✅ УДАЛЕНО: render_as_batch (только для SQLite)
        )

        with context.begin_transaction():
            context.run_migrations()


# Запуск миграций
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
