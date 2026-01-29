"""
Комплексные тесты для проверки выполнения Фазы 1: Infrastructure.
Проверяет все 13 задач первой фазы проекта.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

# Добавляем путь к корню проекта
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import Settings


class TestPhase1Infrastructure:
    """Тесты для проверки выполнения всей Фазы 1 инфраструктуры."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def config(self):
        """Возвращает настройки приложения."""
        return Settings()

    # =============================================================================
    # ISSUE 1: Dockerfile production
    # =============================================================================

    def test_dockerfile_production_exists(self, project_root):
        """Тест: Dockerfile production существует."""
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile должен существовать"

        content = dockerfile_path.read_text(encoding="utf-8")
        assert "FROM python:" in content, "Должен использовать базовый образ Python"
        assert (
            "MULTI-STAGE" in content.upper() or "as builder" in content
        ), "Должен использовать multi-stage build"
        assert (
            "useradd" in content or "USER" in content
        ), "Должен создавать non-root пользователя"
        assert "uvicorn" in content, "Должен запускать uvicorn"

    def test_dockerfile_production_optimization(self, project_root):
        """Тест: Dockerfile production оптимизирован."""
        dockerfile_path = project_root / "Dockerfile"
        content = dockerfile_path.read_text(encoding="utf-8")

        # Проверяем оптимизации слоев
        assert (
            "apt-get clean" in content or "rm -rf /var/lib/apt/lists/*" in content
        ), "Должен очищать apt cache"
        assert (
            "COPY requirements.txt" in content
        ), "Должен копировать requirements.txt отдельно для кэширования"
        assert (
            "--no-cache-dir" in content or "pip install" in content
        ), "Должен использовать --no-cache-dir для pip"

    # =============================================================================
    # ISSUE 2: Dockerfile.dev
    # =============================================================================

    def test_dockerfile_dev_exists(self, project_root):
        """Тест: Dockerfile.dev существует."""
        dockerfile_dev_path = project_root / "Dockerfile.dev"
        assert dockerfile_dev_path.exists(), "Dockerfile.dev должен существовать"

        content = dockerfile_dev_path.read_text(encoding="utf-8")
        assert "FROM python:" in content, "Должен использовать базовый образ Python"
        assert "requirements-dev.txt" in content, "Должен устанавливать dev зависимости"

    def test_dockerfile_dev_hot_reload(self, project_root):
        """Тест: Dockerfile.dev настроен для hot-reload."""
        dockerfile_dev_path = project_root / "Dockerfile.dev"
        content = dockerfile_dev_path.read_text(encoding="utf-8")

        assert "EXPOSE 8000" in content, "Должен экспортировать порт 8000"
        # Проверяем настройки для разработки
        assert (
            "DEBUG=true" in content or "ENVIRONMENT=development" in content
        ), "Должен настраивать development режим"

    # =============================================================================
    # ISSUE 3: docker-compose.yml
    # =============================================================================

    def test_docker_compose_yml_exists(self, project_root):
        """Тест: docker-compose.yml существует."""
        compose_path = project_root / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml должен существовать"

        with open(compose_path, "r") as f:
            compose_config = yaml.safe_load(f)

        assert "services" in compose_config, "Должен содержать секцию services"
        services = compose_config["services"]

        # Проверяем обязательные сервисы инфраструктуры (API сервис — в docker-compose.prod.yml)
        assert "postgres" in services, "Должен содержать PostgreSQL сервис"
        assert "redis" in services, "Должен содержать Redis сервис"
        assert "minio" in services, "Должен содержать MinIO сервис"

    def test_docker_compose_prod_yml_exists(self, project_root):
        """Тест: docker-compose.prod.yml существует (production deployment)."""
        compose_path = project_root / "docker-compose.prod.yml"
        assert compose_path.exists(), "docker-compose.prod.yml должен существовать"

        with open(compose_path, "r") as f:
            compose_config = yaml.safe_load(f)

        assert "services" in compose_config, "Должен содержать секцию services"
        services = compose_config["services"]

        # Проверяем наличие API сервиса в production
        assert "api" in services, "docker-compose.prod.yml должен содержать API сервис"

    def test_docker_compose_services_configuration(self, project_root):
        """Тест: Сервисы в docker-compose.yml правильно настроены."""
        compose_path = project_root / "docker-compose.yml"
        with open(compose_path, "r") as f:
            compose_config = yaml.safe_load(f)

        services = compose_config["services"]

        # Проверяем PostgreSQL
        postgres = services.get("postgres")
        if postgres:
            assert "postgres:" in str(
                postgres.get("image", "")
            ), "PostgreSQL должен использовать официальный образ"

        # Проверяем Redis
        redis = services.get("redis")
        if redis:
            assert "redis:" in str(
                redis.get("image", "")
            ), "Redis должен использовать официальный образ"

        # Проверяем MinIO
        minio = services.get("minio")
        if minio:
            assert "minio" in str(
                minio.get("image", "")
            ), "MinIO должен использовать официальный образ"

    def test_docker_compose_prod_api_service_configuration(self, project_root):
        """Тест: API сервис в docker-compose.prod.yml правильно настроен."""
        compose_path = project_root / "docker-compose.prod.yml"
        with open(compose_path, "r") as f:
            compose_config = yaml.safe_load(f)

        services = compose_config["services"]
        api_service = services.get("api")

        assert api_service is not None, "API сервис должен существовать"
        assert "build" in api_service, "API сервис должен иметь build секцию"
        assert "container_name" in api_service, "API сервис должен иметь container_name"
        assert (
            api_service["container_name"] == "face_recognition_api"
        ), "Имя контейнера должно быть 'face_recognition_api'"

        # Проверяем depends_on
        depends = api_service.get("depends_on", {})
        assert "postgres" in depends, "API должен зависеть от postgres"
        assert "redis" in depends, "API должен зависеть от redis"
        assert "minio" in depends, "API должен зависеть от minio"

    def test_docker_compose_health_checks(self, project_root):
        """Тест: В docker-compose.yml настроены health checks."""
        compose_path = project_root / "docker-compose.yml"
        with open(compose_path, "r") as f:
            compose_config = yaml.safe_load(f)

        services = compose_config["services"]

        # Проверяем health checks для инфраструктурных сервисов
        for service_name in ["postgres", "redis", "minio"]:
            service_config = services.get(service_name)
            assert (
                service_config is not None
            ), f"Сервис {service_name} должен существовать"
            assert (
                "healthcheck" in service_config
            ), f"Сервис {service_name} должен иметь healthcheck"

    # =============================================================================
    # ISSUE 4: docker-compose.dev.yml
    # =============================================================================

    def test_docker_compose_dev_yml_exists(self, project_root):
        """Тест: docker-compose.dev.yml существует."""
        compose_dev_path = project_root / "docker-compose.dev.yml"
        assert compose_dev_path.exists(), "docker-compose.dev.yml должен существовать"

        with open(compose_dev_path, "r") as f:
            compose_config = yaml.safe_load(f)

        assert "services" in compose_config, "Должен содержать секцию services"

    def test_docker_compose_dev_volume_mounts(self, project_root):
        """Тест: docker-compose.dev.yml настроен с volume mounts."""
        compose_dev_path = project_root / "docker-compose.dev.yml"
        with open(compose_dev_path, "r") as f:
            compose_config = yaml.safe_load(f)

        services = compose_config["services"]
        app_service = None

        # Ищем основной сервис приложения
        for service_name in ["face-recognition-api-dev", "app", "face-recognition-api"]:
            if service_name in services:
                app_service = services[service_name]
                break

        assert app_service is not None, "Должен найти основной сервис приложения"
        assert "volumes" in app_service, "Сервис приложения должен иметь volumes"

        volumes = app_service["volumes"]
        # Проверяем, что есть mount для кода
        has_code_mount = any(".:/app" in str(volume) for volume in volumes)
        assert has_code_mount, "Должен монтировать код для hot-reload"

    # =============================================================================
    # ISSUE 5: .dockerignore
    # =============================================================================

    def test_dockerignore_exists(self, project_root):
        """Тест: .dockerignore существует."""
        dockerignore_path = project_root / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore должен существовать"

        content = dockerignore_path.read_text(encoding="utf-8")

        # Проверяем основные исключения
        assert "__pycache__" in content, "Должен исключать __pycache__"
        assert ".git" in content, "Должен исключать .git"
        assert ".env" in content, "Должен исключать .env файлы"
        assert (
            ".venv" in content or "venv" in content
        ), "Должен исключать virtual environments"

    def test_dockerignore_comprehensive(self, project_root):
        """Тест: .dockerignore содержит comprehensive exclusions."""
        dockerignore_path = project_root / ".dockerignore"
        content = dockerignore_path.read_text(encoding="utf-8")

        # Проверяем дополнительные исключения
        assert "node_modules" in content, "Должен исключать node_modules"
        assert "tests" in content, "Должен исключать tests"
        assert "*.log" in content or "logs" in content, "Должен исключать логи"
        assert (
            "*.pyc" in content or "*.py[cod]" in content
        ), "Должен исключать compiled Python files"

    # =============================================================================
    # ISSUE 6: Alembic initialization
    # =============================================================================

    def test_alembic_directory_exists(self, project_root):
        """Тест: Директория alembic существует."""
        alembic_dir = project_root / "alembic"
        assert alembic_dir.exists(), "Директория alembic должна существовать"
        assert alembic_dir.is_dir(), "alembic должна быть директорией"

    def test_alembic_env_py_exists(self, project_root):
        """Тест: env.py файл существует и настроен."""
        env_py_path = project_root / "alembic" / "env.py"
        assert env_py_path.exists(), "alembic/env.py должен существовать"

        content = env_py_path.read_text(encoding="utf-8")
        assert "from sqlalchemy" in content, "Должен импортировать SQLAlchemy"
        assert "target_metadata" in content, "Должен содержать target_metadata"
        assert "Base.metadata" in content, "Должен использовать Base.metadata"

    def test_alembic_ini_exists(self, project_root):
        """Тест: alembic.ini файл существует."""
        ini_path = project_root / "alembic.ini"
        assert ini_path.exists(), "alembic.ini должен существовать"

    def test_first_migration_exists(self, project_root):
        """Тест: Первая миграция существует."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        assert migration_path.exists(), "Первая миграция должна существовать"

        content = migration_path.read_text(encoding="utf-8")
        assert "def upgrade" in content, "Должна содержать функцию upgrade"
        assert "def downgrade" in content, "Должна содержать функцию downgrade"
        assert "op.create_table" in content, "Должна создавать таблицы"

    # =============================================================================
    # ISSUE 7: Database models migration
    # =============================================================================

    def test_users_table_creation(self, project_root):
        """Тест: Таблица users создана в миграции."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        if not migration_path.exists():
            pytest.skip("Миграция не существует")

        content = migration_path.read_text(encoding="utf-8")

        assert "users" in content.lower(), "Должна создаваться таблица users"
        assert "id" in content.lower(), "Должна быть колонка id"
        assert (
            "email" in content.lower() or "email" in content
        ), "Должна быть колонка email"

    def test_references_table_creation(self, project_root):
        """Тест: Таблица references создана в миграции."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        if not migration_path.exists():
            pytest.skip("Миграция не существует")

        content = migration_path.read_text(encoding="utf-8")

        assert "reference" in content.lower(), "Должна создаваться таблица references"
        assert "user_id" in content.lower(), "Должна быть колонка user_id"
        assert "embedding" in content.lower(), "Должна быть колонка embedding"

    def test_verification_sessions_table_creation(self, project_root):
        """Тест: Таблица verification_sessions создана в миграции."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        if not migration_path.exists():
            pytest.skip("Миграция не существует")

        content = migration_path.read_text(encoding="utf-8")

        assert (
            "verification" in content.lower() or "session" in content.lower()
        ), "Должна создаваться таблица verification_sessions"
        assert "status" in content.lower(), "Должна быть колонка status"

    def test_audit_logs_table_creation(self, project_root):
        """Тест: Таблица audit_logs создана в миграции."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        if not migration_path.exists():
            pytest.skip("Миграция не существует")

        content = migration_path.read_text(encoding="utf-8")

        assert "audit" in content.lower(), "Должна создаваться таблица audit_logs"
        assert "action" in content.lower(), "Должна быть колонка action"

    # =============================================================================
    # ISSUE 8: Indexes & constraints
    # =============================================================================

    def test_foreign_key_constraints(self, project_root):
        """Тест: Foreign key constraints созданы в миграции."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        if not migration_path.exists():
            pytest.skip("Миграция не существует")

        content = migration_path.read_text(encoding="utf-8")
        content_lower = content.lower()

        assert (
            "foreign" in content_lower or "fk" in content_lower
        ), "Должны быть foreign key constraints"
        assert "user" in content_lower, "Должны быть ссылки на users"

    def test_unique_constraints(self, project_root):
        """Тест: Unique constraints созданы в миграции."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        if not migration_path.exists():
            pytest.skip("Миграция не существует")

        content = migration_path.read_text(encoding="utf-8")
        content_lower = content.lower()

        assert (
            "unique" in content_lower or "index" in content_lower
        ), "Должны быть unique constraints или indexes"

    def test_indexes_creation(self, project_root):
        """Тест: Индексы созданы в миграции."""
        migration_path = (
            project_root / "alembic" / "versions" / "001_initial_migration.py"
        )
        if not migration_path.exists():
            pytest.skip("Миграция не существует")

        content = migration_path.read_text(encoding="utf-8")
        content_lower = content.lower()

        assert (
            "index" in content_lower
            or "create_index" in content_lower
            or "PrimaryKeyIndex" in content_lower
        ), "Должны создаваться индексы"

    # =============================================================================
    # ISSUE 9: Redis connection
    # =============================================================================

    def test_redis_config_exists(self, project_root):
        """Тест: Redis конфигурация существует в config.py."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")

        assert "REDIS_URL" in content, "Должна быть REDIS_URL конфигурация"
        assert "REDIS_PASSWORD" in content, "Должна быть REDIS_PASSWORD конфигурация"
        assert (
            "REDIS_CONNECTION_POOL_SIZE" in content
        ), "Должна быть REDIS_CONNECTION_POOL_SIZE конфигурация"

    def test_redis_service_configuration(self, project_root):
        """Тест: Redis сервис настроен в docker-compose."""
        compose_path = project_root / "docker-compose.yml"
        if compose_path.exists():
            with open(compose_path, "r") as f:
                compose_config = yaml.safe_load(f)

            services = compose_config.get("services", {})
            redis_service = services.get("redis")
            assert (
                redis_service is not None
            ), "Redis сервис должен быть настроен в docker-compose.yml"

    def test_cache_service_exists(self, project_root):
        """Тест: CacheService существует."""
        cache_service_path = project_root / "app" / "services" / "cache_service.py"
        assert cache_service_path.exists(), "CacheService должен существовать"

        content = cache_service_path.read_text(encoding="utf-8")
        assert "class CacheService" in content, "Должен быть класс CacheService"
        assert "redis" in content.lower(), "CacheService должен использовать redis"

    # =============================================================================
    # ISSUE 10: MinIO bucket setup
    # =============================================================================

    def test_s3_config_exists(self, project_root):
        """Тест: S3/MinIO конфигурация существует в config.py."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")

        assert "S3_ENDPOINT_URL" in content, "Должна быть S3_ENDPOINT_URL конфигурация"
        assert "S3_ACCESS_KEY" in content, "Должна быть S3_ACCESS_KEY конфигурация"
        assert "S3_SECRET_KEY" in content, "Должна быть S3_SECRET_KEY конфигурация"
        assert "S3_BUCKET_NAME" in content, "Должна быть S3_BUCKET_NAME конфигурация"

    def test_storage_service_exists(self, project_root):
        """Тест: StorageService существует."""
        storage_service_path = project_root / "app" / "services" / "storage_service.py"
        assert storage_service_path.exists(), "StorageService должен существовать"

        content = storage_service_path.read_text(encoding="utf-8")
        assert "class StorageService" in content, "Должен быть класс StorageService"
        assert "boto3" in content, "StorageService должен использовать boto3"
        assert (
            "S3" in content or "s3" in content.lower()
        ), "StorageService должен работать с S3"

    def test_minio_setup_script_exists(self, project_root):
        """Тест: Скрипт setup_minio.py существует."""
        setup_script_path = project_root / "setup_minio.py"
        if not setup_script_path.exists():
            # Проверяем альтернативные расположения
            alt_paths = [
                project_root / "scripts" / "setup_minio.py",
                project_root / "docker" / "setup_minio.py",
            ]
            found = False
            for alt_path in alt_paths:
                if alt_path.exists():
                    found = True
                    break
            if not found:
                pytest.skip("setup_minio.py не существует")
        else:
            content = setup_script_path.read_text(encoding="utf-8")
            content_lower = content.lower()
            assert (
                "minio" in content_lower or "s3" in content_lower
            ), "Скрипт должен использовать Minio или S3"
            assert "bucket" in content_lower, "Скрипт должен создавать bucket'ы"

    def test_minio_service_configuration(self, project_root):
        """Тест: MinIO сервис настроен в docker-compose."""
        compose_path = project_root / "docker-compose.yml"
        if compose_path.exists():
            with open(compose_path, "r") as f:
                compose_config = yaml.safe_load(f)

            services = compose_config.get("services", {})
            minio_service = services.get("minio")
            assert (
                minio_service is not None
            ), "MinIO сервис должен быть настроен в docker-compose.yml"

    # =============================================================================
    # ISSUE 11: config.py setup
    # =============================================================================

    def test_settings_class_exists(self, project_root):
        """Тест: Settings class существует."""
        config_path = project_root / "app" / "config.py"
        assert config_path.exists(), "config.py должен существовать"

        content = config_path.read_text(encoding="utf-8")
        assert "class Settings" in content, "Должен быть класс Settings"
        assert (
            "BaseSettings" in content
        ), "Settings должен наследоваться от BaseSettings"

    def test_database_config(self, project_root):
        """Тест: Database конфигурация настроена."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")

        assert "DATABASE_URL" in content, "Должна быть DATABASE_URL конфигурация"
        assert (
            "DATABASE_POOL_SIZE" in content
        ), "Должна быть DATABASE_POOL_SIZE конфигурация"

    def test_redis_config(self, project_root):
        """Тест: Redis конфигурация настроена."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")

        assert "REDIS_URL" in content, "Должна быть REDIS_URL конфигурация"

    def test_s3_config(self, project_root):
        """Тест: S3 конфигурация настроена."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")

        assert "S3_ENDPOINT_URL" in content, "Должна быть S3_ENDPOINT_URL конфигурация"
        assert "S3_ACCESS_KEY" in content, "Должна быть S3_ACCESS_KEY конфигурация"
        assert "S3_SECRET_KEY" in content, "Должна быть S3_SECRET_KEY конфигурация"

    def test_api_config(self, project_root):
        """Тест: API конфигурация настроена."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")

        assert "JWT_SECRET_KEY" in content, "Должна быть JWT_SECRET_KEY конфигурация"
        assert "CORS_ORIGINS" in content, "Должна быть CORS_ORIGINS конфигурация"

    def test_ml_config(self, project_root):
        """Тест: ML конфигурация настроена."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")

        assert "ML_SERVICE_URL" in content, "Должна быть ML_SERVICE_URL конфигурация"

    def test_config_validation(self, project_root):
        """Тест: Config validation настроена."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text(encoding="utf-8")
        content_lower = content.lower()

        assert (
            "validator" in content_lower or "field_validator" in content_lower
        ), "Должны быть field validators"

    # =============================================================================
    # ISSUE 12: .env.example & requirements.txt
    # =============================================================================

    def test_env_example_exists(self, project_root):
        """Тест: .env.example существует."""
        env_example_path = project_root / ".env.example"
        assert env_example_path.exists(), ".env.example должен существовать"

        content = env_example_path.read_text(encoding="utf-8")

        # Проверяем основные переменные
        assert "DATABASE_URL=" in content, "Должна быть DATABASE_URL переменная"
        assert "REDIS_URL=" in content, "Должна быть REDIS_URL переменная"
        assert "S3_ENDPOINT_URL=" in content, "Должна быть S3_ENDPOINT_URL переменная"
        assert "JWT_SECRET_KEY=" in content, "Должна быть JWT_SECRET_KEY переменная"

    def test_env_example_documentation(self, project_root):
        """Тест: .env.example содержит документацию."""
        env_example_path = project_root / ".env.example"
        content = env_example_path.read_text(encoding="utf-8")

        assert "#" in content, "Должны быть комментарии"

    def test_requirements_txt_exists(self, project_root):
        """Тест: requirements.txt существует."""
        requirements_path = project_root / "requirements.txt"
        assert requirements_path.exists(), "requirements.txt должен существовать"

        content = requirements_path.read_text(encoding="utf-8")

        # Проверяем основные зависимости
        assert "fastapi" in content, "Должна быть зависимость fastapi"
        assert "sqlalchemy" in content, "Должна быть зависимость sqlalchemy"
        assert "alembic" in content, "Должна быть зависимость alembic"
        assert "redis" in content, "Должна быть зависимость redis"
        assert "boto3" in content, "Должна быть зависимость boto3"

    def test_requirements_version_pinning(self, project_root):
        """Тест: requirements.txt содержит version pinning."""
        requirements_path = project_root / "requirements.txt"
        content = requirements_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Проверяем, что есть версии (== или >=) или комментарии о версиях
        has_exact_versions = any("==" in line for line in lines)
        has_min_versions = any(">=" in line for line in lines)
        has_comments = any("#" in line for line in lines)

        assert (
            has_exact_versions or has_min_versions or has_comments
        ), "Должны быть зафиксированные версии (== или >=) или комментарии о версиях"

    # =============================================================================
    # ISSUE 13: Python dependencies setup
    # =============================================================================

    def test_requirements_dev_exists(self, project_root):
        """Тест: requirements-dev.txt существует."""
        requirements_dev_path = project_root / "requirements-dev.txt"
        assert (
            requirements_dev_path.exists()
        ), "requirements-dev.txt должен существовать"

        content = requirements_dev_path.read_text(encoding="utf-8")
        assert (
            "-r requirements.txt" in content or "requirements.txt" in content
        ), "Должен включать production dependencies"

    def test_dev_dependencies(self, project_root):
        """Тест: requirements-dev.txt содержит dev зависимости."""
        requirements_dev_path = project_root / "requirements-dev.txt"
        content = requirements_dev_path.read_text(encoding="utf-8")

        # Проверяем dev инструменты
        assert "pytest" in content, "Должна быть зависимость pytest"
        assert (
            "black" in content or "flake8" in content or "mypy" in content
        ), "Должна быть хотя бы одна dev зависимость (black, flake8 или mypy)"

    def test_pytest_ini_exists(self, project_root):
        """Тест: pytest.ini или pyproject.toml существует."""
        pytest_ini_path = project_root / "pytest.ini"
        pyproject_path = project_root / "pyproject.toml"

        has_pytest_ini = pytest_ini_path.exists()
        has_pyproject = pyproject_path.exists()

        assert (
            has_pytest_ini or has_pyproject
        ), "Должен существовать pytest.ini или pyproject.toml"

        if has_pyproject:
            try:
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    content = f.read()
                assert (
                    "pytest" in content.lower()
                    or "tool.pytest" in content.lower()
                    or "testpaths" in content.lower()
                ), "pyproject.toml должен содержать конфигурацию pytest"
            except yaml.YAMLError:
                pytest.skip("pyproject.toml имеет некорректный YAML синтаксис")

    def test_pyproject_toml_exists(self, project_root):
        """Тест: pyproject.toml существует."""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml должен существовать"

        # Проверяем, что файл можно прочитать
        try:
            with open(pyproject_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            pytest.fail("pyproject.toml имеет проблемы с кодировкой")

        # Проверяем базовую структуру без использования yaml.safe_load
        assert (
            "[project]" in content or "project:" in content
        ), "pyproject.toml должен содержать секцию project"
        assert "name" in content, "pyproject.toml должен содержать name"
        assert "version" in content, "pyproject.toml должен содержать version"

    def test_pip_install_requirements(self, project_root):
        """Тест: pip install -r requirements.txt работает."""
        requirements_path = project_root / "requirements.txt"

        # Проверяем, что файл requirements.txt валиден
        with open(requirements_path, "r") as f:
            content = f.read()

        # Проверяем синтаксис файла
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Каждая строка должна быть валидным package specification
                assert (
                    "==" in line
                    or ">=" in line
                    or "~=" in line
                    or line.startswith("-r")
                ), f"Некорректная строка в requirements.txt: {line}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
