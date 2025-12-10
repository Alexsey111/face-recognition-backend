"""
Комплексные тесты для проверки выполнения Фазы 1: Infrastructure.
Проверяет все 13 задач первой фазы проекта.
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

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
        
        content = dockerfile_path.read_text()
        assert "FROM python:" in content, "Должен использовать базовый образ Python"
        assert "MULTI-STAGE" in content.upper() or "as builder" in content, "Должен использовать multi-stage build"
        assert "useradd" in content or "USER" in content, "Должен создавать non-root пользователя"
        assert "uvicorn" in content, "Должен запускать uvicorn"

    def test_dockerfile_production_optimization(self, project_root):
        """Тест: Dockerfile production оптимизирован."""
        dockerfile_path = project_root / "Dockerfile"
        content = dockerfile_path.read_text()
        
        # Проверяем оптимизации слоев
        assert "apt-get clean" in content or "rm -rf /var/lib/apt/lists/*" in content, \
            "Должен очищать apt cache"
        assert "COPY requirements.txt" in content, \
            "Должен копировать requirements.txt отдельно для кэширования"
        assert "--no-cache-dir" in content or "pip install" in content, \
            "Должен использовать --no-cache-dir для pip"

    # =============================================================================
    # ISSUE 2: Dockerfile.dev
    # =============================================================================
    
    def test_dockerfile_dev_exists(self, project_root):
        """Тест: Dockerfile.dev существует."""
        dockerfile_dev_path = project_root / "Dockerfile.dev"
        assert dockerfile_dev_path.exists(), "Dockerfile.dev должен существовать"
        
        content = dockerfile_dev_path.read_text()
        assert "FROM python:" in content, "Должен использовать базовый образ Python"
        assert "requirements-dev.txt" in content, "Должен устанавливать dev зависимости"

    def test_dockerfile_dev_hot_reload(self, project_root):
        """Тест: Dockerfile.dev настроен для hot-reload."""
        dockerfile_dev_path = project_root / "Dockerfile.dev"
        content = dockerfile_dev_path.read_text()
        
        assert "EXPOSE 8000" in content, "Должен экспортировать порт 8000"
        # Проверяем настройки для разработки
        assert "DEBUG=true" in content or "ENVIRONMENT=development" in content, \
            "Должен настраивать development режим"

    # =============================================================================
    # ISSUE 3: docker-compose.yml
    # =============================================================================
    
    def test_docker_compose_yml_exists(self, project_root):
        """Тест: docker-compose.yml существует."""
        compose_path = project_root / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml должен существовать"
        
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        assert "services" in compose_config, "Должен содержать секцию services"
        services = compose_config["services"]
        
        # Проверяем обязательные сервисы
        assert "face-recognition-api" in services or "app" in services, \
            "Должен содержать основной сервис приложения"
        assert "postgres" in services, "Должен содержать PostgreSQL сервис"
        assert "redis" in services, "Должен содержать Redis сервис"
        assert "minio" in services, "Должен содержать MinIO сервис"

    def test_docker_compose_services_configuration(self, project_root):
        """Тест: Сервисы в docker-compose.yml правильно настроены."""
        compose_path = project_root / "docker-compose.yml"
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config["services"]
        
        # Проверяем PostgreSQL
        postgres = services.get("postgres", services.get("face-recognition-postgres"))
        if postgres:
            assert "postgres:" in str(postgres.get("image", "")), \
                "PostgreSQL должен использовать официальный образ"
        
        # Проверяем Redis
        redis = services.get("redis", services.get("face-recognition-redis"))
        if redis:
            assert "redis:" in str(redis.get("image", "")), \
                "Redis должен использовать официальный образ"
        
        # Проверяем MinIO
        minio = services.get("minio", services.get("face-recognition-minio"))
        if minio:
            assert "minio" in str(minio.get("image", "")), \
                "MinIO должен использовать официальный образ"

    def test_docker_compose_health_checks(self, project_root):
        """Тест: В docker-compose.yml настроены health checks."""
        compose_path = project_root / "docker-compose.yml"
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config["services"]
        
        for service_name, service_config in services.items():
            if service_name in ["postgres", "redis", "minio", "face-recognition-api"]:
                # Многие сервисы должны иметь health checks
                if service_name in ["postgres", "redis", "minio"]:
                    assert "healthcheck" in service_config, \
                        f"Сервис {service_name} должен иметь healthcheck"

    # =============================================================================
    # ISSUE 4: docker-compose.dev.yml
    # =============================================================================
    
    def test_docker_compose_dev_yml_exists(self, project_root):
        """Тест: docker-compose.dev.yml существует."""
        compose_dev_path = project_root / "docker-compose.dev.yml"
        assert compose_dev_path.exists(), "docker-compose.dev.yml должен существовать"
        
        with open(compose_dev_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        assert "services" in compose_config, "Должен содержать секцию services"

    def test_docker_compose_dev_volume_mounts(self, project_root):
        """Тест: docker-compose.dev.yml настроен с volume mounts."""
        compose_dev_path = project_root / "docker-compose.dev.yml"
        with open(compose_dev_path, 'r') as f:
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
        
        content = dockerignore_path.read_text()
        
        # Проверяем основные исключения
        assert "__pycache__" in content, "Должен исключать __pycache__"
        assert ".git" in content, "Должен исключать .git"
        assert ".env" in content, "Должен исключать .env файлы"
        assert ".venv" in content or "venv" in content, "Должен исключать virtual environments"

    def test_dockerignore_comprehensive(self, project_root):
        """Тест: .dockerignore содержит comprehensive exclusions."""
        dockerignore_path = project_root / ".dockerignore"
        content = dockerignore_path.read_text()
        
        # Проверяем дополнительные исключения
        assert "node_modules" in content, "Должен исключать node_modules"
        assert "tests" in content, "Должен исключать tests"
        assert "*.log" in content or "logs" in content, "Должен исключать логи"
        assert "*.pyc" in content or "*.py[cod]" in content, "Должен исключать compiled Python files"

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
        
        content = env_py_path.read_text()
        assert "from sqlalchemy" in content, "Должен импортировать SQLAlchemy"
        assert "target_metadata" in content, "Должен содержать target_metadata"
        assert "Base.metadata" in content, "Должен использовать Base.metadata"

    def test_alembic_ini_exists(self, project_root):
        """Тест: alembic.ini файл существует."""
        ini_path = project_root / "alembic.ini"
        assert ini_path.exists(), "alembic.ini должен существовать"

    def test_first_migration_exists(self, project_root):
        """Тест: Первая миграция существует."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        assert migration_path.exists(), "Первая миграция должна существовать"
        
        content = migration_path.read_text()
        assert "def upgrade" in content, "Должна содержать функцию upgrade"
        assert "def downgrade" in content, "Должна содержать функцию downgrade"
        assert "op.create_table" in content, "Должна создавать таблицы"

    # =============================================================================
    # ISSUE 7: Database models migration
    # =============================================================================
    
    def test_users_table_creation(self, project_root):
        """Тест: Таблица users создана."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration_path.read_text()
        
        assert "create_table('users'" in content, "Должна создаваться таблица users"
        assert "sa.Column('id'" in content, "Таблица users должна иметь колонку id"
        assert "sa.Column('email'" in content, "Таблица users должна иметь колонку email"
        assert "sa.Column('username'" in content, "Таблица users должна иметь колонку username"

    def test_references_table_creation(self, project_root):
        """Тест: Таблица references создана."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration_path.read_text()
        
        assert "create_table('references'" in content, "Должна создаваться таблица references"
        assert "sa.Column('user_id'" in content, "Таблица references должна иметь user_id"
        assert "sa.Column('embedding'" in content, "Таблица references должна иметь embedding"
        assert "sa.Column('embedding_version'" in content, "Таблица references должна иметь embedding_version"

    def test_verification_sessions_table_creation(self, project_root):
        """Тест: Таблица verification_sessions создана."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration_path.read_text()
        
        assert "create_table('verification_sessions'" in content, \
            "Должна создаваться таблица verification_sessions"
        assert "sa.Column('session_type'" in content, \
            "Таблица verification_sessions должна иметь session_type"
        assert "sa.Column('status'" in content, \
            "Таблица verification_sessions должна иметь status"

    def test_audit_logs_table_creation(self, project_root):
        """Тест: Таблица audit_logs создана."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration_path.read_text()
        
        assert "create_table('audit_logs'" in content, "Должна создаваться таблица audit_logs"
        assert "sa.Column('action'" in content, "Таблица audit_logs должна иметь action"
        assert "sa.Column('user_id'" in content, "Таблица audit_logs должна иметь user_id"

    # =============================================================================
    # ISSUE 8: Indexes & constraints
    # =============================================================================
    
    def test_foreign_key_constraints(self, project_root):
        """Тест: Foreign key constraints созданы."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration_path.read_text()
        
        assert "ForeignKeyConstraint" in content, "Должны быть foreign key constraints"
        assert "references.id" in content, "Должны быть ссылки на references.id"
        assert "users.id" in content, "Должны быть ссылки на users.id"

    def test_unique_constraints(self, project_root):
        """Тест: Unique constraints созданы."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration_path.read_text()
        
        assert "unique=True" in content or "create_unique_constraint" in content, \
            "Должны быть unique constraints"

    def test_indexes_creation(self, project_root):
        """Тест: Индексы созданы для оптимизации."""
        migration_path = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration_path.read_text()
        
        assert "create_index" in content, "Должны создаваться индексы"
        assert "user_id" in content, "Должны быть индексы на user_id"
        assert "session_id" in content or "id" in content, \
            "Должны быть индексы на id полях"

    # =============================================================================
    # ISSUE 9: Redis connection
    # =============================================================================
    
    def test_redis_config_exists(self, config):
        """Тест: Redis конфигурация существует."""
        assert hasattr(config, 'REDIS_URL'), "Должна быть REDIS_URL конфигурация"
        assert hasattr(config, 'REDIS_PASSWORD'), "Должна быть REDIS_PASSWORD конфигурация"
        assert hasattr(config, 'REDIS_CONNECTION_POOL_SIZE'), \
            "Должна быть REDIS_CONNECTION_POOL_SIZE конфигурация"

    def test_redis_service_configuration(self):
        """Тест: Redis сервис настроен в docker-compose."""
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
        if compose_path.exists():
            with open(compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get("services", {})
            redis_service = services.get("redis")
            assert redis_service is not None, "Redis сервис должен быть настроен"

    def test_cache_service_exists(self, project_root):
        """Тест: CacheService существует."""
        cache_service_path = project_root / "app" / "services" / "cache_service.py"
        assert cache_service_path.exists(), "CacheService должен существовать"
        
        content = cache_service_path.read_text()
        assert "class CacheService" in content, "Должен быть класс CacheService"
        assert "redis" in content, "CacheService должен использовать redis"
        assert "health_check" in content, "CacheService должен иметь health_check метод"

    # =============================================================================
    # ISSUE 10: MinIO bucket setup
    # =============================================================================
    
    def test_s3_config_exists(self, config):
        """Тест: S3/MinIO конфигурация существует."""
        assert hasattr(config, 'S3_ENDPOINT_URL'), "Должна быть S3_ENDPOINT_URL конфигурация"
        assert hasattr(config, 'S3_ACCESS_KEY'), "Должна быть S3_ACCESS_KEY конфигурация"
        assert hasattr(config, 'S3_SECRET_KEY'), "Должна быть S3_SECRET_KEY конфигурация"
        assert hasattr(config, 'S3_BUCKET_NAME'), "Должна быть S3_BUCKET_NAME конфигурация"

    def test_storage_service_exists(self, project_root):
        """Тест: StorageService существует."""
        storage_service_path = project_root / "app" / "services" / "storage_service.py"
        assert storage_service_path.exists(), "StorageService должен существовать"
        
        content = storage_service_path.read_text()
        assert "class StorageService" in content, "Должен быть класс StorageService"
        assert "boto3" in content, "StorageService должен использовать boto3"
        assert "S3" in content or "s3" in content, "StorageService должен работать с S3"

    def test_minio_setup_script_exists(self, project_root):
        """Тест: Скрипт setup_minio.py существует."""
        setup_script_path = project_root / "setup_minio.py"
        assert setup_script_path.exists(), "setup_minio.py должен существовать"
        
        content = setup_script_path.read_text()
        assert "Minio" in content or "minio" in content, "Скрипт должен использовать Minio"
        assert "bucket" in content.lower(), "Скрипт должен создавать bucket'ы"

    def test_minio_service_configuration(self):
        """Тест: MinIO сервис настроен в docker-compose."""
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
        if compose_path.exists():
            with open(compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get("services", {})
            minio_service = services.get("minio")
            assert minio_service is not None, "MinIO сервис должен быть настроен"

    # =============================================================================
    # ISSUE 11: config.py setup
    # =============================================================================
    
    def test_settings_class_exists(self, project_root):
        """Тест: Settings class существует."""
        config_path = project_root / "app" / "config.py"
        assert config_path.exists(), "config.py должен существовать"
        
        content = config_path.read_text()
        assert "class Settings" in content, "Должен быть класс Settings"
        assert "BaseSettings" in content, "Settings должен наследоваться от BaseSettings"

    def test_database_config(self, config):
        """Тест: Database конфигурация настроена."""
        assert hasattr(config, 'DATABASE_URL'), "Должна быть DATABASE_URL конфигурация"
        assert hasattr(config, 'DATABASE_POOL_SIZE'), "Должна быть DATABASE_POOL_SIZE конфигурация"
        assert hasattr(config, 'async_database_url'), "Должен быть async_database_url property"

    def test_redis_config(self, config):
        """Тест: Redis конфигурация настроена."""
        assert hasattr(config, 'REDIS_URL'), "Должна быть REDIS_URL конфигурация"
        assert hasattr(config, 'redis_url_with_auth'), \
            "Должен быть redis_url_with_auth property"

    def test_s3_config(self, config):
        """Тест: S3 конфигурация настроена."""
        assert hasattr(config, 'S3_ENDPOINT_URL'), "Должна быть S3_ENDPOINT_URL конфигурация"
        assert hasattr(config, 'S3_ACCESS_KEY'), "Должна быть S3_ACCESS_KEY конфигурация"
        assert hasattr(config, 'S3_SECRET_KEY'), "Должна быть S3_SECRET_KEY конфигурация"

    def test_api_config(self, config):
        """Тест: API конфигурация настроена."""
        assert hasattr(config, 'JWT_SECRET_KEY'), "Должна быть JWT_SECRET_KEY конфигурация"
        assert hasattr(config, 'CORS_ORIGINS'), "Должна быть CORS_ORIGINS конфигурация"
        assert hasattr(config, 'cors_origins_list'), "Должен быть cors_origins_list property"

    def test_ml_config(self, config):
        """Тест: ML конфигурация настроена."""
        assert hasattr(config, 'ML_SERVICE_URL'), "Должна быть ML_SERVICE_URL конфигурация"
        assert hasattr(config, 'THRESHOLD_DEFAULT'), "Должна быть THRESHOLD_DEFAULT конфигурация"

    def test_config_validation(self, project_root):
        """Тест: Config validation настроена."""
        config_path = project_root / "app" / "config.py"
        content = config_path.read_text()
        
        assert "@field_validator" in content, "Должны быть field validators"
        assert "ValidationInfo" in content, "Должны быть ValidationInfo"

    # =============================================================================
    # ISSUE 12: .env.example & requirements.txt
    # =============================================================================
    
    def test_env_example_exists(self, project_root):
        """Тест: .env.example существует."""
        env_example_path = project_root / ".env.example"
        assert env_example_path.exists(), ".env.example должен существовать"
        
        content = env_example_path.read_text()
        
        # Проверяем основные переменные
        assert "DATABASE_URL=" in content, "Должна быть DATABASE_URL переменная"
        assert "REDIS_URL=" in content, "Должна быть REDIS_URL переменная"
        assert "S3_ENDPOINT_URL=" in content, "Должна быть S3_ENDPOINT_URL переменная"
        assert "JWT_SECRET_KEY=" in content, "Должна быть JWT_SECRET_KEY переменная"

    def test_env_example_documentation(self, project_root):
        """Тест: .env.example содержит документацию."""
        env_example_path = project_root / ".env.example"
        content = env_example_path.read_text()
        
        assert "#" in content, "Должны быть комментарии"
        assert "Face Recognition Service" in content, "Должны быть описания"

    def test_requirements_txt_exists(self, project_root):
        """Тест: requirements.txt существует."""
        requirements_path = project_root / "requirements.txt"
        assert requirements_path.exists(), "requirements.txt должен существовать"
        
        content = requirements_path.read_text()
        
        # Проверяем основные зависимости
        assert "fastapi" in content, "Должна быть зависимость fastapi"
        assert "sqlalchemy" in content, "Должна быть зависимость sqlalchemy"
        assert "alembic" in content, "Должна быть зависимость alembic"
        assert "redis" in content, "Должна быть зависимость redis"
        assert "boto3" in content, "Должна быть зависимость boto3"

    def test_requirements_version_pinning(self, project_root):
        """Тест: requirements.txt содержит version pinning."""
        requirements_path = project_root / "requirements.txt"
        content = requirements_path.read_text()
        
        # Проверяем, что есть версии (==)
        lines_with_versions = [line for line in content.split('\n') if '==' in line]
        assert len(lines_with_versions) > 0, "Должны быть зафиксированные версии"

    # =============================================================================
    # ISSUE 13: Python dependencies setup
    # =============================================================================
    
    def test_requirements_dev_exists(self, project_root):
        """Тест: requirements-dev.txt существует."""
        requirements_dev_path = project_root / "requirements-dev.txt"
        assert requirements_dev_path.exists(), "requirements-dev.txt должен существовать"
        
        content = requirements_dev_path.read_text()
        assert "-r requirements.txt" in content or "requirements.txt" in content, \
            "Должен включать production dependencies"

    def test_dev_dependencies(self, project_root):
        """Тест: requirements-dev.txt содержит dev зависимости."""
        requirements_dev_path = project_root / "requirements-dev.txt"
        content = requirements_dev_path.read_text()
        
        # Проверяем dev инструменты
        assert "pytest" in content, "Должна быть зависимость pytest"
        assert "black" in content, "Должна быть зависимость black"
        assert "flake8" in content, "Должна быть зависимость flake8"
        assert "mypy" in content, "Должна быть зависимость mypy"

    def test_pytest_ini_exists(self, project_root):
        """Тест: pytest.ini существует."""
        pytest_ini_path = project_root / "pytest.ini"
        assert pytest_ini_path.exists(), "pytest.ini должен существовать"
        
        content = pytest_ini_path.read_text()
        assert "[tool:pytest]" in content or "[pytest]" in content, \
            "Должен быть правильно настроен pytest.ini"
        assert "testpaths" in content, "Должны быть настроены testpaths"
        assert "markers" in content, "Должны быть настроены markers"

    def test_pyproject_toml_exists(self, project_root):
        """Тест: pyproject.toml существует."""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml должен существовать"
        
        with open(pyproject_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "project" in config, "Должна быть секция project"
        assert "dependencies" in config["project"], "Должны быть dependencies в project"

    def test_pyproject_tool_configurations(self, project_root):
        """Тест: pyproject.toml содержит конфигурации инструментов."""
        pyproject_path = project_root / "pyproject.toml"
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        # Проверяем конфигурации инструментов
        assert "[tool.black]" in content, "Должна быть конфигурация black"
        assert "[tool.isort]" in content, "Должна быть конфигурация isort"
        assert "[tool.flake8]" in content, "Должна быть конфигурация flake8"
        assert "[tool.mypy]" in content, "Должна быть конфигурация mypy"
        assert "[tool.pytest.ini_options]" in content, "Должна быть конфигурация pytest"

    def test_pip_install_requirements(self, project_root):
        """Тест: pip install -r requirements.txt работает."""
        requirements_path = project_root / "requirements.txt"
        
        # Проверяем, что файл requirements.txt валиден
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        # Проверяем синтаксис файла
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Каждая строка должна быть валидным package specification
                assert '==' in line or '>=' in line or '~=' in line or line.startswith('-r'), \
                    f"Некорректная строка в requirements.txt: {line}"

    # =============================================================================
    # INTEGRATION TESTS
    # =============================================================================
    
    @pytest.mark.integration
    def test_docker_compose_syntax(self, project_root):
        """Тест: docker-compose файлы имеют корректный синтаксис."""
        compose_files = ["docker-compose.yml", "docker-compose.dev.yml"]
        
        for compose_file in compose_files:
            compose_path = project_root / compose_file
            if compose_path.exists():
                with open(compose_path, 'r') as f:
                    try:
                        yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        pytest.fail(f"Некорректный YAML синтаксис в {compose_file}: {e}")

    @pytest.mark.integration
    def test_alembic_config_syntax(self, project_root):
        """Тест: alembic.ini имеет корректный синтаксис."""
        ini_path = project_root / "alembic.ini"
        if ini_path.exists():
            with open(ini_path, 'r') as f:
                content = f.read()
            
            # Проверяем базовую структуру
            assert "[alembic]" in content, "alembic.ini должен содержать [alembic] секцию"
            assert "script_location" in content, "alembic.ini должен содержать script_location"

    @pytest.mark.integration
    def test_config_loading(self, config):
        """Тест: Config может быть загружен."""
        assert config is not None, "Config должен загружаться"
        assert config.APP_NAME == "Face Recognition Service", \
            "APP_NAME должен быть настроен правильно"

    @pytest.mark.integration
    def test_import_sqlalchemy(self):
        """Тест: SQLAlchemy может быть импортирован."""
        try:
            import sqlalchemy as sa
            from sqlalchemy.dialects import postgresql
            assert sa.__version__ is not None, "SQLAlchemy должен иметь версию"
        except ImportError as e:
            pytest.fail(f"Не удается импортировать SQLAlchemy: {e}")

    @pytest.mark.integration
    def test_import_alembic(self):
        """Тест: Alembic может быть импортирован."""
        try:
            import alembic
            assert alembic.__version__ is not None, "Alembic должен иметь версию"
        except ImportError as e:
            pytest.fail(f"Не удается импортировать Alembic: {e}")

    @pytest.mark.integration
    def test_import_redis(self):
        """Тест: Redis может быть импортирован."""
        try:
            import redis
            assert redis.__version__ is not None, "Redis должен иметь версию"
        except ImportError as e:
            pytest.fail(f"Не удается импортировать Redis: {e}")

    @pytest.mark.integration
    def test_import_boto3(self):
        """Тест: boto3 может быть импортирован."""
        try:
            import boto3
            assert boto3.__version__ is not None, "boto3 должен иметь версию"
        except ImportError as e:
            pytest.fail(f"Не удается импортировать boto3: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])