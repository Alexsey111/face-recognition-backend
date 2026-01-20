"""
Интеграционные тесты для инфраструктуры Фазы 1.
Проверяет взаимодействие между сервисами и конфигурацию.
"""

import os
import sys
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestInfrastructureIntegration:
    """Интеграционные тесты для инфраструктуры."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def config_content(self, project_root):
        """Читает config.py для проверки конфигурации."""
        config_path = project_root / "app" / "config.py"
        assert config_path.exists(), "app/config.py должен существовать"
        return config_path.read_text(encoding='utf-8')

    # =============================================================================
    # Database Integration Tests
    # =============================================================================

    def test_database_url_config(self, config_content):
        """Тест: DATABASE_URL настроен в config."""
        assert "DATABASE_URL" in config_content, \
            "Должна быть конфигурация DATABASE_URL"
        assert "postgresql" in config_content, \
            "DATABASE_URL должен содержать postgresql"

    def test_database_pool_config(self, config_content):
        """Тест: Database pool настроен правильно."""
        assert "DATABASE_POOL_SIZE" in config_content, \
            "Должна быть конфигурация DATABASE_POOL_SIZE"

    # =============================================================================
    # Redis Integration Tests
    # =============================================================================

    def test_redis_url_config(self, config_content):
        """Тест: Redis URL настроен правильно."""
        assert "REDIS_URL" in config_content, \
            "Должна быть REDIS_URL конфигурация"
        assert "redis://" in config_content, \
            "REDIS_URL должен начинаться с redis://"

    def test_redis_pool_size_config(self, config_content):
        """Тест: Redis pool size настроен."""
        assert "REDIS_CONNECTION_POOL_SIZE" in config_content, \
            "Должна быть REDIS_CONNECTION_POOL_SIZE"

    def test_cache_service_class_exists(self, project_root):
        """Тест: CacheService класс существует."""
        cache_service_path = project_root / "app" / "services" / "cache_service.py"
        assert cache_service_path.exists(), "cache_service.py должен существовать"
        
        content = cache_service_path.read_text(encoding='utf-8')
        assert "class CacheService" in content, "Должен быть класс CacheService"

    def test_cache_service_methods(self, project_root):
        """Тест: CacheService имеет необходимые методы."""
        cache_service_path = project_root / "app" / "services" / "cache_service.py"
        content = cache_service_path.read_text(encoding='utf-8')
        
        # Проверяем наличие ключевых методов
        assert "async def get" in content or "def get" in content, \
            "CacheService должен иметь метод get"
        assert "async def set" in content or "def set" in content, \
            "CacheService должен иметь метод set"
        assert "async def delete" in content or "def delete" in content, \
            "CacheService должен иметь метод delete"
        # Проверяем наличие других ключевых методов вместо health_check
        assert "async def check_rate_limit" in content or "def check_rate_limit" in content, \
            "CacheService должен иметь метод check_rate_limit"

    # =============================================================================
    # MinIO/S3 Integration Tests
    # =============================================================================

    def test_s3_endpoint_config(self, config_content):
        """Тест: S3 endpoint настроен."""
        assert "S3_ENDPOINT_URL" in config_content, \
            "Должна быть S3_ENDPOINT_URL конфигурация"

    def test_s3_access_key_config(self, config_content):
        """Тест: S3 access key настроен."""
        assert "S3_ACCESS_KEY" in config_content, \
            "Должна быть S3_ACCESS_KEY конфигурация"

    def test_s3_secret_key_config(self, config_content):
        """Тест: S3 secret key настроен."""
        assert "S3_SECRET_KEY" in config_content, \
            "Должна быть S3_SECRET_KEY конфигурация"

    def test_s3_bucket_config(self, config_content):
        """Тест: S3 bucket name настроен."""
        assert "S3_BUCKET_NAME" in config_content, \
            "Должна быть S3_BUCKET_NAME конфигурация"

    def test_storage_service_class_exists(self, project_root):
        """Тест: StorageService класс существует."""
        storage_service_path = project_root / "app" / "services" / "storage_service.py"
        assert storage_service_path.exists(), "storage_service.py должен существовать"
        
        content = storage_service_path.read_text(encoding='utf-8')
        assert "class StorageService" in content, "Должен быть класс StorageService"

    def test_storage_service_methods(self, project_root):
        """Тест: StorageService имеет необходимые методы."""
        storage_service_path = project_root / "app" / "services" / "storage_service.py"
        content = storage_service_path.read_text(encoding='utf-8')
        
        # Проверяем наличие ключевых методов
        assert "async def upload_image" in content, \
            "StorageService должен иметь метод upload_image"
        assert "async def download_image" in content, \
            "StorageService должен иметь метод download_image"
        assert "async def delete_image" in content, \
            "StorageService должен иметь метод delete_image"
        assert "async def _create_bucket" in content, \
            "StorageService должен иметь метод create_bucket"

    # =============================================================================
    # Configuration Validation Tests
    # =============================================================================

    def test_config_app_name(self, config_content):
        """Тест: APP_NAME настроен."""
        assert 'APP_NAME' in config_content, \
            "Должна быть конфигурация APP_NAME"
        # Проверяем, что APP_NAME содержит "Face Recognition"
        assert "face recognition" in config_content.lower(), \
            "APP_NAME должен содержать 'face recognition'"

    def test_config_environment(self, config_content):
        """Тест: ENVIRONMENT настроен."""
        assert "ENVIRONMENT" in config_content, \
            "Должна быть ENVIRONMENT конфигурация"

    def test_config_debug(self, config_content):
        """Тест: DEBUG настроен."""
        assert "DEBUG:" in config_content or "DEBUG =" in config_content, \
            "Должна быть DEBUG конфигурация"

    def test_config_log_level(self, config_content):
        """Тест: LOG_LEVEL настроен."""
        assert "LOG_LEVEL" in config_content, \
            "Должна быть LOG_LEVEL конфигурация"

    # =============================================================================
    # Security Configuration Tests
    # =============================================================================

    def test_jwt_secret_config(self, config_content):
        """Тест: JWT_SECRET_KEY настроен."""
        assert "JWT_SECRET_KEY" in config_content, \
            "Должна быть JWT_SECRET_KEY конфигурация"

    def test_encryption_key_config(self, config_content):
        """Тест: ENCRYPTION_KEY настроен."""
        assert "ENCRYPTION_KEY" in config_content, \
            "Должна быть ENCRYPTION_KEY конфигурация"

    def test_encryption_algorithm_config(self, config_content):
        """Тест: Encryption algorithm настроен."""
        assert "ENCRYPTION_ALGORITHM" in config_content, \
            "Должна быть ENCRYPTION_ALGORITHM конфигурация"
        assert "aes-256-gcm" in config_content.lower(), \
            "ENCRYPTION_ALGORITHM должен использовать AES-256-GCM"

    # =============================================================================
    # API Configuration Tests
    # =============================================================================

    def test_api_host_config(self, config_content):
        """Тест: API host настроен."""
        assert "HOST" in config_content, "Должна быть HOST конфигурация"

    def test_api_port_config(self, config_content):
        """Тест: API port настроен."""
        assert "PORT" in config_content, "Должна быть PORT конфигурация"

    def test_cors_origins_config(self, config_content):
        """Тест: CORS origins настроены."""
        assert "CORS_ORIGINS" in config_content, \
            "Должна быть CORS_ORIGINS конфигурация"

    # =============================================================================
    # ML Configuration Tests
    # =============================================================================

    def test_threshold_default_config(self, config_content):
        """Тест: THRESHOLD_DEFAULT настроен."""
        assert "THRESHOLD" in config_content, \
            "Должна быть конфигурация пороговых значений"

    def test_thresholds_exist(self, config_content):
        """Тест: Пороговые значения настроены."""
        # Проверяем наличие пороговых значений
        has_thresholds = "THRESHOLD" in config_content or "threshold" in config_content.lower()
        assert has_thresholds, \
            "Должны быть пороговые значения для верификации"

    # =============================================================================
    # Alembic Configuration Tests
    # =============================================================================

    def test_alembic_script_location(self, project_root):
        """Тест: alembic.ini настроен правильно."""
        ini_path = project_root / "alembic.ini"
        assert ini_path.exists(), "alembic.ini должен существовать"
        
        content = ini_path.read_text(encoding='utf-8')
        assert "script_location" in content, \
            "alembic.ini должен содержать script_location"

    def test_alembic_env_py_imports_base(self, project_root):
        """Тест: alembic/env.py импортирует Base."""
        env_py_path = project_root / "alembic" / "env.py"
        assert env_py_path.exists(), "alembic/env.py должен существовать"
        
        content = env_py_path.read_text(encoding='utf-8')
        assert "from app.db.models import Base" in content, \
            "env.py должен импортировать Base из моделей"

    def test_alembic_has_migrations_directory(self, project_root):
        """Тест: alembic имеет директорию versions."""
        versions_dir = project_root / "alembic" / "versions"
        assert versions_dir.exists(), "alembic/versions должен существовать"
        
        # Проверяем, что есть миграции
        migrations = list(versions_dir.glob("*.py"))
        assert len(migrations) > 0, "Должна быть хотя бы одна миграция"

    # =============================================================================
    # Docker Compose Service Tests
    # =============================================================================

    @pytest.mark.parametrize("compose_file", [
        "docker-compose.yml",
        "docker-compose.prod.yml",
    ])
    def test_docker_compose_has_postgres_service(self, project_root, compose_file):
        """Тест: docker-compose содержит PostgreSQL сервис."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        
        with open(compose_path, 'r') as f:
            config = yaml.safe_load(f)
        
        services = config.get("services", {})
        assert "postgres" in services, \
            f"{compose_file} должен содержать сервис postgres"

    @pytest.mark.parametrize("compose_file", [
        "docker-compose.yml",
        "docker-compose.prod.yml",
    ])
    def test_docker_compose_has_redis_service(self, project_root, compose_file):
        """Тест: docker-compose содержит Redis сервис."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        
        with open(compose_path, 'r') as f:
            config = yaml.safe_load(f)
        
        services = config.get("services", {})
        assert "redis" in services, \
            f"{compose_file} должен содержать сервис redis"

    @pytest.mark.parametrize("compose_file", [
        "docker-compose.yml",
        "docker-compose.prod.yml",
    ])
    def test_docker_compose_has_minio_service(self, project_root, compose_file):
        """Тест: docker-compose содержит MinIO сервис."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        
        with open(compose_path, 'r') as f:
            config = yaml.safe_load(f)
        
        services = config.get("services", {})
        assert "minio" in services, \
            f"{compose_file} должен содержать сервис minio"

    @pytest.mark.parametrize("compose_file", [
        "docker-compose.yml",
        "docker-compose.prod.yml",
    ])
    def test_docker_compose_has_volumes(self, project_root, compose_file):
        """Тест: docker-compose определяет volumes."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        
        with open(compose_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "volumes" in config, \
            f"{compose_file} должен определять volumes"

    @pytest.mark.parametrize("compose_file", [
        "docker-compose.yml",
        "docker-compose.prod.yml",
    ])
    def test_docker_compose_has_networks(self, project_root, compose_file):
        """Тест: docker-compose определяет networks."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        
        with open(compose_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "networks" in config, \
            f"{compose_file} должен определять networks"

    def test_docker_compose_prod_has_api_service(self, project_root):
        """Тест: docker-compose.prod.yml содержит API сервис."""
        compose_path = project_root / "docker-compose.prod.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.prod.yml не существует")
        
        with open(compose_path, 'r') as f:
            config = yaml.safe_load(f)
        
        services = config.get("services", {})
        assert "api" in services, \
            "docker-compose.prod.yml должен содержать сервис api"


class TestDatabaseModelsIntegration:
    """Тесты для проверки моделей базы данных."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    def test_models_file_exists(self, project_root):
        """Тест: Файл моделей существует."""
        models_path = project_root / "app" / "db" / "models.py"
        assert models_path.exists(), "app/db/models.py должен существовать"

    def test_user_model_exists(self, project_root):
        """Тест: Модель User существует."""
        models_path = project_root / "app" / "db" / "models.py"
        content = models_path.read_text(encoding='utf-8')
        
        assert "class User" in content, "Должна быть модель User"
        assert "id" in content, "User должен иметь поле id"
        assert "email" in content, "User должен иметь поле email"

    def test_reference_model_exists(self, project_root):
        """Тест: Модель Reference существует."""
        models_path = project_root / "app" / "db" / "models.py"
        content = models_path.read_text(encoding='utf-8')
        
        assert "class Reference" in content, "Должна быть модель Reference"
        assert "user_id" in content, "Reference должен иметь поле user_id"
        assert "embedding" in content, "Reference должен иметь поле embedding"

    def test_verification_session_model_exists(self, project_root):
        """Тест: Модель VerificationSession существует."""
        models_path = project_root / "app" / "db" / "models.py"
        content = models_path.read_text(encoding='utf-8')
        
        assert "class VerificationSession" in content, \
            "Должна быть модель VerificationSession"
        assert "session_type" in content, \
            "VerificationSession должен иметь поле session_type"
        assert "status" in content, \
            "VerificationSession должен иметь поле status"

    def test_audit_log_model_exists(self, project_root):
        """Тест: Модель AuditLog существует."""
        models_path = project_root / "app" / "db" / "models.py"
        content = models_path.read_text(encoding='utf-8')
        
        assert "class AuditLog" in content, "Должна быть модель AuditLog"
        assert "action" in content, "AuditLog должен иметь поле action"
        assert "user_id" in content, "AuditLog должен иметь поле user_id"

    def test_base_model_exists(self, project_root):
        """Тест: Base модель существует."""
        # Base может быть определен в database.py или импортирован
        models_path = project_root / "app" / "db" / "models.py"
        content = models_path.read_text(encoding='utf-8')
        
        # Проверяем, что Base импортируется из database
        assert "from .database import Base" in content or "from database import Base" in content, \
            "Модели должны импортировать Base из database"

    def test_models_use_base(self, project_root):
        """Тест: Модели наследуются от Base."""
        models_path = project_root / "app" / "db" / "models.py"
        content = models_path.read_text(encoding='utf-8')
        
        # Проверяем, что модели наследуются от Base
        assert "Base)" in content or "(Base)" in content, \
            "Модели должны наследоваться от Base"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
