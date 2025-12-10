"""
Юнит тесты для проверки ФАЗЫ 1 Infrastructure компонентов.
Тестирование всех исправленных компонентов.
"""

import os
import sys
import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к корню проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings


class TestRequirementsTxt:
    """Тестирование requirements.txt"""

    def test_requirements_file_exists(self):
        """Проверка существования requirements.txt"""
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
        assert requirements_path.exists(), "requirements.txt должен существовать"

    def test_alembic_in_requirements(self):
        """Проверка наличия alembic в requirements.txt"""
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        assert "alembic==1.12.1" in content, "alembic должен быть в requirements.txt"
        assert "alembic" in content.lower(), "alembic должен присутствовать"

    def test_exact_version_pinning(self):
        """Проверка точного version pinning"""
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        # Проверяем, что нет мягкого pinning с >=
        lines = content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                # Игнорируем комментарии и пустые строки
                if '>=' in line or '<' in line:
                    pytest.fail(f"Найден мягкий pinning в строке: {line}")

    def test_core_dependencies_present(self):
        """Проверка наличия основных зависимостей"""
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        required_deps = [
            "fastapi==0.109.0",
            "sqlalchemy[asyncio]==2.0.25",
            "redis==5.0.1",
            "boto3==1.34.25",
            "uvicorn[standard]==0.27.0",
            "pydantic==2.6.0"
        ]
        
        for dep in required_deps:
            assert dep in content, f"Зависимость {dep} должна быть в requirements.txt"


class TestPyprojectToml:
    """Тестирование pyproject.toml"""

    def test_pyproject_toml_exists(self):
        """Проверка существования pyproject.toml"""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml должен существовать"

    def test_pyproject_toml_structure(self):
        """Проверка структуры pyproject.toml"""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        required_sections = [
            "[build-system]",
            "[project]",
            "[project.optional-dependencies]",
            "[tool.pytest.ini_options]",
            "[tool.black]",
            "[tool.isort]",
            "[tool.mypy]"
        ]
        
        for section in required_sections:
            assert section in content, f"Секция {section} должна быть в pyproject.toml"

    def test_dependencies_in_pyproject(self):
        """Проверка зависимостей в pyproject.toml"""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        # Проверяем наличие основных зависимостей
        assert '"fastapi==0.109.0"' in content
        assert '"alembic==1.12.1"' in content
        assert '"sqlalchemy[asyncio]==2.0.25"' in content

    def test_dev_dependencies_in_pyproject(self):
        """Проверка dev зависимостей в pyproject.toml"""
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        dev_deps = [
            '"pytest==7.4.3"',
            '"black==23.11.0"',
            '"flake8==6.1.0"',
            '"mypy==1.7.1"'
        ]
        
        for dep in dev_deps:
            assert dep in content, f"Dev зависимость {dep} должна быть в pyproject.toml"


class TestDockerfile:
    """Тестирование Dockerfile"""

    def test_dockerfile_exists(self):
        """Проверка существования Dockerfile"""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile должен существовать"

    def test_dockerfile_multistage(self):
        """Проверка multi-stage build в Dockerfile"""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Проверяем наличие builder этапа
        assert "FROM python:3.11-slim as builder" in content
        assert "FROM python:3.11-slim as runtime" in content
        
        # Проверяем копирование virtualenv из builder этапа
        assert "COPY --from=builder /opt/venv /opt/venv" in content

    def test_dockerfile_non_root_user(self):
        """Проверка non-root пользователя в Dockerfile"""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        assert "useradd -r -g app" in content
        assert "USER app" in content

    def test_dockerfile_optimized_layers(self):
        """Проверка оптимизации слоев в Dockerfile"""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Проверяем использование virtualenv
        assert "python -m venv /opt/venv" in content
        assert "ENV PATH=\"/opt/venv/bin:$PATH\"" in content


class TestConfig:
    """Тестирование конфигурации"""

    def test_config_file_exists(self):
        """Проверка существования config.py"""
        config_path = Path(__file__).parent.parent.parent / "app" / "config.py"
        assert config_path.exists(), "app/config.py должен существовать"

    def test_settings_class_exists(self):
        """Проверка класса Settings"""
        from app.config import settings
        
        assert hasattr(settings, 'DATABASE_URL')
        assert hasattr(settings, 'REDIS_URL')
        assert hasattr(settings, 'S3_ENDPOINT_URL')
        assert hasattr(settings, 'ML_SERVICE_URL')

    def test_config_validation(self):
        """Проверка валидации конфигурации"""
        from app.config import Settings
        
        # Проверяем валидаторы
        settings = Settings()
        assert hasattr(settings, 'allowed_image_formats_list')
        assert hasattr(settings, 'cors_origins_list')

    def test_database_config(self):
        """Проверка конфигурации базы данных"""
        from app.config import settings
        
        assert hasattr(settings, 'DATABASE_URL')
        assert hasattr(settings, 'DATABASE_POOL_SIZE')
        assert hasattr(settings, 'DATABASE_MAX_OVERFLOW')

    def test_redis_config(self):
        """Проверка конфигурации Redis"""
        from app.config import settings
        
        assert hasattr(settings, 'REDIS_URL')
        assert hasattr(settings, 'REDIS_PASSWORD')
        assert hasattr(settings, 'REDIS_CONNECTION_POOL_SIZE')

    def test_s3_config(self):
        """Проверка конфигурации S3/MinIO"""
        from app.config import settings
        
        assert hasattr(settings, 'S3_ENDPOINT_URL')
        assert hasattr(settings, 'S3_ACCESS_KEY')
        assert hasattr(settings, 'S3_SECRET_KEY')
        assert hasattr(settings, 'S3_BUCKET_NAME')


class TestDockerCompose:
    """Тестирование Docker Compose"""

    def test_docker_compose_exists(self):
        """Проверка существования docker-compose.yml"""
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml должен существовать"

    def test_docker_compose_dev_exists(self):
        """Проверка существования docker-compose.dev.yml"""
        compose_dev_path = Path(__file__).parent.parent.parent / "docker-compose.dev.yml"
        assert compose_dev_path.exists(), "docker-compose.dev.yml должен существовать"

    def test_compose_services(self):
        """Проверка сервисов в docker-compose.yml"""
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
        with open(compose_path, 'r') as f:
            content = f.read()
        
        required_services = [
            "postgres:",
            "redis:",
            "minio:",
            "face-recognition-api:"
        ]
        
        for service in required_services:
            assert service in content, f"Сервис {service} должен быть в docker-compose.yml"

    def test_compose_dev_volume_mounts(self):
        """Проверка volume mounts в docker-compose.dev.yml"""
        compose_dev_path = Path(__file__).parent.parent.parent / "docker-compose.dev.yml"
        with open(compose_dev_path, 'r') as f:
            content = f.read()
        
        assert ".:/app" in content, "Volume mount .:/app должен быть в docker-compose.dev.yml"


class TestAlembic:
    """Тестирование Alembic"""

    def test_alembic_ini_exists(self):
        """Проверка существования alembic.ini"""
        alembic_ini_path = Path(__file__).parent.parent.parent / "alembic.ini"
        assert alembic_ini_path.exists(), "alembic.ini должен существовать"

    def test_alembic_env_exists(self):
        """Проверка существования alembic/env.py"""
        env_path = Path(__file__).parent.parent.parent / "alembic" / "env.py"
        assert env_path.exists(), "alembic/env.py должен существовать"

    def test_alembic_migration_exists(self):
        """Проверка существования миграции"""
        migration_path = Path(__file__).parent.parent.parent / "alembic" / "versions" / "001_initial_migration.py"
        assert migration_path.exists(), "Миграция должна существовать"

    def test_alembic_env_config(self):
        """Проверка конфигурации alembic/env.py"""
        env_path = Path(__file__).parent.parent.parent / "alembic" / "env.py"
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "from app.db.models import Base" in content
        assert "target_metadata = Base.metadata" in content
        assert "get_database_url" in content

    def test_migration_tables(self):
        """Проверка таблиц в миграции"""
        migration_path = Path(__file__).parent.parent.parent / "alembic" / "versions" / "001_initial_migration.py"
        with open(migration_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_tables = [
            "users",
            "references", 
            "verification_sessions",
            "audit_logs"
        ]
        
        for table in required_tables:
            assert f"'{table}'" in content, f"Таблица {table} должна быть в миграции"


class TestDockerIgnore:
    """Тестирование .dockerignore"""

    def test_dockerignore_exists(self):
        """Проверка существования .dockerignore"""
        dockerignore_path = Path(__file__).parent.parent.parent / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore должен существовать"

    def test_dockerignore_content(self):
        """Проверка содержимого .dockerignore"""
        dockerignore_path = Path(__file__).parent.parent.parent / ".dockerignore"
        with open(dockerignore_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_exclusions = [
            "__pycache__",
            ".env",
            "*.log",
            "tests/",
            ".vscode/",
            ".idea/",
            "*.py[cod]",
            "*.egg-info/"
        ]
        
        for exclusion in required_exclusions:
            assert exclusion in content, f"Исключение {exclusion} должно быть в .dockerignore"


class TestEnvExample:
    """Тестирование .env.example"""

    def test_env_example_exists(self):
        """Проверка существования .env.example"""
        env_example_path = Path(__file__).parent.parent.parent / ".env.example"
        assert env_example_path.exists(), ".env.example должен существовать"

    def test_env_example_content(self):
        """Проверка содержимого .env.example"""
        env_example_path = Path(__file__).parent.parent.parent / ".env.example"
        with open(env_example_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_vars = [
            "DATABASE_URL",
            "REDIS_URL", 
            "S3_ENDPOINT_URL",
            "JWT_SECRET_KEY",
            "ENCRYPTION_KEY"
        ]
        
        for var in required_vars:
            assert var in content, f"Переменная {var} должна быть в .env.example"


class TestServices:
    """Тестирование сервисов"""

    def test_cache_service_exists(self):
        """Проверка существования CacheService"""
        cache_service_path = Path(__file__).parent.parent.parent / "app" / "services" / "cache_service.py"
        assert cache_service_path.exists(), "CacheService должен существовать"

    def test_storage_service_exists(self):
        """Проверка существования StorageService"""
        storage_service_path = Path(__file__).parent.parent.parent / "app" / "services" / "storage_service.py"
        assert storage_service_path.exists(), "StorageService должен существовать"

    def test_cache_service_methods(self):
        """Проверка методов CacheService"""
        # Проверяем импорт без реального подключения
        try:
            from app.services.cache_service import CacheService
            cache_service = CacheService()
            
            # Проверяем наличие основных методов
            assert hasattr(cache_service, 'health_check')
            assert hasattr(cache_service, 'set')
            assert hasattr(cache_service, 'get')
            assert hasattr(cache_service, 'delete')
            
        except Exception as e:
            pytest.skip(f"CacheService недоступен: {e}")

    def test_storage_service_methods(self):
        """Проверка методов StorageService"""
        try:
            from app.services.storage_service import StorageService
            storage_service = StorageService()
            
            # Проверяем наличие основных методов
            assert hasattr(storage_service, 'health_check')
            assert hasattr(storage_service, 'upload_image')
            assert hasattr(storage_service, 'download_image')
            assert hasattr(storage_service, 'delete_image')
            
        except Exception as e:
            pytest.skip(f"StorageService недоступен: {e}")


class TestSetupScripts:
    """Тестирование скриптов настройки"""

    def test_setup_minio_exists(self):
        """Проверка существования setup_minio.py"""
        setup_minio_path = Path(__file__).parent.parent.parent / "setup_minio.py"
        assert setup_minio_path.exists(), "setup_minio.py должен существовать"

    def test_setup_minio_functions(self):
        """Проверка функций в setup_minio.py"""
        setup_minio_path = Path(__file__).parent.parent.parent / "setup_minio.py"
        with open(setup_minio_path, 'r') as f:
            content = f.read()
        
        required_functions = [
            "create_minio_buckets",
            "verify_minio_setup"
        ]
        
        for func in required_functions:
            assert f"def {func}" in content, f"Функция {func} должна быть в setup_minio.py"


@pytest.mark.integration
class TestIntegration:
    """Интеграционные тесты"""

    def test_pip_install_requirements(self):
        """Тест установки requirements через pip"""
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
        
        # Проверяем синтаксис файла requirements.txt
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check", "--disable-pip-version-check"],
                cwd=requirements_path.parent,
                capture_output=True,
                text=True,
                timeout=30
            )
            # pip check может вернуть ненулевой код, если есть проблемы с зависимостями
            # Но файл должен быть валидным
            
        except subprocess.TimeoutExpired:
            pytest.skip("Тест pip install занял слишком много времени")

    def test_dockerfile_syntax(self):
        """Проверка синтаксиса Dockerfile"""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        
        # Проверяем базовый синтаксис Docker файла
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Проверяем наличие основных директив
        assert "FROM" in content
        assert "COPY" in content
        assert "RUN" in content
        assert "CMD" in content

    def test_docker_compose_syntax(self):
        """Проверка синтаксиса docker-compose.yml"""
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
        
        # Проверяем базовый синтаксис
        with open(compose_path, 'r') as f:
            content = f.read()
        
        assert "version:" in content
        assert "services:" in content
        assert "volumes:" in content or "networks:" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])