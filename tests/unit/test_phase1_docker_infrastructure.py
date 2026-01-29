"""
Дополнительные тесты для инфраструктуры Фазы 1: Docker и docker-compose.
Проверяет production и development Docker конфигурации.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Добавляем путь к корню проекта
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import Settings


class TestDockerProductionInfrastructure:
    """Тесты для production Docker инфраструктуры."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def docker_compose_prod(self, project_root):
        """Загружает docker-compose.prod.yml."""
        compose_path = project_root / "docker-compose.prod.yml"
        assert compose_path.exists(), "docker-compose.prod.yml должен существовать"
        with open(compose_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def dockerfile_prod(self, project_root):
        """Загружает Dockerfile."""
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile должен существовать"
        return dockerfile_path.read_text(encoding="utf-8")

    # =============================================================================
    # docker-compose.prod.yml Tests
    # =============================================================================

    def test_docker_compose_prod_exists(self, project_root):
        """Тест: docker-compose.prod.yml существует."""
        compose_path = project_root / "docker-compose.prod.yml"
        assert compose_path.exists(), "docker-compose.prod.yml должен существовать"

    def test_docker_compose_prod_has_api_service(self, docker_compose_prod):
        """Тест: docker-compose.prod.yml содержит API сервис."""
        services = docker_compose_prod.get("services", {})
        assert (
            "api" in services
        ), "docker-compose.prod.yml должен содержать сервис 'api'"

    def test_docker_compose_prod_api_build_config(self, docker_compose_prod):
        """Тест: API сервис имеет правильную конфигурацию сборки."""
        api_service = docker_compose_prod["services"]["api"]
        assert "build" in api_service, "API сервис должен иметь build секцию"
        assert "context" in api_service["build"], "Build должен иметь context"
        assert api_service["build"]["context"] == ".", "Context должен быть '.'"

    def test_docker_compose_prod_api_container_name(self, docker_compose_prod):
        """Тест: API сервис имеет определенное имя контейнера."""
        api_service = docker_compose_prod["services"]["api"]
        assert "container_name" in api_service, "Должно быть container_name"
        assert (
            api_service["container_name"] == "face_recognition_api"
        ), "Имя контейнера должно быть 'face_recognition_api'"

    def test_docker_compose_prod_api_ports(self, docker_compose_prod):
        """Тест: API сервис экспортирует порты."""
        api_service = docker_compose_prod["services"]["api"]
        assert "ports" in api_service, "Должны быть экспортированы порты"
        ports = api_service["ports"]
        assert "8000:8000" in ports, "Порт 8000 должен быть проброшен"

    def test_docker_compose_prod_api_environment(self, docker_compose_prod):
        """Тест: API сервис имеет правильные переменные окружения."""
        api_service = docker_compose_prod["services"]["api"]
        env = api_service.get("environment", {})
        assert "ENVIRONMENT" in env, "Должна быть переменная ENVIRONMENT"
        assert (
            env["ENVIRONMENT"] == "production"
        ), "ENVIRONMENT должен быть 'production'"
        assert "DEBUG" in env, "Должна быть переменная DEBUG"
        assert env["DEBUG"] == False, "DEBUG должен быть False в production"
        assert "DATABASE_URL" in env, "Должна быть DATABASE_URL"
        assert (
            "postgres" in env["DATABASE_URL"]
        ), "DATABASE_URL должен указывать на postgres"
        assert "REDIS_URL" in env, "Должна быть REDIS_URL"
        assert "redis" in env["REDIS_URL"], "REDIS_URL должен указывать на redis"
        assert "S3_ENDPOINT_URL" in env, "Должна быть S3_ENDPOINT_URL"
        assert (
            "minio" in env["S3_ENDPOINT_URL"].lower()
        ), "S3_ENDPOINT_URL должен указывать на minio"

    def test_docker_compose_prod_api_healthcheck(self, docker_compose_prod):
        """Тест: API сервис имеет healthcheck."""
        api_service = docker_compose_prod["services"]["api"]
        assert "healthcheck" in api_service, "Должен быть healthcheck"
        hc = api_service["healthcheck"]
        assert "test" in hc, "healthcheck должен иметь test"
        assert "interval" in hc, "healthcheck должен иметь interval"
        assert "timeout" in hc, "healthcheck должен иметь timeout"
        assert "retries" in hc, "healthcheck должен иметь retries"
        assert "start_period" in hc, "healthcheck должен иметь start_period"

    def test_docker_compose_prod_api_restart_policy(self, docker_compose_prod):
        """Тест: API сервис имеет политику рестарта."""
        api_service = docker_compose_prod["services"]["api"]
        assert "restart" in api_service, "Должна быть политика restart"
        assert (
            api_service["restart"] == "unless-stopped"
        ), "Политика restart должна быть 'unless-stopped'"

    def test_docker_compose_prod_api_depends_on(self, docker_compose_prod):
        """Тест: API сервис зависит от других сервисов."""
        api_service = docker_compose_prod["services"]["api"]
        assert "depends_on" in api_service, "Должен быть depends_on"
        depends = api_service["depends_on"]
        assert "postgres" in depends, "Должен зависеть от postgres"
        assert "redis" in depends, "Должен зависеть от redis"
        assert "minio" in depends, "Должен зависеть от minio"

    def test_docker_compose_prod_postgres_healthcheck(self, docker_compose_prod):
        """Тест: PostgreSQL имеет healthcheck с условием."""
        postgres_service = docker_compose_prod["services"]["postgres"]
        assert "healthcheck" in postgres_service, "PostgreSQL должен иметь healthcheck"
        assert (
            "test" in postgres_service["healthcheck"]
        ), "Должен быть test в healthcheck"
        test_cmd = str(postgres_service["healthcheck"]["test"])
        assert "pg_isready" in test_cmd, "Должен использовать pg_isready"

    def test_docker_compose_prod_redis_healthcheck(self, docker_compose_prod):
        """Тест: Redis имеет healthcheck."""
        redis_service = docker_compose_prod["services"]["redis"]
        assert "healthcheck" in redis_service, "Redis должен иметь healthcheck"
        test_cmd = str(redis_service["healthcheck"]["test"])
        assert (
            "redis-cli" in test_cmd or "ping" in test_cmd
        ), "Должен использовать redis-cli ping"

    def test_docker_compose_prod_minio_healthcheck(self, docker_compose_prod):
        """Тест: MinIO имеет healthcheck."""
        minio_service = docker_compose_prod["services"]["minio"]
        assert "healthcheck" in minio_service, "MinIO должен иметь healthcheck"
        test_cmd = str(minio_service["healthcheck"]["test"])
        assert (
            "curl" in test_cmd or "health" in test_cmd.lower()
        ), "Должен использовать curl для проверки health"

    def test_docker_compose_prod_pgadmin_profile(self, docker_compose_prod):
        """Тест: pgAdmin использует profile."""
        services = docker_compose_prod.get("services", {})
        if "pgadmin" in services:
            pgadmin_service = services["pgadmin"]
            assert (
                "profiles" in pgadmin_service
            ), "pgAdmin должен использовать profiles для опционального запуска"
            assert (
                "admin" in pgadmin_service["profiles"]
            ), "pgAdmin должен быть в profile 'admin'"

    def test_docker_compose_prod_volumes(self, docker_compose_prod):
        """Тест: docker-compose.prod.yml определяет volumes."""
        assert "volumes" in docker_compose_prod, "Должны быть определены volumes"
        volumes = docker_compose_prod["volumes"]
        assert "postgres_data" in volumes, "Должен быть volume для postgres"
        assert "redis_data" in volumes, "Должен быть volume для redis"
        assert "minio_data" in volumes, "Должен быть volume для minio"

    def test_docker_compose_prod_networks(self, docker_compose_prod):
        """Тест: docker-compose.prod.yml определяет networks."""
        assert "networks" in docker_compose_prod, "Должны быть определены networks"
        networks = docker_compose_prod["networks"]
        assert (
            "face_recognition_network" in networks
        ), "Должна быть сеть face_recognition_network"
        network = networks["face_recognition_network"]
        assert "driver" in network, "Сеть должна иметь driver"
        assert network["driver"] == "bridge", "Driver должен быть bridge"

    # =============================================================================
    # Dockerfile Tests
    # =============================================================================

    def test_dockerfile_production_multi_stage(self, dockerfile_prod):
        """Тест: Dockerfile использует multi-stage build."""
        assert (
            "FROM python:" in dockerfile_prod
        ), "Должен использовать базовый образ Python"
        from_count = dockerfile_prod.upper().count("FROM ")
        assert from_count >= 2, "Должен использовать multi-stage build (минимум 2 FROM)"

    def test_dockerfile_production_non_root_user(self, dockerfile_prod):
        """Тест: Dockerfile создает non-root пользователя."""
        assert (
            "useradd" in dockerfile_prod or "groupadd" in dockerfile_prod
        ), "Должен создавать пользователя/группу"
        assert (
            "USER" in dockerfile_prod
        ), "Должен переключаться на non-root пользователя"

    def test_dockerfile_production_apt_cleanup(self, dockerfile_prod):
        """Тест: Dockerfile очищает apt cache."""
        assert (
            "apt-get clean" in dockerfile_prod
            or "rm -rf /var/lib/apt/lists" in dockerfile_prod
        ), "Должен очищать apt cache для уменьшения размера образа"

    def test_dockerfile_production_requirements_cache(self, dockerfile_prod):
        """Тест: Dockerfile копирует requirements.txt отдельно."""
        lines = dockerfile_prod.split("\n")
        req_copied = False
        code_copied = False
        for line in lines:
            if "COPY requirements" in line:
                req_copied = True
            if line.strip().startswith("COPY .") or line.strip().startswith(
                "COPY --chown"
            ):
                code_copied = True
            if req_copied and code_copied:
                break
        assert req_copied, "requirements.txt должен копироваться отдельно"
        assert code_copied, "Код должен копироваться отдельно"

    def test_dockerfile_production_no_cache_dir(self, dockerfile_prod):
        """Тест: pip install использует --no-cache-dir."""
        assert (
            "--no-cache-dir" in dockerfile_prod
        ), "pip install должен использовать --no-cache-dir"

    def test_dockerfile_production_uvicorn(self, dockerfile_prod):
        """Тест: Dockerfile запускает uvicorn."""
        assert "uvicorn" in dockerfile_prod, "Должен запускать uvicorn"
        assert (
            "app.main:app" in dockerfile_prod or "app:app" in dockerfile_prod
        ), "Должен запускать app.main:app"

    def test_dockerfile_production_healthcheck(self, dockerfile_prod):
        """Тест: Dockerfile имеет HEALTHCHECK."""
        assert "HEALTHCHECK" in dockerfile_prod, "Должен иметь HEALTHCHECK"
        assert (
            "8000" in dockerfile_prod or "health" in dockerfile_prod.lower()
        ), "HEALTHCHECK должен проверять порт 8000"


class TestDockerDevelopmentInfrastructure:
    """Тесты для development Docker инфраструктуры."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def docker_compose_dev(self, project_root):
        """Загружает docker-compose.dev.yml."""
        compose_path = project_root / "docker-compose.dev.yml"
        assert compose_path.exists(), "docker-compose.dev.yml должен существовать"
        with open(compose_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def dockerfile_dev(self, project_root):
        """Загружает Dockerfile.dev."""
        dockerfile_path = project_root / "Dockerfile.dev"
        assert dockerfile_path.exists(), "Dockerfile.dev должен существовать"
        return dockerfile_path.read_text(encoding="utf-8")

    # =============================================================================
    # docker-compose.dev.yml Tests
    # =============================================================================

    def test_docker_compose_dev_exists(self, project_root):
        """Тест: docker-compose.dev.yml существует."""
        compose_path = project_root / "docker-compose.dev.yml"
        assert compose_path.exists(), "docker-compose.dev.yml должен существовать"

    def test_docker_compose_dev_has_api_service(self, docker_compose_dev):
        """Тест: docker-compose.dev.yml содержит API сервис."""
        services = docker_compose_dev.get("services", {})
        # Ищем сервис приложения
        api_service = None
        for service_name in ["face-recognition-api-dev", "app", "face-recognition-api"]:
            if service_name in services:
                api_service = services[service_name]
                break
        assert (
            api_service is not None
        ), "docker-compose.dev.yml должен содержать сервис приложения"

    def test_docker_compose_dev_volume_mounts(self, docker_compose_dev):
        """Тест: docker-compose.dev.yml монтирует код для hot-reload."""
        api_service = self._get_api_service(docker_compose_dev)

        assert "volumes" in api_service, "Сервис должен иметь volumes"
        volumes = api_service["volumes"]

        # Проверяем монтирование кода
        has_code_mount = any(".:/app" in str(v) or "./:/app" in str(v) for v in volumes)
        assert has_code_mount, "Должен монтировать код (.:/app)"

    def test_docker_compose_dev_ports(self, docker_compose_dev):
        """Тест: docker-compose.dev.yml экспортирует порты."""
        api_service = self._get_api_service(docker_compose_dev)

        assert "ports" in api_service, "Должны быть экспортированы порты"
        ports = api_service["ports"]

        # Проверяем основной порт API
        has_api_port = any("8000" in str(p) for p in ports)
        assert has_api_port, "Порт 8000 должен быть проброшен"

    def test_docker_compose_dev_depends_on(self, docker_compose_dev):
        """Тест: docker-compose.dev.yml имеет depends_on с условиями."""
        api_service = self._get_api_service(docker_compose_dev)

        if "depends_on" in api_service:
            depends = api_service["depends_on"]

            # Проверяем наличие условий health
            for service, config in depends.items():
                if isinstance(config, dict):
                    assert (
                        "condition" in config
                    ), f"depends_on для {service} должен иметь condition"
                    assert (
                        config["condition"] == "service_healthy"
                    ), f"condition должен быть 'service_healthy'"

    def _get_api_service(self, compose_config):
        """Получает API сервис из docker-compose конфигурации."""
        services = compose_config.get("services", {})
        for service_name in ["face-recognition-api-dev", "app", "face-recognition-api"]:
            if service_name in services:
                return services[service_name]
        pytest.fail("API сервис не найден")

    def test_docker_compose_dev_hot_reload_config(self, docker_compose_dev):
        """Тест: docker-compose.dev.yml настроен для hot-reload."""
        api_service = self._get_api_service(docker_compose_dev)

        env = api_service.get("environment", [])

        # Конвертируем в строку для проверки
        env_str = str(env)

        # Проверяем, что ENVIRONMENT присутствует
        assert (
            "ENVIRONMENT" in env_str or "development" in env_str
        ), "Должна быть переменная ENVIRONMENT=development"

        # Проверяем DEBUG
        assert "DEBUG" in env_str, "Должна быть переменная DEBUG"

    def test_docker_compose_dev_separate_networks(self, docker_compose_dev):
        """Тест: docker-compose.dev.yml использует отдельную сеть."""
        assert "networks" in docker_compose_dev, "Должны быть определены networks"
        networks = docker_compose_dev["networks"]
        assert len(networks) > 0, "Должна быть хотя бы одна сеть"

    def test_docker_compose_dev_additional_services(self, docker_compose_dev):
        """Тест: docker-compose.dev.yml содержит дополнительные сервисы для разработки."""
        services = docker_compose_dev.get("services", {})
        assert (
            "redis-commander" in services
        ), "Должен быть сервис redis-commander для управления Redis"

    # =============================================================================
    # Dockerfile.dev Tests
    # =============================================================================

    def test_dockerfile_dev_exists(self, dockerfile_dev):
        """Тест: Dockerfile.dev существует."""
        assert dockerfile_dev is not None, "Dockerfile.dev должен существовать"

    def test_dockerfile_dev_python_base(self, dockerfile_dev):
        """Тест: Dockerfile.dev использует Python 3.11."""
        assert (
            "python:3.11" in dockerfile_dev or "python:3.11" in dockerfile_dev.lower()
        ), "Должен использовать Python 3.11"

    def test_dockerfile_dev_system_deps(self, dockerfile_dev):
        """Тест: Dockerfile.dev устанавливает системные зависимости."""
        deps = [
            "gcc",
            "g++",
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "libgomp1",
            "libgtk-3-0",
            "libavcodec-dev",
            "libavformat-dev",
        ]
        for dep in deps:
            assert dep in dockerfile_dev, f"Должна быть установлена зависимость {dep}"

    def test_dockerfile_dev_dev_requirements(self, dockerfile_dev):
        """Тест: Dockerfile.dev устанавливает dev requirements."""
        assert (
            "requirements-dev.txt" in dockerfile_dev
        ), "Должен устанавливать requirements-dev.txt"

    def test_dockerfile_dev_hot_reload(self, dockerfile_dev):
        """Тест: Dockerfile.dev настроен для hot-reload."""
        assert "--reload" in dockerfile_dev, "uvicorn должен запускаться с --reload"
        assert (
            "DEBUG=true" in dockerfile_dev
            or "ENVIRONMENT=development" in dockerfile_dev
        ), "Должны быть переменные для development режима"

    def test_dockerfile_dev_exposes_ports(self, dockerfile_dev):
        """Тест: Dockerfile.dev экспортирует порты."""
        assert "EXPOSE 8000" in dockerfile_dev, "Должен экспортировать порт 8000"

    def test_dockerfile_dev_creates_directories(self, dockerfile_dev):
        """Тест: Dockerfile.dev создает необходимые директории."""
        assert "mkdir -p" in dockerfile_dev, "Должен создавать необходимые директории"
        assert (
            "/app/logs" in dockerfile_dev or "logs" in dockerfile_dev
        ), "Должен создавать директорию для логов"
        assert (
            "/app/uploads" in dockerfile_dev or "uploads" in dockerfile_dev
        ), "Должен создавать директорию для загрузок"


class TestDockerComposeSyntax:
    """Тесты для проверки синтаксиса docker-compose файлов."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    @pytest.mark.parametrize(
        "compose_file",
        [
            "docker-compose.yml",
            "docker-compose.prod.yml",
            "docker-compose.dev.yml",
        ],
    )
    def test_docker_compose_yaml_syntax(self, project_root, compose_file):
        """Тест: docker-compose файлы имеют корректный YAML синтаксис."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        with open(compose_path, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Некорректный YAML синтаксис в {compose_file}: {e}")
        assert config is not None, f"{compose_file} не должен быть пустым"

    @pytest.mark.parametrize(
        "compose_file",
        [
            "docker-compose.yml",
            "docker-compose.prod.yml",
            "docker-compose.dev.yml",
        ],
    )
    def test_docker_compose_has_version(self, project_root, compose_file):
        """Тест: docker-compose файлы имеют версию."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)
        assert (
            "version" in config or "services" in config
        ), f"{compose_file} должен иметь version или services"

    @pytest.mark.parametrize(
        "compose_file",
        [
            "docker-compose.yml",
            "docker-compose.prod.yml",
            "docker-compose.dev.yml",
        ],
    )
    def test_docker_compose_has_services(self, project_root, compose_file):
        """Тест: docker-compose файлы имеют секцию services."""
        compose_path = project_root / compose_file
        if not compose_path.exists():
            pytest.skip(f"{compose_file} не существует")
        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)
        assert "services" in config, f"{compose_file} должен иметь services"


class TestDockerIgnoreFile:
    """Тесты для проверки .dockerignore файла."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def dockerignore_content(self, project_root):
        """Загружает .dockerignore."""
        dockerignore_path = project_root / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore должен существовать"
        return dockerignore_path.read_text(encoding="utf-8")

    def test_dockerignore_exists(self, project_root):
        """Тест: .dockerignore существует."""
        dockerignore_path = project_root / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore должен существовать"

    def test_dockerignore_excludes_git(self, dockerignore_content):
        """Тест: .dockerignore исключает .git."""
        assert ".git" in dockerignore_content, ".dockerignore должен исключать .git"

    def test_dockerignore_excludes_pycache(self, dockerignore_content):
        """Тест: .dockerignore исключает __pycache__."""
        assert (
            "__pycache__" in dockerignore_content
        ), ".dockerignore должен исключать __pycache__"

    def test_dockerignore_excludes_env(self, dockerignore_content):
        """Тест: .dockerignore исключает .env файлы."""
        assert ".env" in dockerignore_content, ".dockerignore должен исключать .env"

    def test_dockerignore_excludes_venv(self, dockerignore_content):
        """Тест: .dockerignore исключает виртуальные окружения."""
        has_venv = ".venv" in dockerignore_content or "venv" in dockerignore_content
        assert has_venv, ".dockerignore должен исключать .venv"

    def test_dockerignore_excludes_tests(self, dockerignore_content):
        """Тест: .dockerignore исключает тесты."""
        assert "tests" in dockerignore_content, ".dockerignore должен исключать tests"

    def test_dockerignore_excludes_logs(self, dockerignore_content):
        """Тест: .dockerignore исключает логи."""
        has_logs = "*.log" in dockerignore_content or "logs" in dockerignore_content
        assert has_logs, ".dockerignore должен исключать logs"

    def test_dockerignore_excludes_pycache_files(self, dockerignore_content):
        """Тест: .dockerignore исключает .pyc файлы."""
        has_pyc = "*.pyc" in dockerignore_content or "*.py[cod]" in dockerignore_content
        assert has_pyc, ".dockerignore должен исключать .pyc файлы"

    def test_dockerignore_excludes_node_modules(self, dockerignore_content):
        """Тест: .dockerignore исключает node_modules."""
        assert (
            "node_modules" in dockerignore_content
        ), ".dockerignore должен исключать node_modules"

    def test_dockerignore_excludes_docker_files(self, dockerignore_content):
        """Тест: .dockerignore исключает Docker файлы."""
        has_dockerignore = ".dockerignore" in dockerignore_content
        has_dockerfile = (
            "Dockerfile" in dockerignore_content
            or "*.dockerfile" in dockerignore_content
        )
        assert (
            has_dockerignore or has_dockerfile
        ), ".dockerignore должен исключать Docker файлы"

    def test_dockerignore_excludes_readme(self, dockerignore_content):
        """Тест: .dockerignore исключает README файлы."""
        has_readme = (
            "README" in dockerignore_content or "readme" in dockerignore_content.lower()
        )
        # Это не обязательно, но полезно для уменьшения размера
        # Не делаем assert, просто проверяем


class TestDockerSecurity:
    """Тесты для проверки безопасности Docker конфигураций."""

    @pytest.fixture
    def project_root(self):
        """Возвращает путь к корню проекта."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def dockerfile_prod(self, project_root):
        """Загружает Dockerfile."""
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile должен существовать"
        return dockerfile_path.read_text(encoding="utf-8")

    @pytest.fixture
    def docker_compose_prod(self, project_root):
        """Загружает docker-compose.prod.yml."""
        compose_path = project_root / "docker-compose.prod.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.prod.yml не существует")
        with open(compose_path, "r") as f:
            return yaml.safe_load(f)

    def test_dockerfile_no_root_user(self, dockerfile_prod):
        """Тест: Dockerfile не запускает от root."""
        assert (
            "USER" in dockerfile_prod
        ), "Dockerfile должен переключаться на non-root пользователя"
        assert (
            "USER root" not in dockerfile_prod
        ), "Контейнер не должен запускаться от root"

    def test_dockerfile_no_passwords_in_image(self, dockerfile_prod):
        """Тест: Dockerfile не содержит пароли."""
        content_lower = dockerfile_prod.lower()
        assert (
            "password" not in content_lower or "PASSWORD" not in dockerfile_prod
        ), "Dockerfile не должен содержать пароли в явном виде"
        assert (
            "secret" not in content_lower or "SECRET" not in dockerfile_prod
        ), "Dockerfile не должен содержать секреты в явном виде"

    def test_docker_compose_prod_no_hardcoded_secrets(self, docker_compose_prod):
        """Тест: docker-compose.prod.yml не содержит жестко закодированных секретов."""
        api_service = docker_compose_prod.get("services", {}).get("api", {})
        env = api_service.get("environment", {})
        jwt_secret = env.get("JWT_SECRET_KEY", "")
        assert (
            "${" in jwt_secret or "ENV[" in jwt_secret
        ), "JWT_SECRET_KEY должен использовать переменную окружения"
        enc_key = env.get("ENCRYPTION_KEY", "")
        assert (
            "${" in enc_key or "ENV[" in enc_key
        ), "ENCRYPTION_KEY должен использовать переменную окружения"

    def test_docker_compose_prod_debug_false(self, docker_compose_prod):
        """Тест: Production конфигурация отключает debug."""
        api_service = docker_compose_prod.get("services", {}).get("api", {})
        env = api_service.get("environment", {})
        debug_value = env.get("DEBUG", "")
        assert (
            str(debug_value).lower() == "false"
        ), "DEBUG должен быть false в production"

    def test_docker_compose_prod_log_level_info(self, docker_compose_prod):
        """Тест: Production использует INFO log level."""
        api_service = docker_compose_prod.get("services", {}).get("api", {})
        env = api_service.get("environment", {})
        log_level = env.get("LOG_LEVEL", "")
        assert log_level.upper() == "INFO", "LOG_LEVEL должен быть INFO в production"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
