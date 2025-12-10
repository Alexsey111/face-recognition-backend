"""
Быстрая проверка выполнения Фазы 1: Infrastructure.
Запускает базовые проверки всех 13 задач первой фазы.
"""

import os
import sys
from pathlib import Path

# Добавляем путь к корню проекта
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_phase1_completion():
    """Основная функция проверки выполнения Фазы 1."""
    
    print("🔍 ПРОВЕРКА ВЫПОЛНЕНИЯ ФАЗЫ 1: INFRASTRUCTURE")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent.parent
    results = []
    
    # =============================================================================
    # ISSUE 1: Dockerfile production
    # =============================================================================
    try:
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists(), "❌ Dockerfile не найден"
        
        content = dockerfile.read_text()
        assert "FROM python:" in content, "❌ Dockerfile не использует Python base image"
        assert "USER" in content, "❌ Dockerfile не создает non-root user"
        
        print("✅ ISSUE 1: Dockerfile production - ВЫПОЛНЕН")
        results.append(("Issue 1", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 1: Dockerfile production - ОШИБКА: {e}")
        results.append(("Issue 1", False, str(e)))
    
    # =============================================================================
    # ISSUE 2: Dockerfile.dev
    # =============================================================================
    try:
        dockerfile_dev = project_root / "Dockerfile.dev"
        assert dockerfile_dev.exists(), "❌ Dockerfile.dev не найден"
        
        content = dockerfile_dev.read_text()
        assert "FROM python:" in content, "❌ Dockerfile.dev не использует Python base image"
        assert "requirements-dev.txt" in content, "❌ Dockerfile.dev не устанавливает dev dependencies"
        
        print("✅ ISSUE 2: Dockerfile.dev - ВЫПОЛНЕН")
        results.append(("Issue 2", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 2: Dockerfile.dev - ОШИБКА: {e}")
        results.append(("Issue 2", False, str(e)))
    
    # =============================================================================
    # ISSUE 3: docker-compose.yml
    # =============================================================================
    try:
        import yaml
        compose = project_root / "docker-compose.yml"
        assert compose.exists(), "❌ docker-compose.yml не найден"
        
        with open(compose, 'r') as f:
            config = yaml.safe_load(f)
        
        services = config.get("services", {})
        required_services = ["postgres", "redis", "minio"]
        
        for service in required_services:
            assert service in services, f"❌ Сервис {service} не найден в docker-compose.yml"
        
        print("✅ ISSUE 3: docker-compose.yml - ВЫПОЛНЕН")
        results.append(("Issue 3", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 3: docker-compose.yml - ОШИБКА: {e}")
        results.append(("Issue 3", False, str(e)))
    
    # =============================================================================
    # ISSUE 4: docker-compose.dev.yml
    # =============================================================================
    try:
        compose_dev = project_root / "docker-compose.dev.yml"
        assert compose_dev.exists(), "❌ docker-compose.dev.yml не найден"
        
        with open(compose_dev, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "services" in config, "❌ docker-compose.dev.yml не содержит services"
        
        print("✅ ISSUE 4: docker-compose.dev.yml - ВЫПОЛНЕН")
        results.append(("Issue 4", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 4: docker-compose.dev.yml - ОШИБКА: {e}")
        results.append(("Issue 4", False, str(e)))
    
    # =============================================================================
    # ISSUE 5: .dockerignore
    # =============================================================================
    try:
        dockerignore = project_root / ".dockerignore"
        assert dockerignore.exists(), "❌ .dockerignore не найден"
        
        content = dockerignore.read_text()
        required_excludes = ["__pycache__", ".git", ".env", ".venv"]
        
        for exclude in required_excludes:
            assert exclude in content, f"❌ .dockerignore не исключает {exclude}"
        
        print("✅ ISSUE 5: .dockerignore - ВЫПОЛНЕН")
        results.append(("Issue 5", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 5: .dockerignore - ОШИБКА: {e}")
        results.append(("Issue 5", False, str(e)))
    
    # =============================================================================
    # ISSUE 6: Alembic initialization
    # =============================================================================
    try:
        alembic_dir = project_root / "alembic"
        assert alembic_dir.exists(), "❌ Директория alembic не найдена"
        
        env_py = alembic_dir / "env.py"
        assert env_py.exists(), "❌ alembic/env.py не найден"
        
        migration = alembic_dir / "versions" / "001_initial_migration.py"
        assert migration.exists(), "❌ Первая миграция не найдена"
        
        print("✅ ISSUE 6: Alembic initialization - ВЫПОЛНЕН")
        results.append(("Issue 6", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 6: Alembic initialization - ОШИБКА: {e}")
        results.append(("Issue 6", False, str(e)))
    
    # =============================================================================
    # ISSUE 7: Database models migration
    # =============================================================================
    try:
        migration = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration.read_text()
        
        required_tables = ["users", "references", "verification_sessions", "audit_logs"]
        
        for table in required_tables:
            assert f"create_table('{table}'" in content, f"❌ Таблица {table} не создается"
        
        print("✅ ISSUE 7: Database models migration - ВЫПОЛНЕН")
        results.append(("Issue 7", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 7: Database models migration - ОШИБКА: {e}")
        results.append(("Issue 7", False, str(e)))
    
    # =============================================================================
    # ISSUE 8: Indexes & constraints
    # =============================================================================
    try:
        migration = project_root / "alembic" / "versions" / "001_initial_migration.py"
        content = migration.read_text()
        
        assert "create_index" in content, "❌ Индексы не создаются"
        assert "ForeignKeyConstraint" in content, "❌ Foreign key constraints не создаются"
        
        print("✅ ISSUE 8: Indexes & constraints - ВЫПОЛНЕН")
        results.append(("Issue 8", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 8: Indexes & constraints - ОШИБКА: {e}")
        results.append(("Issue 8", False, str(e)))
    
    # =============================================================================
    # ISSUE 9: Redis connection
    # =============================================================================
    try:
        # Проверяем config
        from app.config import Settings
        config = Settings()
        
        assert hasattr(config, 'REDIS_URL'), "❌ REDIS_URL не настроен в config"
        assert hasattr(config, 'REDIS_CONNECTION_POOL_SIZE'), "❌ REDIS_CONNECTION_POOL_SIZE не настроен"
        
        # Проверяем CacheService
        cache_service = project_root / "app" / "services" / "cache_service.py"
        assert cache_service.exists(), "❌ CacheService не найден"
        
        print("✅ ISSUE 9: Redis connection - ВЫПОЛНЕН")
        results.append(("Issue 9", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 9: Redis connection - ОШИБКА: {e}")
        results.append(("Issue 9", False, str(e)))
    
    # =============================================================================
    # ISSUE 10: MinIO bucket setup
    # =============================================================================
    try:
        # Проверяем config
        from app.config import Settings
        config = Settings()
        
        assert hasattr(config, 'S3_ENDPOINT_URL'), "❌ S3_ENDPOINT_URL не настроен"
        assert hasattr(config, 'S3_BUCKET_NAME'), "❌ S3_BUCKET_NAME не настроен"
        
        # Проверяем StorageService
        storage_service = project_root / "app" / "services" / "storage_service.py"
        assert storage_service.exists(), "❌ StorageService не найден"
        
        # Проверяем setup script
        setup_script = project_root / "setup_minio.py"
        assert setup_script.exists(), "❌ setup_minio.py не найден"
        
        print("✅ ISSUE 10: MinIO bucket setup - ВЫПОЛНЕН")
        results.append(("Issue 10", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 10: MinIO bucket setup - ОШИБКА: {e}")
        results.append(("Issue 10", False, str(e)))
    
    # =============================================================================
    # ISSUE 11: config.py setup
    # =============================================================================
    try:
        config_file = project_root / "app" / "config.py"
        assert config_file.exists(), "❌ config.py не найден"
        
        from app.config import Settings
        config = Settings()
        
        # Проверяем основные секции
        assert hasattr(config, 'DATABASE_URL'), "❌ DATABASE_URL не настроен"
        assert hasattr(config, 'REDIS_URL'), "❌ REDIS_URL не настроен"
        assert hasattr(config, 'S3_ENDPOINT_URL'), "❌ S3_ENDPOINT_URL не настроен"
        assert hasattr(config, 'JWT_SECRET_KEY'), "❌ JWT_SECRET_KEY не настроен"
        
        print("✅ ISSUE 11: config.py setup - ВЫПОЛНЕН")
        results.append(("Issue 11", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 11: config.py setup - ОШИБКА: {e}")
        results.append(("Issue 11", False, str(e)))
    
    # =============================================================================
    # ISSUE 12: .env.example & requirements.txt
    # =============================================================================
    try:
        env_example = project_root / ".env.example"
        assert env_example.exists(), "❌ .env.example не найден"
        
        requirements = project_root / "requirements.txt"
        assert requirements.exists(), "❌ requirements.txt не найден"
        
        # Проверяем содержимое requirements.txt
        req_content = requirements.read_text()
        required_deps = ["fastapi", "sqlalchemy", "alembic", "redis", "boto3"]
        
        for dep in required_deps:
            assert dep in req_content, f"❌ {dep} не найден в requirements.txt"
        
        print("✅ ISSUE 12: .env.example & requirements.txt - ВЫПОЛНЕН")
        results.append(("Issue 12", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 12: .env.example & requirements.txt - ОШИБКА: {e}")
        results.append(("Issue 12", False, str(e)))
    
    # =============================================================================
    # ISSUE 13: Python dependencies setup
    # =============================================================================
    try:
        requirements_dev = project_root / "requirements-dev.txt"
        assert requirements_dev.exists(), "❌ requirements-dev.txt не найден"
        
        pytest_ini = project_root / "pytest.ini"
        assert pytest_ini.exists(), "❌ pytest.ini не найден"
        
        pyproject = project_root / "pyproject.toml"
        assert pyproject.exists(), "❌ pyproject.toml не найден"
        
        # Проверяем requirements-dev.txt
        dev_content = requirements_dev.read_text()
        dev_deps = ["pytest", "black", "flake8", "mypy"]
        
        for dep in dev_deps:
            assert dep in dev_content, f"❌ {dep} не найден в requirements-dev.txt"
        
        print("✅ ISSUE 13: Python dependencies setup - ВЫПОЛНЕН")
        results.append(("Issue 13", True, ""))
    except Exception as e:
        print(f"❌ ISSUE 13: Python dependencies setup - ОШИБКА: {e}")
        results.append(("Issue 13", False, str(e)))
    
    # =============================================================================
    # ИТОГОВЫЙ ОТЧЕТ
    # =============================================================================
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ:")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for issue, success, error in results:
        status = "✅ ПРОЙДЕН" if success else f"❌ ОШИБКА: {error}"
        print(f"{issue}: {status}")
    
    print(f"\n🎯 РЕЗУЛЬТАТ: {passed}/{total} задач выполнено")
    
    if passed == total:
        print("🎉 ФАЗА 1 ПОЛНОСТЬЮ ВЫПОЛНЕНА! ✅")
        return True
    else:
        print(f"⚠️  ФАЗА 1 НЕ ПОЛНОСТЬЮ ВЫПОЛНЕНА. Не выполнено: {total - passed} задач")
        return False


if __name__ == "__main__":
    test_phase1_completion()