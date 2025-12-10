#!/usr/bin/env python3
"""
Скрипт для проверки выполнения Фазы 1: Infrastructure.
Простой запуск: python check_phase1.py
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Проверяет существование файла."""
    if file_path.exists():
        print(f"[OK] {description}: {file_path}")
        return True
    else:
        print(f"[FAIL] {description}: {file_path} - НЕ НАЙДЕН")
        return False

def check_file_content(file_path, search_text, description):
    """Проверяет содержимое файла."""
    try:
        content = file_path.read_text()
        if search_text in content:
            print(f"[OK] {description}")
            return True
        else:
            print(f"[FAIL] {description} - ТЕКСТ НЕ НАЙДЕН")
            return False
    except Exception as e:
        print(f"[FAIL] {description} - ОШИБКА ЧТЕНИЯ: {e}")
        return False

def check_python_import(module_name, description):
    """Проверяет возможность импорта Python модуля."""
    try:
        __import__(module_name)
        print(f"[OK] {description}")
        return True
    except ImportError as e:
        print(f"[FAIL] {description} - ИМПОРТ НЕ УДАЛСЯ: {e}")
        return False

def main():
    """Основная функция проверки."""
    
    print("=== ПРОВЕРКА ВЫПОЛНЕНИЯ ФАЗЫ 1: INFRASTRUCTURE ===")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    results = []
    
    # =============================================================================
    # ISSUE 1: Dockerfile production
    # =============================================================================
    print("\nISSUE 1: Dockerfile production")
    dockerfile = project_root / "Dockerfile"
    if check_file_exists(dockerfile, "Dockerfile"):
        results.append(check_file_content(dockerfile, "FROM python:", "Multi-stage Python base"))
        results.append(check_file_content(dockerfile, "USER", "Non-root user"))
        results.append(check_file_content(dockerfile, "uvicorn", "Uvicorn startup"))
    
    # =============================================================================
    # ISSUE 2: Dockerfile.dev
    # =============================================================================
    print("\nISSUE 2: Dockerfile.dev")
    dockerfile_dev = project_root / "Dockerfile.dev"
    if check_file_exists(dockerfile_dev, "Dockerfile.dev"):
        results.append(check_file_content(dockerfile_dev, "FROM python:", "Python base for dev"))
        results.append(check_file_content(dockerfile_dev, "requirements-dev.txt", "Dev dependencies"))
    
    # =============================================================================
    # ISSUE 3: docker-compose.yml
    # =============================================================================
    print("\nISSUE 3: docker-compose.yml")
    try:
        import yaml
        compose = project_root / "docker-compose.yml"
        if check_file_exists(compose, "docker-compose.yml"):
            with open(compose, 'r') as f:
                config = yaml.safe_load(f)
            
            services = config.get("services", {})
            required_services = ["postgres", "redis", "minio"]
            
            for service in required_services:
                if service in services:
                    print(f"[OK] Сервис {service} настроен")
                    results.append(True)
                else:
                    print(f"[FAIL] Сервис {service} НЕ настроен")
                    results.append(False)
        else:
            results.extend([False, False, False])
    except ImportError:
        print("[FAIL] PyYAML не установлен - пропускаем проверку docker-compose.yml")
        results.extend([False, False, False])
    
    # =============================================================================
    # ISSUE 4: docker-compose.dev.yml
    # =============================================================================
    print("\nISSUE 4: docker-compose.dev.yml")
    compose_dev = project_root / "docker-compose.dev.yml"
    if check_file_exists(compose_dev, "docker-compose.dev.yml"):
        results.append(True)
    else:
        results.append(False)
    
    # =============================================================================
    # ISSUE 5: .dockerignore
    # =============================================================================
    print("\nISSUE 5: .dockerignore")
    dockerignore = project_root / ".dockerignore"
    if check_file_exists(dockerignore, ".dockerignore"):
        required_excludes = ["__pycache__", ".git", ".env", ".venv"]
        for exclude in required_excludes:
            results.append(check_file_content(dockerignore, exclude, f"Exclude {exclude}"))
    
    # =============================================================================
    # ISSUE 6: Alembic initialization
    # =============================================================================
    print("\nISSUE 6: Alembic initialization")
    alembic_dir = project_root / "alembic"
    if check_file_exists(alembic_dir, "alembic directory"):
        env_py = alembic_dir / "env.py"
        migration = alembic_dir / "versions" / "001_initial_migration.py"
        
        results.append(check_file_exists(env_py, "alembic/env.py"))
        results.append(check_file_exists(migration, "001_initial_migration.py"))
    
    # =============================================================================
    # ISSUE 7: Database models migration
    # =============================================================================
    print("\nISSUE 7: Database models migration")
    migration = project_root / "alembic" / "versions" / "001_initial_migration.py"
    if check_file_exists(migration, "001_initial_migration.py"):
        required_tables = ["users", "references", "verification_sessions", "audit_logs"]
        for table in required_tables:
            results.append(check_file_content(migration, f"create_table('{table}'", f"Table {table}"))
    
    # =============================================================================
    # ISSUE 8: Indexes & constraints
    # =============================================================================
    print("\nISSUE 8: Indexes & constraints")
    if check_file_exists(migration, "Migration file"):
        results.append(check_file_content(migration, "create_index", "Indexes creation"))
        results.append(check_file_content(migration, "ForeignKeyConstraint", "Foreign keys"))
        results.append(check_file_content(migration, "unique", "Unique constraints"))
    
    # =============================================================================
    # ISSUE 9: Redis connection
    # =============================================================================
    print("\nISSUE 9: Redis connection")
    cache_service = project_root / "app" / "services" / "cache_service.py"
    if check_file_exists(cache_service, "CacheService"):
        results.append(check_file_content(cache_service, "class CacheService", "CacheService class"))
        results.append(check_file_content(cache_service, "redis", "Redis usage"))
        results.append(check_file_content(cache_service, "health_check", "Health check method"))
    
    # =============================================================================
    # ISSUE 10: MinIO bucket setup
    # =============================================================================
    print("\nISSUE 10: MinIO bucket setup")
    storage_service = project_root / "app" / "services" / "storage_service.py"
    setup_minio = project_root / "setup_minio.py"
    
    if check_file_exists(storage_service, "StorageService"):
        results.append(check_file_content(storage_service, "class StorageService", "StorageService class"))
        results.append(check_file_content(storage_service, "boto3", "boto3 usage"))
    
    if check_file_exists(setup_minio, "setup_minio.py"):
        results.append(check_file_content(setup_minio, "Minio", "Minio usage"))
        results.append(check_file_content(setup_minio, "bucket", "Bucket creation"))
    
    # =============================================================================
    # ISSUE 11: config.py setup
    # =============================================================================
    print("\nISSUE 11: config.py setup")
    config_file = project_root / "app" / "config.py"
    if check_file_exists(config_file, "config.py"):
        results.append(check_file_content(config_file, "class Settings", "Settings class"))
        results.append(check_file_content(config_file, "BaseSettings", "BaseSettings inheritance"))
        results.append(check_file_content(config_file, "DATABASE_URL", "Database config"))
        results.append(check_file_content(config_file, "REDIS_URL", "Redis config"))
        results.append(check_file_content(config_file, "S3_ENDPOINT_URL", "S3 config"))
        results.append(check_file_content(config_file, "JWT_SECRET_KEY", "JWT config"))
    
    # =============================================================================
    # ISSUE 12: .env.example & requirements.txt
    # =============================================================================
    print("\nISSUE 12: .env.example & requirements.txt")
    env_example = project_root / ".env.example"
    requirements = project_root / "requirements.txt"
    
    if check_file_exists(env_example, ".env.example"):
        results.append(check_file_content(env_example, "DATABASE_URL", "DATABASE_URL in .env.example"))
        results.append(check_file_content(env_example, "REDIS_URL", "REDIS_URL in .env.example"))
    
    if check_file_exists(requirements, "requirements.txt"):
        required_deps = ["fastapi", "sqlalchemy", "alembic", "redis", "boto3"]
        for dep in required_deps:
            results.append(check_file_content(requirements, dep, f"{dep} in requirements.txt"))
    
    # =============================================================================
    # ISSUE 13: Python dependencies setup
    # =============================================================================
    print("\nISSUE 13: Python dependencies setup")
    requirements_dev = project_root / "requirements-dev.txt"
    pytest_ini = project_root / "pytest.ini"
    pyproject = project_root / "pyproject.toml"
    
    if check_file_exists(requirements_dev, "requirements-dev.txt"):
        results.append(check_file_content(requirements_dev, "pytest", "pytest in dev deps"))
        results.append(check_file_content(requirements_dev, "black", "black in dev deps"))
        results.append(check_file_content(requirements_dev, "flake8", "flake8 in dev deps"))
    
    results.append(check_file_exists(pytest_ini, "pytest.ini"))
    results.append(check_file_exists(pyproject, "pyproject.toml"))
    
    # =============================================================================
    # ПРОВЕРКА ИМПОРТОВ
    # =============================================================================
    print("\nПРОВЕРКА PYTHON ИМПОРТОВ")
    results.append(check_python_import("sqlalchemy", "SQLAlchemy import"))
    results.append(check_python_import("alembic", "Alembic import"))
    results.append(check_python_import("redis", "Redis import"))
    results.append(check_python_import("boto3", "boto3 import"))
    results.append(check_python_import("fastapi", "FastAPI import"))
    
    # =============================================================================
    # ИТОГОВЫЙ ОТЧЕТ
    # =============================================================================
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЕТ:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"ПРОЙДЕНО: {passed}/{total} проверок ({percentage:.1f}%)")
    
    if passed == total:
        print("\nФАЗА 1 ПОЛНОСТЬЮ ВЫПОЛНЕНА!")
        print("Проект готов к переходу к Фазе 2!")
        return True
    elif passed >= total * 0.8:  # 80%
        print(f"\nФАЗА 1 ПРАКТИЧЕСКИ ВЫПОЛНЕНА ({percentage:.1f}%)")
        print("Требуется доработка нескольких компонентов")
        return False
    else:
        print(f"\nФАЗА 1 НЕ ВЫПОЛНЕНА ({percentage:.1f}%)")
        print("Требуется значительная доработка")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)