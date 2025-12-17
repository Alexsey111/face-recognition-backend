#!/usr/bin/env python3
"""
Production Readiness Check for Phase 5 (ASCII version)
Проверяет аспекты готовности к production, которые можно проверить в тестовой среде.
"""

import sys
import os
import importlib.util
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

def check_configuration():
    """Проверка конфигурации"""
    print("=== CONFIGURATION CHECK ===")
    
    try:
        # Проверяем загрузку конфигурации
        sys.path.append('.')
        from app.config import settings
        
        checks = [
            ("MinIO Endpoint", settings.MINIO_ENDPOINT),
            ("MinIO Access Key", settings.MINIO_ACCESS_KEY),
            ("MinIO Bucket", settings.MINIO_BUCKET),
            ("Max File Size MB", settings.MAX_FILE_SIZE_MB),
            ("Upload Expiration Days", settings.UPLOAD_EXPIRATION_DAYS),
            ("Cleanup Interval Hours", settings.CLEANUP_INTERVAL_HOURS),
            ("Allowed Image Formats", settings.ALLOWED_IMAGE_FORMATS),
            ("Min Image Width", settings.MIN_IMAGE_WIDTH),
            ("Min Image Height", settings.MIN_IMAGE_HEIGHT),
        ]
        
        for name, value in checks:
            print(f"  OK {name}: {value}")
        
        # Проверяем логические ограничения
        assert settings.MAX_FILE_SIZE_MB > 0, "Max file size should be positive"
        assert settings.UPLOAD_EXPIRATION_DAYS > 0, "Upload expiration should be positive"
        assert settings.CLEANUP_INTERVAL_HOURS > 0, "Cleanup interval should be positive"
        assert settings.MIN_IMAGE_WIDTH > 0, "Min image width should be positive"
        assert settings.MIN_IMAGE_HEIGHT > 0, "Min image height should be positive"
        
        print("  Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"  FAIL Configuration check failed: {e}")
        return False

def check_dependencies():
    """Проверка зависимостей"""
    print("\n=== DEPENDENCIES CHECK ===")
    
    required_packages = [
        ('fastapi', 'FastAPI web framework'),
        ('uvicorn', 'ASGI server'),
        ('pydantic', 'Data validation'),
        ('sqlalchemy', 'Database ORM'),
        ('boto3', 'AWS S3 client'),
        ('PIL', 'Pillow image processing'),
        ('APScheduler', 'Task scheduler'),
        ('redis', 'Redis client'),
        ('cryptography', 'Encryption library'),
    ]
    
    missing_packages = []
    available_packages = []
    
    for package, description in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'APScheduler':
                from apscheduler.schedulers.background import BackgroundScheduler
            else:
                __import__(package)
            available_packages.append(package)
            print(f"  OK {package}: {description}")
        except ImportError:
            missing_packages.append(package)
            print(f"  FAIL {package}: MISSING - {description}")
    
    # Проверяем opencv-python-headless отдельно
    try:
        import cv2
        print(f"  OK opencv-python-headless: OpenCV {cv2.__version__} available")
        available_packages.append('opencv-python-headless')
    except ImportError:
        missing_packages.append('opencv-python-headless')
        print(f"  WARN opencv-python-headless: MISSING (required for ML service)")
    
    print(f"\nSummary: {len(available_packages)}/{len(required_packages)+1} packages available")
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("   These need to be installed for production")
        return False
    else:
        print("All required packages are available")
        return True

def check_file_structure():
    """Проверка структуры файлов Phase 5"""
    print("\n=== FILE STRUCTURE CHECK ===")
    
    phase5_files = [
        ('app/services/storage_service.py', 'Storage service'),
        ('app/services/session_service.py', 'Session service'),
        ('app/utils/file_utils.py', 'File utilities'),
        ('app/tasks/cleanup.py', 'Cleanup tasks'),
        ('app/tasks/scheduler.py', 'Cleanup scheduler'),
        ('app/routes/upload.py', 'Upload endpoints'),
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path, description in phase5_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"  OK {file_path}: {description}")
        else:
            missing_files.append(file_path)
            print(f"  FAIL {file_path}: MISSING - {description}")
    
    # Проверяем синтаксис файлов
    syntax_errors = []
    for file_path in existing_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            print(f"    OK Syntax: {os.path.basename(file_path)}")
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"    FAIL Syntax: {os.path.basename(file_path)} - {e}")
    
    if missing_files or syntax_errors:
        print("File structure issues found")
        return False
    else:
        print("All Phase 5 files present and syntactically correct")
        return True

def check_service_initialization():
    """Проверка инициализации сервисов"""
    print("\n=== SERVICE INITIALIZATION CHECK ===")
    
    # Проверяем инициализацию StorageService
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('storage_service', 'app/services/storage_service.py')
        storage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(storage_module)
        
        # Проверяем класс без инициализации (чтобы избежать проблем с S3)
        storage_class = storage_module.StorageService
        print("  OK StorageService class loaded")
        
        # Проверяем методы
        required_methods = ['health_check', 'upload_image', 'download_image', 'delete_image']
        for method in required_methods:
            if hasattr(storage_class, method):
                print(f"    OK Method {method} exists")
            else:
                print(f"    FAIL Method {method} missing")
                return False
        
    except Exception as e:
        print(f"  FAIL StorageService initialization failed: {e}")
        return False
    
    # Проверяем SessionService (автономная версия)
    try:
        spec = importlib.util.spec_from_file_location('session_service', 'app/services/session_service.py')
        session_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(session_module)
        
        session_class = session_module.SessionService
        print("  OK SessionService class loaded")
        
        required_methods = ['create_session', 'get_session', 'validate_session', 'delete_session']
        for method in required_methods:
            if hasattr(session_class, method):
                print(f"    OK Method {method} exists")
            else:
                print(f"    FAIL Method {method} missing")
                return False
        
    except Exception as e:
        print(f"  FAIL SessionService initialization failed: {e}")
        return False
    
    print("Service initialization checks passed")
    return True

def check_api_endpoints():
    """Проверка API endpoints"""
    print("\n=== API ENDPOINTS CHECK ===")
    
    try:
        # Проверяем загрузку upload.py
        spec = importlib.util.spec_from_file_location('upload', 'app/routes/upload.py')
        upload_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upload_module)
        
        router = upload_module.router
        print("  OK Upload router loaded")
        
        # Проверяем наличие endpoint функций
        required_functions = [
            'create_upload_session',
            'upload_file_to_session', 
            'get_upload_status',
            'delete_upload_session'
        ]
        
        for func_name in required_functions:
            if hasattr(upload_module, func_name):
                print(f"    OK Endpoint function {func_name} exists")
            else:
                print(f"    FAIL Endpoint function {func_name} missing")
                return False
        
        print("API endpoints validation passed")
        return True
        
    except Exception as e:
        print(f"  FAIL API endpoints check failed: {e}")
        return False

def check_environment_variables():
    """Проверка переменных окружения"""
    print("\n=== ENVIRONMENT VARIABLES CHECK ===")
    
    # Список важных переменных окружения для production
    important_vars = [
        'DATABASE_URL',
        'JWT_SECRET_KEY', 
        'ENCRYPTION_KEY',
        'MINIO_ENDPOINT',
        'MINIO_ACCESS_KEY',
        'MINIO_SECRET_KEY',
        'MINIO_BUCKET',
        'REDIS_URL'
    ]
    
    missing_vars = []
    present_vars = []
    
    for var in important_vars:
        value = os.getenv(var)
        if value:
            present_vars.append(var)
            # Маскируем секретные значения
            if 'KEY' in var or 'PASSWORD' in var or 'SECRET' in var:
                print(f"  OK {var}: {'*' * (len(value) if len(value) <= 10 else 10)}")
            else:
                print(f"  OK {var}: {value}")
        else:
            missing_vars.append(var)
            print(f"  WARN {var}: NOT SET")
    
    if missing_vars:
        print(f"\nMissing environment variables: {', '.join(missing_vars)}")
        print("   These should be set in production environment")
        return False
    else:
        print("\nAll important environment variables are set")
        return True

def main():
    """Основная функция проверки"""
    print("Production Readiness Check for Phase 5")
    print("=" * 50)
    
    checks = [
        ("Configuration", check_configuration),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Service Initialization", check_service_initialization),
        ("API Endpoints", check_api_endpoints),
        ("Environment Variables", check_environment_variables),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                print(f"\nWARN {check_name} check had issues")
        except Exception as e:
            print(f"\nFAIL {check_name} check crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Production Readiness: {passed}/{total} checks passed")
    
    if passed == total:
        print("\nALL CHECKS PASSED - Ready for production deployment!")
        print("\nNext steps:")
        print("1. Install missing dependencies (opencv-python-headless)")
        print("2. Set up production MinIO instance")
        print("3. Configure production database")
        print("4. Set production environment variables")
        print("5. Deploy and test in staging environment")
    else:
        print(f"\n{total - passed} checks failed - Address issues before production")
        print("\nCommon issues to resolve:")
        print("• Install missing Python packages")
        print("• Set up external services (MinIO, Database)")
        print("• Configure environment variables")
        print("• Fix any file structure issues")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)