#!/usr/bin/env python3
"""Скрипт для проверки импортов FastAPI приложения."""

import sys
import os

# Добавляем текущую директорию в Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Тестирование импортов."""
    print("=== Testing imports ===")
    
    try:
        print("1. Importing app.main...")
        import app.main
        print("✅ app.main imported successfully")
        
        print("2. Importing app.config...")
        import app.config
        print("✅ app.config imported successfully")
        
        print("3. Importing app.models.response...")
        import app.models.response
        print("✅ app.models.response imported successfully")
        
        print("4. Importing app.routes.health...")
        import app.routes.health
        print("✅ app.routes.health imported successfully")
        
        print("5. Importing app.middleware.auth...")
        import app.middleware.auth
        print("✅ app.middleware.auth imported successfully")
        
        print("6. Importing app.middleware.rate_limit...")
        import app.middleware.rate_limit
        print("✅ app.middleware.rate_limit imported successfully")
        
        print("7. Importing app.middleware.logging...")
        import app.middleware.logging
        print("✅ app.middleware.logging imported successfully")
        
        print("8. Importing app.middleware.error_handler...")
        import app.middleware.error_handler
        print("✅ app.middleware.error_handler imported successfully")
        
        print("=== All imports successful! ===")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_app_creation():
    """Тестирование создания приложения."""
    print("\n=== Testing app creation ===")
    
    try:
        from app.main import app
        print("✅ App created successfully")
        print(f"App title: {app.title}")
        print(f"App version: {app.version}")
        print(f"OpenAPI URL: {app.openapi_url}")
        return True
        
    except Exception as e:
        print(f"❌ App creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Face Recognition Service - Import Test")
    print("=" * 50)
    
    # Тестируем импорты
    imports_ok = test_imports()
    
    if imports_ok:
        # Тестируем создание приложения
        app_ok = test_app_creation()
        
        if app_ok:
            print("\n🎉 All tests passed! Phase 2 is ready.")
        else:
            print("\n⚠️  App creation failed.")
    else:
        print("\n❌ Import tests failed.")