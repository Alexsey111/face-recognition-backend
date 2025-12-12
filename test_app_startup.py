#!/usr/bin/env python3
"""
Простой тест для проверки запуска FastAPI приложения.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_app_import():
    """Тест импорта приложения."""
    try:
        from app.main import app
        print("App imported successfully")
        print(f"App title: {app.title}")
        print(f"App version: {app.version}")
        print(f"Number of routes: {len(app.routes)}")
        return True
    except Exception as e:
        print(f"App import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_routes():
    """Тест роутов приложения."""
    try:
        from app.main import app
        routes = app.routes
        print(f"Found {len(routes)} routes:")
        for route in routes:
            if hasattr(route, 'path'):
                print(f"  - {route.path}")
        return True
    except Exception as e:
        print(f"Routes test failed: {e}")
        return False

def test_health_endpoint():
    """Тест health endpoint."""
    try:
        import requests
        from threading import Thread
        import time
        import uvicorn
        
        from app.main import app
        
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
        
        # Запускаем сервер в фоне
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Ждем запуска сервера
        time.sleep(2)
        
        # Тестируем health endpoint
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            if response.status_code == 200:
                print("Health endpoint working")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"Health endpoint returned {response.status_code}")
                return False
        except Exception as e:
            print(f"Health endpoint test failed: {e}")
            return False
            
    except Exception as e:
        print(f"Health endpoint test setup failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Face Recognition Service...")
    print("=" * 50)
    
    # Тесты
    tests_passed = 0
    total_tests = 3
    
    if test_app_import():
        tests_passed += 1
    
    if test_app_routes():
        tests_passed += 1
    
    if test_health_endpoint():
        tests_passed += 1
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("All tests passed! Phase 2 is working correctly.")
    else:
        print("Some tests failed. Check the errors above.")