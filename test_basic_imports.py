#!/usr/bin/env python3
"""Упрощенный тест импортов."""

import sys
import os

# Добавляем текущую директорию в Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_basic_imports():
    """Базовые тесты импортов."""
    print("Testing basic imports...")
    
    try:
        # Проверяем, можем ли мы импортировать models как пакет
        print("1. Importing app.models...")
        import app.models
        print("SUCCESS: app.models imported")
        
        # Проверяем, можем ли мы импортировать response как модуль
        print("2. Importing app.models.response...")
        import app.models.response
        print("SUCCESS: app.models.response imported")
        
        # Проверяем, можем ли мы получить классы
        print("3. Importing classes...")
        from app.models.response import HealthResponse, StatusResponse, BaseResponse
        print("SUCCESS: Classes imported")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Basic Import Test")
    print("=" * 30)
    
    success = test_basic_imports()
    if success:
        print("\nAll basic imports work!")
    else:
        print("\nBasic imports failed!")