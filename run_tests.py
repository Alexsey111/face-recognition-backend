#!/usr/bin/env python3
"""
Скрипт для запуска тестов Face Recognition Service
"""
import subprocess
import sys
import os


def run_command(cmd, description):
    """Выполняет команду и выводит результат"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Команда: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="face-recognition-service")
        
        if result.stdout:
            print("📄 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Команда выполнена успешно")
        else:
            print(f"❌ Команда завершилась с кодом: {result.returncode}")
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Ошибка выполнения команды: {e}")
        return False


def main():
    """Основная функция для запуска тестов"""
    print("🚀 Запуск тестов Face Recognition Service")
    
    # Проверяем наличие pytest
    if not run_command("python -m pytest --version", "Проверка pytest"):
        print("❌ pytest не найден. Устанавливаем...")
        run_command("pip install -r requirements-dev.txt", "Установка зависимостей для разработки")
    
    # Запуск тестов
    tests = [
        ("python -m pytest tests/unit/ -v --tb=short", "Запуск юнит-тестов"),
        ("python -m pytest tests/integration/ -v --tb=short", "Запуск интеграционных тестов"),
        ("python -m pytest tests/ -v --cov=app --cov-report=term-missing", "Все тесты с покрытием кода"),
        ("python -m pytest tests/ -m unit -v", "Только быстрые юнит-тесты"),
        ("python -m pytest tests/ -m integration -v", "Только интеграционные тесты"),
    ]
    
    success_count = 0
    for cmd, description in tests:
        if run_command(cmd, description):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 Результаты: {success_count}/{len(tests)} наборов тестов выполнены успешно")
    
    if success_count == len(tests):
        print("🎉 Все тесты прошли успешно!")
    else:
        print("⚠️  Некоторые тесты не прошли. Проверьте вывод выше.")
    
    return success_count == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)