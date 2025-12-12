#!/usr/bin/env python3
"""
Скрипт диагностики виртуальной среды Face Recognition Service
Проверяет состояние Python, зависимостей и конфигурации
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def print_header(title):
    """Печать заголовка секции"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def print_status(item, status, message=""):
    """Печать статуса элемента"""
    icons = {
        'ok': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️'
    }
    icon = icons.get(status, '❓')
    print(f"{icon} {item:<30} {message}")

def check_python_version():
    """Проверка версии Python"""
    print_header("ПРОВЕРКА PYTHON")
    
    version = sys.version_info
    print(f"Версия Python: {version.major}.{version.minor}.{version.micro}")
    print(f"Путь к Python: {sys.executable}")
    
    if version.major == 3 and version.minor >= 11:
        print_status("Версия Python", "ok", "✓ Подходит (3.11+)")
    else:
        print_status("Версия Python", "error", "✗ Требуется 3.11+")

def check_virtual_env():
    """Проверка виртуальной среды"""
    print_header("ПРОВЕРКА ВИРТУАЛЬНОЙ СРЕДЫ")
    
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        print_status("Виртуальная среда", "ok", f"Активна: {venv_path}")
    else:
        print_status("Виртуальная среда", "warning", "Не активна")
    
    # Проверка venv директории
    if Path("venv").exists():
        print_status("Директория venv", "ok", "Существует")
    else:
        print_status("Директория venv", "warning", "Не найдена")

def check_pip():
    """Проверка pip"""
    print_header("ПРОВЕРКА PIP")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print_status("pip", "ok", result.stdout.strip())
        else:
            print_status("pip", "error", "Ошибка получения версии")
    except Exception as e:
        print_status("pip", "error", f"Ошибка: {e}")

def check_dependencies():
    """Проверка ключевых зависимостей"""
    print_header("ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    
    critical_deps = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("sqlalchemy", "SQLAlchemy"),
        ("redis", "Redis"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("httpx", "HTTPX"),
        ("pydantic", "Pydantic"),
    ]
    
    for module_name, display_name in critical_deps:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'неизвестна')
            print_status(display_name, "ok", f"v{version}")
        except ImportError:
            print_status(display_name, "error", "Не установлен")

def check_project_files():
    """Проверка файлов проекта"""
    print_header("ПРОВЕРКА ФАЙЛОВ ПРОЕКТА")
    
    files_to_check = [
        ("requirements.txt", "Зависимости production"),
        ("requirements-dev.txt", "Зависимости разработки"),
        ("pyproject.toml", "Конфигурация проекта"),
        ("app/main.py", "Главный файл приложения"),
        (".env", "Переменные окружения"),
        (".env.example", "Пример переменных"),
    ]
    
    for filename, description in files_to_check:
        path = Path(filename)
        if path.exists():
            print_status(description, "ok", filename)
        else:
            print_status(description, "warning", f"{filename} не найден")

def check_system_info():
    """Информация о системе"""
    print_header("ИНФОРМАЦИЯ О СИСТЕМЕ")
    
    import platform
    print(f"Операционная система: {platform.system()} {platform.release()}")
    print(f"Архитектура: {platform.machine()}")
    print(f"Python реализация: {platform.python_implementation()}")
    
    # Проверка переменных окружения
    important_env_vars = [
        "VIRTUAL_ENV",
        "PATH", 
        "PYTHONPATH",
        "ENVIRONMENT",
        "DEBUG"
    ]
    
    print("\nПеременные окружения:")
    for var in important_env_vars:
        value = os.environ.get(var, "Не установлена")
        if var == "PATH":
            print_status(f"{var}", "info", "Установлена")
        else:
            print_status(f"{var}", "info", value)

def check_disk_space():
    """Проверка места на диске"""
    print_header("ПРОВЕРКА МЕСТА НА ДИСКЕ")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        total_gb = total // (1024**3)
        
        if free_gb > 1:
            print_status("Свободное место", "ok", f"{free_gb}GB из {total_gb}GB")
        else:
            print_status("Свободное место", "warning", f"Мало места: {free_gb}GB")
    except Exception as e:
        print_status("Свободное место", "error", f"Ошибка: {e}")

def suggest_fixes():
    """Предложения по исправлению"""
    print_header("ПРЕДЛОЖЕНИЯ ПО ИСПРАВЛЕНИЮ")
    
    suggestions = [
        "Если Python версия < 3.11: обновите Python",
        "Если нет виртуальной среды: запустите setup_venv.sh или setup_venv.bat",
        "Если отсутствуют зависимости: выполните make install или make setup",
        "Если ошибки OpenCV: установите системные зависимости",
        "Если проблемы с правами: не используйте sudo для venv",
        "Для Docker: используйте make docker-up",
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

def main():
    """Основная функция диагностики"""
    print("🔍 ДИАГНОСТИКА ВИРТУАЛЬНОЙ СРЕДЫ")
    print("Face Recognition Service")
    
    check_python_version()
    check_virtual_env()
    check_pip()
    check_dependencies()
    check_project_files()
    check_system_info()
    check_disk_space()
    suggest_fixes()
    
    print_header("ЗАВЕРШЕНИЕ ДИАГНОСТИКИ")
    print("Диагностика завершена. Проверьте результаты выше.")
    print("\nДля получения помощи:")
    print("- README.md - общая документация")
    print("- VENV_SETUP.md - подробное руководство")
    print("- QUICK_START_VENV.md - быстрый старт")
    print("- make help - доступные команды")

if __name__ == "__main__":
    main()