"""
Тест для проверки исправленных команд запуска в Dockerfile файлах.
"""

import os
import sys
from pathlib import Path


def test_dockerfile_production_command():
    """Проверка команды запуска в production Dockerfile"""
    dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
    with open(dockerfile_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем правильную команду запуска
    assert 'CMD ["uvicorn", "app.main:app"' in content, "Production Dockerfile должен использовать app.main:app"
    
    # Проверяем, что старая команда отсутствует
    assert 'app.main:create_app"' not in content, "Старая команда create_app не должна присутствовать"
    
    print("Production Dockerfile команда исправлена корректно")


def test_dockerfile_dev_command():
    """Проверка команды запуска в development Dockerfile"""
    dockerfile_dev_path = Path(__file__).parent.parent.parent / "Dockerfile.dev"
    with open(dockerfile_dev_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем правильную команду запуска
    assert 'CMD ["uvicorn", "app.main:app"' in content, "Development Dockerfile должен использовать app.main:app"
    
    # Проверяем наличие hot-reload флага
    assert '"--reload"' in content, "Development Dockerfile должен содержать --reload флаг"
    
    # Проверяем, что старая команда отсутствует
    assert 'app.main:create_app"' not in content, "Старая команда create_app не должна присутствовать"
    
    print("Development Dockerfile команда исправлена корректно")


def test_main_py_structure():
    """Проверка структуры app/main.py"""
    main_py_path = Path(__file__).parent.parent.parent / "app" / "main.py"
    with open(main_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие create_app функции
    assert "def create_app() -> FastAPI:" in content, "create_app функция должна существовать"
    
    # Проверяем создание экземпляра приложения
    assert "app = create_app()" in content, "Должен создаваться экземпляр приложения с именем app"
    
    # Проверяем пример запуска в __main__ (более гибкий поиск)
    assert '"app.main:app"' in content, "В файле должен быть пример запуска с app.main:app"
    
    print("Структура app/main.py корректна")


if __name__ == "__main__":
    print("Проверка исправленных команд запуска...")
    
    test_dockerfile_production_command()
    test_dockerfile_dev_command()
    test_main_py_structure()
    
    print("\nВсе проверки прошли успешно!")
    print("Исправления:")
    print("   • Dockerfile (production): app.main:app")
    print("   • Dockerfile.dev (development): app.main:app --reload")
    print("   • Структура app/main.py: корректна")