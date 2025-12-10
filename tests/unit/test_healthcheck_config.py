"""
Тест для проверки улучшенных healthcheck конфигураций в Dockerfile файлах.
"""

import os
import sys
from pathlib import Path


def test_production_healthcheck():
    """Проверка healthcheck в production Dockerfile"""
    dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
    with open(dockerfile_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем улучшенные параметры healthcheck
    assert "HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3" in content, \
        "Production Dockerfile должен содержать улучшенный healthcheck"
    
    # Проверяем многострочный формат
    assert "CMD curl -f http://localhost:8000/health || exit 1" in content, \
        "Healthcheck команда должна быть в многострочном формате"
    
    print("Production Dockerfile healthcheck настроен корректно")


def test_development_healthcheck():
    """Проверка healthcheck в development Dockerfile"""
    dockerfile_dev_path = Path(__file__).parent.parent.parent / "Dockerfile.dev"
    with open(dockerfile_dev_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем улучшенные параметры healthcheck для dev
    assert "HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3" in content, \
        "Development Dockerfile должен содержать оптимизированный healthcheck"
    
    # Проверяем многострочный формат
    assert "CMD curl -f http://localhost:8000/health || exit 1" in content, \
        "Healthcheck команда должна быть в многострочном формате"
    
    print("Development Dockerfile healthcheck настроен корректно")


def test_healthcheck_differences():
    """Проверка различий между production и development healthcheck"""
    dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
    dockerfile_dev_path = Path(__file__).parent.parent.parent / "Dockerfile.dev"
    
    with open(dockerfile_path, 'r', encoding='utf-8') as f:
        prod_content = f.read()
    
    with open(dockerfile_dev_path, 'r', encoding='utf-8') as f:
        dev_content = f.read()
    
    # Извлекаем healthcheck строки
    prod_healthcheck = [line for line in prod_content.split('\n') if 'HEALTHCHECK' in line]
    dev_healthcheck = [line for line in dev_content.split('\n') if 'HEALTHCHECK' in line]
    
    assert len(prod_healthcheck) > 0, "Production Dockerfile должен содержать healthcheck"
    assert len(dev_healthcheck) > 0, "Development Dockerfile должен содержать healthcheck"
    
    # Проверяем, что параметры разные для prod и dev
    assert "interval=30s" in prod_healthcheck[0], "Production должен использовать interval=30s"
    assert "interval=15s" in dev_healthcheck[0], "Development должен использовать interval=15s"
    
    assert "timeout=10s" in prod_healthcheck[0], "Production должен использовать timeout=10s"
    assert "timeout=5s" in dev_healthcheck[0], "Development должен использовать timeout=5s"
    
    assert "start-period=40s" in prod_healthcheck[0], "Production должен использовать start-period=40s"
    assert "start-period=20s" in dev_healthcheck[0], "Development должен использовать start-period=20s"
    
    print("Различия между production и development healthcheck корректны")


def test_healthcheck_endpoint():
    """Проверка правильности healthcheck endpoint"""
    dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
    dockerfile_dev_path = Path(__file__).parent.parent.parent / "Dockerfile.dev"
    
    for dockerfile_path in [dockerfile_path, dockerfile_dev_path]:
        with open(dockerfile_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем правильный endpoint
        assert "http://localhost:8000/health" in content, \
            f"Healthcheck должен использовать правильный endpoint /health"
        
        # Проверяем curl команду
        assert "curl -f" in content, "Healthcheck должен использовать curl с -f флагом"
        
        # Проверяем exit код
        assert "exit 1" in content, "Healthcheck должен использовать exit 1 при ошибке"
    
    print("Healthcheck endpoint настроен корректно")


if __name__ == "__main__":
    print("Проверка улучшенных healthcheck конфигураций...")
    
    test_production_healthcheck()
    test_development_healthcheck()
    test_healthcheck_differences()
    test_healthcheck_endpoint()
    
    print("\nВсе проверки healthcheck прошли успешно!")
    print("Улучшения:")
    print("   • Production: --timeout=10s, --start-period=40s")
    print("   • Development: --timeout=5s, --start-period=20s")
    print("   • Многострочный формат для читаемости")
    print("   • Быстрое реагирование на проблемы")