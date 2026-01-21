# scripts/fix_alembic_heads.py

"""
Скрипт для анализа и починки множественных heads в Alembic
"""

import subprocess
import re

def get_heads():
    """Получить все heads"""
    result = subprocess.run(
        ['alembic', 'heads'],
        capture_output=True,
        text=True
    )
    
    # Парсим вывод
    heads = []
    for line in result.stdout.split('\n'):
        if line.strip():
            # Извлекаем ID (первая часть строки)
            match = re.match(r'^([a-f0-9]+)', line.strip())
            if match:
                heads.append(match.group(1))
    
    return heads

def create_merge_migration(heads):
    """Создать merge миграцию"""
    print(f"Found {len(heads)} heads: {heads}")
    
    if len(heads) <= 1:
        print("✅ Only one head, no merge needed!")
        return
    
    print(f"Creating merge migration for heads: {', '.join(heads)}")
    
    # Создаём merge миграцию
    subprocess.run([
        'alembic', 'merge', 'heads', '-m', 'merge_multiple_heads'
    ])
    
    print("✅ Merge migration created!")
    print("Now run: alembic upgrade head")

if __name__ == "__main__":
    heads = get_heads()
    create_merge_migration(heads)
