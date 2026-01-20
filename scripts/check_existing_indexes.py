# scripts/check_existing_indexes.py

"""
Скрипт для проверки существующих индексов в БД
Запустить перед применением миграции
"""

import asyncio
from sqlalchemy import create_engine, text
from app.config import settings

def check_existing_indexes():
    """Проверка существующих индексов"""
    
    engine = create_engine(settings.DATABASE_URL)
    
    query = text("""
        SELECT
            tablename,
            indexname,
            indexdef
        FROM
            pg_indexes
        WHERE
            schemaname = 'public'
        ORDER BY
            tablename,
            indexname;
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        
        print("=" * 80)
        print("EXISTING INDEXES")
        print("=" * 80)
        
        current_table = None
        for row in result:
            if row.tablename != current_table:
                current_table = row.tablename
                print(f"\n📁 Table: {current_table}")
                print("-" * 80)
            
            print(f"  ✓ {row.indexname}")
            print(f"    {row.indexdef}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    check_existing_indexes()
