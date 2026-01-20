# scripts/check_index_size.py

"""
Проверка размера индексов (важно для production)
"""

from sqlalchemy import create_engine, text
from app.config import settings

def check_index_sizes():
    """Проверка размера индексов"""
    
    engine = create_engine(settings.DATABASE_URL)
    
    query = text("""
        SELECT
            tablename,
            indexname,
            pg_size_pretty(pg_relation_size(indexrelid)) as index_size
        FROM
            pg_stat_user_indexes
        WHERE
            schemaname = 'public'
        ORDER BY
            pg_relation_size(indexrelid) DESC;
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        
        print("=" * 80)
        print("INDEX SIZES")
        print("=" * 80)
        print(f"{'Table':<30} {'Index':<40} {'Size':<10}")
        print("-" * 80)
        
        total_size = 0
        
        for row in result:
            print(f"{row.tablename:<30} {row.indexname:<40} {row.index_size:<10}")
        
        print("=" * 80)

if __name__ == "__main__":
    check_index_sizes()
