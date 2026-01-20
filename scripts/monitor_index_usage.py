# scripts/monitor_index_usage.py

"""
Мониторинг использования индексов (какие используются, какие нет)
"""

from sqlalchemy import create_engine, text
from app.config import settings

def monitor_index_usage():
    """Проверка, какие индексы реально используются"""
    
    engine = create_engine(settings.DATABASE_URL)
    
    # Статистика использования индексов
    query = text("""
        SELECT
            schemaname,
            tablename,
            indexname,
            idx_scan as index_scans,
            idx_tup_read as tuples_read,
            idx_tup_fetch as tuples_fetched
        FROM
            pg_stat_user_indexes
        WHERE
            schemaname = 'public'
        ORDER BY
            idx_scan DESC;
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        
        print("=" * 100)
        print("INDEX USAGE STATISTICS")
        print("=" * 100)
        print(f"{'Table':<30} {'Index':<40} {'Scans':<10} {'Tuples Read':<15}")
        print("-" * 100)
        
        unused_indexes = []
        
        for row in result:
            scans = row.index_scans
            
            print(f"{row.tablename:<30} {row.indexname:<40} {scans:<10} {row.tuples_read:<15}")
            
            # Флаг неиспользуемых индексов
            if scans == 0:
                unused_indexes.append(row.indexname)
        
        print("\n" + "=" * 100)
        
        if unused_indexes:
            print("\n⚠️  UNUSED INDEXES (consider removing):")
            for idx in unused_indexes:
                print(f"  - {idx}")
        else:
            print("\n✅ All indexes are being used!")

if __name__ == "__main__":
    monitor_index_usage()
