# scripts/validate_index_performance.py

"""
Скрипт для валидации улучшения производительности после добавления индексов
"""

import time
from sqlalchemy import create_engine, text
from app.config import settings

def measure_query_performance():
    """Измерение производительности ключевых запросов"""
    
    engine = create_engine(settings.DATABASE_URL)
    
    queries = {
        "User lookup by email": """
            EXPLAIN ANALYZE
            SELECT * FROM users WHERE email = 'test@example.com';
        """,
        
        "Latest reference for user": """
            EXPLAIN ANALYZE
            SELECT * FROM references 
            WHERE user_id = (SELECT id FROM users LIMIT 1)
            ORDER BY version DESC 
            LIMIT 1;
        """,
        
        "Active references for user": """
            EXPLAIN ANALYZE
            SELECT * FROM references 
            WHERE user_id = (SELECT id FROM users LIMIT 1)
              AND is_active = true
            ORDER BY version DESC;
        """,
        
        "User verification history": """
            EXPLAIN ANALYZE
            SELECT * FROM verification_sessions 
            WHERE user_id = (SELECT id FROM users LIMIT 1)
            ORDER BY created_at DESC 
            LIMIT 10;
        """,
        
        "Successful verifications": """
            EXPLAIN ANALYZE
            SELECT * FROM verification_sessions 
            WHERE user_id = (SELECT id FROM users LIMIT 1)
              AND is_match = true
            ORDER BY created_at DESC;
        """,
        
        "User audit logs": """
            EXPLAIN ANALYZE
            SELECT * FROM audit_logs 
            WHERE user_id = (SELECT id FROM users LIMIT 1)
            ORDER BY created_at DESC 
            LIMIT 20;
        """
    }
    
    with engine.connect() as conn:
        print("=" * 80)
        print("QUERY PERFORMANCE VALIDATION")
        print("=" * 80)
        
        for query_name, query_sql in queries.items():
            print(f"\n🔍 {query_name}")
            print("-" * 80)
            
            start = time.time()
            result = conn.execute(text(query_sql))
            execution_time = (time.time() - start) * 1000
            
            # Parse EXPLAIN ANALYZE output
            for row in result:
                line = str(row[0])
                if "Index Scan" in line or "Seq Scan" in line:
                    print(f"  {line}")
                if "Execution Time" in line or "Planning Time" in line:
                    print(f"  ✓ {line}")
            
            print(f"\n  ⏱️  Total execution: {execution_time:.2f}ms")
            
            # Performance target validation
            target = 100  # ms
            if execution_time < target:
                print(f"  ✅ PASS (< {target}ms)")
            else:
                print(f"  ⚠️  SLOW (> {target}ms)")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    measure_query_performance()
