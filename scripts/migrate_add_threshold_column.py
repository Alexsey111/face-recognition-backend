"""
Database migration script: Add missing columns to verification_sessions table.
Run this script to add columns that are missing from the database schema.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.config import settings
from app.db.database import DatabaseManager


async def migrate():
    """Add missing columns to verification_sessions table."""
    
    db_manager = DatabaseManager(settings.DATABASE_URL)
    db_manager._ensure_engine()  # Инициализировать engine
    async with db_manager.engine.connect() as conn:
        # Check if threshold_used column exists
        result = await conn.execute(
            text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'verification_sessions' 
                AND column_name = 'threshold_used'
            """)
        )
        row = result.fetchone()
        
        if row:
            print("✓ Column 'threshold_used' already exists")
        else:
            # Add threshold_used column
            await conn.execute(text("""
                ALTER TABLE verification_sessions 
                ADD COLUMN threshold_used FLOAT
            """))
            await conn.commit()
            print("✓ Added 'threshold_used' column")
        
        # Check other potentially missing columns
        columns_to_check = [
            ('confidence', 'FLOAT'),
            ('face_detected', 'BOOLEAN DEFAULT FALSE'),
            ('face_quality_score', 'FLOAT'),
            ('is_liveness_passed', 'BOOLEAN'),
            ('liveness_score', 'FLOAT'),
            ('liveness_method', 'VARCHAR(50)'),
            ('processing_time_ms', 'INTEGER'),
        ]
        
        for col_name, col_type in columns_to_check:
            result = await conn.execute(
                text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'verification_sessions' 
                    AND column_name = '{col_name}'
                """)
            )
            row = result.fetchone()
            
            if row:
                print(f"✓ Column '{col_name}' already exists")
            else:
                await conn.execute(text(f"""
                    ALTER TABLE verification_sessions 
                    ADD COLUMN {col_name} {col_type}
                """))
                await conn.commit()
                print(f"✓ Added '{col_name}' column")
        
        print("\n✓ Migration completed successfully!")


if __name__ == "__main__":
    asyncio.run(migrate())