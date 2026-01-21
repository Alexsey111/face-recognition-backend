import asyncio
from app.db.database import db_manager

async def delete_user():
    async with db_manager.get_session() as session:
        from sqlalchemy import text
        await session.execute(text("DELETE FROM users WHERE email = 'test@example.com'"))
        await session.commit()
        print("User deleted")

if __name__ == "__main__":
    asyncio.run(delete_user())