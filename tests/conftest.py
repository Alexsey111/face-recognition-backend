import importlib
import pytest
import asyncio
from sqlalchemy import text
from sqlalchemy import create_engine


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def seed_test_reference():
    """Ensure tables exist and seed a test user + reference before each test.

    Uses the synchronous DB URL so it works during TestClient (sync) runs.
    This is a best-effort fixture for tests which expect specific IDs.
    """
    # Local import to avoid early app initialization side-effects
    from app.config import settings
    from app.db.models import Base

    try:
        # Ensure models are imported and metadata is populated
        importlib.import_module("app.db.models")
    except Exception:
        pass

    sync_url = getattr(settings, "sync_database_url", None) or settings.DATABASE_URL
    # For sqlite async URLs like sqlite+aiosqlite:///, convert to sync form
    if sync_url.startswith("sqlite+aiosqlite://"):
        sync_url = sync_url.replace("sqlite+aiosqlite://", "sqlite://", 1)

    # Only run this seeding logic for sqlite (local file or in-memory).
    # Avoid importing DB drivers like psycopg2 in CI/unit environments.
    if not sync_url.startswith("sqlite://"):
        return

    engine = create_engine(sync_url)

    with engine.begin() as conn:
        # Create tables if missing
        Base.metadata.create_all(bind=engine)

        # Remove any previous test rows to keep fixture idempotent
        conn.execute(text('DELETE FROM "references" WHERE id = :id'), {"id": "test-reference-123"})
        conn.execute(text('DELETE FROM users WHERE id = :id'), {"id": "test-user-123"})

        # Insert minimal user required by Reference FK
        conn.execute(
            text(
                "INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications) "
                "VALUES (:id, :email, 1, 0, 0, 0)"
            ),
            {"id": "test-user-123", "email": "test-user@example.com"},
        )

        # Insert minimal reference record with required NOT NULL fields
        conn.execute(
            text(
                'INSERT INTO "references" (id, user_id, file_url, embedding, embedding_encrypted, embedding_hash, '
                'image_filename, image_size_mb, usage_count) VALUES (:id, :user_id, :file_url, :embedding, :embedding_encrypted, '
                ':embedding_hash, :image_filename, :image_size_mb, :usage_count)'
            ),
            {
                "id": "test-reference-123",
                "user_id": "test-user-123",
                "file_url": "http://example.local/test.jpg",
                "embedding": "",
                "embedding_encrypted": b"",  # empty blob
                "embedding_hash": "seed-hash-123",
                "image_filename": "test.jpg",
                "image_size_mb": 0.1,
                "usage_count": 0,
            },
        )

    engine.dispose()
