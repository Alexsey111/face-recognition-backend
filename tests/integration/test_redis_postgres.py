"""Integration tests for Redis and PostgreSQL."""

import uuid
import pytest
import pytest_asyncio
from sqlalchemy import text

from app.services.cache_service import CacheService


class TestPostgreSQLIntegration:
    """PostgreSQL integration tests."""
    
    def test_postgresql_connection(self, sync_engine):
        """Test PostgreSQL connection."""
        with sync_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            assert row[0] == 1
    
    def test_postgresql_tables_exist(self, sync_engine):
        """Test that required tables exist."""
        with sync_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]
            
            required_tables = ['users', 'references', 'verification_sessions', 'audit_logs']
            for table in required_tables:
                assert table in tables, f"Table '{table}' not found"
    
    def test_insert_and_query_user(self, sync_engine):
        """Test inserting and querying a user."""
        test_user_id = f'test-user-{uuid.uuid4().hex[:8]}'
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO users (id, email, password_hash, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, :password_hash, TRUE, 0, 0, 0)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {
                    'id': test_user_id, 
                    'email': f'{test_user_id}@example.com',
                    'password_hash': 'test_hash_placeholder'
                }
            )
            
            result = conn.execute(
                text('SELECT id, email, password_hash, is_active FROM users WHERE id = :id'),
                {'id': test_user_id}
            )
            user = result.fetchone()
            
            assert user is not None
            assert user[0] == test_user_id
            assert user[1] == f'{test_user_id}@example.com'
            assert user[2] == 'test_hash_placeholder'  # ✅ Проверяем password_hash
            assert user[3] is True  # TRUE instead of 1
    
    @pytest.mark.asyncio
    async def test_insert_reference(self, async_engine, test_user):
        """Test inserting a reference image."""
        ref_id = f'ref-{uuid.uuid4().hex[:8]}'
        
        async with async_engine.begin() as conn:
            # "references" is a reserved word in PostgreSQL
            await conn.execute(
                text('''
                    INSERT INTO "references" (id, user_id, file_url, embedding_encrypted, embedding_hash, quality_score, image_filename, image_size_mb, image_format)
                    VALUES (:id, :user_id, :file_url, :embedding_encrypted, :embedding_hash, :quality_score, :image_filename, :image_size_mb, :image_format)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {
                    'id': ref_id,
                    'user_id': test_user,
                    'file_url': f'http://example.local/{ref_id}.jpg',
                    'embedding_encrypted': b'test_encrypted_data',
                    'embedding_hash': f'hash_{uuid.uuid4().hex[:16]}',
                    'quality_score': 0.85,
                    'image_filename': 'test.jpg',
                    'image_size_mb': 1.5,
                    'image_format': 'JPG'
                }
            )
        
            result = await conn.execute(
                text('SELECT id, user_id, image_filename, quality_score, image_format FROM "references" WHERE id = :id'),
                {'id': ref_id}
            )
            ref = result.fetchone()
            
            assert ref is not None
            assert ref[0] == ref_id
            assert ref[1] == test_user
            assert ref[2] == 'test.jpg'
            assert ref[3] == 0.85  # ✅ Проверяем quality_score
            assert ref[4] == 'JPG'  # ✅ Проверяем image_format
    
    @pytest.mark.asyncio
    async def test_reference_cascade_delete(self, async_engine, test_user):
        """Test that references are deleted when user is deleted."""
        ref_id = f'ref-{uuid.uuid4().hex[:8]}'
        
        async with async_engine.begin() as conn:
            await conn.execute(
                text('''
                    INSERT INTO "references" (id, user_id, file_url, embedding_encrypted, embedding_hash, quality_score, image_filename, image_size_mb, image_format)
                    VALUES (:id, :user_id, :file_url, :embedding_encrypted, :embedding_hash, :quality_score, :image_filename, :image_size_mb, :image_format)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {
                    'id': ref_id,
                    'user_id': test_user,
                    'file_url': f'http://example.local/{ref_id}.jpg',
                    'embedding_encrypted': b'test_encrypted_data',
                    'embedding_hash': f'hash_{uuid.uuid4().hex[:16]}',
                    'quality_score': 0.85,
                    'image_filename': 'test.jpg',
                    'image_size_mb': 1.5,
                    'image_format': 'JPG'
                }
            )
        
            await conn.execute(text('DELETE FROM users WHERE id = :id'), {'id': test_user})
            
            result = await conn.execute(
                text('SELECT id FROM "references" WHERE user_id = :user_id'),
                {'user_id': test_user}
            )
            refs = result.fetchall()
            assert len(refs) == 0


class TestRedisIntegration:
    """Redis integration tests (async)."""
    
    @pytest.mark.asyncio
    async def test_redis_connection(self, redis_client):
        """Test Redis connection."""
        result = await redis_client.ping()
        assert result is True
        
    @pytest.mark.asyncio
    async def test_set_and_get(self, redis_client):
        """Test basic set/get operations."""
        key = f"test:key:{uuid.uuid4().hex[:8]}"
        value = "test_value"
        
        await redis_client.set(key, value)
        result = await redis_client.get(key)
        
        assert result == value
        await redis_client.delete(key)
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, redis_client):
        """Test TTL expiration."""
        key = f"test:ttl:{uuid.uuid4().hex[:8]}"
        
        await redis_client.setex(key, 1, "value")
        result = await redis_client.get(key)
        
        assert result == "value"
    
    @pytest.mark.asyncio
    async def test_hash_operations(self, redis_client):
        """Test hash operations."""
        key = f"test:hash:{uuid.uuid4().hex[:8]}"
        
        await redis_client.hset(key, "field1", "value1")
        await redis_client.hset(key, "field2", "value2")
        
        result = await redis_client.hgetall(key)
        
        assert result == {"field1": "value1", "field2": "value2"}
        await redis_client.delete(key)


class TestCacheIntegration:
    """Cache service integration tests."""
    
    @pytest.mark.asyncio
    async def test_cache_embedding(self, cache, test_user):
        """Test embedding caching."""
        image_hash = f"hash_{uuid.uuid4().hex[:16]}"
        embedding = b"embedding_data_test"
        
        result = await cache.cache_embedding(image_hash, embedding, expire_seconds=60)
        assert result is True
        
        cached = await cache.get_cached_embedding(image_hash)
        assert cached == embedding
    
    @pytest.mark.asyncio
    async def test_verification_session_cache(self, cache, test_user):
        """Test verification session caching."""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        session_data = {
            "user_id": test_user,
            "status": "pending",
            "created_at": "2024-01-01T00:00:00",
        }
        
        result = await cache.set_verification_session(session_id, session_data, expire_seconds=300)
        assert result is True
        
        cached = await cache.get_verification_session(session_id)
        assert cached is not None
        assert cached["user_id"] == test_user
        assert cached["status"] == "pending"
        
        await cache.delete_verification_session(session_id)
    
    @pytest.mark.asyncio
    async def test_rate_limit(self, cache, test_user):
        """Test rate limiting."""
        result = await cache.check_rate_limit(
            identifier=test_user,
            limit=10,
            window_seconds=60
        )
        
        assert result["allowed"] is True
        assert result["limit"] == 10
        assert result["current"] >= 1


class TestPerformanceIntegration:
    """Performance-related integration tests."""
    
    @pytest.mark.asyncio
    async def test_redis_pipeline_performance(self, redis_client):
        """Test Redis pipeline performance."""
        key_prefix = f"test:pipe:{uuid.uuid4().hex[:8]}"
        
        pipe = redis_client.pipeline()
        for i in range(10):
            await pipe.set(f"{key_prefix}:{i}", f"value_{i}")
        
        results = await pipe.execute()
        
        # All operations should succeed
        assert all(results)
        
        # Cleanup
        for i in range(10):
            await redis_client.delete(f"{key_prefix}:{i}")