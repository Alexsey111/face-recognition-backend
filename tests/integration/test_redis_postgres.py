"""
Integration tests for Redis and PostgreSQL.
Tests real database connections, caching, and data persistence.
"""
import pytest
import pytest_asyncio
import asyncio
import uuid
import time

import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker


class TestPostgreSQLIntegration:
    """Integration tests for PostgreSQL."""
    
    @pytest.fixture
    def sync_engine(self):
        """Create synchronous SQLAlchemy engine."""
        from app.config import settings
        
        url = getattr(settings, 'sync_database_url', None) or settings.DATABASE_URL
        
        if url.startswith('postgresql+asyncpg://'):
            url = url.replace('postgresql+asyncpg://', 'postgresql://', 1)
        elif url.startswith('sqlite+aiosqlite://'):
            url = url.replace('sqlite+aiosqlite://', 'sqlite:///', 1)
        
        engine = create_engine(url, echo=False)
        yield engine
        engine.dispose()
    
    @pytest.fixture
    def async_engine(self):
        """Create async SQLAlchemy engine."""
        from app.config import settings
        
        url = getattr(settings, 'async_database_url', None) or settings.DATABASE_URL
        
        if url.startswith('postgresql://'):
            url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        elif url.startswith('sqlite://'):
            url = url.replace('sqlite://', 'sqlite+aiosqlite://', 1)
        
        engine = create_async_engine(url, echo=False)
        yield engine
        asyncio.get_event_loop().run_until_complete(engine.dispose())
    
    def test_postgresql_connection(self, sync_engine):
        """Test basic PostgreSQL connection."""
        with sync_engine.connect() as conn:
            result = conn.execute(text('SELECT 1'))
            row = result.fetchone()
            assert row[0] == 1
    
    def test_postgresql_tables_exist(self, sync_engine):
        """Test that required tables exist."""
        from app.db.models import Base
        
        table_names = set(Base.metadata.tables.keys())
        required_tables = {'users', 'references'}
        assert required_tables.issubset(table_names)
    
    def test_insert_and_query_user(self, sync_engine):
        """Test inserting and querying a user."""
        test_user_id = f'test-user-{uuid.uuid4().hex[:8]}'
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, 1, 0, 0, 0)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {'id': test_user_id, 'email': f'{test_user_id}@example.com'}
            )
            
            result = conn.execute(
                text('SELECT id, email, is_active FROM users WHERE id = :id'),
                {'id': test_user_id}
            )
            row = result.fetchone()
            
            assert row is not None
            assert row[0] == test_user_id
            assert row[2] is True
    
    def test_insert_and_query_reference(self, sync_engine):
        """Test inserting and querying a reference."""
        test_user_id = f'test-user-{uuid.uuid4().hex[:8]}'
        test_ref_id = f'test-ref-{uuid.uuid4().hex[:8]}'
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, 1, 0, 0, 0)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {'id': test_user_id, 'email': f'{test_user_id}@example.com'}
            )
            
            conn.execute(
                text('''
                    INSERT INTO references (id, user_id, file_url, embedding, embedding_encrypted, embedding_hash, 
                                          image_filename, image_size_mb, usage_count)
                    VALUES (:id, :user_id, :file_url, :embedding, :embedding_encrypted, :embedding_hash,
                            :image_filename, :image_size_mb, :usage_count)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {
                    'id': test_ref_id,
                    'user_id': test_user_id,
                    'file_url': 'http://example.local/test.jpg',
                    'embedding': None,
                    'embedding_encrypted': b'\x00' * 128,
                    'embedding_hash': f'hash-{uuid.uuid4().hex[:16]}',
                    'image_filename': 'test.jpg',
                    'image_size_mb': 0.1,
                    'usage_count': 0,
                }
            )
            
            result = conn.execute(
                text('SELECT id, user_id, usage_count FROM references WHERE id = :id'),
                {'id': test_ref_id}
            )
            row = result.fetchone()
            
            assert row is not None
            assert row[0] == test_ref_id
            assert row[1] == test_user_id
            assert row[2] == 0
    
    def test_reference_cascade_delete(self, sync_engine):
        """Test that deleting user cascades to references."""
        test_user_id = f'test-user-{uuid.uuid4().hex[:8]}'
        test_ref_id = f'test-ref-{uuid.uuid4().hex[:8]}'
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, 1, 0, 0, 0)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {'id': test_user_id, 'email': f'{test_user_id}@example.com'}
            )
            
            conn.execute(
                text('''
                    INSERT INTO references (id, user_id, file_url, embedding, embedding_encrypted, embedding_hash, 
                                          image_filename, image_size_mb, usage_count)
                    VALUES (:id, :user_id, :file_url, :embedding, :embedding_encrypted, :embedding_hash,
                            :image_filename, :image_size_mb, :usage_count)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {
                    'id': test_ref_id,
                    'user_id': test_user_id,
                    'file_url': 'http://example.local/test.jpg',
                    'embedding': None,
                    'embedding_encrypted': b'\x00' * 128,
                    'embedding_hash': f'hash-{uuid.uuid4().hex[:16]}',
                    'image_filename': 'test.jpg',
                    'image_size_mb': 0.1,
                    'usage_count': 0,
                }
            )
            
            conn.execute(text('DELETE FROM users WHERE id = :id'), {'id': test_user_id})
            
            result = conn.execute(
                text('SELECT COUNT(*) FROM references WHERE id = :id'),
                {'id': test_ref_id}
            )
            count = result.fetchone()[0]
            assert count == 0


class TestRedisIntegration:
    """Integration tests for Redis."""
    
    @pytest_asyncio.fixture
    async def redis_client(self):
        """Create async Redis client."""
        from app.config import settings
        
        redis_url = getattr(settings, 'redis_url_with_auth', None) or settings.REDIS_URL
        
        if redis_url.startswith('redis://') and settings.REDIS_PASSWORD:
            redis_url = redis_url.replace('redis://', f'redis://:{settings.REDIS_PASSWORD}@', 1)
        
        client = redis.from_url(redis_url, decode_responses=True)
        await client.ping()
        
        yield client
        
        await client.acl_safe(None)
        await client.close()
    
    @pytest.mark.asyncio
    async def test_redis_connection(self, redis_client):
        """Test basic Redis connection."""
        result = await redis_client.ping()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, redis_client):
        """Test basic set/get operations."""
        key = f'test:key:{uuid.uuid4().hex[:8]}'
        value = 'test_value'
        
        await redis_client.set(key, value)
        result = await redis_client.get(key)
        assert result == value
        
        await redis_client.delete(key)
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, redis_client):
        """Test TTL/expiration functionality."""
        key = f'test:ttl:{uuid.uuid4().hex[:8]}'
        value = 'expired_value'
        
        await redis_client.setex(key, 2, value)
        result = await redis_client.get(key)
        assert result == value
        
        await asyncio.sleep(2.5)
        result = await redis_client.get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_hash_operations(self, redis_client):
        """Test Redis hash operations."""
        key = f'test:hash:{uuid.uuid4().hex[:8]}'
        
        await redis_client.hset(key, 'field1', 'value1')
        await redis_client.hset(key, 'field2', 'value2')
        
        result = await redis_client.hget(key, 'field1')
        assert result == 'value1'
        
        all_fields = await redis_client.hgetall(key)
        assert all_fields == {'field1': 'value1', 'field2': 'value2'}
        
        await redis_client.delete(key)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, redis_client):
        """Test concurrent Redis operations."""
        key = f'test:concurrent:{uuid.uuid4().hex[:8]}'
        
        async def increment():
            return await redis_client.incr(key)
        
        num_increments = 100
        tasks = [increment() for _ in range(num_increments)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == num_increments
        
        final_value = await redis_client.get(key)
        assert int(final_value) == num_increments
        
        await redis_client.delete(key)


class TestCacheIntegration:
    """Integration tests for application caching layer."""
    
    @pytest_asyncio.fixture
    async def cache(self, redis_client):
        """Create cache service instance."""
        from app.services.cache_service import CacheService
        service = CacheService()
        service._redis = redis_client
        return service
    
    @pytest.mark.asyncio
    async def test_cache_embedding(self, cache):
        """Test caching face embeddings."""
        embedding_id = f'embed:{uuid.uuid4().hex[:8]}'
        embedding = [0.1] * 512
        
        await cache.cache_embedding(embedding_id, embedding, ttl=60)
        cached = await cache.get_embedding(embedding_id)
        
        assert cached is not None
        assert len(cached) == 512
        
        await cache.delete_embedding(embedding_id)
    
    @pytest.mark.asyncio
    async def test_cache_verification_session(self, cache):
        """Test caching verification sessions."""
        session_id = f'session:{uuid.uuid4().hex[:8]}'
        session_data = {
            'user_id': 'user-123',
            'status': 'pending',
            'created_at': time.time(),
        }
        
        await cache.cache_verification_session(session_id, session_data, ttl=300)
        cached = await cache.get_verification_session(session_id)
        
        assert cached is not None
        assert cached['user_id'] == 'user-123'
        
        await cache.delete_verification_session(session_id)
    
    @pytest.mark.asyncio
    async def test_cache_rate_limit(self, cache):
        """Test rate limiting functionality."""
        user_id = f'user:{uuid.uuid4().hex[:8]}'
        
        allowed, remaining = await cache.check_rate_limit(user_id, max_requests=10, window=60)
        assert allowed is True
        assert remaining == 9
        
        for i in range(8):
            allowed, remaining = await cache.check_rate_limit(user_id, max_requests=10, window=60)
            assert allowed is True
        
        allowed, remaining = await cache.check_rate_limit(user_id, max_requests=10, window=60)
        assert allowed is False
        assert remaining == 0


class TestDatabaseCacheIntegration:
    """Integration tests for combined database and cache operations."""
    
    @pytest.mark.asyncio
    async def test_cache_aside_pattern(self, sync_engine, redis_client):
        """Test cache-aside pattern implementation."""
        cache_key = f'user:profile:{uuid.uuid4().hex[:8]}'
        user_id = f'user-{uuid.uuid4().hex[:8]}'
        
        cached_user = await redis_client.get(cache_key)
        assert cached_user is None
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, 1, 0, 0, 0)
                    ON CONFLICT (id) DO NOTHING
                '''),
                {'id': user_id, 'email': f'{user_id}@example.com'}
            )
        
        user_data = {'id': user_id, 'email': f'{user_id}@example.com'}
        await redis_client.setex(cache_key, 300, str(user_data))
        
        cached_user = await redis_client.get(cache_key)
        assert cached_user is not None
        
        await redis_client.delete(cache_key)
        with sync_engine.begin() as conn:
            conn.execute(text('DELETE FROM users WHERE id = :id'), {'id': user_id})
    
    @pytest.mark.asyncio
    async def test_write_through_pattern(self, sync_engine, redis_client):
        """Test write-through caching pattern."""
        user_id = f'user-{uuid.uuid4().hex[:8]}'
        cache_key = f'user:profile:{user_id}'
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, 1, 0, 0, 0)
                '''),
                {'id': user_id, 'email': f'{user_id}@example.com'}
            )
        
        user_data = {'id': user_id, 'email': f'{user_id}@example.com'}
        await redis_client.setex(cache_key, 300, str(user_data))
        
        with sync_engine.begin() as conn:
            result = conn.execute(
                text('SELECT email FROM users WHERE id = :id'),
                {'id': user_id}
            )
            row = result.fetchone()
            assert row[0] == f'{user_id}@example.com'
        
        cached = await redis_client.get(cache_key)
        assert cached is not None
        
        await redis_client.delete(cache_key)
        with sync_engine.begin() as conn:
            conn.execute(text('DELETE FROM users WHERE id = :id'), {'id': user_id})


class TestTransactionIsolation:
    """Test database transaction isolation."""
    
    @pytest.mark.asyncio
    async def test_concurrent_transactions(self, sync_engine):
        """Test concurrent transaction handling."""
        user_id = f'user-{uuid.uuid4().hex[:8]}'
        
        async def update_user():
            import threading
            
            def inner_update():
                with sync_engine.begin() as conn:
                    conn.execute(
                        text('UPDATE users SET total_verifications = total_verifications + 1 WHERE id = :id'),
                        {'id': user_id}
                    )
            
            thread = threading.Thread(target=inner_update)
            thread.start()
            thread.join()
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, 1, 0, 0, 0)
                '''),
                {'id': user_id, 'email': f'{user_id}@example.com'}
            )
        
        tasks = [update_user() for i in range(10)]
        await asyncio.gather(*tasks)
        
        with sync_engine.begin() as conn:
            result = conn.execute(
                text('SELECT total_verifications FROM users WHERE id = :id'),
                {'id': user_id}
            )
            row = result.fetchone()
            assert row[0] == 10
        
        with sync_engine.begin() as conn:
            conn.execute(text('DELETE FROM users WHERE id = :id'), {'id': user_id})


class TestPerformanceIntegration:
    """Performance-related integration tests."""
    
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, sync_engine):
        """Test bulk insert performance."""
        user_prefix = f'perf-user-{uuid.uuid4().hex[:8]}'
        num_users = 100
        
        start_time = time.time()
        
        with sync_engine.begin() as conn:
            values = [
                {'id': f'{user_prefix}-{i}', 'email': f'user{i}@{user_prefix}.com'}
                for i in range(num_users)
            ]
            
            conn.execute(
                text('''
                    INSERT INTO users (id, email, is_active, total_uploads, total_verifications, successful_verifications)
                    VALUES (:id, :email, 1, 0, 0, 0)
                '''),
                values
            )
        
        elapsed = time.time() - start_time
        print(f'Bulk inserted {num_users} users in {elapsed:.3f}s')
        
        with sync_engine.begin() as conn:
            result = conn.execute(
                text('SELECT COUNT(*) FROM users WHERE id LIKE :prefix'),
                {'prefix': f'{user_prefix}%'}
            )
            count = result.fetchone()[0]
            assert count == num_users
        
        with sync_engine.begin() as conn:
            conn.execute(
                text('DELETE FROM users WHERE id LIKE :prefix'),
                {'prefix': f'{user_prefix}%'}
            )
    
    @pytest.mark.asyncio
    async def test_redis_pipeline_performance(self, redis_client):
        """Test Redis pipeline performance."""
        key_prefix = f'perf:key:{uuid.uuid4().hex[:8]}'
        num_operations = 1000
        
        start_time = time.time()
        
        async with redis_client.pipeline(transaction=False) as pipe:
            for i in range(num_operations):
                pipe.set(f'{key_prefix}:{i}', f'value_{i}')
            await pipe.execute()
        
        elapsed = time.time() - start_time
        print(f'Redis pipeline: {num_operations} operations in {elapsed:.3f}s')
        
        for i in range(min(10, num_operations)):
            value = await redis_client.get(f'{key_prefix}:{i}')
            assert value == f'value_{i}'
        
        keys = [f'{key_prefix}:{i}' for i in range(num_operations)]
        await redis_client.delete(*keys)
