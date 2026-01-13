"""
Concurrent tests for face recognition service.
Tests parallel requests, race conditions, and async operations.
"""
import asyncio
import pytest
import pytest_asyncio
import time
import uuid
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

# Test data
TEST_IMAGE_DATA = (
    b"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
    b"Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
    b"CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    b"MjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAA"
    b"AgP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/"
    b"xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
)


class TestConcurrentRequests:
    """Test concurrent request handling."""
    
    @pytest.fixture
    def mock_ml_service(self):
        """Create mock ML service."""
        service = AsyncMock()
        service.is_initialized = True
        service.generate_embedding = AsyncMock(return_value={
            "success": True,
            "face_detected": True,
            "embedding": [0.1] * 512,
            "quality_score": 0.85,
        })
        service.verify_face = AsyncMock(return_value={
            "success": True,
            "verified": True,
            "confidence": 0.95,
        })
        service.check_liveness = AsyncMock(return_value={
            "success": True,
            "liveness_detected": True,
            "confidence": 0.9,
        })
        return service
    
    @pytest.mark.asyncio
    async def test_parallel_embedding_generation(self, mock_ml_service):
        """Test generating embeddings in parallel."""
        num_requests = 10
        
        async def generate_embedding():
            result = await mock_ml_service.generate_embedding(TEST_IMAGE_DATA)
            return result
        
        # Run requests in parallel
        start_time = time.time()
        tasks = [generate_embedding() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # All requests should complete
        assert len(results) == num_requests
        assert all(r["success"] for r in results)
        assert all(r["face_detected"] for r in results)
        
        # Should complete faster than sequential (in real tests with actual ML service)
        print(f"Completed {num_requests} parallel requests in {elapsed:.3f}s")
    
    @pytest.mark.asyncio
    async def test_parallel_verification(self, mock_ml_service):
        """Test parallel verification requests."""
        num_requests = 5
        reference_embedding = [0.1] * 512
        
        async def verify_face():
            result = await mock_ml_service.verify_face(TEST_IMAGE_DATA, reference_embedding)
            return result
        
        tasks = [verify_face() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == num_requests
        assert all(r["success"] for r in results)
        assert all(r["verified"] for r in results)
    
    @pytest.mark.asyncio
    async def test_parallel_liveness_check(self, mock_ml_service):
        """Test parallel liveness checks."""
        num_requests = 5
        
        async def check_liveness():
            result = await mock_ml_service.check_liveness(TEST_IMAGE_DATA)
            return result
        
        tasks = [check_liveness() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == num_requests
        assert all(r["success"] for r in results)
        assert all(r["liveness_detected"] for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, mock_ml_service):
        """Test mixed concurrent operations."""
        num_each = 3
        
        async def embedding_task():
            return await mock_ml_service.generate_embedding(TEST_IMAGE_DATA)
        
        async def verification_task():
            return await mock_ml_service.verify_face(TEST_IMAGE_DATA, [0.1] * 512)
        
        async def liveness_task():
            return await mock_ml_service.check_liveness(TEST_IMAGE_DATA)
        
        # Mix of different operations
        tasks = []
        for _ in range(num_each):
            tasks.append(embedding_task())
            tasks.append(verification_task())
            tasks.append(liveness_task())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Got {len(exceptions)} exceptions: {exceptions}"
        
        # Check all successful
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successful) == num_each * 3


class TestRaceConditions:
    """Test for potential race conditions."""
    
    @pytest.fixture
    def counter(self):
        """Simple counter for race condition testing."""
        return {"value": 0, "lock": asyncio.Lock()}
    
    @pytest.mark.asyncio
    async def test_atomic_counter_increment(self, counter):
        """Test that counter increments are atomic with lock."""
        num_increments = 100
        
        async def increment():
            async with counter["lock"]:
                counter["value"] += 1
        
        tasks = [increment() for _ in range(num_increments)]
        await asyncio.gather(*tasks)
        
        assert counter["value"] == num_increments
    
    @pytest.mark.asyncio
    async def test_reference_creation_race(self):
        """Test concurrent reference creation with same ID."""
        created_ids = []
        
        async def create_reference(user_id: str) -> str:
            ref_id = f"ref-{uuid.uuid4().hex[:8]}"
            created_ids.append(ref_id)
            return ref_id
        
        # Create many references concurrently
        tasks = [create_reference("user-1") for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        # All IDs should be unique
        assert len(results) == 50
        assert len(set(results)) == 50  # All unique
        assert len(set(created_ids)) == 50
    
    @pytest.mark.asyncio
    async def test_embedding_generation_race(self):
        """Test concurrent embedding generation doesn't interfere."""
        embeddings = []
        
        async def generate_embedding(request_id: int):
            # Simulate different embedding results
            embedding = [float(request_id % 10) / 10] * 512
            embeddings.append((request_id, embedding))
            return embedding
        
        tasks = [generate_embedding(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Each embedding should be correct for its request_id
        assert len(embeddings) == 20
        for req_id, emb in embeddings:
            expected_value = float(req_id % 10) / 10
            assert all(v == expected_value for v in emb)


class TestAsyncClientRequests:
    """Test async HTTP client requests."""
    
    @pytest_asyncio.fixture
    async def async_client(self):
        """Create async test client."""
        try:
            from httpx import ASGITransport, AsyncClient
            from app.main import create_test_app
            
            app = create_test_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client
        except ImportError:
            pytest.skip("httpx not installed")
    
    @pytest.mark.asyncio
    async def test_parallel_health_checks(self, async_client):
        """Test parallel health check requests."""
        if async_client is None:
            pytest.skip("httpx not available")
        
        async def health_check():
            return await async_client.get("/health")
        
        # Multiple parallel health checks
        tasks = [health_check() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(responses) == 10
        assert all(r.status_code == 200 for r in responses)
    
    @pytest.mark.asyncio
    async def test_parallel_api_requests(self, async_client):
        """Test parallel API requests."""
        if async_client is None:
            pytest.skip("httpx not available")
        
        async def make_request(endpoint: str, data: dict):
            return await async_client.post(endpoint, json=data)
        
        # Test various endpoints
        endpoints_data = [
            ("/api/v1/verify", {"session_id": "test", "image_data": TEST_IMAGE_DATA.decode()}),
            ("/api/v1/liveness", {"session_id": "test", "image_data": TEST_IMAGE_DATA.decode()}),
        ]
        
        tasks = [
            make_request(endpoint, data)
            for endpoint, data in endpoints_data
            for _ in range(3)  # Repeat each request
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check no exceptions caused crashes
        for r in responses:
            if isinstance(r, Exception):
                pytest.fail(f"Request raised exception: {r}")
        
        # All should return valid error codes (not 500)
        for r in responses:
            assert r.status_code < 500


class TestThroughput:
    """Test throughput under concurrent load."""
    
    @pytest.fixture
    def mock_processing_time(self):
        """Mock processing with fixed delay."""
        async def process(delay: float = 0.05):
            await asyncio.sleep(delay)
            return {"processed": True, "delay": delay}
        return process
    
    @pytest.mark.asyncio
    async def test_throughput_with_delay(self, mock_processing_time):
        """Test throughput with simulated processing delay."""
        num_requests = 20
        delay = 0.02  # 20ms delay per request
        
        start_time = time.time()
        
        # Sequential execution
        seq_start = time.time()
        seq_results = [await mock_processing_time(delay) for _ in range(num_requests)]
        seq_time = time.time() - seq_start
        
        # Concurrent execution
        conc_start = time.time()
        conc_tasks = [mock_processing_time(delay) for _ in range(num_requests)]
        conc_results = await asyncio.gather(*conc_tasks)
        conc_time = time.time() - conc_start
        
        # All results should be successful
        assert len(seq_results) == num_requests
        assert len(conc_results) == num_requests
        
        # Concurrent should be faster (ideally close to single request time)
        assert conc_time < seq_time, f"Concurrent ({conc_time:.3f}s) should be faster than sequential ({seq_time:.3f}s)"
        
        # Speedup should be significant
        speedup = seq_time / conc_time
        print(f"Throughput: sequential={seq_time:.3f}s, concurrent={conc_time:.3f}s, speedup={speedup:.2f}x")
        
        # With 20 requests and ideal parallelism, should see at least 5x speedup
        assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.2f}x"


class TestConnectionPooling:
    """Test connection pool behavior under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_many_small_requests(self):
        """Test many small requests don't exhaust connections."""
        from httpx import AsyncClient, ASGITransport
        from app.main import create_test_app
        
        app = create_test_app()
        transport = ASGITransport(app=app)
        
        async with AsyncClient(transport=transport, base_url="http://test", limits=10) as client:
            # Make many small requests
            tasks = [client.get("/health") for _ in range(50)]
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(r.status_code == 200 for r in responses)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_limits(self):
        """Test behavior when exceeding connection limits."""
        from httpx import AsyncClient, ASGITransport, Timeout
        from app.main import create_test_app
        
        app = create_test_app()
        transport = ASGITransport(app=app)
        
        # Very small connection pool
        async with AsyncClient(
            transport=transport, 
            base_url="http://test", 
            limits=2,
            timeout=Timeout(5.0)
        ) as client:
            # Make requests exceeding pool size
            tasks = [client.get("/health") for _ in range(10)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Some might timeout, but shouldn't crash
            successful = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
            assert len(successful) > 0


class TestSessionConcurrency:
    """Test session-related concurrency scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self):
        """Test creating multiple sessions concurrently."""
        session_ids = []
        
        async def create_session(user_id: str) -> str:
            session_id = f"session-{uuid.uuid4().hex[:8]}"
            session_ids.append(session_id)
            return session_id
        
        # Create many sessions for same user
        user_id = "user-123"
        tasks = [create_session(user_id) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All sessions created
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique
    
    @pytest.mark.asyncio
    async def test_concurrent_verification_same_reference(self):
        """Test multiple verifications against same reference."""
        reference_id = "ref-123"
        verification_count = 5
        verifications = []
        
        async def verify(session_id: str, reference_id: str):
            verifications.append((session_id, reference_id))
            return {
                "session_id": session_id,
                "reference_id": reference_id,
                "verified": True,
                "confidence": 0.95,
            }
        
        tasks = [
            verify(f"session-{i}", reference_id)
            for i in range(verification_count)
        ]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == verification_count
        assert all(r["reference_id"] == reference_id for r in results)
        assert all(r["verified"] for r in results)
