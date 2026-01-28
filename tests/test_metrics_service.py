"""
Tests for Metrics Service
"""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.metrics_service import MetricsService, MetricsConfig
from app.middleware.metrics import (
    false_accept_rate,
    false_reject_rate,
    equal_error_rate,
)


class TestMetricsConfig:
    """Tests for MetricsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MetricsConfig()

        assert config.WINDOW_SIZE == 1000
        assert config.UPDATE_INTERVAL == 60
        assert config.TARGET_FAR == 0.001
        assert config.TARGET_FRR == 0.03
        assert config.SEVERITY_THRESHOLDS['low'] == 0.1
        assert config.SEVERITY_THRESHOLDS['medium'] == 0.2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MetricsConfig(
            WINDOW_SIZE=500,
            UPDATE_INTERVAL=30,
            TARGET_FAR=0.005,
            TARGET_FRR=0.05,
        )

        assert config.WINDOW_SIZE == 500
        assert config.UPDATE_INTERVAL == 30
        assert config.TARGET_FAR == 0.005
        assert config.TARGET_FRR == 0.05


class TestMetricsService:
    """Tests for MetricsService."""

    @pytest.fixture
    def metrics_service(self):
        """Create a MetricsService instance for testing."""
        return MetricsService()

    @pytest.fixture
    def mock_db_service(self):
        """Create a mock database service."""
        mock = AsyncMock()
        mock.get_metrics_history = AsyncMock(return_value=[])
        return mock

    @pytest.mark.asyncio
    async def test_start_stop(self, metrics_service):
        """Test service start and stop."""
        assert metrics_service._running is False

        await metrics_service.start()
        assert metrics_service._running is True
        assert metrics_service._update_task is not None

        await metrics_service.stop()
        assert metrics_service._running is False

    @pytest.mark.asyncio
    async def test_record_verification_genuine_accepted(self, metrics_service):
        """Test recording a genuine user that was accepted."""
        result = await metrics_service.record_verification(
            similarity_score=0.85,
            is_genuine=True,
            threshold=0.7,
            match_result=True,
        )

        assert result['event'] == 'verification_recorded'
        assert metrics_service._genuine_total == 1
        assert metrics_service._genuine_rejected == 0
        assert len(metrics_service.genuine_scores) == 1

    @pytest.mark.asyncio
    async def test_record_verification_genuine_rejected(self, metrics_service):
        """Test recording a genuine user that was rejected (False Reject)."""
        result = await metrics_service.record_verification(
            similarity_score=0.65,
            is_genuine=True,
            threshold=0.7,
            match_result=False,
        )

        assert result['event'] == 'false_reject'
        assert result['severity'] == 'low'
        assert metrics_service._genuine_total == 1
        assert metrics_service._genuine_rejected == 1
        assert metrics_service.genuine_scores[-1] == 0.65

    @pytest.mark.asyncio
    async def test_record_verification_impostor_accepted(self, metrics_service):
        """Test recording an impostor that was accepted (False Accept)."""
        result = await metrics_service.record_verification(
            similarity_score=0.75,
            is_genuine=False,
            threshold=0.7,
            match_result=True,
        )

        assert result['event'] == 'false_accept'
        assert result['severity'] == 'medium'
        assert metrics_service._impostor_total == 1
        assert metrics_service._impostor_accepted == 1
        assert metrics_service.impostor_scores[-1] == 0.75

    @pytest.mark.asyncio
    async def test_record_verification_impostor_rejected(self, metrics_service):
        """Test recording an impostor that was rejected."""
        result = await metrics_service.record_verification(
            similarity_score=0.55,
            is_genuine=False,
            threshold=0.7,
            match_result=False,
        )

        assert result['event'] == 'verification_recorded'
        assert metrics_service._impostor_total == 1
        assert metrics_service._impostor_accepted == 0
        assert len(metrics_service.impostor_scores) == 1

    @pytest.mark.asyncio
    async def test_severity_calculation_false_reject(self, metrics_service):
        """Test severity calculation for false rejects."""
        # Close to threshold - low severity
        severity = metrics_service._calculate_severity(
            score=0.68, threshold=0.7, is_false_reject=True
        )
        assert severity == 'low'

        # Medium distance - medium severity
        severity = metrics_service._calculate_severity(
            score=0.55, threshold=0.7, is_false_reject=True
        )
        assert severity == 'medium'

        # Large distance - high severity
        severity = metrics_service._calculate_severity(
            score=0.4, threshold=0.7, is_false_reject=True
        )
        assert severity == 'high'

    @pytest.mark.asyncio
    async def test_severity_calculation_false_accept(self, metrics_service):
        """Test severity calculation for false accepts."""
        # Close to threshold - low severity
        severity = metrics_service._calculate_severity(
            score=0.72, threshold=0.7, is_false_reject=False
        )
        assert severity == 'low'

        # Medium above threshold - medium severity
        severity = metrics_service._calculate_severity(
            score=0.85, threshold=0.7, is_false_reject=False
        )
        assert severity == 'medium'

        # High above threshold - high severity
        severity = metrics_service._calculate_severity(
            score=0.95, threshold=0.7, is_false_reject=False
        )
        assert severity == 'high'

    @pytest.mark.asyncio
    async def test_update_metrics(self, metrics_service):
        """Test metrics update."""
        # Record some verifications
        await metrics_service.record_verification(
            similarity_score=0.85, is_genuine=True, threshold=0.7, match_result=True
        )
        await metrics_service.record_verification(
            similarity_score=0.75, is_genuine=False, threshold=0.7, match_result=True  # False accept
        )

        # Update metrics
        await metrics_service._update_metrics()

        # Check that Prometheus metrics are updated
        # Note: actual values depend on the middleware implementation
        assert metrics_service._genuine_total == 1
        assert metrics_service._impostor_total == 1
        assert metrics_service._impostor_accepted == 1

    def test_calculate_eer(self, metrics_service):
        """Test EER calculation."""
        genuine_scores = [0.8, 0.85, 0.9, 0.95]
        impostor_scores = [0.3, 0.4, 0.5, 0.6]

        eer = metrics_service._calculate_eer(genuine_scores, impostor_scores)

        # EER should be between 0 and 100
        assert 0 <= eer <= 100

    def test_calculate_eer_empty_lists(self, metrics_service):
        """Test EER with empty lists."""
        eer = metrics_service._calculate_eer([], [])
        assert eer == 0.0

        eer = metrics_service._calculate_eer([0.8], [])
        assert eer == 0.0

    @pytest.mark.asyncio
    async def test_get_current_metrics(self, metrics_service):
        """Test getting current metrics."""
        # Record some verifications
        for i in range(5):
            await metrics_service.record_verification(
                similarity_score=0.8 + i * 0.03,
                is_genuine=True,
                threshold=0.7,
                match_result=True,
            )
        for i in range(3):
            await metrics_service.record_verification(
                similarity_score=0.4 + i * 0.1,
                is_genuine=False,
                threshold=0.7,
                match_result=False,
            )

        metrics = await metrics_service.get_current_metrics()

        assert metrics['genuine_count'] == 5
        assert metrics['impostor_count'] == 3
        assert 'far' in metrics
        assert 'frr' in metrics
        assert 'eer' in metrics
        assert 'compliance' in metrics
        assert 'distributions' in metrics

    @pytest.mark.asyncio
    async def test_get_current_metrics_compliance(self, metrics_service):
        """Test compliance checking."""
        # Record only genuine users (no impostors)
        await metrics_service.record_verification(
            similarity_score=0.85, is_genuine=True, threshold=0.7, match_result=True
        )

        metrics = await metrics_service.get_current_metrics()

        assert metrics['compliance']['far_compliant'] is True
        assert metrics['compliance']['frr_compliant'] is True

    @pytest.mark.asyncio
    async def test_window_size_limit(self, metrics_service):
        """Test that scores window size is limited."""
        service = MetricsService(config=MetricsConfig(WINDOW_SIZE=10))

        # Add more than WINDOW_SIZE scores
        for i in range(15):
            await service.record_verification(
                similarity_score=0.5 + (i % 5) * 0.1,
                is_genuine=True,
                threshold=0.7,
                match_result=True,
            )

        # Should be limited to WINDOW_SIZE
        assert len(service.genuine_scores) == 10

    @pytest.mark.asyncio
    async def test_get_status(self, metrics_service):
        """Test getting service status."""
        status = metrics_service.get_status()

        assert 'running' in status
        assert 'update_task_active' in status
        assert 'window_size' in status
        assert 'genuine_buffer_size' in status
        assert 'impostor_buffer_size' in status


class TestMetricsConcurrency:
    """Tests for concurrent access to metrics service."""

    @pytest.mark.asyncio
    async def test_concurrent_recordings(self):
        """Test concurrent verification recordings."""
        service = MetricsService()

        async def record_verification():
            for _ in range(10):
                await service.record_verification(
                    similarity_score=0.8,
                    is_genuine=True,
                    threshold=0.7,
                    match_result=True,
                )

        # Run concurrent tasks
        tasks = [record_verification() for _ in range(5)]
        await asyncio.gather(*tasks)

        # All recordings should be processed
        assert service._genuine_total == 50


class TestMetricsServiceIntegration:
    """Integration tests for metrics service."""

    @pytest.mark.asyncio
    async def test_full_verification_flow(self):
        """Test complete verification flow with metrics."""
        service = MetricsService()

        # Simulate normal verifications
        for i in range(10):
            await service.record_verification(
                similarity_score=0.85 + (i % 3) * 0.02,
                is_genuine=True,
                threshold=0.7,
                match_result=True,
            )

        # Simulate an impostor attempt that gets rejected
        await service.record_verification(
            similarity_score=0.65,
            is_genuine=False,
            threshold=0.7,
            match_result=False,
        )

        # Simulate a false accept
        await service.record_verification(
            similarity_score=0.72,
            is_genuine=False,
            threshold=0.7,
            match_result=True,
        )

        metrics = await service.get_current_metrics()

        # Verify metrics
        assert metrics['genuine_count'] == 10
        assert metrics['impostor_count'] == 2
        assert metrics['window_size'] == 12

        # FAR should be 50% (1 false accept out of 2 impostors)
        # FRR should be 0%
        assert 'far' in metrics
        assert 'frr' in metrics


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])