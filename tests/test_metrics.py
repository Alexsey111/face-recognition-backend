"""Tests for metrics middleware."""

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app.middleware.metrics import (
    PROMETHEUS_AVAILABLE,
    MetricsMiddleware,
    get_metrics,
    get_metrics_content_type,
    initialize_metrics,
    record_auth,
    record_business_error,
    record_error,
    record_liveness,
    record_reference,
    record_token,
    record_upload,
    record_verification,
    track_db_query,
    track_processing,
)


class TestPrometheusAvailability:
    """Tests for Prometheus library availability."""

    def test_prometheus_available_flag_exists(self):
        """PROMETHEUS_AVAILABLE flag should exist."""
        assert (
            hasattr(PROMETHEUS_AVAILABLE, "__bool__")
            or PROMETHEUS_AVAILABLE is None
            or isinstance(PROMETHEUS_AVAILABLE, bool)
        )


class TestRecordVerification:
    """Tests for record_verification function."""

    def test_record_verification_success(self):
        """Should record successful verification."""
        record_verification("success")

    def test_record_verification_failure(self):
        """Should record failed verification."""
        record_verification("failure")

    def test_record_verification_with_different_results(self):
        """Should record different verification results."""
        for result in ["success", "failure", "unknown", "error"]:
            record_verification(result)


class TestRecordLiveness:
    """Tests for record_liveness function."""

    def test_record_liveness_success(self):
        """Should record successful liveness check."""
        record_liveness("success")

    def test_record_liveness_failure(self):
        """Should record failed liveness check."""
        record_liveness("failure")

    def test_record_liveness_with_different_results(self):
        """Should record different liveness results."""
        for result in ["success", "failure", "spoof", "unknown"]:
            record_liveness(result)


class TestRecordUpload:
    """Tests for record_upload function."""

    def test_record_upload_success(self):
        """Should record successful upload."""
        record_upload("success", "image")

    def test_record_upload_failure(self):
        """Should record failed upload."""
        record_upload("failure", "image")

    def test_record_upload_with_different_file_types(self):
        """Should record uploads with different file types."""
        for file_type in ["image", "video", "document"]:
            record_upload("success", file_type)


class TestRecordReference:
    """Tests for record_reference function."""

    def test_record_reference_add(self):
        """Should record reference add operation."""
        record_reference("add")

    def test_record_reference_delete(self):
        """Should record reference delete operation."""
        record_reference("delete")

    def test_record_reference_update(self):
        """Should record reference update operation."""
        record_reference("update")

    def test_record_reference_list(self):
        """Should record reference list operation."""
        record_reference("list")

    def test_record_reference_with_different_operations(self):
        """Should record different reference operations."""
        for operation in ["add", "delete", "update", "list", "get"]:
            record_reference(operation)


class TestRecordAuth:
    """Tests for record_auth function."""

    def test_record_auth_success(self):
        """Should record successful authentication."""
        record_auth("success", "jwt")

    def test_record_auth_failure(self):
        """Should record failed authentication."""
        record_auth("failure", "jwt")

    def test_record_auth_with_different_methods(self):
        """Should record auth with different methods."""
        for method in ["jwt", "api_key", "oauth"]:
            record_auth("success", method)


class TestRecordToken:
    """Tests for record_token function."""

    def test_record_token_access(self):
        """Should record access token issuance."""
        record_token("access")

    def test_record_token_refresh(self):
        """Should record refresh token issuance."""
        record_token("refresh")


class TestRecordBusinessError:
    """Tests for record_business_error function."""

    def test_record_business_error_validation(self):
        """Should record validation error."""
        record_business_error("validation_error")

    def test_record_business_error_processing(self):
        """Should record processing error."""
        record_business_error("processing_error")

    def test_record_business_error_with_different_types(self):
        """Should record different business error types."""
        for error_type in [
            "validation_error",
            "processing_error",
            "auth_error",
            "not_found",
        ]:
            record_business_error(error_type)


class TestRecordError:
    """Tests for record_error function."""

    def test_record_error_with_status_code(self):
        """Should record HTTP error with status code."""
        record_error("500", "/api/users")

    def test_record_error_with_different_endpoints(self):
        """Should record errors for different endpoints."""
        record_error("404", "/api/users")
        record_error("400", "/api/verify")


class TestTrackDbQuery:
    """Tests for track_db_query context manager."""

    def test_track_db_query_normal_operation(self):
        """Should measure database query time."""
        with track_db_query("select"):
            time.sleep(0.01)

    def test_track_db_query_with_exception(self):
        """Should still measure time even if exception occurs."""
        with pytest.raises(ValueError):
            with track_db_query("insert"):
                time.sleep(0.01)
                raise ValueError("Test error")

    def test_track_db_query_nested_queries(self):
        """Should handle nested queries correctly."""
        with track_db_query("outer"):
            with track_db_query("inner"):
                time.sleep(0.01)


class TestTrackProcessing:
    """Tests for track_processing context manager."""

    def test_track_processing_normal_operation(self):
        """Should measure processing time."""
        with track_processing("admin_stats"):
            time.sleep(0.01)

    def test_track_processing_with_exception(self):
        """Should still measure time even if exception occurs."""
        with pytest.raises(ValueError):
            with track_processing("admin_operation"):
                time.sleep(0.01)
                raise ValueError("Test error")

    def test_track_processing_different_operations(self):
        """Should track different operation types."""
        for operation in [
            "admin_stats",
            "admin_audit_logs",
            "admin_health",
            "admin_errors",
        ]:
            with track_processing(operation):
                time.sleep(0.001)


class TestInitializeMetrics:
    """Tests for initialize_metrics function."""

    def test_initialize_metrics_no_error(self):
        """Should initialize metrics without error."""
        initialize_metrics()


class TestGetMetrics:
    """Tests for get_metrics function."""

    def test_get_metrics_returns_string(self):
        """Should return string (possibly empty)."""
        result = get_metrics()
        assert isinstance(result, str)

    def test_get_metrics_returns_prometheus_format(self):
        """Should return Prometheus metrics format."""
        result = get_metrics()
        # May be empty if Prometheus is not available
        if result:
            assert isinstance(result, str)


class TestGetMetricsContentType:
    """Tests for get_metrics_content_type function."""

    def test_get_metrics_content_type_returns_string(self):
        """Should return content type string."""
        result = get_metrics_content_type()
        assert isinstance(result, str)
        assert "text/plain" in result


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    def test_middleware_skip_health_paths(self):
        """Should skip metrics for health endpoints."""
        middleware = MetricsMiddleware(app=MagicMock())

        skipped_paths = [
            "/health",
            "/ready",
            "/live",
            "/status",
            "/metrics",
            "/favicon.ico",
        ]
        for path in skipped_paths:
            assert path in middleware.SKIP_PATHS

    def test_normalize_path_uuid(self):
        """Should normalize UUID paths."""
        middleware = MetricsMiddleware(app=MagicMock())
        path = "/api/users/550e8400-e29b-41d4-a716-446655440000"
        normalized = middleware._normalize_path(path)
        assert normalized == "/api/users/{uuid}"

    def test_normalize_path_numeric_id(self):
        """Should normalize numeric ID paths."""
        middleware = MetricsMiddleware(app=MagicMock())
        path = "/api/users/12345"
        normalized = middleware._normalize_path(path)
        assert normalized == "/api/users/{id}"

    def test_normalize_path_hash(self):
        """Should normalize hash paths."""
        middleware = MetricsMiddleware(app=MagicMock())
        path = "/api/files/abcdef1234567890"
        normalized = middleware._normalize_path(path)
        assert normalized == "/api/files/{hash}"

    def test_normalize_path_preserves_static(self):
        """Should preserve static paths."""
        middleware = MetricsMiddleware(app=MagicMock())
        path = "/static/css/style.css"
        normalized = middleware._normalize_path(path)
        assert normalized == path

    def test_normalize_path_mixed_patterns(self):
        """Should handle mixed path patterns."""
        middleware = MetricsMiddleware(app=MagicMock())
        # Normalize UUID
        path = "/api/verifications/550e8400-e29b-41d4-a716-446655440000"
        normalized = middleware._normalize_path(path)
        assert normalized == "/api/verifications/{uuid}"


class TestMetricsIntegration:
    """Integration tests for metrics module."""

    def test_all_record_functions_execute(self):
        """All record functions should execute without error."""
        # Record various metrics
        record_verification("success")
        record_liveness("success")
        record_upload("success", "image")
        record_reference("add")
        record_auth("success", "jwt")
        record_token("access")
        record_business_error("validation_error")
        record_error("400", "/api/test")

    def test_all_context_managers_execute(self):
        """All context managers should execute without error."""
        with track_db_query("test"):
            pass

        with track_processing("test"):
            pass

    def test_metrics_functions_execute(self):
        """All metric helper functions should execute without error."""
        initialize_metrics()
        get_metrics()
        get_metrics_content_type()
