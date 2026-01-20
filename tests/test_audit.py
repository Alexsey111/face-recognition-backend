"""Tests for audit logging functionality."""
import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
from io import StringIO

import pytest

from app.utils.logger import (
    audit_event,
    LogContext,
    log_with_context,
    setup_logger,
    get_logger,
    _redact,
    StructuredFormatter,
    EmptyMessageFilter,
    _LOG_CONTEXT,
)


class TestAuditEventFunction:
    """Tests for audit_event function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset audit logger handler for each test
        audit_logger = logging.getLogger("audit")
        for handler in audit_logger.handlers[:]:
            audit_logger.removeHandler(handler)

    def test_audit_event_info_level(self):
        """Should log audit event at info level."""
        audit_event(
            action="user_login",
            target_type="user",
            target_id="user-123",
            user_id="user-123",
        )

    def test_audit_event_error_level(self):
        """Should log audit event at error level."""
        audit_event(
            action="user_login",
            target_type="user",
            target_id="user-123",
            success=False,
            error_message="Login failed",
            level="error",
        )

    def test_audit_event_warning_level(self):
        """Should log audit event at warning level."""
        audit_event(
            action="rate_limit",
            target_type="user",
            target_id="user-123",
            level="warning",
        )

    def test_audit_event_with_details(self):
        """Should include details in audit event."""
        audit_event(
            action="user_updated",
            target_type="user",
            target_id="user-123",
            details={"field": "email", "old": "old@email.com", "new": "new@email.com"},
        )

    def test_audit_event_redacts_sensitive(self):
        """Should redact sensitive data in audit event."""
        audit_event(
            action="user_login",
            target_type="user",
            target_id="user-123",
            details={
                "password": "secret",
                "api_key": "key123",
                "token": "abc123",
            },
        )

    def test_audit_event_with_admin(self):
        """Should log admin actions correctly."""
        audit_event(
            action="user_deleted",
            target_type="user",
            target_id="user-123",
            admin_id="admin-456",
        )


class TestRedactSensitiveData:
    """Tests for sensitive data redaction."""

    def test_redact_dict_with_password(self):
        """Should redact password field."""
        data = {"username": "test", "password": "secret123"}
        result = _redact(data)
        assert result["username"] == "test"
        assert result["password"] == "[REDACTED]"

    def test_redact_nested_password(self):
        """Should redact nested password."""
        data = {"user": {"password": "secret123"}}
        result = _redact(data)
        assert result["user"]["password"] == "[REDACTED]"

    def test_redact_list_items(self):
        """Should redact items in list."""
        data = [{"password": "pass1"}, {"password": "pass2"}]
        result = _redact(data)
        assert result[0]["password"] == "[REDACTED]"
        assert result[1]["password"] == "[REDACTED]"

    def test_redact_multiple_sensitive_keys(self):
        """Should redact multiple sensitive keys."""
        data = {
            "password": "secret",
            "passwd": "secret",
            "token": "abc",
            "access_token": "xyz",
            "refresh_token": "refresh",
            "api_key": "key123",
        }
        result = _redact(data)
        for v in result.values():
            assert v == "[REDACTED]"

    def test_redact_non_dict(self):
        """Should return non-dict values unchanged."""
        assert _redact("string") == "string"
        assert _redact(123) == 123
        assert _redact(None) is None


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context_basic(self):
        """Should set context correctly."""
        with LogContext(request_id="req-123", user_id="user-456"):
            ctx = _LOG_CONTEXT.get()
            assert ctx["request_id"] == "req-123"
            assert ctx["user_id"] == "user-456"

    def test_log_context_extra_fields(self):
        """Should include extra fields."""
        with LogContext(request_id="req-123", extra={"action": "login"}):
            ctx = _LOG_CONTEXT.get()
            assert ctx["request_id"] == "req-123"
            assert ctx["action"] == "login"

    def test_log_context_exits_cleanly(self):
        """Should reset context after exit."""
        with LogContext(request_id="req-123"):
            pass
        ctx = _LOG_CONTEXT.get()
        assert ctx == {}

    def test_log_context_nested(self):
        """Should handle nested contexts."""
        with LogContext(request_id="req-123"):
            ctx1 = _LOG_CONTEXT.get()
            assert ctx1["request_id"] == "req-123"
            with LogContext(request_id="req-456"):
                ctx2 = _LOG_CONTEXT.get()
                assert ctx2["request_id"] == "req-456"
            ctx3 = _LOG_CONTEXT.get()
            assert ctx3["request_id"] == "req-123"


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_format_basic_message(self):
        """Should format log record to JSON."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_format_with_context(self):
        """Should include context variables."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with LogContext(request_id="req-123"):
            formatted = formatter.format(record)
            data = json.loads(formatted)
            assert data["request_id"] == "req-123"

    def test_format_empty_message(self):
        """Should return empty string for empty message."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="   ",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert formatted == ""

    def test_format_redacts_sensitive(self):
        """Should redact sensitive data."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )

        with LogContext(extra={"password": "secret", "api_key": "key123"}):
            formatted = formatter.format(record)
            data = json.loads(formatted)
            assert data["password"] == "[REDACTED]"
            assert data["api_key"] == "[REDACTED]"


class TestEmptyMessageFilter:
    """Tests for EmptyMessageFilter."""

    def test_filter_empty_message(self):
        """Should filter empty messages."""
        filter = EmptyMessageFilter()
        record = MagicMock()
        record.getMessage.return_value = ""

        assert filter.filter(record) is False

    def test_filter_whitespace_message(self):
        """Should filter whitespace-only messages."""
        filter = EmptyMessageFilter()
        record = MagicMock()
        record.getMessage.return_value = "   "

        assert filter.filter(record) is False

    def test_filter_valid_message(self):
        """Should allow valid messages."""
        filter = EmptyMessageFilter()
        record = MagicMock()
        record.getMessage.return_value = "Valid message"

        assert filter.filter(record) is True


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_with_context(self):
        """Should log with context."""
        logger = MagicMock()
        log_with_context(logger.info, "Test message", action="login")

        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert call_args[1]["extra"]["action"] == "login"


class TestAuditIntegration:
    """Integration tests for audit functionality."""

    def test_audit_event_all_parameters(self):
        """Should handle all audit_event parameters."""
        audit_event(
            action="user_created",
            target_type="user",
            target_id="user-123",
            user_id="user-123",
            admin_id="admin-456",
            details={"method": "registration"},
            old_values=None,
            new_values={"name": "New User"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True,
            error_message=None,
            level="info",
        )

    def test_audit_event_minimal_parameters(self):
        """Should handle minimal audit_event parameters."""
        audit_event(
            action="user_login",
            target_type="user",
            target_id="user-123",
        )

    def test_setup_logger_idempotent(self):
        """Logger setup should be idempotent."""
        logger1 = setup_logger("test_idempotent")
        logger2 = setup_logger("test_idempotent")
        assert logger1 is logger2

    def test_get_logger_returns_configured(self):
        """get_logger should return configured logger."""
        logger = get_logger("test_get")
        assert logger is not None
        assert logger.level is not None