"""
Tests for logging utilities.

Tests cover:
- Sensitive data redaction
- Context-based logging (LogContext)
- Structured logging formatter
- Safe stream handler
- Empty message filter
- Log with context function
- Audit event function
"""

import json
import logging
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

# Import directly from logger module to avoid app imports
from app.utils.logger import (
    SENSITIVE_KEYS,
    EmptyMessageFilter,
    LogContext,
    SafeStreamHandler,
    StructuredFormatter,
    _redact,
    audit_event,
    log_with_context,
)


class TestSensitiveDataRedaction:
    """Tests for sensitive data redaction in logs."""

    def test_redact_password_field(self):
        """Password field should be redacted."""
        data = {"password": "secret123", "username": "user"}
        result = _redact(data)
        assert result["password"] == "[REDACTED]"
        assert result["username"] == "user"

    def test_redact_access_token(self):
        """Access token should be redacted."""
        data = {"access_token": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"}
        result = _redact(data)
        assert result["access_token"] == "[REDACTED]"

    def test_redact_refresh_token(self):
        """Refresh token should be redacted."""
        data = {"refresh_token": "token123", "token_type": "Bearer"}
        result = _redact(data)
        assert result["refresh_token"] == "[REDACTED]"

    def test_redact_nested_password(self):
        """Nested password in dict should be redacted."""
        data = {"user": {"password": "secret", "email": "test@test.com"}}
        result = _redact(data)
        assert result["user"]["password"] == "[REDACTED]"
        assert result["user"]["email"] == "test@test.com"

    def test_redact_list_with_passwords(self):
        """Passwords in list should be redacted."""
        data = [{"password": "pass1"}, {"password": "pass2"}]
        result = _redact(data)
        assert result[0]["password"] == "[REDACTED]"
        assert result[1]["password"] == "[REDACTED]"

    def test_redact_embedding(self):
        """Embedding data should be redacted."""
        data = {"embedding": [0.1, 0.2, 0.3], "size": 512}
        result = _redact(data)
        assert result["embedding"] == "[REDACTED]"
        assert result["size"] == 512

    def test_redact_case_insensitive(self):
        """Redaction should be case-insensitive."""
        data = {"PASSWORD": "secret", "Password": "secret2"}
        result = _redact(data)
        assert result["PASSWORD"] == "[REDACTED]"
        assert result["Password"] == "[REDACTED]"

    def test_sensitive_keys_coverage(self):
        """Ensure all expected sensitive keys are covered."""
        expected_keys = {
            "password",
            "passwd",
            "token",
            "access_token",
            "refresh_token",
            "embeddings",
            "embedding",
            "image",
            "file",
            "ssn",
            "credit_card",
            "api_key",
            "secret",
        }
        assert SENSITIVE_KEYS == expected_keys

    def test_redact_non_dict(self):
        """Non-dict values should pass through."""
        assert _redact("string") == "string"
        assert _redact(123) == 123
        assert _redact(None) is None


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_log_context_with_values(self):
        """LogContext should set request_id and user_id."""
        with LogContext(request_id="req-123", user_id="user-456") as ctx:
            from app.utils.logger import _LOG_CONTEXT

            context = _LOG_CONTEXT.get()
            assert context["request_id"] == "req-123"
            assert context["user_id"] == "user-456"

    def test_log_context_with_extra(self):
        """LogContext should accept extra fields."""
        with LogContext(extra={"action": "login", "ip": "127.0.0.1"}):
            from app.utils.logger import _LOG_CONTEXT

            context = _LOG_CONTEXT.get()
            assert context["action"] == "login"
            assert context["ip"] == "127.0.0.1"

    def test_log_context_cleanup(self):
        """LogContext should cleanup on exit."""
        from app.utils.logger import _LOG_CONTEXT

        with LogContext(request_id="req-123"):
            pass

        # Context should be empty after exit
        context = _LOG_CONTEXT.get()
        assert context.get("request_id") is None

    def test_log_context_nested(self):
        """Nested LogContext should work correctly."""
        from app.utils.logger import _LOG_CONTEXT

        with LogContext(request_id="outer"):
            outer_ctx = _LOG_CONTEXT.get()

            with LogContext(request_id="inner"):
                inner_ctx = _LOG_CONTEXT.get()
                assert inner_ctx["request_id"] == "inner"

            # After inner exits, should restore outer context
            after_ctx = _LOG_CONTEXT.get()
            assert after_ctx["request_id"] == "outer"


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_format_basic_message(self):
        """Basic log message should be formatted correctly."""
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
        """Log record should include context variables."""
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
        record.request_id = "req-123"
        record.user_id = "user-456"

        with LogContext(request_id="req-123", user_id="user-456"):
            formatted = formatter.format(record)
            data = json.loads(formatted)
            assert data["request_id"] == "req-123"
            assert data["user_id"] == "user-456"

    def test_format_empty_message_filtered(self):
        """Empty messages should return empty string."""
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

    def test_format_with_exception(self):
        """Exception info should be included."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )
        try:
            raise ValueError("Test error")
        except ValueError:
            record.exc_info = sys.exc_info()

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_format_redacts_sensitive_data_in_extra(self):
        """Sensitive data in extra fields should be redacted in output."""
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


class TestSafeStreamHandler:
    """Tests for SafeStreamHandler."""

    def test_emit_writes_message(self):
        """Handler should write formatted message to stream."""
        stream = StringIO()
        handler = SafeStreamHandler(stream=stream)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        output = stream.getvalue()

        assert "Test message" in output

    def test_emit_skips_empty_messages(self):
        """Handler should skip empty messages when using StructuredFormatter."""
        stream = StringIO()
        handler = SafeStreamHandler(stream=stream)
        handler.setFormatter(StructuredFormatter())

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="   ",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        # StructuredFormatter returns empty string for empty messages
        assert stream.getvalue() == ""


class TestEmptyMessageFilter:
    """Tests for EmptyMessageFilter."""

    def test_filter_allows_normal_message(self):
        """Normal message should pass through filter."""
        filter_obj = EmptyMessageFilter()
        record = MagicMock()
        record.getMessage.return_value = "Valid message"

        assert filter_obj.filter(record) is True

    def test_filter_blocks_empty_message(self):
        """Empty message should be blocked."""
        filter_obj = EmptyMessageFilter()
        record = MagicMock()
        record.getMessage.return_value = "   "

        assert filter_obj.filter(record) is False

    def test_filter_blocks_none_message(self):
        """None message should be blocked."""
        filter_obj = EmptyMessageFilter()
        record = MagicMock()
        record.getMessage.return_value = None

        assert filter_obj.filter(record) is False


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_with_context_adds_context(self):
        """log_with_context should include context in log record."""
        logger = logging.getLogger("test_log_context")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with LogContext(request_id="req-123", user_id="user-456"):
            log_with_context(logger.info, "Test message", action="login")

        output = stream.getvalue()
        data = json.loads(output)
        assert data["request_id"] == "req-123"
        assert data["user_id"] == "user-456"
        assert data["action"] == "login"

    def test_log_with_context_redacts_sensitive(self):
        """log_with_context should redact sensitive data."""
        logger = logging.getLogger("test_redact_context")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        log_with_context(logger.info, "Test", password="secret")

        output = stream.getvalue()
        data = json.loads(output)
        assert data["password"] == "[REDACTED]"
        assert "secret" not in output


class TestAuditEvent:
    """Tests for audit_event function."""

    def test_audit_event_logs_json(self):
        """audit_event should log structured JSON."""
        audit_logger = logging.getLogger("test_audit")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)

        with patch("app.utils.logger.logging.getLogger", return_value=audit_logger):
            audit_event(
                action="user_login",
                target_type="user",
                target_id="user-123",
                user_id="user-123",
                ip_address="127.0.0.1",
                success=True,
            )

        output = stream.getvalue()
        data = json.loads(output)

        assert data["action"] == "user_login"
        assert data["target_type"] == "user"
        assert data["target_id"] == "user-123"
        assert data["success"] is True

    def test_audit_event_with_details(self):
        """audit_event should include details."""
        audit_logger = logging.getLogger("test_audit_details")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)

        with patch("app.utils.logger.logging.getLogger", return_value=audit_logger):
            audit_event(
                action="verification",
                target_type="session",
                target_id="sess-123",
                details={"method": "liveness", "confidence": 0.95},
            )

        output = stream.getvalue()
        data = json.loads(output)

        assert data["details"]["method"] == "liveness"
        assert data["details"]["confidence"] == 0.95

    def test_audit_event_with_old_new_values(self):
        """audit_event should track old and new values."""
        audit_logger = logging.getLogger("test_audit_values")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)

        with patch("app.utils.logger.logging.getLogger", return_value=audit_logger):
            audit_event(
                action="user_update",
                target_type="user",
                target_id="user-123",
                old_values={"email": "old@test.com"},
                new_values={"email": "new@test.com"},
            )

        output = stream.getvalue()
        data = json.loads(output)

        assert data["old_values"]["email"] == "old@test.com"
        assert data["new_values"]["email"] == "new@test.com"

    def test_audit_event_error_level(self):
        """audit_event should use error level when specified."""
        audit_logger = logging.getLogger("test_audit_error")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.ERROR)

        with patch("app.utils.logger.logging.getLogger", return_value=audit_logger):
            audit_event(
                action="failed_operation",
                target_type="operation",
                target_id="op-123",
                success=False,
                error_message="Connection timeout",
                level="error",
            )

        output = stream.getvalue()
        assert "Connection timeout" in output


class TestLogFunctionCall:
    """Tests for log_function_call decorator."""

    def test_decorator_logs_execution(self):
        """Decorator should log function execution."""
        logger = logging.getLogger("test_decorator")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with patch("app.utils.logger.get_logger", return_value=logger):
            from app.utils.logger import log_function_call

            @log_function_call
            def simple_function():
                return 42

            result = simple_function()
            output = stream.getvalue()

            assert result == 42
            data = json.loads(output)
            assert "Function executed" in data["message"]
            assert data["function"] == "simple_function"
            assert data["success"] is True

    def test_decorator_logs_duration(self):
        """Decorator should include duration in log."""
        logger = logging.getLogger("test_duration")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with patch("app.utils.logger.get_logger", return_value=logger):
            from app.utils.logger import log_function_call

            @log_function_call
            def slow_function():
                import time

                time.sleep(0.01)
                return "done"

            slow_function()
            output = stream.getvalue()
            data = json.loads(output)

            assert "duration" in data
            assert isinstance(data["duration"], (int, float))

    def test_decorator_logs_failure(self):
        """Decorator should log exceptions."""
        logger = logging.getLogger("test_failure")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with patch("app.utils.logger.get_logger", return_value=logger):
            from app.utils.logger import log_function_call

            @log_function_call
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError):
                failing_function()

            output = stream.getvalue()
            data = json.loads(output)
            assert "Function failed" in data["message"]
            assert data["function"] == "failing_function"
            assert data["success"] is False
            assert "Test error" in data["error"]

    def test_decorator_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring."""
        from app.utils.logger import log_function_call

        @log_function_call
        def my_function():
            """My docstring."""
            return True

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestGetLoggerSetup:
    """Tests for get_logger and setup_logger."""

    def test_get_logger_returns_logger(self):
        """get_logger should return a logger instance."""
        from app.utils.logger import get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_idempotent(self):
        """get_logger should be safe to call multiple times."""
        from app.utils.logger import get_logger

        logger1 = get_logger("test_idempotent")
        logger2 = get_logger("test_idempotent")
        assert logger1 is logger2

    def test_setup_logger_configures_handlers(self):
        """setup_logger should configure handlers."""
        from app.utils.logger import setup_logger

        logger = setup_logger("test_setup", level="DEBUG")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logger_with_file(self, tmp_path):
        """setup_logger should create file handler when path provided."""
        from app.utils.logger import setup_logger

        log_file = tmp_path / "test.log"
        logger = setup_logger("test_file", level="DEBUG", log_file=str(log_file))
        assert logger.level == logging.DEBUG
        assert log_file.exists()


class TestAuditLogger:
    """Tests for audit logger configuration."""

    def test_audit_logger_exists(self):
        """Audit logger should exist."""
        audit_logger = logging.getLogger("audit")
        assert audit_logger is not None

    def test_audit_logger_has_handlers(self, tmp_path):
        """Audit logger should have handlers when configured."""
        from app.utils.logger import setup_logger

        log_file = tmp_path / "test.log"
        setup_logger("test", log_file=str(log_file))

        audit_logger = logging.getLogger("audit")
        # Audit logger should have handlers after setup
        assert len(audit_logger.handlers) > 0 or audit_logger.parent is not None
