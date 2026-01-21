import logging
import sys
import json
import time
import asyncio
import inspect
import functools
import contextvars
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional, Callable, Any, Dict, Mapping

from ..config import settings


# =========================
# Sensitive data redaction
# =========================

SENSITIVE_KEYS = {
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


def _redact(obj: Any) -> Any:
    """Recursively redact sensitive fields."""
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if k and k.lower() in SENSITIVE_KEYS else _redact(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


# =========================
# Context-based logging
# =========================

_LOG_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "LOG_CONTEXT", default={}
)


class LogContext:
    """
    Context manager for request-scoped logging context.
    Safe for async and threaded execution.
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.extra = extra or {}
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        ctx: Dict[str, Any] = {}
        if self.request_id:
            ctx["request_id"] = self.request_id
        if self.user_id:
            ctx["user_id"] = self.user_id
        ctx.update(self.extra)
        self._token = _LOG_CONTEXT.set(ctx)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _LOG_CONTEXT.reset(self._token)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        self.__exit__(exc_type, exc, tb)


# =========================
# Logging filters / formatters
# =========================


class EmptyMessageFilter(logging.Filter):
    """Blocks empty log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            return bool(msg and str(msg).strip())
        except Exception:
            return True


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if not message or not message.strip():
            return ""

        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # ContextVar context
        try:
            ctx = _LOG_CONTEXT.get()
            if ctx:
                log_data.update(_redact(ctx))
        except Exception:
            pass

        # Standard logging extra fields (non-internal)
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in (
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "name",
            ):
                continue
            log_data[key] = _redact(value)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that skips empty formatted records."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if not msg:
                return
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


# =========================
# Logger setup
# =========================


def _resolve_log_level(level: Optional[str | int]) -> int:
    if isinstance(level, int):
        return level
    raw = level or getattr(settings, "LOG_LEVEL", "INFO")
    try:
        return getattr(logging, str(raw).upper())
    except Exception:
        return logging.INFO


def setup_logger(
    name: Optional[str] = None,
    level: Optional[str | int] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure structured logger.
    Idempotent: safe to call multiple times.
    """

    logger_name = name or "app"
    logger = logging.getLogger(logger_name)

    if getattr(logger, "_configured", False):
        return logger

    lvl = _resolve_log_level(level)
    logger.setLevel(lvl)
    logger.propagate = False

    formatter = StructuredFormatter()
    empty_filter = EmptyMessageFilter()

    # Console
    ch = SafeStreamHandler(sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    ch.addFilter(empty_filter)
    logger.addHandler(ch)

    # File handler
    path = log_file or getattr(settings, "LOG_FILE_PATH", None)
    if path:
        max_size = int(getattr(settings, "LOG_MAX_SIZE", 100 * 1024 * 1024))
        backup = int(getattr(settings, "LOG_BACKUP_COUNT", 10))
        fh = RotatingFileHandler(path, maxBytes=max_size, backupCount=backup)
        fh.setLevel(lvl)
        fh.setFormatter(formatter)
        fh.addFilter(empty_filter)
        logger.addHandler(fh)

    # Audit logger
    audit_logger = logging.getLogger("audit")
    if not getattr(audit_logger, "_configured", False):
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False

        audit_path = f"{path}.audit" if path else "audit.log"
        af = RotatingFileHandler(
            audit_path,
            maxBytes=max_size if path else 100 * 1024 * 1024,
            backupCount=backup if path else 10,
        )
        af.setLevel(logging.INFO)
        af.setFormatter(formatter)
        af.addFilter(empty_filter)
        audit_logger.addHandler(af)
        audit_logger._configured = True

    logger._configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return setup_logger(name)


# =========================
# Helper logging utilities
# =========================


def log_with_context(
    log_method: Callable[..., None],
    message: str,
    **context: Any,
) -> None:
    ctx = {}
    try:
        ctx.update(_LOG_CONTEXT.get())
    except Exception:
        pass
    ctx.update(context)
    log_method(message, extra=_redact(ctx))


def audit_event(
    *,
    action: str,
    target_type: str,
    target_id: str,
    admin_id: Optional[str] = None,
    user_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    old_values: Optional[Mapping[str, Any]] = None,
    new_values: Optional[Mapping[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    level: str = "info",
) -> None:
    audit_logger = logging.getLogger("audit")

    payload = {
        "action": action,
        "target_type": target_type,
        "target_id": target_id,
        "admin_id": admin_id,
        "user_id": user_id,
        "details": _redact(details or {}),
        "old_values": _redact(old_values) if old_values else None,
        "new_values": _redact(new_values) if new_values else None,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "success": success,
        "error_message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    msg = json.dumps(payload, default=str)

    if level.lower() == "error":
        audit_logger.error(msg)
    elif level.lower() == "warning":
        audit_logger.warning(msg)
    else:
        audit_logger.info(msg)


# =========================
# Retry utilities
# =========================


def retry_with_backoff(
    func: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (factor**attempt)
            logging.getLogger(func.__module__).warning(
                "Retry %s in %.2fs: %s", attempt + 1, delay, exc
            )
            time.sleep(delay)
    raise last_exc  # type: ignore


async def async_retry_with_backoff(
    func: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = func()
            return await result if inspect.isawaitable(result) else result
        except exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (factor**attempt)
            logging.getLogger(func.__module__).warning(
                "Async retry %s in %.2fs: %s", attempt + 1, delay, exc
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore


# =========================
# Timing / decorators
# =========================


class Timer:
    def __init__(self):
        self._start: Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start


def log_function_call(func: Callable) -> Callable:
    """Decorator for logging function execution with duration and errors."""

    logger = get_logger(func.__module__)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            timer = Timer().start()
            try:
                result = await func(*args, **kwargs)
                log_with_context(
                    logger.info,
                    "Async function executed",
                    function=func.__name__,
                    duration=timer.elapsed,
                    success=True,
                )
                return result
            except Exception as exc:
                log_with_context(
                    logger.error,
                    "Async function failed",
                    function=func.__name__,
                    duration=timer.elapsed,
                    success=False,
                    error=str(exc),
                )
                raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        timer = Timer().start()
        try:
            result = func(*args, **kwargs)
            log_with_context(
                logger.info,
                "Function executed",
                function=func.__name__,
                duration=timer.elapsed,
                success=True,
            )
            return result
        except Exception as exc:
            log_with_context(
                logger.error,
                "Function failed",
                function=func.__name__,
                duration=timer.elapsed,
                success=False,
                error=str(exc),
            )
            raise

    return sync_wrapper
