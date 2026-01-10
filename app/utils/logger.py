import logging
import sys
import json
import time
import asyncio
import inspect
import functools
from logging.handlers import RotatingFileHandler
try:
    from pythonjsonlogger import jsonlogger
    JsonFormatterBase = jsonlogger.JsonFormatter
except Exception:
    JsonFormatterBase = logging.Formatter
from typing import Optional, Callable, Any, Dict

from ..config import settings


SENSITIVE_KEYS = {
    "password",
    "passwd",
    "token",
    "access_token",
    "refresh_token",
    "embeddings",
    "image",
    "file",
    "ssn",
    "credit_card",
}


def _redact(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return obj
    redacted: Dict[str, Any] = {}
    for k, v in obj.items():
        if k and k.lower() in SENSITIVE_KEYS:
            redacted[k] = "[REDACTED]"
        else:
            redacted[k] = v
    return redacted


class StructuredFormatter(JsonFormatterBase):
    def add_fields(self, log_record, record, message_dict=None):
        # If the base formatter provides add_fields, try to reuse it.
        if hasattr(super(), "add_fields"):
            try:
                super().add_fields(log_record, record, message_dict or {})
            except Exception:
                pass

        # Ensure basic fields exist regardless of underlying formatter
        if "timestamp" not in log_record:
            try:
                log_record["timestamp"] = self.formatTime(record, self.datefmt) if hasattr(self, "formatTime") else time.time()
            except Exception:
                log_record["timestamp"] = time.time()
        if "level" not in log_record:
            log_record["level"] = record.levelname
        log_record.setdefault("logger", record.name)
        if "message" not in log_record:
            log_record["message"] = record.getMessage()

    def format(self, record):
        # Build a JSON-serializable record dict and emit as JSON string
        log_record: Dict[str, Any] = {}
        try:
            self.add_fields(log_record, record, {})
            return json.dumps(log_record, default=str)
        except Exception:
            # Last-resort fallback to plain text
            try:
                return super().format(record)
            except Exception:
                return str(record)


def _make_rotating_handler(path: str, max_bytes: int, backup_count: int):
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(logging.INFO)
    return handler


def setup_logger(name: Optional[str] = None, level: Optional[str] = None, log_file: Optional[str] = None):
    """Configure and return a structured JSON logger.

    This is idempotent: calling multiple times for the same name won't add duplicate handlers.
    """
    # Accept either numeric (e.g. logging.INFO) or string levels (e.g. 'INFO').
    if isinstance(level, int):
        lvl_value = level
    else:
        raw = (level or getattr(settings, "LOG_LEVEL", "INFO") or "INFO")
        # Coerce to string and uppercase to match logging names, but fall back
        # to INT conversion if provided (e.g. '20').
        try:
            lvl_name = str(raw).upper()
            lvl_value = getattr(logging, lvl_name, None)
            if lvl_value is None:
                lvl_value = int(raw)
        except Exception:
            lvl_value = logging.INFO

    logger_name = name or "app"

    logger = logging.getLogger(logger_name)
    logger.setLevel(lvl_value)

    # If logger already configured, return it
    if logger.handlers:
        return logger

    fmt = StructuredFormatter('%(timestamp)s %(level)s %(name)s %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lvl_value)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler if configured
    path = log_file or getattr(settings, "LOG_FILE_PATH", None)
    if path:
        max_size = int(getattr(settings, "LOG_MAX_SIZE", 10 * 1024 * 1024))
        backup = int(getattr(settings, "LOG_BACKUP_COUNT", 5))
        fh = _make_rotating_handler(path, max_size, backup)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Configure audit logger
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    if not audit_logger.handlers:
        audit_path = f"{path}.audit" if path else "./audit.log"
        try:
            af = _make_rotating_handler(audit_path, int(getattr(settings, "LOG_MAX_SIZE", 10 * 1024 * 1024)), int(getattr(settings, "LOG_BACKUP_COUNT", 5)))
            af.setFormatter(fmt)
            audit_logger.addHandler(af)
        except Exception:
            # Fallback: console-only audit logger
            ach = logging.StreamHandler(sys.stdout)
            ach.setFormatter(fmt)
            audit_logger.addHandler(ach)

    return logger


def get_logger(name: Optional[str] = None):
    logger = setup_logger(name)
    return logging.getLogger(logger.name if isinstance(logger, logging.Logger) else (name or "app"))


# Backwards-compatible helpers expected by other modules/tests
class LoggerMixin:
    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__module__)


class StructuredLogger(logging.Logger):
    pass


def structured_logger(name: Optional[str] = None) -> logging.Logger:
    return get_logger(name)


def configure_logging_for_environment(env: Optional[str] = None):
    # Lightweight configuration hook for tests/environments
    setup_logger()


# Context-based logging helpers used by middleware
import contextvars

_LOG_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("_LOG_CONTEXT", default={})


class LogContext:
    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        self.request_id = request_id
        self.user_id = user_id
        self.extra = extra or {}
        self._token = None

    def __enter__(self):
        ctx = _LOG_CONTEXT.get().copy()
        ctx.update({"request_id": self.request_id, "user_id": self.user_id})
        ctx.setdefault("extra", {})
        ctx["extra"].update(self.extra)
        self._token = _LOG_CONTEXT.set(ctx)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            try:
                _LOG_CONTEXT.reset(self._token)
            except Exception:
                pass

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)


def redact_sensitive_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    sensitive = {"password", "passwd", "token", "access_token", "refresh_token", "embeddings", "image", "file"}
    redacted = {}
    for k, v in payload.items():
        if k and k.lower() in sensitive:
            redacted[k] = "[REDACTED]"
        else:
            redacted[k] = v
    return redacted


def log_with_context(log_method, message, **context):
    """
    Log with additional context.
    
    Args:
        log_method: Logger method (logger.info, logger.error, etc.)
        message: Log message
        **context: Additional context fields
    """
    # Просто вызываем переданный метод
    log_method(message, extra=context)



def audit_event(event_type: str, payload: Dict[str, Any], level: str = "info") -> None:
    """Write an audit event to the `audit` logger as structured JSON.

    Payload will be redacted for known sensitive keys.
    """
    audit_logger = logging.getLogger("audit")
    try:
        data = {
            "event_type": event_type,
            "payload": _redact(payload or {}),
            "timestamp": time.time(),
        }
        # Use audit logger to emit JSON structure
        if level.lower() == "info":
            audit_logger.info(json.dumps(data, default=str))
        else:
            audit_logger.warning(json.dumps(data, default=str))
    except Exception:
        # Best-effort; don't crash the app for audit failures
        audit_logger.exception("Failed to emit audit event")


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (factor ** attempt)
            logging.getLogger(func.__module__).warning("Retry %s in %.2fs: %s", attempt + 1, delay, exc)
            time.sleep(delay)
    raise last_exc


async def async_retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            res = func()
            if inspect.isawaitable(res):
                return await res
            return res
        except exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (factor ** attempt)
            logging.getLogger(func.__module__).warning("Async retry %s in %.2fs: %s", attempt + 1, delay, exc)
            await asyncio.sleep(delay)
    raise last_exc


class Timer:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        self.end_time = time.perf_counter()
        return self

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time

    def __str__(self) -> str:
        return f"{self.elapsed:.4f}s"


def log_function_call(func):
    """Decorator for logging sync/async function calls with duration and errors."""

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(
                    "Async function executed",
                    extra={
                        "function": func.__name__,
                        "duration": duration,
                        "success": True,
                    },
                )
                return result
            except Exception as exc:
                duration = time.perf_counter() - start
                logger.error(
                    "Async function failed",
                    extra={
                        "function": func.__name__,
                        "duration": duration,
                        "success": False,
                        "error": str(exc),
                    },
                    exc_info=True,
                )
                raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.info(
                "Function executed",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "success": True,
                },
            )
            return result
        except Exception as exc:
            duration = time.perf_counter() - start
            logger.error(
                "Function failed",
                extra={
                    "function": func.__name__,
                    "duration": duration,
                    "success": False,
                    "error": str(exc),
                },
                exc_info=True,
            )
            raise

    return sync_wrapper
