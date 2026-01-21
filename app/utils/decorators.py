"""
Decorators.
Production-ready async/sync safe decorators.
"""

import asyncio
import functools
import inspect
import time
import uuid
from typing import Callable, Any, Optional, Dict, Tuple

from .logger import get_logger
from .exceptions import (
    ValidationError,
    ProcessingError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitError,
)
from .validators import validate_json_schema

logger = get_logger(__name__)

# =============================================================================
# Helpers
# =============================================================================


def _is_async(func: Callable) -> bool:
    return inspect.iscoroutinefunction(func)


def _make_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    return (
        f"{func.__module__}.{func.__name__}:{repr(args)}:{repr(sorted(kwargs.items()))}"
    )


# =============================================================================
# Validation
# =============================================================================


def validate_input(schema: Optional[Dict[str, Any]] = None):
    """
    Validate kwargs against JSON schema.
    """

    def decorator(func: Callable) -> Callable:
        if _is_async(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if schema:
                    validate_json_schema(kwargs, schema)
                return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if schema:
                validate_json_schema(kwargs, schema)
            return func(*args, **kwargs)

        return sync_wrapper

    return decorator


# =============================================================================
# Logging
# =============================================================================


def log_request(logger_name: Optional[str] = None):
    """
    Logs function execution lifecycle.
    """

    def decorator(func: Callable) -> Callable:
        log = get_logger(logger_name or func.__module__)

        async def _log_async(*args, **kwargs):
            request_id = uuid.uuid4().hex
            start = time.monotonic()

            log.info("Request started", extra={"request_id": request_id})

            try:
                result = await func(*args, **kwargs)
                duration = time.monotonic() - start
                log.info(
                    "Request completed",
                    extra={"request_id": request_id, "duration": duration},
                )
                return result
            except Exception as exc:
                duration = time.monotonic() - start
                log.error(
                    "Request failed",
                    extra={
                        "request_id": request_id,
                        "duration": duration,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                raise

        def _log_sync(*args, **kwargs):
            request_id = uuid.uuid4().hex
            start = time.monotonic()

            log.info("Request started", extra={"request_id": request_id})

            try:
                result = func(*args, **kwargs)
                duration = time.monotonic() - start
                log.info(
                    "Request completed",
                    extra={"request_id": request_id, "duration": duration},
                )
                return result
            except Exception as exc:
                duration = time.monotonic() - start
                log.error(
                    "Request failed",
                    extra={
                        "request_id": request_id,
                        "duration": duration,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                raise

        return functools.wraps(func)(_log_async if _is_async(func) else _log_sync)

    return decorator


# =============================================================================
# Timing
# =============================================================================


def measure_time(threshold: Optional[float] = None):
    """
    Measures execution time.
    """

    def decorator(func: Callable) -> Callable:
        log = get_logger(func.__module__)

        async def _async(*args, **kwargs):
            start = time.monotonic()
            result = await func(*args, **kwargs)
            duration = time.monotonic() - start

            if threshold and duration > threshold:
                log.warning("Slow execution", extra={"duration": duration})
            else:
                log.debug("Execution time", extra={"duration": duration})

            return result

        def _sync(*args, **kwargs):
            start = time.monotonic()
            result = func(*args, **kwargs)
            duration = time.monotonic() - start

            if threshold and duration > threshold:
                log.warning("Slow execution", extra={"duration": duration})
            else:
                log.debug("Execution time", extra={"duration": duration})

            return result

        return functools.wraps(func)(_async if _is_async(func) else _sync)

    return decorator


log_execution_time = measure_time


# =============================================================================
# Retry
# =============================================================================


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
):
    """
    Retry with exponential backoff + jitter.
    """

    def decorator(func: Callable) -> Callable:
        log = get_logger(func.__module__)

        async def _async(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_retries:
                        log.error("Retries exhausted", exc_info=True)
                        raise
                    wait = delay * (backoff**attempt)
                    await asyncio.sleep(wait)

        def _sync(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_retries:
                        log.error("Retries exhausted", exc_info=True)
                        raise
                    time.sleep(delay * (backoff**attempt))

        return functools.wraps(func)(_async if _is_async(func) else _sync)

    return decorator


# =============================================================================
# Cache / Memoization
# =============================================================================


def cache_result(ttl: int = 300):
    """
    In-memory TTL cache (thread-safe).
    """

    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Tuple[Any, float]] = {}
        lock = asyncio.Lock()

        async def _async(*args, **kwargs):
            key = _make_cache_key(func, args, kwargs)
            async with lock:
                if key in cache:
                    value, ts = cache[key]
                    if time.time() - ts < ttl:
                        return value

            result = await func(*args, **kwargs)

            async with lock:
                cache[key] = (result, time.time())

            return result

        def _sync(*args, **kwargs):
            key = _make_cache_key(func, args, kwargs)
            if key in cache:
                value, ts = cache[key]
                if time.time() - ts < ttl:
                    return value

            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        return functools.wraps(func)(_async if _is_async(func) else _sync)

    return decorator


memoize = cache_result


# =============================================================================
# Auth
# =============================================================================


def require_auth(required_role: Optional[str] = None):
    """
    Requires authenticated request with optional role.
    """

    def decorator(func: Callable) -> Callable:
        async def _async(*args, **kwargs):
            request = next((a for a in args if hasattr(a, "state")), None)
            if not request or not getattr(request.state, "user_id", None):
                raise UnauthorizedError("Authentication required")

            if required_role and request.state.user_role != required_role:
                raise ForbiddenError(f"Role {required_role} required")

            return await func(*args, **kwargs)

        def _sync(*args, **kwargs):
            request = next((a for a in args if hasattr(a, "state")), None)
            if not request or not getattr(request.state, "user_id", None):
                raise UnauthorizedError("Authentication required")

            if required_role and request.state.user_role != required_role:
                raise ForbiddenError(f"Role {required_role} required")

            return func(*args, **kwargs)

        return functools.wraps(func)(_async if _is_async(func) else _sync)

    return decorator


# =============================================================================
# Rate limiting
# =============================================================================


def rate_limit(requests_per_minute: int = 60):
    """
    Simple per-function rate limiter.
    """

    def decorator(func: Callable) -> Callable:
        calls = []
        lock = asyncio.Lock()

        async def _async(*args, **kwargs):
            async with lock:
                now = time.time()
                calls[:] = [t for t in calls if t > now - 60]
                if len(calls) >= requests_per_minute:
                    raise RateLimitError("Rate limit exceeded", requests_per_minute)
                calls.append(now)

            return await func(*args, **kwargs)

        def _sync(*args, **kwargs):
            now = time.time()
            calls[:] = [t for t in calls if t > now - 60]
            if len(calls) >= requests_per_minute:
                raise RateLimitError("Rate limit exceeded", requests_per_minute)
            calls.append(now)
            return func(*args, **kwargs)

        return functools.wraps(func)(_async if _is_async(func) else _sync)

    return decorator


# =============================================================================
# Deprecation
# =============================================================================


def deprecated(reason: str = "Deprecated"):
    """
    Marks function as deprecated.
    """

    def decorator(func: Callable) -> Callable:
        async def _async(*args, **kwargs):
            logger.warning("Deprecated call", extra={"reason": reason})
            return await func(*args, **kwargs)

        def _sync(*args, **kwargs):
            logger.warning("Deprecated call", extra={"reason": reason})
            return func(*args, **kwargs)

        return functools.wraps(func)(_async if _is_async(func) else _sync)

    return decorator
