"""
Helper функции.
Общие вспомогательные функции для всего приложения.
"""

import uuid
import time
import base64
import hashlib
import hmac
import secrets
import re
from typing import Any, Dict, Optional, List, Union, Callable
from datetime import datetime, timezone
from pathlib import Path

from .logger import get_logger
from .exceptions import (
    ValidationError,
    ProcessingError,
    RetryExhaustedError,
)

logger = get_logger(__name__)


# ============================================================
# ID / REQUEST HELPERS
# ============================================================

def generate_unique_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex
    return f"{prefix}_{uid}" if prefix else uid


def generate_request_id() -> str:
    timestamp = int(time.time() * 1_000_000)
    return f"req_{timestamp}_{secrets.token_hex(4)}"


# ============================================================
# FILE / SIZE HELPERS
# ============================================================

def format_file_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0

    while size_bytes >= 1024 and index < len(units) - 1:
        size_bytes /= 1024
        index += 1

    return f"{size_bytes:.1f} {units[index]}"


def sanitize_filename(filename: str) -> str:
    if not filename:
        return "unnamed_file"

    filename = Path(filename).name
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    if len(filename) > 255:
        stem = Path(filename).stem[:240]
        suffix = Path(filename).suffix
        filename = f"{stem}{suffix}"

    return filename.strip()


# ============================================================
# IMAGE / BASE64 HELPERS
# ============================================================

def validate_image_base64(image_data: str) -> None:
    if not image_data:
        raise ValidationError("Empty base64 image")

    try:
        if image_data.startswith("data:image/"):
            _, payload = image_data.split(",", 1)
            base64.b64decode(payload, validate=True)
        else:
            base64.b64decode(image_data, validate=True)
    except Exception as e:
        raise ValidationError(
            message="Invalid base64 image",
            details={"error": str(e)},
        )


def extract_image_metadata(image_data: str) -> Dict[str, Any]:
    try:
        validate_image_base64(image_data)

        if image_data.startswith("data:image/"):
            header, payload = image_data.split(",", 1)
            mime = header.split(";")[0].split("/")[1].lower()
            binary = base64.b64decode(payload)
        else:
            mime = "unknown"
            binary = base64.b64decode(image_data)

        return {
            "format": mime.upper(),
            "size_bytes": len(binary),
            "hash": hashlib.sha256(binary).hexdigest(),
            "is_valid": True,
        }

    except Exception as e:
        logger.warning("Failed to extract image metadata", exc_info=True)
        return {
            "is_valid": False,
            "error": str(e),
        }


def calculate_similarity_score(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Returns a float in range [-1, 1]. If either vector has zero magnitude, returns 0.0.
    """
    try:
        if not a or not b:
            return 0.0
        if len(a) != len(b):
            raise ValueError("Vectors must be same length")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception as e:
        logger.warning("Failed to calculate similarity score: %s", e)
        return 0.0


# ============================================================
# HASHING / SECURITY
# ============================================================

def calculate_file_hash(
    data: Union[str, bytes],
    algorithm: str = "sha256",
) -> str:
    if isinstance(data, str):
        try:
            if data.startswith("data:image/"):
                _, payload = data.split(",", 1)
                data = base64.b64decode(payload)
            else:
                data = base64.b64decode(data)
        except Exception:
            data = data.encode("utf-8")

    algorithm = algorithm.lower()

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValidationError(f"Unsupported hash algorithm: {algorithm}")

    hasher.update(data)
    return hasher.hexdigest()


def verify_hmac_signature(
    payload: bytes,
    signature: str,
    secret: str,
    algorithm: str = "sha256",
) -> bool:
    digest = hmac.new(
        secret.encode(),
        payload,
        getattr(hashlib, algorithm),
    ).hexdigest()

    return hmac.compare_digest(digest, signature)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_email_format(email: str) -> None:
    if not email or not isinstance(email, str):
        raise ValidationError("Invalid email")

    pattern = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )

    if not pattern.match(email):
        raise ValidationError(
            message="Invalid email format",
            field="email",
            value=email,
        )


def validate_threshold(
    threshold: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> None:
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Threshold must be numeric")

    if not (min_val <= threshold <= max_val):
        raise ValidationError(
            message="Threshold out of bounds",
            details={
                "value": threshold,
                "min": min_val,
                "max": max_val,
            },
        )


# ============================================================
# API RESPONSE
# ============================================================

def create_api_response(
    success: bool,
    data: Any = None,
    message: Optional[str] = None,
    error_code: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    response = {
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id or generate_request_id(),
    }

    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    if error_code:
        response["error_code"] = error_code

    return response


# ============================================================
# RETRY / BATCH / TIMING
# ============================================================

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_exceptions: tuple = (Exception,),
):
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retry_exceptions as e:
            last_error = e

            if attempt >= max_retries:
                break

            delay = base_delay * (backoff_factor ** attempt)
            logger.warning(
                "Retry attempt failed",
                extra={
                    "attempt": attempt + 1,
                    "delay": delay,
                    "error": str(e),
                },
            )
            time.sleep(delay)

    raise RetryExhaustedError(
        message="All retry attempts exhausted",
        retry_count=max_retries,
    ) from last_error


def batch_process(
    items: List[Any],
    batch_size: int,
    processor: Callable[[List[Any]], List[Any]],
) -> List[Any]:
    results: List[Any] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        results.extend(processor(batch))

    return results


class Timer:
    """Контекстный таймер"""

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> "Timer":
        self._start = time.time()
        return self

    def stop(self) -> "Timer":
        self._end = time.time()
        return self

    @property
    def elapsed(self) -> float:
        if not self._start:
            return 0.0
        end = self._end or time.time()
        return end - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.3f}s"


# ============================================================
# CACHE KEYS
# ============================================================

class CacheKeyGenerator:
    """Генератор ключей кэша"""

    @staticmethod
    def user_references(user_id: str) -> str:
        return f"user:{user_id}:references"

    @staticmethod
    def user_session(user_id: str, session_id: str) -> str:
        return f"user:{user_id}:session:{session_id}"

    @staticmethod
    def ml_embedding(image_hash: str) -> str:
        return f"ml:embedding:{image_hash}"

    @staticmethod
    def liveness_check(session_id: str) -> str:
        return f"liveness:{session_id}"


def create_thumbnail_key(original_key: str) -> str:
    """Generate a thumbnail cache/object key for an original image key."""
    return f"{original_key}:thumbnail"


def mask_sensitive_data(obj: Any) -> Any:
    """Redact common sensitive fields in a dict-like object."""
    if not isinstance(obj, dict):
        return obj
    keys_to_redact = {"password", "passwd", "token", "access_token", "refresh_token", "ssn", "credit_card"}
    redacted = {}
    for k, v in obj.items():
        if k and k.lower() in keys_to_redact:
            redacted[k] = "[REDACTED]"
        else:
            redacted[k] = v
    return redacted


def parse_user_agent(ua_string: Optional[str]) -> Dict[str, str]:
    """Very small UA parser returning basic fields."""
    if not ua_string:
        return {"browser": "unknown", "os": "unknown"}
    ua = ua_string.lower()
    browser = "unknown"
    if "chrome" in ua and "edg" not in ua:
        browser = "chrome"
    elif "firefox" in ua:
        browser = "firefox"
    elif "safari" in ua and "chrome" not in ua:
        browser = "safari"
    elif "edg" in ua or "edge" in ua:
        browser = "edge"

    os = "unknown"
    if "windows" in ua:
        os = "windows"
    elif "mac os" in ua or "macintosh" in ua:
        os = "mac"
    elif "linux" in ua:
        os = "linux"

    return {"browser": browser, "os": os}


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "0s"
    try:
        return f"{seconds:.3f}s"
    except Exception:
        return str(seconds)
