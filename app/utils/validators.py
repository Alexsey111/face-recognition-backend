"""
Валидаторы данных.
Проверка корректности входных данных и форматов.
"""

import re
import os
import uuid
import base64
import hashlib
import html
import math
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from .constants import (
    IMAGE_FORMATS,
    FILE_LIMITS,
    MAGIC_NUMBERS,
    SIMILARITY_LIMITS,
    PASSWORD_REGEX,
    EMAIL_REGEX,
    USERNAME_REGEX,
)
from .exceptions import ValidationError


# =============================================================================
# BASIC VALIDATORS
# =============================================================================


def validate_email(email: str) -> bool:
    if not isinstance(email, str) or not email:
        raise ValidationError("Email is required")

    if len(email) > 255:
        raise ValidationError("Email is too long (max 255 characters)")

    local_part = email.split("@")[0]
    if ".." in local_part:
        raise ValidationError("Invalid email format")

    if not EMAIL_REGEX.match(email):
        raise ValidationError("Invalid email format")

    return True


def validate_username(username: str) -> bool:
    if not isinstance(username, str) or not username:
        raise ValidationError("Username is required")

    if not 3 <= len(username) <= 50:
        raise ValidationError("Username length must be 3–50 characters")

    if not USERNAME_REGEX.match(username):
        raise ValidationError(
            "Username may contain letters, numbers, underscores and hyphens only"
        )

    return True


def validate_password(password: str) -> bool:
    if not isinstance(password, str) or not password:
        raise ValidationError("Password is required")

    if not 8 <= len(password) <= 128:
        raise ValidationError("Password length must be 8–128 characters")

    if not PASSWORD_REGEX.match(password):
        raise ValidationError(
            "Password must contain upper, lower, digit and special character"
        )

    return True


def validate_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except Exception:
        raise ValidationError("Invalid UUID format")


def validate_date(date_string: str, fmt: str = "%Y-%m-%d") -> bool:
    try:
        datetime.strptime(date_string, fmt)
        return True
    except Exception:
        raise ValidationError(f"Invalid date format, expected {fmt}")


def validate_url(url: str) -> bool:
    if not isinstance(url, str):
        raise ValidationError("URL must be a string")

    pattern = re.compile(
        r"^https?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,}|"
        r"localhost|"
        r"\d{1,3}(?:\.\d{1,3}){3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    if not pattern.match(url):
        raise ValidationError("Invalid URL format")

    return True


# =============================================================================
# IMAGE VALIDATION
# =============================================================================


def validate_image_format(image_data: Union[str, bytes]) -> bool:
    detected = _detect_image_format(image_data)

    if detected not in IMAGE_FORMATS:
        raise ValidationError(
            f"Unsupported image format: {detected}. "
            f"Supported: {', '.join(IMAGE_FORMATS)}"
        )

    return True


def validate_image_size(
    image_data: Union[str, bytes],
    max_size: int = FILE_LIMITS["max_image_size"],
) -> bool:
    try:
        if isinstance(image_data, bytes):
            size = len(image_data)
        else:
            if image_data.startswith("data:image/"):
                _, b64 = image_data.split(",", 1)
            else:
                b64 = image_data
            size = len(base64.b64decode(b64))
    except Exception:
        raise ValidationError("Invalid image data")

    if size > max_size:
        raise ValidationError(
            f"Image is too large: {size / 1024 / 1024:.2f}MB "
            f"(max {max_size / 1024 / 1024:.2f}MB)"
        )

    return True


# =============================================================================
# FILE / HASH
# =============================================================================


def validate_file_hash(
    data: Union[str, bytes],
    expected_hash: str,
    algorithm: str = "sha256",
) -> bool:
    if algorithm not in {"sha256", "md5"}:
        raise ValidationError("Unsupported hash algorithm")

    hash_obj = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()

    if isinstance(data, str):
        try:
            if data.startswith("data:image/"):
                _, data = data.split(",", 1)
            data = base64.b64decode(data)
        except Exception:
            raise ValidationError("Invalid data for hashing")

    hash_obj.update(data)

    if hash_obj.hexdigest().lower() != expected_hash.lower():
        raise ValidationError("Hash mismatch")

    return True


def validate_file_upload(
    filename: str,
    content_type: str,
    file_size: int,
) -> bool:
    if not filename or not isinstance(filename, str):
        raise ValidationError("Filename is required")

    if len(filename) > FILE_LIMITS["max_filename_length"]:
        raise ValidationError("Filename too long")

    if not content_type.startswith("image/"):
        raise ValidationError("Only image uploads are allowed")

    if file_size <= 0 or file_size > FILE_LIMITS["max_image_size"]:
        raise ValidationError("Invalid file size")

    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    if os.path.splitext(filename.lower())[1] not in allowed_ext:
        raise ValidationError("Unsupported file extension")

    return True


# =============================================================================
# ML / EMBEDDINGS
# =============================================================================


def validate_embedding(embedding: List[float]) -> bool:
    if not isinstance(embedding, (list, tuple)):
        raise ValidationError("Embedding must be list or tuple")

    size = len(embedding)
    if (
        not FILE_LIMITS["min_embedding_size"]
        <= size
        <= FILE_LIMITS["max_embedding_size"]
    ):
        raise ValidationError(f"Invalid embedding size: {size}")

    for i, v in enumerate(embedding):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValidationError(f"Non-numeric value at index {i}")
        if math.isnan(v) or math.isinf(v):
            raise ValidationError(f"Invalid float at index {i}")

    return True


def validate_similarity_threshold(threshold: float) -> bool:
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValidationError("Threshold must be numeric")

    if (
        not SIMILARITY_LIMITS["min_threshold"]
        <= threshold
        <= SIMILARITY_LIMITS["max_threshold"]
    ):
        raise ValidationError("Threshold out of bounds")

    return True


# =============================================================================
# SECURITY / SANITIZATION
# =============================================================================


def sanitize_string(
    text: str,
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None,
) -> str:
    if not text:
        return ""

    if allowed_chars:
        text = "".join(c for c in text if c in allowed_chars)
    else:
        text = re.sub(r'[<>"\']', "", text)

    return text[:max_length].strip() if max_length else text.strip()


def sanitize_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return html.escape(text, quote=True)


def validate_sql_safe(text: str) -> bool:
    if not isinstance(text, str):
        return True

    patterns = [
        r"(\bOR\b|\bAND\b).*=.*",
        r";\s*(DROP|DELETE|INSERT|UPDATE|SELECT|ALTER)",
        r"--",
        r"/\*.*\*/",
        r"UNION\s+SELECT",
        r"INFORMATION_SCHEMA",
        r"XP_CMDSHELL",
    ]

    upper = text.upper()
    for p in patterns:
        if re.search(p, upper):
            raise ValidationError("Potential SQL injection detected")

    return True


# =============================================================================
# JSON SCHEMA VALIDATION
# =============================================================================


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Простая валидация JSON схемы для декораторов.
    Упрощенная версия для базовых случаев.
    """
    # Базовые проверки типов
    for key, expected_type in schema.items():
        if key not in data:
            raise ValidationError(f"Missing required field: {key}")

        actual_value = data[key]

        # Обработка Union типов
        if isinstance(expected_type, list):
            if not any(isinstance(actual_value, t) for t in expected_type):
                raise ValidationError(f"Field {key} has invalid type")
            continue

        # Обработка Optional типов (Union с None)
        if isinstance(expected_type, type) and expected_type.__name__ == "Union":
            continue

        # Простая проверка типа
        if not isinstance(actual_value, expected_type):
            raise ValidationError(
                f"Field {key} must be of type {expected_type.__name__}"
            )

    return True


# =============================================================================
# HELPERS
# =============================================================================


def _detect_image_format(image_data: Union[str, bytes]) -> str:
    try:
        if isinstance(image_data, bytes):
            return _detect_by_magic(image_data)

        if "." in image_data:
            ext = image_data.rsplit(".", 1)[-1].upper()
            return {
                "JPG": "JPEG",
                "JPEG": "JPEG",
                "PNG": "PNG",
                "WEBP": "WEBP",
                "GIF": "GIF",
                "BMP": "BMP",
                "HEIC": "HEIC",
                "HEIF": "HEIC",
            }.get(ext, "UNKNOWN")

        if image_data.startswith("data:image/"):
            mime = image_data.split(";")[0].split("/")[1].upper()
            return mime if mime in IMAGE_FORMATS else "UNKNOWN"

        decoded = base64.b64decode(image_data)
        return _detect_by_magic(decoded)

    except Exception:
        return "UNKNOWN"


def _detect_by_magic(data: bytes) -> str:
    for fmt, signatures in MAGIC_NUMBERS.items():
        if any(data.startswith(sig) for sig in signatures):
            return fmt

    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"

    return "UNKNOWN"


# =============================================================================
# CLASS-BASED VALIDATORS (for backward compatibility)
# =============================================================================


class Validators:
    """
    Класс-обертка для всех валидаторов.
    Предоставляет статические методы для валидации данных.
    """

    # Basic validators
    validate_email = staticmethod(validate_email)
    validate_username = staticmethod(validate_username)
    validate_password = staticmethod(validate_password)
    validate_uuid = staticmethod(validate_uuid)
    validate_date = staticmethod(validate_date)
    validate_url = staticmethod(validate_url)

    # Image validation
    validate_image_format = staticmethod(validate_image_format)
    validate_image_size = staticmethod(validate_image_size)

    # File validation
    validate_file_hash = staticmethod(validate_file_hash)
    validate_file_upload = staticmethod(validate_file_upload)

    # ML / Embeddings
    validate_embedding = staticmethod(validate_embedding)
    validate_similarity_threshold = staticmethod(validate_similarity_threshold)

    # Security
    sanitize_string = staticmethod(sanitize_string)
    sanitize_html = staticmethod(sanitize_html)
    validate_sql_safe = staticmethod(validate_sql_safe)

    # JSON Schema
    validate_json_schema = staticmethod(validate_json_schema)
