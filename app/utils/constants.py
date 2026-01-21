"""
Application constants
Centralized immutable values for validation, security and file detection.
"""

import re

# =============================================================================
# Image formats (canonical names only)
# =============================================================================
IMAGE_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
    "HEIC",
    "HEIF",
    "TIFF",
    "GIF",
}

# Aliases for backward compatibility / user input normalization
IMAGE_FORMAT_ALIASES = {
    "JPG": "JPEG",
    "JPEG": "JPEG",
    "PNG": "PNG",
    "WEBP": "WEBP",
    "BMP": "BMP",
    "HEIC": "HEIC",
    "HEIF": "HEIF",
    "TIF": "TIFF",
    "TIFF": "TIFF",
    "GIF": "GIF",
}

# =============================================================================
# Magic numbers for secure file type detection
# =============================================================================
# NOTE:
# - WEBP requires additional check: header[:4] == b"RIFF" and header[8:12] == b"WEBP"
# - HEIC/HEIF must be detected by presence of ftyp* within first ~32 bytes
MAGIC_NUMBERS = {
    "JPEG": [b"\xFF\xD8\xFF"],  # JPEG SOI
    "PNG": [b"\x89PNG\r\n\x1a\n"],
    "GIF": [b"GIF87a", b"GIF89a"],
    "BMP": [b"BM"],
    "WEBP": [b"RIFF"],  # must also check b"WEBP" at offset 8
    "HEIC": [b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1", b"ftypmsf1"],
    "TIFF": [b"II*\x00", b"MM\x00*"],
}

# =============================================================================
# File limits
# =============================================================================
FILE_LIMITS = {
    "max_image_size": 10 * 1024 * 1024,  # 10 MB
    "max_filename_length": 255,
    "min_embedding_size": 128,
    "max_embedding_size": 2048,
}

# =============================================================================
# Similarity limits for face verification
# =============================================================================
SIMILARITY_LIMITS = {
    "min_threshold": 0.0,
    "max_threshold": 1.0,
    "default_threshold": 0.6,
}

# =============================================================================
# Confidence levels for face verification
# =============================================================================
CONFIDENCE_LEVELS = {
    "very_high": 0.85,  # Very confident match
    "high": 0.75,  # High confidence
    "medium": 0.65,  # Medium confidence
    "low": 0.55,  # Low confidence
    "very_low": 0.45,  # Very low confidence
}

# =============================================================================
# Regex patterns for validation
# =============================================================================
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")

# Strong but UX-friendly password policy
PASSWORD_REGEX = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9])[^\s]{8,128}$"
)

# =============================================================================
# User roles
# =============================================================================
USER_ROLES = {
    "USER": "user",
    "ADMIN": "admin",
    "SYSTEM": "system",
}

# =============================================================================
# Time periods in seconds
# =============================================================================
TIME_PERIODS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
}
