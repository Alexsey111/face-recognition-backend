"""Application constants"""
import re

# Image formats
IMAGE_FORMATS = ["JPEG", "JPG", "PNG", "WEBP", "GIF", "BMP", "HEIC", "HEIF"]

# File limits
FILE_LIMITS = {
    "max_image_size": 10 * 1024 * 1024,  # 10MB
    "max_embedding_size": 1024,
    "min_embedding_size": 128,
    "max_filename_length": 255
}

# Similarity thresholds
SIMILARITY_LIMITS = {
    "min_threshold": 0.0,
    "max_threshold": 1.0,
    "default_threshold": 0.75
}

# ML and Face Recognition constants
EPSILON = 1e-7  # Small value for numerical stability in ML calculations
CONFIDENCE_LEVELS = {
    "LOW": 0.6,
    "MEDIUM": 0.75,
    "HIGH": 0.85,
    "VERY_HIGH": 0.95
}

# Rate limits
RATE_LIMITS = {
    "default_requests_per_minute": 60,
    "default_burst_size": 10,
    "login_requests_per_minute": 5,
    "upload_requests_per_minute": 20
}

# Magic numbers for file format detection
MAGIC_NUMBERS = {
    "JPEG": [
        b"\xFF\xD8\xFF\xE0",  # JFIF
        b"\xFF\xD8\xFF\xE1",  # EXIF
        b"\xFF\xD8\xFF\xE2",  # Canon
        b"\xFF\xD8\xFF\xE3",  # Samsung
    ],
    "PNG": [b"\x89PNG\r\n\x1a\n"],
    "GIF": [b"GIF87a", b"GIF89a"],
    "BMP": [b"BM"],
    "WEBP": [b"RIFF"],  # Проверяется отдельно
    "HEIC": [b"\x00\x00\x00\x18ftypheic", b"\x00\x00\x00\x18ftypheix"],
    "TIFF": [b"II*\x00", b"MM\x00*"],
}

# Regex patterns
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")
PASSWORD_REGEX = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&\-_.+#])[A-Za-z\d@$!%*?&\-_.+#]{8,128}$"
)

# Security
SECURITY_CONFIG = {
    "password_min_length": 8,
    "password_max_length": 128,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 15,
    "session_timeout_minutes": 30,
    "token_expire_minutes": 30,
    "refresh_token_expire_days": 7
}

# CORS
CORS_CONFIG = {
    "allow_origins": ["http://localhost:3000", "http://localhost:8000"],
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    "allow_headers": ["*"],
    "expose_headers": ["X-Total-Count", "X-Rate-Limit-Limit"],
    "max_age": 86400
}

# Logging levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Time formats
TIME_FORMATS = {
    "iso": "%Y-%m-%dT%H:%M:%S",
    "log": "%Y-%m-%d %H:%M:%S",
    "file": "%Y%m%d_%H%M%S"
}
