# app/config.py
"""
Конфигурация приложения (PostgreSQL Only).
Загрузка настроек из переменных окружения.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator, model_validator, ConfigDict, ValidationInfo
from pydantic_settings import BaseSettings
from pydantic import Field 
import os
import secrets
import warnings

from dotenv import load_dotenv

# Загружаем .env явно
env_path = Path(__file__).parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

# Определяем окружение
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

class Settings(BaseSettings):
    """
    Настройки приложения для PostgreSQL.
    """
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Базовые настройки приложения
    APP_NAME: str = "Face Recognition API"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = Field(default="INFO")

    # CORS настройки
    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8000")

    # ============================================================================
    # PostgreSQL Database
    # ============================================================================
    DATABASE_URL: str = Field(
        default="postgresql://face_user:face_password_2024@localhost:5432/face_recognition"
    )
    DATABASE_POOL_SIZE: int = Field(default=20)
    DATABASE_MAX_OVERFLOW: int = Field(default=40)
    DATABASE_POOL_TIMEOUT: int = Field(default=30)
    DATABASE_POOL_RECYCLE: int = Field(default=3600)

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_CONNECTION_POOL_SIZE: int = 10
    CACHE_KEY_PREFIX: str = "face_recognition:"

    # S3/MinIO Storage
    S3_ENDPOINT_URL: str = Field(default="http://localhost:9000")
    S3_ACCESS_KEY: str = Field(default="minioadmin")
    S3_SECRET_KEY: str = Field(default="minioadmin2024")
    S3_BUCKET_NAME: str = "face-recognition"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = Field(default=False)
    S3_PUBLIC_READ: bool = False

    # MinIO aliases
    MINIO_ENDPOINT: Optional[str] = None
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_KEY: Optional[str] = None
    MINIO_REGION: Optional[str] = None
    MINIO_BUCKET: Optional[str] = None
    MINIO_SSL: Optional[bool] = None

    # Настройки файлов
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_EXPIRATION_DAYS: int = 30
    CLEANUP_INTERVAL_HOURS: int = 24
    STORE_ORIGINAL_IMAGES: bool = False
    DELETE_SOURCE_AFTER_PROCESSING: bool = True

    # ML сервис
    USE_LOCAL_ML_SERVICE: bool = True
    ML_SERVICE_URL: str = "http://localhost:8001"
    ML_SERVICE_TIMEOUT: int = 30
    ML_SERVICE_API_KEY: Optional[str] = None
    LOCAL_ML_DEVICE: str = "auto"
    LOCAL_ML_ENABLE_CUDA: bool = True
    LOCAL_ML_MODEL_CACHE_SIZE: int = 100
    LOCAL_ML_BATCH_SIZE: int = 1
    LOCAL_ML_FACE_DETECTION_THRESHOLD: float = 0.9
    LOCAL_ML_QUALITY_THRESHOLD: float = 0.5
    LOCAL_ML_ENABLE_PERFORMANCE_MONITORING: bool = True
    
    # Liveness
    USE_CERTIFIED_LIVENESS: bool = Field(default=False)
    CERTIFIED_LIVENESS_MODEL_PATH: Optional[str] = Field(default=None)
    CERTIFIED_LIVENESS_THRESHOLD: float = Field(default=0.98)

    # JWT
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Encryption
    ENCRYPTION_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ENCRYPTION_ALGORITHM: str = "aes-256-gcm"

    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    rate_limit_on_redis_failure: str = Field(default="allow")

    # Security
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    ALLOWED_IMAGE_FORMATS: str = "JPEG,JPG,PNG,WEBP,HEIC,HEIF"
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".jpg", ".jpeg", ".png", ".webp"])
    MIN_IMAGE_WIDTH: int = 224
    MIN_IMAGE_HEIGHT: int = 224
    MAX_IMAGE_WIDTH: int = 4096
    MAX_IMAGE_HEIGHT: int = 4096

    # Verification thresholds
    THRESHOLD_DEFAULT: float = 0.80
    THRESHOLD_MIN: float = 0.50
    THRESHOLD_MAX: float = 0.95
    TARGET_FAR: float = 0.001
    TARGET_FRR: float = 0.02

    # Liveness thresholds
    LIVENESS_THRESHOLD: float = 0.5
    LIVENESS_THRESHOLD_MIN: float = 0.3
    LIVENESS_THRESHOLD_MAX: float = 0.9
    LIVENESS_CONFIDENCE_THRESHOLD: float = 0.7
    LIVENESS_ANTI_SPOOFING_THRESHOLD: float = 0.6

    # Webhooks
    WEBHOOK_URL: Optional[str] = None
    WEBHOOK_SECRET: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_MAX_RETRIES: int = 3
    WEBHOOK_RETRY_DELAY: int = 1

    # Notifications
    ENABLE_SLACK_NOTIFICATIONS: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    EMAIL_SMTP_HOST: Optional[str] = None
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None

    # Logging & Audit
    LOG_FORMAT: str = Field(default="json")
    LOG_FILE_PATH: Optional[str] = Field(default=None)
    LOG_MAX_SIZE: int = 100 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 10
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_RETENTION_DAYS: int = 90
    AUDIT_LOG_INCLUDE_REQUEST_BODY: bool = False
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    METRICS_PATH: str = "/metrics"
    HEALTH_CHECK_INTERVAL: int = 30
    REQUEST_ID_HEADER: str = "X-Request-ID"
    GENERATE_REQUEST_ID_IF_NOT_PRESENT: bool = True
  

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS:
            return ["*"] if self.DEBUG else []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def allowed_image_formats_list(self) -> List[str]:
        return [fmt.strip().upper() for fmt in self.ALLOWED_IMAGE_FORMATS.split(",")]

    @property
    def allowed_extensions_list(self) -> List[str]:
        return self.ALLOWED_EXTENSIONS

    @property
    def async_database_url(self) -> str:
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url

    @property
    def sync_database_url(self) -> str:
        url = self.DATABASE_URL
        return url.replace("postgresql+asyncpg://", "postgresql://").replace("postgres+asyncpg://", "postgresql://")

    @property
    def redis_url_with_auth(self) -> str:
        if self.REDIS_PASSWORD:
            parts = self.REDIS_URL.split("://")
            if len(parts) == 2:
                protocol, rest = parts
                if "@" not in rest:
                    return f"{protocol}://:{self.REDIS_PASSWORD}@{rest}"
        return self.REDIS_URL

    @field_validator("LOG_FILE_PATH")
    @classmethod
    def validate_log_path(cls, v: Optional[str]) -> Optional[str]:
        if v:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_postgresql_url(self) -> "Settings":
        if not self.DATABASE_URL.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                f"❌ Invalid DATABASE_URL: must start with 'postgresql://' or 'postgres://'\n"
                f"Got: {self.DATABASE_URL[:50]}...\n"
                f"SQLite is no longer supported. Please use PostgreSQL."
            )
        return self

    @model_validator(mode="after")
    def validate_debug_environment(self) -> "Settings":
        if self.DEBUG and self.ENVIRONMENT == "production":
            raise ValueError("DEBUG mode cannot be enabled in production environment")
        return self

    @model_validator(mode="after")
    def validate_rate_limit_policy(self) -> "Settings":
        """Валидация политики rate limiting при недоступности Redis."""
        allowed = ["allow", "block", "error"]
        v_lower = self.rate_limit_on_redis_failure.lower()  # ← ИСПРАВЛЕНО: lowercase
        if v_lower not in allowed:
            raise ValueError(
                f"rate_limit_on_redis_failure must be one of {allowed}, got: {self.rate_limit_on_redis_failure}"
            )
        self.rate_limit_on_redis_failure = v_lower  # ← ИСПРАВЛЕНО: lowercase
        return self


    @model_validator(mode="after")
    def warn_about_generated_secrets(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            if not os.getenv("JWT_SECRET_KEY"):
                warnings.warn(
                    "⚠️  JWT_SECRET_KEY не задан в .env! "
                    "Токены будут инвалидированы при перезапуске.",
                    UserWarning
                )
            if not os.getenv("ENCRYPTION_KEY"):
                warnings.warn(
                    "⚠️  ENCRYPTION_KEY не задан в .env! "
                    "Биометрические данные будут недоступны при перезапуске.",
                    UserWarning
                )
        return self

    @model_validator(mode="after")
    def setup_minio_aliases(self) -> "Settings":
        if self.MINIO_ENDPOINT is None:
                self.MINIO_ENDPOINT = self.S3_ENDPOINT_URL
        if self.MINIO_ACCESS_KEY is None:
                self.MINIO_ACCESS_KEY = self.S3_ACCESS_KEY
        if self.MINIO_SECRET_KEY is None:
                self.MINIO_SECRET_KEY = self.S3_SECRET_KEY
        if self.MINIO_REGION is None:
                self.MINIO_REGION = self.S3_REGION
        if self.MINIO_BUCKET is None:
                self.MINIO_BUCKET = self.S3_BUCKET_NAME
        if self.MINIO_SSL is None:
                self.MINIO_SSL = self.S3_USE_SSL
        return self


# ============================================================================
# СОЗДАНИЕ ЭКЗЕМПЛЯРА SETTINGS (В КОНЦЕ ФАЙЛА)
# ============================================================================

settings = Settings()

# Минимальное логирование только при прямом запуске модуля
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"🚀 Face Recognition Service - {settings.ENVIRONMENT.upper()} mode")
    logger.info(f"🐘 PostgreSQL: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'configured'}")
    logger.info(f"📦 Redis: {settings.REDIS_URL}")
    logger.info(f"🗄️  MinIO: {settings.S3_ENDPOINT_URL}")
    logger.info(f"🔧 Rate limit policy: {settings.rate_limit_on_redis_failure}")
    logger.info("=" * 80)