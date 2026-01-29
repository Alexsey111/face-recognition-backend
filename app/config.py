# app/config.py
"""
Конфигурация приложения (PostgreSQL Only).
Загрузка настроек из переменных окружения.
"""

import os
import secrets
import warnings
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings

# Загружаем .env явно
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Определяем окружение
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


class Settings(BaseSettings):
    """
    Настройки приложения для PostgreSQL с полной поддержкой Anti-Spoofing.
    """

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ============================================================================
    # Базовые настройки приложения
    # ============================================================================
    APP_NAME: str = "Face Recognition API"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = Field(default="INFO")

    # ============================================================================
    # CORS настройки
    # ============================================================================
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

    # ============================================================================
    # Redis
    # ============================================================================
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_CONNECTION_POOL_SIZE: int = 10
    CACHE_KEY_PREFIX: str = "face_recognition:"

    # ============================================================================
    # S3/MinIO Storage
    # ============================================================================
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

    # ============================================================================
    # Настройки файлов
    # ============================================================================
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_EXPIRATION_DAYS: int = 30
    CLEANUP_INTERVAL_HOURS: int = 24
    STORE_ORIGINAL_IMAGES: bool = False
    DELETE_SOURCE_AFTER_PROCESSING: bool = True

    # ============================================================================
    # Политика хранения биометрических данных (GDPR/FZ-152 Compliance)
    # ============================================================================
    # Сроки хранения в днях (если не указано иное)
    BIOMETRIC_RETENTION_DAYS: int = 1095  # 3 года для биометрических шаблонов
    BIOMETRIC_INACTIVITY_DAYS: int = 1095  # 3 года неактивности перед удалением
    RAW_PHOTO_RETENTION_DAYS: int = 30  # 30 дней для исходных фото
    AUDIT_LOG_RETENTION_DAYS: int = 365  # 1 год для аудит логов
    WEBHOOK_LOG_RETENTION_DAYS: int = 30  # 30 дней для webhook логов

    # GDPR Compliance: Право на удаление данных (Right to be Forgotten)
    ENABLE_GDPR_AUTO_DELETE: bool = True
    GDPR_DELETE_BATCH_SIZE: int = 100

    # Политика хранения для корпоративных клиентов (расширенная)
    CORPORATE_RAW_PHOTO_RETENTION_DAYS: int = 90
    CORPORATE_BIOMETRIC_RETENTION_DAYS: int = 1825  # 5 лет

    # ============================================================================
    # ML Service Configuration
    # ============================================================================
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

    # ============================================================================
    # Anti-Spoofing (MiniFASNetV2) Configuration
    # ============================================================================

    # Enable/Disable certified liveness detection
    # ИЗМЕНЕНО: False по умолчанию для обратной совместимости
    USE_CERTIFIED_LIVENESS: bool = Field(
        default=False,
        description="Enable certified liveness detection with MiniFASNetV2",
    )

    # Model path - должен указывать на файл .pth
    # ИЗМЕНЕНО: Optional для обратной совместимости
    CERTIFIED_LIVENESS_MODEL_PATH: Optional[str] = Field(
        default="models/minifasnet_v2.pth",
        description="Path to MiniFASNetV2 model file (.pth)",
    )

    # Classification threshold (Real vs Spoof)
    # 0.5 = balanced, >0.5 = more strict (lower FAR, higher FRR)
    CERTIFIED_LIVENESS_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for real/spoof classification (0-1)",
    )

    # Expected model version for validation
    EXPECTED_MODEL_VERSION: str = Field(default="v2.0.1")

    # Input configuration for MiniFASNetV2
    ANTISPOOFING_INPUT_WIDTH: int = Field(default=80)
    ANTISPOOFING_INPUT_HEIGHT: int = Field(default=80)

    # Normalization parameters
    # Standard MiniFASNet normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # This normalizes images to [-1, 1] range
    ANTISPOOFING_NORMALIZE_MEAN: List[float] = Field(default=[0.5, 0.5, 0.5])
    ANTISPOOFING_NORMALIZE_STD: List[float] = Field(default=[0.5, 0.5, 0.5])

    # Alternative: ImageNet normalization (uncomment if needed)
    # ANTISPOOFING_NORMALIZE_MEAN: List[float] = Field(default=[0.485, 0.456, 0.406])
    # ANTISPOOFING_NORMALIZE_STD: List[float] = Field(default=[0.229, 0.224, 0.225])

    # Auxiliary checks configuration
    ANTISPOOFING_ENABLE_AUXILIARY_CHECKS: bool = Field(
        default=True,
        description="Enable additional spoofing indicators analysis (moiré, texture, etc.)",
    )

    # Performance targets (for monitoring)
    ANTISPOOFING_TARGET_INFERENCE_TIME_CPU: float = Field(default=0.1)  # 100ms
    ANTISPOOFING_TARGET_INFERENCE_TIME_GPU: float = Field(default=0.02)  # 20ms

    # Model auto-download configuration
    ANTISPOOFING_AUTO_DOWNLOAD_MODEL: bool = Field(
        default=True, description="Automatically download model if not found"
    )
    ANTISPOOFING_MODEL_DOWNLOAD_URL: str = Field(
        default="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    )

    # ============================================================================
    # General Liveness Configuration (compatibility with old settings)
    # ============================================================================
    LIVENESS_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    LIVENESS_THRESHOLD_MIN: float = Field(default=0.3, ge=0.0, le=1.0)
    LIVENESS_THRESHOLD_MAX: float = Field(default=0.9, ge=0.0, le=1.0)
    LIVENESS_CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    LIVENESS_ANTI_SPOOFING_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)

    # ============================================================================
    # JWT
    # ============================================================================
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ============================================================================
    # Encryption
    # ============================================================================
    ENCRYPTION_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ENCRYPTION_ALGORITHM: str = "aes-256-gcm"

    # ============================================================================
    # Rate limiting
    # ============================================================================
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    rate_limit_on_redis_failure: str = Field(default="allow")

    # ============================================================================
    # Security
    # ============================================================================
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    ALLOWED_IMAGE_FORMATS: str = "JPEG,JPG,PNG,WEBP,HEIC,HEIF"
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".jpg", ".jpeg", ".png", ".webp"])
    MIN_IMAGE_WIDTH: int = 224
    MIN_IMAGE_HEIGHT: int = 224
    MAX_IMAGE_WIDTH: int = 4096
    MAX_IMAGE_HEIGHT: int = 4096

    # ============================================================================
    # Face Verification Thresholds (Accuracy Requirements: FAR<0.1%, FRR<1-3%)
    # ============================================================================
    # Основной порог верификации (cosine similarity)
    # Рекомендуемые значения:
    # - 0.60-0.70: Высокий FRR (много отказов), низкий FAR
    # - 0.70-0.80: Баланс (рекомендуется для production)
    # - 0.80-0.90: Низкий FRR (много совпадений), высокий FAR
    VERIFICATION_THRESHOLD: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Порог cosine similarity для верификации. "
        "Рекомендуется: 0.75 (баланс FAR/FRR)",
    )

    # Дополнительные пороги для уровней уверенности
    CONFIDENCE_LOW: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Низкий уровень совпадения (требуется повторная проверка)",
    )
    CONFIDENCE_MEDIUM: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Средний уровень совпадения (стандартная верификация)",
    )
    CONFIDENCE_HIGH: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Высокий уровень совпадения (высокая уверенность)",
    )

    # Целевые показатели точности (из ТЗ)
    # FAR (False Accept Rate) - доля ложных срабатываний
    TARGET_FAR: float = Field(
        default=0.001,  # 0.1%
        ge=0.0,
        le=1.0,
        description="Целевой False Accept Rate (0.1% = 1/1000)",
    )

    # FRR (False Reject Rate) - доля ложных отказов
    TARGET_FRR: float = Field(
        default=0.03,  # 3%
        ge=0.0,
        le=1.0,
        description="Целевой False Reject Rate (1-3% = 1-3/100)",
    )

    # Расчётный порог на основе FAR/FRR
    # Для FaceNet (casia-webface):
    # - При FAR=0.1% порог ~0.65-0.70
    # - При FAR=0.01% порог ~0.75-0.80
    # - При FAR=0.001% порог ~0.85-0.90
    CALCULATED_THRESHOLD_FAR_001: float = Field(
        default=0.65,
        description="Порог для FAR=0.1% (высокая безопасность)",
    )
    CALCULATED_THRESHOLD_FAR_0001: float = Field(
        default=0.75,
        description="Порог для FAR=0.01% (баланс)",
    )
    CALCULATED_THRESHOLD_FAR_00001: float = Field(
        default=0.85,
        description="Порог для FAR=0.001% (низкая FRR)",
    )

    # Евклидово расстояние пороги (альтернативный метод)
    EUCLIDEAN_THRESHOLD_DEFAULT: float = Field(
        default=1.0,
        ge=0.0,
        description="Максимальное евклидово расстояние для совпадения (L2 norm)",
    )

    # ============================================================================
    # Legacy Verification thresholds (backward compatibility)
    # ============================================================================
    THRESHOLD_DEFAULT: float = 0.80
    THRESHOLD_MIN: float = 0.50
    THRESHOLD_MAX: float = 0.95

    # ============================================================================
    # Webhooks
    # ============================================================================
    WEBHOOK_URL: Optional[str] = None
    WEBHOOK_SECRET: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_MAX_RETRIES: int = 3
    WEBHOOK_RETRY_DELAY: int = 1

    # ============================================================================
    # Notifications
    # ============================================================================
    ENABLE_SLACK_NOTIFICATIONS: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    EMAIL_SMTP_HOST: Optional[str] = None
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None

    # ============================================================================
    # Logging & Audit
    # ============================================================================
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

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS:
            return ["*"] if self.DEBUG else []
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

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
        return url.replace("postgresql+asyncpg://", "postgresql://").replace(
            "postgres+asyncpg://", "postgresql://"
        )

    @property
    def redis_url_with_auth(self) -> str:
        if self.REDIS_PASSWORD:
            parts = self.REDIS_URL.split("://")
            if len(parts) == 2:
                protocol, rest = parts
                if "@" not in rest:
                    return f"{protocol}://:{self.REDIS_PASSWORD}@{rest}"
        return self.REDIS_URL

    @property
    def antispoofing_model_path_resolved(self) -> Path:
        """Resolve model path relative to project root."""
        if self.CERTIFIED_LIVENESS_MODEL_PATH is None:
            return Path("models/minifasnet_v2.pth")  # Fallback path

        model_path = Path(self.CERTIFIED_LIVENESS_MODEL_PATH)
        if not model_path.is_absolute():
            # Resolve relative to project root
            project_root = Path(__file__).parent.parent
            model_path = project_root / model_path
        return model_path

    # ============================================================================
    # Validators
    # ============================================================================

    @field_validator("LOG_FILE_PATH")
    @classmethod
    def validate_log_path(cls, v: Optional[str]) -> Optional[str]:
        if v:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("ANTISPOOFING_NORMALIZE_MEAN", "ANTISPOOFING_NORMALIZE_STD")
    @classmethod
    def validate_normalization_params(cls, v: List[float]) -> List[float]:
        """Validate normalization parameters."""
        if len(v) != 3:
            raise ValueError(
                "Normalization parameters must have exactly 3 values (RGB)"
            )
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Normalization parameters must be numeric")
        return v

    @field_validator("CERTIFIED_LIVENESS_THRESHOLD")
    @classmethod
    def validate_liveness_threshold(cls, v: float) -> float:
        """Validate liveness threshold range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("CERTIFIED_LIVENESS_THRESHOLD must be between 0.0 and 1.0")
        if v > 0.9:
            warnings.warn(
                f"⚠️  CERTIFIED_LIVENESS_THRESHOLD is very high ({v}). "
                "This may result in high False Rejection Rate (FRR). "
                "Recommended range: 0.4-0.6",
                UserWarning,
            )
        return v

    @model_validator(mode="after")
    def validate_antispoofing_config(self) -> "Settings":
        """Validate anti-spoofing configuration with graceful degradation."""

        # Проверяем только если функция включена
        if not self.USE_CERTIFIED_LIVENESS:
            return self

        # Если путь None, отключаем anti-spoofing
        if self.CERTIFIED_LIVENESS_MODEL_PATH is None:
            warnings.warn(
                "⚠️  USE_CERTIFIED_LIVENESS=True but CERTIFIED_LIVENESS_MODEL_PATH is None. "
                "Anti-spoofing will be disabled.",
                UserWarning,
            )
            self.USE_CERTIFIED_LIVENESS = False
            return self

        # Check model path
        model_path = self.antispoofing_model_path_resolved

        if not model_path.exists():
            if self.ANTISPOOFING_AUTO_DOWNLOAD_MODEL:
                warnings.warn(
                    f"⚠️  Anti-spoofing model not found at: {model_path}\n"
                    f"It will be automatically downloaded on first use.",
                    UserWarning,
                )
            else:
                warnings.warn(
                    f"⚠️  Anti-spoofing model not found at: {model_path}\n"
                    f"Please download it manually or set ANTISPOOFING_AUTO_DOWNLOAD_MODEL=True",
                    UserWarning,
                )

        # Validate input dimensions
        if self.ANTISPOOFING_INPUT_WIDTH != 80 or self.ANTISPOOFING_INPUT_HEIGHT != 80:
            warnings.warn(
                f"⚠️  Non-standard input dimensions for MiniFASNetV2: "
                f"{self.ANTISPOOFING_INPUT_WIDTH}x{self.ANTISPOOFING_INPUT_HEIGHT}. "
                f"Standard is 80x80. This may affect accuracy.",
                UserWarning,
            )

        # Validate normalization
        if self.ANTISPOOFING_NORMALIZE_MEAN != [
            0.5,
            0.5,
            0.5,
        ] and self.ANTISPOOFING_NORMALIZE_MEAN != [0.485, 0.456, 0.406]:
            warnings.warn(
                f"⚠️  Non-standard normalization mean: {self.ANTISPOOFING_NORMALIZE_MEAN}. "
                f"Ensure this matches your model's training configuration.",
                UserWarning,
            )

        return self

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
        v_lower = self.rate_limit_on_redis_failure.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"rate_limit_on_redis_failure must be one of {allowed}, got: {self.rate_limit_on_redis_failure}"
            )
        self.rate_limit_on_redis_failure = v_lower
        return self

    @model_validator(mode="after")
    def warn_about_generated_secrets(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            if not os.getenv("JWT_SECRET_KEY"):
                warnings.warn(
                    "⚠️  JWT_SECRET_KEY не задан в .env! "
                    "Токены будут инвалидированы при перезапуске.",
                    UserWarning,
                )
            if not os.getenv("ENCRYPTION_KEY"):
                warnings.warn(
                    "⚠️  ENCRYPTION_KEY не задан в .env! "
                    "Биометрические данные будут недоступны при перезапуске.",
                    UserWarning,
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
    logger.info(
        f"🐘 PostgreSQL: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'configured'}"
    )
    logger.info(f"📦 Redis: {settings.REDIS_URL}")
    logger.info(f"🗄️  MinIO: {settings.S3_ENDPOINT_URL}")
    logger.info(f"🔧 Rate limit policy: {settings.rate_limit_on_redis_failure}")

    # Anti-spoofing info
    if settings.USE_CERTIFIED_LIVENESS:
        logger.info(f"🛡️  Anti-Spoofing: ENABLED")
        logger.info(f"   Model: {settings.antispoofing_model_path_resolved}")
        logger.info(f"   Threshold: {settings.CERTIFIED_LIVENESS_THRESHOLD}")
        logger.info(
            f"   Input size: {settings.ANTISPOOFING_INPUT_WIDTH}x{settings.ANTISPOOFING_INPUT_HEIGHT}"
        )
        if settings.antispoofing_model_path_resolved.exists():
            logger.info(f"   Status: ✓ Model file found")
        else:
            logger.warning(
                f"   Status: ⚠ Model file NOT found (will download if enabled)"
            )
    else:
        logger.info(
            f"🛡️  Anti-Spoofing: DISABLED (can be enabled via USE_CERTIFIED_LIVENESS=true)"
        )

    logger.info("=" * 80)
