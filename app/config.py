"""
Конфигурация приложения.
Загрузка настроек из переменных окружения.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import field_validator, model_validator, ValidationInfo
from pydantic_settings import BaseSettings
import os
import secrets


class Settings(BaseSettings):
    """
    Настройки приложения, загружаемые из переменных окружения.
    """

    # Базовые настройки приложения
    APP_NAME: str = "Face Recognition API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT: str = "production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS настройки
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")

    # База данных
    DATABASE_URL: str = "sqlite:///./dev.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_CONNECTION_POOL_SIZE: int = 10

    # S3/MinIO Storage Configuration
    # Unified S3-compatible storage settings (works with both AWS S3 and MinIO)
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "face-recognition"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False
    S3_PUBLIC_READ: bool = False  # по умолчанию приватное хранилище

    # MinIO aliases for backward compatibility
    MINIO_ENDPOINT: Optional[str] = None
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_KEY: Optional[str] = None
    MINIO_REGION: Optional[str] = None
    MINIO_BUCKET: Optional[str] = None
    MINIO_SSL: Optional[bool] = None

    # Настройки файлов (Phase 5)
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_EXPIRATION_DAYS: int = 30
    CLEANUP_INTERVAL_HOURS: int = 24

    # Хранение исходных данных
    STORE_ORIGINAL_IMAGES: bool = False  # не сохранять исходники по умолчанию
    DELETE_SOURCE_AFTER_PROCESSING: bool = True

    # ML сервис
    USE_LOCAL_ML_SERVICE: bool = True  # Использовать локальный ML сервис вместо внешнего
    ML_SERVICE_URL: str = "http://localhost:8001"  # Внешний ML сервис (если не локальный)
    ML_SERVICE_TIMEOUT: int = 30
    ML_SERVICE_API_KEY: Optional[str] = None

    # Локальный ML сервис настройки
    LOCAL_ML_DEVICE: str = "auto"  # auto, cpu, cuda
    LOCAL_ML_ENABLE_CUDA: bool = True
    LOCAL_ML_MODEL_CACHE_SIZE: int = 100  # Количество кэшируемых эмбеддингов
    LOCAL_ML_BATCH_SIZE: int = 1  # Размер батча для обработки
    LOCAL_ML_FACE_DETECTION_THRESHOLD: float = 0.9  # Порог детекции лица
    LOCAL_ML_QUALITY_THRESHOLD: float = 0.5  # Минимальное качество изображения
    LOCAL_ML_ENABLE_PERFORMANCE_MONITORING: bool = True

    # JWT токены
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Шифрование
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", secrets.token_urlsafe(32))
    ENCRYPTION_ALGORITHM: str = "fernet"

    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    # Безопасность
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_FORMATS: str = "JPEG,JPG,PNG,WEBP,HEIC,HEIF"
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    MIN_IMAGE_WIDTH: int = 224
    MIN_IMAGE_HEIGHT: int = 224
    MAX_IMAGE_WIDTH: int = 4096
    MAX_IMAGE_HEIGHT: int = 4096

    # Пороговые значения для верификации
    THRESHOLD_DEFAULT: float = 0.80
    THRESHOLD_MIN: float = 0.50
    THRESHOLD_MAX: float = 0.95
    TARGET_FAR: float = 0.001  # < 0.1%
    TARGET_FRR: float = 0.02  # < 2%

    # Webhook настройки
    WEBHOOK_URL: Optional[str] = None  # Основной webhook URL для уведомлений
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_MAX_RETRIES: int = 3
    WEBHOOK_RETRY_DELAY: int = 5

    # Уведомления
    ENABLE_SLACK_NOTIFICATIONS: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    EMAIL_SMTP_HOST: Optional[str] = None
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None

    # Мониторинг и метрики
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30

    # Логирование
    LOG_FORMAT: str = "json"
    LOG_FILE_PATH: Optional[str] = None
    LOG_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    LOG_BACKUP_COUNT: int = 5

    @property
    def cors_origins_list(self) -> List[str]:
        """Получение списка CORS origins из строки."""
        if not self.CORS_ORIGINS:
            return ["*"] if self.DEBUG else []  # ✅ Разрешаем все в DEBUG
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

    @property
    def allowed_image_formats_list(self) -> List[str]:
        """Получение списка допустимых форматов изображений."""
        return [fmt.strip().upper() for fmt in self.ALLOWED_IMAGE_FORMATS.split(",")]

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Получение списка допустимых расширений файлов."""
        return self.ALLOWED_EXTENSIONS

    @property
    def async_database_url(self) -> str:
        """Получение async DATABASE_URL."""
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        elif url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///")
        return url

    @property
    def sync_database_url(self) -> str:
        """Получение sync DATABASE_URL (для Alembic)."""
        return self.DATABASE_URL  # Оставляем как есть

    @property
    def redis_url_with_auth(self) -> str:
        """Получение Redis URL с аутентификацией."""
        if self.REDIS_PASSWORD:
            # Вставляем пароль в URL: redis://:password@host:port/db
            parts = self.REDIS_URL.split("://")
            if len(parts) == 2:
                protocol, rest = parts
                if "@" not in rest:
                    # Добавляем пароль перед хостом
                    return f"{protocol}://:{self.REDIS_PASSWORD}@{rest}"
        return self.REDIS_URL

    @model_validator(mode="after")
    def validate_debug_environment(self) -> "Settings":
        """Валидация debug режима в production."""
        if self.DEBUG and self.ENVIRONMENT == "production":
            raise ValueError("DEBUG mode cannot be enabled in production environment")
        return self

    @field_validator("LOG_FILE_PATH")
    @classmethod
    def validate_log_path(cls, v: Optional[str]) -> Optional[str]:
        """Валидация пути к лог файлу."""
        if v:
            log_path = Path(v)
            # Создаем директорию если не существует
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def setup_minio_aliases(self) -> "Settings":
        """Настройка алиасов для MINIO переменных."""
        # Если MINIO_* переменные не заданы, используем S3_* значения
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

    class Config:
        env_file = ".env"
        case_sensitive = True


# Создание экземпляра настроек
settings = Settings()
