"""
Pydantic модели для запросов API.
Модели для валидации входных данных.
"""

import base64
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


def validate_image_data_static(v: str) -> str:
    """Статическая функция валидации формата изображения."""
    if not v:
        raise ValueError("image_data cannot be empty")

    # Проверка base64
    if v.startswith("data:image/"):
        # data:image/jpeg;base64,/9j/4AAQ...
        try:
            header, encoded = v.split(",", 1)
            base64.b64decode(encoded, validate=True)
        except Exception:
            raise ValueError("Invalid base64 image data")
    # Проверка URL
    elif v.startswith("http://") or v.startswith("https://"):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid image URL")
    else:
        raise ValueError("image_data must be base64 or URL")

    return v


class UploadRequest(BaseModel):
    """
    Модель для запроса загрузки изображения.
    """

    image_data: str = Field(
        ...,
        description="Изображение в формате base64 или URL",
        json_schema_extra={
            "example": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        },
    )
    user_id: Optional[str] = Field(
        None,
        description="Идентификатор пользователя",
        json_schema_extra={"example": "user_123"},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Дополнительные метаданные",
        json_schema_extra={
            "example": {"source": "mobile_app", "device_id": "device_456"}
        },
    )
    reference_id: Optional[str] = Field(
        None, description="ID эталонного изображения для сравнения"
    )

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if v is not None:
            return validate_image_data_static(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        """Валидация метаданных."""
        if v is not None:
            # Проверяем размер метаданных
            import json

            metadata_str = json.dumps(v)
            if len(metadata_str) > 10240:  # 10KB
                raise ValueError("Metadata too large (max 10KB)")
        return v


class VerifyRequest(BaseModel):
    """
    Модель для запроса верификации лица.
    """

    session_id: str = Field(
        ...,
        description="ID сессии верификации",
        json_schema_extra={"example": "verify_session_123"},
    )
    image_data: str = Field(
        ..., description="Изображение для верификации в формате base64 или URL"
    )
    reference_id: Optional[str] = Field(
        None, description="ID эталонного изображения для сравнения"
    )
    user_id: Optional[str] = Field(None, description="Идентификатор пользователя")
    threshold: Optional[float] = Field(
        0.8, ge=0.0, le=1.0, description="Порог схожести для положительной верификации"
    )
    auto_enroll: bool = Field(
        False,
        description="Автоматически добавить в базу эталонов при успешной верификации",
    )

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if v is not None:
            return validate_image_data_static(v)
        return v


class LivenessRequest(BaseModel):
    """
    Модель для запроса проверки живости.
    """

    session_id: str = Field(
        ...,
        description="ID сессии проверки живости",
        json_schema_extra={"example": "liveness_session_456"},
    )
    image_data: str = Field(
        ..., description="Изображение для проверки живости в формате base64 или URL"
    )
    challenge_type: Optional[str] = Field(
        "passive",
        description="Тип проверки: passive, active, blink, smile, turn_head, video_blink, video_smile, video_head_turn",
        json_schema_extra={"example": "passive"},
    )
    challenge_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Данные для активной проверки (движения, повороты головы)",
        json_schema_extra={
            "example": {
                "rotation_x": 15,
                "rotation_y": 10,
                "instruction": "Turn head slightly",
            }
        },
    )

    @field_validator("challenge_type")
    @classmethod
    def validate_challenge_type(cls, v):
        """Валидация типа проверки живости."""
        allowed_types = [
            "passive",
            "active",
            "blink",
            "smile",
            "turn_head",
            "video_blink",
            "video_smile",
            "video_head_turn",
        ]
        if v not in allowed_types:
            raise ValueError(f"Challenge type must be one of: {allowed_types}")
        return v

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if v is not None:
            return validate_image_data_static(v)
        return v


class VideoLivenessRequest(BaseModel):
    """
    Модель для запроса проверки живости по видео.
    """

    session_id: str = Field(..., description="ID сессии проверки живости")
    video_data: str = Field(
        ..., description="Видео для проверки живости в формате base64 или URL"
    )
    challenge_type: str = Field(
        "video_blink",
        description="Тип анализа видео: video_blink, video_smile, video_head_turn",
    )
    frame_count: int = Field(
        10, ge=3, le=30, description="Количество кадров для анализа (3-30)"
    )
    challenge_data: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные данные для анализа видео"
    )

    @field_validator("challenge_type")
    @classmethod
    def validate_challenge_type(cls, v):
        """Валидация типа анализа видео."""
        allowed_types = ["video_blink", "video_smile", "video_head_turn"]
        if v not in allowed_types:
            raise ValueError(f"Video challenge type must be one of: {allowed_types}")
        return v


class BatchEmbeddingRequest(BaseModel):
    """
    Модель для пакетной генерации эмбеддингов.
    """

    images: List[str] = Field(
        ...,
        description="Список изображений в формате base64 или URL",
        min_length=1,
        max_length=100,
    )
    batch_size: int = Field(
        8, ge=1, le=32, description="Размер батча для обработки (1-32)"
    )
    user_id: Optional[str] = Field(None, description="ID пользователя для ассоциации")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метаданные для батча")

    @field_validator("images")
    @classmethod
    def validate_images(cls, v):
        """Валидация списка изображений."""
        for i, image_data in enumerate(v):
            if not image_data or len(image_data.strip()) == 0:
                raise ValueError(f"Image {i} cannot be empty")
        return v


class BatchVerificationRequest(BaseModel):
    """
    Модель для пакетной верификации лиц.
    """

    images: List[str] = Field(
        ...,
        description="Список изображений для верификации",
        min_length=1,
        max_length=100,
    )
    reference_embedding: List[float] = Field(
        ...,
        description="Эталонный эмбеддинг для сравнения",
        min_length=128,
        max_length=512,
    )
    threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Порог схожести для положительной верификации"
    )
    batch_size: int = Field(8, ge=1, le=32, description="Размер батча для обработки")

    @field_validator("images")
    @classmethod
    def validate_images(cls, v):
        """Валидация списка изображений."""
        for i, image_data in enumerate(v):
            if not image_data or len(image_data.strip()) == 0:
                raise ValueError(f"Image {i} cannot be empty")
        return v

    @field_validator("reference_embedding")
    @classmethod
    def validate_reference_embedding(cls, v):
        """Валидация эталонного эмбеддинга."""
        # Проверяем что все значения в разумном диапазоне
        for i, value in enumerate(v):
            if not isinstance(value, (int, float)) or abs(value) > 10:
                raise ValueError(
                    f"Embedding value {i} must be a number with |value| <= 10"
                )
        return v


class AdvancedAntiSpoofingRequest(BaseModel):
    """
    Модель для продвинутой проверки anti-spoofing.
    """

    session_id: str = Field(..., description="ID сессии проверки")
    image_data: str = Field(..., description="Изображение для анализа")
    analysis_type: str = Field(
        "comprehensive",
        description="Тип анализа: comprehensive, depth, texture, certified",
    )
    certification_level: Optional[str] = Field(
        None, description="Уровень сертификации: basic, standard, premium"
    )
    include_reasoning: bool = Field(
        True, description="Включить детальное объяснение reasoning"
    )

    @field_validator("analysis_type")
    @classmethod
    def validate_analysis_type(cls, v):
        """Валидация типа анализа."""
        allowed_types = ["comprehensive", "depth", "texture", "certified"]
        if v not in allowed_types:
            raise ValueError(f"Analysis type must be one of: {allowed_types}")
        return v

    @field_validator("certification_level")
    @classmethod
    def validate_certification_level(cls, v):
        """Валидация уровня сертификации."""
        if v is not None:
            allowed_levels = ["basic", "standard", "premium"]
            if v not in allowed_levels:
                raise ValueError(
                    f"Certification level must be one of: {allowed_levels}"
                )
        return v


class ReferenceCreateRequest(BaseModel):
    """
    Модель для создания эталонного изображения.
    """

    image_data: str = Field(
        ..., description="Изображение для создания эталона в формате base64 или URL"
    )
    user_id: str = Field(
        ...,
        description="Идентификатор пользователя",
        json_schema_extra={"example": "user_123"},
    )
    label: Optional[str] = Field(
        None,
        description="Метка для эталона",
        json_schema_extra={"example": "reference_photo_1"},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные метаданные"
    )
    quality_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Минимальное качество для сохранения эталона"
    )

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if v is not None:
            return validate_image_data_static(v)
        return v


class ReferenceUpdateRequest(BaseModel):
    """
    Модель для обновления эталонного изображения.
    """

    image_data: Optional[str] = Field(
        None, description="Новое изображение эталона (опционально)"
    )
    label: Optional[str] = Field(None, description="Новая метка эталона")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Обновленные метаданные"
    )
    quality_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Новый порог качества"
    )
    is_active: Optional[bool] = Field(None, description="Активен ли эталон")

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if v is not None:
            return validate_image_data_static(v)
        return v


class AdminStatsRequest(BaseModel):
    """
    Модель для запроса статистики администратора.
    """

    date_from: Optional[str] = Field(
        None,
        description="Начальная дата в формате YYYY-MM-DD",
        json_schema_extra={"example": "2024-01-01"},
    )
    date_to: Optional[str] = Field(
        None,
        description="Конечная дата в формате YYYY-MM-DD",
        json_schema_extra={"example": "2024-01-31"},
    )
    include_user_stats: bool = Field(
        False, description="Включить статистику по пользователям"
    )
    include_performance: bool = Field(
        True, description="Включить метрики производительности"
    )

    @field_validator("date_from", "date_to")
    @classmethod
    def validate_dates(cls, v, info: ValidationInfo):
        """Валидация формата дат."""
        if v:
            try:
                from datetime import datetime

                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"Invalid date format for {info.field_name}. Expected YYYY-MM-DD"
                )
        return v


class TokenRefreshRequest(BaseModel):
    """
    Модель для обновления токена.
    """

    refresh_token: str = Field(..., description="Refresh токен")


class UserCreate(BaseModel):
    """Schema для создания пользователя."""

    email: str = Field(..., description="Email пользователя")
    password: str = Field(..., min_length=8, description="Пароль пользователя")
    full_name: Optional[str] = Field(None, description="Полное имя")
    phone: Optional[str] = Field(None, description="Телефон")

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "full_name": "John Doe",
                "phone": "+1234567890",
            }
        }
    }


class UserLogin(BaseModel):
    """Schema для логина."""

    email: str = Field(..., description="Email пользователя")
    password: str = Field(..., description="Пароль")
    phone: Optional[str] = Field(None, description="Телефон (альтернатива email)")

    model_config = {
        "json_schema_extra": {
            "example": {"email": "user@example.com", "password": "SecurePass123!"}
        }
    }


class CompareRequest(BaseModel):
    """
    Модель для запроса сравнения лица с эталонами.
    """

    image_data: str = Field(
        ..., description="Изображение для сравнения в формате base64 или URL"
    )
    user_id: Optional[str] = Field(
        None, description="ID пользователя для поиска эталонов"
    )
    reference_ids: Optional[List[str]] = Field(
        None, description="Список ID конкретных эталонов для сравнения"
    )
    threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Порог схожести для определения совпадения"
    )
    max_results: int = Field(
        10, ge=1, le=100, description="Максимальное количество результатов"
    )
    include_metadata: bool = Field(
        False, description="Включить метаданные эталонов в результат"
    )

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if v is not None:
            return validate_image_data_static(v)
        return v

    @field_validator("reference_ids")
    @classmethod
    def validate_reference_ids(cls, v):
        """Валидация списка ID эталонов."""
        if v is not None and len(v) == 0:
            raise ValueError("reference_ids list cannot be empty")
        return v

    @model_validator(mode="after")
    def check_user_or_reference_ids(self):
        """Проверка, что указан либо user_id, либо reference_ids."""
        if not self.user_id and not self.reference_ids:
            raise ValueError("Either user_id or reference_ids must be provided")
        return self
