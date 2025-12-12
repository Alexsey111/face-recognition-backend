"""
Pydantic модели для запросов API.
Модели для валидации входных данных.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo
import base64
from urllib.parse import urlparse


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
        example="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    )
    user_id: Optional[str] = Field(
        None, description="Идентификатор пользователя", example="user_123"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Дополнительные метаданные",
        example={"source": "mobile_app", "device_id": "device_456"},
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
        ..., description="ID сессии верификации", example="verify_session_123"
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
        ..., description="ID сессии проверки живости", example="liveness_session_456"
    )
    image_data: str = Field(
        ..., description="Изображение для проверки живости в формате base64 или URL"
    )
    challenge_type: Optional[str] = Field(
        "passive",
        description="Тип проверки: passive, active, blink, smile, turn_head",
        example="passive",
    )
    challenge_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Данные для активной проверки (движения, повороты головы)",
        example={"rotation_x": 15, "rotation_y": 10},
    )

    @field_validator("challenge_type")
    @classmethod
    def validate_challenge_type(cls, v):
        """Валидация типа проверки живости."""
        allowed_types = ["passive", "active", "blink", "smile", "turn_head"]
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


class ReferenceCreateRequest(BaseModel):
    """
    Модель для создания эталонного изображения.
    """

    image_data: str = Field(
        ..., description="Изображение для создания эталона в формате base64 или URL"
    )
    user_id: str = Field(
        ..., description="Идентификатор пользователя", example="user_123"
    )
    label: Optional[str] = Field(
        None, description="Метка для эталона", example="reference_photo_1"
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
        None, description="Начальная дата в формате YYYY-MM-DD", example="2024-01-01"
    )
    date_to: Optional[str] = Field(
        None, description="Конечная дата в формате YYYY-MM-DD", example="2024-01-31"
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


class AuthRequest(BaseModel):
    """
    Модель для запроса аутентификации.
    """

    username: str = Field(
        ..., description="Имя пользователя или email", example="admin@example.com"
    )
    password: str = Field(..., description="Пароль", min_length=8)


class TokenRefreshRequest(BaseModel):
    """
    Модель для обновления токена.
    """

    refresh_token: str = Field(..., description="Refresh токен")
