"""
Pydantic модели для работы с эталонными изображениями.
Модели для создания, обновления и представления эталонов.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
import uuid


class ReferenceModel(BaseModel):
    """
    Базовая модель эталонного изображения.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID эталона"
    )
    user_id: str = Field(
        ..., description="ID пользователя, которому принадлежит эталон"
    )
    label: Optional[str] = Field(None, max_length=100, description="Метка эталона")
    file_url: Optional[str] = Field(None, description="URL файла эталона")
    file_size: Optional[int] = Field(None, description="Размер файла в байтах")
    image_format: Optional[str] = Field(None, description="Формат изображения")
    image_dimensions: Optional[Dict[str, int]] = Field(
        None, description="Размеры изображения", example={"width": 1920, "height": 1080}
    )
    embedding: Optional[bytes] = Field(None, description="Зашифрованный эмбеддинг лица")
    embedding_version: int = Field(default=1, description="Версия алгоритма эмбеддинга")
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка качества"
    )
    is_active: bool = Field(default=True, description="Активен ли эталон")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Дата создания"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Дата последнего обновления"
    )
    last_used: Optional[datetime] = Field(
        None, description="Дата последнего использования"
    )
    usage_count: int = Field(default=0, description="Количество использований")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные метаданные"
    )

    # Техническая информация
    original_filename: Optional[str] = Field(None, description="Оригинальное имя файла")
    checksum: Optional[str] = Field(None, description="Контрольная сумма файла")
    processing_time: Optional[float] = Field(
        None, description="Время обработки в секундах"
    )

    class Config:
        from_attributes = True

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        """Валидация метки эталона."""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                raise ValueError("Label cannot be empty")
            if len(v) > 100:
                raise ValueError("Label cannot be longer than 100 characters")
        return v


class ReferenceCreate(BaseModel):
    """
    Модель для создания эталонного изображения.
    """

    user_id: str = Field(..., description="ID пользователя")
    image_data: str = Field(
        ...,
        description="Изображение в формате base64 или URL",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    )
    label: Optional[str] = Field(None, max_length=100, description="Метка эталона")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные метаданные"
    )
    quality_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Минимальное качество для сохранения эталона"
    )
    auto_process: bool = Field(
        default=True, description="Автоматически обрабатывать изображение"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        """Валидация метки эталона."""
        return ReferenceModel.validate_label(v)

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Image data cannot be empty")
        return v


class ReferenceUpdate(BaseModel):
    """
    Модель для обновления эталонного изображения.
    """

    label: Optional[str] = Field(None, max_length=100, description="Метка эталона")
    image_data: Optional[str] = Field(
        None, description="Новое изображение эталона (опционально)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Обновленные метаданные"
    )
    quality_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Новый порог качества"
    )
    is_active: Optional[bool] = Field(None, description="Активен ли эталон")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v):
        """Валидация метки эталона."""
        return ReferenceModel.validate_label(v)

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if v is not None:
            return ReferenceCreate.validate_image_data(v)
        return v


class ReferenceSearch(BaseModel):
    """
    Модель для поиска эталонных изображений.
    """

    user_id: Optional[str] = Field(None, description="ID пользователя")
    label: Optional[str] = Field(None, description="Метка эталона")
    is_active: Optional[bool] = Field(None, description="Статус активности")
    quality_min: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Минимальное качество"
    )
    quality_max: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Максимальное качество"
    )
    created_after: Optional[datetime] = Field(None, description="Создано после")
    created_before: Optional[datetime] = Field(None, description="Создано до")
    usage_count_min: Optional[int] = Field(
        None, ge=0, description="Минимальное количество использований"
    )
    usage_count_max: Optional[int] = Field(
        None, ge=0, description="Максимальное количество использований"
    )

    @model_validator(mode="after")
    def validate_ranges(self):
        """Валидация диапазонов качества и использования."""
        if (
            self.quality_min is not None 
            and self.quality_max is not None 
            and self.quality_min > self.quality_max
        ):
            raise ValueError("quality_min cannot be greater than quality_max")
        
        if (
            self.usage_count_min is not None 
            and self.usage_count_max is not None 
            and self.usage_count_min > self.usage_count_max
        ):
            raise ValueError("usage_count_min cannot be greater than usage_count_max")
        
        return self


class ReferenceCompare(BaseModel):
    """
    Модель для сравнения с эталонными изображениями.
    """

    image_data: str = Field(
        ..., description="Изображение для сравнения в формате base64 или URL"
    )
    reference_ids: Optional[List[str]] = Field(
        None,
        description="Список ID эталонов для сравнения (если None, используются все активные)",
    )
    user_id: Optional[str] = Field(
        None, description="ID пользователя для фильтрации эталонов"
    )
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Порог схожести")
    max_results: int = Field(
        10, ge=1, le=100, description="Максимальное количество результатов"
    )
    include_metadata: bool = Field(
        default=False, description="Включить метаданные в результат"
    )

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Image data cannot be empty")
        return v

    @field_validator("reference_ids")
    @classmethod
    def validate_reference_ids(cls, v):
        """Валидация списка ID эталонов."""
        if v is not None and len(v) > 100:
            raise ValueError("Cannot compare with more than 100 references at once")
        return v


class ReferenceCompareResult(BaseModel):
    """
    Модель результата сравнения с эталонами.
    """

    reference_id: str = Field(..., description="ID эталона")
    label: Optional[str] = Field(None, description="Метка эталона")
    user_id: str = Field(..., description="ID пользователя")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Оценка схожести")
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Качество эталона"
    )
    distance: float = Field(..., description="Евклидово расстояние между эмбеддингами")
    processing_time: float = Field(..., description="Время обработки в секундах")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метаданные эталона")


class ReferenceListResponse(BaseModel):
    """
    Модель для списка эталонных изображений.
    """

    references: List[ReferenceModel] = Field(..., description="Список эталонов")
    total_count: int = Field(..., description="Общее количество эталонов")
    page: int = Field(..., description="Номер текущей страницы")
    per_page: int = Field(..., description="Количество элементов на странице")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")
    filters_applied: Optional[Dict[str, Any]] = Field(
        None, description="Примененные фильтры"
    )


class ReferenceStats(BaseModel):
    """
    Модель для статистики эталонных изображений.
    """

    total_references: int = Field(..., description="Общее количество эталонов")
    active_references: int = Field(..., description="Количество активных эталонов")
    inactive_references: int = Field(..., description="Количество неактивных эталонов")
    average_quality: float = Field(..., description="Среднее качество эталонов")
    total_usage_count: int = Field(..., description="Общее количество использований")
    most_used_reference: Optional[ReferenceModel] = Field(
        None, description="Самый используемый эталон"
    )
    quality_distribution: Dict[str, int] = Field(
        ...,
        description="Распределение по качеству",
        example={"excellent": 45, "good": 30, "fair": 20, "poor": 5},
    )
    user_distribution: Dict[str, int] = Field(
        ...,
        description="Распределение по пользователям",
        example={"user_123": 15, "user_456": 8, "user_789": 3},
    )
