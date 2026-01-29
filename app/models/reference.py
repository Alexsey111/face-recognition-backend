"""
Pydantic модели для работы с эталонными изображениями.
Модели для создания, обновления и представления эталонов.
"""

import base64
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ============================================================================
# Вспомогательные функции для валидации (DRY принцип)
# ============================================================================


def validate_label_value(v: Optional[str]) -> Optional[str]:
    """
    Валидация метки эталона.

    Args:
        v: Значение метки

    Returns:
        Валидированная метка

    Raises:
        ValueError: Если метка не валидна
    """
    if v is not None:
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Label cannot be empty")
        if len(v) > 100:
            raise ValueError("Label cannot be longer than 100 characters")
    return v


def validate_image_data_value(v: Optional[str]) -> Optional[str]:
    """
    Валидация формата изображения.

    Args:
        v: Данные изображения (base64 или URL)

    Returns:
        Валидированные данные изображения

    Raises:
        ValueError: Если данные не валидны
    """
    if v is not None:
        v_stripped = v.strip()
        if len(v_stripped) == 0:
            raise ValueError("Image data cannot be empty")

        # Проверка минимальной длины для base64
        if v_stripped.startswith("data:image"):
            # data:image/jpeg;base64,... минимум ~50 символов для валидного изображения
            if len(v_stripped) < 100:
                raise ValueError("Image data appears to be too short for a valid image")

        return v
    return v


# ============================================================================
# Основные модели
# ============================================================================


class ReferenceModel(BaseModel):
    """
    Базовая модель эталонного изображения.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            bytes: lambda v: base64.b64encode(v).decode("utf-8") if v else None
        },
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID эталона"
    )
    user_id: str = Field(
        ..., description="ID пользователя, которому принадлежит эталон"
    )
    label: Optional[str] = Field(None, max_length=100, description="Метка эталона")
    file_url: Optional[str] = Field(None, description="URL файла эталона")
    file_size: Optional[int] = Field(None, ge=0, description="Размер файла в байтах")
    image_format: Optional[str] = Field(
        None, description="Формат изображения (jpeg, png, heic)"
    )
    image_dimensions: Optional[Dict[str, int]] = Field(
        None,
        description="Размеры изображения",
        json_schema_extra={"example": {"width": 1920, "height": 1080}},
    )

    # Эмбеддинг исключаем из JSON по умолчанию из соображений безопасности
    embedding: Optional[bytes] = Field(
        None,
        exclude=True,  # Не включаем в JSON по умолчанию
        description="Зашифрованный эмбеддинг лица",
    )
    embedding_version: int = Field(
        default=1, ge=1, description="Версия алгоритма эмбеддинга"
    )
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка качества изображения"
    )
    is_active: bool = Field(default=True, description="Активен ли эталон")

    # Временные метки
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Дата создания"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Дата последнего обновления"
    )
    last_used: Optional[datetime] = Field(
        None, description="Дата последнего использования"
    )

    # Статистика использования
    usage_count: int = Field(default=0, ge=0, description="Количество использований")

    # Дополнительные данные
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные метаданные"
    )

    # Техническая информация
    original_filename: Optional[str] = Field(
        None, max_length=255, description="Оригинальное имя файла"
    )
    checksum: Optional[str] = Field(
        None, description="Контрольная сумма файла (SHA256)"
    )
    processing_time: Optional[float] = Field(
        None, ge=0.0, description="Время обработки в секундах"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: Optional[str]) -> Optional[str]:
        """Валидация метки эталона."""
        return validate_label_value(v)

    @field_validator("image_format")
    @classmethod
    def validate_image_format(cls, v: Optional[str]) -> Optional[str]:
        """Валидация формата изображения."""
        if v is not None:
            allowed_formats = {"jpeg", "jpg", "png", "heic", "webp"}
            v_lower = v.lower()
            if v_lower not in allowed_formats:
                raise ValueError(
                    f"Image format must be one of {allowed_formats}, got '{v}'"
                )
            return v_lower
        return v

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, v: Optional[str]) -> Optional[str]:
        """Валидация контрольной суммы (SHA256)."""
        if v is not None:
            if len(v) != 64:
                raise ValueError("Checksum must be a valid SHA256 hash (64 characters)")
        return v


class ReferenceCreate(BaseModel):
    """
    Модель для создания эталонного изображения.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
                "label": "Passport Photo",
                "quality_threshold": 0.8,
                "auto_process": True,
            }
        }
    )

    user_id: str = Field(..., min_length=1, description="ID пользователя")
    image_data: str = Field(
        ..., min_length=100, description="Изображение в формате base64 или URL"
    )
    label: Optional[str] = Field(None, max_length=100, description="Метка эталона")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные метаданные"
    )
    quality_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Минимальное качество для сохранения эталона",
    )
    auto_process: bool = Field(
        default=True, description="Автоматически обрабатывать изображение"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: Optional[str]) -> Optional[str]:
        """Валидация метки эталона."""
        return validate_label_value(v)

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v: str) -> str:
        """Валидация формата изображения."""
        validated = validate_image_data_value(v)
        if validated is None:
            raise ValueError("Image data is required")
        return validated

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Валидация ID пользователя."""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("User ID cannot be empty")
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
    def validate_label(cls, v: Optional[str]) -> Optional[str]:
        """Валидация метки эталона."""
        return validate_label_value(v)

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v: Optional[str]) -> Optional[str]:
        """Валидация формата изображения."""
        return validate_image_data_value(v)

    @model_validator(mode="after")
    def validate_at_least_one_field(self):
        """Проверка, что хотя бы одно поле для обновления указано."""
        fields_set = {k: v for k, v in self.model_dump().items() if v is not None}
        if not fields_set:
            raise ValueError("At least one field must be provided for update")
        return self


class ReferenceSearch(BaseModel):
    """
    Модель для поиска эталонных изображений.
    """

    user_id: Optional[str] = Field(None, description="ID пользователя")
    label: Optional[str] = Field(None, description="Метка эталона (partial match)")
    is_active: Optional[bool] = Field(None, description="Статус активности")
    quality_min: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Минимальное качество"
    )
    quality_max: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Максимальное качество"
    )
    created_after: Optional[datetime] = Field(
        None, description="Создано после указанной даты"
    )
    created_before: Optional[datetime] = Field(
        None, description="Создано до указанной даты"
    )
    usage_count_min: Optional[int] = Field(
        None, ge=0, description="Минимальное количество использований"
    )
    usage_count_max: Optional[int] = Field(
        None, ge=0, description="Максимальное количество использований"
    )

    @model_validator(mode="after")
    def validate_ranges(self):
        """Валидация диапазонов качества, дат и использования."""
        # Проверка качества
        if (
            self.quality_min is not None
            and self.quality_max is not None
            and self.quality_min > self.quality_max
        ):
            raise ValueError("quality_min cannot be greater than quality_max")

        # Проверка использований
        if (
            self.usage_count_min is not None
            and self.usage_count_max is not None
            and self.usage_count_min > self.usage_count_max
        ):
            raise ValueError("usage_count_min cannot be greater than usage_count_max")

        # Проверка дат
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after > self.created_before
        ):
            raise ValueError("created_after cannot be later than created_before")

        return self


class ReferenceCompare(BaseModel):
    """
    Модель для сравнения с эталонными изображениями.
    """

    image_data: str = Field(
        ...,
        min_length=100,
        description="Изображение для сравнения в формате base64 или URL",
    )
    reference_ids: Optional[List[str]] = Field(
        None,
        description="Список ID эталонов для сравнения (если None, используются все активные)",
    )
    user_id: Optional[str] = Field(
        None, description="ID пользователя для фильтрации эталонов"
    )
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Порог схожести")
    max_results: int = Field(
        default=10, ge=1, le=100, description="Максимальное количество результатов"
    )
    include_metadata: bool = Field(
        default=False, description="Включить метаданные в результат"
    )

    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v: str) -> str:
        """Валидация формата изображения."""
        validated = validate_image_data_value(v)
        if validated is None:
            raise ValueError("Image data is required")
        return validated

    @field_validator("reference_ids")
    @classmethod
    def validate_reference_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Валидация списка ID эталонов."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("reference_ids list cannot be empty")
            if len(v) > 100:
                raise ValueError("Cannot compare with more than 100 references at once")
            # Проверка на дубликаты
            if len(v) != len(set(v)):
                raise ValueError("reference_ids contains duplicates")
        return v


class ReferenceCompareResult(BaseModel):
    """
    Модель результата сравнения с эталонами.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reference_id": "ref_12345",
                "label": "Passport Photo",
                "user_id": "user_12345",
                "similarity_score": 0.95,
                "quality_score": 0.92,
                "distance": 0.23,
                "processing_time": 0.15,
                "is_match": True,
            }
        }
    )

    reference_id: str = Field(..., description="ID эталона")
    label: Optional[str] = Field(None, description="Метка эталона")
    user_id: str = Field(..., description="ID пользователя")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Оценка схожести")
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Качество эталона"
    )
    distance: float = Field(
        ..., ge=0.0, description="Евклидово расстояние между эмбеддингами"
    )
    processing_time: float = Field(
        ..., ge=0.0, description="Время обработки в секундах"
    )
    is_match: bool = Field(
        ..., description="Совпадает ли лицо с эталоном на основе порога"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метаданные эталона")


class ReferenceListResponse(BaseModel):
    """
    Модель для списка эталонных изображений с пагинацией.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "references": [],
                "total_count": 42,
                "page": 1,
                "per_page": 20,
                "total_pages": 3,
                "has_next": True,
                "has_prev": False,
            }
        }
    )

    references: List[ReferenceModel] = Field(..., description="Список эталонов")
    total_count: int = Field(..., ge=0, description="Общее количество эталонов")
    page: int = Field(..., ge=1, description="Номер текущей страницы")
    per_page: int = Field(..., ge=1, description="Количество элементов на странице")
    total_pages: int = Field(..., ge=0, description="Общее количество страниц")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")
    filters_applied: Optional[Dict[str, Any]] = Field(
        None, description="Примененные фильтры"
    )

    @model_validator(mode="after")
    def validate_pagination(self):
        """Валидация корректности пагинации."""
        # Вычисляем ожидаемое количество страниц
        expected_pages = (self.total_count + self.per_page - 1) // self.per_page

        if self.total_pages != expected_pages:
            raise ValueError(
                f"total_pages mismatch: expected {expected_pages}, got {self.total_pages}"
            )

        # Проверяем корректность has_next/has_prev
        if self.has_next and self.page >= self.total_pages:
            raise ValueError("has_next is True but page is at or beyond total_pages")

        if self.has_prev and self.page <= 1:
            raise ValueError("has_prev is True but page is 1")

        return self


class ReferenceStats(BaseModel):
    """
    Модель для статистики эталонных изображений.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_references": 150,
                "active_references": 142,
                "inactive_references": 8,
                "average_quality": 0.87,
                "total_usage_count": 1532,
                "quality_distribution": {
                    "excellent": 45,
                    "good": 30,
                    "fair": 20,
                    "poor": 5,
                },
            }
        }
    )

    total_references: int = Field(..., ge=0, description="Общее количество эталонов")
    active_references: int = Field(
        ..., ge=0, description="Количество активных эталонов"
    )
    inactive_references: int = Field(
        ..., ge=0, description="Количество неактивных эталонов"
    )
    average_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Среднее качество эталонов"
    )
    total_usage_count: int = Field(
        ..., ge=0, description="Общее количество использований"
    )
    most_used_reference: Optional[ReferenceModel] = Field(
        None, description="Самый используемый эталон"
    )
    quality_distribution: Dict[str, int] = Field(
        ...,
        description="Распределение по качеству (excellent/good/fair/poor)",
    )
    user_distribution: Dict[str, int] = Field(
        ...,
        description="Распределение по пользователям (user_id -> count)",
    )

    @model_validator(mode="after")
    def validate_stats_consistency(self):
        """Проверка консистентности статистики."""
        # Сумма активных и неактивных должна равняться общему количеству
        if self.active_references + self.inactive_references != self.total_references:
            raise ValueError(
                "Sum of active and inactive references must equal total_references"
            )

        # Проверка суммы в quality_distribution
        quality_sum = sum(self.quality_distribution.values())
        if quality_sum > 0 and quality_sum != self.total_references:
            raise ValueError(
                f"Quality distribution sum ({quality_sum}) doesn't match total_references ({self.total_references})"
            )

        # Проверка суммы в user_distribution
        user_sum = sum(self.user_distribution.values())
        if user_sum > 0 and user_sum != self.total_references:
            raise ValueError(
                f"User distribution sum ({user_sum}) doesn't match total_references ({self.total_references})"
            )

        return self


# ============================================================================
# Алиасы для обратной совместимости
# ============================================================================

Reference = ReferenceModel


# ============================================================================
# Дополнительные модели для операций
# ============================================================================


class ReferenceBatchDelete(BaseModel):
    """Модель для массового удаления эталонов."""

    reference_ids: List[str] = Field(
        ..., min_length=1, max_length=100, description="Список ID эталонов для удаления"
    )
    user_id: Optional[str] = Field(
        None, description="ID пользователя (для проверки прав доступа)"
    )
    force: bool = Field(
        default=False, description="Принудительное удаление без дополнительных проверок"
    )

    @field_validator("reference_ids")
    @classmethod
    def validate_no_duplicates(cls, v: List[str]) -> List[str]:
        """Проверка на дубликаты."""
        if len(v) != len(set(v)):
            raise ValueError("reference_ids contains duplicates")
        return v


class ReferenceBatchDeleteResponse(BaseModel):
    """Ответ на массовое удаление эталонов."""

    deleted_count: int = Field(..., ge=0, description="Количество удаленных эталонов")
    failed_count: int = Field(..., ge=0, description="Количество неудачных удалений")
    deleted_ids: List[str] = Field(..., description="ID успешно удаленных эталонов")
    failed_ids: List[str] = Field(
        ..., description="ID эталонов, которые не удалось удалить"
    )
    errors: Optional[Dict[str, str]] = Field(
        None, description="Детали ошибок для неудачных удалений (id -> error_message)"
    )
