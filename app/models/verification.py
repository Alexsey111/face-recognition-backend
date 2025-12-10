"""
Pydantic модели для работы с сессиями верификации.
Модели для создания, управления и представления сессий верификации.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import uuid


class VerificationSessionModel(BaseModel):
    """
    Базовая модель сессии верификации.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID сессии")
    user_id: Optional[str] = Field(None, description="ID пользователя (опционально)")
    session_type: str = Field(..., description="Тип сессии: verification, liveness")
    status: str = Field(default="pending", description="Статус сессии")
    reference_id: Optional[str] = Field(None, description="ID эталонного изображения")
    request_data: Optional[Dict[str, Any]] = Field(None, description="Данные запроса")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Данные ответа")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Дата создания")
    started_at: Optional[datetime] = Field(None, description="Дата начала обработки")
    completed_at: Optional[datetime] = Field(None, description="Дата завершения")
    expires_at: datetime = Field(..., description="Дата истечения сессии")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные метаданные")
    
    # Техническая информация
    ip_address: Optional[str] = Field(None, description="IP адрес клиента")
    user_agent: Optional[str] = Field(None, description="User Agent клиента")
    processing_time: Optional[float] = Field(None, description="Время обработки в секундах")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    
    class Config:
        orm_mode = True
    
    @validator("session_type")
    def validate_session_type(cls, v):
        """Валидация типа сессии."""
        allowed_types = ["verification", "liveness", "enrollment", "identification"]
        if v not in allowed_types:
            raise ValueError(f"Session type must be one of: {allowed_types}")
        return v
    
    @validator("status")
    def validate_status(cls, v):
        """Валидация статуса сессии."""
        allowed_statuses = ["pending", "processing", "completed", "failed", "expired", "cancelled"]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v


class VerificationSessionCreate(BaseModel):
    """
    Модель для создания сессии верификации.
    """
    session_type: str = Field(..., description="Тип сессии")
    user_id: Optional[str] = Field(None, description="ID пользователя")
    reference_id: Optional[str] = Field(None, description="ID эталонного изображения")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные метаданные")
    expires_in_minutes: int = Field(default=30, ge=1, le=1440, description="Время жизни сессии в минутах")
    
    @validator("session_type")
    def validate_session_type(cls, v):
        """Валидация типа сессии."""
        return VerificationSessionModel.validate_session_type(v)


class VerificationSessionUpdate(BaseModel):
    """
    Модель для обновления сессии верификации.
    """
    status: Optional[str] = Field(None, description="Новый статус сессии")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Данные ответа")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Обновленные метаданные")
    
    @validator("status")
    def validate_status(cls, v):
        """Валидация статуса сессии."""
        if v is not None:
            return VerificationSessionModel.validate_status(v)
        return v


class VerificationRequest(BaseModel):
    """
    Модель для запроса верификации в рамках сессии.
    """
    session_id: str = Field(..., description="ID сессии верификации")
    image_data: str = Field(
        ..., 
        description="Изображение для верификации в формате base64 или URL"
    )
    reference_id: Optional[str] = Field(None, description="ID эталонного изображения")
    threshold: Optional[float] = Field(
        0.8, 
        ge=0.0, 
        le=1.0, 
        description="Порог схожести для положительной верификации"
    )
    auto_enroll: bool = Field(
        False,
        description="Автоматически добавить в базу эталонов при успешной верификации"
    )
    
    @validator("image_data")
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Image data cannot be empty")
        return v


class LivenessRequest(BaseModel):
    """
    Модель для запроса проверки живости в рамках сессии.
    """
    session_id: str = Field(..., description="ID сессии проверки живости")
    image_data: str = Field(
        ..., 
        description="Изображение для проверки живости в формате base64 или URL"
    )
    challenge_type: Optional[str] = Field(
        "passive",
        description="Тип проверки: passive, active, blink, smile, turn_head"
    )
    challenge_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Данные для активной проверки"
    )
    
    @validator("challenge_type")
    def validate_challenge_type(cls, v):
        """Валидация типа проверки живости."""
        if v is not None:
            allowed_types = ["passive", "active", "blink", "smile", "turn_head"]
            if v not in allowed_types:
                raise ValueError(f"Challenge type must be one of: {allowed_types}")
        return v
    
    @validator("image_data")
    def validate_image_data(cls, v):
        """Валидация формата изображения."""
        if not v or len(v.strip()) == 0:
            raise ValueError("Image data cannot be empty")
        return v


class VerificationResult(BaseModel):
    """
    Модель результата верификации.
    """
    session_id: str = Field(..., description="ID сессии")
    verified: bool = Field(..., description="Результат верификации")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Оценка схожести")
    threshold_used: float = Field(..., ge=0.0, le=1.0, description="Использованный порог")
    reference_id: Optional[str] = Field(None, description="ID использованного эталона")
    reference_label: Optional[str] = Field(None, description="Метка использованного эталона")
    processing_time: float = Field(..., description="Время обработки в секундах")
    face_detected: bool = Field(..., description="Обнаружено лицо на изображении")
    face_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Качество распознанного лица")
    embedding_distance: Optional[float] = Field(None, description="Евклидово расстояние между эмбеддингами")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Дополнительные метаданные")


class LivenessResult(BaseModel):
    """
    Модель результата проверки живости.
    """
    session_id: str = Field(..., description="ID сессии")
    liveness_detected: bool = Field(..., description="Обнаружены признаки живости")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    challenge_type: str = Field(..., description="Тип выполненной проверки")
    processing_time: float = Field(..., description="Время обработки в секундах")
    anti_spoofing_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Оценка антиспуфинга")
    face_detected: bool = Field(..., description="Обнаружено лицо на изображении")
    multiple_faces: bool = Field(..., description="Обнаружено несколько лиц")
    image_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Качество изображения")
    recommendations: Optional[List[str]] = Field(None, description="Рекомендации для улучшения")
    depth_analysis: Optional[Dict[str, float]] = Field(None, description="Анализ глубины изображения")


class SessionListResponse(BaseModel):
    """
    Модель для списка сессий верификации.
    """
    sessions: List[VerificationSessionModel] = Field(..., description="Список сессий")
    total_count: int = Field(..., description="Общее количество сессий")
    page: int = Field(..., description="Номер текущей страницы")
    per_page: int = Field(..., description="Количество элементов на странице")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")
    status_summary: Dict[str, int] = Field(
        ..., 
        description="Сводка по статусам",
        example={"pending": 5, "processing": 2, "completed": 45, "failed": 3}
    )


class SessionStats(BaseModel):
    """
    Модель для статистики сессий верификации.
    """
    total_sessions: int = Field(..., description="Общее количество сессий")
    active_sessions: int = Field(..., description="Количество активных сессий")
    completed_sessions: int = Field(..., description="Количество завершенных сессий")
    failed_sessions: int = Field(..., description="Количество неуспешных сессий")
    expired_sessions: int = Field(..., description="Количество истекших сессий")
    average_processing_time: float = Field(..., description="Среднее время обработки")
    success_rate: float = Field(..., description="Процент успешных верификаций")
    liveness_success_rate: float = Field(..., description="Процент успешных проверок живости")
    sessions_by_type: Dict[str, int] = Field(
        ..., 
        description="Распределение по типам сессий",
        example={"verification": 30, "liveness": 20, "enrollment": 5}
    )
    sessions_by_status: Dict[str, int] = Field(
        ..., 
        description="Распределение по статусам",
        example={"completed": 45, "pending": 5, "processing": 2, "failed": 3}
    )
    hourly_distribution: Dict[str, int] = Field(
        ..., 
        description="Распределение по часам",
        example={"00": 2, "01": 1, "02": 0, "03": 3}
    )


class SessionSearch(BaseModel):
    """
    Модель для поиска сессий верификации.
    """
    user_id: Optional[str] = Field(None, description="ID пользователя")
    session_type: Optional[str] = Field(None, description="Тип сессии")
    status: Optional[str] = Field(None, description="Статус сессии")
    reference_id: Optional[str] = Field(None, description="ID эталона")
    created_after: Optional[datetime] = Field(None, description="Создано после")
    created_before: Optional[datetime] = Field(None, description="Создано до")
    ip_address: Optional[str] = Field(None, description="IP адрес")
    has_error: Optional[bool] = Field(None, description="Есть ли ошибка")
    
    @validator("session_type")
    def validate_session_type(cls, v):
        """Валидация типа сессии."""
        if v is not None:
            return VerificationSessionModel.validate_session_type(v)
        return v
    
    @validator("status")
    def validate_status(cls, v):
        """Валидация статуса сессии."""
        if v is not None:
            return VerificationSessionModel.validate_status(v)
        return v