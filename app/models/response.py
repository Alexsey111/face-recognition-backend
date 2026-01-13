"""
Pydantic модели для ответов API.
Модели для валидации выходных данных.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
import uuid


class BaseResponse(BaseModel):
    """
    Базовый класс для всех ответов API.
    """
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(..., description="Статус выполнения операции")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Время выполнения",
    )
    message: Optional[str] = Field(None, description="Сообщение о результате")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID запроса"
    )

class ErrorResponse(BaseResponse):
    """
    Модель для ответов с ошибками.
    """

    success: bool = False
    error_code: str = Field(..., description="Код ошибки")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Детали ошибки")


class UploadResponse(BaseResponse):
    """
    Модель для ответа на загрузку изображения.
    """

    image_id: str = Field(..., description="Уникальный ID изображения")
    file_url: Optional[str] = Field(None, description="URL загруженного файла")
    file_size: Optional[int] = Field(None, description="Размер файла в байтах")
    image_format: Optional[str] = Field(None, description="Формат изображения")
    image_dimensions: Optional[Dict[str, int]] = Field(
        None, description="Размеры изображения", json_schema_extra={"example": {"width": 1920, "height": 1080}}
    )
    processing_time: Optional[float] = Field(
        None, description="Время обработки в секундах"
    )
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка качества изображения"
    )


class VerifyResponse(BaseResponse):
    """
    Модель для ответа верификации лица.
    """

    verification_id: str = Field(..., description="Уникальный ID верификации")
    session_id: str = Field(..., description="ID сессии верификации")
    verified: bool = Field(..., description="Результат верификации")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Оценка схожести")
    threshold_used: float = Field(
        ..., ge=0.0, le=1.0, description="Использованный порог"
    )
    reference_id: Optional[str] = Field(None, description="ID использованного эталона")
    processing_time: float = Field(..., description="Время обработки в секундах")
    face_detected: bool = Field(..., description="Обнаружено лицо на изображении")
    face_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Качество распознанного лица"
    )
    liveness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка живости"
    )
    liveness_passed: Optional[bool] = Field(
        None, description="Прошла ли проверка живости"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные метаданные"
    )


class LivenessResponse(BaseResponse):
    """
    Модель для ответа проверки живости.
    """

    session_id: str = Field(..., description="ID сессии проверки живости")
    liveness_detected: bool = Field(..., description="Обнаружены признаки живости")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    challenge_type: str = Field(..., description="Тип выполненной проверки")
    processing_time: float = Field(..., description="Время обработки в секундах")
    anti_spoofing_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка антиспуфинга"
    )
    face_detected: bool = Field(..., description="Обнаружено лицо на изображении")
    multiple_faces: bool = Field(..., description="Обнаружено несколько лиц")
    image_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Качество изображения"
    )
    recommendations: Optional[List[str]] = Field(
        None, description="Рекомендации для улучшения"
    )
    # Дополнительные поля для новой функциональности
    challenge_specific_data: Optional[Dict[str, Any]] = Field(
        None, description="Специфичные данные для типа челленджа"
    )
    liveness_type: Optional[str] = Field(
        None, description="Детальный тип проверки живости"
    )
    depth_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Анализ глубины (для advanced anti-spoofing)"
    )
    reasoning_result: Optional[Dict[str, Any]] = Field(
        None, description="Результат multi-turn reasoning"
    )


class VideoLivenessResponse(BaseResponse):
    """
    Модель для ответа проверки живости по видео.
    """

    session_id: str = Field(..., description="ID сессии проверки")
    liveness_detected: bool = Field(..., description="Обнаружены признаки живости")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    challenge_type: str = Field(..., description="Тип анализа видео")
    frames_processed: int = Field(..., description="Количество обработанных кадров")
    processing_time: float = Field(..., description="Время обработки в секундах")
    sequence_data: Optional[Dict[str, Any]] = Field(
        None, description="Данные анализа последовательности кадров"
    )
    anti_spoofing_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка антиспуфинга"
    )
    face_detected: bool = Field(..., description="Обнаружено лицо в кадрах")
    recommendations: Optional[List[str]] = Field(
        None, description="Рекомендации для улучшения"
    )


class BatchEmbeddingResponse(BaseResponse):
    """
    Модель для ответа пакетной генерации эмбеддингов.
    """

    batch_id: str = Field(..., description="ID батча")
    total_images: int = Field(..., description="Общее количество изображений")
    successful_embeddings: int = Field(..., description="Успешно обработанные эмбеддинги")
    failed_embeddings: int = Field(..., description="Неудачные попытки")
    processing_time: float = Field(..., description="Время обработки в секундах")
    results: List[Dict[str, Any]] = Field(
        ..., description="Результаты для каждого изображения"
    )
    performance_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Метрики производительности батча"
    )


class BatchVerificationResponse(BaseResponse):
    """
    Модель для ответа пакетной верификации лиц.
    """

    batch_id: str = Field(..., description="ID батча")
    total_images: int = Field(..., description="Общее количество изображений")
    verified_matches: int = Field(..., description="Количество совпадений")
    failed_verifications: int = Field(..., description="Неудачные верификации")
    threshold_used: float = Field(..., description="Использованный порог")
    processing_time: float = Field(..., description="Время обработки в секундах")
    results: List[Dict[str, Any]] = Field(
        ..., description="Результаты для каждого изображения"
    )
    summary: Optional[Dict[str, Any]] = Field(
        None, description="Сводка результатов верификации"
    )


class AdvancedAntiSpoofingResponse(BaseResponse):
    """
    Модель для ответа продвинутой проверки anti-spoofing.
    """

    session_id: str = Field(..., description="ID сессии проверки")
    liveness_detected: bool = Field(..., description="Обнаружены признаки живости")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    analysis_type: str = Field(..., description="Тип выполненного анализа")
    processing_time: float = Field(..., description="Время обработки в секундах")
    anti_spoofing_score: float = Field(..., ge=0.0, le=1.0, description="Общая оценка anti-spoofing")
    
    # Результаты различных анализов
    depth_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Результаты анализа глубины"
    )
    texture_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Результаты анализа текстуры"
    )
    certified_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Результаты сертифицированной проверки"
    )
    
    # Multi-turn reasoning
    reasoning_result: Optional[Dict[str, Any]] = Field(
        None, description="Результат multi-turn reasoning анализа"
    )
    reasoning_summary: Optional[str] = Field(
        None, description="Краткое описание reasoning процесса"
    )
    
    # Компонентные оценки
    component_scores: Optional[Dict[str, float]] = Field(
        None, description="Оценки отдельных компонентов анализа"
    )
    
    # Сертификация
    certification_level: Optional[str] = Field(
        None, description="Уровень сертификации (basic, standard, premium)"
    )
    certification_passed: bool = Field(
        ..., description="Прошла ли сертификация"
    )
    
    face_detected: bool = Field(..., description="Обнаружено лицо на изображении")
    multiple_faces: bool = Field(..., description="Обнаружено несколько лиц")
    recommendations: Optional[List[str]] = Field(
        None, description="Рекомендации для улучшения"
    )


class ChallengeResponse(BaseResponse):
    """
    Модель для ответа challenge-response проверки.
    """

    session_id: str = Field(..., description="ID сессии")
    challenge_type: str = Field(..., description="Тип челленджа")
    instruction: str = Field(..., description="Инструкция для пользователя")
    expected_action: Optional[str] = Field(
        None, description="Ожидаемое действие"
    )
    response_detected: bool = Field(..., description="Обнаружен ответ на челлендж")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности в ответе")
    response_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Анализ ответа на челлендж"
    )
    processing_time: float = Field(..., description="Время обработки")
    recommendations: Optional[List[str]] = Field(
        None, description="Рекомендации"
    )


class ActiveLivenessResponse(BaseResponse):
    """
    Модель для ответа активной проверки живости.
    """

    session_id: str = Field(..., description="ID сессии")
    challenge_type: str = Field(..., description="Тип активной проверки")
    liveness_detected: bool = Field(..., description="Обнаружены признаки живости")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    processing_time: float = Field(..., description="Время обработки в секундах")
    
    # Специфичные данные для типа активной проверки
    blink_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Анализ моргания (для blink challenge)"
    )
    smile_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Анализ улыбки (для smile challenge)"
    )
    head_turn_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Анализ поворота головы (для turn_head challenge)"
    )
    challenge_response_analysis: Optional[Dict[str, Any]] = Field(
        None, description="Анализ challenge-response"
    )
    
    # Общие результаты
    anti_spoofing_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка anti-spoofing"
    )
    face_detected: bool = Field(..., description="Обнаружено лицо")
    image_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Качество изображения"
    )
    recommendations: Optional[List[str]] = Field(
        None, description="Рекомендации"
    )


class ReferenceResponse(BaseResponse):
    """
    Модель для ответа работы с эталонами.
    """

    reference_id: str = Field(..., description="ID эталона")
    user_id: str = Field(..., description="ID пользователя")
    label: Optional[str] = Field(None, description="Метка эталона")
    file_url: Optional[str] = Field(None, description="URL файла эталона")
    created_at: datetime = Field(..., description="Дата создания")
    updated_at: Optional[datetime] = Field(
        None, description="Дата последнего обновления"
    )
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка качества"
    )
    usage_count: int = Field(default=0, description="Количество использований")
    last_used: Optional[datetime] = Field(
        None, description="Дата последнего использования"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метаданные эталона")


class ReferenceListResponse(BaseResponse):
    """
    Модель для ответа со списком эталонов.
    """

    references: List[ReferenceResponse] = Field(..., description="Список эталонов")
    total_count: int = Field(..., description="Общее количество эталонов")
    page: int = Field(..., description="Номер текущей страницы")
    per_page: int = Field(..., description="Количество элементов на странице")
    has_next: bool = Field(..., description="Есть ли следующая страница")
    has_prev: bool = Field(..., description="Есть ли предыдущая страница")


class HealthResponse(BaseResponse):
    """
    Модель для ответа проверки здоровья сервиса.
    """

    status: str = Field(..., description="Статус сервиса")
    version: str = Field(..., description="Версия сервиса")
    uptime: float = Field(..., description="Время работы в секундах")
    services: Dict[str, str] = Field(..., description="Статус внешних сервисов")
    system_info: Dict[str, Any] = Field(..., description="Информация о системе")


class StatusResponse(BaseResponse):
    """
    Модель для ответа детального статуса.
    """

    database_status: str = Field(..., description="Статус подключения к БД")
    redis_status: str = Field(..., description="Статус подключения к Redis")
    storage_status: str = Field(..., description="Статус хранилища файлов")
    ml_service_status: str = Field(..., description="Статус ML сервиса")
    last_heartbeat: datetime = Field(..., description="Время последней проверки")


class AdminStatsResponse(BaseResponse):
    """
    Модель для ответа статистики администратора.
    """

    period: Dict[str, str] = Field(..., description="Период статистики")
    total_requests: int = Field(..., description="Общее количество запросов")
    successful_requests: int = Field(..., description="Успешные запросы")
    failed_requests: int = Field(..., description="Неуспешные запросы")
    average_response_time: float = Field(..., description="Среднее время ответа")
    verification_stats: Dict[str, int] = Field(
        ..., description="Статистика верификации"
    )
    liveness_stats: Dict[str, int] = Field(
        ..., description="Статистика проверки живости"
    )
    user_stats: Optional[Dict[str, Any]] = Field(
        None, description="Статистика по пользователям"
    )
    performance_metrics: Optional[Dict[str, float]] = Field(
        None, description="Метрики производительности"
    )


class AuthResponse(BaseResponse):
    """
    Модель для ответа аутентификации.
    """

    access_token: str = Field(..., description="Access токен")
    refresh_token: str = Field(..., description="Refresh токен")
    token_type: str = Field(default="bearer", description="Тип токена")
    expires_in: int = Field(..., description="Время жизни токена в секундах")
    user_id: str = Field(..., description="ID пользователя")
    user_role: str = Field(..., description="Роль пользователя")


class TokenResponse(BaseResponse):
    """
    Модель для ответа обновления токена.
    """

    access_token: str = Field(..., description="Новый access токен")
    token_type: str = Field(default="bearer", description="Тип токена")
    expires_in: int = Field(..., description="Время жизни токена в секундах")


class SessionResponse(BaseResponse):
    """
    Модель для ответа создания сессии.
    """

    session_id: str = Field(..., description="Уникальный ID сессии")
    session_type: str = Field(..., description="Тип сессии")
    expires_at: str = Field(..., description="Время истечения сессии (ISO формат)")
    user_id: Optional[str] = Field(None, description="ID пользователя")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Метаданные сессии")


class PaginatedResponse(BaseResponse):
    """
    Базовая модель для пагинированных ответов.
    """

    data: List[Dict[str, Any]] = Field(..., description="Данные")
    pagination: Dict[str, Union[int, bool]] = Field(
        ..., description="Информация о пагинации"
    )

    @classmethod
    def create(
        cls, data: List[Dict[str, Any]], page: int, per_page: int, total: int, **kwargs
    ) -> "PaginatedResponse":
        """Создание пагинированного ответа."""
        has_next = page * per_page < total
        has_prev = page > 1

        pagination = {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page,
            "has_next": has_next,
            "has_prev": has_prev,
        }

        return cls(data=data, pagination=pagination, **kwargs)
