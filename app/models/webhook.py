"""
Pydantic модели для webhook системы.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from enum import Enum


class WebhookEventType(str, Enum):
    """Типы событий для webhook"""

    VERIFICATION_COMPLETED = "verification.completed"
    LIVENESS_COMPLETED = "liveness.completed"
    REFERENCE_CREATED = "reference.created"
    USER_ACTIVITY = "user.activity"
    SYSTEM_ALERT = "system.alert"
    WEBHOOK_TEST = "webhook.test"


class WebhookStatus(str, Enum):
    """Статусы отправки webhook"""

    PENDING = "pending"
    SENDING = "sending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    EXPIRED = "expired"


class WebhookConfigBase(BaseModel):
    """Базовая модель конфигурации webhook"""

    user_id: str = Field(..., description="ID пользователя")
    webhook_url: str = Field(..., description="URL webhook endpoint")
    secret: str = Field(..., description="Секретный ключ для HMAC подписи")
    event_types: List[WebhookEventType] = Field(
        default=[WebhookEventType.VERIFICATION_COMPLETED],
        description="Типы событий для отправки",
    )
    is_active: bool = Field(default=True, description="Активна ли конфигурация")
    timeout: int = Field(
        default=10, ge=1, le=60, description="Таймаут запроса в секундах"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Максимальное количество попыток"
    )
    retry_delay: int = Field(
        default=1, ge=1, le=60, description="Базовая задержка retry в секундах"
    )

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("Webhook URL must start with http:// or https://")
        return v


class WebhookConfigCreate(WebhookConfigBase):
    """Модель для создания webhook конфигурации"""

    pass


class WebhookConfigUpdate(BaseModel):
    """Модель для обновления webhook конфигурации"""

    webhook_url: Optional[str] = None
    secret: Optional[str] = None
    event_types: Optional[List[WebhookEventType]] = None
    is_active: Optional[bool] = None
    timeout: Optional[int] = Field(None, ge=1, le=60)
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    retry_delay: Optional[int] = Field(None, ge=1, le=60)


class WebhookConfig(WebhookConfigBase):
    """Модель webhook конфигурации"""

    id: str = Field(..., description="Уникальный ID конфигурации")
    created_at: datetime = Field(..., description="Дата создания")
    updated_at: datetime = Field(..., description="Дата обновления")

    model_config = ConfigDict(from_attributes=True)


class WebhookLogBase(BaseModel):
    """Базовая модель лога webhook"""

    webhook_config_id: str = Field(..., description="ID конфигурации webhook")
    event_type: WebhookEventType = Field(..., description="Тип события")
    payload: Dict[str, Any] = Field(..., description="Отправляемый payload")
    payload_hash: str = Field(..., description="Хеш payload для deduplication")
    attempts: int = Field(
        default=0, ge=0, le=10, description="Количество попыток отправки"
    )
    last_attempt_at: Optional[datetime] = Field(
        None, description="Время последней попытки"
    )
    next_retry_at: Optional[datetime] = Field(
        None, description="Время следующей попытки"
    )
    status: WebhookStatus = Field(
        default=WebhookStatus.PENDING, description="Статус отправки"
    )
    http_status: Optional[int] = Field(None, description="HTTP статус ответа")
    response_body: Optional[str] = Field(None, description="Тело ответа")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    signature: Optional[str] = Field(None, description="HMAC подпись")


class WebhookLogCreate(WebhookLogBase):
    """Модель для создания записи лога webhook"""

    pass


class WebhookLogUpdate(BaseModel):
    """Модель для обновления записи лога webhook"""

    attempts: Optional[int] = Field(None, ge=0, le=10)
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    status: Optional[WebhookStatus] = None
    http_status: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None


class WebhookLog(WebhookLogBase):
    """Модель лога webhook"""

    id: str = Field(..., description="Уникальный ID записи")
    created_at: datetime = Field(..., description="Дата создания")

    model_config = ConfigDict(from_attributes=True)


class WebhookPayload(BaseModel):
    """Модель payload для webhook"""

    event: str = Field(..., description="Тип события")
    timestamp: str = Field(..., description="Временная метка в ISO формате")
    data: Dict[str, Any] = Field(..., description="Данные события")
    signature: Optional[str] = Field(None, description="HMAC подпись")


class WebhookVerificationData(BaseModel):
    """Модель данных для webhook верификации"""

    user_id: str = Field(..., description="ID пользователя")
    session_id: str = Field(..., description="ID сессии")
    is_match: bool = Field(..., description="Результат верификации")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Оценка схожести")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    is_live: bool = Field(..., description="Результат проверки живости")
    liveness_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка живости"
    )
    processing_time_ms: int = Field(
        ..., ge=0, description="Время обработки в миллисекундах"
    )
    reference_version: Optional[int] = Field(None, ge=1, description="Версия эталона")

    model_config = ConfigDict(
        json_schema_extra={  # ✅ ИСПРАВЛЕНО (было schema_extra)
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "660e8400-e29b-41d4-a716-446655440111",
                "is_match": True,
                "similarity_score": 0.92,
                "confidence": 0.98,
                "is_live": True,
                "liveness_score": 0.87,
                "processing_time_ms": 450,
                "reference_version": 2,
            }
        }
    )


class WebhookLivenessData(BaseModel):
    """Модель данных для webhook проверки живости"""

    user_id: str = Field(..., description="ID пользователя")
    session_id: str = Field(..., description="ID сессии")
    is_live: bool = Field(..., description="Результат проверки живости")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уровень уверенности")
    challenge_type: str = Field(..., description="Тип челленджа")
    anti_spoofing_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка антиспуфинга"
    )
    processing_time_ms: int = Field(
        ..., ge=0, description="Время обработки в миллисекундах"
    )
    face_detected: bool = Field(..., description="Обнаружено лицо")
    multiple_faces: bool = Field(..., description="Обнаружено несколько лиц")
    recommendations: List[str] = Field(default=[], description="Рекомендации")

    model_config = ConfigDict(
        json_schema_extra={  # ✅ ИСПРАВЛЕНО (было schema_extra)
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "660e8400-e29b-41d4-a716-446655440111",
                "is_live": True,
                "confidence": 0.87,
                "challenge_type": "passive",
                "anti_spoofing_score": 0.85,
                "processing_time_ms": 320,
                "face_detected": True,
                "multiple_faces": False,
                "recommendations": [],
            }
        }
    )


class WebhookTestRequest(BaseModel):
    """Модель запроса для тестирования webhook"""

    webhook_url: Optional[str] = Field(None, description="URL для тестирования")
    event_type: WebhookEventType = Field(
        default=WebhookEventType.WEBHOOK_TEST, description="Тип тестового события"
    )
    custom_data: Optional[Dict[str, Any]] = Field(
        None, description="Кастомные данные для теста"
    )


class WebhookTestResponse(BaseModel):
    """Модель ответа тестирования webhook"""

    success: bool = Field(..., description="Успешность теста")
    webhook_url: str = Field(..., description="Тестируемый URL")
    response_time: float = Field(..., description="Время ответа в секундах")
    status_code: Optional[int] = Field(None, description="HTTP статус код")
    error: Optional[str] = Field(None, description="Ошибка если есть")
    timestamp: str = Field(..., description="Время теста")
    signature: Optional[str] = Field(None, description="HMAC подпись тестового запроса")


class WebhookStatistics(BaseModel):
    """Модель статистики webhook"""

    total_webhooks: int = Field(..., description="Общее количество webhook")
    successful_webhooks: int = Field(..., description="Успешно отправленных webhook")
    failed_webhooks: int = Field(..., description="Неудачных webhook")
    pending_webhooks: int = Field(..., description="Ожидающих отправки webhook")
    retry_webhooks: int = Field(..., description="Webhook на retry")
    success_rate: float = Field(..., description="Процент успешности")
    average_response_time: float = Field(..., description="Среднее время ответа")
    last_webhook_at: Optional[datetime] = Field(
        None, description="Время последнего webhook"
    )


class WebhookRetryRequest(BaseModel):
    """Модель запроса на повторную отправку webhook"""

    webhook_log_id: str = Field(
        ..., description="ID записи лога для повторной отправки"
    )
    force: bool = Field(
        default=False, description="Принудительно отправить игнорируя лимиты"
    )


class WebhookBulkAction(BaseModel):
    """Модель для массовых операций с webhook"""

    webhook_config_ids: List[str] = Field(..., description="Список ID конфигураций")
    action: str = Field(
        ..., description="Действие (activate, deactivate, delete, test)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Дополнительные параметры"
    )
