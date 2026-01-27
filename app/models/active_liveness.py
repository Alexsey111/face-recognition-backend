"""
Pydantic models for Active Liveness Detection.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


# ============================================================================
# Challenge Types
# ============================================================================

ChallengeType = Literal[
    "blink",           # Моргание
    "smile",           # Улыбка
    "turn_head_left",  # Поворот головы влево
    "turn_head_right", # Поворот головы вправо
    "turn_head_up",    # Наклон головы вверх
    "turn_head_down",  # Наклон головы вниз
    "open_mouth",      # Открыть рот
    "random",          # Случайный челлендж (генерируется сервером)
]


# ============================================================================
# Request Models
# ============================================================================

class ActiveLivenessChallengeRequest(BaseModel):
    """Запрос на создание Active Liveness челленджа."""
    
    challenge_type: ChallengeType = Field(
        default="random",
        description="Тип челленджа для проверки живости"
    )
    
    timeout_seconds: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Время на выполнение челленджа (секунды)"
    )
    
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="Сложность челленджа"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "challenge_type": "blink",
                "timeout_seconds": 10,
                "difficulty": "medium"
            }
        }


class ActiveLivenessVerifyRequest(BaseModel):
    """Запрос на верификацию Active Liveness с видео/последовательностью изображений."""
    
    challenge_id: str = Field(
        ...,
        description="ID челленджа, полученный при создании"
    )
    
    video_data: Optional[bytes] = Field(
        default=None,
        description="Видео в формате MP4/WebM (base64 encoded)"
    )
    
    image_sequence: Optional[List[bytes]] = Field(
        default=None,
        description="Последовательность изображений (если нет видео)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Дополнительные метаданные (FPS, device info, etc.)"
    )
    
    @field_validator('video_data', 'image_sequence')
    @classmethod
    def validate_data_provided(cls, v, info):
        """Проверка что предоставлено видео ИЛИ последовательность."""
        if info.data.get('video_data') is None and info.data.get('image_sequence') is None:
            raise ValueError("Either video_data or image_sequence must be provided")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "challenge_id": "550e8400-e29b-41d4-a716-446655440000",
                "metadata": {
                    "fps": 30,
                    "device": "iPhone 13",
                    "camera": "front"
                }
            }
        }


class BlinkDetectionRequest(BaseModel):
    """Запрос на детекцию моргания."""
    
    video_data: Optional[bytes] = None
    image_sequence: Optional[List[bytes]] = None
    
    min_blinks: int = Field(default=1, ge=1, le=5, description="Минимальное количество морганий")
    timeout_seconds: int = Field(default=10, ge=5, le=30)
    
    class Config:
        json_schema_extra = {
            "example": {
                "min_blinks": 2,
                "timeout_seconds": 10
            }
        }


class HeadMovementRequest(BaseModel):
    """Запрос на детекцию движения головы."""
    
    video_data: Optional[bytes] = None
    image_sequence: Optional[List[bytes]] = None
    
    movement_type: Literal["left", "right", "up", "down", "nod", "shake"] = Field(
        ...,
        description="Тип движения головы"
    )
    
    min_angle_degrees: float = Field(
        default=15.0,
        ge=10.0,
        le=45.0,
        description="Минимальный угол поворота (градусы)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "movement_type": "left",
                "min_angle_degrees": 20.0
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class ChallengeInstruction(BaseModel):
    """Инструкция для пользователя."""
    
    text: str = Field(..., description="Текст инструкции")
    icon: Optional[str] = Field(None, description="Иконка/эмодзи для UI")
    duration_seconds: int = Field(..., description="Время на выполнение")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Моргните 2 раза",
                "icon": "👁️",
                "duration_seconds": 10
            }
        }


class ActiveLivenessChallengeResponse(BaseModel):
    """Ответ с деталями челленджа."""
    
    success: bool = True
    challenge_id: str = Field(..., description="Уникальный ID челленджа")
    challenge_type: ChallengeType
    instruction: ChallengeInstruction
    
    expires_at: datetime = Field(..., description="Время истечения челленджа")
    server_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Дополнительные параметры для клиента
    expected_duration_seconds: int = Field(..., description="Ожидаемая длительность действия")
    min_frames_required: int = Field(default=30, description="Минимум кадров для анализа")
    recommended_fps: int = Field(default=30, description="Рекомендуемый FPS")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "challenge_id": "550e8400-e29b-41d4-a716-446655440000",
                "challenge_type": "blink",
                "instruction": {
                    "text": "Моргните 2 раза",
                    "icon": "👁️",
                    "duration_seconds": 10
                },
                "expires_at": "2026-01-25T12:00:00Z",
                "expected_duration_seconds": 3,
                "min_frames_required": 30,
                "recommended_fps": 30
            }
        }


class BlinkDetectionResult(BaseModel):
    """Результат детекции моргания."""
    
    blinks_detected: int = Field(..., description="Количество обнаруженных морганий")
    blinks_required: int = Field(..., description="Требуемое количество морганий")
    success: bool = Field(..., description="Успешно ли выполнено задание")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность детекции")
    
    # Детальная статистика
    average_ear: float = Field(..., description="Средний Eye Aspect Ratio")
    blink_frames: List[int] = Field(default_factory=list, description="Номера кадров с морганиями")
    total_frames_analyzed: int = Field(..., description="Всего проанализировано кадров")
    
    quality_issues: List[str] = Field(default_factory=list, description="Проблемы с качеством")


class HeadMovementResult(BaseModel):
    """Результат детекции движения головы."""
    
    movement_detected: bool = Field(..., description="Обнаружено ли движение")
    movement_type: str = Field(..., description="Тип обнаруженного движения")
    
    angle_degrees: float = Field(..., description="Угол поворота (градусы)")
    required_angle: float = Field(..., description="Требуемый угол")
    
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Euler angles
    yaw: float = Field(..., description="Поворот вокруг вертикальной оси (влево/вправо)")
    pitch: float = Field(..., description="Наклон вперед/назад")
    roll: float = Field(..., description="Наклон влево/вправо")
    
    frames_analyzed: int = Field(...)


class OcclusionDetectionResult(BaseModel):
    """Результат детекции окклюзий."""
    
    has_mask: bool = Field(default=False)
    has_sunglasses: bool = Field(default=False)
    has_regular_glasses: bool = Field(default=False)
    has_vr_headset: bool = Field(default=False)
    has_hand_covering: bool = Field(default=False)
    
    occlusion_score: float = Field(..., ge=0.0, le=1.0, description="Оценка видимости лица (1=полностью видимо)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    details: Dict[str, Any] = Field(default_factory=dict)


class ActiveLivenessVerifyResponse(BaseModel):
    """Ответ верификации Active Liveness."""
    
    success: bool
    liveness_detected: bool = Field(..., description="Обнаружена ли живость")
    challenge_completed: bool = Field(..., description="Выполнен ли челлендж")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Общая уверенность")
    
    # Результаты специфичных проверок
    blink_result: Optional[BlinkDetectionResult] = None
    head_movement_result: Optional[HeadMovementResult] = None
    occlusion_result: Optional[OcclusionDetectionResult] = None
    
    # Пассивная liveness проверка
    passive_liveness_score: float = Field(..., ge=0.0, le=1.0, description="Оценка пассивной живости")
    anti_spoofing_score: float = Field(..., ge=0.0, le=1.0, description="Anti-spoofing оценка")
    
    # Качество видео
    video_quality: Dict[str, Any] = Field(default_factory=dict)
    
    # Метаданные
    processing_time_seconds: float = Field(...)
    frames_analyzed: int = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Диагностика
    failure_reasons: List[str] = Field(default_factory=list, description="Причины неудачи")
    warnings: List[str] = Field(default_factory=list, description="Предупреждения")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "liveness_detected": True,
                "challenge_completed": True,
                "confidence": 0.95,
                "blink_result": {
                    "blinks_detected": 2,
                    "blinks_required": 2,
                    "success": True,
                    "confidence": 0.92,
                    "average_ear": 0.25,
                    "total_frames_analyzed": 90
                },
                "passive_liveness_score": 0.97,
                "anti_spoofing_score": 0.94,
                "video_quality": {
                    "quality_score": 0.85,
                    "resolution": [640, 480],
                    "fps": 30
                },
                "processing_time_seconds": 2.3,
                "frames_analyzed": 90,
                "failure_reasons": [],
                "warnings": []
            }
        }


class SmileDetectionResult(BaseModel):
    """Результат детекции улыбки."""
    
    smile_detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    mouth_aspect_ratio: float = Field(..., description="MAR (Mouth Aspect Ratio)")
    smile_intensity: float = Field(..., ge=0.0, le=1.0, description="Интенсивность улыбки")
    
    frames_with_smile: List[int] = Field(default_factory=list)
    total_frames: int


class CombinedLivenessScore(BaseModel):
    """Комбинированная оценка живости (Active + Passive)."""
    
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Общая оценка живости")
    is_live: bool = Field(..., description="Финальное решение: живой или нет")
    
    # Компоненты оценки
    active_liveness_score: float = Field(..., ge=0.0, le=1.0)
    passive_liveness_score: float = Field(..., ge=0.0, le=1.0)
    anti_spoofing_score: float = Field(..., ge=0.0, le=1.0)
    
    # Веса компонентов
    weights: Dict[str, float] = Field(
        default={
            "active": 0.5,
            "passive": 0.3,
            "anti_spoofing": 0.2
        }
    )
    
    # Метаданные
    method: str = Field(default="weighted_average")
    threshold: float = Field(default=0.7)
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 0.92,
                "is_live": True,
                "active_liveness_score": 0.95,
                "passive_liveness_score": 0.88,
                "anti_spoofing_score": 0.93,
                "weights": {
                    "active": 0.5,
                    "passive": 0.3,
                    "anti_spoofing": 0.2
                },
                "method": "weighted_average",
                "threshold": 0.7
            }
        }


# ============================================================================
# Challenge Storage Models (for internal use)
# ============================================================================

class ChallengeSession(BaseModel):
    """Хранение активного челленджа."""
    
    challenge_id: str
    challenge_type: ChallengeType
    created_at: datetime
    expires_at: datetime
    
    # Параметры челленджа
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Статус
    status: Literal["pending", "completed", "failed", "expired"] = "pending"
    
    # Результат (если completed)
    result: Optional[Dict[str, Any]] = None


# ============================================================================
# Statistics Models
# ============================================================================

class ActiveLivenessStatistics(BaseModel):
    """Статистика Active Liveness."""
    
    total_challenges_created: int = 0
    total_challenges_completed: int = 0
    total_challenges_failed: int = 0
    
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # По типам челленджей
    by_challenge_type: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    
    # Средние метрики
    average_processing_time: float = 0.0
    average_confidence: float = 0.0
    
    # Последние N челленджей
    recent_results: List[Dict[str, Any]] = Field(default_factory=list)
