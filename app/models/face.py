"""
models/face.py
Pydantic модели для работы с распознаванием лиц, эмбеддингами и liveness detection.
Централизованное хранилище всех face-related моделей данных.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# FACE DETECTION MODELS
# ============================================================================


class FaceDetectionResult(BaseModel):
    """
    Результат детекции лица на изображении.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    face_detected: bool = Field(..., description="Обнаружено ли лицо")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность детекции")
    bounding_box: Optional[List[float]] = Field(
        None, description="Координаты [x, y, width, height]"
    )
    face_count: int = Field(0, ge=0, description="Количество обнаруженных лиц")
    multiple_faces: bool = Field(False, description="Обнаружено несколько лиц")
    face_area_percentage: Optional[float] = Field(
        None, description="Процент площади лица на изображении"
    )

    @field_validator("bounding_box")
    @classmethod
    def validate_bounding_box(cls, v):
        """Валидация bounding box."""
        if v is not None:
            if len(v) != 4:
                raise ValueError(
                    "Bounding box must have 4 coordinates [x, y, width, height]"
                )
            if any(coord < 0 for coord in v):
                raise ValueError("Bounding box coordinates must be non-negative")
        return v


class FaceLandmarks(BaseModel):
    """
    Ключевые точки лица (facial landmarks).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    landmarks_type: Literal["68_points", "5_points", "106_points"] = Field(
        ..., description="Тип landmarks"
    )
    points: List[List[float]] = Field(..., description="Список координат [[x, y], ...]")
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Уверенность детекции landmarks"
    )

    # Специфичные метрики для лица
    eye_distance: Optional[float] = Field(None, description="Расстояние между глазами")
    face_ratio: Optional[float] = Field(
        None, description="Соотношение ширины к высоте лица"
    )
    head_pose: Optional[Dict[str, float]] = Field(
        None, description="Угол наклона головы {yaw, pitch, roll}"
    )

    @field_validator("points")
    @classmethod
    def validate_points(cls, v, info):
        """Валидация landmarks points."""
        landmarks_type = info.data.get("landmarks_type")
        expected_count = {"68_points": 68, "5_points": 5, "106_points": 106}

        if landmarks_type and len(v) != expected_count.get(landmarks_type, 0):
            raise ValueError(
                f"Expected {expected_count.get(landmarks_type)} points for {landmarks_type}, "
                f"got {len(v)}"
            )

        # Проверка формата каждой точки
        for point in v:
            if len(point) != 2:
                raise ValueError("Each landmark point must have 2 coordinates [x, y]")

        return v


# ============================================================================
# FACE EMBEDDING MODELS
# ============================================================================


class FaceEmbedding(BaseModel):
    """
    Эмбеддинг лица (векторное представление).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: List[float] = Field(..., description="Вектор эмбеддинга")
    dimension: int = Field(..., description="Размерность эмбеддинга (обычно 512)")
    model_name: str = Field(default="facenet-vggface2", description="Название модели")
    model_version: str = Field(default="1.0.0", description="Версия модели")
    quality_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Оценка качества эмбеддинга"
    )

    # Метаданные
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Дата создания"
    )
    source_image_hash: Optional[str] = Field(
        None, description="Хеш исходного изображения"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v, info):
        """Валидация эмбеддинга."""
        dimension = info.data.get("dimension", 512)

        if len(v) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {dimension}, got {len(v)}"
            )

        # Проверка на NaN и Inf
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Embedding contains NaN or Inf values")

        return v

    def to_numpy(self) -> np.ndarray:
        """Конвертация в numpy array."""
        return np.array(self.embedding, dtype=np.float32)

    @classmethod
    def from_numpy(cls, embedding: np.ndarray, **kwargs) -> "FaceEmbedding":
        """Создание из numpy array."""
        return cls(embedding=embedding.tolist(), dimension=len(embedding), **kwargs)


class FaceEmbeddingComparison(BaseModel):
    """
    Результат сравнения двух эмбеддингов.
    """

    similarity_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Косинусная схожесть"
    )
    euclidean_distance: float = Field(..., ge=0.0, description="Евклидово расстояние")
    is_match: bool = Field(..., description="Совпадение по порогу")
    threshold_used: float = Field(
        ..., ge=0.0, le=1.0, description="Использованный порог"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Уверенность в результате"
    )

    # Дополнительные метрики
    normalized_distance: Optional[float] = Field(
        None, description="Нормализованное расстояние [0, 1]"
    )
    confidence_level: Optional[
        Literal["very_low", "low", "medium", "high", "very_high"]
    ] = Field(None, description="Уровень уверенности")


# ============================================================================
# FACE QUALITY ASSESSMENT
# ============================================================================


class FaceQualityAssessment(BaseModel):
    """
    Оценка качества изображения лица.
    """

    overall_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Общая оценка качества"
    )

    # Компоненты качества
    sharpness_score: float = Field(0.0, ge=0.0, le=1.0, description="Резкость")
    brightness_score: float = Field(0.0, ge=0.0, le=1.0, description="Яркость")
    contrast_score: float = Field(0.0, ge=0.0, le=1.0, description="Контрастность")
    noise_level: float = Field(0.0, ge=0.0, le=1.0, description="Уровень шума")
    blur_score: float = Field(0.0, ge=0.0, le=1.0, description="Размытие")

    # Параметры лица
    face_size_score: float = Field(0.0, ge=0.0, le=1.0, description="Размер лица")
    face_angle_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Угол наклона лица"
    )
    occlusion_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Окклюзия (закрытие лица)"
    )

    # Проблемы
    issues: List[str] = Field(
        default_factory=list, description="Список обнаруженных проблем"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Рекомендации по улучшению"
    )

    # Пороговая проверка
    passes_quality_check: bool = Field(
        ..., description="Проходит ли минимальные требования"
    )
    minimum_quality_threshold: float = Field(
        0.5, description="Минимальный порог качества"
    )


# ============================================================================
# LIGHTING ANALYSIS
# ============================================================================


class LightingAnalysis(BaseModel):
    """
    Анализ освещения лица.
    """

    overall_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Общее качество освещения"
    )

    # Компоненты освещения
    exposure_score: float = Field(0.0, ge=0.0, le=1.0, description="Экспозиция")
    shadow_evenness: float = Field(
        0.0, ge=0.0, le=1.0, description="Равномерность теней"
    )
    left_right_balance: float = Field(
        0.0, ge=0.0, le=1.0, description="Баланс освещения слева/справа"
    )
    contrast_score: float = Field(0.0, ge=0.0, le=1.0, description="Контраст")

    # Флаги проблем
    is_overexposed: bool = Field(False, description="Переэкспонировано")
    is_underexposed: bool = Field(False, description="Недоэкспонировано")
    has_harsh_shadows: bool = Field(False, description="Жесткие тени")
    has_uneven_lighting: bool = Field(False, description="Неравномерное освещение")

    # Проблемы и рекомендации
    issues: List[str] = Field(default_factory=list, description="Обнаруженные проблемы")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")


# ============================================================================
# DEPTH ANALYSIS (3D)
# ============================================================================


class DepthAnalysis(BaseModel):
    """
    Анализ глубины для anti-spoofing (3D оценка).
    """

    depth_score: float = Field(..., ge=0.0, le=1.0, description="Оценка глубины сцены")
    flatness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Оценка плоскости (0=3D, 1=плоско)"
    )

    # Результат проверки
    is_likely_real: bool = Field(
        ..., description="Вероятно реальное лицо (не фото/экран)"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность в оценке")

    # Детали анализа
    gradient_variance: Optional[float] = Field(None, description="Вариация градиентов")
    edge_density: Optional[float] = Field(None, description="Плотность краев")
    texture_complexity: Optional[float] = Field(None, description="Сложность текстуры")

    # Аномалии
    anomalies: List[str] = Field(
        default_factory=list, description="Обнаруженные аномалии"
    )

    # Метод анализа
    analysis_method: str = Field(
        default="gradient_based", description="Метод анализа глубины"
    )


# ============================================================================
# LIVENESS DETECTION
# ============================================================================


class LivenessDetectionResult(BaseModel):
    """
    Результат проверки живости (liveness detection).
    """

    is_live: bool = Field(..., description="Обнаружена живость")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность")
    liveness_score: float = Field(..., ge=0.0, le=1.0, description="Оценка живости")

    # Тип проверки
    detection_method: Literal[
        "passive",
        "active",
        "blink",
        "smile",
        "turn_head",
        "certified",
        "depth_based",
        "video_based",
    ] = Field(..., description="Метод проверки живости")

    # Anti-spoofing оценки
    anti_spoofing_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Оценка anti-spoofing модели"
    )
    spoof_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Вероятность подделки"
    )

    # Дополнительные анализы
    depth_analysis: Optional[DepthAnalysis] = Field(None, description="Анализ глубины")
    texture_analysis: Optional[Dict[str, float]] = Field(
        None, description="Анализ текстуры"
    )

    # Флаги проблем
    detected_issues: List[str] = Field(
        default_factory=list,
        description="Обнаруженные проблемы (print_attack, screen_replay, mask, etc.)",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Рекомендации для улучшения"
    )

    # Сертификация
    is_certified: bool = Field(
        False, description="Проверка с сертифицированной моделью (MiniFASNetV2)"
    )
    certification_level: Optional[str] = Field(
        None, description="Уровень сертификации (ISO, NIST, etc.)"
    )

    # Метаданные
    processing_time: float = Field(0.0, description="Время обработки (секунды)")
    model_version: str = Field(default="1.0.0", description="Версия модели")


class ActiveLivenessChallenge(BaseModel):
    """
    Данные активного челленджа для проверки живости.
    """

    challenge_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID челленджа"
    )
    challenge_type: Literal["blink", "smile", "turn_head", "nod", "custom"] = Field(
        ..., description="Тип челленджа"
    )
    instructions: str = Field(..., description="Инструкция для пользователя")
    expected_action: Dict[str, Any] = Field(..., description="Ожидаемое действие")
    timeout_seconds: int = Field(30, description="Таймаут челленджа")

    # Результат (заполняется после проверки)
    completed: bool = Field(False, description="Челлендж завершен")
    success: bool = Field(False, description="Челлендж пройден успешно")
    detected_action: Optional[Dict[str, Any]] = Field(
        None, description="Обнаруженное действие"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Уверенность в результате"
    )


# ============================================================================
# FACE ALIGNMENT
# ============================================================================


class FaceAlignmentResult(BaseModel):
    """
    Результат выравнивания лица.
    """

    face_alignment_applied: bool = Field(..., description="Выравнивание применено")
    rotation_angle: float = Field(0.0, description="Угол поворота (градусы)")
    alignment_method: str = Field(
        default="68_point_landmarks", description="Метод выравнивания"
    )

    # Параметры выравнивания
    landmarks_type: str = Field(default="dlib_style_68", description="Тип landmarks")
    landmarks_detected: int = Field(0, description="Количество обнаруженных landmarks")

    # Метрики лица
    eye_distance: Optional[float] = Field(None, description="Расстояние между глазами")
    face_ratio: Optional[float] = Field(None, description="Соотношение лица")
    alignment_quality: float = Field(
        0.0, ge=0.0, le=1.0, description="Качество выравнивания"
    )

    # Трансформация
    transformation_matrix: Optional[List[List[float]]] = Field(
        None, description="Матрица трансформации"
    )


# ============================================================================
# COMPREHENSIVE FACE ANALYSIS
# ============================================================================


class ComprehensiveFaceAnalysis(BaseModel):
    """
    Полный анализ лица (все компоненты вместе).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ID и метаданные
    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID анализа"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Время анализа"
    )

    # Детекция и качество
    detection: FaceDetectionResult = Field(..., description="Результат детекции")
    quality: FaceQualityAssessment = Field(..., description="Оценка качества")

    # Landmarks и выравнивание
    landmarks: Optional[FaceLandmarks] = Field(None, description="Ключевые точки лица")
    alignment: Optional[FaceAlignmentResult] = Field(
        None, description="Результат выравнивания"
    )

    # Эмбеддинг
    embedding: Optional[FaceEmbedding] = Field(None, description="Эмбеддинг лица")

    # Освещение и глубина
    lighting: Optional[LightingAnalysis] = Field(None, description="Анализ освещения")
    depth: Optional[DepthAnalysis] = Field(None, description="Анализ глубины")

    # Liveness
    liveness: Optional[LivenessDetectionResult] = Field(
        None, description="Проверка живости"
    )

    # Общий результат
    overall_pass: bool = Field(..., description="Общий результат проверки")
    overall_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Общая уверенность"
    )

    # Проблемы и рекомендации (сводные)
    all_issues: List[str] = Field(
        default_factory=list, description="Все обнаруженные проблемы"
    )
    all_recommendations: List[str] = Field(
        default_factory=list, description="Все рекомендации"
    )

    # Производительность
    processing_time: float = Field(0.0, description="Общее время обработки (секунды)")

    def summary(self) -> Dict[str, Any]:
        """Краткая сводка анализа."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "face_detected": self.detection.face_detected,
            "quality_score": self.quality.overall_quality,
            "liveness_detected": self.liveness.is_live if self.liveness else None,
            "overall_pass": self.overall_pass,
            "confidence": self.overall_confidence,
            "issues_count": len(self.all_issues),
            "processing_time": self.processing_time,
        }


# ============================================================================
# BATCH PROCESSING MODELS
# ============================================================================


class BatchFaceProcessingRequest(BaseModel):
    """
    Запрос на пакетную обработку лиц.
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Уникальный ID пакета"
    )
    images: List[str] = Field(
        ..., min_length=1, max_length=100, description="Список изображений (base64)"
    )

    # Параметры обработки
    generate_embeddings: bool = Field(True, description="Генерировать эмбеддинги")
    check_liveness: bool = Field(False, description="Проверять живость")
    align_faces: bool = Field(True, description="Выравнивать лица")

    # Настройки
    batch_size: int = Field(
        10, ge=1, le=50, description="Размер подпакета для обработки"
    )
    parallel_processing: bool = Field(True, description="Параллельная обработка")


class BatchFaceProcessingResult(BaseModel):
    """
    Результат пакетной обработки лиц.
    """

    batch_id: str = Field(..., description="ID пакета")
    total_images: int = Field(..., description="Общее количество изображений")
    successful_count: int = Field(0, description="Успешно обработано")
    failed_count: int = Field(0, description="Ошибок обработки")

    # Результаты
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Результаты по каждому изображению"
    )

    # Производительность
    total_processing_time: float = Field(0.0, description="Общее время обработки")
    average_time_per_image: float = Field(
        0.0, description="Среднее время на изображение"
    )

    # Статус
    status: Literal["completed", "partial", "failed"] = Field(
        ..., description="Статус обработки"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Время завершения",
    )


# ============================================================================
# СПУФИНГ ИНДИКАТОРЫ
# ============================================================================


class SpoofingIndicators(BaseModel):
    """
    Индикаторы попытки обмана (spoofing).
    """

    overall_spoof_score: float = Field(
        ..., ge=0.0, le=1.0, description="Общая оценка подделки"
    )
    is_likely_spoof: bool = Field(..., description="Вероятно подделка")

    # Типы атак
    print_attack_probability: float = Field(
        0.0, ge=0.0, le=1.0, description="Вероятность атаки фото"
    )
    screen_replay_probability: float = Field(
        0.0, ge=0.0, le=1.0, description="Вероятность атаки экраном"
    )
    mask_attack_probability: float = Field(
        0.0, ge=0.0, le=1.0, description="Вероятность атаки маской"
    )
    deepfake_probability: float = Field(
        0.0, ge=0.0, le=1.0, description="Вероятность deepfake"
    )

    # Детальные индикаторы
    moire_pattern_detected: bool = Field(False, description="Обнаружен муаровый узор")
    screen_glare_detected: bool = Field(False, description="Обнаружен блик экрана")
    paper_texture_detected: bool = Field(
        False, description="Обнаружена текстура бумаги"
    )
    unnatural_eye_reflection: bool = Field(
        False, description="Неестественное отражение в глазах"
    )

    # Флаги
    flags: List[str] = Field(
        default_factory=list, description="Список обнаруженных флагов подделки"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность в оценке")
