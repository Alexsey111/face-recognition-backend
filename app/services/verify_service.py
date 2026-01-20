"""
Verify Service - Верификация лиц против эталонных изображений.

Ответственность:
- Верификация лица против reference
- Расчёт similarity и confidence
- Управление сессиями верификации
- Динамический threshold на основе качества
- Webhook уведомления
"""

import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.models import VerificationSession
from ..db.crud import VerificationSessionCRUD, ReferenceCRUD
from ..services.ml_service import MLService
from ..services.encryption_service import EncryptionService
from ..services.validation_service import ValidationService
from ..services.cache_service import CacheService
from ..services.webhook_service import WebhookService
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, ProcessingError, NotFoundError
from ..utils.constants import CONFIDENCE_LEVELS
from ..middleware.metrics import record_verification, track_processing
from ..services.audit_service import AuditService

logger = get_logger(__name__)


class VerifyService:
    """Сервис для верификации лиц."""

    def __init__(self, db: AsyncSession):
        """
        Инициализация сервиса.

        Args:
            db: Асинхронная сессия базы данных
        """
        self.db = db
        self.ml_service = MLService()
        self.encryption_service = EncryptionService()
        self.validation_service = ValidationService()
        self.cache_service = CacheService()
        self.webhook_service = WebhookService(db)

    # =========================================================================
    # Основная верификация
    # =========================================================================

    async def verify_face(
        self,
        user_id: str,
        image_data: bytes,
        threshold: Optional[float] = None,
        session_id: Optional[str] = None,
        reference_id: Optional[str] = None,
        auto_enroll: bool = False,
    ) -> Dict[str, Any]:
        """
        Основная логика верификации лица против reference.

        Args:
            user_id: ID пользователя
            image_data: Бинарные данные изображения
            threshold: Порог similarity (если None, используется динамический)
            session_id: ID сессии верификации (опционально)
            reference_id: Конкретный reference для сравнения (опционально)
            auto_enroll: Автоматически создать reference при успешной верификации

        Returns:
            Словарь с результатами верификации

        Raises:
            NotFoundError: Если не найден reference
            ValidationError: Если изображение не прошло валидацию
            ProcessingError: Если ML обработка не удалась
        """
        start_time = time.monotonic()
        request_id = session_id or str(uuid.uuid4())

        logger.info(f"Starting face verification for user {user_id}, request {request_id}")

        try:
            with track_processing("verification"):
                # 1. Валидация изображения
                validation_result = await self.validation_service.validate_image(
                    image_data,
                    max_size=settings.MAX_UPLOAD_SIZE,
                    allowed_formats=settings.ALLOWED_IMAGE_FORMATS,
                )

                if not validation_result.is_valid:
                    raise ValidationError(
                        f"Image validation failed: {validation_result.error_message}"
                    )

                # 2. Получение reference
                if reference_id:
                    reference = await ReferenceCRUD.get_reference_by_id(self.db, reference_id)
                else:
                    reference = await self._get_user_reference(user_id)

                if not reference:
                    raise NotFoundError(
                        f"Reference image not found for user {user_id}. "
                        "Please register your face first."
                    )

                # 3. Расшифровка reference embedding
                reference_embedding = await self.encryption_service.decrypt_embedding(
                    reference.embedding_encrypted
                )

                # 4. ML верификация
                ml_result = await self.ml_service.verify_face(
                    image_data=validation_result.image_data,
                    reference_embedding=reference_embedding,
                    threshold=threshold or settings.THRESHOLD_DEFAULT,
                )

                if not ml_result.get("success"):
                    raise ProcessingError(
                        f"ML verification failed: {ml_result.get('error', 'Unknown error')}"
                    )

                # 5. Sanitize numpy types
                ml_result = self._sanitize_ml_result(ml_result)

                # 6. Динамический порог
                quality_score = ml_result.get("quality_score", 0.7)
                dynamic_threshold = self.calculate_dynamic_threshold(
                    requested_threshold=threshold,
                    quality_score=quality_score,
                )

                # 7. Определение результата верификации
                similarity = float(ml_result.get("similarity", 0.0))
                confidence = float(ml_result.get("confidence", 0.0))
                liveness_score = float(ml_result.get("liveness", 0.0))

                is_verified = similarity >= dynamic_threshold

                # 8. Определение уровня confidence
                confidence_level = self.determine_confidence_level(
                    similarity=similarity,
                    threshold=dynamic_threshold,
                    quality_score=quality_score,
                )

                # 9. Сохранение результата в БД
                verification = await self._save_verification(
                    user_id=user_id,
                    reference_id=reference.id,
                    session_id=request_id,
                    similarity=similarity,
                    confidence=confidence,
                    liveness_score=liveness_score,
                    threshold=dynamic_threshold,
                    verified=is_verified,
                    face_detected=ml_result.get("face_detected", True),
                    face_quality=quality_score,
                    processing_time=time.monotonic() - start_time,
                )

                # 10. Кэширование результата
                await self._cache_verification_result(
                    verification_id=str(verification.id),
                    result=ml_result,
                )

                # 11. Auto-enroll (если включен и верификация успешна)
                if auto_enroll and is_verified and confidence >= 0.8:
                    await self._auto_enroll_reference(
                        user_id=user_id,
                        image_data=validation_result.image_data,
                        quality_score=quality_score,
                        confidence=confidence,
                    )

                processing_time = time.monotonic() - start_time

                result = {
                    "verification_id": str(verification.id),
                    "session_id": request_id,
                    "verified": is_verified,
                    "similarity_score": similarity,
                    "confidence": confidence,
                    "confidence_level": confidence_level,
                    "threshold_used": dynamic_threshold,
                    "processing_time": processing_time,
                    "face_detected": ml_result.get("face_detected", True),
                    "face_quality": quality_score,
                    "liveness_score": liveness_score,
                    "liveness_passed": liveness_score >= 0.5 if liveness_score > 0 else None,
                    "reference_id": reference.id,
                    "metadata": {
                        "ml_model_version": ml_result.get("model_version"),
                        "detection_confidence": ml_result.get("detection_confidence"),
                    },
                    "request_id": request_id,
                }

                logger.info(
                    f"Verification completed: verified={is_verified}, "
                    f"similarity={similarity:.3f}, "
                    f"confidence={confidence:.3f}, "
                    f"time={processing_time:.3f}s"
                )

                # Запись метрики верификации
                result_label = "match" if is_verified else "no_match"
                record_verification(result_label)

                # Audit log для критичной операции верификации
                audit_service = AuditService(self.db)
                await audit_service.log_event(
                    action="verification_completed",
                    resource_type="verification_session",
                    resource_id=str(verification.id),
                    user_id=user_id,
                    new_values={"is_match": is_verified, "similarity": similarity},
                    success=True,
                    details={
                        "confidence": confidence,
                        "confidence_level": confidence_level,
                        "threshold_used": dynamic_threshold,
                        "processing_time": processing_time,
                    }
                )

                return result

        except Exception as e:
            logger.error(f"Verification failed for user {user_id}: {e}")
            raise

    # =========================================================================
    # Управление сессиями
    # =========================================================================

    async def create_verification_session(
        self,
        user_id: str,
        reference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_in_minutes: int = 30,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> VerificationSession:
        """
        Создание сессии верификации.

        Args:
            user_id: ID пользователя
            reference_id: ID конкретного reference (опционально)
            metadata: Дополнительные метаданные
            expires_in_minutes: Время жизни сессии в минутах
            ip_address: IP адрес клиента
            user_agent: User-Agent клиента

        Returns:
            Созданная сессия верификации
        """
        session_id = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)

        session = await VerificationSessionCRUD.create_session(
            db=self.db,
            user_id=user_id,
            session_id=session_id,
            reference_id=reference_id,
            session_type="verification",
            image_filename="pending",
            image_size_mb=0.0,
            expires_at=expires_at,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Кэшируем сессию
        await self.cache_service.set_verification_session(
            session_id=session_id,
            session_data={
                "session_id": session_id,
                "user_id": user_id,
                "reference_id": reference_id,
                "session_type": "verification",
                "status": "pending",
                "expires_at": expires_at.isoformat(),
                "metadata": metadata,
            },
            expire_seconds=expires_in_minutes * 60,
        )

        logger.info(f"Verification session created: {session_id}")
        return session

    async def get_verification_session(
        self,
        session_id: str,
    ) -> Optional[VerificationSession]:
        """
        Получение сессии верификации.

        Args:
            session_id: ID сессии

        Returns:
            Сессия или None
        """
        # Сначала проверяем кэш
        cached = await self.cache_service.get_verification_session(session_id)
        if cached:
            return cached

        # Если нет в кэше, ищем в БД
        return await VerificationSessionCRUD.get_session(self.db, session_id)

    async def get_verification_history(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VerificationSession]:
        """
        Получение истории верификаций пользователя.

        Args:
            user_id: ID пользователя
            filters: Фильтры (status, verified, date_from, date_to)
            limit: Количество записей
            offset: Смещение для пагинации

        Returns:
            Список сессий верификации
        """
        sessions = await VerificationSessionCRUD.get_user_sessions(
            db=self.db,
            user_id=user_id,
            limit=limit * 2,  # Получаем больше для фильтрации
        )

        # Применение фильтров
        if filters:
            sessions = self._apply_filters(sessions, filters)

        return sessions[offset:offset + limit]

    # =========================================================================
    # Динамический threshold
    # =========================================================================

    def calculate_dynamic_threshold(
        self,
        requested_threshold: Optional[float] = None,
        quality_score: float = 0.7,
        base_threshold: float = None,
    ) -> float:
        """
        Расчёт динамического порога верификации на основе качества изображения.

        Args:
            requested_threshold: Запрошенный порог (приоритет)
            quality_score: Оценка качества изображения (0.0-1.0)
            base_threshold: Базовый порог (default: из settings)

        Returns:
            Эффективный порог верификации
        """
        base_threshold = base_threshold or settings.THRESHOLD_DEFAULT

        # Если явно указан порог, используем его с ограничениями
        if requested_threshold is not None:
            return float(
                np.clip(
                    requested_threshold,
                    settings.THRESHOLD_MIN,
                    settings.THRESHOLD_MAX,
                )
            )

        # Динамическая корректировка на основе качества
        # Высокое качество -> можно использовать более строгий порог
        # Низкое качество -> нужно понизить порог, чтобы избежать FRR
        delta = (0.5 - quality_score) * 0.1  # -0.05 до +0.05
        adjusted_threshold = base_threshold + delta

        return float(
            np.clip(
                adjusted_threshold,
                settings.THRESHOLD_MIN,
                settings.THRESHOLD_MAX,
            )
        )

    # =========================================================================
    # Confidence level
    # =========================================================================

    def determine_confidence_level(
        self,
        similarity: float,
        threshold: float,
        quality_score: float,
    ) -> str:
        """
        Определение уровня уверенности в результате верификации.

        Args:
            similarity: Коэффициент similarity
            threshold: Использованный порог
            quality_score: Качество изображения

        Returns:
            Уровень confidence: 'high', 'medium', 'low', 'very_low'
        """
        # Базовая уверенность на основе similarity
        if similarity >= CONFIDENCE_LEVELS["high"]:
            base_confidence = "high"
        elif similarity >= CONFIDENCE_LEVELS["medium"]:
            base_confidence = "medium"
        elif similarity >= CONFIDENCE_LEVELS["low"]:
            base_confidence = "low"
        else:
            base_confidence = "very_low"

        # Корректировка на основе качества
        if quality_score < 0.5 and base_confidence == "high":
            base_confidence = "medium"
        elif quality_score < 0.3 and base_confidence == "medium":
            base_confidence = "low"

        # Если similarity близко к порогу, понижаем уверенность
        if abs(similarity - threshold) < 0.05 and base_confidence in ["high", "medium"]:
            base_confidence = "medium" if base_confidence == "high" else "low"

        return base_confidence

    # =========================================================================
    # Webhook notifications
    # =========================================================================

    async def send_verification_webhook(
        self,
        user_id: str,
        verification_id: str,
        verified: bool,
        similarity: float,
        confidence: float,
    ) -> None:
        """
        Отправка webhook уведомления о результате верификации.

        Args:
            user_id: ID пользователя
            verification_id: ID верификации
            verified: Результат верификации
            similarity: Коэффициент similarity
            confidence: Уровень confidence
        """
        try:
            payload = {
                "event": "face.verified",
                "verification_id": verification_id,
                "user_id": user_id,
                "verified": verified,
                "similarity": similarity,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await self.webhook_service.emit_event(
                event_type="face.verified",
                user_id=user_id,
                payload=payload,
            )

            logger.info(f"Webhook sent for verification {verification_id}")

        except Exception as e:
            logger.warning(f"Failed to send webhook for verification {verification_id}: {e}")

    # =========================================================================
    # Вспомогательные методы
    # =========================================================================

    async def _get_user_reference(self, user_id: str):
        """Получение активного reference пользователя."""
        references = await ReferenceCRUD.get_all_references(self.db, user_id)
        active_refs = [ref for ref in references if ref.is_active]

        if not active_refs:
            return None

        # Возвращаем самый новый
        return max(active_refs, key=lambda r: r.version)

    async def _save_verification(
        self,
        user_id: str,
        reference_id: str,
        session_id: str,
        similarity: float,
        confidence: float,
        liveness_score: float,
        threshold: float,
        verified: bool,
        face_detected: bool,
        face_quality: float,
        processing_time: float,
    ) -> VerificationSession:
        """Сохранение результата верификации в БД."""
        return await VerificationSessionCRUD.create_session(
            db=self.db,
            user_id=user_id,
            session_id=session_id,
            reference_id=reference_id,
            session_type="verification",
            status="completed",
            image_filename="verification_image",
            image_size_mb=0.0,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            is_match=verified,
            similarity_score=similarity,
            confidence=confidence,
            liveness_score=liveness_score,
            face_detected=face_detected,
            face_quality_score=face_quality,
            processing_time=processing_time,
            metadata={
                "threshold_used": threshold,
                "verified": verified,
            },
        )

    async def _cache_verification_result(
        self,
        verification_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Кэширование результата верификации."""
        try:
            await self.cache_service.set(
                key=f"verification:{verification_id}",
                value=result,
                ttl=300,  # 5 минут
            )
        except Exception as e:
            logger.warning(f"Failed to cache verification result: {e}")

    async def _auto_enroll_reference(
        self,
        user_id: str,
        image_data: bytes,
        quality_score: float,
        confidence: float,
    ) -> None:
        """
        Автоматическое создание reference при успешной верификации.

        Args:
            user_id: ID пользователя
            image_data: Бинарные данные изображения
            quality_score: Оценка качества
            confidence: Уровень confidence
        """
        try:
            from .reference_service import ReferenceService

            reference_service = ReferenceService(self.db)
            await reference_service.create_reference(
                user_id=user_id,
                image_data=image_data,
                label=f"auto_enroll_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                quality_threshold=quality_score * 0.9,  # Немного ниже текущего
                metadata={
                    "source": "auto_enroll",
                    "confidence": confidence,
                },
            )

            logger.info(f"Auto-enroll completed for user {user_id}")

        except Exception as e:
            logger.warning(f"Auto-enroll failed for user {user_id}: {e}")

    def _sanitize_ml_result(self, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразование numpy типов в Python типы."""
        def sanitize(value):
            if hasattr(value, "item"):  # numpy scalar
                return value.item()
            if isinstance(value, (list, tuple)):
                return [sanitize(v) for v in value]
            if isinstance(value, dict):
                return {k: sanitize(v) for k, v in value.items()}
            return value

        return sanitize(ml_result)

    def _apply_filters(
        self,
        sessions: List[VerificationSession],
        filters: Dict[str, Any],
    ) -> List[VerificationSession]:
        """Применение фильтров к списку сессий."""
        filtered = sessions

        if "status" in filters:
            filtered = [s for s in filtered if s.status == filters["status"]]

        if "verified" in filters:
            filtered = [s for s in filtered if s.is_match == filters["verified"]]

        if "date_from" in filters:
            date_from = datetime.fromisoformat(filters["date_from"])
            filtered = [s for s in filtered if s.created_at >= date_from]

        if "date_to" in filters:
            date_to = datetime.fromisoformat(filters["date_to"])
            filtered = [s for s in filtered if s.created_at <= date_to]

        return filtered
