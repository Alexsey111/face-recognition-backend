"""
Reference Service - Управление эталонными изображениями лиц.

Ответственность:
- Создание/обновление/удаление reference
- Управление версионированием
- Сравнение embedding с предыдущими версиями
- Валидация quality score
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db.crud import ReferenceCRUD
from ..db.models import Reference
from ..services.encryption_service import EncryptionService
from ..services.ml_service import MLService
from ..services.storage_service import StorageService
from ..services.validation_service import ValidationService
from ..utils.exceptions import NotFoundError, ProcessingError, ValidationError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReferenceService:
    """Сервис для управления эталонными изображениями лиц."""

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
        self.storage_service = StorageService()

    # =========================================================================
    # Создание reference
    # =========================================================================

    async def create_reference(
        self,
        user_id: str,
        image_data: bytes,
        label: Optional[str] = None,
        quality_threshold: float = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_original: bool = None,
    ) -> Reference:
        """
        Создание нового reference с автоматическим версионированием.

        Args:
            user_id: ID пользователя
            image_data: Бинарные данные изображения (base64 или bytes)
            label: Опциональная метка для reference
            quality_threshold: Минимальный порог качества (default: из settings)
            metadata: Дополнительные метаданные
            store_original: Сохранять ли оригинальное изображение (default: из settings)

        Returns:
            Reference: Созданный reference объект

        Raises:
            ValidationError: Если изображение не прошло валидацию
            ProcessingError: Если не удалось сгенерировать embedding
        """
        logger.info(f"Creating reference for user {user_id}")

        # 1. Валидация изображения
        validation_result = await self.validation_service.validate_image(
            image_data,
            max_size=settings.MAX_UPLOAD_SIZE,
            allowed_formats=settings.allowed_image_formats_list,
        )

        if not validation_result.is_valid:
            raise ValidationError(
                f"Image validation failed: {validation_result.error_message}"
            )

        # 2. Генерация embedding
        embedding_result = await self.ml_service.generate_embedding(
            validation_result.image_data
        )

        if not embedding_result.get("success"):
            raise ProcessingError(
                f"Embedding generation failed: {embedding_result.get('error', 'Unknown error')}"
            )

        # 3. Проверка quality threshold
        quality_threshold = quality_threshold or settings.LOCAL_ML_QUALITY_THRESHOLD
        if embedding_result["quality_score"] < quality_threshold:
            raise ValidationError(
                f"Image quality ({embedding_result['quality_score']:.3f}) "
                f"below threshold ({quality_threshold:.3f})"
            )

        # 4. Шифрование embedding
        embedding_encrypted = await self.encryption_service.encrypt_embedding(
            embedding_result["embedding"]
        )

        # 5. Вычисление hash для дедупликации
        embedding_hash = hashlib.sha256(embedding_encrypted).hexdigest()

        # 6. Проверка на дубликат
        await self._check_duplicate_embedding(user_id, embedding_hash)

        # 7. Загрузка оригинального изображения (опционально)
        file_url = None
        if store_original is None:
            store_original = settings.STORE_ORIGINAL_IMAGES

        if store_original:
            upload_result = await self.storage_service.upload_image(
                image_data=validation_result.image_data,
                metadata={
                    "user_id": user_id,
                    "label": label,
                    "quality_score": embedding_result["quality_score"],
                },
            )
            file_url = upload_result.get("file_url")

        # 8. Получение версии (автоинкремент)
        version = await self._get_next_version(user_id)

        # 9. Получение предыдущего reference для версионирования
        previous_reference = await self.get_latest_reference(user_id)
        previous_reference_id = previous_reference.id if previous_reference else None

        # 10. Расчёт similarity с предыдущим reference (если есть)
        similarity_with_previous = None
        if previous_reference:
            similarity_with_previous = await self.calculate_similarity_with_old(
                new_embedding=embedding_result["embedding"],
                old_reference_id=previous_reference.id,
            )
            logger.info(
                f"Similarity with previous reference: {similarity_with_previous:.3f}"
            )

        # 11. Создание reference в БД
        reference = await ReferenceCRUD.create_reference(
            db=self.db,
            user_id=user_id,
            embedding_encrypted=embedding_encrypted,
            embedding_hash=embedding_hash,
            quality_score=embedding_result["quality_score"],
            image_filename=f"reference_{uuid.uuid4()}.jpg",
            image_size_mb=len(validation_result.image_data) / (1024 * 1024),
            image_format=validation_result.image_format,
            file_url=file_url,
            face_landmarks=embedding_result.get("landmarks"),
            label=label,
            version=version,
            previous_reference_id=previous_reference_id,
            metadata={
                **(metadata or {}),
                "similarity_with_previous": similarity_with_previous,
                "face_detected": embedding_result.get("face_detected", True),
            },
        )

        logger.info(
            f"Reference created successfully: {reference.id} "
            f"(version {version}, quality {embedding_result['quality_score']:.3f})"
        )

        return reference

    # =========================================================================
    # Получение reference
    # =========================================================================

    async def get_reference(self, reference_id: str) -> Optional[Reference]:
        """
        Получение reference по ID.

        Args:
            reference_id: ID reference

        Returns:
            Reference или None
        """
        return await ReferenceCRUD.get_reference_by_id(self.db, reference_id)

    async def get_latest_reference(self, user_id: str) -> Optional[Reference]:
        """
        Получение последнего активного reference для пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            Reference или None
        """
        stmt = (
            select(Reference)
            .where(Reference.user_id == user_id)
            .where(Reference.is_active == True)
            .order_by(desc(Reference.version))
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_references(
        self,
        user_id: str,
        include_inactive: bool = False,
    ) -> List[Reference]:
        """
        Получение всех references пользователя.

        Args:
            user_id: ID пользователя
            include_inactive: Включать ли неактивные references

        Returns:
            Список references
        """
        stmt = select(Reference).where(Reference.user_id == user_id)

        if not include_inactive:
            stmt = stmt.where(Reference.is_active == True)

        stmt = stmt.order_by(desc(Reference.created_at))

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # =========================================================================
    # Обновление reference
    # =========================================================================

    async def update_reference(
        self,
        reference_id: str,
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_active: Optional[bool] = None,
    ) -> Reference:
        """
        Обновление metadata reference.

        Args:
            reference_id: ID reference
            label: Новая метка
            metadata: Новые метаданные
            is_active: Статус активности

        Returns:
            Обновлённый reference

        Raises:
            NotFoundError: Если reference не найден
        """
        reference = await self.get_reference(reference_id)
        if not reference:
            raise NotFoundError(f"Reference {reference_id} not found")

        updated = await ReferenceCRUD.update_reference(
            db=self.db,
            reference_id=reference_id,
            label=label,
            metadata=metadata,
            is_active=is_active,
        )

        logger.info(f"Reference {reference_id} updated successfully")
        return updated

    # =========================================================================
    # Удаление reference
    # =========================================================================

    async def delete_reference(
        self,
        reference_id: str,
        soft_delete: bool = True,
    ) -> bool:
        """
        Удаление reference.

        Args:
            reference_id: ID reference
            soft_delete: Использовать soft delete (is_active=False)

        Returns:
            True если удалён успешно

        Raises:
            NotFoundError: Если reference не найден
        """
        reference = await self.get_reference(reference_id)
        if not reference:
            raise NotFoundError(f"Reference {reference_id} not found")

        if soft_delete:
            await self.update_reference(reference_id, is_active=False)
            logger.info(f"Reference {reference_id} soft deleted")
        else:
            await ReferenceCRUD.delete_reference(self.db, reference_id)
            logger.info(f"Reference {reference_id} permanently deleted")

        return True

    # =========================================================================
    # Сравнение с references
    # =========================================================================

    async def compare_with_references(
        self,
        image_data: bytes,
        reference_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        threshold: float = 0.6,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Сравнение лица с несколькими references.

        Args:
            image_data: Бинарные данные изображения
            reference_ids: Список ID references для сравнения (приоритет)
            user_id: ID пользователя (если reference_ids не указаны)
            threshold: Минимальный порог similarity
            max_results: Максимальное количество результатов

        Returns:
            Список результатов сравнения, отсортированных по similarity

        Raises:
            ValidationError: Если не указаны reference_ids и user_id
        """
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

        # 2. Получение references для сравнения
        if reference_ids:
            references = [await self.get_reference(rid) for rid in reference_ids]
            references = [ref for ref in references if ref is not None]
        elif user_id:
            references = await self.get_all_references(user_id, include_inactive=False)
        else:
            raise ValidationError("Either reference_ids or user_id must be provided")

        if not references:
            logger.warning("No references found for comparison")
            return []

        if len(references) > max_results:
            raise ValidationError(
                f"Too many references ({len(references)}). Maximum allowed: {max_results}"
            )

        # 3. Генерация embedding для входного изображения
        embedding_result = await self.ml_service.generate_embedding(
            validation_result.image_data
        )

        if not embedding_result.get("success"):
            raise ProcessingError("Failed to generate embedding for input image")

        input_embedding = embedding_result["embedding"]

        # 4. Сравнение с каждым reference
        results = []

        for reference in references:
            try:
                # Расшифровка reference embedding
                ref_embedding = await self.encryption_service.decrypt_embedding(
                    reference.embedding_encrypted
                )

                # Сравнение embeddings
                comparison_result = await self.ml_service.compare_faces(
                    image_data_1=validation_result.image_data,
                    embedding_2=ref_embedding,
                    threshold=threshold,
                )

                if comparison_result.get("success"):
                    similarity = comparison_result["similarity_score"]

                    results.append(
                        {
                            "reference_id": reference.id,
                            "user_id": reference.user_id,
                            "label": reference.label,
                            "similarity_score": similarity,
                            "distance": comparison_result.get(
                                "distance", 1.0 - similarity
                            ),
                            "is_match": similarity >= threshold,
                            "quality_score": reference.quality_score,
                            "version": reference.version,
                            "created_at": (
                                reference.created_at.isoformat()
                                if reference.created_at
                                else None
                            ),
                        }
                    )

            except Exception as e:
                logger.warning(f"Failed to compare with reference {reference.id}: {e}")
                continue

        # 5. Сортировка по similarity (убывание)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        logger.info(
            f"Comparison completed: {len(results)} matches out of {len(references)} references"
        )

        return results[:max_results]

    # =========================================================================
    # Вспомогательные методы
    # =========================================================================

    async def calculate_similarity_with_old(
        self,
        new_embedding: bytes,
        old_reference_id: str,
    ) -> float:
        """
        Расчёт similarity нового embedding с предыдущим reference.

        Args:
            new_embedding: Новый embedding (незашифрованный)
            old_reference_id: ID предыдущего reference

        Returns:
            Коэффициент similarity (0.0-1.0)
        """
        old_reference = await self.get_reference(old_reference_id)
        if not old_reference:
            logger.warning(f"Old reference {old_reference_id} not found")
            return 0.0

        try:
            # Расшифровка старого embedding
            old_embedding = await self.encryption_service.decrypt_embedding(
                old_reference.embedding_encrypted
            )

            # Сравнение embeddings напрямую (без изображения)
            similarity = await self.ml_service.compare_embeddings(
                embedding_1=new_embedding,
                embedding_2=old_embedding,
            )

            return similarity

        except Exception as e:
            logger.error(f"Failed to calculate similarity with old reference: {e}")
            return 0.0

    async def _get_next_version(self, user_id: str) -> int:
        """
        Получение следующей версии reference для пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            Номер следующей версии
        """
        stmt = select(func.max(Reference.version)).where(Reference.user_id == user_id)
        result = await self.db.execute(stmt)
        max_version = result.scalar()

        return (max_version or 0) + 1

    async def _check_duplicate_embedding(
        self,
        user_id: str,
        embedding_hash: str,
    ) -> None:
        """
        Проверка на дубликат embedding.

        Args:
            user_id: ID пользователя
            embedding_hash: Hash embedding

        Raises:
            ValidationError: Если найден дубликат
        """
        stmt = (
            select(Reference)
            .where(Reference.user_id == user_id)
            .where(Reference.embedding_hash == embedding_hash)
            .where(Reference.is_active == True)
        )
        result = await self.db.execute(stmt)
        duplicate = result.scalar_one_or_none()

        if duplicate:
            raise ValidationError(
                f"Duplicate embedding detected. Reference {duplicate.id} already exists."
            )

    # =========================================================================
    # Статистика
    # =========================================================================

    async def get_reference_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Получение статистики по references пользователя.

        Args:
            user_id: ID пользователя

        Returns:
            Словарь со статистикой
        """
        references = await self.get_all_references(user_id, include_inactive=True)

        active_references = [ref for ref in references if ref.is_active]
        inactive_references = [ref for ref in references if not ref.is_active]

        avg_quality = (
            sum(ref.quality_score or 0 for ref in active_references)
            / len(active_references)
            if active_references
            else 0.0
        )

        return {
            "total_references": len(references),
            "active_references": len(active_references),
            "inactive_references": len(inactive_references),
            "average_quality_score": avg_quality,
            "latest_version": references[0].version if references else 0,
            "oldest_reference": (
                references[-1].created_at.isoformat() if references else None
            ),
            "newest_reference": (
                references[0].created_at.isoformat() if references else None
            ),
        }
