"""
BiometricStorage Service - Шифрованное хранение биометрических шаблонов.

Обеспечивает:
- AES-256-GCM шифрование эмбеддингов
- Сжатие перед шифрованием (blosc2/lz4)
- Версионирование шаблонов
- Аудит всех операций
- Автоматическую ротацию ключей

================================================================================
ПОЛИТИКА ХРАНЕНИЯ И УДАЛЕНИЯ БИОМЕТРИЧЕСКИХ ДАННЫХ
================================================================================

Согласно требованиям GDPR и ФЗ-152, биометрические данные хранятся с учётом:

1. СРОКИ ХРАНЕНИЯ:
   - Эмбеддинги (хешированные биометрические шаблоны): 3 года с последнего использования
   - Эталонные изображения (raw photos): 30 дней после верификации
   - Audit логи операций с биометрикой: 1 год

2. АВТОМАТИЧЕСКОЕ УДАЛЕНИЕ:
   - При отсутствии активности > 3 лет → полное удаление
   - При запросе пользователя на удаление (GDPR "right to be forgotten")
   - При деактивации аккаунта > 30 дней

3. ПРИНЦИПЫ МИНИМИЗАЦИИ ДАННЫХ:
   - Храним только эмбеддинги, не исходные изображения
   - Шифрование AES-256-GCM с аутентификацией
   - Сжатие данных перед хранением (blosc2/lz4)

4. ПРАВА ПОЛЬЗОВАТЕЛЯ:
   - Запрос на экспорт данных (получение всех эмбеддингов)
   - Запрос на удаление данных (безвозвратное удаление)
   - Запрос на отзыв согласия (GDPR Article 7)

================================================================================
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

try:
    import blosc2  # type: ignore
    BLOSC2_AVAILABLE = True
except ImportError:
    blosc2 = None
    BLOSC2_AVAILABLE = False

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import StorageError, ValidationError, NotFoundError
from ..services.encryption_service import EncryptionService
from ..services.audit_service import AuditService
from ..db.database import get_async_db_manager
from ..db.models import BiometricTemplate

logger = get_logger(__name__)


class BiometricStorage:
    """
    Сервис для безопасного хранения биометрических шаблонов.
    
    Security Features:
    - AES-256-GCM шифрование с аутентификацией
    - Сжатие данных перед шифрованием (blosc2)
    - Версионирование шаблонов
    - Аудит всех операций чтения/записи
    
    Data Format:
    {
        "v": "1.0",           # Версия формата
        "user_id": "uuid",    # Идентификатор пользователя
        "embedding_dim": 512, # Размерность эмбеддинга
        "model_type": "FaceNet",  # Тип модели
        "created_at": "iso",  # Время создания
        "data": "<compressed_base64>"  # Сжатые данные
    }
    """

    def __init__(
        self,
        encryption_service: Optional[EncryptionService] = None,
        audit_service: Optional[AuditService] = None,
    ):
        """
        Args:
            encryption_service: Сервис шифрования (создается если не передан)
            audit_service: Сервис аудита (создается если не передан)
        """
        self.encryption = encryption_service or EncryptionService()
        self.audit = audit_service
        self.db_manager = get_async_db_manager()
        
        # Настройки сжатия
        self.compressor = getattr(settings, "BIOMETRIC_COMPRESSOR", "lz4")
        self.compress_level = getattr(settings, "BIOMETRIC_COMPRESS_LEVEL", 5)
        
        logger.info(
            f"BiometricStorage initialized | Compressor: {self.compressor} | "
            f"Level: {self.compress_level}"
        )

    async def save_template(
        self,
        user_id: str,
        embedding: np.ndarray,
        model_type: str = "FaceNet",
        metadata: Optional[Dict[str, Any]] = None,
        replace: bool = True,
    ) -> str:
        """
        Сохранить зашифрованный биометрический шаблон.
        
        Args:
            user_id: Идентификатор пользователя
            embedding: NumPy массив эмбеддинга
            model_type: Тип модели (FaceNet, ArcFace, etc.)
            metadata: Дополнительные метаданные
            replace: Заменить существующий шаблон
            
        Returns:
            template_id: UUID сохраненного шаблона
        """
        start_time = time.time()
        template_id = str(uuid.uuid4())
        
        try:
            # Валидация эмбеддинга
            if not isinstance(embedding, np.ndarray):
                raise ValidationError("Embedding must be a numpy array")
            
            if embedding.size == 0:
                raise ValidationError("Embedding cannot be empty")
            
            if embedding.ndim != 1:
                raise ValidationError(f"Embedding must be 1D array, got {embedding.ndim}D")
            
            # Подготовка данных
            data = self._prepare_data(
                user_id=user_id,
                embedding=embedding,
                model_type=model_type,
                metadata=metadata,
            )
            
            # Сжатие
            compressed = self._compress(data)
            
            # Шифрование
            encrypted = await self.encryption.encrypt(
                data=compressed,
                metadata={
                    "type": "biometric_template",
                    "user_id": user_id,
                    "model": model_type,
                }
            )
            
            # Сохранение в БД
            async with self.db_manager.get_session() as db:
                # Проверка наличия существующего шаблона
                existing = await db.execute(
                    f"SELECT id FROM {BiometricTemplate.__tablename__} WHERE user_id = :user_id",
                    {"user_id": user_id}
                )
                existing_id = existing.scalar_oneOrNone()
                
                if existing_id and replace:
                    # Обновление существующего
                    await db.execute(
                        f"""UPDATE {BiometricTemplate.__tablename__} 
                            SET encrypted_data = :encrypted, 
                                version = version + 1,
                                updated_at = :updated_at
                            WHERE user_id = :user_id""",
                        {
                            "encrypted": encrypted,
                            "updated_at": datetime.now(timezone.utc),
                            "user_id": user_id,
                        }
                    )
                    template_id = existing_id
                    operation = "updated"
                    
                elif existing_id and not replace:
                    raise StorageError(f"Template for user {user_id} already exists")
                    
                else:
                    # Создание нового
                    template = BiometricTemplate(
                        id=template_id,
                        user_id=user_id,
                        encrypted_data=encrypted,
                        embedding_dim=embedding.shape[0],
                        model_type=model_type,
                        metadata=metadata or {},
                    )
                    db.add(template)
                    operation = "created"
                    
                await db.commit()
            
            # Аудит
            if self.audit:
                await self.audit.log(
                    action="biometric_template_" + operation,
                    resource_type="biometric_template",
                    resource_id=template_id,
                    user_id=user_id,
                    details={
                        "embedding_dim": embedding.shape[0],
                        "model_type": model_type,
                        "encrypted_size": len(encrypted),
                        "processing_time": time.time() - start_time,
                    }
                )
            
            logger.info(
                f"Biometric template {operation}: user={user_id}, "
                f"dim={embedding.shape[0]}, time={time.time() - start_time:.3f}s"
            )
            
            return template_id
            
        except Exception as e:
            logger.error(f"Failed to save biometric template: {e}")
            raise StorageError(f"Failed to save template: {str(e)}")

    async def get_template(
        self,
        user_id: str,
        version: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Получить и расшифровать биометрический шаблон.
        
        Args:
            user_id: Идентификатор пользователя
            version: Конкретная версия (latest = None)
            
        Returns:
            Tuple[embedding, metadata]
        """
        start_time = time.time()
        
        try:
            async with self.db_manager.get_session() as db:
                if version is None:
                    result = await db.execute(
                        f"""SELECT encrypted_data, metadata, version 
                            FROM {BiometricTemplate.__tablename__} 
                            WHERE user_id = :user_id
                            ORDER BY version DESC LIMIT 1""",
                        {"user_id": user_id}
                    )
                else:
                    result = await db.execute(
                        f"""SELECT encrypted_data, metadata, version 
                            FROM {BiometricTemplate.__tablename__} 
                            WHERE user_id = :user_id AND version = :version""",
                        {"user_id": user_id, "version": version}
                    )
                
                row = result.fetchone()
                
                if not row:
                    raise NotFoundError(f"Template not found for user {user_id}")
                
                encrypted_data, metadata, template_version = row
            
            # Расшифровка
            decrypted = await self.encryption.decrypt(encrypted_data)
            data = self._decompress(decrypted)
            embedding = self._extract_embedding(data)
            
            # Аудит
            if self.audit:
                await self.audit.log(
                    action="biometric_template_retrieved",
                    resource_type="biometric_template",
                    resource_id=user_id,
                    user_id=user_id,
                    details={
                        "version": template_version,
                        "embedding_dim": embedding.shape[0],
                        "processing_time": time.time() - start_time,
                    }
                )
            
            logger.info(
                f"Biometric template retrieved: user={user_id}, "
                f"version={template_version}, time={time.time() - start_time:.3f}s"
            )
            
            return embedding, metadata
            
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get biometric template: {e}")
            raise StorageError(f"Failed to retrieve template: {str(e)}")

    async def delete_template(self, user_id: str) -> bool:
        """
        Удалить биометрический шаблон (soft delete).
        
        Args:
            user_id: Идентификатор пользователя
            
        Returns:
            True если удален, False если не найден
        """
        try:
            async with self.db_manager.get_session() as db:
                result = await db.execute(
                    f"""UPDATE {BiometricTemplate.__tablename__} 
                        SET is_active = False, updated_at = :updated_at
                        WHERE user_id = :user_id AND is_active = True""",
                    {
                        "user_id": user_id,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                await db.commit()
                
                deleted = result.rowcount > 0
            
            if deleted and self.audit:
                await self.audit.log(
                    action="biometric_template_deleted",
                    resource_type="biometric_template",
                    resource_id=user_id,
                    user_id=user_id,
                )
            
            logger.info(f"Biometric template deleted: user={user_id}, deleted={deleted}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete biometric template: {e}")
            raise StorageError(f"Failed to delete template: {str(e)}")

    async def list_versions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Получить список всех версий шаблонов пользователя.
        
        Args:
            user_id: Идентификатор пользователя
            
        Returns:
            Список версий с метаданными
        """
        try:
            async with self.db_manager.get_session() as db:
                result = await db.execute(
                    f"""SELECT id, version, embedding_dim, model_type, 
                               created_at, updated_at, is_active
                        FROM {BiometricTemplate.__tablename__} 
                        WHERE user_id = :user_id
                        ORDER BY version DESC""",
                    {"user_id": user_id}
                )
                rows = result.fetchall()
            
            return [
                {
                    "template_id": row[0],
                    "version": row[1],
                    "embedding_dim": row[2],
                    "model_type": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "updated_at": row[5].isoformat() if row[5] else None,
                    "is_active": row[6],
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Failed to list template versions: {e}")
            raise StorageError(f"Failed to list versions: {str(e)}")

    async def compare_templates(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Сравнить два эмбеддинга без сохранения в БД.
        
        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
            
        Returns:
            Dict с метриками схожести
        """
        # Нормализация
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Косинусная схожесть
        cosine_sim = float(np.dot(e1, e2))
        
        # Евклидово расстояние
        euclidean_dist = float(np.linalg.norm(e1 - e2))
        
        return {
            "cosine_similarity": round(cosine_sim, 6),
            "euclidean_distance": round(euclidean_dist, 6),
            "is_match": cosine_sim >= settings.VERIFICATION_THRESHOLD,
            "threshold_used": settings.VERIFICATION_THRESHOLD,
        }

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _prepare_data(
        self,
        user_id: str,
        embedding: np.ndarray,
        model_type: str,
        metadata: Optional[Dict[str, Any]],
    ) -> bytes:
        """
        Подготовить данные для сжатия.
        """
        data_dict = {
            "v": "1.0",
            "user_id": user_id,
            "embedding_dim": embedding.shape[0],
            "model_type": model_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data": embedding.tobytes().hex(),  # Hex для JSON совместимости
        }
        
        if metadata:
            data_dict["metadata"] = metadata
        
        return json.dumps(data_dict, separators=(",", ":")).encode("utf-8")

    def _compress(self, data: bytes) -> bytes:
        """Сжать данные перед шифрованием."""
        if not BLOSC2_AVAILABLE or blosc2 is None:
            logger.warning("blosc2 not available, storing uncompressed data")
            return data

        try:
            compressed = blosc2.compress(
                data,
                cname=self.compressor,
                clevel=self.compress_level,
                shuffle=blosc2.SHUFFLE,
            )
            return compressed
        except Exception:
            logger.warning("Compression failed, storing uncompressed data")
            return data

    def _decompress(self, data: bytes) -> bytes:
        """Расшифровать данные."""
        if not BLOSC2_AVAILABLE or blosc2 is None:
            return data

        try:
            return blosc2.decompress(data)
        except Exception:
            # Данные не были сжаты
            return data

    def _extract_embedding(self, data: bytes) -> np.ndarray:
        """Извлечь эмбеддинг из данных."""
        data_dict = json.loads(data.decode("utf-8"))
        
        embedding_hex = data_dict["data"]
        embedding = np.frombuffer(bytes.fromhex(embedding_hex), dtype=np.float32)
        
        return embedding

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику сервиса."""
        return {
            "compressor": self.compressor if BLOSC2_AVAILABLE else "none",
            "compress_level": self.compress_level if BLOSC2_AVAILABLE else 0,
            "blosc2_available": BLOSC2_AVAILABLE,
            "encryption_algorithm": "AES-256-GCM",
            "supports_versioning": True,
        }


# =============================================================================
# Singleton
# =============================================================================

_biometric_storage: Optional[BiometricStorage] = None


async def get_biometric_storage() -> BiometricStorage:
    """Получение singleton экземпляра BiometricStorage."""
    global _biometric_storage
    
    if _biometric_storage is None:
        _biometric_storage = BiometricStorage()
    
    return _biometric_storage


async def reset_biometric_storage() -> None:
    """Сброс singleton для тестирования."""
    global _biometric_storage
    _biometric_storage = None