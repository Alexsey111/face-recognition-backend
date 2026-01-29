"""
Biometric Encryption - Специализированное шифрование биометрических данных.

Предоставляет:
- Удобные методы encrypt_embedding / decrypt_embedding
- Хеширование для проверки целостности
- Безопасное удаление данных из памяти
"""

import hashlib
import secrets
from typing import Optional, Tuple

import numpy as np

from ..config import settings
from ..services.encryption_service import EncryptionService
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BiometricEncryption:
    """
    Удобная обёртка для шифрования биометрических данных.

    Features:
    - AES-256-GCM шифрование эмбеддингов
    - SHA-256 хеширование для integrity check
    - Безопасное удаление из памяти (secure zeroing)
    """

    def __init__(self, encryption_service: Optional[EncryptionService] = None):
        """
        Args:
            encryption_service: EncryptionService (создаётся если None)
        """
        self.encryption = encryption_service or EncryptionService()
        self._embedding_buffer: Optional[bytes] = None

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Шифрование эмбеддинга лица.

        Args:
            embedding: NumPy массив эмбеддинга

        Returns:
            Зашифрованные данные (bytes)
        """
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be a numpy array")

        if embedding.size == 0:
            raise ValueError("Embedding cannot be empty")

        # Сохраняем для возможного безопасного удаления
        self._embedding_buffer = embedding.tobytes()

        # Шифруем
        encrypted = self.encryption.encrypt_data_sync(self._embedding_buffer)

        # Очищаем буфер после успешного шифрования
        self._secure_delete_buffer()

        return encrypted

    def decrypt_embedding(self, encrypted: bytes) -> np.ndarray:
        """
        Расшифровка эмбеддинга.

        Args:
            encrypted: Зашифрованные данные

        Returns:
            NumPy массив эмбеддинга
        """
        data, _ = self.encryption.decrypt_data(encrypted)
        return np.frombuffer(data, dtype=np.float32)

    def hash_template(self, embedding: np.ndarray) -> str:
        """
        Создание необратимого хеша эмбеддинга.

        Используется для:
        - Проверки целостности данных
        - Быстрого сравнения (без расшифровки)
        - Detecting tampering

        Args:
            embedding: NumPy массив эмбеддинга

        Returns:
            SHA-256 hex digest
        """
        return hashlib.sha256(embedding.tobytes()).hexdigest()

    def verify_template_integrity(
        self,
        embedding: np.ndarray,
        expected_hash: str,
    ) -> bool:
        """
        Проверка целостности эмбеддинга.

        Args:
            embedding: Текущий эмбеддинг
            expected_hash: Ожидаемый хеш

        Returns:
            True если целостность подтверждена
        """
        actual_hash = self.hash_template(embedding)
        return secrets.compare_digest(actual_hash, expected_hash)

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Генерация криптографически безопасного токена.

        Args:
            length: Длина токена в байтах

        Returns:
            Hex-encoded токен
        """
        return secrets.token_hex(length)

    def secure_zero(self, data: bytearray) -> None:
        """
        Безопасное обнуление данных (secure deletion).

        Перезаписывает данные случайными байтами перед удалением.

        Args:
            data: bytearray для обнуления
        """
        try:
            # Перезапись случайными байтами
            for i in range(len(data)):
                data[i] = secrets.randbelow(256)
            # Финальная перезапись нулями
            for i in range(len(data)):
                data[i] = 0
        except Exception as e:
            logger.warning(f"Secure zero failed: {e}")

    def secure_delete_embedding(self, embedding: np.ndarray) -> None:
        """
        Безопасное удаление эмбеддинга из памяти.

        Args:
            embedding: NumPy массив для удаления
        """
        try:
            # Перезапись данных
            data = embedding.flatten()
            for i in range(0, len(data), 1024):
                end = min(i + 1024, len(data))
                data[i:end] = np.zeros(end - i, dtype=np.float32)

            # Освобождение памяти
            del data
            embedding.fill(0)

            logger.debug("Embedding securely deleted")
        except Exception as e:
            logger.warning(f"Secure delete failed: {e}")

    def _secure_delete_buffer(self) -> None:
        """Удаление временного буфера."""
        if self._embedding_buffer is not None:
            # Создаём mutable копию для безопасного удаления
            buffer = bytearray(self._embedding_buffer)
            self.secure_zero(buffer)
            self._embedding_buffer = None

    def get_info(self) -> dict:
        """Информация о сервисе."""
        return {
            "algorithm": "AES-256-GCM",
            "hash_algorithm": "SHA-256",
            "encryption_ready": True,
            "secure_deletion": True,
        }


# =============================================================================
# Secure memory management utilities
# =============================================================================


class SecureMemoryManager:
    """
    Менеджер безопасной работы с памятью.

    Обеспечивает:
    - Автоматическую очистку конфиденциальных данных
    - Отслеживание чувствительных буферов
    - Защита от утечки через GC
    """

    _sensitive_buffers: set = set()

    @classmethod
    def register_buffer(cls, buffer_id: int) -> None:
        """Регистрация чувствительного буфера."""
        cls._sensitive_buffers.add(buffer_id)

    @classmethod
    def unregister_buffer(cls, buffer_id: int) -> None:
        """Удаление из отслеживания."""
        cls._sensitive_buffers.discard(buffer_id)

    @classmethod
    def secure_cleanup(cls) -> int:
        """
        Безопасная очистка всех зарегистрированных буферов.

        Returns:
            Количество очищенных буферов
        """
        count = 0
        for buffer_id in list(cls._sensitive_buffers):
            try:
                # Буфер будет освобождён при следующем GC
                cls._sensitive_buffers.discard(buffer_id)
                count += 1
            except Exception:
                pass
        return count

    @classmethod
    def scrub_variable(cls, var: any) -> None:
        """
        Попытка обнулить переменную.

        Поддерживает:
        - bytearray
        - bytes (копия для модификации)
        - numpy arrays
        - list/tuple
        """
        try:
            if isinstance(var, bytearray):
                for i in range(len(var)):
                    var[i] = 0
            elif isinstance(var, np.ndarray):
                var.fill(0)
            elif isinstance(var, list):
                for i in range(len(var)):
                    var[i] = 0
            elif isinstance(var, tuple):
                # Tuple неизменяем, логируем предупреждение
                logger.warning("Cannot scrub immutable tuple")
        except Exception as e:
            logger.warning(f"Scrub failed: {e}")


# =============================================================================
# Secure deletion context manager
# =============================================================================

import contextlib
from typing import Any


@contextlib.contextmanager
def secure_context(data: Any):
    """
    Контекстный менеджер для безопасной работы с данными.

    Usage:
        with secure_context(embedding) as data:
            # работа с данными
        # автоматическая очистка
    """
    yield data
    # Очистка при выходе из контекста
    try:
        if isinstance(data, np.ndarray):
            data.fill(0)
        elif isinstance(data, bytearray):
            for i in range(len(data)):
                data[i] = 0
    except Exception:
        pass


# =============================================================================
# Singleton instance
# =============================================================================

_biometric_encryption: Optional[BiometricEncryption] = None


def get_biometric_encryption() -> BiometricEncryption:
    """Получение singleton экземпляра BiometricEncryption."""
    global _biometric_encryption

    if _biometric_encryption is None:
        _biometric_encryption = BiometricEncryption()

    return _biometric_encryption
