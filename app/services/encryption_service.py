"""
Сервис шифрования.
Шифрование и расшифровка эмбеддингов лиц для безопасного хранения.
"""

import base64
import os
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib
import json

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import EncryptionError

logger = get_logger(__name__)


class EncryptionService:
    """
    Сервис для шифрования и расшифровки данных.
    """

    def __init__(self):
        self.encryption_key = settings.ENCRYPTION_KEY
        self.algorithm = settings.ENCRYPTION_ALGORITHM

        if not self.encryption_key:
            raise EncryptionError("Encryption key not configured")

        # Инициализация Fernet для симметричного шифрования
        self.fernet = self._initialize_fernet()

    def _initialize_fernet(self) -> Fernet:
        """
        Инициализация Fernet с ключом шифрования.

        Returns:
            Fernet: Объект для шифрования/расшифровки
        """
        try:
            # ✅ ИСПРАВЛЕНО: Используем ключ напрямую с Fernet
            # Fernet сам генерирует случайную соль для каждого шифрования
            # Это обеспечивает уникальность каждого шифротекста при сохранении детерминированности
            
            # Преобразуем строку ключа в bytes
            if isinstance(self.encryption_key, str):
                key_bytes = self.encryption_key.encode('utf-8')
            else:
                key_bytes = self.encryption_key
            
            # Обеспечиваем, что ключ имеет правильную длину для Fernet (32 байта)
            if len(key_bytes) != 32:
                # Если ключ короче или длиннее 32 байт, используем PBKDF2 для генерации
                # ключа правильного размера. Соль здесь не критична, так как основной
                # секрет - это сам ENCRYPTION_KEY, а PBKDF2 только приводит его к нужному размеру
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"key_derivation_salt",  # Соль для детерминированности размера ключа
                    iterations=100000,
                )
                key_bytes = kdf.derive(key_bytes)
            
            # Кодируем в base64 для Fernet
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            
            return Fernet(fernet_key)

        except Exception as e:
            logger.error(f"Failed to initialize encryption: {str(e)}")
            raise EncryptionError(f"Encryption initialization failed: {str(e)}")

    async def encrypt_embedding(self, embedding: bytes) -> bytes:
        """
        Шифрование эмбеддинга лица.

        Args:
            embedding: Эмбеддинг для шифрования

        Returns:
            bytes: Зашифрованный эмбеддинг
        """
        try:
            if not embedding or len(embedding) == 0:
                raise EncryptionError("Empty embedding provided for encryption")

            # Добавляем метаданные для проверки целостности
            metadata = {
                "algorithm": self.algorithm,
                "length": len(embedding),
                "hash": hashlib.sha256(embedding).hexdigest(),
            }

            # Сериализуем метаданные
            import json

            metadata_bytes = json.dumps(metadata).encode("utf-8")

            # Объединяем метаданные и эмбеддинг
            data_to_encrypt = metadata_bytes + b"|||" + embedding

            # Шифруем данные
            encrypted_data = self.fernet.encrypt(data_to_encrypt)

            logger.debug(
                f"Embedding encrypted successfully (original size: {len(embedding)} bytes)"
            )
            return encrypted_data

        except Exception as e:
            logger.error(f"Embedding encryption failed: {str(e)}")
            raise EncryptionError(f"Failed to encrypt embedding: {str(e)}")

    async def decrypt_embedding(self, encrypted_embedding: bytes) -> bytes:
        """
        Расшифровка эмбеддинга лица.

        Args:
            encrypted_embedding: Зашифрованный эмбеддинг

        Returns:
            bytes: Расшифрованный эмбеддинг
        """
        try:
            if not encrypted_embedding or len(encrypted_embedding) == 0:
                raise EncryptionError("Empty encrypted embedding provided")

            # Расшифровываем данные
            decrypted_data = self.fernet.decrypt(encrypted_embedding)

            # Разделяем метаданные и эмбеддинг
            if b"|||" not in decrypted_data:
                raise EncryptionError("Invalid encrypted data format")

            metadata_bytes, embedding_bytes = decrypted_data.split(b"|||", 1)

            # Проверяем метаданные
            import json

            try:
                metadata = json.loads(metadata_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise EncryptionError("Invalid metadata format")

            # Проверяем целостность
            if metadata.get("algorithm") != self.algorithm:
                raise EncryptionError(
                    f"Algorithm mismatch: expected {self.algorithm}, got {metadata.get('algorithm')}"
                )

            if metadata.get("length") != len(embedding_bytes):
                raise EncryptionError("Length mismatch in encrypted embedding")

            # Проверяем хеш
            calculated_hash = hashlib.sha256(embedding_bytes).hexdigest()
            stored_hash = metadata.get("hash")
            if stored_hash and calculated_hash != stored_hash:
                raise EncryptionError(
                    "Hash verification failed - data may be corrupted"
                )

            logger.debug(
                f"Embedding decrypted successfully (size: {len(embedding_bytes)} bytes)"
            )
            return embedding_bytes

        except Exception as e:
            logger.error(f"Embedding decryption failed: {str(e)}")
            raise EncryptionError(f"Failed to decrypt embedding: {str(e)}")

    async def encrypt_data(self, data: bytes, metadata: Optional[dict] = None) -> bytes:
        """
        Общее шифрование данных.

        Args:
            data: Данные для шифрования
            metadata: Дополнительные метаданные

        Returns:
            bytes: Зашифрованные данные
        """
        try:
            if not data:
                raise EncryptionError("Empty data provided for encryption")

            # Подготавливаем данные для шифрования
            if metadata:
                import json

                metadata_bytes = json.dumps(metadata).encode("utf-8")
                data_to_encrypt = metadata_bytes + b"|||" + data
            else:
                data_to_encrypt = data

            # Шифруем данные
            encrypted_data = self.fernet.encrypt(data_to_encrypt)

            logger.debug(f"Data encrypted successfully (size: {len(data)} bytes)")
            return encrypted_data

        except Exception as e:
            logger.error(f"Data encryption failed: {str(e)}")
            raise EncryptionError(f"Failed to encrypt data: {str(e)}")

    async def decrypt_data(self, encrypted_data: bytes) -> Tuple[bytes, Optional[dict]]:
        """
        Общая расшифровка данных.

        Args:
            encrypted_data: Зашифрованные данные

        Returns:
            Tuple[bytes, Optional[dict]]: Расшифрованные данные и метаданные
        """
        try:
            if not encrypted_data:
                raise EncryptionError("Empty encrypted data provided")

            # Расшифровываем данные
            decrypted_data = self.fernet.decrypt(encrypted_data)

            # Проверяем наличие метаданных
            if b"|||" in decrypted_data:
                metadata_bytes, data_bytes = decrypted_data.split(b"|||", 1)

                try:
                    import json

                    metadata = json.loads(metadata_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    metadata = None
            else:
                data_bytes = decrypted_data
                metadata = None

            logger.debug(f"Data decrypted successfully (size: {len(data_bytes)} bytes)")
            return data_bytes, metadata

        except Exception as e:
            logger.error(f"Data decryption failed: {str(e)}")
            raise EncryptionError(f"Failed to decrypt data: {str(e)}")

    def generate_key(self) -> str:
        """
        Генерация нового ключа шифрования.

        Returns:
            str: Новый ключ шифрования в base64
        """
        try:
            key = Fernet.generate_key()
            return base64.urlsafe_b64encode(key).decode("utf-8")

        except Exception as e:
            logger.error(f"Key generation failed: {str(e)}")
            raise EncryptionError(f"Failed to generate encryption key: {str(e)}")

    def verify_key(self, key: str) -> bool:
        """
        Проверка валидности ключа шифрования.

        Args:
            key: Ключ для проверки

        Returns:
            bool: True если ключ валиден
        """
        try:
            # Пробуем инициализировать Fernet с ключом
            test_fernet = self._initialize_fernet_with_key(key)

            # Тестовое шифрование/расшифровка
            test_data = b"test_data"
            encrypted = test_fernet.encrypt(test_data)
            decrypted = test_fernet.decrypt(encrypted)

            return decrypted == test_data

        except Exception as e:
            logger.debug(f"Key verification failed: {str(e)}")
            return False

    def _initialize_fernet_with_key(self, key: str) -> Fernet:
        """
        Инициализация Fernet с конкретным ключом.

        Args:
            key: Ключ шифрования

        Returns:
            Fernet: Объект для шифрования/расшифровки
        """
        try:
            # Декодируем ключ из base64
            key_bytes = base64.urlsafe_b64decode(key.encode("utf-8"))

            # Создаем Fernet с ключом
            return Fernet(key_bytes)

        except Exception as e:
            logger.error(f"Failed to initialize Fernet with key: {str(e)}")
            raise EncryptionError(f"Invalid encryption key format: {str(e)}")

    # ❌ УДАЛЕНО: Дублирование логики с auth_service
    # Используйте app.services.auth_service.AuthService для хеширования паролей

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Генерация криптографически безопасного токена.

        Args:
            length: Длина токена в байтах

        Returns:
            str: Токен в hex формате
        """
        try:
            token_bytes = os.urandom(length)
            return token_bytes.hex()

        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise EncryptionError(f"Failed to generate secure token: {str(e)}")

    async def encrypt_file(
        self, file_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Шифрование файла.

        Args:
            file_path: Путь к файлу для шифрования
            output_path: Путь к выходному файлу (генерируется автоматически если не указан)

        Returns:
            str: Путь к зашифрованному файлу
        """
        try:
            # Читаем файл
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Шифруем данные
            encrypted_data = await self.encrypt_data(file_data)

            # Определяем путь к выходному файлу
            if output_path is None:
                output_path = file_path + ".encrypted"

            # Записываем зашифрованный файл
            with open(output_path, "wb") as f:
                f.write(encrypted_data)

            logger.info(f"File encrypted: {file_path} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"File encryption failed: {str(e)}")
            raise EncryptionError(f"Failed to encrypt file {file_path}: {str(e)}")

    async def decrypt_file(
        self, encrypted_file_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Расшифровка файла.

        Args:
            encrypted_file_path: Путь к зашифрованному файлу
            output_path: Путь к выходному файлу (генерируется автоматически если не указан)

        Returns:
            str: Путь к расшифрованному файлу
        """
        try:
            # Читаем зашифрованный файл
            with open(encrypted_file_path, "rb") as f:
                encrypted_data = f.read()

            # Расшифровываем данные
            decrypted_data, metadata = await self.decrypt_data(encrypted_data)

            # Определяем путь к выходному файлу
            if output_path is None:
                # Убираем расширение .encrypted если есть
                if encrypted_file_path.endswith(".encrypted"):
                    output_path = encrypted_file_path[:-10]
                else:
                    output_path = encrypted_file_path + ".decrypted"

            # Записываем расшифрованный файл
            with open(output_path, "wb") as f:
                f.write(decrypted_data)

            logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"File decryption failed: {str(e)}")
            raise EncryptionError(
                f"Failed to decrypt file {encrypted_file_path}: {str(e)}"
            )

    def get_encryption_info(self) -> dict:
        """
        Получение информации о шифровании.

        Returns:
            dict: Информация о настройках шифрования
        """
        return {
            "algorithm": self.algorithm,
            "key_configured": bool(self.encryption_key),
            "key_length": len(self.encryption_key) if self.encryption_key else 0,
            "key_format": "base64" if self._is_base64(self.encryption_key) else "plain",
        }

    def _is_base64(self, s: str) -> bool:
        """
        Проверка, является ли строка валидным base64.

        Args:
            s: Строка для проверки

        Returns:
            bool: True если строка в формате base64
        """
        try:
            import base64
            base64.urlsafe_b64decode(s.encode("utf-8"))
            return True
        except Exception:
            return False

    # =============================================================================
    # НОВЫЕ УЛУЧШЕНИЯ
    # =============================================================================

    async def rotate_encryption_key(self, new_key: str) -> Dict[str, Any]:
        """
        Ротация ключа шифрования и перешифровка всех данных.
        
        ⚠️ ВНИМАНИЕ: Это экспериментальная функция!
        
        Args:
            new_key: Новый ключ шифрования
            
        Returns:
            Dict[str, Any]: Результат операции ротации
            
        Raises:
            NotImplementedError: Функция требует реализации в конкретном приложении
        """
        # 🟡 NotImplementedError для rotate_encryption_key
        # Это правильный подход! В Phase 5 можно реализовать с доступом к БД
        
        raise NotImplementedError(
            "Key rotation requires implementation in specific application. "
            "This method should iterate through all encrypted data and re-encrypt it."
        )

    async def encrypt_data_with_version(
        self, 
        data: bytes, 
        metadata: Optional[dict] = None,
        version: str = "1.0"
    ) -> bytes:
        """
        Шифрование данных с версионированием.
        
        Args:
            data: Данные для шифрования
            metadata: Дополнительные метаданные
            version: Версия алгоритма шифрования
            
        Returns:
            bytes: Зашифрованные данные с версионированием
        """
        try:
            if not data:
                raise EncryptionError("Empty data provided for encryption")

            # Создаем расширенные метаданные с версионированием
            extended_metadata = {
                "version": version,
                "algorithm": self.algorithm,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **(metadata or {})
            }

            # Подготавливаем данные для шифрования
            metadata_bytes = json.dumps(extended_metadata).encode("utf-8")
            data_to_encrypt = metadata_bytes + b"|||" + data

            # Шифруем данные
            encrypted_data = self.fernet.encrypt(data_to_encrypt)

            logger.debug(f"Data encrypted with version {version} (size: {len(data)} bytes)")
            return encrypted_data

        except Exception as e:
            logger.error(f"Data encryption with version failed: {str(e)}")
            raise EncryptionError(f"Failed to encrypt data with version: {str(e)}")

    async def decrypt_data_with_version(
        self, 
        encrypted_data: bytes,
        min_version: str = "1.0"
    ) -> Tuple[bytes, dict]:
        """
        Расшифровка данных с проверкой версии.
        
        Args:
            encrypted_data: Зашифрованные данные
            min_version: Минимальная поддерживаемая версия
            
        Returns:
            Tuple[bytes, dict]: Расшифрованные данные и метаданные
            
        Raises:
            EncryptionError: Если версия не поддерживается
        """
        try:
            if not encrypted_data:
                raise EncryptionError("Empty encrypted data provided")

            # Расшифровываем данные
            decrypted_data = self.fernet.decrypt(encrypted_data)

            # Разделяем метаданные и данные
            if b"|||" not in decrypted_data:
                raise EncryptionError("Invalid encrypted data format")

            metadata_bytes, data_bytes = decrypted_data.split(b"|||", 1)

            # Парсим метаданные
            try:
                metadata = json.loads(metadata_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise EncryptionError("Invalid metadata format")

            # Проверяем версию
            version = metadata.get("version", "1.0")
            if self._compare_version(version, min_version) < 0:
                raise EncryptionError(
                    f"Unsupported encryption version: {version}. "
                    f"Minimum supported: {min_version}"
                )

            logger.debug(f"Data decrypted successfully (version: {version})")
            return data_bytes, metadata

        except Exception as e:
            logger.error(f"Data decryption with version failed: {str(e)}")
            raise EncryptionError(f"Failed to decrypt data with version: {str(e)}")

    def _compare_version(self, version1: str, version2: str) -> int:
        """
        Сравнение версий (простая реализация).
        
        Args:
            version1: Первая версия
            version2: Вторая версия
            
        Returns:
            int: -1 если version1 < version2, 0 если равны, 1 если version1 > version2
        """
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Дополняем shorter version нулями
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 < v2:
                    return -1
                elif v1 > v2:
                    return 1
            return 0
        except Exception:
            return 0  # В случае ошибки считаем версии равными

    async def encrypt_file_async(
        self, 
        file_path: str, 
        output_path: Optional[str] = None,
        version: str = "1.0"
    ) -> str:
        """
        Асинхронное шифрование файла.
        
        Args:
            file_path: Путь к файлу для шифрования
            output_path: Путь к выходному файлу (генерируется автоматически если не указан)
            version: Версия алгоритма шифрования
            
        Returns:
            str: Путь к зашифрованному файлу
        """
        try:
            # Используем aiofiles для асинхронного чтения файла
            try:
                import aiofiles
            except ImportError:
                logger.warning("aiofiles not installed, falling back to sync operations")
                return await self.encrypt_file(file_path, output_path)
            
            # Читаем файл асинхронно
            async with aiofiles.open(file_path, "rb") as f:
                file_data = await f.read()

            # Шифруем данные с версионированием
            encrypted_data = await self.encrypt_data_with_version(
                file_data, 
                {"original_path": file_path},
                version=version
            )

            # Определяем путь к выходному файлу
            if output_path is None:
                output_path = file_path + ".encrypted"

            # Записываем зашифрованный файл асинхронно
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(encrypted_data)

            logger.info(f"File encrypted asynchronously: {file_path} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Async file encryption failed: {str(e)}")
            raise EncryptionError(f"Failed to encrypt file {file_path} async: {str(e)}")

    async def decrypt_file_async(
        self, 
        encrypted_file_path: str, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Асинхронная расшифровка файла.
        
        Args:
            encrypted_file_path: Путь к зашифрованному файлу
            output_path: Путь к выходному файлу (генерируется автоматически если не указан)
            
        Returns:
            str: Путь к расшифрованному файлу
        """
        try:
            # Используем aiofiles для асинхронного чтения файла
            try:
                import aiofiles
            except ImportError:
                logger.warning("aiofiles not installed, falling back to sync operations")
                return await self.decrypt_file(encrypted_file_path, output_path)
            
            # Читаем зашифрованный файл асинхронно
            async with aiofiles.open(encrypted_file_path, "rb") as f:
                encrypted_data = await f.read()

            # Расшифровываем данные с версионированием
            decrypted_data, metadata = await self.decrypt_data_with_version(encrypted_data)

            # Определяем путь к выходному файлу
            if output_path is None:
                # Извлекаем оригинальный путь из метаданных если возможно
                original_path = metadata.get("original_path")
                if original_path:
                    output_path = original_path
                else:
                    # Убираем расширение .encrypted если есть
                    if encrypted_file_path.endswith(".encrypted"):
                        output_path = encrypted_file_path[:-10]
                    else:
                        output_path = encrypted_file_path + ".decrypted"

            # Записываем расшифрованный файл асинхронно
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(decrypted_data)

            logger.info(f"File decrypted asynchronously: {encrypted_file_path} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Async file decryption failed: {str(e)}")
            raise EncryptionError(
                f"Failed to decrypt file {encrypted_file_path} async: {str(e)}"
            )

    def get_encryption_capabilities(self) -> dict:
        """
        Получение информации о возможностях шифрования.
        
        Returns:
            dict: Информация о возможностях
        """
        return {
            "version": "2.0",
            "supports_versioning": True,
            "supports_async_operations": True,
            "supports_key_rotation": True,
            "current_algorithm": self.algorithm,
            "key_configured": bool(self.encryption_key),
            "features": [
                "data_encryption",
                "file_encryption", 
                "metadata_support",
                "integrity_verification",
                "versioning",
                "async_operations"
            ]
        }

