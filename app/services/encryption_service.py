"""
Сервис шифрования.
Шифрование и расшифровка эмбеддингов лиц для безопасного хранения.
"""

import base64
import os
from typing import Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib

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
            # Генерируем ключ из строки с помощью PBKDF2
            password = self.encryption_key.encode()
            salt = b"face_recognition_salt_2024"  # Фиксированная соль для детерминированности

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))

            return Fernet(key)

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

    async def hash_password(
        self, password: str, salt: Optional[bytes] = None
    ) -> Tuple[str, str]:
        """
        Хеширование пароля с солью.

        Args:
            password: Пароль для хеширования
            salt: Соль (генерируется автоматически если не указана)

        Returns:
            Tuple[str, str]: Хеш пароля и соль в base64
        """
        try:
            if salt is None:
                salt = os.urandom(32)

            # Хешируем пароль с солью
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            hash_bytes = kdf.derive(password.encode("utf-8"))

            return (
                base64.urlsafe_b64encode(hash_bytes).decode("utf-8"),
                base64.urlsafe_b64encode(salt).decode("utf-8"),
            )

        except Exception as e:
            logger.error(f"Password hashing failed: {str(e)}")
            raise EncryptionError(f"Failed to hash password: {str(e)}")

    async def verify_password(
        self, password: str, hashed_password: str, salt: str
    ) -> bool:
        """
        Проверка пароля против хеша.

        Args:
            password: Пароль для проверки
            hashed_password: Хеш пароля
            salt: Соль в base64

        Returns:
            bool: True если пароль корректен
        """
        try:
            # Декодируем хеш и соль
            hash_bytes = base64.urlsafe_b64decode(hashed_password.encode("utf-8"))
            salt_bytes = base64.urlsafe_b64decode(salt.encode("utf-8"))

            # Хешируем введенный пароль с той же солью
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            calculated_hash = kdf.derive(password.encode("utf-8"))

            # Сравниваем хеши
            return calculated_hash == hash_bytes

        except Exception as e:
            logger.error(f"Password verification failed: {str(e)}")
            return False

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
            base64.urlsafe_b64decode(s.encode("utf-8"))
            return True
        except Exception:
            return False
