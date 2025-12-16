"""
Тесты для сервиса шифрования.
Проверка функций шифрования/дешифрования и безопасности.
"""

import pytest
import numpy as np
import base64
import json
import os
from unittest.mock import Mock, patch, AsyncMock

from app.services.encryption_service import EncryptionService
from app.utils.exceptions import EncryptionError
from app.config import settings


class TestEncryptionService:
    """Тесты для EncryptionService."""

    @pytest.fixture
    def encryption_service(self):
        """Создание экземпляра сервиса шифрования для тестов."""
        return EncryptionService()

    @pytest.fixture
    def sample_embedding(self):
        """Создание примера эмбеддинга для тестов."""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    @pytest.fixture
    def sample_data(self):
        """Создание примера данных для тестов."""
        return b"sample_data_for_testing"

    def test_initialization_success(self):
        """Тест успешной инициализации сервиса."""
        service = EncryptionService()
        assert service.encryption_key is not None
        assert service.algorithm is not None
        assert service.fernet is not None

    def test_initialization_without_key(self, monkeypatch):
        """Тест инициализации без ключа шифрования."""
        # Убираем ключ шифрования
        monkeypatch.setattr(settings, 'ENCRYPTION_KEY', None)
        
        with pytest.raises(EncryptionError, match="Encryption key not configured"):
            EncryptionService()

    @pytest.mark.asyncio
    async def test_encrypt_embedding_success(self, encryption_service, sample_embedding):
        """Тест успешного шифрования эмбеддинга."""
        embedding_bytes = sample_embedding.tobytes()
        
        encrypted = await encryption_service.encrypt_embedding(embedding_bytes)
        
        assert encrypted is not None
        assert isinstance(encrypted, bytes)
        assert len(encrypted) > len(embedding_bytes)  # Зашифрованные данные больше

    @pytest.mark.asyncio
    async def test_encrypt_empty_embedding(self, encryption_service):
        """Тест шифрования пустого эмбеддинга."""
        with pytest.raises(EncryptionError, match="Empty embedding provided"):
            await encryption_service.encrypt_embedding(b"")

    @pytest.mark.asyncio
    async def test_encrypt_none_embedding(self, encryption_service):
        """Тест шифрования None эмбеддинга."""
        with pytest.raises(EncryptionError, match="Empty embedding provided"):
            await encryption_service.encrypt_embedding(None)

    @pytest.mark.asyncio
    async def test_decrypt_embedding_success(self, encryption_service, sample_embedding):
        """Тест успешного дешифрования эмбеддинга."""
        embedding_bytes = sample_embedding.tobytes()
        
        # Шифруем
        encrypted = await encryption_service.encrypt_embedding(embedding_bytes)
        
        # Дешифруем
        decrypted = await encryption_service.decrypt_embedding(encrypted)
        
        assert decrypted == embedding_bytes
        np.testing.assert_array_equal(
            np.frombuffer(decrypted, dtype=np.float32),
            sample_embedding
        )

    @pytest.mark.asyncio
    async def test_decrypt_invalid_data(self, encryption_service):
        """Тест дешифрования невалидных данных."""
        invalid_data = b"invalid_encrypted_data"
        
        with pytest.raises(EncryptionError):
            await encryption_service.decrypt_embedding(invalid_data)

    @pytest.mark.asyncio
    async def test_decrypt_empty_data(self, encryption_service):
        """Тест дешифрования пустых данных."""
        with pytest.raises(EncryptionError, match="Empty encrypted embedding provided"):
            await encryption_service.decrypt_embedding(b"")

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_data_with_metadata(self, encryption_service, sample_data):
        """Тест шифрования данных с метаданными."""
        metadata = {"version": "1.0", "type": "test"}
        
        # Шифруем с метаданными
        encrypted = await encryption_service.encrypt_data(sample_data, metadata)
        
        # Дешифруем
        decrypted, decrypted_metadata = await encryption_service.decrypt_data(encrypted)
        
        assert decrypted == sample_data
        assert decrypted_metadata == metadata

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_data_without_metadata(self, encryption_service, sample_data):
        """Тест шифрования данных без метаданных."""
        # Шифруем без метаданных
        encrypted = await encryption_service.encrypt_data(sample_data)
        
        # Дешифруем
        decrypted, metadata = await encryption_service.decrypt_data(encrypted)
        
        assert decrypted == sample_data
        assert metadata is None

    def test_generate_secure_token(self, encryption_service):
        """Тест генерации безопасного токена (encryption_service version)."""
        token = encryption_service.generate_secure_token(32)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) == 64  # 32 байта в hex = 64 символа

    def test_generate_key(self, encryption_service):
        """Тест генерации ключа шифрования."""
        key = encryption_service.generate_key()
        
        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0

    def test_verify_valid_key(self, encryption_service):
        """Тест проверки валидного ключа."""
        # Генерируем валидный ключ
        valid_key = encryption_service.generate_key()
        
        # Проверяем его
        is_valid = encryption_service.verify_key(valid_key)
        
        assert is_valid is True

    def test_verify_invalid_key(self, encryption_service):
        """Тест проверки невалидного ключа."""
        invalid_key = "invalid_key_123"
        
        is_valid = encryption_service.verify_key(invalid_key)
        
        assert is_valid is False

    def test_get_encryption_info(self, encryption_service):
        """Тест получения информации о шифровании."""
        info = encryption_service.get_encryption_info()
        
        assert "algorithm" in info
        assert "key_configured" in info
        assert "key_length" in info
        assert "key_format" in info
        assert info["algorithm"] is not None
        assert info["key_configured"] is True
        assert info["key_length"] > 0

    @pytest.mark.asyncio
    async def test_encrypt_file_success(self, encryption_service, tmp_path):
        """Тест успешного шифрования файла."""
        # Создаем тестовый файл
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Шифруем файл
        encrypted_file = await encryption_service.encrypt_file(str(test_file))
        
        assert encrypted_file.endswith(".encrypted")
        assert encrypted_file != str(test_file)
        
        # Проверяем, что зашифрованный файл существует
        import os
        assert os.path.exists(encrypted_file)

    @pytest.mark.asyncio
    async def test_decrypt_file_success(self, encryption_service, tmp_path):
        """Тест успешного дешифрования файла."""
        # Создаем тестовый файл
        test_file = tmp_path / "test.txt"
        test_content = "test content"
        test_file.write_text(test_content)
        
        # Шифруем файл
        encrypted_file = await encryption_service.encrypt_file(str(test_file))
        
        # Дешифруем файл
        decrypted_file = await encryption_service.decrypt_file(encrypted_file)
        
        # Проверяем содержимое
        assert os.path.exists(decrypted_file)
        with open(decrypted_file, 'r') as f:
            decrypted_content = f.read()
        assert decrypted_content == test_content

    @pytest.mark.asyncio
    async def test_encrypt_file_not_found(self, encryption_service):
        """Тест шифрования несуществующего файла."""
        non_existent_file = "/path/to/non/existent/file.txt"
        
        with pytest.raises(EncryptionError):
            await encryption_service.encrypt_file(non_existent_file)


class TestEncryptionSecurity:
    """Тесты безопасности шифрования."""

    @pytest.fixture
    def encryption_service(self):
        return EncryptionService()

    @pytest.mark.asyncio
    async def test_encryption_consistency(self, encryption_service):
        """Тест консистентности шифрования (одинаковые данные -> разные шифротексты)."""
        data = b"same_data_for_testing"
        
        # Шифруем одни и те же данные дважды
        encrypted1 = await encryption_service.encrypt_data(data)
        encrypted2 = await encryption_service.encrypt_data(data)
        
        # Шифротексты должны быть разными (из-за случайности)
        assert encrypted1 != encrypted2
        
        # Но дешифровка должна давать одинаковый результат
        decrypted1, _ = await encryption_service.decrypt_data(encrypted1)
        decrypted2, _ = await encryption_service.decrypt_data(encrypted2)
        
        assert decrypted1 == decrypted2 == data

    @pytest.mark.asyncio
    async def test_integrity_verification(self, encryption_service):
        """Тест проверки целостности данных."""
        data = b"test_data_for_integrity"
        
        # Шифруем данные
        encrypted = await encryption_service.encrypt_data(data)
        
        # Изменяем зашифрованные данные
        modified_encrypted = bytearray(encrypted)
        modified_encrypted[0] ^= 1  # Инвертируем первый бит
        
        # Попытка дешифрации должна провалиться
        with pytest.raises(EncryptionError):
            await encryption_service.decrypt_data(bytes(modified_encrypted))

    @pytest.mark.asyncio
    async def test_metadata_integrity(self, encryption_service):
        """Тест целостности метаданных."""
        data = b"test_data"
        metadata = {"version": "1.0", "type": "test"}
        
        # Шифруем с метаданными
        encrypted = await encryption_service.encrypt_data(data, metadata)
        
        # Проверяем, что метаданные не изменились при дешифрации
        decrypted_data, decrypted_metadata = await encryption_service.decrypt_data(encrypted)
        
        assert decrypted_data == data
        assert decrypted_metadata == metadata

    def test_key_derivation_consistency(self, encryption_service):
        """Тест консистентности генерации ключей."""
        # Проверяем, что с одинаковым ключом получаем одинаковые результаты
        test_key = encryption_service.encryption_key
        
        # Создаем два сервиса с одинаковым ключом
        service1 = EncryptionService()
        service2 = EncryptionService()
        
        # Оба должны иметь одинаковые настройки
        assert service1.algorithm == service2.algorithm

    # ❌ УДАЛЕНО: Дублирование логики с auth_service
    # Тесты для хеширования паролей находятся в test_auth.py


if __name__ == "__main__":
    pytest.main([__file__])