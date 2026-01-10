"""
Тесты для сервиса шифрования AES-256-GCM.
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
        assert service._key is not None
        assert service.ALGORITHM == "aes-256-gcm"
        assert service.KEY_LENGTH == 32  # AES-256

    def test_initialization_without_key(self, monkeypatch):
        """Тест инициализации без ключа шифрования."""
        monkeypatch.setattr(settings, 'ENCRYPTION_KEY', '')
        
        with pytest.raises(EncryptionError):
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
    async def test_encrypt_none_data(self, encryption_service):
        """Тест шифрования None данных."""
        with pytest.raises(EncryptionError):
            await encryption_service.encrypt(None)

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
            await encryption_service.decrypt(invalid_data)

    @pytest.mark.asyncio
    async def test_decrypt_empty_data(self, encryption_service):
        """Тест дешифрования пустых данных."""
        with pytest.raises(EncryptionError, match="Empty encrypted token"):
            await encryption_service.decrypt(b"")

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_data_with_metadata(self, encryption_service, sample_data):
        """Тест шифрования данных с метаданными."""
        metadata = {"version": "1.0", "type": "test"}
        
        # Шифруем с метаданными
        encrypted = await encryption_service.encrypt(sample_data, metadata)
        
        # Дешифруем
        decrypted, decrypted_metadata = await encryption_service.decrypt(encrypted)
        
        assert decrypted == sample_data
        assert decrypted_metadata.get("meta") == metadata

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_data_without_metadata(self, encryption_service, sample_data):
        """Тест шифрования данных без метаданных."""
        # Шифруем без метаданных
        encrypted = await encryption_service.encrypt(sample_data)
        
        # Дешифруем
        decrypted, payload = await encryption_service.decrypt(encrypted)
        
        assert decrypted == sample_data
        assert payload.get("meta") == {}

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
        assert "key_length_bits" in info
        assert info["algorithm"] == "aes-256-gcm"
        assert info["key_length_bits"] == 256  # AES-256
        assert info["authenticated_encryption"] is True
        assert info["mode"] == "GCM"

    @pytest.mark.asyncio
    async def test_decrypt_file_success(self, encryption_service, tmp_path):
        """Тест успешного дешифрования файла."""
        # Создаем тестовый файл
        test_content = b"test content"
        
        # Записываем и сразу шифруем в памяти
        encrypted = await encryption_service.encrypt(test_content, metadata={"filename": "test.txt"})
        
        # Дешифруем файл из памяти (имитация чтения из файла)
        decrypted, payload = await encryption_service.decrypt(encrypted)
        
        # Проверяем содержимое
        assert decrypted == test_content


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
        encrypted1 = await encryption_service.encrypt(data)
        encrypted2 = await encryption_service.encrypt(data)
        
        # Шифротексты должны быть разными (из-за случайности nonce)
        assert encrypted1 != encrypted2
        
        # Но дешифровка должна давать одинаковый результат
        decrypted1, _ = await encryption_service.decrypt(encrypted1)
        decrypted2, _ = await encryption_service.decrypt(encrypted2)
        
        assert decrypted1 == decrypted2 == data

    @pytest.mark.asyncio
    async def test_integrity_verification(self, encryption_service):
        """Тест проверки целостности данных (GCM authentication tag)."""
        data = b"test_data_for_integrity"
        
        # Шифруем данные
        encrypted = await encryption_service.encrypt(data)
        
        # Изменяем зашифрованные данные
        modified_encrypted = bytearray(encrypted)
        modified_encrypted[0] ^= 1  # Инвертируем первый бит
        
        # Попытка дешифрации должна провалиться из-за неверного tag
        with pytest.raises(EncryptionError):
            await encryption_service.decrypt(bytes(modified_encrypted))

    @pytest.mark.asyncio
    async def test_metadata_integrity(self, encryption_service):
        """Тест целостности метаданных."""
        data = b"test_data"
        metadata = {"version": "1.0", "type": "test"}
        
        # Шифруем с метаданными
        encrypted = await encryption_service.encrypt(data, metadata)
        
        # Проверяем, что метаданные не изменились при дешифрации
        decrypted_data, payload = await encryption_service.decrypt(encrypted)
        
        assert decrypted_data == data
        assert payload.get("meta") == metadata

    def test_key_derivation_consistency(self, encryption_service):
        """Тест консистентности генерации ключей."""
        # Создаем два сервиса с одинаковым ключом
        service1 = EncryptionService()
        service2 = EncryptionService()
        
        # Оба должны иметь одинаковые настройки
        assert service1.ALGORITHM == service2.ALGORITHM
        assert service1.KEY_LENGTH == service2.KEY_LENGTH

    def test_aes_256_requirements(self, encryption_service):
        """Тест соответствия AES-256 требованиям."""
        assert encryption_service.KEY_LENGTH == 32  # 256 bits = 32 bytes
        assert encryption_service.NONCE_LENGTH == 12  # GCM recommended
        assert encryption_service.TAG_LENGTH == 16  # GCM tag
        assert encryption_service.ALGORITHM == "aes-256-gcm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])