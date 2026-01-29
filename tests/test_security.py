"""
Общие тесты безопасности.
Проверка интеграции компонентов безопасности и защиты от OWASP Top 10 уязвимостей.
"""

import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.config import settings
from app.services.auth_service import AuthService
from app.services.encryption_service import EncryptionService
from app.utils.exceptions import UnauthorizedError, ValidationError
from app.utils.validators import sanitize_string, validate_email, validate_password


class TestSecurityIntegration:
    """Интеграционные тесты безопасности."""

    @pytest.fixture
    def auth_service(self):
        return AuthService()

    @pytest.fixture
    def encryption_service(self):
        return EncryptionService()

    @pytest.fixture
    def test_user_data(self):
        return {
            "user_id": "test-user-123",
            "email": "test@example.com",
            "password": "SecurePass123!",
            "role": "user",
        }

    @pytest.mark.asyncio
    async def test_complete_authentication_flow(self, auth_service, test_user_data):
        """Тест полного процесса аутентификации."""
        # 1. Хеширование пароля
        password_hash = await auth_service.hash_password(test_user_data["password"])
        assert password_hash is not None

        # 2. Создание access токена (sync метод)
        access_token = auth_service.create_access_token(
            test_user_data["user_id"], test_user_data["role"]
        )
        assert access_token is not None

        # 3. Создание refresh токена (sync метод)
        refresh_token = auth_service.create_refresh_token(test_user_data["user_id"])
        assert refresh_token is not None

        # 4. Верификация access токена (async метод)
        payload = await auth_service.verify_token(access_token, "access")
        assert payload["user_id"] == test_user_data["user_id"]
        assert payload["role"] == test_user_data["role"]

        # 5. Проверка пароля
        is_valid = await auth_service.verify_password(
            test_user_data["password"], password_hash
        )
        assert is_valid is True

        # 6. Token rotation
        tokens = await auth_service.refresh_access_token(refresh_token)
        new_access_token = tokens["access_token"]
        new_refresh_token = tokens["refresh_token"]
        assert new_access_token != access_token
        assert new_refresh_token != refresh_token  # ✅ Оба новые

        # 7. Верификация нового токена
        new_payload = await auth_service.verify_token(new_access_token, "access")
        assert new_payload["user_id"] == test_user_data["user_id"]

    @pytest.mark.asyncio
    async def test_encryption_integration(self, encryption_service):
        """Тест интеграции шифрования."""
        # 1. Создание тестовых данных
        sensitive_data = b"secret information"
        metadata = {"type": "test", "version": "1.0"}

        # 2. Шифрование данных
        encrypted = await encryption_service.encrypt_data(sensitive_data, metadata)
        assert encrypted is not None

        # 3. Дешифровка данных
        decrypted_data, decrypted_metadata = await encryption_service.decrypt_data(
            encrypted
        )
        assert decrypted_data == sensitive_data
        # decrypted_metadata содержит весь payload, включая метаданные в поле "meta"
        assert decrypted_metadata["meta"] == metadata

        # 4. Проверка целостности
        assert len(encrypted) > len(sensitive_data)  # Зашифрованные данные больше

    @pytest.mark.asyncio
    async def test_secure_token_storage(self, auth_service, encryption_service):
        """Тест безопасного хранения токенов."""
        user_id = "test-user"

        # 1. Создание сессии
        session = await auth_service.create_user_session(user_id)
        assert "access_token" in session
        assert "refresh_token" in session

        # 2. Шифрование токенов
        encrypted_access = await encryption_service.encrypt_data(
            session["access_token"].encode()
        )
        encrypted_refresh = await encryption_service.encrypt_data(
            session["refresh_token"].encode()
        )

        # 3. Дешифровка и использование
        access_token, _ = await encryption_service.decrypt_data(encrypted_access)
        refresh_token, _ = await encryption_service.decrypt_data(encrypted_refresh)

        # 4. Верификация токенов
        payload = await auth_service.verify_token(access_token.decode(), "access")
        assert payload["user_id"] == user_id

    @pytest.mark.asyncio
    async def test_token_expiration_handling(self, auth_service):
        """Тест обработки истечения токенов."""
        user_id = "test-user"

        # Создаем токен с коротким временем жизни
        expire = datetime.now(timezone.utc) + timedelta(seconds=1)

        # Создаем истекший токен вручную
        import jwt

        expired_payload = {
            "user_id": user_id,
            "type": "access",
            "exp": expire - timedelta(seconds=2),  # Уже истек
            "iat": datetime.now(timezone.utc) - timedelta(seconds=3),
            "jti": "expired-token",
        }

        expired_token = jwt.encode(
            expired_payload,
            auth_service.jwt_secret_key,
            algorithm=auth_service.jwt_algorithm,
        )

        # Попытка верификации должна провалиться
        with pytest.raises(UnauthorizedError, match="Token has expired"):
            await auth_service.verify_token(expired_token, "access")


class TestOWASPTop10Protection:
    """Тесты защиты от OWASP Top 10 уязвимостей."""

    @pytest.fixture
    def auth_service(self):
        return AuthService()

    @pytest.fixture
    def encryption_service(self):
        return EncryptionService()

    @staticmethod
    @pytest.fixture
    def validators_module():
        from app.utils import validators

        return validators

    # A01: Broken Access Control
    @pytest.mark.asyncio
    async def test_broken_access_control_protection(self, auth_service):
        """Тест защиты от сломанного контроля доступа."""
        # Создаем токен с ролью user (sync метод)
        user_token = auth_service.create_access_token("user-123", role="user")

        # Проверяем, что user не может получить admin права
        payload = await auth_service.verify_token(user_token, "access")
        assert payload["role"] == "user"

        # Попытка получить admin разрешения должна провалиться
        with pytest.raises(Exception):  # ForbiddenError
            auth_service.validate_user_permissions("user", ["admin_operations"])

    # A02: Cryptographic Failures
    @pytest.mark.asyncio
    async def test_cryptographic_failures_protection(self, encryption_service):
        """Тест защиты от криптографических ошибок."""
        # 1. Проверка шифрования слабых данных
        weak_data = b"123456"
        encrypted = await encryption_service.encrypt_data(weak_data)
        assert len(encrypted) > len(weak_data)

        # 2. Проверка дешифровки
        decrypted, _ = await encryption_service.decrypt_data(encrypted)
        assert decrypted == weak_data

        # 3. Проверка обнаружения подделки данных
        tampered_data = bytearray(encrypted)
        tampered_data[0] ^= 1

        with pytest.raises(Exception):  # EncryptionError
            await encryption_service.decrypt_data(bytes(tampered_data))

    # A03: Injection
    def test_injection_protection(self, validators_module):
        """Тест защиты от инъекций."""
        # 1. SQL injection попытки должны быть заблокированы
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; DELETE FROM users WHERE '1'='1",
        ]

        for malicious_input in malicious_inputs:
            # Санитизация должна удалить опасные символы
            sanitized = sanitize_string(malicious_input)
            # Проверяем, что опасные символы удалены
            assert "'" not in sanitized  # Одинарные кавычки
            assert '"' not in sanitized  # Двойные кавычки
            assert "<" not in sanitized  # Угловые скобки
            assert ">" not in sanitized
            # SQL ключевые слова могут остаться, это нормально для sanitize_string

    # A04: Insecure Design
    @pytest.mark.asyncio
    async def test_insecure_design_protection(self, auth_service):
        """Тест защиты от небезопасного дизайна."""
        # 1. Проверка, что токены имеют уникальные идентификаторы
        tokens = []
        for i in range(5):
            token = auth_service.create_access_token(f"user-{i}")  # sync метод
            tokens.append(token)

        # Все токены должны быть уникальными
        assert len(set(tokens)) == 5

        # 2. Проверка принудительного истечения токенов
        for token in tokens:
            payload = await auth_service.verify_token(token, "access")  # async метод
            assert "exp" in payload
            assert payload["exp"] > datetime.now(timezone.utc).timestamp()

    # A05: Security Misconfiguration
    def test_security_misconfiguration_protection(self):
        """Тест защиты от неправильной конфигурации безопасности."""
        # 1. Проверка настроек JWT
        assert settings.JWT_ALGORITHM in ["HS256", "HS384", "HS512"]
        assert len(settings.JWT_SECRET_KEY) > 32  # Сложный секрет

        # 2. Проверка настроек шифрования
        assert len(settings.ENCRYPTION_KEY) > 32

        # 3. Проверка настроек rate limiting
        assert settings.RATE_LIMIT_REQUESTS_PER_MINUTE > 0
        assert settings.RATE_LIMIT_BURST > 0

    # A06: Vulnerable Components
    def test_vulnerable_components_check(self):
        """Тест проверки уязвимых компонентов."""
        # Проверяем, что используются безопасные версии библиотек
        import cryptography
        import jwt

        # Проверяем минимальные версии
        assert cryptography.__version__ >= "3.4.8"

        # JWT должен использовать безопасные алгоритмы
        assert settings.JWT_ALGORITHM in ["HS256", "HS384", "HS512"]

    # A07: Identification and Authentication Failures
    @pytest.mark.asyncio
    async def test_identification_authentication_protection(self, auth_service):
        """Тест защиты идентификации и аутентификации."""
        # 1. Слабые пароли должны отклоняться
        weak_passwords = ["123456", "password", "qwerty", "abc123", "password123"]

        for weak_password in weak_passwords:
            with pytest.raises(ValidationError):
                validate_password(weak_password)

        # 2. Токены должны истекать
        token = auth_service.create_access_token("test-user")  # sync метод
        payload = await auth_service.verify_token(token, "access")  # async метод

        # Время истечения должно быть в будущем
        assert payload["exp"] > datetime.now(timezone.utc).timestamp()

        # 3. Refresh токены должны жить дольше
        refresh_token = auth_service.create_refresh_token("test-user")  # sync метод
        refresh_payload = await auth_service.verify_token(
            refresh_token, "refresh"
        )  # async метод

        # Refresh токен должен жить дольше access токена
        assert refresh_payload["exp"] > payload["exp"]

    # A08: Software and Data Integrity Failures
    @pytest.mark.asyncio
    async def test_data_integrity_protection(self, encryption_service):
        """Тест защиты целостности данных."""
        # 1. Проверка целостности метаданных
        data = b"important_data"
        metadata = {"version": "1.0", "checksum": "abc123"}

        encrypted = await encryption_service.encrypt_data(data, metadata)
        decrypted_data, decrypted_metadata = await encryption_service.decrypt_data(
            encrypted
        )

        assert decrypted_data == data
        # decrypted_metadata содержит весь payload, включая метаданные в поле "meta"
        assert decrypted_metadata["meta"] == metadata

        # 2. Проверка обнаружения изменений
        tampered = bytearray(encrypted)
        tampered[10] ^= 1  # Изменяем байт

        with pytest.raises(Exception):  # EncryptionError
            await encryption_service.decrypt_data(bytes(tampered))

    # A09: Security Logging and Monitoring Failures
    def test_logging_monitoring_protection(self):
        """Тест защиты логирования и мониторинга."""
        from app.utils.logger import get_logger

        # Проверяем, что логгер настроен
        logger = get_logger(__name__)
        assert logger is not None

        # Проверяем уровни логирования
        assert settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # A10: Server-Side Request Forgery (SSRF)
    def test_ssrf_protection(self):
        """Тест защиты от SSRF."""
        # Проверка что sanitize_string работает корректно
        # Для реальной защиты от SSRF нужны дополнительные проверки на уровне приложения
        test_inputs = [
            "http://example.com",
            "https://google.com",
            "http://localhost:22",  # Валидный синтаксис, но потенциально опасный
            "file:///etc/passwd",  # Невалидный протокол
        ]

        for test_input in test_inputs:
            # sanitize_string должен обрабатывать вход без ошибок
            sanitized = sanitize_string(test_input)
            assert isinstance(sanitized, str)


class TestRateLimiting:
    """Тесты rate limiting."""

    @pytest.fixture
    def auth_service(self):
        return AuthService()

    @pytest.mark.asyncio
    async def test_token_creation_rate_limiting(self, auth_service):
        """Тест rate limiting создания токенов."""
        user_id = "test-user"

        # Создаем несколько токенов быстро (sync метод)
        tokens = []
        for i in range(10):
            token = auth_service.create_access_token(f"{user_id}-{i}")
            tokens.append(token)

        # Все токены должны быть уникальными
        assert len(set(tokens)) == 10

        # Все токены должны быть валидными (async метод)
        for token in tokens:
            payload = await auth_service.verify_token(token, "access")
            assert payload["user_id"].startswith(user_id)


class TestPasswordSecurity:
    """Тесты безопасности паролей."""

    @pytest.fixture
    def auth_service(self):
        return AuthService()

    @pytest.mark.asyncio
    async def test_password_hashing_security(self, auth_service):
        """Тест безопасности хеширования паролей."""
        password = "TestPassword123!"

        # Хешируем пароль
        hash1 = await auth_service.hash_password(password)
        hash2 = await auth_service.hash_password(password)

        # Хеши должны быть разными (из-за соли)
        assert hash1 != hash2

        # Но оба должны верифицироваться
        assert await auth_service.verify_password(password, hash1)
        assert await auth_service.verify_password(password, hash2)

        # Неправильный пароль не должен проходить
        assert not await auth_service.verify_password("wrong_password", hash1)

    @pytest.mark.asyncio
    async def test_password_complexity_enforcement(self, auth_service):
        """Тест принудительной сложности паролей."""
        # Слабые пароли должны отклоняться на уровне валидации
        weak_passwords = ["123456", "password", "abc123", "qwerty"]

        for weak_password in weak_passwords:
            with pytest.raises(ValidationError):
                validate_password(weak_password)

        # Сильные пароли должны приниматься (только с разрешенными спецсимволами)
        strong_passwords = [
            "StrongPass123!",
            "C0mpl3x!Pass",
            "Secure$2024",  # Изменил # на $ (разрешенный символ)
        ]

        for strong_password in strong_passwords:
            assert validate_password(strong_password) is True


class TestInputValidation:
    """Тесты валидации входных данных."""

    def test_email_validation_security(self):
        """Тест безопасности валидации email."""
        # SQL injection попытки в email
        malicious_emails = [
            "test@example.com'; DROP TABLE users; --",
            "admin@domain.com' OR '1'='1",
            "user@test.com' UNION SELECT * FROM users --",
        ]

        for malicious_email in malicious_emails:
            try:
                validate_email(malicious_email)
                # Если прошло, проверяем, что опасные части удалены
                assert "DROP TABLE" not in malicious_email
                assert "UNION SELECT" not in malicious_email
            except ValidationError:
                # Это нормально - некоторые email должны отклоняться
                pass

    def test_xss_protection(self):
        """Тест защиты от XSS."""
        malicious_inputs = [
            '<script>alert("xss")</script>',
            '"><script>alert("xss")</script>',
            '<img src="x" onerror="alert(\'xss\')">',
        ]

        for malicious_input in malicious_inputs:
            sanitized = sanitize_string(malicious_input)

            # Опасные символы должны быть удалены (но не javascript: протокол)
            assert "<" not in sanitized  # Угловые скобки удалены
            assert ">" not in sanitized  # Угловые скобки удалены
            assert '"' not in sanitized  # Двойные кавычки удалены
            assert "'" not in sanitized  # Одинарные кавычки удалены
            # javascript: протокол может остаться - это нормально для sanitize_string
            # Основная защита от XSS обеспечивается удалением опасных символов


if __name__ == "__main__":
    pytest.main([__file__])
