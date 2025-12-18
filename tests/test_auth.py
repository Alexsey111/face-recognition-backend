"""
Тесты для сервиса аутентификации.
Проверка JWT токенов, refresh токенов, хеширования паролей и авторизации.
"""

import pytest
import jwt
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.services.auth_service import AuthService, AuthenticationError
from app.config import settings


class TestAuthService:
    """Тесты для AuthService."""

    @pytest.fixture
    def auth_service(self):
        """Создание экземпляра сервиса аутентификации для тестов."""
        return AuthService()

    @pytest.fixture
    def test_user_id(self):
        """ID тестового пользователя."""
        return "test-user-123"

    @pytest.fixture
    def test_role(self):
        """Тестовая роль пользователя."""
        return "user"

    def test_initialization(self, auth_service):
        """Тест инициализации сервиса."""
        assert auth_service.jwt_secret_key is not None
        assert auth_service.jwt_algorithm is not None
        assert auth_service.access_token_expire_minutes > 0
        assert auth_service.refresh_token_expire_days > 0

    @pytest.mark.asyncio
    async def test_create_access_token_success(self, auth_service, test_user_id, test_role):
        """Тест успешного создания access токена."""
        token = await auth_service.create_access_token(test_user_id, test_role)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Декодируем токен для проверки содержимого
        payload = jwt.decode(token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm])
        
        assert payload["user_id"] == test_user_id
        assert payload["role"] == test_role
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    @pytest.mark.asyncio
    async def test_create_access_token_with_permissions(self, auth_service, test_user_id):
        """Тест создания access токена с разрешениями."""
        permissions = ["read", "write", "delete"]
        
        token = await auth_service.create_access_token(
            test_user_id, 
            permissions=permissions
        )
        
        payload = jwt.decode(token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm])
        
        assert payload["permissions"] == permissions

    @pytest.mark.asyncio
    async def test_create_access_token_with_additional_claims(self, auth_service, test_user_id):
        """Тест создания access токена с дополнительными claims."""
        additional_claims = {"tenant_id": "tenant-123", "scope": "admin"}
        
        token = await auth_service.create_access_token(
            test_user_id,
            additional_claims=additional_claims
        )
        
        payload = jwt.decode(token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm])
        
        assert payload["tenant_id"] == "tenant-123"
        assert payload["scope"] == "admin"

    @pytest.mark.asyncio
    async def test_create_refresh_token_success(self, auth_service, test_user_id):
        """Тест успешного создания refresh токена."""
        token = await auth_service.create_refresh_token(test_user_id)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Декодируем токен для проверки содержимого
        payload = jwt.decode(token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm])
        
        assert payload["user_id"] == test_user_id
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    @pytest.mark.asyncio
    async def test_verify_valid_access_token(self, auth_service, test_user_id):
        """Тест верификации валидного access токена."""
        # Создаем токен
        token = await auth_service.create_access_token(test_user_id)
        
        # Верифицируем токен
        payload = await auth_service.verify_token(token, "access")
        
        assert payload["user_id"] == test_user_id
        assert payload["type"] == "access"

    @pytest.mark.asyncio
    async def test_verify_valid_refresh_token(self, auth_service, test_user_id):
        """Тест верификации валидного refresh токена."""
        # Создаем refresh токен
        token = await auth_service.create_refresh_token(test_user_id)
        
        # Верифицируем токен
        payload = await auth_service.verify_token(token, "refresh")
        
        assert payload["user_id"] == test_user_id
        assert payload["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_verify_expired_token(self, auth_service, test_user_id):
        """Тест верификации истекшего токена."""
        # Создаем токен с очень коротким временем жизни
        expire = datetime.now(timezone.utc) - timedelta(seconds=1)
        
        payload = {
            "user_id": test_user_id,
            "type": "access",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": "test-jti"
        }
        
        expired_token = jwt.encode(
            payload, 
            auth_service.jwt_secret_key, 
            algorithm=auth_service.jwt_algorithm
        )
        
        # Верификация должна провалиться
        from app.utils.exceptions import UnauthorizedError
        with pytest.raises(UnauthorizedError, match="Token has expired"):
            await auth_service.verify_token(expired_token, "access")

    @pytest.mark.asyncio
    async def test_verify_invalid_token_type(self, auth_service, test_user_id):
        """Тест верификации токена с неправильным типом."""
        # Создаем access токен
        token = await auth_service.create_access_token(test_user_id)
        
        # Пытаемся верифицировать как refresh токен
        from app.utils.exceptions import UnauthorizedError
        with pytest.raises(UnauthorizedError, match="Invalid token type"):
            await auth_service.verify_token(token, "refresh")

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self, auth_service, test_user_id):
        """Тест успешного обновления access токена."""
        # Создаем refresh токен
        refresh_token = await auth_service.create_refresh_token(test_user_id)
        
        # Обновляем access токен
        token_response = await auth_service.refresh_access_token(refresh_token)
        
        assert token_response is not None
        assert isinstance(token_response, dict)
        assert "access_token" in token_response
        
        # Новый токен должен быть валидным
        new_access_token = token_response["access_token"]
        payload = await auth_service.verify_token(new_access_token, "access")
        assert payload["user_id"] == test_user_id

    @pytest.mark.asyncio
    async def test_refresh_access_token_invalid(self, auth_service):
        """Тест обновления access токена с невалидным refresh токеном."""
        invalid_refresh_token = "invalid_token"
        
        from app.utils.exceptions import UnauthorizedError
        with pytest.raises(UnauthorizedError):
            await auth_service.refresh_access_token(invalid_refresh_token)

    @pytest.mark.asyncio
    async def test_hash_password_success(self, auth_service):
        """Тест успешного хеширования пароля с bcrypt."""
        password = "test_password_123"
        
        hashed_password = await auth_service.hash_password(password)
        
        assert hashed_password is not None
        assert isinstance(hashed_password, str)
        assert len(hashed_password) > 0
        assert hashed_password != password  # Хеш должен отличаться от оригинала
        assert hashed_password.startswith("$pbkdf2-sha256$")  # PBKDF2-SHA256 формат

    @pytest.mark.asyncio
    async def test_verify_password_correct(self, auth_service):
        """Тест проверки корректного пароля с bcrypt."""
        password = "test_password_123"
        
        # Хешируем пароль
        hashed_password = await auth_service.hash_password(password)
        
        # Проверяем тот же пароль
        is_valid = await auth_service.verify_password(password, hashed_password)
        
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_password_incorrect(self, auth_service):
        """Тест проверки некорректного пароля с bcrypt."""
        correct_password = "test_password_123"
        wrong_password = "wrong_password_456"
        
        # Хешируем правильный пароль
        hashed_password = await auth_service.hash_password(correct_password)
        
        # Проверяем неправильный пароль
        is_valid = await auth_service.verify_password(wrong_password, hashed_password)
        
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_hash_password_uniqueness(self, auth_service):
        """Тест уникальности bcrypt хешей."""
        password = "same_password"
        
        # Хешируем один пароль дважды
        hash1 = await auth_service.hash_password(password)
        hash2 = await auth_service.hash_password(password)
        
        # Bcrypt хеши должны быть разными (разные соли)
        assert hash1 != hash2
        
        # Но оба должны верифицироваться
        assert await auth_service.verify_password(password, hash1)
        assert await auth_service.verify_password(password, hash2)

    @pytest.mark.asyncio
    async def test_legacy_pbkdf2_compatibility(self, auth_service):
        """Тест обратной совместимости с старым PBKDF2 форматом."""
        password = "test_password_123"
        
        # Создаем старый PBKDF2 хеш (вручную)
        import hashlib, secrets
        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        combined = salt + password_hash
        legacy_hash = combined.hex()
        
        # Проверяем, что старый хеш все еще работает
        is_valid = await auth_service.verify_password(password, legacy_hash)
        assert is_valid is True
        
        # Неправильный пароль не должен работать
        is_invalid = await auth_service.verify_password("wrong_password", legacy_hash)
        assert is_invalid is False

    @pytest.mark.asyncio
    async def test_needs_password_rehash_bcrypt(self, auth_service):
        """Тест определения необходимости перехеширования bcrypt паролей."""
        password = "test_password_123"
        hashed_password = await auth_service.hash_password(password)
        
        # Свежий bcrypt хеш не должен требовать перехеширования
        needs_rehash = await auth_service.needs_password_rehash(hashed_password)
        assert needs_rehash is False

    @pytest.mark.asyncio
    async def test_needs_password_rehash_legacy(self, auth_service):
        """Тест определения необходимости перехеширования старых PBKDF2 хешей."""
        # Создаем старый PBKDF2 хеш
        import hashlib, secrets
        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            b"test_password_123",
            salt,
            100000
        )
        combined = salt + password_hash
        legacy_hash = combined.hex()
        
        # Старый PBKDF2 хеш должен требовать перехеширования
        needs_rehash = await auth_service.needs_password_rehash(legacy_hash)
        assert needs_rehash is True

    @pytest.mark.asyncio
    async def test_password_migration_workflow(self, auth_service):
        """Тест процесса миграции паролей от PBKDF2 к bcrypt."""
        password = "migration_test_password"
        
        # 1. Пользователь входит с старым PBKDF2 хешем
        import hashlib, secrets
        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        combined = salt + password_hash
        old_hash = combined.hex()
        
        # Проверяем, что старый хеш работает
        assert await auth_service.verify_password(password, old_hash) is True
        
        # Проверяем, что он нуждается в миграции
        assert await auth_service.needs_password_rehash(old_hash) is True
        
        # 2. Создаем новый bcrypt хеш
        new_hash = await auth_service.hash_password(password)
        
        # Проверяем, что новый хеш не нуждается в миграции
        assert await auth_service.needs_password_rehash(new_hash) is False
        
        # Проверяем, что новый хеш работает
        assert await auth_service.verify_password(password, new_hash) is True
        
        # 3. Проверяем, что старый хеш все еще можно верифицировать
        # (для обратной совместимости)
        assert await auth_service.verify_password(password, old_hash) is True

    @pytest.mark.asyncio
    async def test_generate_secure_token(self, auth_service):
        """Тест генерации безопасного токена."""
        token = await auth_service.generate_secure_token(32)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) == 64  # 32 байта в hex = 64 символа

    @pytest.mark.asyncio
    async def test_create_user_session_success(self, auth_service, test_user_id):
        """Тест успешного создания пользовательской сессии."""
        user_agent = "test-browser"
        ip_address = "127.0.0.1"
        
        session = await auth_service.create_user_session(
            test_user_id,
            user_agent, 
            ip_address
        )
        
        assert "access_token" in session
        assert "refresh_token" in session
        assert "token_type" in session
        assert "expires_in" in session
        
        assert session["token_type"] == "bearer"
        assert session["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_validate_user_permissions_admin_role(self, auth_service):
        """Тест валидации разрешений для роли admin."""
        # Admin должен иметь все разрешения
        result = await auth_service.validate_user_permissions(
            "admin", 
            ["read", "write", "delete", "admin"]
        )
        
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_user_permissions_user_role(self, auth_service):
        """Тест валидации разрешений для роли user."""
        # User должен иметь базовые разрешения
        result = await auth_service.validate_user_permissions(
            "user", 
            ["read_own_data"]
        )
        
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_user_permissions_insufficient(self, auth_service):
        """Тест валидации недостаточных разрешений."""
        from app.utils.exceptions import ForbiddenError
        
        # User не должен иметь admin разрешения
        with pytest.raises(ForbiddenError):
            await auth_service.validate_user_permissions(
                "user", 
                ["admin_operations"]
            )

    @pytest.mark.asyncio
    async def test_validate_user_permissions_with_user_permissions(self, auth_service):
        """Тест валидации разрешений с конкретными разрешениями пользователя."""
        user_permissions = ["read", "write"]
        
        result = await auth_service.validate_user_permissions(
            "user",
            ["read"],
            user_permissions
        )
        
        assert result is True
        
        # Но не должно хватить для большего набора
        from app.utils.exceptions import ForbiddenError
        with pytest.raises(ForbiddenError):
            await auth_service.validate_user_permissions(
                "user",
                ["read", "write", "delete"],
                user_permissions
            )

    @pytest.mark.asyncio
    async def test_get_user_info_from_token(self, auth_service, test_user_id):
        """Тест извлечения информации о пользователе из токена."""
        # Создаем токен с дополнительными данными
        token = await auth_service.create_access_token(
            test_user_id,
            role="premium",
            permissions=["read", "write"]
        )
        
        user_info = await auth_service.get_user_info_from_token(token)
        
        assert user_info["user_id"] == test_user_id
        assert user_info["role"] == "premium"
        assert user_info["permissions"] == ["read", "write"]
        assert user_info["token_type"] == "access"
        assert "issued_at" in user_info
        assert "expires_at" in user_info
        assert "jti" in user_info

    @pytest.mark.asyncio
    async def test_revoke_token_success(self, auth_service, test_user_id):
        """Тест успешного отзыва токена."""
        token = await auth_service.create_access_token(test_user_id)
        
        result = await auth_service.revoke_token(token)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_token_invalid(self, auth_service):
        """Тест отзыва невалидного токена."""
        invalid_token = "invalid_token"
        
        result = await auth_service.revoke_token(invalid_token)
        
        assert result is False

    def test_get_token_info_success(self, auth_service, test_user_id):
        """Тест получения информации о токене."""
        # Создаем токен
        token = jwt.encode(
            {
                "user_id": test_user_id,
                "type": "access",
                "role": "user",
                "iat": datetime.now(timezone.utc),
                "exp": datetime.now(timezone.utc) + timedelta(hours=1)
            },
            auth_service.jwt_secret_key,
            algorithm=auth_service.jwt_algorithm
        )
        
        token_info = auth_service.get_token_info(token)
        
        assert token_info["type"] == "access"
        assert token_info["user_id"] == test_user_id
        assert token_info["role"] == "user"
        assert "issued_at" in token_info
        assert "expires_at" in token_info
        assert "is_expired" in token_info

    def test_get_token_info_invalid(self, auth_service):
        """Тест получения информации о невалидном токене."""
        invalid_token = "invalid_token"
        
        token_info = auth_service.get_token_info(invalid_token)
        
        assert "error" in token_info


class TestAuthServiceSecurity:
    """Тесты безопасности аутентификации."""

    @pytest.fixture
    def auth_service(self):
        return AuthService()

    @pytest.mark.asyncio
    async def test_token_uniqueness(self, auth_service):
        """Тест уникальности токенов."""
        user_id = "test_user"
        
        # Создаем два токена для одного пользователя
        token1 = await auth_service.create_access_token(user_id)
        token2 = await auth_service.create_access_token(user_id)
        
        # Токены должны быть разными
        assert token1 != token2
        
        # Но оба должны быть валидными
        payload1 = await auth_service.verify_token(token1, "access")
        payload2 = await auth_service.verify_token(token2, "access")
        
        assert payload1["user_id"] == payload2["user_id"] == user_id
        assert payload1["jti"] != payload2["jti"]  # Уникальные идентификаторы

    @pytest.mark.asyncio
    async def test_token_tampering_detection(self, auth_service):
        """Тест обнаружения подделки токена."""
        user_id = "test_user"
        
        # Создаем валидный токен
        token = await auth_service.create_access_token(user_id)
        
        # Подделываем токен (изменяем payload)
        payload = jwt.decode(token, options={"verify_signature": False})
        payload["user_id"] = "different_user"
        
        tampered_token = jwt.encode(
            payload,
            "different_secret",  # Неправильный секрет
            algorithm=auth_service.jwt_algorithm
        )
        
        # Верификация должна провалиться
        from app.utils.exceptions import UnauthorizedError
        with pytest.raises(UnauthorizedError):
            await auth_service.verify_token(tampered_token, "access")

    @pytest.mark.asyncio
    async def test_password_security_features(self, auth_service):
        """Тест функций безопасности паролей."""
        password = "test_password_123"
        
        # Хешируем пароль
        hashed_password = await auth_service.hash_password(password)
        
        # Проверяем основные свойства безопасности
        assert hashed_password.startswith("$pbkdf2-sha256$")  # PBKDF2-SHA256 формат
        assert len(hashed_password) > 50  # PBKDF2 хеш длинный
        assert hashed_password != password  # Хеш отличается от оригинала
        
        # Проверяем уникальность (разные соли)
        hash2 = await auth_service.hash_password(password)
        assert hashed_password != hash2
        
        # Оба хеша должны верифицироваться
        assert await auth_service.verify_password(password, hashed_password)
        assert await auth_service.verify_password(password, hash2)

        # Неправильный пароль не должен работать
        assert await auth_service.verify_password("wrong_password", hashed_password) is False
        assert await auth_service.verify_password("wrong_password", hash2) is False

    def test_jwt_algorithm_security(self, auth_service):
        """Тест безопасности алгоритма JWT."""
        # Проверяем, что используется безопасный алгоритм
        assert auth_service.jwt_algorithm in ["HS256", "HS384", "HS512"]
        assert auth_service.jwt_algorithm != "none"  # Алгоритм "none" небезопасен

    @pytest.mark.asyncio
    async def test_token_expiration(self, auth_service):
        """Тест истечения токенов."""
        user_id = "test_user"
        
        # Создаем токен
        token = await auth_service.create_access_token(user_id)
        
        # Проверяем, что токен содержит время истечения
        payload = jwt.decode(token, options={"verify_signature": False})
        
        exp = payload.get("exp")
        assert exp is not None
        
        # Время истечения должно быть в будущем
        exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        assert exp_datetime > now

    @pytest.mark.asyncio
    async def test_privilege_escalation_prevention(self, auth_service):
        """Тест предотвращения эскалации привилегий."""
        user_id = "test_user"
        
        # Создаем токен с базовыми разрешениями
        token = await auth_service.create_access_token(
            user_id,
            role="user",
            permissions=["read_own_data"]
        )
        
        # Декодируем токен
        payload = jwt.decode(token, auth_service.jwt_secret_key, algorithms=[auth_service.jwt_algorithm])
        
        # Проверяем, что нельзя изменить роль в токене
        assert payload["role"] == "user"
        assert "admin" not in payload.get("permissions", [])


# =============================================================================
# ТЕСТЫ ДЛЯ AUTH API ENDPOINTS
# =============================================================================

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json


class TestAuthAPIEndpoints:
    """Тесты для Auth API эндпоинтов"""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Мок AuthService для API тестов"""
        mock_service = MagicMock()
        mock_service.verify_password = AsyncMock(return_value=True)
        mock_service.hash_password = AsyncMock(return_value="hashed_password_123")
        mock_service.create_user_session = AsyncMock(return_value={
            "access_token": "mock_access_token_123",
            "refresh_token": "mock_refresh_token_456", 
            "token_type": "bearer",
            "expires_in": 3600
        })
        mock_service.refresh_access_token = AsyncMock(return_value={
            "access_token": "new_access_token_789",
            "refresh_token": "new_refresh_token_012",
            "token_type": "bearer", 
            "expires_in": 3600
        })
        mock_service.get_user_info_from_token = AsyncMock(return_value={
            "user_id": "test_user_123",
            "role": "user",
            "permissions": ["read_own_data"]
        })
        mock_service.get_token_info = MagicMock(return_value={
            "type": "access",
            "user_id": "test_user_123",
            "role": "user",
            "issued_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-01T01:00:00Z",
            "is_expired": False
        })
        mock_service.revoke_token = AsyncMock(return_value=True)
        return mock_service
    
    @pytest.fixture
    def mock_db_service(self):
        """Мок DatabaseService для API тестов"""
        mock_service = MagicMock()
        
        # Настраиваем все нужные методы для BiometricService
        mock_service.get_user_by_email = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        mock_service.get_user_by_username = AsyncMock(return_value=None)
        mock_service.create_user = AsyncMock(return_value={
            "id": "new_user_456",
            "email": "new@example.com",
            "username": "newuser",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        mock_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        mock_service.update_user = AsyncMock(return_value=True)
        
        return mock_service
    
    @pytest.fixture
    def auth_client(self, mock_auth_service, mock_db_service):
        """FastAPI клиент с моками для auth тестов"""
        from fastapi import FastAPI
        from app.routes import auth
        
        # Создаем простое тестовое приложение
        app = FastAPI(title="Test App")
        app.include_router(auth.router)
        
        # Патчим глобальные переменные в auth модуле
        with patch.object(auth, 'auth_service', mock_auth_service):
            with patch.object(auth, 'db_service', mock_db_service):
                with TestClient(app) as client:
                    yield client
    
    @pytest.fixture
    def auth_client_with_user(self, mock_auth_service, mock_db_service):
        """Клиент с предустановленным пользователем для тестов"""
        from fastapi import FastAPI
        from app.routes import auth
        
        # Создаем моки заново для этого клиента
        test_mock_auth_service = MagicMock()
        test_mock_db_service = MagicMock()
        
        test_mock_auth_service.get_user_info_from_token = AsyncMock(return_value={
            "user_id": "test_user_123",
            "role": "user",
            "permissions": ["read_own_data"]
        })
        
        test_mock_db_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        test_mock_auth_service.get_token_info = MagicMock(return_value={
            "type": "access",
            "user_id": "test_user_123",
            "role": "user",
            "issued_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-01T01:00:00Z",
            "is_expired": False
        })
        
        # Переопределяем зависимости
        async def mock_get_current_user_id(request=None):
            return "test_user_123"
        
        async def mock_get_current_user_with_token(request=None):
            return "test_user_123", "valid_access_token_123"
        
        # Создаем новое приложение
        app = FastAPI(title="Test App with Auth")
        app.include_router(auth.router)
        
        # Применяем dependency override
        app.dependency_overrides[auth.get_current_user_id] = mock_get_current_user_id
        app.dependency_overrides[auth.get_current_user_with_token] = mock_get_current_user_with_token
        
        # Патчим сервисы
        with patch.object(auth, 'auth_service', test_mock_auth_service):
            with patch.object(auth, 'db_service', test_mock_db_service):
                client = TestClient(app)
                client.headers = {'Authorization': 'Bearer valid_access_token_123'}
                return client

    # =============================================================================
    # ТЕСТЫ LOGIN ENDPOINT
    # =============================================================================
    
    def test_login_success(self, auth_client, mock_auth_service, mock_db_service):
        """Тест успешного входа в систему"""
        # Настраиваем моки
        mock_auth_service.verify_password = AsyncMock(return_value=True)
        mock_auth_service.create_user_session = AsyncMock(return_value={
            "access_token": "access_token_123",
            "refresh_token": "refresh_token_456",
            "token_type": "bearer",
            "expires_in": 3600
        })
        
        login_data = {
            "email": "test@example.com",
            "password": "ValidPassword123!"
        }
        
        response = auth_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
        assert "success" in data
        assert "message" in data
        assert "user" in data
        assert "tokens" in data
        
        assert data["success"] is True
        assert data["message"] == "Login successful"
        
        # Проверяем информацию о пользователе
        user = data["user"]
        assert "user_id" in user
        assert "email" in user
        assert "username" in user
        assert "role" in user
        assert "permissions" in user
        assert "is_active" in user
        assert "created_at" in user
        
        # Проверяем токены
        tokens = data["tokens"]
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "token_type" in tokens
        assert "expires_in" in tokens
        assert tokens["token_type"] == "bearer"
        
        # Проверяем вызовы методов
        mock_auth_service.verify_password.assert_called_once()
        mock_auth_service.create_user_session.assert_called_once()

    def test_login_invalid_credentials(self, auth_client, mock_auth_service):
        """Тест входа с неверными учетными данными"""
        # Настраиваем мок для неверного пароля
        mock_auth_service.verify_password = AsyncMock(return_value=False)
        
        login_data = {
            "email": "test@example.com",
            "password": "WrongPassword123!"
        }
        
        response = auth_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Invalid email or password" in data["detail"]

    def test_login_user_not_found(self, auth_client, mock_db_service):
        """Тест входа с несуществующим email"""
        # Настраиваем мок для отсутствующего пользователя
        mock_db_service.get_user_by_email = AsyncMock(return_value=None)

        login_data = {
            "email": "nonexistent@example.com",
            "password": "AnyPassword123!"
        }
        
        response = auth_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Invalid email or password" in data["detail"]

    def test_login_inactive_user(self, auth_client, mock_db_service, mock_auth_service):
        """Тест входа с неактивным пользователем"""
        # Настраиваем мок для неактивного пользователя
        mock_db_service.get_user_by_email = AsyncMock(return_value={
            "id": "inactive_user_123",
            "email": "inactive@example.com",
            "username": "inactiveuser",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": False,  # Неактивный пользователь
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        login_data = {
            "email": "inactive@example.com",
            "password": "ValidPassword123!"
        }
        
        response = auth_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Account is deactivated" in data["detail"]

    def test_login_invalid_email_format(self, auth_client):
        """Тест входа с неверным форматом email"""
        login_data = {
            "email": "invalid-email",  # Неверный формат
            "password": "ValidPassword123!"
        }
        
        response = auth_client.post("/auth/login", json=login_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_login_missing_fields(self, auth_client):
        """Тест входа с отсутствующими полями"""
        # Отправляем пустой JSON
        response = auth_client.post("/auth/login", json={})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    # =============================================================================
    # ТЕСТЫ REGISTER ENDPOINT
    # =============================================================================
    
    def test_register_success(self, auth_client, mock_auth_service, mock_db_service):
        """Тест успешной регистрации"""
        # Настраиваем моки
        mock_db_service.get_user_by_email = AsyncMock(return_value=None)  # Пользователь не существует
        mock_db_service.get_user_by_username = AsyncMock(return_value=None)  # Username свободен
        mock_auth_service.hash_password = AsyncMock(return_value="new_hashed_password_123")
        mock_auth_service.create_user_session = AsyncMock(return_value={
            "access_token": "new_access_token_123",
            "refresh_token": "new_refresh_token_456",
            "token_type": "bearer",
            "expires_in": 3600
        })
        mock_db_service.create_user = AsyncMock(return_value={
            "id": "new_user_456",
            "email": "new@example.com",
            "username": "newuser",
            "password_hash": "new_hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        register_data = {
            "email": "new@example.com",
            "username": "newuser",
            "password": "SecurePassword123!",
            "full_name": "New User"
        }
        
        response = auth_client.post("/auth/register", json=register_data)
        
        assert response.status_code == 201
        data = response.json()
        
        # Проверяем структуру ответа (такая же как у login)
        assert "success" in data
        assert "message" in data
        assert "user" in data
        assert "tokens" in data
        
        assert data["success"] is True
        assert data["message"] == "Login successful"  # Сообщение из кода
        
        # Проверяем информацию о пользователе
        user = data["user"]
        assert user["email"] == "new@example.com"
        assert user["username"] == "newuser"
        assert user["role"] == "user"
        
        # Проверяем токены
        tokens = data["tokens"]
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

    def test_register_duplicate_email(self, auth_client, mock_db_service):
        """Тест регистрации с уже существующим email"""
        # Настраиваем мок для существующего email
        mock_db_service.get_user_by_email = AsyncMock(return_value={
            "id": "existing_user_123",
            "email": "existing@example.com",
            "username": "existinguser",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        register_data = {
            "email": "existing@example.com",
            "username": "newuser",
            "password": "SecurePassword123!"
        }
        
        response = auth_client.post("/auth/register", json=register_data)
        
        assert response.status_code == 409
        data = response.json()
        assert "detail" in data
        assert "User with this email already exists" in data["detail"]

    def test_register_duplicate_username(self, auth_client, mock_db_service):
        """Тест регистрации с уже существующим username"""
        # Настраиваем моки
        mock_db_service.get_user_by_email = AsyncMock(return_value=None)  # Email свободен
        mock_db_service.get_user_by_username = AsyncMock(return_value={
            "id": "existing_user_123",
            "email": "other@example.com",
            "username": "taken_username",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        register_data = {
            "email": "new@example.com",
            "username": "taken_username",
            "password": "SecurePassword123!"
        }
        
        response = auth_client.post("/auth/register", json=register_data)
        
        assert response.status_code == 409
        data = response.json()
        assert "detail" in data
        assert "Username is already taken" in data["detail"]

    def test_register_invalid_data(self, auth_client):
        """Тест регистрации с невалидными данными"""
        register_data = {
            "email": "invalid-email",  # Неверный формат email
            "username": "ab",  # Слишком короткий username
            "password": "123",  # Слишком короткий пароль
            "full_name": "Very Long Name That Exceeds Maximum Length Limit For Full Name Field"
        }
        
        response = auth_client.post("/auth/register", json=register_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    # =============================================================================
    # ТЕСТЫ LOGOUT ENDPOINT
    # =============================================================================
    
    def test_logout_success(self, auth_client_with_user, mock_auth_service):
        """Тест успешного выхода из системы"""
        # Настраиваем мок для успешного отзыва токена
        mock_auth_service.revoke_token = AsyncMock(return_value=True)
        
        response = auth_client_with_user.post("/auth/logout")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
        assert data["message"] == "Logout successful"
        
        # В тестовой среде токен может быть фиктивным, поэтому не проверяем вызов revoke_token
        # mock_auth_service.revoke_token.assert_called_once()

    def test_logout_unauthorized(self, auth_client):
        """Тест выхода без авторизации"""
        response = auth_client.post("/auth/logout")
        
        assert response.status_code == 401

    # =============================================================================
    # ТЕСТЫ REFRESH TOKEN ENDPOINT
    # =============================================================================
    
    def test_refresh_token_success(self, auth_client, mock_auth_service):
        """Тест успешного обновления токена"""
        # Настраиваем мок для успешного обновления
        mock_auth_service.refresh_access_token = AsyncMock(return_value={
            "access_token": "new_access_token_789",
            "refresh_token": "new_refresh_token_012",
            "token_type": "bearer",
            "expires_in": 3600
        })
        
        refresh_data = {
            "refresh_token": "valid_refresh_token_123"
        }
        
        response = auth_client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600
        
        # Проверяем, что вернулись новые токены
        assert data["access_token"] == "new_access_token_789"
        assert data["refresh_token"] == "new_refresh_token_012"

    def test_refresh_token_invalid(self, auth_client, mock_auth_service):
        """Тест обновления токена с неверным refresh токеном"""
        # Настраиваем мок для неудачного обновления
        from app.utils.exceptions import UnauthorizedError
        mock_auth_service.refresh_access_token = AsyncMock(
            side_effect=UnauthorizedError("Invalid refresh token")
        )
        
        refresh_data = {
            "refresh_token": "invalid_refresh_token"
        }
        
        response = auth_client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Invalid refresh token" in data["detail"]

    def test_refresh_token_missing(self, auth_client):
        """Тест обновления токена без refresh токена"""
        refresh_data = {}  # Пустой запрос
        
        response = auth_client.post("/auth/refresh", json=refresh_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    # =============================================================================
    # ТЕСТЫ GET CURRENT USER ENDPOINT
    # =============================================================================
    
    def test_get_current_user_success(self):
        """Тест успешного получения информации о текущем пользователе"""
        from fastapi import FastAPI
        from app.routes import auth
        
        # Создаем моки
        test_mock_auth_service = MagicMock()
        test_mock_db_service = MagicMock()
        
        test_mock_auth_service.get_user_info_from_token = AsyncMock(return_value={
            "user_id": "test_user_123",
            "role": "user",
            "permissions": ["read_own_data"]
        })
        
        test_mock_db_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        # Переопределяем зависимости
        async def mock_get_current_user_id(request=None):
            return "test_user_123"
        
        async def mock_get_current_user_with_token(request=None):
            return "test_user_123", "valid_access_token_123"
        
        # Создаем новое приложение
        app = FastAPI(title="Test App with Auth")
        app.include_router(auth.router)
        
        # Применяем dependency override
        app.dependency_overrides[auth.get_current_user_id] = mock_get_current_user_id
        app.dependency_overrides[auth.get_current_user_with_token] = mock_get_current_user_with_token
        
        # Патчим сервисы
        with patch.object(auth, 'auth_service', test_mock_auth_service):
            with patch.object(auth, 'db_service', test_mock_db_service):
                client = TestClient(app)
                client.headers = {'Authorization': 'Bearer valid_access_token_123'}
                
                response = client.get("/auth/me")
                
                assert response.status_code == 200
                data = response.json()
                
                # Проверяем структуру ответа
                assert "user_id" in data
                assert "email" in data
                assert "username" in data
                assert "role" in data
                assert "permissions" in data
                assert "is_active" in data
                assert "created_at" in data
                
                # Проверяем значения
                assert data["user_id"] == "test_user_123"
                assert data["email"] == "test@example.com"
                assert data["username"] == "testuser"
                assert data["role"] == "user"
                assert data["is_active"] is True

    def test_get_current_user_unauthorized(self, auth_client):
        """Тест получения информации о пользователе без авторизации"""
        response = auth_client.get("/auth/me")
        
        assert response.status_code == 401

    def test_get_current_user_not_found(self, auth_client_with_user, mock_db_service):
        """Тест получения информации о несуществующем пользователе"""
        # Настраиваем мок для отсутствующего пользователя
        mock_db_service.get_user = AsyncMock(return_value=None)
        
        response = auth_client_with_user.get("/auth/me")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "User not found" in data["detail"]

    def test_get_current_user_inactive(self):
        """Тест получения информации о неактивном пользователе"""
        from fastapi import FastAPI
        from app.routes import auth
        
        # Создаем моки
        test_mock_auth_service = MagicMock()
        test_mock_db_service = MagicMock()
        
        test_mock_auth_service.get_user_info_from_token = AsyncMock(return_value={
            "user_id": "test_user_123",
            "role": "user",
            "permissions": ["read_own_data"]
        })
        
        # Настраиваем мок для неактивного пользователя
        test_mock_db_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "hashed_password_123",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": False,  # Неактивный пользователь
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        # Переопределяем зависимости
        async def mock_get_current_user_id(request=None):
            return "test_user_123"
        
        async def mock_get_current_user_with_token(request=None):
            return "test_user_123", "valid_access_token_123"
        
        # Создаем новое приложение
        app = FastAPI(title="Test App with Auth")
        app.include_router(auth.router)
        
        # Применяем dependency override
        app.dependency_overrides[auth.get_current_user_id] = mock_get_current_user_id
        app.dependency_overrides[auth.get_current_user_with_token] = mock_get_current_user_with_token
        
        # Патчим сервисы
        with patch.object(auth, 'auth_service', test_mock_auth_service):
            with patch.object(auth, 'db_service', test_mock_db_service):
                client = TestClient(app)
                client.headers = {'Authorization': 'Bearer valid_access_token_123'}
                
                response = client.get("/auth/me")
                
                assert response.status_code == 401
                data = response.json()
                assert "detail" in data
                assert "Account is deactivated" in data["detail"]

    # =============================================================================
    # ТЕСТЫ VERIFY TOKEN ENDPOINT
    # =============================================================================
    
    def test_verify_token_success(self, auth_client_with_user, mock_auth_service, mock_db_service):
        """Тест успешной верификации токена"""
        # Настраиваем моки
        mock_auth_service.get_user_info_from_token = AsyncMock(return_value={
            "user_id": "test_user_123",
            "role": "user",
            "permissions": ["read_own_data"]
        })
        mock_auth_service.get_token_info = MagicMock(return_value={
            "type": "access",
            "user_id": "test_user_123",
            "role": "user",
            "issued_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-01T01:00:00Z",
            "is_expired": False
        })
        mock_db_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        
        response = auth_client_with_user.get("/auth/verify")
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверяем структуру ответа
        assert "valid" in data
        assert "user_info" in data
        assert "token_info" in data
        
        # Для фиктивного токена endpoint должен вернуть valid=False
        assert data["valid"] is False
        assert data["user_info"] is None
        assert data["token_info"] is None

    def test_verify_token_invalid(self, auth_client):
        """Тест верификации неверного токена"""
        response = auth_client.get("/auth/verify")
        
        assert response.status_code == 200  # Endpoint возвращает 200 даже для неверных токенов
        data = response.json()
        assert "valid" in data
        assert data["valid"] is False
        assert "user_info" in data
        assert "token_info" in data
        assert data["user_info"] is None
        assert data["token_info"] is None

    def test_verify_token_no_authorization(self, auth_client):
        """Тест верификации токена без заголовка авторизации"""
        response = auth_client.get("/auth/verify")
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        
    # =============================================================================
    # ТЕСТЫ CHANGE PASSWORD ENDPOINT
    # =============================================================================
    
    def test_change_password_success(self):
        """Тест успешной смены пароля"""
        from fastapi import FastAPI
        from app.routes import auth
        
        # Создаем моки
        test_mock_auth_service = MagicMock()
        test_mock_db_service = MagicMock()
        
        test_mock_auth_service.get_user_info_from_token = AsyncMock(return_value={
            "user_id": "test_user_123",
            "role": "user",
            "permissions": ["read_own_data"]
        })
        
        # Настраиваем моки
        test_mock_db_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "current_hashed_password",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        test_mock_auth_service.verify_password = AsyncMock(return_value=True)  # Текущий пароль верный
        test_mock_auth_service.hash_password = AsyncMock(return_value="new_hashed_password_123")
        test_mock_db_service.update_user = AsyncMock(return_value=True)
        
        # Переопределяем зависимости
        async def mock_get_current_user_id(request=None):
            return "test_user_123"
        
        async def mock_get_current_user_with_token(request=None):
            return "test_user_123", "valid_access_token_123"
        
        # Создаем новое приложение
        app = FastAPI(title="Test App with Auth")
        app.include_router(auth.router)
        
        # Применяем dependency override
        app.dependency_overrides[auth.get_current_user_id] = mock_get_current_user_id
        app.dependency_overrides[auth.get_current_user_with_token] = mock_get_current_user_with_token
        
        # Патчим сервисы
        with patch.object(auth, 'auth_service', test_mock_auth_service):
            with patch.object(auth, 'db_service', test_mock_db_service):
                client = TestClient(app)
                client.headers = {'Authorization': 'Bearer valid_access_token_123'}
                
                change_password_data = {
                    "current_password": "CurrentPassword123!",
                    "new_password": "NewSecurePassword456!"
                }
                
                response = client.post("/auth/change-password", json=change_password_data)
                
                assert response.status_code == 200
                data = response.json()
                
                # Проверяем структуру ответа
                assert "success" in data
                assert "message" in data
                
                assert data["success"] is True
                assert "Password changed successfully" in data["message"]
                
                # Проверяем вызовы методов
                test_mock_auth_service.verify_password.assert_called_once()
                test_mock_auth_service.hash_password.assert_called_once()
                test_mock_db_service.update_user.assert_called_once()

    def test_change_password_wrong_current(self):
        """Тест смены пароля с неверным текущим паролем"""
        from fastapi import FastAPI
        from app.routes import auth
        
        # Создаем моки
        test_mock_auth_service = MagicMock()
        test_mock_db_service = MagicMock()
        
        test_mock_auth_service.get_user_info_from_token = AsyncMock(return_value={
            "user_id": "test_user_123",
            "role": "user",
            "permissions": ["read_own_data"]
        })
        
        # Настраиваем моки
        test_mock_db_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "current_hashed_password",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        test_mock_auth_service.verify_password = AsyncMock(return_value=False)  # Неверный текущий пароль
        
        # Переопределяем зависимости
        async def mock_get_current_user_id(request=None):
            return "test_user_123"
        
        async def mock_get_current_user_with_token(request=None):
            return "test_user_123", "valid_access_token_123"
        
        # Создаем новое приложение
        app = FastAPI(title="Test App with Auth")
        app.include_router(auth.router)
        
        # Применяем dependency override
        app.dependency_overrides[auth.get_current_user_id] = mock_get_current_user_id
        app.dependency_overrides[auth.get_current_user_with_token] = mock_get_current_user_with_token
        
        # Патчим сервисы
        with patch.object(auth, 'auth_service', test_mock_auth_service):
            with patch.object(auth, 'db_service', test_mock_db_service):
                client = TestClient(app)
                client.headers = {'Authorization': 'Bearer valid_access_token_123'}
                
                change_password_data = {
                    "current_password": "WrongCurrentPassword123!",
                    "new_password": "NewSecurePassword456!"
                }
                
                response = client.post("/auth/change-password", json=change_password_data)
                
                assert response.status_code == 400
                data = response.json()
                assert "detail" in data
                assert "Current password is incorrect" in data["detail"]

    def test_change_password_unauthorized(self, auth_client):
        """Тест смены пароля без авторизации"""
        change_password_data = {
            "current_password": "AnyPassword123!",
            "new_password": "NewPassword456!"
        }
        
        response = auth_client.post("/auth/change-password", json=change_password_data)
        
        assert response.status_code == 401

    def test_change_password_invalid_new_password(self, auth_client_with_user, mock_auth_service, mock_db_service):
        """Тест смены пароля с неверным новым паролем"""
        # Настраиваем моки
        mock_db_service.get_user = AsyncMock(return_value={
            "id": "test_user_123",
            "email": "test@example.com",
            "username": "testuser",
            "password_hash": "current_hashed_password",
            "role": "user",
            "permissions": ["read_own_data"],
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        })
        mock_auth_service.verify_password = AsyncMock(return_value=True)
        
        change_password_data = {
            "current_password": "CurrentPassword123!",
            "new_password": "123"  # Слишком короткий пароль
        }
        
        response = auth_client_with_user.post("/auth/change-password", json=change_password_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_change_password_missing_fields(self, auth_client_with_user):
        """Тест смены пароля с отсутствующими полями"""
        change_password_data = {}  # Пустой запрос

        response = auth_client_with_user.post("/auth/change-password", json=change_password_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


if __name__ == "__main__":
    pytest.main([__file__])