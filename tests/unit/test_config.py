import pytest
import os
from unittest.mock import patch
from app.config import Settings, settings


class TestSettings:
    """Тесты для настроек приложения"""
    
    def test_settings_creation(self):
        """Тест создания настроек"""
        settings_obj = Settings()
        assert settings_obj is not None
        assert hasattr(settings_obj, 'DATABASE_URL')
        assert hasattr(settings_obj, 'REDIS_URL')
        assert hasattr(settings_obj, 'DEBUG')
        
    def test_settings_instance(self):
        """Тест глобального экземпляра настроек"""
        assert settings is not None
        assert isinstance(settings, Settings)
    
    @patch.dict(os.environ, {
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test_db',
        'DEBUG': 'true',
        'JWT_SECRET_KEY': 'test-secret-key'
    })
    def test_settings_from_env(self):
        """Тест загрузки настроек из переменных окружения"""
        settings_obj = Settings()
        assert settings_obj.DATABASE_URL == 'postgresql://test:test@localhost:5432/test_db'
        assert settings_obj.DEBUG is True
        assert settings_obj.JWT_SECRET_KEY == 'test-secret-key'
        
    def test_allowed_image_formats_list(self):
        """Тест списка разрешенных форматов изображений"""
        settings_obj = Settings()
        formats = settings_obj.allowed_image_formats_list
        assert isinstance(formats, list)
        assert 'JPEG' in formats
        assert 'PNG' in formats
        assert 'JPG' in formats
        
    def test_cors_origins_list(self):
        """Тест списка CORS источников"""
        settings_obj = Settings()
        origins = settings_obj.cors_origins_list
        assert isinstance(origins, list)
        
    def test_validate_debug(self):
        """Тест валидации debug режима через model_validator"""
        # Тест валидного значения (development + debug)
        with patch.dict(os.environ, {'ENVIRONMENT': 'development', 'DEBUG': 'true'}):
            settings = Settings()
            assert settings.DEBUG is True
            assert settings.ENVIRONMENT == "development"
        
        # Тест невалидного значения (production + debug должно вызвать ошибку)
        with patch.dict(os.environ, {'ENVIRONMENT': 'production', 'DEBUG': 'true'}):
            with pytest.raises(ValueError, match="DEBUG mode cannot be enabled in production environment"):
                Settings()
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'development',
        'DEBUG': 'true'
    })
    def test_debug_validation_with_development(self):
        """Тест валидации debug в development режиме"""
        settings_obj = Settings()
        # В development режиме debug может быть True
        assert settings_obj.DEBUG is True
        
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'production',
        'DEBUG': 'false'
    })
    def test_debug_validation_with_production(self):
        """Тест валидации debug в production режиме"""
        settings_obj = Settings()
        # В production режиме debug должен быть False
        assert settings_obj.DEBUG is False
    
    def test_debug_validation_production_fails(self, monkeypatch):
        """Тест того, что debug=True в production вызывает ошибку"""
        # Устанавливаем переменные окружения для теста
        monkeypatch.setenv('DEBUG', 'true')
        monkeypatch.setenv('ENVIRONMENT', 'production')
        
        # Валидатор DEBUG использует правильный синтаксис Pydantic v2
        # Он должен вызвать ошибку при DEBUG=True в production
        with pytest.raises(ValueError, match="DEBUG mode cannot be enabled in production environment"):
            Settings()
    
    def test_default_values(self, monkeypatch):
        """Тест значений по умолчанию"""
        # Используем monkeypatch для установки значений по умолчанию
        monkeypatch.setenv('DEBUG', 'false')
        monkeypatch.setenv('ENVIRONMENT', 'production')
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///./test.db')
        
        settings_obj = Settings()
        
        # Проверяем важные значения по умолчанию
        assert settings_obj.APP_NAME == "Face Recognition Service"
        assert settings_obj.DEBUG is False
        assert settings_obj.ENVIRONMENT == "production"
        assert settings_obj.HOST == "0.0.0.0"
        assert settings_obj.PORT == 8000
        assert settings_obj.MAX_UPLOAD_SIZE == 10 * 1024 * 1024  # 10MB
        assert settings_obj.THRESHOLD_DEFAULT == 0.80
        assert settings_obj.RATE_LIMIT_REQUESTS_PER_MINUTE == 60
    
    def test_database_configuration(self, monkeypatch):
        """Тест конфигурации базы данных"""
        # Используем monkeypatch для установки тестовых значений
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///./test.db')
        monkeypatch.setenv('DATABASE_POOL_SIZE', '10')
        monkeypatch.setenv('DATABASE_MAX_OVERFLOW', '20')
        
        settings_obj = Settings()
        
        assert hasattr(settings_obj, 'DATABASE_URL')
        assert hasattr(settings_obj, 'DATABASE_POOL_SIZE')
        assert hasattr(settings_obj, 'DATABASE_MAX_OVERFLOW')
        assert settings_obj.DATABASE_POOL_SIZE == 10
        assert settings_obj.DATABASE_MAX_OVERFLOW == 20
    
    def test_redis_configuration(self):
        """Тест конфигурации Redis"""
        settings_obj = Settings()
        
        assert hasattr(settings_obj, 'REDIS_URL')
        assert hasattr(settings_obj, 'REDIS_CONNECTION_POOL_SIZE')
        assert settings_obj.REDIS_CONNECTION_POOL_SIZE == 10
        assert settings_obj.REDIS_SOCKET_TIMEOUT == 5
    
    def test_s3_configuration(self):
        """Тест конфигурации S3/MinIO"""
        settings_obj = Settings()
        
        assert hasattr(settings_obj, 'S3_ENDPOINT_URL')
        assert hasattr(settings_obj, 'S3_ACCESS_KEY')
        assert hasattr(settings_obj, 'S3_SECRET_KEY')
        assert hasattr(settings_obj, 'S3_BUCKET_NAME')
        assert settings_obj.S3_BUCKET_NAME == "test-bucket"  # Исправлено: используем тестовое значение
        assert settings_obj.S3_USE_SSL is False
    
    def test_security_configuration(self):
        """Тест конфигурации безопасности"""
        settings_obj = Settings()
        
        assert hasattr(settings_obj, 'JWT_SECRET_KEY')
        assert hasattr(settings_obj, 'ENCRYPTION_KEY')
        assert hasattr(settings_obj, 'MAX_UPLOAD_SIZE')
        assert settings_obj.MAX_UPLOAD_SIZE == 10 * 1024 * 1024
    
    def test_threshold_configuration(self):
        """Тест конфигурации пороговых значений"""
        settings_obj = Settings()
        
        assert hasattr(settings_obj, 'THRESHOLD_DEFAULT')
        assert hasattr(settings_obj, 'THRESHOLD_MIN')
        assert hasattr(settings_obj, 'THRESHOLD_MAX')
        assert settings_obj.THRESHOLD_DEFAULT == 0.80
        assert settings_obj.THRESHOLD_MIN == 0.50
        assert settings_obj.THRESHOLD_MAX == 0.95
        assert settings_obj.TARGET_FAR == 0.001
        assert settings_obj.TARGET_FRR == 0.02
