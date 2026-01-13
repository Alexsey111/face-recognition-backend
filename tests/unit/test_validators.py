import pytest
from unittest.mock import patch, MagicMock
from app.utils.validators import (
    validate_email,
    validate_username,
    validate_password,
    validate_image_format,
    validate_image_size,
    validate_uuid,
    validate_date,
    validate_url,
    validate_phone_number,
    sanitize_string
)
from app.utils.exceptions import ValidationError


class TestValidators:
    """Тесты для валидаторов"""
    
    
    
    
    
    
    
    
    
    # New tests from the edit
    def test_validate_email_valid(self):
        """Тест валидных email адресов"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "test+tag@example.org",
            "user123@test-domain.com"
        ]
        
        for email in valid_emails:
            result = validate_email(email)
            assert result is True
    
    def test_validate_email_invalid(self):
        """Тест невалидных email адресов"""
        invalid_emails = [
            "",
            "invalid_email",
            "@example.com",
            "test@",
            "test..test@example.com",
            "a" * 256 + "@example.com"  # Слишком длинный
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                validate_email(email)
    
    def test_validate_username_valid(self):
        """Тест валидных имен пользователя"""
        valid_usernames = [
            "user123",
            "user_name",
            "user-name",
            "UserName",
            "user_name_123"
        ]
        
        for username in valid_usernames:
            result = validate_username(username)
            assert result is True
    
    def test_validate_username_invalid(self):
        """Тест невалидных имен пользователя"""
        invalid_usernames = [
            "",
            "ab",  # Слишком короткий
            "a" * 51,  # Слишком длинный
            "user@name",  # Неразрешенный символ
            "user name",  # Пробел
            "user.name"  # Точка не разрешена
        ]
        
        for username in invalid_usernames:
            with pytest.raises(ValidationError):
                validate_username(username)
    
    def test_validate_password_valid(self):
        """Тест валидных паролей"""
        valid_passwords = [
            "Password123!",
            "SecurePass@2023",
            "MyP@ssw0rd",
            "Test123#Password"
        ]
        
        for password in valid_passwords:
            result = validate_password(password)
            assert result is True
    
    def test_validate_password_invalid(self):
        """Тест невалидных паролей"""
        invalid_passwords = [
            "",
            "short",  # Слишком короткий
            "a" * 129,  # Слишком длинный
            "lowercaseonly",
            "UPPERCASEONLY",
            "123456789",
            "NoSpecialChars123"
        ]
        
        for password in invalid_passwords:
            with pytest.raises(ValidationError):
                validate_password(password)
    
    def test_validate_image_format_valid(self):
        """Тест валидных форматов изображений"""
        valid_formats = [
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD",
            "image.jpg",
            "photo.png",
            "picture.webp"
        ]
        
        for image_data in valid_formats:
            result = validate_image_format(image_data)
            assert result is True
    
    def test_validate_image_format_invalid(self):
        """Тест невалидных форматов изображений"""
        invalid_formats = [
            "",
            "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP",
            "document.pdf",
            "video.mp4"
        ]
        
        for image_data in invalid_formats:
            with pytest.raises(ValidationError):
                validate_image_format(image_data)
    
    def test_validate_image_size_valid(self):
        """Тест валидного размера изображения"""
        # Тест с размером меньше лимита (10MB по умолчанию)
        valid_size = 1024 * 1024  # 1MB
        result = validate_image_size("dGVzdCBkYXRh", max_size=valid_size + 1024)
        assert result is True
    
    def test_validate_image_size_too_large(self):
        """Тест слишком большого размера изображения"""
        # Тест с размером больше лимита
        large_size = 20 * 1024 * 1024  # 20MB
        with pytest.raises(ValidationError):
            validate_image_size("x" * large_size, max_size=1024)  # 1KB лимит
    
    def test_validate_image_size_empty(self):
        """Тест пустых данных изображения"""
        with pytest.raises(ValidationError):
            validate_image_size("")
    
    def test_validate_image_size_invalid_base64(self):
        """Тест некорректных base64 данных"""
        with pytest.raises(ValidationError):
            validate_image_size("invalid_base64!")
    
    def test_validate_uuid_valid(self):
        """Тест валидных UUID"""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "123e4567-e89b-12d3-a456-426614174000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        ]
        
        for uuid_str in valid_uuids:
            result = validate_uuid(uuid_str)
            assert result is True
    
    def test_validate_uuid_invalid(self):
        """Тест невалидных UUID"""
        invalid_uuids = [
            "",
            "invalid-uuid",
            "550e8400-e29b-41d4-a716-4466554400",  # Слишком короткий
            "550e8400-e29b-41d4-a716-4466554400000",  # Слишком длинный
            "550e8400-e29b-41d4-a716-44665544000g",  # Неверный символ
        ]
        
        for uuid_str in invalid_uuids:
            with pytest.raises(ValidationError):
                validate_uuid(uuid_str)
    
    def test_validate_date_valid(self):
        """Тест валидных дат"""
        valid_dates = [
            "2024-01-01",
            "2023-12-31",
            "2024-02-29"  # Високосный год
        ]
        
        for date_str in valid_dates:
            result = validate_date(date_str)
            assert result is True
    
    def test_validate_date_invalid(self):
        """Тест невалидных дат"""
        invalid_dates = [
            "",
            "2024-13-01",  # Невалидный месяц
            "2024-02-30",  # Невалидный день
            "2024/01/01",  # Неправильный формат
            "not-a-date"
        ]
        
        for date_str in invalid_dates:
            with pytest.raises(ValidationError):
                validate_date(date_str)
    
    def test_validate_url_valid(self):
        """Тест валидных URL"""
        valid_urls = [
            "https://example.com",
            "http://localhost:8000",
            "https://sub.domain.com/path?query=value",
            "http://192.168.1.1:8080"
        ]
        
        for url in valid_urls:
            result = validate_url(url)
            assert result is True
    
    def test_validate_url_invalid(self):
        """Тест невалидных URL"""
        invalid_urls = [
            "",
            "not_a_url",
            "ftp://example.com",  # Не поддерживаемый протокол
            "https://",
            "example.com"  # Отсутствует протокол
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validate_url(url)
    
    def test_validate_phone_number_valid(self):
        """Тест валидных номеров телефонов"""
        valid_phones = [
            "+1234567890",
            "1234567890",
            "+1 (234) 567-890",
            "123-456-7890"
        ]
        
        for phone in valid_phones:
            result = validate_phone_number(phone)
            assert result is True
    
    def test_validate_phone_number_invalid(self):
        """Тест невалидных номеров телефонов"""
        invalid_phones = [
            "",
            "123",  # Слишком короткий
            "a" * 16,  # Слишком длинный
            "abc123def",  # Не только цифры и плюс
        ]
        
        for phone in invalid_phones:
            with pytest.raises(ValidationError):
                validate_phone_number(phone)
    
    def test_sanitize_string_basic(self):
        """Тест базовой санитизации строки"""
        test_cases = [
            ("hello world", "hello world"),
            ("hello<script>alert('xss')</script>", "helloscriptalert(xss)/script"),
            ('text with "quotes"', 'text with quotes'),
            ("text with 'quotes'", "text with quotes"),
            ("text with <tags>", "text with tags")
        ]
        
        for input_text, expected in test_cases:
            result = sanitize_string(input_text)
            assert result == expected
    
    def test_sanitize_string_with_limits(self):
        """Тест санитизации с ограничением длины"""
        long_text = "a" * 100
        result = sanitize_string(long_text, max_length=50)
        assert len(result) == 50
        assert result == "a" * 50
    
    def test_sanitize_string_with_allowed_chars(self):
        """Тест санитизации с разрешенными символами"""
        text = "hello123!@#"
        result = sanitize_string(text, allowed_chars="hello")
        assert result == "hello"
    
    def test_sanitize_string_empty(self):
        """Тест санитизации пустой строки"""
        result = sanitize_string("")
        assert result == ""
    
    def test_sanitize_string_none(self):
        """Тест санитизации None"""
        result = sanitize_string(None)
        assert result == ""
