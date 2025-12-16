"""
Валидаторы данных.
Проверка корректности входных данных и форматов.
"""

import re
import base64
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
import uuid
import hashlib
import html

from .constants import (
    IMAGE_FORMATS, FILE_LIMITS, MAGIC_NUMBERS, SIMILARITY_LIMITS, 
    RATE_LIMITS, PASSWORD_REGEX, EMAIL_REGEX, USERNAME_REGEX
)
from .exceptions import ValidationError

# 🟡 Регулярные выражения для валидации теперь импортируются из constants.py
# PASSWORD_REGEX определен в constants.py с расширенными спецсимволами


def validate_email(email: str) -> bool:
    """
    Валидация email адреса.

    Args:
        email: Email адрес для проверки

    Returns:
        bool: True если email валиден

    Raises:
        ValidationError: Если email невалиден
    """
    if not email or not isinstance(email, str):
        raise ValidationError("Email is required")

    if len(email) > 255:
        raise ValidationError("Email is too long (max 255 characters)")

    if not EMAIL_REGEX.match(email):
        raise ValidationError("Invalid email format")

    return True


def validate_username(username: str) -> bool:
    """
    Валидация имени пользователя.

    Args:
        username: Имя пользователя для проверки

    Returns:
        bool: True если имя пользователя валидно

    Raises:
        ValidationError: Если имя пользователя невалидно
    """
    if not username or not isinstance(username, str):
        raise ValidationError("Username is required")

    if len(username) < 3:
        raise ValidationError("Username is too short (min 3 characters)")

    if len(username) > 50:
        raise ValidationError("Username is too long (max 50 characters)")

    if not USERNAME_REGEX.match(username):
        raise ValidationError(
            "Username can only contain letters, numbers, underscores, and hyphens"
        )

    return True


def validate_password(password: str) -> bool:
    """
    Валидация пароля.

    Args:
        password: Пароль для проверки

    Returns:
        bool: True если пароль валиден

    Raises:
        ValidationError: Если пароль невалиден
    """
    if not password or not isinstance(password, str):
        raise ValidationError("Password is required")

    if len(password) < 8:
        raise ValidationError("Password is too short (min 8 characters)")

    if len(password) > 128:
        raise ValidationError("Password is too long (max 128 characters)")

    # Проверяем сложность пароля
    if not PASSWORD_REGEX.match(password):
        raise ValidationError(
            "Password must contain at least one uppercase letter, "
            "one lowercase letter, one digit, and one special character"
        )

    return True


def validate_image_format(image_data: str) -> bool:
    """
    Валидация формата изображения.

    Args:
        image_data: Данные изображения (base64, URL или путь к файлу)

    Returns:
        bool: True если формат поддерживается

    Raises:
        ValidationError: Если формат не поддерживается
    """
    if not image_data:
        raise ValidationError("Image data is required")

    # Определяем формат по префиксу или расширению
    format_type = _detect_image_format(image_data)

    if format_type not in IMAGE_FORMATS:
        raise ValidationError(
            f"Unsupported image format: {format_type}. "
            f"Supported formats: {', '.join(IMAGE_FORMATS)}"
        )

    return True


def validate_image_size(
    image_data: Union[str, bytes], max_size: int = FILE_LIMITS["max_image_size"]
) -> bool:
    """
    Валидация размера изображения.

    Args:
        image_data: Данные изображения
        max_size: Максимальный размер в байтах

    Returns:
        bool: True если размер в пределах нормы

    Raises:
        ValidationError: Если размер превышает лимит
    """
    if not image_data:
        raise ValidationError("Image data is required")

    # Если данные в виде строки (base64), декодируем для проверки размера
    if isinstance(image_data, str):
        try:
            if image_data.startswith("data:image/"):
                # Data URL формат
                _, base64_data = image_data.split(",", 1)
                decoded_data = base64.b64decode(base64_data)
            else:
                # Предполагаем, что это чистый base64
                decoded_data = base64.b64decode(image_data)

            size = len(decoded_data)
        except Exception:
            # Если не удалось декодировать, возвращаем False
            raise ValidationError("Invalid image data format")
    else:
        # Если данные уже в виде bytes
        size = len(image_data)

    if size > max_size:
        size_mb = size / (1024 * 1024)
        max_size_mb = max_size / (1024 * 1024)
        raise ValidationError(
            f"Image is too large: {size_mb:.2f}MB. "
            f"Maximum allowed size: {max_size_mb:.2f}MB"
        )

    return True


def validate_uuid(uuid_string: str) -> bool:
    """
    Валидация UUID.

    Args:
        uuid_string: Строка UUID для проверки

    Returns:
        bool: True если UUID валиден

    Raises:
        ValidationError: Если UUID невалиден
    """
    if not uuid_string or not isinstance(uuid_string, str):
        raise ValidationError("UUID is required")

    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        raise ValidationError("Invalid UUID format")


def validate_date(date_string: str, format: str = "%Y-%m-%d") -> bool:
    """
    Валидация даты.

    Args:
        date_string: Строка даты для проверки
        format: Формат даты

    Returns:
        bool: True если дата валидна

    Raises:
        ValidationError: Если дата невалидна
    """
    if not date_string or not isinstance(date_string, str):
        raise ValidationError("Date string is required")

    try:
        datetime.strptime(date_string, format)
        return True
    except ValueError:
        raise ValidationError(f"Invalid date format. Expected: {format}")


def validate_url(url: str) -> bool:
    """
    Валидация URL.

    Args:
        url: URL для проверки

    Returns:
        bool: True если URL валиден

    Raises:
        ValidationError: Если URL невалиден
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL is required")

    # Простая проверка URL
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    if not url_pattern.match(url):
        raise ValidationError("Invalid URL format")

    return True


def validate_file_hash(
    data: Union[str, bytes], expected_hash: str, algorithm: str = "sha256"
) -> bool:
    """
    Валидация хеша файла.

    Args:
        data: Данные файла
        expected_hash: Ожидаемый хеш
        algorithm: Алгоритм хеширования

    Returns:
        bool: True если хеши совпадают

    Raises:
        ValidationError: Если хеши не совпадают
    """
    if not data or not expected_hash:
        raise ValidationError("Data and hash are required")

    # Вычисляем хеш данных
    if algorithm.lower() == "sha256":
        hash_obj = hashlib.sha256()
    elif algorithm.lower() == "md5":
        hash_obj = hashlib.md5()
    else:
        raise ValidationError(f"Unsupported hash algorithm: {algorithm}")

    if isinstance(data, str):
        try:
            # Если это base64 строка, декодируем
            if data.startswith("data:image/"):
                _, base64_data = data.split(",", 1)
                data = base64.b64decode(base64_data)
            else:
                data = base64.b64decode(data)
        except Exception:
            raise ValidationError("Invalid data format for hash calculation")

    hash_obj.update(data)
    calculated_hash = hash_obj.hexdigest()

    if calculated_hash.lower() != expected_hash.lower():
        raise ValidationError(
            f"Hash mismatch. Expected: {expected_hash}, "
            f"Calculated: {calculated_hash}"
        )

    return True


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Простая валидация JSON схемы.

    Args:
        data: Данные для валидации
        schema: Схема для проверки

    Returns:
        bool: True если данные соответствуют схеме

    Raises:
        ValidationError: Если данные не соответствуют схеме
    """
    if not isinstance(data, dict):
        raise ValidationError("Data must be a dictionary")

    # Проверяем обязательные поля
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Required field missing: {field}")

    # Проверяем типы полей
    field_types = schema.get("properties", {})
    for field, value in data.items():
        if field in field_types:
            expected_type = field_types[field].get("type")
            if expected_type and not _check_type(value, expected_type):
                raise ValidationError(
                    f"Field '{field}' must be of type {expected_type}, "
                    f"got {type(value).__name__}"
                )

    # Проверяем ограничения значений
    constraints = schema.get("constraints", {})
    for field, value in data.items():
        if field in constraints:
            constraint = constraints[field]

            # Минимальное/максимальное значение для чисел
            if isinstance(value, (int, float)):
                if "min" in constraint and value < constraint["min"]:
                    raise ValidationError(
                        f"Field '{field}' must be >= {constraint['min']}"
                    )
                if "max" in constraint and value > constraint["max"]:
                    raise ValidationError(
                        f"Field '{field}' must be <= {constraint['max']}"
                    )

            # Длина для строк
            if isinstance(value, str):
                if "min_length" in constraint and len(value) < constraint["min_length"]:
                    raise ValidationError(
                        f"Field '{field}' must be at least {constraint['min_length']} characters"
                    )
                if "max_length" in constraint and len(value) > constraint["max_length"]:
                    raise ValidationError(
                        f"Field '{field}' must be at most {constraint['max_length']} characters"
                    )

                # Регулярное выражение
                if "pattern" in constraint:
                    if not re.match(constraint["pattern"], value):
                        raise ValidationError(
                            f"Field '{field}' does not match required pattern"
                        )

    return True


def validate_list_items(data: List[Any], item_validator: callable = None) -> bool:
    """
    Валидация элементов списка.

    Args:
        data: Список для валидации
        item_validator: Функция для валидации каждого элемента

    Returns:
        bool: True если все элементы валидны

    Raises:
        ValidationError: Если найдены невалидные элементы
    """
    if not isinstance(data, list):
        raise ValidationError("Data must be a list")

    if item_validator:
        for i, item in enumerate(data):
            try:
                item_validator(item)
            except ValidationError as e:
                raise ValidationError(f"Invalid item at index {i}: {str(e)}")

    return True


def sanitize_string(
    text: str, max_length: int = None, allowed_chars: str = None
) -> str:
    """
    Санитизация строки.

    Args:
        text: Строка для санитизации
        max_length: Максимальная длина
        allowed_chars: Разрешенные символы

    Returns:
        str: Санитизированная строка
    """
    if not text:
        return ""

    # Удаляем опасные символы
    if allowed_chars:
        # Оставляем только разрешенные символы
        sanitized = "".join(
            c for c in text if c in allowed_chars or c.isalnum() or c.isspace()
        )
    else:
        # Удаляем только опасные символы
        sanitized = re.sub(r'[<>"\']', "", text)

    # Обрезаем до максимальной длины
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized.strip()


def validate_phone_number(phone: str) -> bool:
    """
    Валидация номера телефона.

    Args:
        phone: Номер телефона для проверки

    Returns:
        bool: True если номер валиден

    Raises:
        ValidationError: Если номер невалиден
    """
    if not phone or not isinstance(phone, str):
        raise ValidationError("Phone number is required")

    # Удаляем все кроме цифр и плюса
    cleaned_phone = re.sub(r"[^\d+]", "", phone)

    # Проверяем, что номер содержит только цифры и возможно плюс в начале
    if not re.match(r"^\+?\d{10,15}$", cleaned_phone):
        raise ValidationError("Invalid phone number format")

    return True


def validate_coordinates(lat: float, lng: float) -> bool:
    """
    Валидация географических координат.

    Args:
        lat: Широта
        lng: Долгота

    Returns:
        bool: True если координаты валидны

    Raises:
        ValidationError: Если координаты невалидны
    """
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        raise ValidationError("Coordinates must be numbers")

    if not (-90 <= lat <= 90):
        raise ValidationError("Latitude must be between -90 and 90")

    if not (-180 <= lng <= 180):
        raise ValidationError("Longitude must be between -180 and 180")

    return True


# =============================================================================
# НОВЫЕ ФУНКЦИИ БЕЗОПАСНОСТИ
# =============================================================================

def sanitize_html(text: str) -> str:
    """
    Защита от XSS - удаление HTML тегов и экранирование специальных символов.
    
    Args:
        text: Текст для санитизации
        
    Returns:
        str: Санитизированный текст
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Удаляем HTML теги
    text = re.sub(r"<[^>]+>", "", text)
    
    # Экранируем специальные HTML символы
    return html.escape(text, quote=True)


def validate_sql_safe(text: str) -> bool:
    """
    Защита от SQL injection - проверка на опасные паттерны.
    
    Args:
        text: Текст для проверки
        
    Returns:
        bool: True если текст безопасен
        
    Raises:
        ValidationError: Если обнаружены потенциально опасные паттерны
    """
    if not text or not isinstance(text, str):
        return True
    
    # Паттерны SQL injection
    dangerous_patterns = [
        r"(\bOR\b|\bAND\b).*=.*",  # OR 1=1, AND password=...
        r";\s*(DROP|DELETE|UPDATE|INSERT|SELECT|CREATE|ALTER)",  # ; DROP TABLE
        r"--",  # SQL комментарии
        r"/\*.*\*/",  # Блочные комментарии
        r"UNION\s+SELECT",  # UNION атаки
        r"EXEC(UTE)?\s+",  # EXECUTE команды
        r"INFORMATION_SCHEMA",  # Попытки доступа к метаданным
        r"XP_CMDSHELL",  # Опасные системные процедуры
    ]
    
    text_upper = text.upper()
    for pattern in dangerous_patterns:
        if re.search(pattern, text_upper, re.IGNORECASE):
            raise ValidationError("Potentially dangerous input detected")
    
    return True


def validate_embedding(embedding: List[float]) -> bool:
    """
    Валидация вектора эмбеддинга для ML моделей.
    
    Args:
        embedding: Вектор эмбеддинга
        
    Returns:
        bool: True если эмбеддинг валиден
        
    Raises:
        ValidationError: Если эмбеддинг невалиден
    """
    if not isinstance(embedding, (list, tuple)):
        raise ValidationError("Embedding must be a list or tuple")
    
    if not embedding:
        raise ValidationError("Embedding cannot be empty")
    
    # Проверяем размер
    embedding_size = len(embedding)
    if embedding_size < FILE_LIMITS["min_embedding_size"]:
        raise ValidationError(
            f"Embedding too small: {embedding_size}. "
            f"Minimum required: {FILE_LIMITS['min_embedding_size']}"
        )
    
    if embedding_size > FILE_LIMITS["max_embedding_size"]:
        raise ValidationError(
            f"Embedding too large: {embedding_size}. "
            f"Maximum allowed: {FILE_LIMITS['max_embedding_size']}"
        )
    
    # Проверяем, что все значения - числа
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in embedding):
        raise ValidationError("Embedding must contain only numeric values")
    
    # Проверяем на NaN и бесконечность
    import math
    for i, value in enumerate(embedding):
        if math.isnan(value) or math.isinf(value):
            raise ValidationError(f"Embedding contains invalid value at index {i}")
    
    return True


def validate_similarity_threshold(threshold: float) -> bool:
    """
    Валидация порога схожести для сравнения эмбеддингов.
    
    Args:
        threshold: Порог схожести (0.0 - 1.0)
        
    Returns:
        bool: True если порог валиден
        
    Raises:
        ValidationError: Если порог вне допустимых пределов
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValidationError("Threshold must be a number")
    
    min_threshold = SIMILARITY_LIMITS["min_threshold"]
    max_threshold = SIMILARITY_LIMITS["max_threshold"]
    
    if not (min_threshold <= threshold <= max_threshold):
        raise ValidationError(
            f"Threshold must be between {min_threshold} and {max_threshold}"
        )
    
    return True


def validate_file_upload(filename: str, content_type: str, file_size: int) -> bool:
    """
    Комплексная валидация загружаемого файла.
    
    Args:
        filename: Имя файла
        content_type: MIME тип файла
        file_size: Размер файла в байтах
        
    Returns:
        bool: True если файл валиден
        
    Raises:
        ValidationError: Если файл невалиден
    """
    if not filename or not isinstance(filename, str):
        raise ValidationError("Filename is required")
    
    if not content_type or not isinstance(content_type, str):
        raise ValidationError("Content type is required")
    
    if not isinstance(file_size, int) or file_size <= 0:
        raise ValidationError("File size must be a positive integer")
    
    # Валидация размера файла
    if file_size > FILE_LIMITS["max_image_size"]:
        size_mb = file_size / (1024 * 1024)
        max_size_mb = FILE_LIMITS["max_image_size"] / (1024 * 1024)
        raise ValidationError(
            f"File too large: {size_mb:.2f}MB. "
            f"Maximum allowed: {max_size_mb:.2f}MB"
        )
    
    # Валидация имени файла
    if len(filename) > FILE_LIMITS["max_filename_length"]:
        raise ValidationError(
            f"Filename too long: {len(filename)} characters. "
            f"Maximum allowed: {FILE_LIMITS['max_filename_length']}"
        )
    
    # Проверяем расширение файла
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        raise ValidationError(
            f"Unsupported file extension. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Валидация MIME типа
    if not content_type.startswith("image/"):
        raise ValidationError("Only image files are allowed")
    
    return True


def validate_api_key(api_key: str) -> bool:
    """
    Валидация API ключа.
    
    Args:
        api_key: API ключ для проверки
        
    Returns:
        bool: True если ключ валиден
        
    Raises:
        ValidationError: Если ключ невалиден
    """
    if not api_key or not isinstance(api_key, str):
        raise ValidationError("API key is required")
    
    # Проверяем формат с помощью regex
    if not re.match(r"^[A-Za-z0-9_-]{32,128}$", api_key):
        raise ValidationError(
            "Invalid API key format. Must be 32-128 alphanumeric characters with underscores/hyphens"
        )
    
    return True


def validate_rate_limit_config(config: Dict[str, Any]) -> bool:
    """
    Валидация конфигурации rate limiting.
    
    Args:
        config: Конфигурация rate limiting
        
    Returns:
        bool: True если конфигурация валидна
        
    Raises:
        ValidationError: Если конфигурация невалидна
    """
    if not isinstance(config, dict):
        raise ValidationError("Rate limit config must be a dictionary")
    
    # Проверяем обязательные поля
    required_fields = ["requests_per_minute", "burst_size"]
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}")
    
    # Валидируем типы и значения
    if not isinstance(config["requests_per_minute"], int) or config["requests_per_minute"] <= 0:
        raise ValidationError("requests_per_minute must be a positive integer")
    
    if not isinstance(config["burst_size"], int) or config["burst_size"] <= 0:
        raise ValidationError("burst_size must be a positive integer")
    
    if config["burst_size"] > config["requests_per_minute"]:
        raise ValidationError("burst_size cannot exceed requests_per_minute")
    
    # Опциональные поля
    if "block_duration" in config:
        if not isinstance(config["block_duration"], int) or config["block_duration"] <= 0:
            raise ValidationError("block_duration must be a positive integer")
    
    return True


# Вспомогательные функции


def _detect_image_format(image_data: Union[str, bytes]) -> str:
    """
    Определение формата изображения по данным с поддержкой magic numbers.
    
    Улучшенная версия с поддержкой бинарных данных и определением формата
    по file signatures (magic numbers).

    Args:
        image_data: Данные изображения (строка или bytes)

    Returns:
        str: Формат изображения
    """
    try:
        # Если данные в виде bytes, проверяем magic numbers
        if isinstance(image_data, bytes):
            return _detect_format_by_magic_number(image_data)
        
        # Если это строка
        if isinstance(image_data, str):
            # Сначала проверяем по расширению файла (самый быстрый способ)
            if "." in image_data:
                extension = image_data.split(".")[-1].upper()
                extension_mapping = {
                    "JPG": "JPEG",
                    "JPEG": "JPEG",
                    "PNG": "PNG", 
                    "WEBP": "WEBP",
                    "GIF": "GIF",
                    "BMP": "BMP",
                    "HEIC": "HEIC",
                    "HEIF": "HEIC"
                }
                detected_format = extension_mapping.get(extension)
                if detected_format:
                    return detected_format
            
            # Data URL формат
            if image_data.startswith("data:image/"):
                mime_type = image_data.split(";")[0].split("/")[1].upper()
                format_mapping = {
                    "JPEG": "JPEG",
                    "JPG": "JPEG", 
                    "PNG": "PNG",
                    "WEBP": "WEBP",
                    "GIF": "GIF",
                    "BMP": "BMP",
                    "HEIC": "HEIC"
                }
                return format_mapping.get(mime_type, "UNKNOWN")
            
            # Если это base64 строка, декодируем и проверяем magic numbers
            try:
                if image_data.startswith("data:image/"):
                    # Data URL формат
                    _, base64_data = image_data.split(",", 1)
                    binary_data = base64.b64decode(base64_data)
                else:
                    # Предполагаем чистый base64
                    binary_data = base64.b64decode(image_data)
                
                return _detect_format_by_magic_number(binary_data)
                
            except Exception:
                # Если не удалось декодировать, возвращаем результат по расширению если был
                pass
    
    except Exception as e:
        # Логируем ошибку но продолжаем
        import logging
        logging.getLogger(__name__).debug(f"Error detecting image format: {e}")
    
    # По умолчанию
    return "UNKNOWN"


def _detect_format_by_magic_number(data: bytes) -> str:
    """
    Определение формата файла по magic numbers (file signatures).
    
    Args:
        data: Бинарные данные файла
        
    Returns:
        str: Формат файла
    """
    if not data or len(data) < 4:
        return "UNKNOWN"
    
    # Проверяем magic numbers из констант
    for format_name, magic_signatures in MAGIC_NUMBERS.items():
        for signature in magic_signatures:
            if data.startswith(signature):
                return format_name
    
    # Дополнительные проверки для сложных форматов
    # WEBP: RIFFxxxxWEBP
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    
    # TIFF: II* или MM*
    if len(data) >= 4 and data[:2] in [b"II", b"MM"] and data[2:4] in [b"*\x00", b"\x00*"]:
        return "TIFF"
    
    return "UNKNOWN"


def _check_type(value: Any, expected_type: str) -> bool:
    """
    Проверка типа значения.

    Args:
        value: Значение для проверки
        expected_type: Ожидаемый тип

    Returns:
        bool: True если тип соответствует
    """
    type_mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    python_type = type_mapping.get(expected_type)
    if python_type:
        return isinstance(value, python_type)

    return False
