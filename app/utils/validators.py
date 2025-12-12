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

from .constants import IMAGE_FORMATS, FILE_LIMITS
from .exceptions import ValidationError

# Регулярные выражения для валидации
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")
PASSWORD_REGEX = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
)


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


# Вспомогательные функции


def _detect_image_format(image_data: str) -> str:
    """
    Определение формата изображения по данным.

    Args:
        image_data: Данные изображения

    Returns:
        str: Формат изображения
    """
    if image_data.startswith("data:image/"):
        # Data URL формат
        mime_type = image_data.split(";")[0].split("/")[1].upper()
        if mime_type == "JPEG":
            return "JPEG"
        elif mime_type == "PNG":
            return "PNG"
        elif mime_type == "WEBP":
            return "WEBP"

    # По расширению файла
    if "." in image_data:
        extension = image_data.split(".")[-1].upper()
        if extension in ["JPG", "JPEG"]:
            return "JPEG"
        elif extension == "PNG":
            return "PNG"
        elif extension == "WEBP":
            return "WEBP"

    # По умолчанию
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
