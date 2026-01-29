"""
Утилиты для работы с файлами изображений

Поддерживаемые форматы: JPG, PNG, HEIC, HEIF, WebP
"""

import io
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pillow_heif
from PIL import Image

from app.utils.exceptions import ValidationError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Регистрируем HEIF плагин для Pillow
pillow_heif.register_heif_opener()

# Поддерживаемые форматы
SUPPORTED_IMAGE_FORMATS = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/heic": [".heic"],
    "image/heif": [".heif"],
    "image/webp": [".webp"],
}

SUPPORTED_EXTENSIONS = [
    ext for exts in SUPPORTED_IMAGE_FORMATS.values() for ext in exts
]

# Максимальный размер файла (в байтах)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 4096  # Максимальное разрешение по одной стороне


class ImageFileHandler:
    """Обработчик файлов изображений с поддержкой HEIC"""

    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """
        Проверка расширения файла

        Args:
            filename: Имя файла

        Returns:
            True если расширение поддерживается
        """
        ext = Path(filename).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    @staticmethod
    def validate_file_size(file_data: bytes) -> bool:
        """
        Проверка размера файла

        Args:
            file_data: Байты файла

        Returns:
            True если размер в пределах лимита
        """
        size = len(file_data)
        if size > MAX_FILE_SIZE:
            raise ValidationError(
                f"Размер файла {size / 1024 / 1024:.2f} MB превышает лимит {MAX_FILE_SIZE / 1024 / 1024:.2f} MB"
            )
        return True

    @staticmethod
    def detect_mime_type(file_data: bytes, filename: Optional[str] = None) -> str:
        """
        Определение MIME-типа файла

        Args:
            file_data: Байты файла
            filename: Имя файла (опционально)

        Returns:
            MIME-тип
        """
        # Проверяем по magic bytes
        if file_data[:4] == b"\xff\xd8\xff\xe0" or file_data[:4] == b"\xff\xd8\xff\xe1":
            return "image/jpeg"
        elif file_data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif file_data[4:12] == b"ftypheic" or file_data[4:12] == b"ftypmif1":
            return "image/heic"
        elif file_data[4:12] == b"ftypheif":
            return "image/heif"
        elif file_data[:4] == b"RIFF" and file_data[8:12] == b"WEBP":
            return "image/webp"

        # Fallback на расширение файла
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type in SUPPORTED_IMAGE_FORMATS:
                return mime_type

        raise ValidationError("Неподдерживаемый формат файла")

    @staticmethod
    def is_heic_format(file_data: bytes, filename: Optional[str] = None) -> bool:
        """
        Проверка, является ли файл HEIC/HEIF форматом

        Args:
            file_data: Байты файла
            filename: Имя файла

        Returns:
            True если это HEIC/HEIF
        """
        mime_type = ImageFileHandler.detect_mime_type(file_data, filename)
        return mime_type in ["image/heic", "image/heif"]

    @staticmethod
    def convert_heic_to_jpeg(file_data: bytes, quality: int = 95) -> bytes:
        """
        Конвертация HEIC/HEIF в JPEG

        Args:
            file_data: Байты HEIC файла
            quality: Качество JPEG (1-100)

        Returns:
            Байты JPEG файла
        """
        try:
            logger.info(f"Конвертация HEIC в JPEG (size: {len(file_data)} bytes)")

            # Открываем HEIC через Pillow с поддержкой HEIF
            image = Image.open(io.BytesIO(file_data))

            # Конвертируем в RGB (HEIC может содержать RGBA)
            if image.mode in ("RGBA", "LA", "P"):
                # Создаем белый фон для прозрачности
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode == "P":
                    image = image.convert("RGBA")
                background.paste(
                    image, mask=image.split()[-1] if image.mode == "RGBA" else None
                )
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Сохраняем в JPEG
            jpeg_buffer = io.BytesIO()
            image.save(jpeg_buffer, format="JPEG", quality=quality, optimize=True)
            jpeg_data = jpeg_buffer.getvalue()

            logger.info(f"Конвертация завершена (output size: {len(jpeg_data)} bytes)")
            return jpeg_data

        except Exception as e:
            logger.error(f"Ошибка конвертации HEIC: {e}")
            raise ValidationError(f"Не удалось конвертировать HEIC файл: {str(e)}")

    @staticmethod
    def load_image_from_bytes(
        file_data: bytes, filename: Optional[str] = None
    ) -> np.ndarray:
        """
        Загрузка изображения из байтов в формат OpenCV (BGR)
        Автоматически конвертирует HEIC в JPEG

        Args:
            file_data: Байты изображения
            filename: Имя файла (для определения типа)

        Returns:
            Массив numpy (BGR формат для OpenCV)
        """
        try:
            # Валидация размера
            ImageFileHandler.validate_file_size(file_data)

            # Определяем тип файла
            mime_type = ImageFileHandler.detect_mime_type(file_data, filename)
            logger.info(
                f"Загрузка изображения: {filename or 'unknown'}, type: {mime_type}"
            )

            # Конвертируем HEIC в JPEG если необходимо
            if mime_type in ["image/heic", "image/heif"]:
                logger.info("Обнаружен HEIC формат, выполняется конвертация...")
                file_data = ImageFileHandler.convert_heic_to_jpeg(file_data)

            # Декодируем в numpy array
            nparr = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValidationError("Не удалось декодировать изображение")

            # Проверяем размеры
            height, width = image.shape[:2]
            if max(height, width) > MAX_IMAGE_DIMENSION:
                raise ValidationError(
                    f"Разрешение изображения {width}x{height} превышает максимальное {MAX_IMAGE_DIMENSION}px"
                )

            logger.info(
                f"Изображение загружено успешно: {width}x{height}, channels: {image.shape[2]}"
            )
            return image

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения: {e}")
            raise ValidationError(f"Не удалось загрузить изображение: {str(e)}")

    @staticmethod
    def save_image_to_bytes(
        image: np.ndarray, format: str = "JPEG", quality: int = 95
    ) -> bytes:
        """
        Сохранение OpenCV изображения в байты

        Args:
            image: Массив numpy (BGR формат)
            format: Формат вывода ('JPEG', 'PNG')
            quality: Качество для JPEG (1-100)

        Returns:
            Байты изображения
        """
        try:
            if format.upper() == "JPEG":
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                ext = ".jpg"
            elif format.upper() == "PNG":
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                ext = ".png"
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")

            success, encoded = cv2.imencode(ext, image, encode_param)
            if not success:
                raise Exception("Ошибка кодирования изображения")

            return encoded.tobytes()

        except Exception as e:
            logger.error(f"Ошибка сохранения изображения: {e}")
            raise ValidationError(f"Не удалось сохранить изображение: {str(e)}")

    @staticmethod
    def get_image_info(file_data: bytes, filename: Optional[str] = None) -> dict:
        """
        Получение информации об изображении без полной загрузки

        Args:
            file_data: Байты файла
            filename: Имя файла

        Returns:
            Словарь с информацией
        """
        try:
            mime_type = ImageFileHandler.detect_mime_type(file_data, filename)

            # Для HEIC используем Pillow
            if mime_type in ["image/heic", "image/heif"]:
                image = Image.open(io.BytesIO(file_data))
                width, height = image.size
                mode = image.mode
            else:
                # Для остальных форматов используем OpenCV
                nparr = np.frombuffer(file_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                height, width = image.shape[:2]
                mode = "BGR" if len(image.shape) == 3 else "Gray"

            return {
                "width": width,
                "height": height,
                "mode": mode,
                "mime_type": mime_type,
                "size_bytes": len(file_data),
                "size_mb": len(file_data) / 1024 / 1024,
                "format": mime_type.split("/")[-1].upper(),
            }

        except Exception as e:
            logger.error(f"Ошибка получения информации об изображении: {e}")
            raise ValidationError(
                f"Не удалось прочитать информацию об изображении: {str(e)}"
            )


# Экспортируем основные функции для обратной совместимости
def load_image(file_data: bytes, filename: Optional[str] = None) -> np.ndarray:
    """Загрузка изображения (обертка для ImageFileHandler)"""
    return ImageFileHandler.load_image_from_bytes(file_data, filename)


def save_image(image: np.ndarray, format: str = "JPEG", quality: int = 95) -> bytes:
    """Сохранение изображения (обертка для ImageFileHandler)"""
    return ImageFileHandler.save_image_to_bytes(image, format, quality)


def validate_image_file(file_data: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Валидация файла изображения

    Returns:
        (is_valid, error_message)
    """
    try:
        # Проверка расширения
        if not ImageFileHandler.validate_file_extension(filename):
            return (
                False,
                f"Неподдерживаемое расширение файла. Поддерживаются: {', '.join(SUPPORTED_EXTENSIONS)}",
            )

        # Проверка размера
        ImageFileHandler.validate_file_size(file_data)

        # Проверка формата
        ImageFileHandler.detect_mime_type(file_data, filename)

        # Пробуем загрузить
        ImageFileHandler.load_image_from_bytes(file_data, filename)

        return True, None

    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Ошибка валидации: {str(e)}"


# ============================================================================
# Aliases for backward compatibility (используются в app/utils/__init__.py)
# ============================================================================


class FileUtils:
    """Alias for ImageFileHandler - for backward compatibility"""

    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        return ImageFileHandler.validate_file_extension(filename)

    @staticmethod
    def validate_file_size(file_data: bytes) -> bool:
        return ImageFileHandler.validate_file_size(file_data)

    @staticmethod
    def detect_mime_type(file_data: bytes, filename: Optional[str] = None) -> str:
        return ImageFileHandler.detect_mime_type(file_data, filename)

    @staticmethod
    def is_heic_format(file_data: bytes, filename: Optional[str] = None) -> bool:
        return ImageFileHandler.is_heic_format(file_data, filename)

    @staticmethod
    def convert_heic_to_jpeg(file_data: bytes, quality: int = 95) -> bytes:
        return ImageFileHandler.convert_heic_to_jpeg(file_data, quality)

    @staticmethod
    def load_image_from_bytes(
        file_data: bytes, filename: Optional[str] = None
    ) -> np.ndarray:
        return ImageFileHandler.load_image_from_bytes(file_data, filename)

    @staticmethod
    def save_image_to_bytes(
        image: np.ndarray, format: str = "JPEG", quality: int = 95
    ) -> bytes:
        return ImageFileHandler.save_image_to_bytes(image, format, quality)

    @staticmethod
    def get_image_info(file_data: bytes, filename: Optional[str] = None) -> dict:
        return ImageFileHandler.get_image_info(file_data, filename)


class ImageValidator:
    """Alias for validation functions - for backward compatibility"""

    @staticmethod
    def validate_image(file_data: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        return validate_image_file(file_data, filename)

    @staticmethod
    def check_image_quality(image: np.ndarray) -> bool:
        """Check image quality, return True if acceptable"""
        # Basic quality checks are already done in load_image_from_bytes
        return True

    @staticmethod
    def get_supported_formats() -> dict:
        """Get supported formats information"""
        return {
            "supported_extensions": SUPPORTED_EXTENSIONS,
            "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
            "max_dimension": MAX_IMAGE_DIMENSION,
        }
