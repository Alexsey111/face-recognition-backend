"""Тесты поддержки формата HEIC"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.utils.exceptions import ValidationError
from app.utils.file_utils import ImageFileHandler


class TestHEICSupport:
    """Тесты работы с HEIC форматом"""

    def create_test_jpeg(self, width: int = 640, height: int = 480) -> bytes:
        """Создание тестового JPEG изображения (симуляция HEIC контента)"""
        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    def test_detect_heic_format(self):
        """Тест определения HEIC формата по magic bytes"""
        # HEIC magic bytes: ftypheic
        heic_header = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00"
        result = ImageFileHandler.is_heic_format(heic_header + b"\x00" * 100)
        assert result is True

    def test_heic_mif1_variant(self):
        """Тест определения HEIF mif1 варианта (iPhone)"""
        # iPhone использует ftypmif1
        mif1_header = b"\x00\x00\x00\x20ftypmif1\x00\x00\x00\x00"
        result = ImageFileHandler.is_heic_format(mif1_header + b"\x00" * 100)
        assert result is True

    def test_heif_format_detection(self):
        """Тест определения HEIF формата"""
        heif_data = b"\x00\x00\x00\x18ftypheif\x00\x00\x00\x00" + b"\x00" * 100
        mime_type = ImageFileHandler.detect_mime_type(heif_data, "test.heif")
        assert mime_type == "image/heif"

    def test_heic_mime_type_detection(self):
        """Тест определения MIME типа HEIC"""
        heic_data = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00" + b"\x00" * 100
        mime_type = ImageFileHandler.detect_mime_type(heic_data, "test.heic")
        assert mime_type == "image/heic"

    def test_heic_extension_validation(self):
        """Тест валидации расширения HEIC"""
        assert ImageFileHandler.validate_file_extension("photo.heic") is True
        assert ImageFileHandler.validate_file_extension("photo.heif") is True
        assert ImageFileHandler.validate_file_extension("photo.HEIC") is True
        assert ImageFileHandler.validate_file_extension("photo.HEIF") is True

    def test_jpeg_extension_validation(self):
        """Тест валидации расширения JPEG"""
        assert ImageFileHandler.validate_file_extension("photo.jpg") is True
        assert ImageFileHandler.validate_file_extension("photo.jpeg") is True
        assert ImageFileHandler.validate_file_extension("photo.JPG") is True

    def test_png_extension_validation(self):
        """Тест валидации расширения PNG"""
        assert ImageFileHandler.validate_file_extension("photo.png") is True
        assert ImageFileHandler.validate_file_extension("photo.PNG") is True

    def test_webp_extension_validation(self):
        """Тест валидации WebP"""
        assert ImageFileHandler.validate_file_extension("photo.webp") is True
        assert ImageFileHandler.validate_file_extension("photo.WEBP") is True

    def test_unsupported_extension_validation(self):
        """Тест отклонения неподдерживаемых расширений"""
        assert ImageFileHandler.validate_file_extension("photo.gif") is False
        assert ImageFileHandler.validate_file_extension("photo.bmp") is False
        assert ImageFileHandler.validate_file_extension("photo.tiff") is False
        assert ImageFileHandler.validate_file_extension("photo.raw") is False

    def test_jpeg_to_jpeg_conversion(self):
        """Тест конвертации JPEG (симуляция HEIC→JPEG)"""
        test_image = self.create_test_jpeg(640, 480)

        # Конвертируем (для JPEG это просто пересохранение)
        result = ImageFileHandler.convert_heic_to_jpeg(test_image, quality=95)

        assert result is not None
        assert len(result) > 0
        assert result[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_load_jpeg_image(self):
        """Тест загрузки JPEG изображения"""
        test_image = self.create_test_jpeg(640, 480)

        # Загружаем через обработчик
        image = ImageFileHandler.load_image_from_bytes(test_image, "test.jpg")

        assert image is not None
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # BGR формат
        assert image.shape[2] == 3

    def test_image_info_from_jpeg(self):
        """Тест получения информации о JPEG изображении"""
        test_image = self.create_test_jpeg(800, 600)

        info = ImageFileHandler.get_image_info(test_image, "test.jpg")

        assert info["width"] == 800
        assert info["height"] == 600
        assert info["size_bytes"] > 0
        assert "mime_type" in info
        assert info["format"] == "JPEG"

    def test_rgba_to_rgb_conversion(self):
        """Тест конвертации RGBA → RGB (для прозрачных изображений)"""
        # Создаем изображение с альфа-каналом
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")  # PNG для сохранения альфы

        # Конвертируем (симуляция HEIC с альфой)
        result = ImageFileHandler.convert_heic_to_jpeg(buffer.getvalue())

        # Проверяем, что результат - валидный JPEG без альфа-канала
        result_img = Image.open(io.BytesIO(result))
        assert result_img.mode == "RGB"

    def test_p_mode_conversion(self):
        """Тест конвертации палитрового изображения (P)"""
        # Создаем палитровое изображение
        img = Image.new("P", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # Конвертируем
        result = ImageFileHandler.convert_heic_to_jpeg(buffer.getvalue())

        # Проверяем результат
        result_img = Image.open(io.BytesIO(result))
        assert result_img.mode == "RGB"

    def test_grayscale_conversion(self):
        """Тест конвертации grayscale изображения"""
        img = Image.new("L", (100, 100), color=128)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # Конвертируем
        result = ImageFileHandler.convert_heic_to_jpeg(buffer.getvalue())

        # Проверяем результат
        result_img = Image.open(io.BytesIO(result))
        assert result_img.mode == "RGB"

    def test_file_size_limit(self):
        """Тест проверки размера файла"""
        # Создаем изображение
        test_image = self.create_test_jpeg(100, 100)

        # Проверяем что размер в пределах лимита
        from app.utils.file_utils import MAX_FILE_SIZE

        assert len(test_image) < MAX_FILE_SIZE

    def test_invalid_image_data(self):
        """Тест обработки невалидных данных изображения"""
        invalid_data = b"invalid image data"

        with pytest.raises(ValidationError):
            ImageFileHandler.load_image_from_bytes(invalid_data, "test.jpg")

    def test_max_dimension_validation(self):
        """Тест проверки максимального разрешения"""
        # Создаем очень большое изображение
        img = Image.new("RGB", (5000, 5000), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        with pytest.raises(ValidationError) as exc_info:
            ImageFileHandler.load_image_from_bytes(buffer.getvalue(), "large.jpg")

        assert "превышает максимальное" in str(exc_info.value)

    def test_min_dimension_validation(self):
        """Тест проверки минимального разрешения"""
        # Создаем слишком маленькое изображение
        img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        # Проверяем что изображение загружается (предупреждение, не ошибка)
        image = ImageFileHandler.load_image_from_bytes(buffer.getvalue(), "small.jpg")
        assert image is not None

    def test_jpeg_magic_bytes_detection(self):
        """Тест определения JPEG по magic bytes"""
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100
        mime = ImageFileHandler.detect_mime_type(jpeg_data, None)
        assert mime == "image/jpeg"

    def test_png_magic_bytes_detection(self):
        """Тест определения PNG по magic bytes"""
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        mime = ImageFileHandler.detect_mime_type(png_data, None)
        assert mime == "image/png"

    def test_webp_magic_bytes_detection(self):
        """Тест определения WebP по magic bytes"""
        webp_data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 100
        mime = ImageFileHandler.detect_mime_type(webp_data, None)
        assert mime == "image/webp"

    def test_fallback_to_extension(self):
        """Тест fallback определения формата по расширению"""
        # JPEG данные без magic bytes, но с правильным расширением
        jpeg_data = b"\x00" * 100
        mime = ImageFileHandler.detect_mime_type(jpeg_data, "photo.jpg")
        assert mime == "image/jpeg"

    def test_unsupported_mime_type(self):
        """Тест ошибки при неподдерживаемом формате"""
        unsupported_data = b"\x00" * 100

        with pytest.raises(ValidationError):
            ImageFileHandler.detect_mime_type(unsupported_data, "photo.gif")

    @pytest.mark.parametrize(
        "extension,expected",
        [
            ("photo.heic", True),
            ("photo.HEIC", True),
            ("photo.heif", True),
            ("photo.HEIF", True),
            ("photo.jpg", True),
            ("photo.jpeg", True),
            ("photo.png", True),
            ("photo.webp", True),
            ("photo.gif", False),
            ("photo.bmp", False),
            ("photo.tiff", False),
            ("photo.raw", False),
            ("photo.unknown", False),
        ],
    )
    def test_supported_extensions_parametrized(self, extension, expected):
        """Тест поддерживаемых расширений (parameterized)"""
        result = ImageFileHandler.validate_file_extension(extension)
        assert result is expected


class TestHEICValidationService:
    """Тесты интеграции с ValidationService"""

    def test_validation_service_get_formats_info(self):
        """Тест получения информации о форматах"""
        from app.services.validation_service import ValidationService

        service = ValidationService()
        info = service.get_supported_formats_info()

        assert "supported_extensions" in info
        assert "max_file_size_mb" in info
        assert "formats" in info
        assert "HEIC" in info["formats"]
        assert "JPEG" in info["formats"]
        assert "PNG" in info["formats"]
        assert "WebP" in info["formats"]

    def test_validation_service_check_quality(self):
        """Тест проверки качества изображения"""
        from app.services.validation_service import ValidationService

        service = ValidationService()

        # Создаем тестовое изображение
        img = Image.new("RGB", (640, 480), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        from app.utils.file_utils import ImageFileHandler

        image = ImageFileHandler.load_image_from_bytes(buffer.getvalue(), "test.jpg")

        # Проверка качества не должна вызывать исключений
        service._check_image_quality(image)

    def test_validation_service_low_resolution(self):
        """Тест ошибки о низком разрешении"""
        import numpy as np

        from app.services.validation_service import ValidationService
        from app.utils.exceptions import ValidationError

        service = ValidationService()

        # Создаем изображение низкого разрешения
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        # Должно выбросить ValidationError (изменение в логике)
        with pytest.raises(ValidationError) as exc_info:
            service._check_image_quality(image)

        assert "Слишком низкое разрешение" in str(exc_info.value)
        assert "160x160" in str(exc_info.value)


class TestHEICFileUtils:
    """Тесты утилит работы с файлами"""

    def test_validate_image_file_valid(self):
        """Тест валидации валидного файла"""
        from app.utils.file_utils import validate_image_file

        # Создаем тестовое изображение
        img = Image.new("RGB", (640, 480), color=(0, 255, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        file_data = buffer.getvalue()

        is_valid, error = validate_image_file(file_data, "test.jpg")

        assert is_valid is True
        assert error is None

    def test_validate_image_file_invalid_extension(self):
        """Тест валидации файла с неподдерживаемым расширением"""
        from app.utils.file_utils import validate_image_file

        file_data = b"\x00" * 100

        is_valid, error = validate_image_file(file_data, "test.gif")

        assert is_valid is False
        assert "Неподдерживаемое расширение" in error

    def test_validate_image_file_too_large(self):
        """Тест валидации файла слишком большого размера"""
        from app.utils.file_utils import MAX_FILE_SIZE, validate_image_file

        # Создаем данные больше лимита
        file_data = b"\x00" * (MAX_FILE_SIZE + 1)

        is_valid, error = validate_image_file(file_data, "test.jpg")

        assert is_valid is False
        assert "превышает лимит" in error


# =============================================================================
# fixtures для интеграционных тестов
# =============================================================================


@pytest.fixture
def test_images_dir(tmp_path):
    """Создание временной директории с тестовыми изображениями"""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Создаем тестовые изображения
    for i in range(3):
        img = Image.new("RGB", (640, 480), color=(i * 80, 128, 255 - i * 80))
        (images_dir / f"person{i+1}_img1.jpg").write_bytes(_image_to_bytes(img, "JPEG"))
        (images_dir / f"person{i+1}_img2.jpg").write_bytes(_image_to_bytes(img, "JPEG"))

    return images_dir


def _image_to_bytes(img: Image.Image, format: str) -> bytes:
    """Вспомогательная функция для сохранения изображения в байты"""
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=95)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# smoke тесты
# =============================================================================


class TestHEICSmoke:
    """Smoke тесты для быстрой проверки"""

    def test_image_file_handler_import(self):
        """Проверка импорта ImageFileHandler"""
        from app.utils.file_utils import ImageFileHandler

        assert hasattr(ImageFileHandler, "load_image_from_bytes")
        assert hasattr(ImageFileHandler, "convert_heic_to_jpeg")
        assert hasattr(ImageFileHandler, "get_image_info")

    def test_validation_service_import(self):
        """Проверка импорта ValidationService"""
        from app.services.validation_service import ValidationService

        assert hasattr(ValidationService, "validate_uploaded_image")
        assert hasattr(ValidationService, "get_supported_formats_info")

    def test_supported_extensions_defined(self):
        """Проверка определения поддерживаемых расширений"""
        from app.utils.file_utils import SUPPORTED_EXTENSIONS

        assert isinstance(SUPPORTED_EXTENSIONS, list)
        assert len(SUPPORTED_EXTENSIONS) > 0
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".heic" in SUPPORTED_EXTENSIONS

    def test_heif_support_flag(self):
        """Проверка флага поддержки HEIF"""
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
            heif_supported = True
        except ImportError:
            heif_supported = False

        # pillow-heif должен быть установлен
        assert heif_supported is True
