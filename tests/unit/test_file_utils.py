"""
Тесты для app/utils/file_utils.py
Валидация, конвертация, ресайз и метаданные изображений.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from PIL import Image, ImageFile
import hashlib

from app.utils.file_utils import FileUtils, ImageValidator
from app.utils.exceptions import ValidationError, ProcessingError


class TestFileUtils:
    """Тесты для FileUtils"""

    # ======================================================
    # Basic helpers
    # ======================================================

    def test_get_file_extension(self):
        """Тест получения расширения файла"""
        assert FileUtils.get_file_extension("image.jpg") == "jpg"
        assert FileUtils.get_file_extension("photo.JPEG") == "jpeg"
        assert FileUtils.get_file_extension("picture.png") == "png"
        assert FileUtils.get_file_extension("file.heic") == "heic"
        assert FileUtils.get_file_extension("noextension") == ""
        assert FileUtils.get_file_extension("path/to/image.jpg") == "jpg"

    def test_is_valid_image_format(self):
        """Тест проверки валидного формата изображения"""
        assert FileUtils.is_valid_image_format("test.jpg") is True
        assert FileUtils.is_valid_image_format("test.jpeg") is True
        assert FileUtils.is_valid_image_format("test.png") is True
        assert FileUtils.is_valid_image_format("test.heic") is True
        assert FileUtils.is_valid_image_format("test.gif") is False
        assert FileUtils.is_valid_image_format("test.bmp") is False
        assert FileUtils.is_valid_image_format("test.pdf") is False

    def test_get_file_size_mb(self):
        """Тест получения размера файла в МБ"""
        # 1 МБ данных
        content = b"x" * (1024 * 1024)
        assert FileUtils.get_file_size_mb(content) == 1.0

        # 500 КБ данных
        content = b"x" * (512 * 1024)
        assert FileUtils.get_file_size_mb(content) == 0.5

        # 0 байт
        assert FileUtils.get_file_size_mb(b"") == 0.0

    def test_calculate_file_hash(self):
        """Тест вычисления хеша файла"""
        content = b"test content"
        expected_hash = hashlib.sha256(content).hexdigest()

        result = FileUtils.calculate_file_hash(content)
        assert result == expected_hash
        assert len(result) == 64  # SHA256 hex digest length

    def test_calculate_file_hash_different_content(self):
        """Тест что разный контент дает разный хеш"""
        content1 = b"content 1"
        content2 = b"content 2"

        hash1 = FileUtils.calculate_file_hash(content1)
        hash2 = FileUtils.calculate_file_hash(content2)

        assert hash1 != hash2

    # ======================================================
    # Keys
    # ======================================================

    def test_generate_upload_key(self):
        """Тест генерации ключа загрузки"""
        user_id = "user123"
        filename = "photo.jpg"

        key = FileUtils.generate_upload_key(user_id, filename)

        assert key.startswith("uploads/user123/")
        assert key.endswith(".jpg")
        assert "photo" in key

    def test_generate_upload_key_special_chars(self):
        """Тест генерации ключа с спецсимволами в имени"""
        user_id = "user456"
        filename = "my photo (1).jpg"

        key = FileUtils.generate_upload_key(user_id, filename)

        assert key.startswith("uploads/user456/")
        assert key.endswith(".jpg")

    def test_generate_file_key_alias(self):
        """Тест что generate_file_key это алиас для generate_upload_key"""
        user_id = "user789"
        filename = "test.jpg"

        upload_key = FileUtils.generate_upload_key(user_id, filename)
        file_key = FileUtils.generate_file_key(user_id, filename)

        assert upload_key == file_key

    def test_generate_reference_key(self):
        """Тест генерации ключа референса"""
        user_id = "user123"
        version = 5

        key = FileUtils.generate_reference_key(user_id, version)

        assert key == "references/user123/v5.jpg"


class TestFileUtilsImageOperations:
    """Тесты для операций с изображениями в FileUtils"""

    @pytest.fixture
    def sample_image_content(self):
        """Создание тестового изображения в памяти"""
        img = Image.new("RGB", (200, 200), color="red")
        output = BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

    def test_open_image_valid(self, sample_image_content):
        """Тест открытия валидного изображения"""
        img = FileUtils.open_image(sample_image_content)

        assert isinstance(img, Image.Image)
        assert img.size == (200, 200)

    def test_open_image_invalid(self):
        """Тест открытия невалидного изображения"""
        invalid_content = b"not an image"

        with pytest.raises(ValidationError):
            FileUtils.open_image(invalid_content)

    def test_open_image_corrupted(self):
        """Тест открытия поврежденного изображения"""
        # Создаем файл, который выглядит как изображение, но поврежден
        corrupted_content = b"\xff\xd8\xff\xe0" + b"x" * 100

        with pytest.raises(ValidationError):
            FileUtils.open_image(corrupted_content)

    def test_get_image_dimensions(self, sample_image_content):
        """Тест получения размеров изображения"""
        width, height = FileUtils.get_image_dimensions(sample_image_content)

        assert width == 200
        assert height == 200

    def test_ensure_rgb_rgba(self):
        """Тест конвертации RGBA в RGB"""
        img_rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img_rgb = FileUtils.ensure_rgb(img_rgba)

        assert img_rgb.mode == "RGB"
        assert img_rgb.size == (100, 100)

    def test_ensure_rgb_la(self):
        """Тест конвертации LA в RGB"""
        img_la = Image.new("LA", (100, 100), color=(128, 255))
        img_rgb = FileUtils.ensure_rgb(img_la)

        assert img_rgb.mode == "RGB"

    def test_ensure_rgb_palette(self):
        """Тест конвертации палитрового изображения в RGB"""
        img_p = Image.new("P", (100, 100))
        img_rgb = FileUtils.ensure_rgb(img_p)

        assert img_rgb.mode == "RGB"

    def test_ensure_rgb_already_rgb(self):
        """Тест что RGB изображение не меняется"""
        img_rgb = Image.new("RGB", (100, 100), color="blue")
        result = FileUtils.ensure_rgb(img_rgb)

        assert result is img_rgb  # Should return same object


class TestImageValidator:
    """Тесты для ImageValidator"""

    @pytest.fixture
    def valid_image_content(self):
        """Создание валидного изображения"""
        img = Image.new("RGB", (200, 200), color="white")
        output = BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

    @pytest.fixture
    def small_image_content(self):
        """Создание слишком маленького изображения"""
        img = Image.new("RGB", (30, 30), color="black")
        output = BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

    @pytest.fixture
    def large_image_content(self):
        """Создание слишком большого изображения (больше 10 МБ)"""
        # Создаем изображение, которое будет занимать больше 10 МБ
        # 4000x4000 пикселей RGB = ~48 МБ в несжатом виде
        img = Image.new("RGB", (4000, 4000), color="blue")
        output = BytesIO()
        img.save(output, format="JPEG", quality=95)
        return output.getvalue()

    # ======================================================
    # Validation tests
    # ======================================================

    @patch('app.utils.file_utils.logger')
    def test_validate_valid_image(self, mock_logger, valid_image_content):
        """Тест валидации валидного изображения"""
        # Не должно вызывать исключение
        ImageValidator.validate(valid_image_content, "test.jpg")

    def test_validate_invalid_format(self):
        """Тест валидации с неподдерживаемым форматом"""
        content = b"fake image data"

        with pytest.raises(ValidationError) as exc_info:
            ImageValidator.validate(content, "test.gif")

        assert "Unsupported image format" in str(exc_info.value)

    def test_validate_file_too_large(self):
        """Тест валидации слишком большого файла"""
        # Примечание: может не хватить памяти для создания 10МБ+ изображения в тесте
        # Проверяем логику с моком
        with patch("app.utils.file_utils.FileUtils.get_file_size_mb") as mock_size:
            mock_size.return_value = 15.0  # 15 МБ

            with pytest.raises(ValidationError) as exc_info:
                ImageValidator.validate(b"x" * 100, "test.jpg")

            assert "File size exceeds limit" in str(exc_info.value)

    def test_validate_too_small_resolution(self, small_image_content):
        """Тест валидации изображения с низким разрешением"""
        with pytest.raises(ValidationError) as exc_info:
            ImageValidator.validate(small_image_content, "test.jpg")

        assert "Image resolution too small" in str(exc_info.value)

    @patch('app.utils.file_utils.logger')
    def test_validate_minimum_dimensions(self, mock_logger):
        """Тест что изображение ровно 50x50 проходит валидацию"""
        img = Image.new("RGB", (50, 50), color="white")
        output = BytesIO()
        img.save(output, format="JPEG")
        content = output.getvalue()

        # Должно пройти без исключения
        ImageValidator.validate(content, "test.jpg")

    def test_validate_49x49_fails(self):
        """Тест что изображение 49x49 не проходит валидацию"""
        img = Image.new("RGB", (49, 49), color="white")
        output = BytesIO()
        img.save(output, format="JPEG")
        content = output.getvalue()

        with pytest.raises(ValidationError):
            ImageValidator.validate(content, "test.jpg")

    def test_validate_heic_format(self):
        """Тест валидации HEIC формата (поддерживается)"""
        # HEIC файлы требуют pillow-heif
        # Проверяем что формат считается валидным
        assert FileUtils.is_valid_image_format("photo.heic") is True

    # ======================================================
    # validate_image API
    # ======================================================

    @patch('app.utils.file_utils.logger')
    def test_validate_image_valid(self, mock_logger, valid_image_content):
        """Тест validate_image с валидным изображением"""
        is_valid, error_msg = ImageValidator.validate_image(
            valid_image_content, "test.jpg"
        )

        assert is_valid is True
        assert error_msg == ""

    def test_validate_image_invalid_format(self):
        """Тест validate_image с невалидным форматом"""
        is_valid, error_msg = ImageValidator.validate_image(b"data", "test.bmp")

        assert is_valid is False
        assert "Unsupported image format" in error_msg

    def test_validate_image_invalid_content(self):
        """Тест validate_image с битым содержимым"""
        is_valid, error_msg = ImageValidator.validate_image(
            b"not an image", "test.jpg"
        )

        assert is_valid is False
        assert len(error_msg) > 0

    # ======================================================
    # get_info
    # ======================================================

    def test_get_info_valid_image(self, valid_image_content):
        """Тест получения информации о валидном изображении"""
        info = ImageValidator.get_info(valid_image_content, "test.jpg")

        assert info["filename"] == "test.jpg"
        assert info["width"] == 200
        assert info["height"] == 200
        assert "size_mb" in info
        assert info["format"] == "JPEG"
        assert info["mode"] == "RGB"
        assert "file_hash" in info

    def test_get_info_invalid_image(self):
        """Тест получения информации о невалидном изображении"""
        with pytest.raises(ProcessingError):
            ImageValidator.get_info(b"not an image", "test.jpg")

    def test_get_info_file_hash(self, valid_image_content):
        """Тест что file_hash корректно вычисляется"""
        info = ImageValidator.get_info(valid_image_content, "test.jpg")

        # Хеш должен совпадать с прямым вычислением
        expected_hash = FileUtils.calculate_file_hash(valid_image_content)
        assert info["file_hash"] == expected_hash


class TestFileUtilsEdgeCases:
    """Тесты для граничных случаев FileUtils"""

    def test_empty_file(self):
        """Тест обработки пустого файла"""
        content = b""

        # Проверяем что функции работают с пустым файлом
        assert FileUtils.get_file_size_mb(content) == 0.0
        assert FileUtils.calculate_file_hash(content) == hashlib.sha256(b"").hexdigest()

    def test_unicode_filename(self):
        """Тест с unicode символами в имени файла"""
        filename = "фото_на_русском.jpg"
        ext = FileUtils.get_file_extension(filename)
        assert ext == "jpg"

    def test_multiple_dots_filename(self):
        """Тест с несколькими точками в имени файла"""
        filename = "my.file.name.jpg"
        ext = FileUtils.get_file_extension(filename)
        assert ext == "jpg"

    def test_leading_dot(self):
        """Тест с точкой в начале расширения"""
        filename = ".hidden.png"
        ext = FileUtils.get_file_extension(filename)
        assert ext == "png"

    def test_very_long_filename(self):
        """Тест с очень длинным именем файла"""
        filename = "a" * 300 + ".jpg"
        ext = FileUtils.get_file_extension(filename)
        assert ext == "jpg"

    def test_uppercase_extension(self):
        """Тест с верхним регистром расширения"""
        filename = "image.JPG"
        ext = FileUtils.get_file_extension(filename)
        is_valid = FileUtils.is_valid_image_format(filename)

        assert ext == "jpg"
        assert is_valid is True