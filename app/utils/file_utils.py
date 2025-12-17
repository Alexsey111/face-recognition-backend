"""
Утилиты для работы с файлами.
Валидация, конвертирование и обработка изображений.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image
from io import BytesIO
from datetime import datetime
from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FileUtils:
    """Утилиты для работы с файлами"""
    
    ALLOWED_FORMATS = {'jpg', 'jpeg', 'png', 'heic'}
    TARGET_FORMAT = 'jpg'
    MAX_FILE_SIZE_MB = 10
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Получение расширения файла"""
        return Path(filename).suffix.lower().lstrip('.')
    
    @staticmethod
    def is_valid_image_format(filename: str) -> bool:
        """Проверка валидности формата изображения"""
        ext = FileUtils.get_file_extension(filename)
        return ext in FileUtils.ALLOWED_FORMATS
    
    @staticmethod
    def get_file_size_mb(file_content: bytes) -> float:
        """Получение размера файла в МБ"""
        return len(file_content) / (1024 * 1024)
    
    @staticmethod
    def generate_file_key(user_id: str, filename: str) -> str:
        """
        Генерация ключа S3 объекта
        Формат: uploads/{user_id}/{timestamp}_{original_filename}
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = os.path.basename(filename)
        return f"uploads/{user_id}/{timestamp}_{safe_filename}"
    
    @staticmethod
    def generate_reference_key(user_id: str, version: int) -> str:
        """
        Генерация ключа S3 объекта для референсного изображения
        Формат: references/{user_id}/v{version}.jpg
        """
        return f"references/{user_id}/v{version}.jpg"
    
    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """Вычисление SHA256 хеша файла"""
        hash_obj = hashlib.sha256(file_content)
        return hash_obj.hexdigest()
    
    @staticmethod
    def convert_image_to_jpg(
        file_content: bytes,
        filename: str,
        quality: int = 85
    ) -> Tuple[bytes, str]:
        """
        Конвертирование изображения в формат JPG
        
        Args:
            file_content: Исходное изображение
            filename: Исходное имя файла
            quality: Качество JPG (1-100)
            
        Returns:
            Tuple[bytes, str]: (конвертированный_контент, новое_имя_файла)
        """
        try:
            # Открываем изображение
            img = Image.open(BytesIO(file_content))
            
            # Конвертируем RGBA в RGB если необходимо
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Сохраняем как JPG
            output = BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            converted_content = output.getvalue()
            
            # Генерируем новое имя файла
            base_name = Path(filename).stem
            new_filename = f"{base_name}.jpg"
            
            logger.info(
                f"Изображение конвертировано: {filename} -> {new_filename} "
                f"({len(file_content)} -> {len(converted_content)} байт)"
            )
            
            return converted_content, new_filename
            
        except Exception as e:
            logger.error(f"Ошибка конвертирования изображения: {e}")
            raise
    
    @staticmethod
    def get_image_dimensions(file_content: bytes) -> Tuple[int, int]:
        """Получение размеров изображения"""
        try:
            img = Image.open(BytesIO(file_content))
            return img.width, img.height
        except Exception as e:
            logger.error(f"Ошибка получения размеров изображения: {e}")
            raise
    
    @staticmethod
    def resize_image(
        file_content: bytes,
        max_width: int = 1024,
        max_height: int = 1024
    ) -> bytes:
        """
        Изменение размера изображения если превышает максимальные размеры
        
        Args:
            file_content: Контент изображения
            max_width: Максимальная ширина
            max_height: Максимальная высота
            
        Returns:
            bytes: Измененное изображение
        """
        try:
            img = Image.open(BytesIO(file_content))
            
            # Вычисляем новый размер сохраняя пропорции
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Сохраняем измененное изображение
            output = BytesIO()
            img.save(output, format='JPEG', quality=90, optimize=True)
            resized_content = output.getvalue()
            
            logger.info(
                f"Изображение изменено в размере: {img.width}x{img.height} "
                f"({len(file_content)} -> {len(resized_content)} байт)"
            )
            
            return resized_content
            
        except Exception as e:
            logger.error(f"Ошибка изменения размера изображения: {e}")
            raise
    
    @staticmethod
    def validate_image_quality(file_content: bytes, filename: str) -> Tuple[bool, str]:
        """
        Базовая валидация качества изображения
        
        Args:
            file_content: Контент изображения
            filename: Имя файла
            
        Returns:
            Tuple[bool, str]: (валидно, сообщение_ошибки)
        """
        try:
            # Проверка размера файла
            size_mb = FileUtils.get_file_size_mb(file_content)
            if size_mb > FileUtils.MAX_FILE_SIZE_MB:
                return False, f"Файл слишком большой: {size_mb:.1f}МБ (макс {FileUtils.MAX_FILE_SIZE_MB}МБ)"
            
            # Проверка валидности изображения
            try:
                img = Image.open(BytesIO(file_content))
                width, height = img.size
            except Exception as e:
                return False, f"Недействительный файл изображения: {str(e)}"
            
            # Проверка минимальных размеров для распознавания лиц
            if width < 50 or height < 50:
                return False, f"Изображение слишком маленькое: {width}x{height} (мин 50x50)"
            
            logger.info(f"Изображение прошло валидацию: {filename} ({width}x{height}, {size_mb:.1f}МБ)")
            return True, ""
            
        except Exception as e:
            logger.error(f"Ошибка валидации изображения: {e}")
            return False, f"Ошибка валидации: {str(e)}"


class ImageValidator:
    """Валидатор изображений для распознавания лиц"""
    
    # Минимальные размеры для детекции лиц
    MIN_WIDTH = 50
    MIN_HEIGHT = 50
    MAX_FILE_SIZE_MB = 10
    
    @staticmethod
    def validate_image(file_content: bytes, filename: str) -> Tuple[bool, str]:
        """
        Валидация изображения для распознавания лиц
        
        Args:
            file_content: Контент изображения
            filename: Имя файла
            
        Returns:
            Tuple[bool, str]: (валидно, сообщение_ошибки)
        """
        try:
            # Проверка формата
            if not FileUtils.is_valid_image_format(filename):
                return False, f"Недействительный формат изображения: {FileUtils.get_file_extension(filename)}"
            
            # Проверка размера файла
            size_mb = FileUtils.get_file_size_mb(file_content)
            if size_mb > ImageValidator.MAX_FILE_SIZE_MB:
                return False, f"Файл слишком большой: {size_mb:.1f}МБ (макс {ImageValidator.MAX_FILE_SIZE_MB}МБ)"
            
            # Проверка валидности изображения
            try:
                img = Image.open(BytesIO(file_content))
                width, height = img.size
            except Exception as e:
                return False, f"Недействительный файл изображения: {str(e)}"
            
            # Проверка размеров
            if width < ImageValidator.MIN_WIDTH or height < ImageValidator.MIN_HEIGHT:
                return False, f"Изображение слишком маленькое: {width}x{height} (мин {ImageValidator.MIN_WIDTH}x{ImageValidator.MIN_HEIGHT})"
            
            logger.info(f"Изображение прошло валидацию: {filename} ({width}x{height}, {size_mb:.1f}МБ)")
            return True, ""
            
        except Exception as e:
            logger.error(f"Ошибка валидации изображения: {e}")
            return False, f"Ошибка валидации: {str(e)}"
    
    @staticmethod
    def get_image_info(file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Получение информации об изображении
        
        Args:
            file_content: Контент изображения
            filename: Имя файла
            
        Returns:
            Dict[str, Any]: Информация об изображении
        """
        try:
            img = Image.open(BytesIO(file_content))
            width, height = img.size
            size_mb = FileUtils.get_file_size_mb(file_content)
            
            return {
                "filename": filename,
                "width": width,
                "height": height,
                "size_mb": size_mb,
                "format": img.format,
                "mode": img.mode,
                "file_hash": FileUtils.calculate_file_hash(file_content),
                "is_valid": (
                    width >= ImageValidator.MIN_WIDTH and 
                    height >= ImageValidator.MIN_HEIGHT and
                    size_mb <= ImageValidator.MAX_FILE_SIZE_MB
                )
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения информации об изображении: {e}")
            return {
                "filename": filename,
                "error": str(e),
                "is_valid": False
            }