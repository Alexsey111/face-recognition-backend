"""
Сервис валидации данных.
Валидация изображений, безопасность и проверка форматов.
"""

import base64
import hashlib
import io
import re
from typing import Optional, Tuple, List, Dict, Any

import cv2
import httpx
import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    # pillow-heif is optional; HEIC images may fail to open without it
    pass

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class ValidationResult:
    """
    Результат валидации изображения.
    """
    def __init__(
        self,
        is_valid: bool,
        image_data: Optional[bytes] = None,
        image_format: Optional[str] = None,
        dimensions: Optional[Dict[str, int]] = None,
        quality_score: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        self.is_valid = is_valid
        self.image_data = image_data
        self.image_format = image_format
        self.dimensions = dimensions
        self.quality_score = quality_score
        self.error_message = error_message


class ValidationService:
    """
    Сервис для валидации данных и изображений.
    """
    
    def __init__(self):
        self.max_file_size = settings.MAX_UPLOAD_SIZE
        self.allowed_formats = settings.allowed_image_formats_list
        self.min_width = settings.MIN_IMAGE_WIDTH
        self.min_height = settings.MIN_IMAGE_HEIGHT
        self.max_width = settings.MAX_IMAGE_WIDTH
        self.max_height = settings.MAX_IMAGE_HEIGHT
        self._http_client = httpx.AsyncClient(timeout=10.0)
    
    async def validate_image(
        self,
        image_data: str,
        max_size: Optional[int] = None,
        allowed_formats: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Валидация изображения.
        
        Args:
            image_data: Данные изображения (base64, data URL или URL)
            max_size: Максимальный размер файла
            allowed_formats: Разрешенные форматы изображений
            
        Returns:
            ValidationResult: Результат валидации
        """
        try:
            # Используем настройки по умолчанию
            max_size = max_size or self.max_file_size
            if allowed_formats:
                allowed_formats = [fmt.strip().upper() for fmt in allowed_formats]
            else:
                allowed_formats = self.allowed_formats
            
            logger.info("Starting image validation")
            
            # Декодирование изображения
            decoded_data, format_type = await self._decode_image_data(image_data, max_size)
            
            if not decoded_data:
                return ValidationResult(
                    is_valid=False,
                    error_message="Failed to decode image data"
                )
            
            # Проверка размера файла
            if len(decoded_data) > max_size:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Image size {len(decoded_data)} exceeds maximum allowed size {max_size}"
                )
            
            # Определение формата изображения
            image_format = self._detect_image_format(decoded_data)
            if image_format not in allowed_formats:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Image format {image_format} not allowed. Allowed: {allowed_formats}"
                )
            
            # Открытие и проверка изображения
            try:
                with Image.open(io.BytesIO(decoded_data)) as img:
                    # Проверка размеров
                    width, height = img.size
                    if width < self.min_width or height < self.min_height:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Image dimensions {width}x{height} too small. Minimum: {self.min_width}x{self.min_height}"
                        )
                    
                    if width > self.max_width or height > self.max_height:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Image dimensions {width}x{height} too large. Maximum: {self.max_width}x{self.max_height}"
                        )
                    
                    # Проверка качества изображения
                    quality_score = await self._assess_image_quality(decoded_data, img)
                    
                    # Проверка на наличие лица (базовая)
                    face_detected = await self._detect_face_basic(decoded_data)
                    
                    if not face_detected:
                        return ValidationResult(
                            is_valid=False,
                            error_message="No face detected in image"
                        )
                    
                    logger.info(f"Image validation successful: {image_format}, {width}x{height}, quality: {quality_score:.3f}")
                    
                    return ValidationResult(
                        is_valid=True,
                        image_data=decoded_data,
                        image_format=image_format,
                        dimensions={"width": width, "height": height},
                        quality_score=quality_score
                    )
                    
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Failed to process image: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"Image validation error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
        finally:
            # nothing to cleanup here, placeholder to satisfy linter rule
            ...
    
    async def _decode_image_data(self, image_data: str, max_size: int) -> Tuple[Optional[bytes], str]:
        """
        Декодирование данных изображения.
        
        Args:
            image_data: Строка с данными изображения
            
        Returns:
            Tuple[Optional[bytes], str]: Декодированные данные и тип источника
        """
        try:
            # Data URL формат
            if image_data.startswith("data:image/"):
                # Извлекаем base64 часть
                if "," in image_data:
                    header, data = image_data.split(",", 1)
                    try:
                        decoded_data = base64.b64decode(data)
                        return decoded_data, "data_url"
                    except Exception:
                        pass
            
            # HTTP/HTTPS URL
            elif image_data.startswith(("http://", "https://")):
                fetched = await self._fetch_image_from_url(image_data, max_size)
                return fetched, "url"
            
            # Попытка декодировать как чистый base64
            else:
                try:
                    # Проверяем, что это base64
                    decoded_data = base64.b64decode(image_data)
                    return decoded_data, "base64"
                except Exception:
                    pass
            
            return None, "unknown"
            
        except Exception as e:
            logger.error(f"Error decoding image data: {str(e)}")
            return None, "unknown"
    
    def _detect_image_format(self, image_data: bytes) -> str:
        """
        Определение формата изображения.
        
        Args:
            image_data: Двоичные данные изображения
            
        Returns:
            str: Формат изображения
        """
        try:
            # Определяем формат по заголовкам файлов (magic bytes)
            if image_data.startswith(b"\xff\xd8\xff"):
                return "JPEG"
            elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
                return "PNG"
            elif image_data.startswith(b"RIFF") and b"WEBP" in image_data[:12]:
                return "WEBP"
            elif image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
                return "GIF"
            elif image_data.startswith(b"BM"):
                return "BMP"
            elif image_data.startswith(b"\x00\x00\x01\x00"):
                return "ICO"
            elif image_data[:12].lower().find(b"ftypheic") != -1 or image_data[:12].lower().find(b"ftypheif") != -1:
                return "HEIC"
            elif image_data[:12].lower().find(b"ftyphevc") != -1:
                return "HEIF"
            else:
                return "UNKNOWN"
        except Exception:
            return "UNKNOWN"
    
    async def _fetch_image_from_url(self, url: str, max_size: int) -> Optional[bytes]:
        """
        Загрузка изображения по URL с ограничением размера.
        """
        try:
            async with self._http_client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch image from URL {url}: status {resp.status_code}")
                    return None

                content = bytearray()
                async for chunk in resp.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > max_size:
                        logger.warning(f"Image from URL {url} exceeds max size {max_size}")
                        return None
                return bytes(content)
        except Exception as e:
            logger.error(f"Error fetching image from URL {url}: {e}")
            return None
    
    async def _assess_image_quality(self, image_data: bytes, img: Image.Image) -> float:
        """
        Оценка качества изображения.
        
        Args:
            image_data: Двоичные данные изображения
            img: PIL Image объект
            
        Returns:
            float: Оценка качества от 0 до 1
        """
        try:
            quality_score = 0.0
            
            # 1. Анализ размера файла (больше файл = лучше качество, до разумных пределов)
            file_size_score = min(len(image_data) / (1024 * 1024), 2.0) / 2.0  # Нормализация до 0-1
            quality_score += file_size_score * 0.2
            
            # 2. Анализ резкости (используем Laplacian variance)
            img_array = np.array(img.convert("L"))  # Конвертируем в grayscale
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)  # Нормализация
            quality_score += sharpness_score * 0.3
            
            # 3. Анализ яркости и контрастности
            mean_brightness = np.mean(img_array)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128  # Оптимальная яркость ~128
            quality_score += max(0, brightness_score) * 0.2

            # 3.1 Тени/пересвет: доля пикселей в очень тёмном/очень светлом диапазоне
            dark_ratio = np.mean(img_array < 30)
            bright_ratio = np.mean(img_array > 225)
            shadow_highlight_penalty = max(0, (dark_ratio + bright_ratio) - 0.2)  # штраф если >20% экстремумов
            quality_score -= shadow_highlight_penalty * 0.2
            
            # 4. Анализ цветового баланса
            if img.mode == "RGB":
                r_mean = np.mean(img_array)
                g_mean = np.mean(img_array)
                b_mean = np.mean(img_array)
                color_balance = 1.0 - (abs(r_mean - g_mean) + abs(g_mean - b_mean)) / 255
                quality_score += max(0, color_balance) * 0.15
            
            # 5. Анализ шума (через стандартное отклонение)
            noise_level = np.std(img_array)
            noise_score = max(0, 1.0 - noise_level / 64)  # Меньше шума = лучше
            quality_score += noise_score * 0.15
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error assessing image quality: {str(e)}")
            return 0.5  # Возвращаем среднее значение при ошибке

    async def analyze_spoof_signs(self, image_data: bytes) -> Dict[str, Any]:
        """
        Простые эвристики антиспуфинга (экраны/печать/блики/плоскость).
        Возвращает оценку и признаки. Не заменяет ML-модель.
        """
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"score": 0.5, "flags": ["decode_failed"]}

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Блики: анализ ярких пятен
            bright_mask = gray > 240
            bright_ratio = float(np.mean(bright_mask))

            # Плоскость/экран: низкая вариативность глубины по текстуре (низкий Laplacian)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Муар/повторяющийся паттерн: FFT энергия в высоких частотах
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
            high_freq_energy = float(np.mean(magnitude_spectrum[-50:, -50:]))

            # Насыщенность: печать/экран часто имеет низкую насыщенность
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation_mean = float(np.mean(hsv[:, :, 1])) / 255.0

            flags = []
            if bright_ratio > 0.12:
                flags.append("glare")
            if lap_var < 50:
                flags.append("flat_surface")
            if high_freq_energy < 10:
                flags.append("moire_low")
            if saturation_mean < 0.15:
                flags.append("low_saturation")

            # Простая агрегированная оценка (0–1, выше — живее)
            score = 0.5
            score += min(lap_var / 400.0, 0.3)  # резкость
            score += max(0, 0.15 - bright_ratio)  # меньше бликов — лучше
            score += max(0, saturation_mean - 0.1) * 0.2
            score = float(np.clip(score, 0.0, 1.0))

            return {
                "score": score,
                "flags": flags,
                "laplacian_var": lap_var,
                "bright_ratio": bright_ratio,
                "high_freq_energy": high_freq_energy,
                "saturation_mean": saturation_mean,
            }
        except Exception as e:
            logger.warning(f"Error in spoof analysis: {e}")
            return {"score": 0.5, "flags": ["analysis_error"]}
    
    async def _detect_face_basic(self, image_data: bytes) -> bool:
        """
        Базовая проверка наличия лица в изображении.
        
        Args:
            image_data: Двоичные данные изображения
            
        Returns:
            bool: True если лицо обнаружено
        """
        try:
            # Загружаем изображение в OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return False
            
            # Загружаем Haar каскад для обнаружения лиц
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Конвертируем в grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Обнаруживаем лица
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Возвращаем True если найдено хотя бы одно лицо
            return len(faces) > 0
            
        except Exception as e:
            logger.warning(f"Error in basic face detection: {str(e)}")
            return False  # При ошибке считаем, что лица нет
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Валидация метаданных.
        
        Args:
            metadata: Словарь с метаданными
            
        Returns:
            bool: True если метаданные валидны
        """
        try:
            if not isinstance(metadata, dict):
                return False
            
            # Проверяем размер метаданных
            import json
            metadata_str = json.dumps(metadata)
            if len(metadata_str) > 10240:  # 10KB
                return False
            
            # Проверяем глубину вложенности
            def check_depth(obj, depth=0):
                if depth > 5:  # Максимум 5 уровней вложенности
                    return False
                if isinstance(obj, dict):
                    return all(check_depth(v, depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, depth + 1) for item in obj)
                else:
                    return True
            
            return check_depth(metadata)
            
        except Exception as e:
            logger.warning(f"Error validating metadata: {str(e)}")
            return False
    
    def validate_user_input(self, data: Dict[str, Any], validation_rules: Dict[str, Any]) -> bool:
        """
        Валидация пользовательского ввода по правилам.
        
        Args:
            data: Данные для валидации
            validation_rules: Правила валидации
            
        Returns:
            bool: True если данные валидны
        """
        try:
            for field, rules in validation_rules.items():
                if field not in data:
                    if rules.get("required", False):
                        return False
                    continue
                
                value = data[field]
                
                # Проверка типа
                if "type" in rules:
                    expected_type = rules["type"]
                    if not isinstance(value, expected_type):
                        return False
                
                # Проверка длины
                if "min_length" in rules and len(str(value)) < rules["min_length"]:
                    return False
                if "max_length" in rules and len(str(value)) > rules["max_length"]:
                    return False
                
                # Проверка диапазона
                if "min_value" in rules and value < rules["min_value"]:
                    return False
                if "max_value" in rules and value > rules["max_value"]:
                    return False
                
                # Проверка паттерна
                if "pattern" in rules and not re.match(rules["pattern"], str(value)):
                    return False
                
                # Пользовательская валидация
                if "validator" in rules and callable(rules["validator"]):
                    if not rules["validator"](value):
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating user input: {str(e)}")
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Санитизация имени файла.
        
        Args:
            filename: Исходное имя файла
            
        Returns:
            str: Санитизированное имя файла
        """
        try:
            # Удаляем опасные символы
            sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            # Ограничиваем длину
            if len(sanitized) > 255:
                name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
                sanitized = name[:250 - len(ext)] + ('.' + ext if ext else '')
            
            return sanitized
            
        except Exception:
            return "sanitized_file"
    
    def generate_file_hash(self, file_data: bytes) -> str:
        """
        Генерация хеша файла.
        
        Args:
            file_data: Двоичные данные файла
            
        Returns:
            str: SHA256 хеш файла
        """
        try:
            return hashlib.sha256(file_data).hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
            return ""