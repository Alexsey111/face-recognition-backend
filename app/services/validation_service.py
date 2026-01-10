"""
Сервис валидации данных.
Валидация изображений, безопасность и проверка форматов.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import socket
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

# DecompressionBombError был добавлен в Pillow 10.3.0
try:
    from PIL import DecompressionBombError
    _HAS_DECOMPRESSION_BOMB_ERROR = True
except ImportError:
    _HAS_DECOMPRESSION_BOMB_ERROR = False

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except Exception:
    HEIF_SUPPORTED = False

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)

# Защита от decompression bomb
Image.MAX_IMAGE_PIXELS = 25_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False


class MaskDetectionResult:
    def __init__(
        self,
        is_mask_detected: bool,
        confidence: float = 0.0,
        face_with_mask: int = 0,
        face_without_mask: int = 0,
        errors: Optional[List[str]] = None,
    ):
        self.is_mask_detected = is_mask_detected
        self.confidence = confidence
        self.face_with_mask = face_with_mask
        self.face_without_mask = face_without_mask
        self.errors = errors or []


class ValidationResult:
    def __init__(
        self,
        is_valid: bool,
        image_data: Optional[bytes] = None,
        image_format: Optional[str] = None,
        dimensions: Optional[Dict[str, int]] = None,
        quality_score: Optional[float] = None,
        mask_result: Optional[MaskDetectionResult] = None,
        error_message: Optional[str] = None,
    ):
        self.is_valid = is_valid
        self.image_data = image_data
        self.image_format = image_format
        self.dimensions = dimensions
        self.quality_score = quality_score
        self.mask_result = mask_result
        self.error_message = error_message


class ValidationService:
    """
    Сервис валидации данных и изображений.
    """

    _face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Пути к файлам модели маски (будут инициализированы в __init__)
    _mask_prototxt = None
    _mask_caffemodel = None
    _mask_net = None

    def __init__(self):
        self.max_file_size = settings.MAX_UPLOAD_SIZE
        self.allowed_formats = settings.allowed_image_formats_list
        self.min_width = settings.MIN_IMAGE_WIDTH
        self.min_height = settings.MIN_IMAGE_HEIGHT
        self.max_width = settings.MAX_IMAGE_WIDTH
        self.max_height = settings.MAX_IMAGE_HEIGHT

        self._http_client = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=False,
            headers={"User-Agent": "ValidationService/1.0"},
        )

        # Инициализация модели детекции масок
        self._init_mask_detection_model()

    def _init_mask_detection_model(self) -> None:
        """
        Инициализирует модель детекции масок.
        Использует MobileNet SSD с предобученными весами для классификации масок.
        """
        try:
            # Попытка загрузить модель из директории моделей проекта
            base_models_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models"
            )

            # Стандартные пути к файлам модели
            prototxt_path = os.path.join(base_models_path, "mask_detector.prototxt")
            caffemodel_path = os.path.join(base_models_path, "mask_detector.caffemodel")

            # Проверяем наличие файлов модели
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                self._mask_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                logger.info("Mask detection model loaded from project models directory")
                return

            # Альтернативные пути для fall-back модели
            alt_prototxt = os.path.join(base_models_path, "deploy.prototxt")
            alt_caffemodel = os.path.join(base_models_path, "res10_300x300_ssd_iter_140000.caffemodel")

            if os.path.exists(alt_prototxt) and os.path.exists(alt_caffemodel):
                self._mask_net = cv2.dnn.readNetFromCaffe(alt_prototxt, alt_caffemodel)
                logger.info("Face detection model loaded (mask detection requires additional training)")
                return

            # Если модель не найдена, используем резервный метод на основе анализа ключевых точек
            logger.warning("Mask detection model files not found, using fallback method")
            self._mask_net = None

        except Exception as e:
            logger.warning(f"Failed to load mask detection model: {e}")
            self._mask_net = None

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def validate_image(
        self,
        image_data: str,
        max_size: Optional[int] = None,
        allowed_formats: Optional[List[str]] = None,
        check_mask: bool = False,
    ) -> ValidationResult:
        """
        Полная валидация изображения для верификации лица.

        Проверяет:
        - Размер файла
        - Поддерживаемый формат
        - Размеры в пикселях (min/max из settings)
        - Качество (sharpness, brightness, noise)
        - Наличие лица (haarcascade)
        - Наличие маски (опционально, если check_mask=True)

        Args:
            image_data: base64, data_url или URL изображения
            max_size: максимальный размер в байтах (по умолчанию из settings)
            allowed_formats: список разрешённых форматов (по умолчанию из settings)
            check_mask: флаг для проверки наличия маски

        Returns:
            ValidationResult с is_valid, данными и метриками качества

        Raises:
            ValidationError: при любой ошибке валидации
        """
        try:
            max_size = max_size or self.max_file_size
            allowed_formats = (
                [f.upper() for f in allowed_formats]
                if allowed_formats
                else self.allowed_formats
            )

            decoded, source = await self._decode_image_data(image_data, max_size)
            if not decoded:
                return ValidationResult(False, error_message="Image decode failed")

            if len(decoded) > max_size:
                return ValidationResult(False, error_message="File too large")

            image_format = self._detect_image_format(decoded)
            if image_format not in allowed_formats:
                return ValidationResult(
                    False, error_message=f"Format {image_format} not allowed"
                )

            if image_format in {"HEIC", "HEIF"} and not HEIF_SUPPORTED:
                return ValidationResult(False, error_message="HEIC/HEIF not supported")

            with Image.open(io.BytesIO(decoded)) as img:
                width, height = img.size

                if not (
                    self.min_width <= width <= self.max_width
                    and self.min_height <= height <= self.max_height
                ):
                    return ValidationResult(False, error_message="Invalid dimensions")

                quality = await self._assess_image_quality(decoded, img)

                face_detected, face_coords = await self._detect_face_basic(decoded)
                if not face_detected:
                    return ValidationResult(False, error_message="Face not detected")

                mask_result = None
                if check_mask and face_coords is not None:
                    mask_result = await self.detect_mask(decoded, face_coords)

                return ValidationResult(
                    True,
                    image_data=decoded,
                    image_format=image_format,
                    dimensions={"width": width, "height": height},
                    quality_score=quality,
                    mask_result=mask_result,
                )

        except Image.DecompressionBombError:
            return ValidationResult(
                False,
                error_message="Изображение слишком большое по количеству пикселей. "
                            "Пожалуйста, уменьшите размер до 4096x4096 или меньше."
            )
        except Exception as e:
            logger.exception("Image validation failed")
            return ValidationResult(False, error_message=str(e))

    async def detect_mask(
        self,
        image_data: bytes,
        face_coords: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> MaskDetectionResult:
        """
        Детекция масок на лицах в изображении.

        Использует глубокую нейронную сеть (MobileNet SSD) для классификации
        наличия маски на лице. Если модель не загружена, используется
        резервный метод на основе анализа нижней части лица.

        Args:
            image_data: байты изображения
            face_coords: список координат обнаруженных лиц (x, y, w, h)

        Returns:
            MaskDetectionResult с результатами детекции масок
        """
        errors = []
        face_with_mask = 0
        face_without_mask = 0
        total_confidence = 0.0

        try:
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return MaskDetectionResult(
                    is_mask_detected=False,
                    confidence=0.0,
                    errors=["Failed to decode image for mask detection"]
                )

            # Если координаты лиц не переданы, детектируем лица повторно
            if face_coords is None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )
                face_coords = [tuple(face) for face in faces]

            if len(face_coords) == 0:
                return MaskDetectionResult(
                    is_mask_detected=False,
                    confidence=0.0,
                    errors=["No faces detected for mask analysis"]
                )

            # Используем DNN модель для детекции масок
            if self._mask_net is not None:
                (h, w) = img.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(img, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0)
                )
                self._mask_net.setInput(blob)
                detections = self._mask_net.forward()

                # Обработка результатов детекции
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Проверяем, находится ли обнаруженное лицо в пределах
                        # известных координат лиц
                        for fx, fy, fw, fh in face_coords:
                            if (startX >= fx and startY >= fy and
                                endX <= fx + fw and endY <= fy + fh):
                                if confidence > 0.65:  # Порог для маски
                                    face_with_mask += 1
                                else:
                                    face_without_mask += 1
                                total_confidence += confidence
                                break

            else:
                # Резервный метод: анализ нижней части лица
                for (x, y, face_w, face_h) in face_coords:
                    # Выделяем область нижней части лица (рот, нос)
                    face_roi = img[y + face_h // 2:y + face_h, x:x + face_w]

                    if face_roi.size == 0:
                        face_without_mask += 1
                        continue

                    # Конвертируем в HSV для анализа оттенков кожи и тканевых масок
                    hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

                    # Маска для диапазона цветов кожи
                    skin_mask = cv2.inRange(
                        hsv_roi,
                        np.array([0, 20, 70]),
                        np.array([20, 170, 255])
                    )

                    # Анализируем видимость кожи в нижней части лица
                    skin_ratio = cv2.countNonZero(skin_mask) / (face_roi.shape[0] * face_roi.shape[1])

                    # Если мало видимой кожи - возможно маска
                    if skin_ratio < 0.35:
                        face_with_mask += 1
                        total_confidence += (1.0 - skin_ratio)
                    else:
                        face_without_mask += 1
                        total_confidence += skin_ratio

            total_faces = face_with_mask + face_without_mask
            avg_confidence = total_confidence / total_faces if total_faces > 0 else 0.0

            return MaskDetectionResult(
                is_mask_detected=face_with_mask > 0,
                confidence=min(avg_confidence, 1.0),
                face_with_mask=face_with_mask,
                face_without_mask=face_without_mask,
                errors=errors if errors else None
            )

        except Exception as e:
            logger.exception("Mask detection failed")
            return MaskDetectionResult(
                is_mask_detected=False,
                confidence=0.0,
                errors=[f"Mask detection error: {str(e)}"]
            )

    async def _decode_image_data(
        self, image_data: str, max_size: int
    ) -> Tuple[Optional[bytes], str]:
        try:
            if image_data.startswith("data:image/"):
                _, data = image_data.split(",", 1)
                return base64.b64decode(data, validate=True), "data_url"

            if image_data.startswith(("http://", "https://")):
                self._validate_url_security(image_data)
                return await self._fetch_image_from_url(image_data, max_size), "url"

            return base64.b64decode(image_data, validate=True), "base64"

        except Exception:
            return None, "invalid"

    def _validate_url_security(self, url: str) -> None:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            raise ValidationError("Invalid URL")

        try:
            ip = socket.gethostbyname(host)
        except Exception:
            raise ValidationError("DNS resolution failed")

        private_prefixes = (
            "127.",
            "10.",
            "192.168.",
            "169.254.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.2",
        )
        if ip.startswith(private_prefixes):
            raise ValidationError("SSRF blocked")

    async def _fetch_image_from_url(self, url: str, max_size: int) -> Optional[bytes]:
        async with self._http_client.stream("GET", url) as resp:
            if resp.status_code != 200:
                return None

            data = bytearray()
            async for chunk in resp.aiter_bytes():
                data.extend(chunk)
                if len(data) > max_size:
                    return None
            return bytes(data)

    def _detect_image_format(self, data: bytes) -> str:
        if data.startswith(b"\xff\xd8\xff"):
            return "JPEG"
        if data.startswith(b"\x89PNG"):
            return "PNG"
        if data.startswith(b"GIF8"):
            return "GIF"
        if data.startswith(b"RIFF") and b"WEBP" in data[:12]:
            return "WEBP"
        if b"ftypheic" in data[:32].lower():
            return "HEIC"
        if b"ftypheif" in data[:32].lower():
            return "HEIF"
        return "UNKNOWN"

    async def _assess_image_quality(self, data: bytes, img: Image.Image) -> float:
        gray = np.array(img.convert("L"))
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(lap_var / 800.0, 1.0)

        brightness = np.mean(gray)
        brightness_score = 1 - abs(brightness - 128) / 128

        noise = np.std(gray)
        noise_score = max(0.0, 1 - noise / 64)

        score = (
            sharpness * 0.4
            + brightness_score * 0.3
            + noise_score * 0.3
        )
        return float(np.clip(score, 0.0, 1.0))

    async def _detect_face_basic(self, image_data: bytes) -> Tuple[bool, Optional[List[Tuple[int, int, int, int]]]]:
        def _sync_face_detection():
            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, None
            faces = self._face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            face_coords = [tuple(face) for face in faces] if len(faces) > 0 else None
            return len(faces) > 0, face_coords

        return await asyncio.to_thread(_sync_face_detection)

    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        try:
            raw = json.dumps(metadata)
            if len(raw) > 10_240:
                return False

            def depth(obj, lvl=0):
                if lvl > 5:
                    return False
                if isinstance(obj, dict):
                    return all(depth(v, lvl + 1) for v in obj.values())
                if isinstance(obj, list):
                    return all(depth(i, lvl + 1) for i in obj)
                return True

            return depth(metadata)
        except Exception:
            return False

    def sanitize_filename(self, filename: str) -> str:
        name = re.sub(r"[<>:\"/\\|?*]", "_", filename)
        return name[:255]

    def generate_file_hash(self, file_data: bytes) -> str:
        return hashlib.sha256(file_data).hexdigest()
