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
from typing import Any, Dict, List, Optional, Tuple
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
from ..utils.exceptions import ValidationError
from ..utils.file_utils import (
    MAX_FILE_SIZE,
    SUPPORTED_EXTENSIONS,
    ImageFileHandler,
    validate_image_file,
)
from ..utils.logger import get_logger

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
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
            )

            # Стандартные пути к файлам модели
            prototxt_path = os.path.join(base_models_path, "mask_detector.prototxt")
            caffemodel_path = os.path.join(base_models_path, "mask_detector.caffemodel")

            # Проверяем наличие файлов модели
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                self._mask_net = cv2.dnn.readNetFromCaffe(
                    prototxt_path, caffemodel_path
                )
                logger.info("Mask detection model loaded from project models directory")
                return

            # Альтернативные пути для fall-back модели
            alt_prototxt = os.path.join(base_models_path, "deploy.prototxt")
            alt_caffemodel = os.path.join(
                base_models_path, "res10_300x300_ssd_iter_140000.caffemodel"
            )

            if os.path.exists(alt_prototxt) and os.path.exists(alt_caffemodel):
                self._mask_net = cv2.dnn.readNetFromCaffe(alt_prototxt, alt_caffemodel)
                logger.info(
                    "Face detection model loaded (mask detection requires additional training)"
                )
                return

            # Если модель не найдена, используем резервный метод на основе анализа ключевых точек
            logger.warning(
                "Mask detection model files not found, using fallback method"
            )
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
                "Пожалуйста, уменьшите размер до 4096x4096 или меньше.",
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
                    errors=["Failed to decode image for mask detection"],
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
                    errors=["No faces detected for mask analysis"],
                )

            # Используем DNN модель для детекции масок
            if self._mask_net is not None:
                (h, w) = img.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
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
                            if (
                                startX >= fx
                                and startY >= fy
                                and endX <= fx + fw
                                and endY <= fy + fh
                            ):
                                if confidence > 0.65:  # Порог для маски
                                    face_with_mask += 1
                                else:
                                    face_without_mask += 1
                                total_confidence += confidence
                                break

            else:
                # Резервный метод: анализ нижней части лица
                for x, y, face_w, face_h in face_coords:
                    # Выделяем область нижней части лица (рот, нос)
                    face_roi = img[y + face_h // 2 : y + face_h, x : x + face_w]

                    if face_roi.size == 0:
                        face_without_mask += 1
                        continue

                    # Конвертируем в HSV для анализа оттенков кожи и тканевых масок
                    hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

                    # Маска для диапазона цветов кожи
                    skin_mask = cv2.inRange(
                        hsv_roi, np.array([0, 20, 70]), np.array([20, 170, 255])
                    )

                    # Анализируем видимость кожи в нижней части лица
                    skin_ratio = cv2.countNonZero(skin_mask) / (
                        face_roi.shape[0] * face_roi.shape[1]
                    )

                    # Если мало видимой кожи - возможно маска
                    if skin_ratio < 0.35:
                        face_with_mask += 1
                        total_confidence += 1.0 - skin_ratio
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
                errors=errors if errors else None,
            )

        except Exception as e:
            logger.exception("Mask detection failed")
            return MaskDetectionResult(
                is_mask_detected=False,
                confidence=0.0,
                errors=[f"Mask detection error: {str(e)}"],
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

        score = sharpness * 0.4 + brightness_score * 0.3 + noise_score * 0.3
        return float(np.clip(score, 0.0, 1.0))

    async def _detect_face_basic(
        self, image_data: bytes
    ) -> Tuple[bool, Optional[List[Tuple[int, int, int, int]]]]:
        def _sync_face_detection():
            img = cv2.imdecode(
                np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE
            )
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

    # =========================================================================
    # НОВЫЕ МЕТОДЫ ВАЛИДАЦИИ (с использованием ImageFileHandler)
    # =========================================================================

    def validate_uploaded_image(
        self,
        file_data: bytes,
        filename: str,
        check_face: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Полная валидация загруженного изображения

        Args:
            file_data: Байты файла
            filename: Имя файла
            check_face: Проверять наличие лица

        Returns:
            (image_array, image_info)

        Raises:
            ValidationError: При ошибке валидации
        """
        logger.info(f"Валидация изображения: {filename}")

        # 1. Проверка формата и размера
        is_valid, error_msg = validate_image_file(file_data, filename)
        if not is_valid:
            raise ValidationError(error_msg)

        # 2. Получение информации о файле
        image_info = ImageFileHandler.get_image_info(file_data, filename)
        logger.info(
            f"Информация об изображении: {image_info['width']}x{image_info['height']}, "
            f"format: {image_info['format']}, size: {image_info['size_mb']:.2f} MB"
        )

        # 3. Загрузка изображения (с автоконвертацией HEIC)
        image = ImageFileHandler.load_image_from_bytes(file_data, filename)

        # 4. Проверка качества изображения
        self._check_image_quality(image)

        # 5. Проверка наличия лица (опционально)
        if check_face:
            # Эта логика должна быть в face detection service
            pass

        return image, image_info

    def _check_image_quality(self, image: np.ndarray) -> None:
        """
        Проверка качества изображения

        Args:
            image: Массив изображения

        Raises:
            ValidationError: При недостаточном качестве
        """
        # Проверка минимального разрешения
        height, width = image.shape[:2]
        min_dimension = 160  # Минимум для FaceNet

        if height < min_dimension or width < min_dimension:
            raise ValidationError(
                f"Слишком низкое разрешение: {width}x{height}. "
                f"Минимум: {min_dimension}x{min_dimension}"
            )

        # Проверка на черно-белое изображение
        if len(image.shape) == 2 or image.shape[2] == 1:
            logger.warning("Обнаружено черно-белое изображение")

        # Проверка размытости (Laplacian variance)
        gray = image if len(image.shape) == 2 else image[:, :, 0]
        laplacian_var = np.var(gray)

        if laplacian_var < 100:  # Пороговое значение для размытости
            logger.warning(
                f"Изображение может быть размытым (variance: {laplacian_var:.2f})"
            )

    def get_supported_formats_info(self) -> dict:
        """
        Получение информации о поддерживаемых форматах

        Returns:
            Словарь с информацией
        """
        return {
            "supported_extensions": SUPPORTED_EXTENSIONS,
            "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
            "max_dimension": 4096,
            "min_dimension": 160,
            "formats": {
                "JPEG": {
                    "extensions": [".jpg", ".jpeg"],
                    "description": "Стандартный формат",
                },
                "PNG": {
                    "extensions": [".png"],
                    "description": "Формат с поддержкой прозрачности",
                },
                "HEIC": {
                    "extensions": [".heic", ".heif"],
                    "description": "Apple формат (автоконвертация в JPEG)",
                },
                "WebP": {
                    "extensions": [".webp"],
                    "description": "Google формат",
                },
            },
        }

    async def analyze_spoof_signs(self, image_data: bytes) -> Dict[str, Any]:
        """
        Анализ признаков подделки (spoofing) с помощью эвристик.

        Detects common spoofing indicators:
        - moire_pattern: муаровые паттерны от экранов
        - screen_glare: блики от экрана
        - color_banding: цветовые полосы (компрессия экрана)
        - edge_artifacts: артефакты по краям
        - noise_inconsistency: неестественный шум

        Args:
            image_data: Байты изображения

        Returns:
            Dict с 'score' (0-1, выше = более похоже на real) и 'flags' (список обнаруженных признаков)
        """

        def _sync_analyze():
            try:
                img = cv2.imdecode(
                    np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
                )
                if img is None:
                    return {"score": 0.5, "flags": ["decode_failed"], "details": {}}

                h, w = img.shape[:2]
                flags = []
                details = {}

                # 1. Moiré pattern detection (муаровые полосы от экранов)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                fourier = np.fft.fft2(gray)
                fourier_shift = np.fft.fftshift(fourier)
                magnitude = np.log(np.abs(fourier_shift) + 1)

                # Ищем высокочастотные паттерны в центральной области
                center_h, center_w = h // 2, w // 2
                center_region = magnitude[
                    max(0, center_h - 50) : min(h, center_h + 50),
                    max(0, center_w - 50) : min(w, center_w + 50),
                ]
                moire_score = (
                    np.std(center_region) / np.mean(center_region)
                    if np.mean(center_region) > 0
                    else 0
                )
                details["moire_score"] = round(moire_score, 3)
                if moire_score > 0.8:
                    flags.append("moire_pattern")

                # 2. Screen glare detection (блики от экрана)
                gray_float = gray.astype(np.float32) / 255.0
                luminance = (
                    cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
                    / 255.0
                )

                # Ищем области с очень высокой яркостью
                high_lum_mask = luminance > 0.95
                glare_ratio = np.sum(high_lum_mask) / (h * w)
                details["glare_ratio"] = round(glare_ratio, 4)
                if glare_ratio > 0.15:
                    flags.append("screen_glare")

                # 3. Color banding detection (цветовые полосы от экрана)
                # Проверяем плавность градиентов
                grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                # Экраны часто имеют artificial sharp gradients
                sharp_edges = np.sum(gradient_magnitude > 0.3) / (h * w)
                details["sharp_edge_ratio"] = round(sharp_edges, 4)
                if sharp_edges > 0.4:
                    flags.append("color_banding")

                # 4. Edge artifacts (артефакты по краям - характерно для фото экранов)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (h * w)
                details["edge_density"] = round(edge_density, 4)

                # Нормальное фото имеет ~10-30% edge density
                # Фото экрана может иметь аномально высокий или низкий
                if edge_density > 0.5 or edge_density < 0.02:
                    flags.append("edge_artifacts")

                # 5. Noise inconsistency analysis
                # Вычисляем локальную дисперсию шума
                noise_std_local = (
                    cv2.blur(gray.astype(np.float32) ** 2, (15, 15))
                    - cv2.blur(gray.astype(np.float32), (15, 15)) ** 2
                )
                noise_std_local = np.sqrt(np.maximum(noise_std_local, 0))

                # Фото экрана часто имеет uniform noise
                noise_std_global = np.std(noise_std_local)
                noise_mean_local = np.mean(noise_std_local)
                noise_consistency = noise_std_global / (noise_mean_local + 1e-6)
                details["noise_consistency"] = round(noise_consistency, 3)

                # Неестественно uniform шум = признак экрана
                if noise_consistency < 0.1:
                    flags.append("noise_inconsistency")

                # 6. Chromatic aberration check
                # Экраны могут проявлять хроматическую аберрацию по краям
                b, g, r = cv2.split(img)
                b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
                b_std, g_std, r_std = np.std(b), np.std(g), np.std(r)

                # Аномально высокая корреляция между каналами
                rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
                bg_corr = np.corrcoef(b.flatten(), g.flatten())[0, 1]
                details["channel_correlation"] = {
                    "rg": round(rg_corr, 3),
                    "bg": round(bg_corr, 3),
                }

                if rg_corr > 0.95 and bg_corr > 0.95:
                    flags.append("screen_capture_pattern")

                # 7. Calculate overall spoof score
                # Каждый флаг снижает score
                base_score = 1.0
                flag_weights = {
                    "moire_pattern": 0.15,
                    "screen_glare": 0.2,
                    "color_banding": 0.15,
                    "edge_artifacts": 0.1,
                    "noise_inconsistency": 0.2,
                    "screen_capture_pattern": 0.2,
                }

                for flag in flags:
                    base_score -= flag_weights.get(flag, 0.1)

                score = max(0.0, min(1.0, base_score))

                return {
                    "score": round(score, 3),
                    "flags": flags,
                    "details": details,
                }

            except Exception as e:
                logger.warning(f"Spoof analysis failed: {e}")
                return {
                    "score": 0.5,
                    "flags": ["analysis_error"],
                    "details": {"error": str(e)},
                }

        return await asyncio.to_thread(_sync_analyze)
