"""
Утилиты для работы с файлами и изображениями.
Валидация, конвертация, ресайз и метаданные.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from io import BytesIO
from typing import Tuple, Dict, Any

from PIL import Image, ImageFile

from .logger import get_logger
from .exceptions import (
    ValidationError,
    ProcessingError,
)

logger = get_logger(__name__)

# Защита от decompression bomb
Image.MAX_IMAGE_PIXELS = 20_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False


class FileUtils:
    """Низкоуровневые утилиты работы с файлами"""

    ALLOWED_FORMATS = {"jpg", "jpeg", "png", "heic"}
    TARGET_FORMAT = "jpg"
    MAX_FILE_SIZE_MB = 10

    # ======================================================
    # Basic helpers
    # ======================================================

    @staticmethod
    def get_file_extension(filename: str) -> str:
        return Path(filename).suffix.lower().lstrip(".")

    @staticmethod
    def get_file_size_mb(content: bytes) -> float:
        return len(content) / (1024 * 1024)

    @staticmethod
    def calculate_file_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    # ======================================================
    # Keys
    # ======================================================

    @staticmethod
    def generate_upload_key(user_id: str, filename: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = os.path.basename(filename)
        return f"uploads/{user_id}/{timestamp}_{safe_name}"

    @staticmethod
    def generate_reference_key(user_id: str, version: int) -> str:
        return f"references/{user_id}/v{version}.jpg"

    # ======================================================
    # Image operations
    # ======================================================

    @staticmethod
    def open_image(content: bytes) -> Image.Image:
        try:
            img = Image.open(BytesIO(content))
            img.verify()
            img = Image.open(BytesIO(content))
            return img
        except Exception as e:
            logger.warning("Invalid image file")
            raise ValidationError(
                message="Invalid image file",
                details={"error": str(e)},
            )

    @staticmethod
    def ensure_rgb(img: Image.Image) -> Image.Image:
        if img.mode in {"RGBA", "LA", "P"}:
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            return background
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    @staticmethod
    def convert_to_jpeg(
        content: bytes,
        filename: str,
        quality: int = 85,
    ) -> Tuple[bytes, str]:
        img = FileUtils.open_image(content)
        img = FileUtils.ensure_rgb(img)

        output = BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)

        new_name = f"{Path(filename).stem}.jpg"
        converted = output.getvalue()

        logger.info(
            "Image converted to JPEG",
            extra={
                "original": filename,
                "new": new_name,
                "before_bytes": len(content),
                "after_bytes": len(converted),
            },
        )

        return converted, new_name

    @staticmethod
    def resize_if_needed(
        content: bytes,
        max_width: int = 1024,
        max_height: int = 1024,
    ) -> bytes:
        img = FileUtils.open_image(content)
        img = FileUtils.ensure_rgb(img)

        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        output = BytesIO()
        img.save(output, format="JPEG", quality=90, optimize=True)
        resized = output.getvalue()

        logger.info(
            "Image resized",
            extra={
                "width": img.width,
                "height": img.height,
                "bytes": len(resized),
            },
        )

        return resized


class ImageValidator:
    """Валидация изображений для face recognition"""

    MIN_WIDTH = 50
    MIN_HEIGHT = 50
    MAX_FILE_SIZE_MB = 10

    @staticmethod
    def validate(content: bytes, filename: str) -> None:
        ext = FileUtils.get_file_extension(filename)
        if ext not in FileUtils.ALLOWED_FORMATS:
            raise ValidationError(
                message="Unsupported image format",
                field="filename",
                value=ext,
            )

        size_mb = FileUtils.get_file_size_mb(content)
        if size_mb > ImageValidator.MAX_FILE_SIZE_MB:
            raise ValidationError(
                message="File size exceeds limit",
                field="file_size_mb",
                value=round(size_mb, 2),
                details={"max_mb": ImageValidator.MAX_FILE_SIZE_MB},
            )

        img = FileUtils.open_image(content)
        width, height = img.size

        if width < ImageValidator.MIN_WIDTH or height < ImageValidator.MIN_HEIGHT:
            raise ValidationError(
                message="Image resolution too small",
                details={
                    "width": width,
                    "height": height,
                    "min_width": ImageValidator.MIN_WIDTH,
                    "min_height": ImageValidator.MIN_HEIGHT,
                },
            )

        logger.info(
            "Image validation passed",
            extra={
                "filename": filename,
                "width": width,
                "height": height,
                "size_mb": round(size_mb, 2),
            },
        )

    @staticmethod
    def get_info(content: bytes, filename: str) -> Dict[str, Any]:
        try:
            img = FileUtils.open_image(content)
            return {
                "filename": filename,
                "width": img.width,
                "height": img.height,
                "size_mb": FileUtils.get_file_size_mb(content),
                "format": img.format,
                "mode": img.mode,
                "file_hash": FileUtils.calculate_file_hash(content),
            }
        except Exception as e:
            raise ProcessingError(
                message="Failed to extract image metadata",
                details={"error": str(e)},
            )
