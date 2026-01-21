"""
File Storage Service (S3 / MinIO)

Асинхронный сервис работы с объектным хранилищем.
boto3 используется через executor (boto3 НЕ async).

⚠️ Важно:
- Этот сервис безопасен для production при корректных настройках
- Public-read используется ТОЛЬКО если явно включён в settings
"""

from __future__ import annotations

import io
import mimetypes
import asyncio
import functools
import json
import socket
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from PIL import Image, UnidentifiedImageError

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import StorageError, ValidationError

logger = get_logger(__name__)


# =============================================================================
# Constants & Limits
# =============================================================================

ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}

MAX_IMAGE_SIZE_BYTES = getattr(
    settings, "MAX_IMAGE_SIZE_BYTES", 10 * 1024 * 1024
)  # 10MB


# =============================================================================
# Storage Service
# =============================================================================


class StorageService:
    """
    S3 / MinIO storage service
    """

    def __init__(self) -> None:
        self.endpoint_url = settings.S3_ENDPOINT_URL
        self.access_key = settings.S3_ACCESS_KEY
        self.secret_key = settings.S3_SECRET_KEY
        self.bucket_name = settings.S3_BUCKET_NAME
        self.region = settings.S3_REGION
        self.use_ssl = settings.S3_USE_SSL
        self.public_read = bool(getattr(settings, "S3_PUBLIC_READ", False))
        self.auto_create_bucket = bool(
            getattr(settings, "S3_AUTO_CREATE_BUCKET", False)
        )

        self.s3_client = self._init_client()
        self._reconnect_lock = asyncio.Lock()  # асинхронный lock

    # ---------------------------------------------------------------------

    def _init_client(self):
        try:
            config = Config(
                region_name=self.region,
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            )

            return boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=config,
                use_ssl=self.use_ssl,
            )
        except Exception as e:
            raise StorageError(f"S3 client initialization failed: {e}")

    # ---------------------------------------------------------------------

    async def _run(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        max_retries = 5
        delays = [1, 2, 4, 8, 16]

        for attempt in range(max_retries):
            try:
                return await loop.run_in_executor(
                    None, functools.partial(func, *args, **kwargs)
                )
            except (
                ClientError,
                ConnectionError,
                EndpointConnectionError,
                socket.timeout,
            ) as e:
                if attempt == max_retries - 1:
                    raise

                async with self._reconnect_lock:  # только один поток делает reconnect
                    logger.warning(
                        f"S3 error (attempt {attempt + 1}/{max_retries}): {e}. Reconnecting..."
                    )
                    self.s3_client = self._init_client()

                await asyncio.sleep(delays[attempt])

        raise RuntimeError("Unexpected exit from retry loop")

    # --------------------------------------------------

    async def health_check(self) -> bool:
        try:
            await self._run(self.s3_client.head_bucket, Bucket=self.bucket_name)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                if self.auto_create_bucket:
                    await self._create_bucket()
                    return True
                return False
            return False

        except NoCredentialsError:
            return False

    # ---------------------------------------------------------------------

    async def _create_bucket(self) -> None:
        try:
            if self.region == "us-east-1":
                await self._run(
                    self.s3_client.create_bucket,
                    Bucket=self.bucket_name,
                )
            else:
                await self._run(
                    self.s3_client.create_bucket,
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )

            if self.public_read:
                await self._setup_public_policy()

        except Exception as e:
            raise StorageError(f"Bucket creation failed: {e}")

    # ---------------------------------------------------------------------

    async def _setup_public_policy(self) -> None:
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/*",
                }
            ],
        }

        try:
            await self._run(
                self.s3_client.put_bucket_policy,
                Bucket=self.bucket_name,
                Policy=json.dumps(policy),
            )
        except Exception as e:
            logger.warning("Failed to set bucket policy: %s", e)

    # ---------------------------------------------------------------------
    """
    Загружает изображение в S3/MinIO хранилище.

    Args:
        image_data: Байты изображения
        key: Ключ объекта (если None — генерируется автоматически)
        metadata: Дополнительные метаданные (опционально)

    Returns:
        dict с key, file_url, file_size, content_type

    Raises:
        ValidationError: Если изображение не проходит валидацию (размер, тип, формат)
        StorageError: При ошибке загрузки в S3
    """

    async def upload_image(
        self,
        image_data: bytes,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if not image_data:
            raise ValidationError("Empty image data")

        mime_type, width, height = self._detect_image_mime(image_data)

        if width > settings.MAX_IMAGE_WIDTH or height > settings.MAX_IMAGE_HEIGHT:
            raise ValidationError(
                f"Image dimensions {width}x{height} exceed maximum allowed "
                f"{settings.MAX_IMAGE_WIDTH}x{settings.MAX_IMAGE_HEIGHT}"
            )

        if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValidationError(f"Unsupported image type: {mime_type}")

        if not key:
            key = self._generate_object_key(mime_type)

        s3_metadata = self._prepare_metadata(metadata)

        put_args = {
            "Bucket": self.bucket_name,
            "Key": key,
            "Body": image_data,
            "ContentType": mime_type,
            "Metadata": s3_metadata,
        }

        if self.public_read:
            put_args["ACL"] = "public-read"

        try:
            await self._run(self.s3_client.put_object, **put_args)
        except Exception as e:
            raise StorageError(f"Upload failed: {e}")

        return {
            "key": key,
            "file_url": self._build_file_url(key),
            "file_size": len(image_data),
            "content_type": mime_type,
        }

    # ---------------------------------------------------------------------

    async def download_image(self, file_key: str) -> bytes:
        try:
            response = await self._run(
                self.s3_client.get_object,
                Bucket=self.bucket_name,
                Key=file_key,
            )
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise StorageError("File not found")
            raise StorageError(f"Download failed: {e}")

    # ---------------------------------------------------------------------

    async def delete_image(self, file_key: str) -> bool:
        try:
            await self._run(
                self.s3_client.delete_object,
                Bucket=self.bucket_name,
                Key=file_key,
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return False
            raise StorageError(f"Delete failed: {e}")

    # ---------------------------------------------------------------------

    async def list_files(
        self,
        prefix: str = "",
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Список файлов в хранилище.

        Args:
            prefix: Префикс для фильтрации (например, "uploads/")
            limit: Максимальное количество файлов

        Returns:
            Список словарей с информацией о файлах
        """
        try:

            def _list():
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=limit,
                )
                return response.get("Contents", [])

            files = await self._run(_list)

            result = []
            for f in files:
                result.append(
                    {
                        "key": f["Key"],
                        "size": f["Size"],
                        "last_modified": f["LastModified"],
                        "etag": f.get("ETag", ""),
                    }
                )

            return result

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                return []
            raise StorageError(f"List files failed: {e}")

    # ---------------------------------------------------------------------

    async def generate_presigned_url(
        self,
        file_key: str,
        expires_in: int = 3600,
        operation: str = "get_object",
    ) -> str:

        if operation not in {"get_object", "put_object"}:
            raise ValidationError("Invalid presigned URL operation")

        def _gen():
            return self.s3_client.generate_presigned_url(
                operation,
                Params={"Bucket": self.bucket_name, "Key": file_key},
                ExpiresIn=expires_in,
            )

        try:
            return await self._run(_gen)
        except Exception as e:
            raise StorageError(f"Presigned URL failed: {e}")

    # =============================================================================
    # Helpers
    # =============================================================================

    def _detect_image_mime(self, data: bytes) -> Tuple[str, int, int]:
        try:
            # Первое открытие для получения формата
            img = Image.open(io.BytesIO(data))
            width, height = img.size
            img_format = img.format  # Сохраняем формат ДО verify()
            img.verify()
            mime_type = Image.MIME.get(img_format, "application/octet-stream")
            return mime_type, width, height
        except UnidentifiedImageError:
            raise ValidationError("Invalid image data")
        except Exception as e:
            raise ValidationError(f"Image processing failed: {str(e)}")

    def _generate_object_key(self, mime_type: str) -> str:
        ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }.get(mime_type, ".img")

        now = datetime.now(timezone.utc)
        return (
            f"images/{now.year}/{now.month:02d}/{now.day:02d}/"
            f"{uuid.uuid4().hex}{ext}"
        )

    def _prepare_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
        result: Dict[str, str] = {}

        if not metadata:
            return result

        for key, value in metadata.items():
            k = str(key)[:128]
            v = str(value)[:1024]
            result[k] = v

        return result

    def _build_file_url(self, key: str) -> str:
        if self.endpoint_url:
            base = self.endpoint_url.rstrip("/")
            return f"{base}/{self.bucket_name}/{key}"
        return f"https://{self.bucket_name}.s3." f"{self.region}.amazonaws.com/{key}"
