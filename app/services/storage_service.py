"""
Сервис хранилища файлов.
Работа с S3/MinIO для загрузки, хранения и управления файлами изображений.
"""

import io
import mimetypes
from typing import Optional, Dict, Any, Tuple
import aiohttp
import aiofiles
from botocore.exceptions import ClientError, NoCredentialsError
import boto3
from botocore.config import Config

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import StorageError, ValidationError

logger = get_logger(__name__)


class StorageService:
    """
    Сервис для работы с хранилищем файлов (S3/MinIO).
    """
    
    def __init__(self):
        self.endpoint_url = settings.S3_ENDPOINT_URL
        self.access_key = settings.S3_ACCESS_KEY
        self.secret_key = settings.S3_SECRET_KEY
        self.bucket_name = settings.S3_BUCKET_NAME
        self.region = settings.S3_REGION
        self.use_ssl = settings.S3_USE_SSL
        self.public_read = bool(getattr(settings, "S3_PUBLIC_READ", False))
        
        # Инициализация S3 клиента
        self.s3_client = self._initialize_s3_client()
    
    def _initialize_s3_client(self):
        """
        Инициализация S3 клиента.
        
        Returns:
            boto3.client: S3 клиент
        """
        try:
            config = Config(
                region_name=self.region,
                signature_version='s3v4',
                retries={
                    'max_attempts': 3,
                    'mode': 'standard'
                }
            )
            
            client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=config,
                use_ssl=self.use_ssl
            )
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise StorageError(f"S3 client initialization failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Проверка состояния хранилища.
        
        Returns:
            bool: True если хранилище доступно
        """
        try:
            # Пробуем получить информацию о бакете
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Бакет не существует, пытаемся создать
                await self._create_bucket_if_not_exists()
                return True
            else:
                logger.error(f"S3 health check failed: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"S3 health check error: {str(e)}")
            return False
    
    async def _create_bucket_if_not_exists(self):
        """
        Создание бакета если он не существует.
        """
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Настройка политики доступа (публичное чтение для файлов)
            await self._setup_bucket_policy()
            
            logger.info(f"Bucket {self.bucket_name} created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create bucket {self.bucket_name}: {str(e)}")
            raise StorageError(f"Bucket creation failed: {str(e)}")
    
    async def _setup_bucket_policy(self):
        """
        Настройка политики доступа к бакету.
        """
        try:
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "PublicReadGetObject",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                    }
                ]
            }
            
            self.s3_client.put_bucket_policy(
                Bucket=self.bucket_name,
                Policy=str(policy).replace("'", '"')
            )
            
        except Exception as e:
            logger.warning(f"Failed to setup bucket policy: {str(e)}")
    
    async def upload_image(
        self,
        image_data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Загрузка изображения в хранилище.
        
        Args:
            image_data: Двоичные данные изображения
            metadata: Дополнительные метаданные
            
        Returns:
            Dict[str, Any]: Информация о загруженном файле
        """
        try:
            # Определяем MIME тип
            mime_type, _ = mimetypes.guess_type("image.jpg")
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # Генерируем уникальное имя файла
            import uuid
            import time
            from datetime import datetime
            
            file_extension = self._get_extension_from_mime_type(mime_type)
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{file_extension}"
            
            # Создаем путь с организацией по датам
            now = datetime.now()
            key = f"images/{now.year}/{now.month:02d}/{now.day:02d}/{filename}"
            
            # Подготавливаем метаданные
            s3_metadata = self._prepare_s3_metadata(metadata)
            
            # Загружаем файл
            try:
                put_kwargs = dict(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=image_data,
                    ContentType=mime_type,
                    Metadata=s3_metadata,
                )
                if self._is_public_bucket():
                    put_kwargs["ACL"] = "public-read"

                self.s3_client.put_object(**put_kwargs)
            except Exception as e:
                logger.error(f"S3 upload failed for key {key}: {str(e)}")
                raise StorageError(f"Failed to upload image: {str(e)}")
            
            # Формируем URL файла
            file_url = self._generate_file_url(key)
            
            logger.info(f"Image uploaded successfully: {key} ({len(image_data)} bytes)")
            
            return {
                "image_id": key,
                "file_url": file_url,
                "file_size": len(image_data),
                "content_type": mime_type,
                "key": key,
                "bucket": self.bucket_name,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Image upload failed: {str(e)}")
            raise StorageError(f"Failed to upload image: {str(e)}")
    
    async def download_image(self, file_key: str) -> bytes:
        """
        Скачивание изображения из хранилища.
        
        Args:
            file_key: Ключ файла в хранилище
            
        Returns:
            bytes: Двоичные данные изображения
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            image_data = response['Body'].read()
            
            logger.info(f"Image downloaded successfully: {file_key} ({len(image_data)} bytes)")
            return image_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise StorageError(f"File not found: {file_key}")
            else:
                logger.error(f"S3 download failed for key {file_key}: {str(e)}")
                raise StorageError(f"Failed to download image: {str(e)}")
        except Exception as e:
            logger.error(f"Image download failed: {str(e)}")
            raise StorageError(f"Failed to download image: {str(e)}")
    
    async def delete_image(self, file_key: str) -> bool:
        """
        Удаление изображения из хранилища.
        
        Args:
            file_key: Ключ файла для удаления
            
        Returns:
            bool: True если файл удален
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            logger.info(f"Image deleted successfully: {file_key}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.warning(f"File not found for deletion: {file_key}")
                return False
            else:
                logger.error(f"S3 delete failed for key {file_key}: {str(e)}")
                raise StorageError(f"Failed to delete image: {str(e)}")
        except Exception as e:
            logger.error(f"Image deletion failed: {str(e)}")
            raise StorageError(f"Failed to delete image: {str(e)}")
    
    async def delete_image_by_url(self, file_url: str) -> bool:
        """
        Удаление изображения по URL.
        
        Args:
            file_url: URL файла для удаления
            
        Returns:
            bool: True если файл удален
        """
        try:
            # Извлекаем ключ из URL
            file_key = self._extract_key_from_url(file_url)
            if not file_key:
                logger.warning(f"Could not extract key from URL: {file_url}")
                return False
            
            return await self.delete_image(file_key)
            
        except Exception as e:
            logger.error(f"Failed to delete image by URL {file_url}: {str(e)}")
            return False
    
    async def get_image_info(self, file_key: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о изображении.
        
        Args:
            file_key: Ключ файла
            
        Returns:
            Optional[Dict[str, Any]]: Информация о файле
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            
            metadata = response.get('Metadata', {})
            content_type = response.get('ContentType', 'application/octet-stream')
            
            return {
                "file_url": self._generate_file_url(file_key),
                "file_size": response.get('ContentLength'),
                "image_format": self._get_format_from_content_type(content_type),
                "image_dimensions": metadata.get('dimensions'),
                "created_at": response.get('LastModified'),
                "metadata": metadata
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                return None
            else:
                logger.error(f"S3 head object failed for key {file_key}: {str(e)}")
                raise StorageError(f"Failed to get image info: {str(e)}")
        except Exception as e:
            logger.error(f"Get image info failed: {str(e)}")
            raise StorageError(f"Failed to get image info: {str(e)}")
    
    async def list_images(
        self,
        prefix: Optional[str] = None,
        max_keys: int = 1000
    ) -> list:
        """
        Список изображений в хранилище.
        
        Args:
            prefix: Префикс для фильтрации
            max_keys: Максимальное количество ключей
            
        Returns:
            list: Список ключей файлов
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            keys = []
            if 'Contents' in response:
                keys = [obj['Key'] for obj in response['Contents']]
            
            logger.info(f"Listed {len(keys)} images with prefix: {prefix}")
            return keys
            
        except Exception as e:
            logger.error(f"List images failed: {str(e)}")
            raise StorageError(f"Failed to list images: {str(e)}")
    
    async def generate_presigned_url(
        self,
        file_key: str,
        expires_in: int = 3600,
        operation: str = 'get_object'
    ) -> str:
        """
        Генерация подписанного URL для доступа к файлу.
        
        Args:
            file_key: Ключ файла
            expires_in: Время жизни URL в секундах
            operation: Операция ('get_object' или 'put_object')
            
        Returns:
            str: Подписанный URL
        """
        try:
            if operation == 'get_object':
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.bucket_name, 'Key': file_key},
                    ExpiresIn=expires_in
                )
            elif operation == 'put_object':
                url = self.s3_client.generate_presigned_url(
                    'put_object',
                    Params={'Bucket': self.bucket_name, 'Key': file_key},
                    ExpiresIn=expires_in
                )
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {file_key}: {str(e)}")
            raise StorageError(f"Failed to generate presigned URL: {str(e)}")
    
    def _get_extension_from_mime_type(self, mime_type: str) -> str:
        """
        Получение расширения файла из MIME типа.
        
        Args:
            mime_type: MIME тип
            
        Returns:
            str: Расширение файла
        """
        extension_map = {
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/png': '.png',
            'image/webp': '.webp',
            'image/gif': '.gif'
        }
        return extension_map.get(mime_type, '.jpg')
    
    def _get_format_from_content_type(self, content_type: str) -> str:
        """
        Получение формата изображения из Content-Type.
        
        Args:
            content_type: Content-Type заголовок
            
        Returns:
            str: Формат изображения
        """
        if 'jpeg' in content_type.lower() or 'jpg' in content_type.lower():
            return 'JPEG'
        elif 'png' in content_type.lower():
            return 'PNG'
        elif 'webp' in content_type.lower():
            return 'WEBP'
        elif 'gif' in content_type.lower():
            return 'GIF'
        else:
            return 'UNKNOWN'
    
    def _prepare_s3_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """
        Подготовка метаданных для S3.
        
        Args:
            metadata: Метаданные для подготовки
            
        Returns:
            Dict[str, str]: Подготовленные метаданные
        """
        s3_metadata = {}
        
        if metadata:
            # Преобразуем все значения в строки для S3
            for key, value in metadata.items():
                if key.lower() in ['cache-control', 'content-type', 'content-encoding', 'content-disposition']:
                    continue  # Пропускаем заголовки HTTP
                
                # Ограничиваем длину ключей и значений
                str_key = str(key)[:128]
                str_value = str(value)[:2048]
                
                s3_metadata[str_key] = str_value
        
        return s3_metadata
    
    def _generate_file_url(self, file_key: str) -> str:
        """
        Генерация URL файла.
        
        Args:
            file_key: Ключ файла
            
        Returns:
            str: URL файла
        """
        if self.endpoint_url:
            # Для S3 совместимых сервисов
            base_url = self.endpoint_url.rstrip('/')
            if base_url.endswith(self.bucket_name):
                return f"{base_url}/{file_key}"
            else:
                return f"{base_url}/{self.bucket_name}/{file_key}"
        else:
            # Стандартный S3 URL
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_key}"
    
    def _extract_key_from_url(self, file_url: str) -> Optional[str]:
        """
        Извлечение ключа файла из URL.
        
        Args:
            file_url: URL файла
            
        Returns:
            Optional[str]: Ключ файла или None
        """
        try:
            # Простая логика извлечения ключа из URL
            if self.bucket_name in file_url:
                parts = file_url.split(f"{self.bucket_name}/")
                if len(parts) > 1:
                    return parts[1]
            return None
            
        except Exception:
            return None
    
    def _is_public_bucket(self) -> bool:
        """
        Проверка, является ли бакет публичным.
        
        Returns:
            bool: True если бакет публичный
        """
        # Логика определения публичности бакета
        return self.public_read
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Получение статистики хранилища.
        
        Returns:
            Dict[str, Any]: Статистика хранилища
        """
        try:
            # Получаем информацию о бакete
            response = self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Получаем список объектов для подсчета
            objects_response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            
            return {
                "bucket_name": self.bucket_name,
                "region": self.region,
                "endpoint_url": self.endpoint_url,
                "total_objects": objects_response.get('KeyCount', 0),
                "is_public": self._is_public_bucket(),
                "status": "accessible"
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {
                "bucket_name": self.bucket_name,
                "status": "error",
                "error": str(e)
            }