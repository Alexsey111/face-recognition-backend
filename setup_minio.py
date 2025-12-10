"""
Скрипт для настройки MinIO bucket для Face Recognition Service.
Создает необходимые buckets и настраивает политики доступа.
"""

import os
import sys
from minio import Minio
from minio.error import S3Error
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_minio_buckets():
    """Создание необходимых buckets в MinIO."""
    
    # Получение настроек из переменных окружения
    minio_endpoint = os.getenv('S3_ENDPOINT_URL', 'localhost:9000').replace('http://', '').replace('https://', '')
    access_key = os.getenv('S3_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('S3_SECRET_KEY', 'minioadmin')
    bucket_name = os.getenv('S3_BUCKET_NAME', 'face-recognition')
    
    # Создание клиента MinIO
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # Для локальной разработки
    )
    
    try:
        logger.info(f"Подключение к MinIO: {minio_endpoint}")
        
        # Список buckets для создания
        buckets_to_create = [
            bucket_name,
            f"{bucket_name}-temp",
            f"{bucket_name}-processed",
            f"{bucket_name}-thumbnails"
        ]
        
        for bucket in buckets_to_create:
            try:
                # Проверка существования bucket
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
                    logger.info(f"✅ Bucket '{bucket}' создан успешно")
                else:
                    logger.info(f"ℹ️  Bucket '{bucket}' уже существует")
            except S3Error as e:
                logger.error(f"❌ Ошибка при создании bucket '{bucket}': {e}")
        
        # Настройка политик для основного bucket
        try:
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "PublicReadGetObject",
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                    },
                    {
                        "Sid": "UserFullAccess",
                        "Effect": "Allow",
                        "Principal": {"AWS": [access_key]},
                        "Action": ["s3:*"],
                        "Resource": [
                            f"arn:aws:s3:::{bucket_name}",
                            f"arn:aws:s3:::{bucket_name}/*"
                        ]
                    }
                ]
            }
            
            # Применение политики (может не работать в некоторых версиях MinIO)
            try:
                policy_json = str(bucket_policy).replace("'", '"')
                client.set_bucket_policy(bucket_name, policy_json)
                logger.info(f"✅ Политика доступа для bucket '{bucket_name}' настроена")
            except Exception as e:
                logger.warning(f"⚠️  Не удалось настроить политику для bucket '{bucket_name}': {e}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при настройке политик: {e}")
        
        # Создание папок внутри bucket
        folders_to_create = [
            "references/",
            "uploads/",
            "temp/",
            "thumbnails/",
            "exports/",
            "backups/"
        ]
        
        for folder in folders_to_create:
            try:
                # Создание объекта-папки (пустой файл с суффиксом /)
                client.put_object(
                    bucket_name,
                    folder,
                    data=b'',
                    length=0,
                    content_type='application/x-directory'
                )
                logger.info(f"✅ Папка '{folder}' создана в bucket '{bucket_name}'")
            except S3Error as e:
                if "Object already exists as a directory" in str(e):
                    logger.info(f"ℹ️  Папка '{folder}' уже существует")
                else:
                    logger.warning(f"⚠️  Ошибка при создании папки '{folder}': {e}")
        
        logger.info("🎉 Настройка MinIO завершена успешно!")
        return True
        
    except S3Error as e:
        logger.error(f"❌ Ошибка подключения к MinIO: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}")
        return False


def verify_minio_setup():
    """Проверка настройки MinIO."""
    
    minio_endpoint = os.getenv('S3_ENDPOINT_URL', 'localhost:9000').replace('http://', '').replace('https://', '')
    access_key = os.getenv('S3_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('S3_SECRET_KEY', 'minioadmin')
    bucket_name = os.getenv('S3_BUCKET_NAME', 'face-recognition')
    
    client = Minio(
        minio_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    
    try:
        logger.info("🔍 Проверка настройки MinIO...")
        
        # Проверка подключения
        client.list_buckets()
        logger.info("✅ Подключение к MinIO успешно")
        
        # Проверка bucket
        if client.bucket_exists(bucket_name):
            logger.info(f"✅ Bucket '{bucket_name}' доступен")
            
            # Проверка объектов в bucket
            objects = list(client.list_objects(bucket_name, recursive=True))
            logger.info(f"ℹ️  В bucket '{bucket_name}' найдено {len(objects)} объектов")
            
            return True
        else:
            logger.error(f"❌ Bucket '{bucket_name}' не найден")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке MinIO: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Настройка MinIO для Face Recognition Service")
    print("=" * 50)
    
    # Проверка переменных окружения
    required_vars = ['S3_ENDPOINT_URL', 'S3_ACCESS_KEY', 'S3_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Отсутствуют переменные окружения: {', '.join(missing_vars)}")
        logger.info("💡 Убедитесь, что .env файл настроен правильно")
        sys.exit(1)
    
    # Создание buckets
    if create_minio_buckets():
        logger.info("✅ Создание buckets завершено")
        
        # Проверка настройки
        if verify_minio_setup():
            logger.info("✅ Проверка настройки завершена")
            print("\n🎉 MinIO настроен успешно!")
            print(f"📊 Endpoint: {os.getenv('S3_ENDPOINT_URL')}")
            print(f"🪣 Bucket: {os.getenv('S3_BUCKET_NAME')}")
            print(f"🔑 Access Key: {os.getenv('S3_ACCESS_KEY')}")
        else:
            logger.error("❌ Проверка настройки не пройдена")
            sys.exit(1)
    else:
        logger.error("❌ Создание buckets не удалось")
        sys.exit(1)