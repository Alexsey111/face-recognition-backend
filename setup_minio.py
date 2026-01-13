#!/usr/bin/env python3
"""
Скрипт инициализации MinIO/S3 хранилища.

Создаёт необходимые bucket'ы для хранения изображений лиц.
Используется при первом запуске системы или для проверки конфигурации.

Пример запуска:
    python setup_minio.py

Переменные окружения (могут быть переопределены в .env):
    S3_ENDPOINT_URL - URL MinIO сервера (по умолчанию: localhost:9000)
    S3_ACCESS_KEY   - Access key для аутентификации
    S3_SECRET_KEY   - Secret key для аутентификации
    S3_BUCKET_NAME  - Имя bucket'а для изображений (по умолчанию: face-images)
    S3_SECURE       - Использовать HTTPS (по умолчанию: False)
"""

import os
import sys
from pathlib import Path

# Добавляем путь к приложению для импортов
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from minio import Minio
    from minio.error import S3Error, BucketAlreadyOwnedByYou, BucketAlreadyExists
except ImportError:
    print("Ошибка: minio library не установлена.")
    print("Установите зависимости: pip install minio")
    sys.exit(1)

from app.config import Settings


def get_minio_client(settings: Settings) -> Minio:
    """
    Создаёт и возвращает клиент MinIO.
    
    Args:
        settings: Объект настроек приложения
        
    Returns:
        Настроенный клиент MinIO
    """
    return Minio(
        endpoint=settings.S3_ENDPOINT_URL.replace("http://", "").replace("https://", ""),
        access_key=settings.S3_ACCESS_KEY,
        secret_key=settings.S3_SECRET_KEY,
        secure="https" in settings.S3_ENDPOINT_URL.lower()
    )


def create_bucket_if_not_exists(client: Minio, bucket_name: str) -> bool:
    """
    Создаёт bucket, если он не существует.
    
    Args:
        client: MinIO клиент
        bucket_name: Имя bucket'а
        
    Returns:
        True если bucket создан или уже существует, False в случае ошибки
    """
    try:
        # Проверяем, существует ли bucket
        if client.bucket_exists(bucket_name):
            print(f"✅ Bucket '{bucket_name}' уже существует")
            return True
        
        # Создаём bucket
        client.make_bucket(bucket_name)
        print(f"✅ Bucket '{bucket_name}' успешно создан")
        return True
        
    except BucketAlreadyOwnedByYou:
        print(f"✅ Bucket '{bucket_name}' уже принадлежит вам")
        return True
        
    except BucketAlreadyExists:
        print(f"✅ Bucket '{bucket_name}' уже существует")
        return True
        
    except S3Error as e:
        print(f"❌ Ошибка при создании bucket '{bucket_name}': {e}")
        return False


def setup_minio_buckets(settings: Settings) -> bool:
    """
    Настраивает все необходимые bucket'ы для приложения.
    
    Args:
        settings: Объект настроек приложения
        
    Returns:
        True если все bucket'ы созданы, False в случае ошибки
    """
    print("🔧 Инициализация MinIO хранилища...")
    print(f"   Endpoint: {settings.S3_ENDPOINT_URL}")
    print(f"   Bucket:   {settings.S3_BUCKET_NAME}")
    print("-" * 50)
    
    try:
        client = get_minio_client(settings)
        
        # Создаём основной bucket для изображений
        success = create_bucket_if_not_exists(client, settings.S3_BUCKET_NAME)
        
        if success:
            print("-" * 50)
            print("✅ Инициализация MinIO завершена успешно")
        else:
            print("-" * 50)
            print("❌ Инициализация MinIO завершена с ошибками")
            
        return success
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return False


def verify_minio_connection(settings: Settings) -> bool:
    """
    Проверяет соединение с MinIO сервером.
    
    Args:
        settings: Объект настроек приложения
        
    Returns:
        True если соединение успешно, False в случае ошибки
    """
    print("🔍 Проверка соединения с MinIO...")
    
    try:
        client = get_minio_client(settings)
        
        # Пытаемся получить информацию о сервере (list_buckets требует аутентификации)
        buckets = client.list_buckets()
        
        print(f"✅ Соединение успешно. Доступно bucket'ов: {len(buckets)}")
        for bucket in buckets:
            print(f"   - {bucket.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False


def main():
    """
    Основная функция запуска скрипта.
    
    Поддерживает аргументы командной строки:
        --verify   - только проверить соединение
        --setup    - создать bucket'ы (по умолчанию)
        --help     - показать справку
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Инициализация MinIO для Face Recognition Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
    python setup_minio.py              # Создать bucket'ы
    python setup_minio.py --verify     # Проверить соединение
    python setup_minio.py --setup      # Явно создать bucket'ы
        """
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Только проверить соединение с MinIO"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Создать bucket'ы (действие по умолчанию)"
    )
    
    args = parser.parse_args()
    
    # Загружаем настройки
    print("📋 Загрузка конфигурации...")
    try:
        settings = Settings()
        print(f"   Environment: {settings.ENVIRONMENT}")
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        print("   Убедитесь, что файл .env создан и содержит необходимые переменные")
        sys.exit(1)
    
    # Выполняем запрошенное действие
    if args.verify:
        success = verify_minio_connection(settings)
    else:
        # По умолчанию создаём bucket'ы
        success = setup_minio_buckets(settings)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
