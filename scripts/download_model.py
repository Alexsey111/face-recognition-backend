"""
Скрипт для загрузки предобученной модели MiniFASNetV2.

Использование:
    python scripts/download_model.py

Параметры:
    --force     Перезаписать существующий файл
    --verify    Проверить MD5 хеш (если известен)
    --md5       Ожидаемый MD5 хеш для проверки
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

# Добавляем путь к проекту
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings


# Конфигурация
MODEL_URL = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
MODEL_PATH = PROJECT_ROOT / "models" / "minifasnet_v2.pth"
EXPECTED_MD5 = None  # Установите если известен хеш файла


def get_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """Вычисление хеша файла."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_with_progress(url: str, filepath: Path, desc: str = "Download") -> bool:
    """
    Загрузка файла с отображением прогресса.

    Args:
        url: URL для загрузки
        filepath: Путь для сохранения
        desc: Описание для прогресс-бара

    Returns:
        True если успешно, False иначе
    """
    import urllib.request

    class ProgressTracker:
        def __init__(self):
            self.last_percent = -1

        def __call__(self, block_num, block_size, total_size):
            percent = min(100, (block_num * block_size * 100) // total_size)
            if percent != self.last_percent:
                print(f"\r{desc}: {percent}%", end="", flush=True)
                self.last_percent = percent

    try:
        print(f"\n{'='*60}")
        print(f"Загрузка модели из: {url}")
        print(f"Сохранение в: {filepath}")
        print(f"{'='*60}\n")

        filepath.parent.mkdir(parents=True, exist_ok=True)

        urlretrieve(url, filepath, ProgressTracker())

        print(f"\n\n✓ Загрузка завершена!")
        return True

    except HTTPError as e:
        print(f"\n\n✗ Ошибка HTTP: {e.code} - {e.reason}")
        return False
    except URLError as e:
        print(f"\n\n✗ Ошибка URL: {e.reason}")
        return False
    except KeyboardInterrupt:
        print("\n\n✗ Загрузка отменена пользователем")
        if filepath.exists():
            filepath.unlink()
        return False
    except Exception as e:
        print(f"\n\n✗ Неизвестная ошибка: {e}")
        return False


def verify_model_file(filepath: Path) -> dict:
    """
    Проверка загруженного файла модели.

    Returns:
        dict с информацией о файле
    """
    if not filepath.exists():
        return {"valid": False, "reason": "Файл не найден"}

    result = {
        "valid": True,
        "path": filepath,
        "size_bytes": filepath.stat().st_size,
        "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
    }

    # Проверка расширения
    if filepath.suffix.lower() == ".pth":
        result["extension"] = "valid"
    else:
        result["extension"] = "warning"
        result["reason"] = "Ожидается .pth расширение"

    # Вычисление хешей
    result["md5"] = get_file_hash(filepath, "md5")
    result["sha256"] = get_file_hash(filepath, "sha256")

    # Проверка MD5 если задан
    if EXPECTED_MD5:
        result["md5_valid"] = result["md5"] == EXPECTED_MD5
        if result["md5_valid"]:
            result["reason"] = "MD5 хеш подтвержден"
        else:
            result["valid"] = False
            result["reason"] = "MD5 хеш НЕ совпадает!"

    # Проверка размера (MiniFASNetV2 ~ 1.6 MB)
    expected_size_mb = 1.6
    if result["size_mb"] < 0.5:
        result["valid"] = False
        result["reason"] = "Файл слишком маленький, возможно ошибка загрузки"
    elif result["size_mb"] > 10:
        result["valid"] = False
        result["reason"] = "Файл слишком большой"

    return result


def download_model(
    force: bool = False,
    md5_check: str = None,
) -> bool:
    """
    Загрузка модели.

    Args:
        force: Перезаписать существующий файл
        md5_check: Ожидаемый MD5 хеш

    Returns:
        True если успешно
    """
    global EXPECTED_MD5
    EXPECTED_MD5 = md5_check

    # Проверка существующего файла
    if MODEL_PATH.exists():
        if not force:
            print(f"\n✓ Модель уже существует: {MODEL_PATH}")

            # Верификация
            verify_result = verify_model_file(MODEL_PATH)

            print(f"\n{'='*60}")
            print("Проверка файла:")
            print(f"  Размер: {verify_result['size_mb']} MB")
            print(f"  MD5: {verify_result['md5']}")
            print(f"  SHA256: {verify_result['sha256'][:16]}...")
            print(f"  Статус: {'✓ Валидный' if verify_result['valid'] else '✗ Невалидный'}")
            print(f"{'='*60}\n")

            if verify_result.get("md5_valid"):
                print("MD5 хеш подтвержден!")
                return True
            else:
                print("⚠ Файл существует, но не прошел проверку.")
                print("Используйте --force для перезагрузки.\n")
                return True

        print(f"\n⚠ Перезапись существующего файла: {MODEL_PATH}")
        MODEL_PATH.unlink()

    # Загрузка
    success = download_with_progress(MODEL_URL, MODEL_PATH, "Загрузка")

    if success and MODEL_PATH.exists():
        print("\n" + "="*60)
        print("Проверка загруженного файла:")
        print("="*60)

        verify_result = verify_model_file(MODEL_PATH)

        print(f"\n  Путь: {verify_result['path']}")
        print(f"  Размер: {verify_result['size_mb']} MB")
        print(f"  MD5: {verify_result['md5']}")
        print(f"  SHA256: {verify_result['sha256'][:16]}...")

        if verify_result["valid"]:
            print("\n✓ Модель успешно загружена и проверена!")
            return True
        else:
            print(f"\n✗ Внимание: {verify_result['reason']}")
            return False

    return False


def main():
    """Точка входа."""
    parser = argparse.ArgumentParser(
        description="Загрузка предобученной модели MiniFASNetV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Загрузка модели
  python scripts/download_model.py

  # Принудительная перезагрузка
  python scripts/download_model.py --force

  # С проверкой MD5
  python scripts/download_model.py --verify --md5 abc123...
        """
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Перезаписать существующий файл",
    )

    parser.add_argument(
        "--verify",
        "-v",
        action="store_true",
        help="Проверить MD5 хеш после загрузки",
    )

    parser.add_argument(
        "--md5",
        type=str,
        help="Ожидаемый MD5 хеш для проверки",
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("MiniFASNetV2 Model Downloader")
    print("="*60)
    print(f"\nКонфигурация:")
    print(f"  URL: {MODEL_URL}")
    print(f"  Путь: {MODEL_PATH}")
    print(f"  Force: {args.force}")
    print(f"  Verify: {args.verify}")
    if args.md5:
        print(f"  MD5: {args.md5}")
    print()

    success = download_model(
        force=args.force,
        md5_check=args.md5 if args.verify else None,
    )

    if success:
        print("\n✓ Готово! Модель доступна по пути:")
        print(f"  {MODEL_PATH}")
        print("\nДля использования в приложении убедитесь, что:")
        print("  1. USE_CERTIFIED_LIVENESS=true в .env")
        print("  2. CERTIFIED_LIVENESS_MODEL_PATH указан верно")
        print()
    else:
        print("\n✗ Ошибка загрузки модели!")
        sys.exit(1)


if __name__ == "__main__":
    main()