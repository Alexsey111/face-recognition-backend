# API Reference

## Overview

Face Recognition API предоставляет REST endpoints для:
- Загрузки изображений
- Управления эталонными изображениями
- Верификации личности
- Проверки живости (Liveness Detection)
- Сертифицированной Anti-Spoofing проверки

## Base URL

```
http://localhost:8000/api/v1
```

## Endpoints

### Authentication
- `POST /api/v1/auth/login` - Аутентификация пользователя
- `POST /api/v1/auth/refresh` - Обновление токена

### Upload
- `POST /api/v1/upload` - Создание сессии загрузки
- `POST /api/v1/upload/{session_id}/file` - Загрузка файла в сессию
- `GET /api/v1/upload/{session_id}` - Получение статуса сессии

### Reference
- `POST /api/v1/reference` - Создание эталона
- `GET /api/v1/reference` - Список эталонов
- `GET /api/v1/reference/{reference_id}` - Получение эталона
- `PUT /api/v1/reference/{reference_id}` - Обновление эталона
- `DELETE /api/v1/reference/{reference_id}` - Удаление эталона

### Verification
- `POST /api/v1/verify` - Верификация лица

### Liveness Detection (Anti-Spoofing)
- `POST /api/v1/liveness` - Проверка живости
- `POST /api/v1/liveness/session/{session_id}` - Проверка в сессии

### Health & Metrics
- `GET /api/v1/health` - Health check
- `GET /metrics` - Prometheus метрики

---

## Liveness Detection API

### POST /api/v1/liveness

Проверка живости лица с использованием **сертифицированной модели MiniFASNetV2**.

#### Request Body

```json
{
  "image_data": "base64_encoded_image",
  "challenge_type": "passive",
  "return_features": false
}
```

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| image_data | string | Да | Base64-кодированное изображение (JPEG/PNG) |
| challenge_type | string | Нет | Тип проверки: `passive` (по умолчанию) |
| return_features | boolean | Нет | Возвращать дополнительные признаки |

#### Response

```json
{
  "success": true,
  "liveness_detected": true,
  "confidence": 0.985,
  "real_probability": 0.985,
  "spoof_probability": 0.015,
  "spoof_score": 0.015,
  "threshold": 0.98,
  "model_version": "MiniFASNetV2-certified",
  "model_type": "MiniFASNetV2",
  "accuracy_claim": ">98%",
  "processing_time": 0.125,
  "face_detected": true,
  "liveness_type": "certified_anti_spoofing",
  "spoof_type": "unknown"
}
```

#### Поля ответа

| Поле | Тип | Описание |
|------|-----|----------|
| liveness_detected | boolean | Обнаружено живое лицо (true) или подделка (false) |
| confidence | float | Уверенность в результате (0.0 - 1.0) |
| real_probability | float | Вероятность, что это реальное лицо |
| spoof_probability | float | Вероятность, что это подделка |
| spoof_score | float | Оценка спуфинга (1.0 - spoof, 0.0 - real) |
| model_version | string | Версия модели: `MiniFASNetV2-certified` |
| accuracy_claim | string | Заявленная точность: `>98%` |
| liveness_type | string | Тип проверки: `certified_anti_spoofing` |

### Активная проверка живости

#### Blink Detection

```json
{
  "image_data": "base64...",
  "challenge_type": "blink"
}
```

#### Smile Detection

```json
{
  "image_data": "base64...",
  "challenge_type": "smile"
}
```

#### Head Turn Detection

```json
{
  "image_data": "base64...",
  "challenge_type": "turn_head",
  "challenge_data": {
    "rotation_y": 30
  }
}
```

---

## Anti-Spoofing Capabilities

### Поддерживаемые типы атак

| Тип атаки | Описание | Защита |
|-----------|----------|--------|
| Print Attack | Распечатанная фотография | ✅ Защищено |
| Replay Attack | Видео на экране | ✅ Защищено |
| Digital Attack | Цифровое изображение | ✅ Защищено |
| Mask Attack | Маска на лице | ⚠️ Частично |
| Deepfake | Синтезированное видео | ⚠️ Требуется дообучение |

### Модели

| Модель | Тип | Точность | Сертифицирована |
|--------|-----|----------|-----------------|
| MiniFASNetV2 | Passive Liveness | >98% | ✅ Да |
| Heuristic | Passive Liveness | ~85% | ❌ Нет |
| Blink Detection | Active Liveness | ~90% | ❌ Нет |

---

## Configuration

### Environment Variables

```bash
# Liveness Settings
USE_CERTIFIED_LIVENESS=true           # Использовать сертифицированную модель
CERTIFIED_LIVENESS_MODEL_PATH=/path/to/model.pth  # Путь к модели
CERTIFIED_LIVENESS_THRESHOLD=0.98     # Порог принятия решения
```

### Thresholds

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| LIVENESS_THRESHOLD | 0.98 | Минимальная уверенность для признания лица живым |
| LIVENESS_CONFIDENCE_THRESHOLD | 0.7 | Порог для дополнительных проверок |

---

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": "Error message",
  "error_code": "ERROR_CODE"
}
```

### Common Error Codes

| Код | Описание |
|-----|----------|
| NO_FACE_DETECTED | Лицо не найдено на изображении |
| INVALID_IMAGE | Некорректный формат изображения |
| LIVENESS_CHECK_FAILED | Ошибка проверки живости |
| MODEL_NOT_LOADED | Модель не загружена |

---

## Rate Limiting

- 60 запросов в минуту на endpoint
- Burst: 10 запросов

---

## OpenAPI Documentation

Интерактивная документация доступна по адресу: `/docs`

---

## Version

API Version: 1.0.0

Last Updated: 2025-01-06

---

## Поддержка форматов изображений

### Поддерживаемые форматы

Сервис поддерживает следующие форматы изображений:

| Формат | Расширения | Описание | Автоконвертация |
|--------|-----------|----------|----------------|
| JPEG   | .jpg, .jpeg | Стандартный формат | - |
| PNG    | .png | Формат с поддержкой прозрачности | - |
| **HEIC** | **.heic, .heif** | **Apple формат (iOS/macOS)** | **✓ Да → JPEG** |
| WebP   | .webp | Google формат | - |

### HEIC (High Efficiency Image Container)

HEIC — современный формат сжатия изображений, используемый Apple начиная с iOS 11.

**Особенности обработки:**
- ✅ Автоматическое определение формата по magic bytes
- ✅ Конвертация в JPEG с сохранением качества (95%)
- ✅ Обработка прозрачности (конвертация в RGB с белым фоном)
- ✅ Поддержка HEIF (вариант HEIC)
- ⚠️ Небольшое увеличение времени обработки (~100-200ms)

### Эндпоинты для работы с изображениями

#### POST /api/v1/upload/validate

Валидация загруженного изображения.

```bash
curl -X POST "http://localhost:8000/api/v1/upload/validate" \
  -F "file=@photo.heic"
```

**Response:**

```json
{
  "status": "success",
  "message": "Изображение валидно",
  "image_info": {
    "width": 3024,
    "height": 4032,
    "format": "HEIC",
    "mime_type": "image/heic",
    "size_mb": 2.45
  },
  "converted_from_heic": true,
  "dimensions": "3024x4032"
}
```

#### GET /api/v1/upload/supported-formats

Получение списка поддерживаемых форматов.

```bash
curl "http://localhost:8000/api/v1/upload/supported-formats"
```

**Response:**

```json
{
  "supported_extensions": [".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"],
  "max_file_size_mb": 10,
  "max_dimension": 4096,
  "min_dimension": 160,
  "formats": {
    "JPEG": {
      "extensions": [".jpg", ".jpeg"],
      "description": "Стандартный формат"
    },
    "PNG": {
      "extensions": [".png"],
      "description": "Формат с поддержкой прозрачности"
    },
    "HEIC": {
      "extensions": [".heic", ".heif"],
      "description": "Apple формат (автоконвертация в JPEG)"
    },
    "WebP": {
      "extensions": [".webp"],
      "description": "Google формат"
    }
  }
}
```

#### POST /api/v1/upload/convert-heic

Конвертация HEIC/HEIF в JPEG.

```bash
curl -X POST "http://localhost:8000/api/v1/upload/convert-heic" \
  -F "file=@photo.heic"
```

**Response:**

```json
{
  "status": "success",
  "message": "Конвертация завершена",
  "jpeg_size_bytes": 524288,
  "jpeg_size_mb": 0.5,
  "dimensions": "1920x1080",
  "jpeg_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD..."
}
```

### Ограничения

| Параметр | Значение |
|----------|----------|
| Максимальный размер файла | 10 MB |
| Максимальное разрешение | 4096x4096 px |
| Минимальное разрешение | 160x160 px (для распознавания лиц) |
| Качество JPEG при конвертации | 95% |

### Технические детали

#### Определение формата

Формат определяется по **magic bytes** (сигнатуре файла):

| Формат | Magic Bytes |
|--------|-------------|
| JPEG   | `FF D8 FF E0` или `FF D8 FF E1` |
| PNG    | `89 50 4E 47 0D 0A 1A 0A` |
| HEIC   | `ftypheic` или `ftypmif1` (смещение 4) |
| HEIF   | `ftypheif` (смещение 4) |
| WebP   | `RIFF....WEBP` |

#### Конвертация HEIC → JPEG

```python
# Автоматическая конвертация при загрузке
def convert_heic_to_jpeg(file_data: bytes, quality: int = 95) -> bytes:
    image = Image.open(io.BytesIO(file_data))
    # Обработка прозрачности
    if image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    # Сохранение в JPEG
    return image_to_bytes(image, format='JPEG', quality=quality)
```

### Интеграция в другие эндпоинты

HEIC файлы автоматически конвертируются в следующих эндпоинтах:

- `POST /api/v1/upload/{session_id}/file` - Загрузка файла
- `POST /api/v1/verify` - Верификация лица
- `POST /api/v1/liveness` - Проверка живости
- `POST /api/v1/reference` - Создание эталона

---

## Используемые модели машинного обучения

### Краткий обзор

Сервис использует следующие модели машинного обучения в едином pipeline обработки:

| Задача | Модель | Точность | Скорость (GPU) |
|--------|--------|----------|----------------|
| Face Detection | MTCNN | 95% precision | 10ms |
| Face Verification | FaceNet (Inception-ResNet-v1) | 99.65% (LFW) | 5ms |
| Anti-Spoofing | MiniFASNetV2 | 98.95% (ACER) | 2ms |

**Полная документация:** См. [MODELS.md](./docs/MODELS.md)

### Pipeline обработки

```
Input Image → Face Detection (MTCNN) → Face Alignment → 
├─→ Anti-Spoofing (MiniFASNetV2) → is_live
└─→ Embedding Extraction (FaceNet) → 512-dim vector → Comparison
```

### Конфигурация моделей

Параметры моделей настраиваются через `config.py`:

```python
# Face Verification
VERIFICATION_THRESHOLD = 0.6  # Standard (FAR < 0.1%, FRR < 1-3%)
FACENET_VERSION = "1.0.0"
EMBEDDING_SIZE = 512

# Liveness Detection
LIVENESS_THRESHOLD = 0.5  # Balanced
MINIFASNET_VERSION = "2.0.1"

# Face Detection
FACE_DETECTOR = "mtcnn"  # or "retinaface"
MIN_FACE_SIZE = 20
```

### Метрики производительности

#### Latency breakdown (полный pipeline, 1024×1024 image)

| Этап | CPU (ms) | GPU (ms) |
|------|----------|----------|
| Image loading & decode | 10 | 10 |
| Face detection (MTCNN) | 100 | 10 |
| Face alignment | 5 | 5 |
| Liveness check (MiniFASNet) | 15 | 2 |
| Embedding extraction (FaceNet) | 50 | 5 |
| Comparison | 1 | 0.5 |
| **Total** | **~201ms** | **~53ms** |

#### Throughput

| Конфигурация | Запросов/сек |
|--------------|--------------|
| Single CPU | 5 |
| Single GPU | 18 |
| 10 pods (GPU) | ~170 |

### Модели в ответах API

#### Verification Response

```json
{
  "verification_id": "ver_123456",
  "is_match": true,
  "similarity_score": 0.89,
  "confidence": 0.95,
  "threshold_used": 0.7,
  "model_info": {
    "face_detector": "mtcnn",
    "embedder": "facenet",
    "liveness_enabled": true
  }
}
```

#### Liveness Response

```json
{
  "liveness_id": "live_789xyz",
  "is_live": true,
  "liveness_score": 0.94,
  "confidence": 0.97,
  "model": "MiniFASNetV2",
  "spoof_type": null
}
```

### Health Check

Эндпоинт `/api/v1/health` возвращает статус моделей:

```json
{
  "status": "healthy",
  "services": {
    "ml_models": "healthy"
  },
  "model_versions": {
    "facenet": "1.0.0",
    "minifasnet": "2.0.1",
    "mtcnn": "1.0.0"
  }
}
```

### Метрики Prometheus

```prometheus
# HELP model_inference_seconds Model inference latency
# TYPE model_inference_seconds histogram
model_inference_seconds_bucket{model="facenet",le="0.01"} 1523
model_inference_seconds_bucket{model="facenet",le="0.05"} 1890
model_inference_seconds_bucket{model="minifasnet",le="0.005"} 1450

# HELP verifications_total Total verifications
verifications_total{model="facenet"} 5234

# HELP liveness_checks_total Total liveness checks
liveness_checks_total{model="minifasnet"} 4123
```