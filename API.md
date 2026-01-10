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