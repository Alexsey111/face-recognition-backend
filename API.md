# API Documentation - Face Recognition Service

![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-blue.svg)

Полная документация API для сервиса распознавания лиц. Все endpoints поддерживают асинхронную обработку и возвращают структурированные ответы.

## 📋 Содержание

- [Аутентификация](#аутентификация)
- [Общие ответы](#общие-ответы)
- [Health Check](#health-check)
- [Reference Management](#reference-management)
- [Verification](#verification)
- [Liveness Detection](#liveness-detection)
- [Upload](#upload)
- [Admin Endpoints](#admin-endpoints)
- [Webhooks](#webhooks)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [SDK Examples](#sdk-examples)

## 🔐 Аутентификация

### API Key Authentication

Большинство endpoints требуют API ключ в заголовке:

```http
Authorization: Bearer YOUR_API_KEY
```

### JWT Token Authentication

Для некоторых endpoints можно использовать JWT токены:

```http
Authorization: Bearer YOUR_JWT_TOKEN
```

### Получение API ключа

```http
POST /api/v1/auth/api-key
Content-Type: application/json

{
  "user_id": "user_uuid",
  "name": "My API Key",
  "permissions": ["read", "write"]
}
```

**Ответ:**

```json
{
  "api_key": "fr_api_1234567890abcdef",
  "key_id": "key_uuid",
  "name": "My API Key",
  "permissions": ["read", "write"],
  "created_at": "2024-01-01T12:00:00Z",
  "expires_at": "2025-01-01T12:00:00Z"
}
```

## 📊 Общие ответы

### Success Response

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {
    // Response data
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456789"
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "image",
      "reason": "File size exceeds 10MB limit"
    }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456789"
}
```

## 🏥 Health Check

### Basic Health Check

```http
GET /health
```

**Ответ:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "storage": "healthy"
  }
}
```

### Detailed Health Check

```http
GET /health/detailed
```

**Ответ:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "services": {
    "database": {
      "status": "healthy",
      "response_time": 5,
      "connections": 8,
      "max_connections": 100
    },
    "redis": {
      "status": "healthy",
      "response_time": 2,
      "memory_usage": "45MB",
      "connected_clients": 3
    },
    "storage": {
      "status": "healthy",
      "response_time": 10,
      "bucket_size": "1.2GB",
      "objects_count": 150
    }
  },
  "metrics": {
    "requests_per_minute": 45,
    "average_response_time": 85,
    "error_rate": 0.02
  }
}
```

### Readiness Check

```http
GET /health/ready
```

**Ответ:**

```json
{
  "ready": true,
  "checks": {
    "database": true,
    "redis": true,
    "storage": true,
    "ml_service": true
  }
}
```

## 📸 Reference Management

### Upload Reference Image

Загрузка эталонного изображения для пользователя.

```http
POST /api/v1/reference
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data

image: (binary) # JPEG, PNG, WEBP (max 10MB)
label: "john_doe" # Опционально
user_id: "user_uuid" # Опционально
metadata: {"source": "passport"} # Опционально
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "reference_id": "ref_123456789",
    "user_id": "user_uuid",
    "label": "john_doe",
    "file_url": "https://storage.example.com/references/ref_123456789.jpg",
    "file_size": 2048576,
    "image_format": "JPEG",
    "image_dimensions": {
      "width": 1920,
      "height": 1080
    },
    "quality_score": 0.85,
    "embedding_version": 1,
    "created_at": "2024-01-01T12:00:00Z",
    "metadata": {
      "source": "passport",
      "processing_time": 0.125
    }
  }
}
```

### Get Reference

Получение информации об эталонном изображении.

```http
GET /api/v1/reference/{reference_id}
Authorization: Bearer YOUR_API_KEY
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "reference_id": "ref_123456789",
    "user_id": "user_uuid",
    "label": "john_doe",
    "file_url": "https://storage.example.com/references/ref_123456789.jpg",
    "quality_score": 0.85,
    "is_active": true,
    "usage_count": 5,
    "last_used": "2024-01-01T11:30:00Z",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

### List References

Получение списка эталонных изображений пользователя.

```http
GET /api/v1/reference?user_id=user_uuid&limit=20&offset=0
Authorization: Bearer YOUR_API_KEY
```

**Параметры:**
- `user_id` (string, optional): Фильтр по пользователю
- `limit` (int, default: 20): Количество записей
- `offset` (int, default: 0): Смещение для пагинации
- `active_only` (bool, default: true): Только активные записи

**Ответ:**

```json
{
  "success": true,
  "data": {
    "references": [
      {
        "reference_id": "ref_123456789",
        "label": "john_doe",
        "quality_score": 0.85,
        "is_active": true,
        "usage_count": 5,
        "created_at": "2024-01-01T12:00:00Z"
      }
    ],
    "total": 1,
    "limit": 20,
    "offset": 0,
    "has_more": false
  }
}
```

### Delete Reference

Удаление эталонного изображения.

```http
DELETE /api/v1/reference/{reference_id}
Authorization: Bearer YOUR_API_KEY
```

**Ответ:**

```json
{
  "success": true,
  "message": "Reference deleted successfully"
}
```

## ✅ Verification

### Face Verification

Верификация личности по эталонному изображению.

```http
POST /api/v1/verify
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data

image: (binary) # Тестовое изображение
reference_id: "ref_123456789" # ID эталонного изображения
threshold: 0.7 # Опционально, порог принятия решения
return_details: true # Опционально, вернуть детальную информацию
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "verification_id": "ver_123456789",
    "is_match": true,
    "confidence_score": 0.85,
    "threshold_used": 0.7,
    "similarity_score": 0.85,
    "processing_time": 0.125,
    "timestamp": "2024-01-01T12:00:00Z",
    "details": {
      "face_detected": true,
      "face_quality": 0.92,
      "reference_quality": 0.85,
      "embedding_distance": 0.15,
      "liveness_score": 0.98
    },
    "session_id": "sess_123456789"
  }
}
```

### Batch Verification

Пакетная верификация нескольких изображений.

```http
POST /api/v1/verify/batch
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "reference_id": "ref_123456789",
  "images": [
    {
      "image_data": "base64_encoded_image1",
      "filename": "test1.jpg"
    },
    {
      "image_data": "base64_encoded_image2", 
      "filename": "test2.jpg"
    }
  ],
  "threshold": 0.7
}
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "batch_id": "batch_123456789",
    "total_images": 2,
    "processed_images": 2,
    "results": [
      {
        "image_id": "img_1",
        "filename": "test1.jpg",
        "is_match": true,
        "confidence_score": 0.85,
        "processing_time": 0.125
      },
      {
        "image_id": "img_2",
        "filename": "test2.jpg",
        "is_match": false,
        "confidence_score": 0.45,
        "processing_time": 0.118
      }
    ],
    "summary": {
      "matches_found": 1,
      "average_confidence": 0.65,
      "total_processing_time": 0.243
    }
  }
}
```

## 🔍 Liveness Detection

### Liveness Check

Проверка живой личности vs фотографии/маски.

```http
POST /api/v1/liveness
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data

image: (binary) # Изображение для проверки
method: "blink" # Опционально: blink, pose, depth
return_score: true # Опционально, вернуть детальный score
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "liveness_id": "live_123456789",
    "is_live": true,
    "liveness_score": 0.92,
    "method_used": "blink",
    "confidence": 0.92,
    "processing_time": 0.085,
    "timestamp": "2024-01-01T12:00:00Z",
    "details": {
      "face_detected": true,
      "multiple_faces": false,
      "face_quality": 0.88,
      "eye_blink_detected": true,
      "head_movement_detected": false,
      "image_sharpness": 0.91
    }
  }
}
```

### Advanced Liveness

Расширенная проверка liveness с дополнительными методами.

```http
POST /api/v1/liveness/advanced
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data

image: (binary)
methods: ["blink", "pose", "depth"] # Множественные методы
return_details: true
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "liveness_id": "live_123456789",
    "is_live": true,
    "overall_score": 0.89,
    "methods": {
      "blink": {
        "score": 0.95,
        "detected": true,
        "confidence": 0.95
      },
      "pose": {
        "score": 0.85,
        "detected": true,
        "confidence": 0.85,
        "pose_variance": 0.3
      },
      "depth": {
        "score": 0.87,
        "detected": true,
        "confidence": 0.87,
        "depth_map_quality": 0.82
      }
    },
    "consensus": "live",
    "processing_time": 0.245
  }
}
```

## 📤 Upload

### Upload Image

Загрузка изображения в систему.

```http
POST /api/v1/upload
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data

image: (binary)
purpose: "reference" # reference, verification, liveness
user_id: "user_uuid" # Опционально
metadata: {"source": "mobile_app"} # Опционально
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "upload_id": "upl_123456789",
    "file_url": "https://storage.example.com/uploads/upl_123456789.jpg",
    "file_size": 2048576,
    "image_format": "JPEG",
    "image_dimensions": {
      "width": 1920,
      "height": 1080
    },
    "checksum": "sha256:abc123...",
    "processing_time": 0.065,
    "expires_at": "2024-01-08T12:00:00Z"
  }
}
```

### Upload Status

Проверка статуса загрузки.

```http
GET /api/v1/upload/{upload_id}/status
Authorization: Bearer YOUR_API_KEY
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "upload_id": "upl_123456789",
    "status": "completed",
    "progress": 100,
    "file_url": "https://storage.example.com/uploads/upl_123456789.jpg",
    "processing_results": {
      "face_detected": true,
      "face_count": 1,
      "image_quality": 0.87
    }
  }
}
```

## 👥 Admin Endpoints

### User Management

#### List Users

```http
GET /api/v1/admin/users?limit=20&offset=0&search=john
Authorization: Bearer YOUR_ADMIN_API_KEY
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "users": [
      {
        "user_id": "user_123456789",
        "username": "john_doe",
        "email": "john@example.com",
        "role": "user",
        "is_active": true,
        "is_verified": true,
        "created_at": "2024-01-01T12:00:00Z",
        "last_login": "2024-01-01T11:30:00Z",
        "total_references": 5,
        "total_verifications": 150
      }
    ],
    "total": 1,
    "limit": 20,
    "offset": 0
  }
}
```

#### Create User

```http
POST /api/v1/admin/users
Authorization: Bearer YOUR_ADMIN_API_KEY
Content-Type: application/json

{
  "username": "jane_doe",
  "email": "jane@example.com",
  "password": "secure_password",
  "role": "user",
  "first_name": "Jane",
  "last_name": "Doe"
}
```

#### Update User

```http
PUT /api/v1/admin/users/{user_id}
Authorization: Bearer YOUR_ADMIN_API_KEY
Content-Type: application/json

{
  "email": "jane.new@example.com",
  "role": "moderator",
  "is_active": false
}
```

### System Statistics

#### Get Statistics

```http
GET /api/v1/admin/stats?period=24h
Authorization: Bearer YOUR_ADMIN_API_KEY
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "period": "24h",
    "timestamp": "2024-01-01T12:00:00Z",
    "users": {
      "total": 1250,
      "active": 890,
      "new_registrations": 15
    },
    "verifications": {
      "total_requests": 15420,
      "successful_matches": 12890,
      "failed_matches": 2530,
      "average_confidence": 0.78
    },
    "references": {
      "total": 3200,
      "active": 3150,
      "deleted": 50
    },
    "system": {
      "average_response_time": 85,
      "error_rate": 0.02,
      "uptime": 99.9,
      "requests_per_minute": 45
    }
  }
}
```

#### Get Performance Metrics

```http
GET /api/v1/admin/metrics
Authorization: Bearer YOUR_ADMIN_API_KEY
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "cpu_usage": 45.2,
    "memory_usage": "2.1GB",
    "disk_usage": "15.6GB",
    "database_connections": 8,
    "redis_memory": "45MB",
    "storage_usage": "1.2GB",
    "network_io": {
      "bytes_in": "125MB",
      "bytes_out": "89MB"
    }
  }
}
```

## 🔔 Webhooks

### Register Webhook

Регистрация webhook для уведомлений.

```http
POST /api/v1/webhooks
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "url": "https://your-app.com/webhook",
  "events": ["verification.completed", "reference.created"],
  "secret": "webhook_secret_key",
  "active": true
}
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "webhook_id": "webhook_123456789",
    "url": "https://your-app.com/webhook",
    "events": ["verification.completed", "reference.created"],
    "secret": "webhook_secret_key",
    "active": true,
    "created_at": "2024-01-01T12:00:00Z"
  }
}
```

### Webhook Events

#### Verification Completed

```json
{
  "event": "verification.completed",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "verification_id": "ver_123456789",
    "user_id": "user_123456789",
    "is_match": true,
    "confidence_score": 0.85,
    "processing_time": 0.125
  }
}
```

#### Reference Created

```json
{
  "event": "reference.created",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "reference_id": "ref_123456789",
    "user_id": "user_123456789",
    "label": "john_doe",
    "quality_score": 0.85,
    "file_size": 2048576
  }
}
```

## ❌ Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `AUTHENTICATION_FAILED` | Invalid API key or token | Check your credentials |
| `AUTHORIZATION_DENIED` | Insufficient permissions | Verify your access level |
| `VALIDATION_ERROR` | Invalid input data | Check request format |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist | Verify IDs |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Reduce request rate |
| `FILE_TOO_LARGE` | Uploaded file exceeds limit | Reduce file size |
| `UNSUPPORTED_FORMAT` | Invalid image format | Use JPEG, PNG, or WEBP |
| `PROCESSING_FAILED` | Image processing error | Check image quality |
| `SERVICE_UNAVAILABLE` | External service down | Try again later |
| `QUOTA_EXCEEDED` | Usage limit reached | Upgrade plan |

### Error Response Examples

#### Validation Error

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "image",
      "reason": "File size exceeds 10MB limit"
    },
    "validation_errors": [
      {
        "field": "image",
        "message": "File size must be less than 10MB",
        "code": "file_too_large"
      },
      {
        "field": "threshold",
        "message": "Threshold must be between 0 and 1",
        "code": "invalid_range"
      }
    ]
  }
}
```

#### Rate Limit Error

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 60,
      "window": "1 minute",
      "reset_at": "2024-01-01T12:01:00Z"
    }
  }
}
```

## 🚦 Rate Limiting

### Default Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/v1/verify` | 60 requests | 1 minute |
| `/api/v1/liveness` | 30 requests | 1 minute |
| `/api/v1/reference` | 20 requests | 1 minute |
| `/api/v1/upload` | 10 requests | 1 minute |
| Admin endpoints | 100 requests | 1 minute |

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 60
```

### Custom Limits

Для увеличения лимитов обратитесь в поддержку или обновите план подписки.

## 💻 SDK Examples

### Python SDK

```python
from face_recognition_client import FaceRecognitionClient

# Инициализация клиента
client = FaceRecognitionClient(
    api_key="your_api_key",
    base_url="https://api.face-recognition.com"
)

# Загрузка эталонного изображения
reference = client.reference.upload(
    image_path="reference.jpg",
    label="john_doe"
)

# Верификация
result = client.verify.verify(
    image_path="test.jpg",
    reference_id=reference.id,
    threshold=0.7
)

print(f"Match: {result.is_match}")
print(f"Confidence: {result.confidence_score}")
```

### JavaScript SDK

```javascript
import { FaceRecognitionClient } from '@face-recognition/sdk';

const client = new FaceRecognitionClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.face-recognition.com'
});

// Загрузка эталонного изображения
const reference = await client.reference.upload({
  image: imageFile,
  label: 'john_doe'
});

// Верификация
const result = await client.verify.verify({
  image: testImageFile,
  referenceId: reference.id,
  threshold: 0.7
});

console.log(`Match: ${result.isMatch}`);
console.log(`Confidence: ${result.confidenceScore}`);
```

### cURL Examples

#### Basic Verification

```bash
curl -X POST "https://api.face-recognition.com/api/v1/verify" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg" \
  -F "reference_id=ref_123456789" \
  -F "threshold=0.7"
```

#### Batch Verification

```bash
curl -X POST "https://api.face-recognition.com/api/v1/verify/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_id": "ref_123456789",
    "images": [
      {
        "image_data": "base64_encoded_image1",
        "filename": "test1.jpg"
      }
    ],
    "threshold": 0.7
  }'
```

#### Liveness Check

```bash
curl -X POST "https://api.face-recognition.com/api/v1/liveness" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg" \
  -F "method=blink"
```

## 📈 SDK Statistics

### Response Time Statistics

```http
GET /api/v1/stats/response-times
Authorization: Bearer YOUR_API_KEY
```

**Ответ:**

```json
{
  "success": true,
  "data": {
    "period": "1h",
    "endpoints": {
      "/api/v1/verify": {
        "average": 85,
        "p50": 78,
        "p95": 145,
        "p99": 234,
        "requests": 1240
      },
      "/api/v1/liveness": {
        "average": 65,
        "p50": 58,
        "p95": 112,
        "p99": 189,
        "requests": 890
      }
    }
  }
}
```

## 🔄 WebSocket Support

### Real-time Verification

```javascript
const ws = new WebSocket('wss://api.face-recognition.com/ws/verify');

ws.onopen = function(event) {
  console.log('Connected to WebSocket');
  
  // Отправка изображения для real-time верификации
  ws.send(JSON.stringify({
    type: 'verify',
    reference_id: 'ref_123456789',
    image: imageBase64
  }));
};

ws.onmessage = function(event) {
  const result = JSON.parse(event.data);
  console.log('Verification result:', result);
};
```

**WebSocket Response:**

```json
{
  "type": "verification_result",
  "data": {
    "verification_id": "ver_123456789",
    "is_match": true,
    "confidence_score": 0.85,
    "processing_time": 0.125
  }
}
```

## 📝 Changelog

### v1.0.0 (2024-01-01)
- Initial release
- Basic face recognition functionality
- Verification and liveness detection
- REST API with OpenAPI documentation
- Docker support
- Admin panel endpoints

### Planned Features
- Face enrollment from video streams
- 3D face recognition
- Emotion detection
- Age and gender estimation
- Mobile SDK
- Face recognition in video streams

---

**📚 Полная документация доступна на**: https://docs.face-recognition.com

**🔗 Interactive API Explorer**: https://api.face-recognition.com/docs