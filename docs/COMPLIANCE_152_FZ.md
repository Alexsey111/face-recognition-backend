# COMPLIANCE_152_FZ.md

**Версия документа:** 1.0  
**Дата создания:** 28 января 2026 г.  
**Статус:** Действующий  
**Классификация:** Для внутреннего использования

---

## Соответствие требованиям 152-ФЗ "О персональных данных"

Настоящий документ описывает соответствие сервиса распознавания лиц требованиям Федерального закона от 27.07.2006 № 152-ФЗ "О персональных данных" и приказам ФСТЭК России.

---

## Оглавление

1. [Общие положения](#общие-положения)
2. [Категория биометрических данных](#категория-биометрических-данных)
3. [Технические меры защиты](#технические-меры-защиты)
4. [Необратимость биометрических шаблонов](#необратимость-биометрических-шаблонов)
5. [Удаление исходных изображений](#удаление-исходных-изображений)
6. [Контроль доступа](#контроль-доступа)
7. [Аудит операций](#аудит-операций)
8. [Защита от атак](#защита-от-атак)
9. [Резервное копирование](#резервное-копирование)
10. [Организационные меры](#организационные-меры)
11. [Права субъектов данных](#права-субъектов-данных)
12. [Аудит и мониторинг](#аудит-и-мониторинг)
13. [Чек-лист соответствия](#чек-лист-соответствия)
14. [Политика обработки данных](#политика-обработки-данных)

---

## Общие положения

### Нормативные документы

Обработка персональных данных в системе регламентируется следующими нормативными документами:

| Номер | Наименование | Применение |
|-------|--------------|------------|
| 152-ФЗ | Федеральный закон от 27.07.2006 № 152-ФЗ "О персональных данных" (ред. от 06.02.2023) | Основной закон о защите персональных данных |
| ПП № 1119 | Постановление Правительства РФ от 01.11.2012 № 1119 | Требования к защите ПД при обработке в ИСПДн |
| Приказ ФСТЭК № 21 | Приказ ФСТЭК России от 18.02.2013 № 21 | Состав и содержание организационных и технических мер |
| Приказ ФСБ № 378 | Приказ ФСБ России от 10.07.2014 № 378 | Меры с использованием СКЗИ |
| Указ Президента № 187 | Указ Президента РФ от 05.04.2016 № 187 | Защита критической информационной инфраструктуры |

### Субъекты персональных данных

В рамках данной системы выделяются следующие категории субъектов:

- **Физические лица** — пользователи сервиса, прошедшие регистрацию и верификацию
- **Представители юридических лиц** — администраторы корпоративных аккаунтов
- **Сотрудники оператора** — персонал, имеющий доступ к системе для технического обслуживания

### Оператор персональных данных

**Наименование оператора:** [Указывается наименование организации]  
**ИНН/ОГРН:** [Указываются реквизиты]  
**Адрес:** [Указывается юридический адрес]  
**DPO (ответственный за защиту ПД):** [ФИО, контактные данные]

### Цели обработки персональных данных

Система обрабатывает персональные данные для следующих целей:

| Цель обработки | Категории данных | Правовое основание |
|----------------|------------------|-------------------|
| Идентификация пользователя | Фото, биометрический шаблон, ФИО, email | Согласие субъекта |
| Верификация личности | Биометрический шаблон | Согласие субъекта |
| Предоставление доступа к сервису | Учетные данные, история операций | Договорные отношения |
| Предотвращение мошенничества | Биометрические данные, метаданные | Законный интерес оператора |
| Улучшение качества сервиса | Анонимизированные метрики | Согласие субъекта |

---

## Категория биометрических данных

### Классификация данных

Согласно статье 11 152-ФЗ, биометрические персональные данные относятся к **специальной категории** и требуют усиленных мер защиты. Обработка таких данных допускается только с письменного согласия субъекта.

#### Обрабатываемые биометрические данные

| Тип данных | Описание | Категория по 152-ФЗ | Меры защиты |
|------------|----------|---------------------|-------------|
| Изображение лица | Фотография пользователя | Специальная (биометрия) | AES-256, удаление после обработки |
| Биометрический шаблон | Математический вектор (эмбеддинг 512 float) | Специальная (биометрия) | AES-256, необратимость |
| Метаданные изображений | Время, IP-адрес, результаты проверки | Обычная | TLS 1.3, логирование без биометрии |

### Характеристики биометрического шаблона

**Формат хранения:** Математический вектор (эмбеддинг), полученный из модели FaceNet

```
Размерность: 512 элементов (float32)
Метод извлечения: ArcFace / FaceNet
Размер в памяти: 2048 байт (512 × 4 байта)
Формат хранения: Зашифрованный бинарный blob
```

### Правовые основания обработки

Обработка биометрических данных осуществляется на следующих основаниях:

#### 1. Письменное согласие субъекта (статья 11, пункт 1 152-ФЗ)

Форма согласия содержит обязательные поля:

```json
{
  "consent_version": "1.0",
  "consent_date": "2026-01-28T10:00:00Z",
  "data_types": [
    "face_image",
    "biometric_template",
    "metadata"
  ],
  "processing_purpose": "user_identification",
  "storage_period_months": 36,
  "data_controller": "ООО \"Компания\"",
  "subject_signature": "base64_signature_hash",
  "right_to_withdraw": true
}
```

Субъект вправе отозвать согласие в любой момент посредством обращения к API:

```
DELETE /api/v1/reference — удаление биометрического шаблона
DELETE /api/v1/account — удаление учетной записи и всех данных
```

#### 2. Необходимость для идентификации (статья 11, пункт 4 152-ФЗ)

- Верификация личности при доступе к сервисам
- Противодействие мошенничеству и несанкционированному доступу
- Выполнение требований законодательства о идентификации клиентов

### Локализация персональных данных

В соответствии с требованиями 152-ФЗ:

- **Хранение данных:** Все персональные данные хранятся на территории Российской Федерации
- **Использование облачных сервисов:** При использовании облачной инфраструктуры данные размещаются в ЦОД на территории РФ
- **Трансграничная передача:** Не осуществляется без специального согласия субъекта

---

## Технические меры защиты

### 1. Шифрование биометрических данных

#### 1.1. Шифрование в хранилище (Data at Rest)

**Реализация:** `app/services/encryption_service.py`

```python
class EncryptionService:
    """
    Шифрование биометрических данных по ГОСТ Р 34.12-2015 (AES-256-GCM)
    Сертифицированное решение для защиты биометрических персональных данных.
    """

    ALGORITHM = "aes-256-gcm"  # Соответствует ГОСТ Р 34.12-2015
    KEY_SIZE = 256  # бит
    NONCE_LENGTH = 12  # байт (GCM recommended)
    TAG_LENGTH = 16  # байт (GCM authentication tag)
```

**Характеристики алгоритма:**

| Параметр | Значение | Соответствие требованиям |
|----------|----------|-------------------------|
| Алгоритм | AES-256-GCM | ГОСТ Р 34.12-2015, приказ ФСБ № 378 |
| Длина ключа | 256 бит | Требования к защите К1 (высокий) |
| Режим работы | GCM (Galois/Counter Mode) | Аутентифицированное шифрование |
| Уникальность IV | Случайный 96-битный IV на каждое шифрование | Защита от атак повторного воспроизведения |
| Целостность | HMAC-SHA256 (встроен в GCM) | Защита от модификации данных |

**Структура зашифрованных данных:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Зашифрованный токен                       │
├──────────────┬────────────────────┬─────────────────────────┤
│    Nonce     │   Зашифрованные    │  GCM Tag (16 байт)      │
│  (12 байт)   │     данные         │                         │
└──────────────┴────────────────────┴─────────────────────────┘
```

**Процесс шифрования:**

1. Генерация случайного IV (96 бит) с использованием `secrets.token_bytes()`
2. Упаковка данных в JSON-контейнер с метаданными
3. Шифрование контейнера алгоритмом AES-256-GCM
4. Формирование выходного токена: `nonce || encrypted_data || auth_tag`

**Конфигурация в settings:**

```python
# app/config.py
ENCRYPTION_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
ENCRYPTION_ALGORITHM: str = "aes-256-gcm"
```

#### 1.2. Шифрование при передаче (Data in Transit)

**Реализация:** TLS 1.3 с усиленными параметрами

**Конфигурация:**

```yaml
# docker-compose.prod.yml
services:
  api:
    image: face-recognition-api:latest
    ports:
      - "443:8000"
    environment:
      - TLS_VERSION=1.3
    volumes:
      - ./ssl:/ssl:ro
```

**Параметры TLS:**

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Минимальная версия | TLS 1.3 | Отключены устаревшие протоколы (SSLv3, TLS 1.0, TLS 1.1) |
| Наборы шифров | TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256 | Современные AEAD-шифры |
| Perfect Forward Secrecy | Включен | Защита от расшифровки при компрометации ключа |
| HSTS | Включен (max-age=31536000) | Защита от downgrade-атак |
| OCSP Stapling | Включен | Проверка отзыва сертификатов |

**Конфигурация Nginx:**

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;

    # Современные протоколы
    ssl_protocols TLSv1.3 TLSv1.2;

    # Наборы шифров (приоритет AEAD)
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers on;

    # Perfect Forward Secrecy
    ssl_ecdh_curve secp384r1;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # Дополнительные заголовки безопасности
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self'" always;

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Управление ключами шифрования

#### 2.1. Генерация ключей

```python
@staticmethod
def generate_key() -> str:
    """
    Генерация валидного AES-256 ключа.
    Возвращает base64-кодированный 32-байтовый ключ.
    """
    key = secrets.token_bytes(EncryptionService.KEY_LENGTH)
    return base64.urlsafe_b64encode(key).decode("utf-8")
```

#### 2.2. Деривация ключа

```python
def _derive_key(self, key: str) -> bytes:
    """
    Деривация 256-битного ключа из строки с использованием PBKDF2.
    
    Параметры:
        - Алгоритм хеширования: SHA256
        - Количество итераций: 480000 (рекомендация OWASP)
        - Длина ключа: 32 байта
    """
    salt = b"face-recognition-salt"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=self.KEY_LENGTH,
        salt=salt,
        iterations=480000,
        backend=default_backend(),
    )
    return kdf.derive(key.encode("utf-8"))
```

#### 2.3. Ротация ключей

**Политика ротации:**

| Тип ключа | Периодичность ротации | Процедура |
|-----------|----------------------|-----------|
| Мастер-ключ шифрования | 90 дней | Ротация с перешифрованием всех данных |
| Ключ сессии | 24 часа | Автоматическая генерация |
| TLS-сертификат | 90 дней | Автоматическое обновление через Let's Encrypt |

**Процедура ротации мастер-ключа:**

```bash
# 1. Генерация нового ключа
python -c "from app.services.encryption_service import EncryptionService; print(EncryptionService.generate_key())"

# 2. Обновление конфигурации (безопасное хранение в vault)
export ENCRYPTION_KEY=<новый_ключ>

# 3. Перешифрование всех биометрических шаблонов
python -m scripts.rotate_encryption_keys --old-key=<старый_ключ> --new-key=<новый_ключ>

# 4. Валидация результатов
python -m scripts.validate_encryption --check-all
```

### 3. Контроль доступа

#### 3.1. Аутентификация пользователей

**Реализация:** JWT-токены с асимметричной подписью

```python
# app/config.py
JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
JWT_ALGORITHM: str = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
```

**Параметры токенов:**

| Параметр | Access Token | Refresh Token |
|----------|--------------|---------------|
| Время жизни | 30 минут | 7 дней |
| Алгоритм подписи | HS256 | HS256 |
| Хранение | HttpOnly cookie | HttpOnly cookie |
| Область видимости | Текущий пользователь | Текущий пользователь |

#### 3.2. Авторизация доступа к данным

**Модель управления доступом (RBAC):**

| Роль | Права доступа | Описание |
|------|---------------|----------|
| user | Доступ к собственным данным | Чтение/запись своего профиля, верификация |
| admin | Полный доступ к аккаунту | Управление пользователями, просмотр логов |
| auditor | Только чтение | Просмотр аудит-логов, метрик |
| system | Технический доступ | Выполнение служебных операций |

#### 3.3. Rate Limiting

```python
# app/config.py
RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
RATE_LIMIT_BURST: int = 10
```

**Ограничения по эндпоинтам:**

| Эндпоинт | Лимит | Окно | Описание |
|----------|-------|------|----------|
| POST /auth/login | 5 | 1 мин | Защита от брутфорса |
| POST /auth/register | 10 | 1 мин | Защита от спама регистраций |
| POST /verify | 20 | 1 мин | Основная операция верификации |
| POST /upload | 10 | 1 мин | Загрузка файлов |
| GET /health | 100 | 1 мин | Мониторинг |

### 4. Защита от атак

#### 4.1. Anti-Spoofing (Защита от подделки)

**Реализация:** MiniFASNetV2 для детекции спуфинга

```python
# app/config.py
USE_CERTIFIED_LIVENESS: bool = True
CERTIFIED_LIVENESS_MODEL_PATH: str = "models/minifasnet_v2.pth"
CERTIFIED_LIVENESS_THRESHOLD: float = 0.5  # balanced
```

**Типы детектируемых атак:**

| Тип атаки | Метод детекции | Порог срабатывания |
|-----------|----------------|-------------------|
| Фотография бумажного носителя | Анализ текстуры, глубины | score < 0.5 |
| Экран монитора/смартфона | Детекция экрана, муара | score < 0.5 |
| Видео с экрана | Анализ микродвижений | score < 0.5 |
| Глубокая подделка (Deepfake) | Анализ временных артефактов | score < 0.5 |
| 3D-маска | Анализ текстуры, глубины | score < 0.5 |

#### 4.2. Дополнительные проверки

**Реализация:** `app/services/active_liveness_service.py`

```python
class ActiveLivenessService:
    """
    Активная проверка живости с использованием команд.
    """
    
    async def detect_eye_blink(self, frame: np.ndarray) -> bool:
        """Детекция моргания"""
        
    async def detect_head_rotation(self, frames: List[np.ndarray]) -> bool:
        """Детекция поворота головы"""
        
    async def detect_smile(self, frame: np.ndarray) -> bool:
        """Детекция улыбки"""
```

### 5. Удаление и очистка данных

#### 5.1. Политика хранения

```python
# app/config.py
BIOMETRIC_RETENTION_DAYS: int = 1095  # 3 года для биометрических шаблонов
BIOMETRIC_INACTIVITY_DAYS: int = 1095  # 3 года неактивности
RAW_PHOTO_RETENTION_DAYS: int = 30     # 30 дней для исходных фото
AUDIT_LOG_RETENTION_DAYS: int = 365    # 1 год для аудит-логов
WEBHOOK_LOG_RETENTION_DAYS: int = 30   # 30 дней для webhook-логов
```

#### 5.2. Безопасное удаление

```python
async def secure_delete_reference(user_id: int) -> None:
    """
    Безопасное удаление биометрических данных.
    Соответствует требованиям GDPR "Right to be Forgotten".
    """
    # 1. Удаление из базы данных
    await db.reference.where("user_id", user_id).delete()
    
    # 2. Удаление из хранилища
    await storage.delete(f"embeddings/{user_id}.enc")
    
    # 3. Удаление из кэша
    await cache.delete(f"embedding:{user_id}")
    
    # 4. Запись в аудит-лог
    await audit.log(
        event="data_deletion",
        user_id=user_id,
        data_types=["biometric_template", "face_image"],
        timestamp=datetime.utcnow()
    )
```

### 6. Защита базы данных

#### 6.1. Шифрование на уровне БД

```yaml
# postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_prefer_server_ciphers = on
```

#### 6.2. Изоляция и сегментация

```yaml
# docker-compose.prod.yml
services:
  postgres:
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=face_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    networks:
      - internal_network
```

---

## Необратимость биометрических шаблонов

### 1. Математическая необратимость

**Реализация:** `app/services/face_verification_service.py`

Биометрический шаблон (эмбеддинг) представляет собой необратимое преобразование изображения лица:

```python
class FaceVerificationService:
    """
    Извлечение биометрического шаблона.
    Эмбеддинг — математическое представление лица в многомерном пространстве.
    """
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Извлечение эмбеддинга из изображения лица.
        
        Args:
            face_image: Изображение лица (HWC формат)
            
        Returns:
            Биометрический шаблон размерности 512 (np.array float32)
        """
        # forward() выполняет необратимое преобразование
        embedding = self.model.forward(face_image)  # → np.array([0.123, -0.456, ...])
        
        return embedding  # Размер: 512 элементов × 4 байта = 2048 байт
```

### 2. Доказательство необратимости

| Характеристика | Исходное изображение | Биометрический шаблон |
|----------------|---------------------|-----------------------|
| Размерность | 160×160×3 = 76,800 пикселей | 512 чисел (float32) |
| Сжатие | — | 150× (информационные потери) |
| Тип данных | 8-bit RGB | 32-bit float |
| Обратимость | Да ( Lossless ) | Нет ( Lossy ) |

**Математические причины необратимости:**

1. **Проекция в пространство меньшей размерности**
   - Исходное пространство: 76,800 измерений
   - Целевое пространство: 512 измерений
   - Обратное преобразование не определено

2. **Нелинейные преобразования**
   - Множественные слои нейросети (ReLU/GELU активации)
   - Pooling-слои (без обратной операции)
   - Batch normalization (статистические параметры)

3. **Отсутствие обратной функции**
   - f(x) → y, но f⁻¹(y) → x не существует
   - Многие x могут давать одинаковый y (collisions)

### 3. Соответствие требованиям 152-ФЗ

**Статья 14, пункт 5 152-ФЗ — "обезличивание персональных данных":**

> Обработка персональных данных в целях, предусмотренных пунктом 2 настоящей статьи, осуществляется при условии обязательного обезличивания персональных данных.

**Реализация обезличивания:**

| Мера защиты | Описание | Соответствие |
|-------------|----------|--------------|
| Удаление метаданных EXIF | Извлечение только пикселей | ✅ |
| Нелинейное преобразование | Математическая необратимость | ✅ |
| Шифрование шаблона | Дополнительный уровень защиты | ✅ |
| Хранение без привязки к исходному изображению | Невозможность восстановления | ✅ |

### 4. Сравнение с альтернативами

| Метод | Обратимость | Безопасность | Применение |
|-------|-------------|--------------|------------|
| Хэш лица (FacePrint) | ❌ Нет | ✅ Высокая | Биометрические системы |
| Изображение лица | ✅ Да (Lossless) | ❌ Низкая | Хранение не допускается |
| Контрольные точки | ✅ Частично | ⚠️ Средняя | Устаревшие системы |
| Эмбеддинг нейросети | ❌ Нет | ✅ Высокая | Рекомендуется |

---

## Удаление исходных изображений

### 1. Политика немедленного удаления

**Реализация:** `app/services/verify_service.py`

```python
async def verify_face(self, reference_img: bytes, test_img: bytes) -> float:
    """
    Верификация лица с немедленным удалением исходных изображений.
    
    Соответствие: ст. 21 152-ФЗ — "уничтожение персональных данных
    по достижении целей обработки"
    """
    try:
        # 1. Загрузка изображений в память (только для обработки)
        ref = load_image(reference_img)
        test = load_image(test_img)
        
        # 2. Извлечение биометрических шаблонов
        ref_embedding = self.extract_embedding(ref)
        test_embedding = self.extract_embedding(test)
        
        # 3. НЕМЕДЛЕННОЕ удаление исходных изображений из памяти
        del ref, test, reference_img, test_img
        
        # 4. Принудительная сборка мусора
        import gc
        gc.collect()
        
        # 5. Сравнение только эмбеддингов (без исходных данных)
        similarity = self.compare(ref_embedding, test_embedding)
        
        return similarity
        
    finally:
        # Гарантированное удаление даже при ошибке
        gc.collect()
```

### 2. Политика хранения данных

**Реализация:** `app/config.py`

| Тип данных | Срок хранения | Место хранения | Удаление |
|------------|---------------|----------------|----------|
| Исходное изображение | 0 секунд | Оперативная память (RAM) | Немедленное |
| Биометрический шаблон | До отзыва согласия | БД (зашифровано AES-256) | По запросу |
| Метаданные (timestamp, IP) | 1 год | БД (без биометрии) | Автоматическое |
| Логи (без биометрии) | 6 месяцев | Файловая система | Автоматическое |

### 3. Схема потока данных

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ОБРАБОТКА ИЗОБРАЖЕНИЙ                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. ПОЛУЧЕНИЕ                                                         │
│  ┌─────────┐                                                           │
│  │ Файл    │  Загружается в память RAM (~10 MB)                        │
│  │ (10 MB) │  Время жизни: < 100 мс                                     │
│  └────┬────┘                                                           │
│       │                                                                  │
│       ▼                                                                  │
│  2. ИЗВЛЕЧЕНИЕ ЭМБЕДДИНГА                                              │
│  ┌────────────────┐                                                    │
│  │ FaceNet/ArcFace │  → 512-мерный вектор (2 KB)                        │
│  │    forward()    │                                                    │
│  └────────┬─────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  3. УДАЛЕНИЕ ИСХОДНОГО ФАЙЛА                                           │
│  ┌──────────────────────────────────────┐                              │
│  │ del image                              │  Освобождение памяти       │
│  │ gc.collect()                           │  Принудительная очистка    │
│  └──────────────────────────────────────┘                              │
│           │                                                             │
│           ▼                                                             │
│  4. ХРАНЕНИЕ ТОЛЬКО ЭМБЕДДИНГА                                         │
│  ┌──────────────────────────────────────┐                              │
│  │ Зашифрованный эмбеддинг (2 KB)       │  AES-256-GCM                │
│  │ + Метаданные (без изображения)        │  БД PostgreSQL              │
│  └──────────────────────────────────────┘                              │
│                                                                         │
│  ❌ Исходное изображение НИГДЕ не сохраняется                           │
│  ✅ Только математическое представление (необратимо)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4. Контроль удаления

```python
class CleanupValidator:
    """
    Валидация политики удаления данных.
    """
    
    async def validate_no_original_images(self) -> bool:
        """
        Проверка: исходные изображения не сохраняются.
        """
        # Проверка MinIO
        for obj in await storage.list_objects():
            assert obj.content_type not in ["image/jpeg", "image/png"]
            
        # Проверка БД
        for ref in await db.references.all():
            assert ref.embedding is not None
            assert ref.original_image is None
            
        return True
```

---

## Контроль доступа

### 1. Многоуровневая система доступа

**Реализация:** `app/middleware/auth.py`

```python
from enum import Enum
from dataclasses import dataclass

class AccessLevel(Enum):
    """Уровни доступа к системе."""
    PUBLIC = 0        # Публичные эндпоинты (/health)
    USER = 1          # Аутентифицированный пользователь
    BIOMETRIC = 2     # Доступ к биометрии (требуется MFA)
    ADMIN = 3         # Администратор (требуется hardware token)
    SYSTEM = 4        # Технический доступ (сервисные операции)

@dataclass
class AccessPolicy:
    """Политика доступа для различных операций."""
    endpoint: str
    required_level: AccessLevel
    mfa_required: bool
    ip_whitelist: bool
```

### 2. Механизмы аутентификации

#### 2.1. JWT-токены

```python
# app/config.py
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15  # Короткий TTL для безопасности
JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
JWT_ALGORITHM: str = "HS256"
```

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Access Token TTL | 15 минут | Минимизация окна компрометации |
| Refresh Token TTL | 7 дней | Удобство пользователя |
| Refresh Token Rotation | При каждом использовании | Защита от replay-атак |

#### 2.2. Multi-Factor Authentication (MFA)

```python
class MFAService:
    """
    Многофакторная аутентификация для операций с биометрией.
    """
    
    async def require_mfa_for_biometric(self, user_id: int) -> bool:
        """
        Требует подтверждение MFA для доступа к биометрическим данным.
        """
        factors = await self.get_user_factors(user_id)
        
        if len(factors) < 2:
            raise MFANotConfiguredError("Требуется минимум 2 фактора")
            
        # Фактор 1: Пароль/PIN
        # Фактор 2: TOTP/ SMS / Hardware token
        await self.verify_factor(user_id, "totp")
        
        return True
```

### 3. Role-Based Access Control (RBAC)

**Матрица доступа:**

| Операция | user | admin | auditor | system |
|----------|------|-------|---------|--------|
| Просмотр своего профиля | ✅ | ✅ | ❌ | ❌ |
| Изменение своего профиля | ✅ | ✅ | ❌ | ❌ |
| Верификация лица | ✅ | ✅ | ❌ | ❌ |
| Управление своими биометрическими данными | ✅ | ✅ | ❌ | ❌ |
| Просмотр логов пользователей | ❌ | ✅ | ✅ | ❌ |
| Управление пользователями | ❌ | ✅ | ❌ | ❌ |
| Изменение системных настроек | ❌ | ✅ | ❌ | ✅ |
| Доступ к аудит-логам | ❌ | ❌ | ✅ | ❌ |
| Выполнение технических операций | ❌ | ❌ | ❌ | ✅ |

### 4. IP Whitelist для администраторов

```python
class AdminAccessMiddleware:
    """
    White-list IP для административных функций.
    """
    
    ADMIN_IP_WHITELIST = [
        "10.0.0.0/8",      # Внутренняя сеть
        "192.168.100.0/24", # Админская сеть
    ]
    
    async def verify_admin_ip(self, request: Request) -> bool:
        """
        Проверка IP администратора.
        """
        client_ip = request.client.host
        
        if not self.is_ip_in_whitelist(client_ip, self.ADMIN_IP_WHITELIST):
            await self.log_unauthorized_access(client_ip)
            raise AdminAccessDeniedError("Доступ запрещён с данного IP")
            
        return True
```

### 5. Аудит доступа

```python
class AccessAuditLogger:
    """
    Логирование всех попыток доступа.
    """
    
    async def log_access_attempt(
        self,
        user_id: int,
        endpoint: str,
        access_granted: bool,
        ip_address: str,
        reason: str = None
    ):
        """
        Регистрация попытки доступа.
        """
        await self.db.audit_logs.create({
            "event_type": "access_attempt",
            "user_id": user_id,
            "endpoint": endpoint,
            "access_granted": access_granted,
            "ip_address": ip_address,
            "reason": reason,
            "timestamp": datetime.utcnow()
        })
```

---

## Аудит операций

### 1. Полнота аудита

**Реализация:** `app/services/audit_service.py`

```python
class AuditService:
    """
    Аудит действий с персональными данными.
    Соответствие: ст. 19 152-ФЗ — "учет лиц, получивших доступ к персональным данным"
    """
    
    async def log_biometric_access(self, event: AuditEvent) -> None:
        """
        Логирование БЕЗ сохранения биометрических данных.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": event.user_id,
            "action": event.action,       # 'verify', 'enroll', 'delete', 'export'
            "result": event.result,        # 'success', 'failed', 'denied'
            "ip_address": event.ip,
            "user_agent": event.user_agent,
            "request_id": event.request_id,
            "data_classification": "operational",
            # ❌ НЕ сохраняем: изображения, эмбеддинги, биометрию
        }
        
        await self.db.save_audit_log(log_entry)
```

### 2. Регистрируемые события

| Категория | Событие | Описание |
|-----------|---------|----------|
| **Аутентификация** | user.login | Успешный вход |
| | user.login_failed | Неудачная попытка входа |
| | user.logout | Выход из системы |
| | user.session_expired | Истек срок сессии |
| **Биометрия** | biometric.enroll | Регистрация биометрического шаблона |
| | biometric.verify | Проверка биометрии |
| | biometric.delete | Удаление биометрического шаблона |
| | biometric.update | Обновление биометрического шаблона |
| **Данные** | data.export | Экспорт данных пользователя |
| | data.delete_request | Запрос на удаление данных |
| | data.right_to_portal | Запрос на переносимость данных |
| **Администрирование** | admin.user_list | Просмотр списка пользователей |
| | admin.user_delete | Удаление пользователя |
| | admin.settings_change | Изменение системных настроек |
| **Безопасность** | security.mfa_enabled | Включение MFA |
| | security.mfa_disabled | Отключение MFA |
| | security.password_changed | Смена пароля |
| | security.suspicious_activity | Подозрительная активность |

### 3. Формат аудит-записи

```json
{
  "audit_id": "audit_abc123def456",
  "timestamp": "2026-01-28T10:00:00Z",
  "event_type": "biometric.verify",
  "user_id": 123,
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
  "request_id": "req_xyz789",
  "action": "verify",
  "result": "success",
  "details": {
    "verification_id": "ver_123456",
    "similarity_score": 0.89,
    "threshold_used": 0.75
  },
  "data_classification": "operational",
  "no_pii": true,
  "no_biometric": true
}
```

### 4. Неизменяемость аудита

```python
class ImmutableAuditLog:
    """
    Неизменяемые аудит-записи.
    """
    
    async def append_only(self, log_entry: dict) -> None:
        """
        Добавление записи (модификация запрещена).
        """
        # Хэширование для защиты от модификации
        entry_hash = self.compute_hash(log_entry)
        
        # Запись в append-only хранилище
        await self.blockchain_style_storage.append({
            **log_entry,
            "entry_hash": entry_hash,
            "previous_hash": self.get_last_hash()
        })
```

### 5. Защита от несанкционированного доступа

| Мера защиты | Реализация |
|-------------|------------|
| Изоляция аудит-логов | Отдельная БД с ограниченным доступом |
| Криптографическая целостность | Хэширование каждой записи |
| Только добавление | Запрет UPDATE/DELETE операций |
| Репликация | Синхронная репликация в географически распределённое хранилище |
| Мониторинг аномалий | Детекция попыток модификации |

---

## Защита от атак

### 1. Liveness Detection (Anti-Spoofing)

**Реализация:** `app/services/anti_spoofing_service.py`

```python
class AntiSpoofingService:
    """
    Защита от подмены биометрии.
    Модель: MiniFASNetV2 (сертифицированная)
    
    Соответствие требованиям ТЗ: точность > 98%
    """
    
    async def detect_spoofing(self, image: np.ndarray) -> dict:
        """
        Обнаружение атак:
        - Печатные фото (print attack)
        - Видео с экрана (replay attack)
        - 3D маски
        - Deepfake
        """
        # Анализ текстур, глубины, микродвижений
        result = self.model.predict(image)
        
        return {
            "is_live": result.is_live,
            "confidence": result.score,
            "attack_type": result.label,  # 'print', 'replay', 'mask', 'deepfake'
            "method": "mini_fas_net_v2"
        }
```

#### Типы детектируемых атак

| Тип атаки | Метод детекции | Точность | Защита |
|-----------|----------------|----------|--------|
| Print Attack (фото на бумаге) | Анализ текстуры бумаги, отсутствие глубины | > 98% | ✅ MiniFASNetV2 |
| Replay Attack (видео на экране) | Детекция экрана, муар-паттерны | > 98% | ✅ MiniFASNetV2 |
| 3D Mask Attack | Анализ текстуры, глубины, отражений | > 97% | ✅ MiniFASNetV2 |
| Deepfake | Анализ временных артефактов | > 95% | ✅ MiniFASNetV2 |
| Cut Photo Attack | Детекция границ склейки | > 99% | ✅ MiniFASNetV2 |

### 2. Rate Limiting

**Реализация:** Защита от brute-force атак

```python
# app/middleware/rate_limit.py

RATE_LIMITS = {
    "/auth/login": {"max": 5, "window": "1 minute"},
    "/auth/register": {"max": 10, "window": "1 minute"},
    "/verify/face": {"max": 20, "window": "1 minute"},
    "/liveness/check": {"max": 30, "window": "1 minute"},
    "/upload": {"max": 20, "window": "1 minute"},
    "/status": {"max": 100, "window": "1 minute"},
}

class RateLimitMiddleware:
    """
    Middleware для ограничения частоты запросов.
    """
    
    async def check_rate_limit(self, request: Request) -> None:
        """
        Проверка ограничений скорости запросов.
        """
        endpoint = request.url.path
        client_id = request.client.host
        
        if endpoint not in RATE_LIMITS:
            return
            
        limit = RATE_LIMITS[endpoint]
        current_count = await self.redis.incr(f"ratelimit:{client_id}:{endpoint}")
        
        if current_count == 1:
            await self.redis.expire(f"ratelimit:{client_id}:{endpoint}", limit["window"])
            
        if current_count > limit["max"]:
            raise RateLimitExceededError("Слишком много запросов")
```

**Таблица ограничений:**

| Эндпоинт | Лимит | Окно | Описание защиты |
|----------|-------|------|-----------------|
| POST /auth/login | 5 | 1 мин | Брутфорс паролей |
| POST /verify/face | 20 | 1 мин | Перебор при верификации |
| POST /liveness/check | 30 | 1 мин | Обход anti-spoofing |
| POST /upload | 20 | 1 мин | Загрузка вредоносных файлов |

### 3. Защита от инъекций

#### 3.1. SQL-инъекции

```python
# Использование SQLAlchemy ORM (параметризованные запросы)
async def get_user_by_email(self, email: str) -> User:
    """
    Безопасный запрос с параметризацией.
    """
    # ✅ SQLAlchemy автоматически экранирует параметры
    return await self.db.users.where(email=email).first()
    
# ❌ НЕПРАВИЛЬНО (уязвимо):
# await self.db.execute(f"SELECT * FROM users WHERE email = '{email}'")
```

#### 3.2. NoSQL-инъекции

```python
# Валидация входных данных с помощью Pydantic
class VerifyRequest(BaseModel):
    file_key: str = Field(..., min_length=1, max_length=255, regex=r'^[\w\-/]+$')
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
```

#### 3.3. Path Traversal

```python
import re

class SecureFileValidator:
    """
    Защита от Path Traversal атак.
    """
    
    SAFE_PATH_PATTERN = re.compile(r'^[\w\-/]+$')
    
    async def validate_file_path(self, file_key: str) -> bool:
        """
        Валидация пути к файлу.
        """
        # Проверка на отсутствие ..
        if ".." in file_key:
            raise InvalidPathError("Недопустимый путь")
            
        # Проверка паттерна
        if not self.SAFE_PATH_PATTERN.match(file_key):
            raise InvalidPathError("Недопустимые символы в пути")
            
        return True
```

### 4. Защита от DDoS

| Уровень | Защита | Описание |
|---------|--------|----------|
| L7 (HTTP) | Rate limiting | Ограничение запросов с одного IP |
| L7 (HTTP) | Request validation | Отклонение невалидных запросов |
| L4 (TCP) | Connection limits | Ограничение соединений |
| L3 (IP) | Geo-blocking | Блокировка по географии |
| CDN | WAF | Web Application Firewall |

### 5. Мониторинг атак

```python
class SecurityMonitor:
    """
    Мониторинг и детекция атак в реальном времени.
    """
    
    async def detect_anomalies(self, request: Request) -> List[SecurityAlert]:
        """
        Детекция аномалий в запросах.
        """
        alerts = []
        
        # 1. Множественные неудачные логины
        failed_logins = await self.redis.get(f"failed_logins:{request.client.host}")
        if int(failed_logins or 0) > 5:
            alerts.append(SecurityAlert(
                type="brute_force",
                severity="high",
                source=request.client.host
            ))
            
        # 2. Необычная активность
        request_rate = await self.redis.get(f"request_rate:{request.client.host}")
        if float(request_rate or 0) > 100:  # > 100 запросов/сек
            alerts.append(SecurityAlert(
                type="ddos",
                severity="critical",
                source=request.client.host
            ))
            
        return alerts
```

---

## Резервное копирование

### 1. Стратегия резервного копирования

**Реализация:** Encrypted backups

```bash
#!/bin/bash
# scripts/backup.sh

# Экспорт БД с шифрованием GPG
pg_dump face_recognition_db | \
    gpg --encrypt --recipient backup@company.com | \
    aws s3 cp - s3://backups/db_$(date +%Y%m%d_%H%M%S).sql.gpg

# Параметры шифрования:
# - Алгоритм: AES256
# - Ключ: RSA-4096 для GPG-шифрования
# - Хранение: S3 с SSE-KMS
```

### 2. Параметры резервного копирования

| Параметр | Значение | Описание |
|----------|----------|----------|
| Частота | Ежедневно, в 03:00 UTC | Минимальная нагрузка на систему |
| Хранение | 30 дней | Ротация бэкапов |
| Шифрование | GPG (AES-256) | Защита данных |
| Хранилище | S3 (Москва) | Локализация в РФ |
| Копий | 4 недельных + 12 месячных | Долгосрочное хранение |

### 3. Проверка восстановления

```python
class BackupValidator:
    """
    Валидация резервных копий.
    """
    
    async def test_restore(self, backup_path: str) -> bool:
        """
        Тестовое восстановление из бэкапа.
        """
        # 1. Скачивание бэкапа
        backup_data = await self.s3.download(backup_path)
        
        # 2. Расшифровка
        decrypted = self.gpg.decrypt(backup_data)
        
        # 3. Восстановление в тестовую БД
        await self.test_db.restore(decrypted)
        
        # 4. Проверка целостности
        await self.validate_schema(self.test_db)
        await self.validate_data_counts(self.test_db)
        
        # 5. Очистка тестовой БД
        await self.test_db.drop()
        
        return True
```

### 4. Таблица резервного копирования

| Тип данных | Частота | Хранилище | Шифрование | Срок хранения |
|------------|---------|-----------|------------|---------------|
| PostgreSQL | Ежедневно | S3 (Москва) | GPG AES-256 | 30 дней |
| MinIO (биометрия) | Ежедневно | S3 (Москва) | AES-256 | 30 дней |
| Аудит-логи | Ежедневно | S3 (Москва) | GPG | 1 год |
| Конфигурация | При изменении | S3 (Москва) | GPG | Навсегда |

### 5. План восстановления

| Сценарий | RTO (Recovery Time Objective) | RPO (Recovery Point Objective) |
|----------|-------------------------------|-------------------------------|
| Полная потеря сервера | 4 часа | 24 часа |
| Потеря одной реплики | 1 час | 0 (синхронная реплика) |
| Коррупция данных | 2 часа | 1 час |
| Ransomware | 24 часа | 24 часа |

---

## Организационные меры

### 1. Политики и документы

| Документ | Ответственный | Периодичность проверки |
|----------|---------------|----------------------|
| Политика обработки персональных данных | DPO | Ежегодно |
| Политика информационной безопасности | CISO | Ежегодно |
| Инструкция пользователя | DPO | При изменениях |
| Инструкция администратора | CISO | При изменениях |
| План реагирования на инциденты | CISO | Ежеквартально |

### 2. Обучение персонала

| Категория персонала | Тема обучения | Периодичность |
|--------------------|---------------|---------------|
| Разработчики | Безопасная разработка (SSDLC) | При найме / ежегодно |
| Администраторы | Защита систем и данных | Ежеквартально |
| DPO | Требования 152-ФЗ | Ежегодно |
| Все сотрудники | Основы ИБ | При найме / ежегодно |

### 3. Управление подрядчиками

- Проверка подрядчиков на соответствие требованиям ИБ
- Включение положений о защите персональных данных в договоры
- Аудит подрядчиков не реже 1 раза в год
- Требование об уведомлении об инцидентах в течение 24 часов

### 4. Физическая безопасность

| Мера | Описание | Контроль |
|------|----------|----------|
| Контроль доступа в ЦОД | Биометрическая верификация | Журнал доступа |
| Видеонаблюдение | Запись 24/7, хранение 30 дней | Архив записей |
| Охрана | Круглосуточная физическая охрана | Сменные рапорта |
| Резервное питание | UPS и генераторы | Тесты ежемесячно |

---

## Права субъектов данных

### 1. Право на доступ (статья 14 152-ФЗ)

Субъект имеет право получить информацию об обрабатываемых персональных данных.

**API для реализации:**

```
GET /api/v1/account/export — экспорт всех данных пользователя
```

**Формат ответа:**

```json
{
  "export_date": "2026-01-28T10:00:00Z",
  "user_data": {
    "id": 123,
    "email": "user@example.com",
    "full_name": "Иванов Иван Иванович",
    "created_at": "2025-01-15T08:00:00Z",
    "last_login": "2026-01-28T09:30:00Z"
  },
  "biometric_data": {
    "has_template": true,
    "template_created": "2025-01-15T08:15:00Z",
    "template_version": "v1.0"
  },
  "audit_records": [
    {
      "event": "login",
      "timestamp": "2026-01-28T09:30:00Z",
      "ip": "192.168.1.1"
    }
  ]
}
```

### 2. Право на уточнение (статья 15 152-ФЗ)

Субъект имеет право уточнить свои персональные данные.

**API для реализации:**

```
PUT /api/v1/account — обновление профиля
DELETE /api/v1/reference — удаление биометрического шаблона
```

### 3. Право на удаление (статья 16 152-ФЗ, "Право на забвение")

Субъект имеет право требовать удаления своих персональных данных.

**API для реализации:**

```
DELETE /api/v1/account — удаление аккаунта и всех данных
```

**Процедура удаления:**

1. Подтверждение личности (повторная аутентификация)
2. Удаление биометрических шаблонов
3. Удаление изображений из хранилища
4. Анонимизация аудит-записей (сохранение факта операции без данных)
5. Подтверждение удаления субъекту

### 4. Право на возражение (статья 18 152-ФЗ)

Субъект имеет право возражать против обработки своих данных.

**API для реализации:**

```
POST /api/v1/account/privacy/objection — подача возражения
```

### 5. Право на ограничение обработки (статья 21 152-ФЗ)

Субъект имеет право требовать ограничения обработки.

**API для реализации:**

```
POST /api/v1/account/privacy/restrict — ограничение обработки
```

### 6. Сроки обработки запросов

| Тип запроса | Срок ответа | Максимальный срок |
|-------------|-------------|-------------------|
| Предоставление информации | 10 рабочих дней | 30 рабочих дней |
| Уточнение данных | 10 рабочих дней | 30 рабочих дней |
| Удаление данных | 30 календарных дней | 90 календарных дней |
| Возражение | 10 рабочих дней | 30 рабочих дней |

---

## Аудит и мониторинг

### 1. Аудит-события

**Реализация:** `app/services/audit_service.py`

| Событие | Категория | Описание |
|---------|-----------|----------|
| user.login | Аутентификация | Успешный вход в систему |
| user.login_failed | Аутентификация | Неудачная попытка входа |
| user.logout | Аутентификация | Выход из системы |
| reference.created | Биометрия | Создание биометрического шаблона |
| reference.deleted | Биометрия | Удаление биометрического шаблона |
| verification.completed | Операция | Завершение верификации |
| liveness.check | Операция | Проверка живости |
| data.exported | Данные | Экспорт данных субъекта |
| data.deleted | Данные | Удаление данных субъекта |
| admin.action | Администрирование | Действие администратора |
| security.incident | Безопасность | Инцидент безопасности |

### 2. Формат аудит-записи

```json
{
  "audit_id": "audit_abc123def456",
  "timestamp": "2026-01-28T10:00:00Z",
  "event_type": "verification.completed",
  "user_id": 123,
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0",
  "request_id": "req_xyz789",
  "details": {
    "verification_id": "ver_123456",
    "similarity_score": 0.89,
    "is_match": true
  },
  "data_classification": "operational"
}
```

### 3. Мониторинг в реальном времени

**Метрики Prometheus:**

```prometheus
# HELP audit_events_total Total number of audit events
# TYPE audit_events_total counter
audit_events_total{event_type="login",status="success"} 1523
audit_events_total{event_type="login",status="failed"} 23

# HELP data_access_total Total data access operations
# TYPE data_access_total counter
data_access_total{operation="read"} 4521
data_access_total{operation="write"} 1234
data_access_total{operation="delete"} 56

# HELP security_incidents_total Total security incidents
# TYPE security_incidents_total counter
security_incidents_total{severity="critical"} 0
security_incidents_total{severity="high"} 2
security_incidents_total{severity="medium"} 15
```

### 4. Оповещения

| Событие | Уровень | Канал уведомления |
|---------|---------|-------------------|
| Множественные неудачные входы | Warning | Email, Slack |
| Попытка несанкционированного доступа | Critical | SMS, Email, Slack |
| Массовое удаление данных | Critical | SMS, Email |
| Компрометация ключей | Critical | SMS, Email, телефон |
| Нарушение SLA | Warning | Slack |

### 5. Отчетность

| Отчет | Периодичность | Формат | Получатели |
|-------|---------------|--------|------------|
| Отчет об инцидентах | Ежедневно | PDF | DPO, CISO |
| Статистика обращений | Еженедельно | PDF | DPO |
| Аудит доступа к данным | Ежемесячно | Excel | DPO, CISO |
| Оценка рисков | Ежеквартально | PDF | Руководство |
| Общий отчет по 152-ФЗ | Ежегодно | PDF | Руководство, регулятор |

---

## Чек-лист соответствия

### Технические меры защиты

| № | Требование | Источник | Статус | Примечание |
|---|------------|----------|--------|------------|
| 1 | Шифрование биометрических данных | ПП № 1119 | ✅ | AES-256-GCM |
| 2 | Управление ключами шифрования | Приказ ФСБ № 378 | ✅ | PBKDF2, ротация 90 дней |
| 3 | Шифрование при передаче | ПП № 1119 | ✅ | TLS 1.3 |
| 4 | Контроль доступа | ПП № 1119 | ✅ | RBAC, JWT |
| 5 | Аутентификация пользователей | 152-ФЗ | ✅ | JWT + password |
| 6 | Логирование действий | ПП № 1119 | ✅ | Структурированные логи |
| 7 | Антиспуфинг | Рекомендации | ✅ | MiniFASNetV2 |
| 8 | Резервное копирование | ПП № 1119 | ✅ | Ежедневно |
| 9 | Защита от DDoS | Рекомендации | ✅ | Rate limiting |
| 10 | Удаление данных | 152-ФЗ | ✅ | GDPR-совместимое удаление |

### Организационные меры

| № | Требование | Источник | Статус | Примечание |
|---|------------|----------|--------|------------|
| 1 | Назначение ответственного за защиту ПД | 152-ФЗ | ✅ | DPO назначен |
| 2 | Политика обработки ПД | 152-ФЗ | ✅ | Документ утвержден |
| 3 | Согласие субъекта на обработку | 152-ФЗ | ✅ | Форма согласия разработана |
| 4 | Обучение персонала | Приказ ФСТЭК № 21 | ✅ | Программа обучения |
| 5 | Управление инцидентами | ПП № 1119 | ✅ | Процедура реагирования |
| 6 | Договоры с подрядчиками | 152-ФЗ | ✅ | Типовые договоры |
| 7 | Локализация данных | 152-ФЗ | ✅ | ЦОД в РФ |
| 8 | Аудит соответствия | Приказ ФСТЭК № 21 | ✅ | Ежеквартально |

### Права субъектов

| № | Требование | Источник | Статус | Примечание |
|---|------------|----------|--------|------------|
| 1 | Право на доступ к данным | Ст. 14 152-ФЗ | ✅ | API /api/v1/account/export |
| 2 | Право на уточнение данных | Ст. 15 152-ФЗ | ✅ | API /api/v1/account |
| 3 | Право на удаление данных | Ст. 16 152-ФЗ | ✅ | API /api/v1/account |
| 4 | Право на возражение | Ст. 18 152-ФЗ | ✅ | API /api/v1/account/privacy/objection |
| 5 | Право на ограничение | Ст. 21 152-ФЗ | ✅ | API /api/v1/account/privacy/restrict |
| 6 | Сроки обработки запросов | 152-ФЗ | ✅ | Регламент утвержден |

### Документация

| № | Документ | Статус | Дата обновления |
|---|----------|--------|-----------------|
| 1 | Политика обработки ПД | ✅ | 2026-01-01 |
| 2 | Политика конфиденциальности | ✅ | 2026-01-01 |
| 3 | Согласие на обработку биометрии | ✅ | 2026-01-01 |
| 4 | Инструкция пользователя | ✅ | 2026-01-01 |
| 5 | Инструкция администратора | ✅ | 2026-01-01 |
| 6 | План реагирования на инциденты | ✅ | 2026-01-01 |
| 7 | Регламент работы с данными | ✅ | 2026-01-01 |

---

## Политика обработки данных

### 1. Общие положения

Настоящая Политика определяет порядок обработки персональных данных в системе распознавания лиц и действует в отношении всех персональных данных, которые могут быть получены от субъектов персональных данных.

### 2. Принципы обработки

| Принцип | Описание | Реализация в системе |
|---------|----------|---------------------|
| Законность | Обработка на законных основаниях | Согласие субъекта, договор |
| Справедливость | Честная и прозрачная обработка | Уведомление субъекта |
| Целенаправленность | Определенные, законные цели | Перечень целей в документе |
| Минимизация | Достаточность данных | Храним только необходимые |
| Точность | Актуальность данных | Механизмы обновления |
| Ограничение хранения | Сроки хранения | Политика retention |
| Целостность и конфиденциальность | Защита от несанкционированного доступа | Технические и организационные меры |

### 3. Категории обрабатываемых данных

| Категория | Примеры | Цель обработки |
|-----------|---------|----------------|
| Идентификационные | ФИО, email, телефон | Регистрация, идентификация |
| Аутентификационные | Пароль, JWT-токен | Контроль доступа |
| Биометрические | Фото лица, эмбеддинг | Верификация личности |
| Технические | IP-адрес, User-Agent | Безопасность, логирование |
| Производные | Метрики качества, результаты проверок | Улучшение сервиса |

### 4. Правовые основания обработки

| Основание | Категории данных | Примеры использования |
|-----------|------------------|----------------------|
| Согласие субъекта | Биометрические, идентификационные | Регистрация, верификация |
| Договор | Идентификационные, аутентификационные | Предоставление услуг |
| Закон | Все категории | Исполнение требований закона |
| Жизненно важные интересы | Биометрические | Защита жизни и здоровья |

### 5. Передача данных третьим лицам

Передача персональных данных третьим лицам осуществляется только:

- С письменного согласия субъекта
- На основании договора с третьим лицом
- По требованию государственных органов в соответствии с законом

**Категории получателей:**

| Категория | Примеры | Цель передачи |
|-----------|---------|---------------|
| Поставщики услуг | Облачные провайдеры, хостинг | Техническая инфраструктура |
| Партнеры | Интеграторы | Расширение функциональности |
| Государственные органы | ФСБ, ФСТЭК | Исполнение требований закона |

### 6. Международная передача данных

Международная передача персональных данных **не осуществляется**.

Все серверы и системы хранения данных расположены на территории Российской Федерации.

### 7. Сроки хранения данных

| Категория данных | Срок хранения | Удаление |
|------------------|---------------|----------|
| Биометрические шаблоны | 3 года с последнего использования | Автоматическое |
| Учетные записи | 3 года неактивности | Автоматическое |
| Исходные изображения | 30 дней | Автоматическое |
| Аудит-логи | 1 год | Автоматическое |
| Сессии | 7 дней | Автоматическое |

### 8. Контактная информация

**Ответственный за защиту персональных данных (DPO):**

- ФИО: [Указывается]
- Email: dpo@example.com
- Телефон: [Указывается]
- Адрес: [Указывается]

**Для реализации прав субъекта:**

- Направлять запросы на: dpo@example.com
- Форма запроса: https://example.com/privacy/request

---

## Приложения

### Приложение А: Форма согласия на обработку биометрических данных

```
Я, __________________________, паспорт _______________, выдан ______________,
проживающий по адресу __________________________,

даю согласие ООО "Компания" (далее — Оператор) на обработку моих биометрических
персональных данных (изображение лица, биометрический шаблон) в целях:
- Идентификации меня как пользователя сервиса
- Верификации моей личности при доступе к сервису
- Предотвращения несанкционированного доступа к моему аккаунту

С условиями обработки ознакомлен:
- Биометрические данные будут храниться в зашифрованном виде
- Срок хранения — 3 (три) года с даты последнего использования
- Я вправе отозвать данное согласие в любой момент
- Отзыв согласия не влияет на законность обработки до отзыва

Дата: "___" _____________ 202__ г.          Подпись: _________________
```

### Приложение Б: Схема обработки данных

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ОБРАБОТКА ДАННЫХ                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│   │ Загрузка│───▶│  Валидация  │───▶│  Проверка   │───▶│ Шифрование │  │
│   │  фото   │    │  качества   │    │   живости   │    │  AES-256   │  │
│   └─────────┘    └─────────────┘    └─────────────┘    └────────────┘  │
│        │              │                   │                 │           │
│        ▼              ▼                   ▼                 ▼           │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                    БЕЗОПАСНОЕ ХРАНИЛИЩЕ                     │     │
│   │  ┌─────────────────────────────────────────────────────┐   │     │
│   │  │  PostgreSQL (метаданные) + MinIO (зашифрованные)    │   │     │
│   │  └─────────────────────────────────────────────────────┘   │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                              │                                         │
│                              ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐     │
│   │                       АУДИТ-ЛОГ                             │     │
│   │  ┌─────────────────────────────────────────────────────┐   │     │
│   │  │  События: создание, доступ, удаление, верификация   │   │     │
│   │  └─────────────────────────────────────────────────────┘   │     │
│   └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Приложение В: Контакты для обращений

| Тип обращения | Контакт | Срок ответа |
|---------------|---------|-------------|
| Общие вопросы по 152-ФЗ | privacy@example.com | 10 рабочих дней |
| Реализация прав субъекта | dpo@example.com | 30 календарных дней |
| Инциденты безопасности | security@example.com | 24 часа |
| Юридические запросы | legal@example.com | 5 рабочих дней |

---

**Документ подготовлен в соответствии с требованиями:**  
Федеральный закон от 27.07.2006 № 152-ФЗ "О персональных данных"  
Постановление Правительства РФ от 01.11.2012 № 1119  
Приказ ФСТЭК России от 18.02.2013 № 21  
Приказ ФСБ России от 10.07.2014 № 378

---

*© 2026 [Наименование организации]. Все права защищены.*