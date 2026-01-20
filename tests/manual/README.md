# Manual Testing Guide

## Подготовка тестовых изображений

### 1. Соберите набор фотографий:

**Категория A: Реальные лица (Real Faces)**
- `person_a_front.jpg` - фото спереди
- `person_a_angle.jpg` - фото под углом
- `person_a_smile.jpg` - фото с улыбкой
- `person_a_serious.jpg` - серьезное выражение
- `person_b_*.jpg` - то же для другого человека

**Категория B: Вариации условий**
- `good_lighting.jpg` - хорошее освещение
- `bad_lighting.jpg` - плохое освещение
- `backlight.jpg` - контровой свет
- `shadow.jpg` - сильные тени
- `high_quality.jpg` - высокое разрешение
- `low_quality.jpg` - низкое разрешение

**Категория C: Spoofing Attacks**
- `screen_photo.jpg` - фото с экрана телефона
- `print_photo.jpg` - фото распечатанного фото
- `video_frame.jpg` - кадр из видео

**Категория D: Edge Cases**
- `no_face.jpg` - изображение без лица
- `multiple_faces.jpg` - несколько лиц
- `partial_face.jpg` - частично видимое лицо
- `occluded_face.jpg` - лицо с окклюзией (очки, маска)

---

## Тестовые сценарии

### Сценарий 1: Базовая верификация

**Шаги:**
1. Запустите сервер: `python -m app.main`
2. Откройте Swagger UI: `http://localhost:8000/docs`
3. Зарегистрируйте пользователя через `/api/v1/auth/register`
4. Войдите через `/api/v1/auth/login` и скопируйте токен
5. Нажмите "Authorize" и вставьте токен
6. Создайте reference через `/api/v1/reference`:
   - Загрузите `person_a_front.jpg` (в base64)
   - Запомните `reference_id`
7. Верифицируйте через `/api/v1/verify`:
   - Используйте `person_a_angle.jpg`
   - Проверьте `similarity_score` (должен быть > 0.7)

**Ожидаемый результат:**
- `verified: true`
- `similarity_score: 0.75-0.95`
- `confidence: 0.8-1.0`

---

### Сценарий 2: Liveness Detection

**Шаги:**
1. Сделайте селфи с телефона (реальное фото)
2. Вызовите `/api/v1/liveness` с вашим фото
3. Проверьте `liveness_detected` и `confidence`
4. Сделайте фото экрана телефона с тем же селфи
5. Вызовите `/api/v1/liveness/anti-spoofing/advanced`
6. Проверьте, что система обнаружила spoofing

**Ожидаемый результат:**
- Реальное фото: `liveness_detected: true`, `confidence > 0.7`
- Фото экрана: `liveness_detected: false`, `anti_spoofing_score < 0.5`

---

### Сценарий 3: Различные условия освещения

**Шаги:**
1. Создайте reference с `good_lighting.jpg`
2. Верифицируйте с `bad_lighting.jpg` того же человека
3. Проверьте `lighting_analysis` в ответе
4. Проверьте recommendations

**Ожидаемый результат:**
- Верификация должна пройти (но с lower confidence)
- `lighting_analysis.overall_quality < 0.5` для плохого освещения
- Recommendations должны содержать советы по улучшению

---

### Сценарий 4: Edge Cases

**Тест 4.1: Нет лица**
- Загрузите `no_face.jpg`
- Ожидается: `face_detected: false`, `error: "No face detected"`

**Тест 4.2: Несколько лиц**
- Загрузите `multiple_faces.jpg`
- Ожидается: `face_detected: true`, `multiple_faces: true`

**Тест 4.3: Частичное лицо**
- Загрузите `partial_face.jpg`
- Ожидается: `face_detected: false` или low `quality_score`

---

## Чеклист для ручного тестирования

### Функциональность
- [ ] Регистрация и вход работают
- [ ] Создание reference с хорошим фото
- [ ] Верификация одного и того же человека (разные фото)
- [ ] Отказ в верификации для разных людей
- [ ] Liveness detection для реальных лиц
- [ ] Обнаружение spoofing attacks
- [ ] Проверка качества изображения
- [ ] Multiple references для одного пользователя

### Performance
- [ ] Embedding generation < 500ms
- [ ] Verification < 1s
- [ ] Liveness check < 2s
- [ ] API response time < 100ms (без ML обработки)

### Security
- [ ] Токен аутентификации обязателен
- [ ] Невалидный токен → 401
- [ ] Эмбеддинги не возвращаются в API ответах
- [ ] Большие файлы отклоняются

### Usability
- [ ] Понятные error messages
- [ ] Recommendations полезны
- [ ] Swagger UI документация корректна

---

## Репортинг багов

При обнаружении бага создайте issue с:

**Заголовок:** [BUG] Краткое описание

**Описание:**
- Шаги для воспроизведения
- Ожидаемый результат
- Фактический результат
- Скриншоты/логи
- Версия ПО
- Окружение (OS, Python version, GPU/CPU)

**Пример:**
