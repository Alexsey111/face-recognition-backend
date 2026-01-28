# Тестовые датасеты для валидации метрик

Структура каталогов для тестовых данных, используемых при валидации системы верификации лиц.

## Структура датасета

```
tests/datasets/validation_dataset/
├── genuine_pairs/      # Пары изображений одного человека (для тестирования FRR)
├── impostor_pairs/      # Изображения разных людей (для тестирования FAR)
├── live_faces/         # Фотографии живых лиц (для Liveness TP)
└── spoofed_faces/      # Поддельные изображения (печать, экран, маска) для Liveness TN
```

## Требования к датасету

### 1. genuine_pairs/

**Назначение**: Тестирование False Reject Rate (FRR)

**Требование ТЗ**: FRR < 1-3%

**Структура**: Пары фотографий одного человека

**Формат имен**: `personID_img1.jpg`, `personID_img2.jpg`

**Пример**:
```
person001_img1.jpg    # Фото человека 001, ракурс 1
person001_img2.jpg    # Фото человека 001, ракурс 2
person002_img1.jpg
person002_img2.jpg
```

**Рекомендации**:
- Минимум 100 пар (200 изображений)
- Разные условия: освещение, ракурсы, выражения лица
- Временной интервал между снимками (дни/месяцы)

---

### 2. impostor_pairs/

**Назначение**: Тестирование False Accept Rate (FAR)

**Требование ТЗ**: FAR < 0.1%

**Структура**: Фотографии разных людей

**Формат имен**: `personID_img.jpg`

**Пример**:
```
personA_img1.jpg
personB_img1.jpg
personC_img1.jpg
```

**Рекомендации**:
- Минимум 50 разных людей (2500+ комбинаций пар)
- Включить похожих людей (близнецы, родственники)
- Разнообразие по возрасту, полу, этничности

---

### 3. live_faces/

**Назначение**: Тестирование Liveness Detection (True Positives)

**Требование ТЗ**: Liveness Accuracy > 98%

**Структура**: Фотографии реальных живых лиц

**Формат имен**: `live_001.jpg`, `live_002.jpg`

**Рекомендации**:
- Минимум 100 изображений
- Снимки с камер смартфонов/веб-камер
- Различные условия освещения
- Естественные выражения лица

---

### 4. spoofed_faces/

**Назначение**: Тестирование Liveness Detection (True Negatives)

**Требование ТЗ**: Liveness Accuracy > 98%

**Типы атак**:
- `print_*.jpg` — фото с бумажной распечатки
- `screen_*.jpg` — фото с экрана монитора/смартфона
- `mask_*.jpg` — реалистичные маски
- `video_*.jpg` — кадры из видеозаписи на экране

**Рекомендации**:
- Минимум 100 изображений (25 каждого типа)
- Высокое качество печати/экранов для реалистичности
- 3D-маски, если доступны

---

## Источники публичных датасетов

### Для верификации (FAR/FRR)

| Датасет | Описание | Ссылка |
|---------|-----------|--------|
| LFW | Labeled Faces in the Wild | http://vis-www.cs.umass.edu/lfw/ |
| CelebA | Large-scaleCelebFaces Attributes | http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html |
| VGGFace2 | Large-scale face recognition | https://github.com/ox-vgg/vgg_face2 |

### Для Liveness

| Датасет | Описание | Ссылка |
|---------|-----------|--------|
| NUAA | Photograph Imposter Database | http://parnec.nuaa.edu.cn/ |
| CASIA | Face Anti-Spoofing Database | http://www.cbsr.ia.ac.cn/ |
| Replay-Attack | Replay Attack Database | https://www.idiap.ch/dataset/replayattack |
| OULU-NPU | OULU Anti-Spoofing Database | https://sites.google.com/site/oulunpudatabase/ |

---

## Использование

```bash
# Запуск валидации
python scripts/validate_metrics.py \
  --dataset tests/datasets/validation_dataset \
  --threshold 0.6 \
  --output validation_results.json
```

---

## Результаты

Скрипт генерирует `validation_results.json` с подробными метриками:

```json
{
  "timestamp": "2026-01-27T14:30:00",
  "threshold": 0.6,
  "verification": {
    "far": {
      "value": 0.08,
      "passed": true
    },
    "frr": {
      "value": 1.5,
      "passed": true
    }
  },
  "liveness": {
    "accuracy": {
      "value": 98.5,
      "passed": true
    }
  },
  "summary": {
    "overall_status": "PASSED",
    "compliance_152_fz": true
  }
}
```

---

## Требования Compliance (152-ФЗ)

| Метрика | Требование | Статус |
|---------|------------|--------|
| FAR | < 0.1% | ✅ |
| FRR | < 3% | ✅ |
| Accuracy | > 99% | ✅ |
| Liveness | > 98% | ✅ |