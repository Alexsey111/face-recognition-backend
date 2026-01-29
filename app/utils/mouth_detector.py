"""
Mouth Detection Utilities for Active Liveness.

- Smile detection using Mouth Aspect Ratio (MAR)
- Open mouth detection
- Mouth shape analysis

MAR (Mouth Aspect Ratio):
$MAR = \frac{||p_{61} - p_{67}|| + ||p_{62} - p_{66}|| + ||p_{63} - p_{65}||}{3 \times ||p_{60} - p_{64}||}$

Где p60-p67 — точки рта (68-point landmarks).
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Индексы точек рта для 68-point landmarks
MOUTH_OUTER_INDICES = list(range(48, 60))  # Точки 48-59
MOUTH_INNER_INDICES = list(range(60, 68))  # Точки 60-67

# MAR пороги
DEFAULT_SMILE_MAR_THRESHOLD = 0.5
DEFAULT_OPEN_MOUTH_MAR_THRESHOLD = 0.6


# ============================================================================
# Mouth Aspect Ratio
# ============================================================================


def calculate_mouth_aspect_ratio(
    landmarks: np.ndarray,
    mouth_indices: List[int] = MOUTH_INNER_INDICES,
) -> float:
    """
    Вычисление Mouth Aspect Ratio (MAR).

    Формула аналогична EAR:
    MAR = (||p61-p67|| + ||p62-p66|| + ||p63-p65||) / (3 * ||p60-p64||)

    Args:
        landmarks: Массив 68 точек лица
        mouth_indices: Индексы точек рта (8 точек: 60-67)

    Returns:
        MAR значение
    """
    try:
        mouth_points = landmarks[mouth_indices]

        # Вертикальные расстояния (3 пары точек)
        A = np.linalg.norm(mouth_points[1] - mouth_points[7])  # p61-p67
        B = np.linalg.norm(mouth_points[2] - mouth_points[6])  # p62-p66
        C = np.linalg.norm(mouth_points[3] - mouth_points[5])  # p63-p65

        # Горизонтальное расстояние
        D = np.linalg.norm(mouth_points[0] - mouth_points[4])  # p60-p64

        if D < 1e-6:
            return 0.0

        mar = (A + B + C) / (3.0 * D)

        return float(mar)

    except Exception as e:
        logger.warning(f"Failed to calculate MAR: {e}")
        return 0.0


def calculate_mouth_width_height_ratio(landmarks: np.ndarray) -> float:
    """
    Соотношение ширины к высоте рта (для детекции улыбки).

    Улыбка: ширина увеличивается, высота уменьшается → ratio увеличивается
    """
    try:
        mouth_outer = landmarks[MOUTH_OUTER_INDICES]

        # Ширина рта
        width = np.linalg.norm(mouth_outer[0] - mouth_outer[6])  # p48-p54

        # Высота рта (средняя)
        top_middle = mouth_outer[3]  # p51
        bottom_middle = mouth_outer[9]  # p57
        height = np.linalg.norm(top_middle - bottom_middle)

        if height < 1e-6:
            return 0.0

        ratio = width / height

        return float(ratio)

    except Exception as e:
        logger.warning(f"Failed to calculate mouth w/h ratio: {e}")
        return 0.0


# ============================================================================
# Smile Detection
# ============================================================================


def detect_smile_in_sequence(
    landmarks_sequence: List[np.ndarray],
    min_intensity: float = DEFAULT_SMILE_MAR_THRESHOLD,
    min_frames_with_smile: int = 5,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Детекция улыбки в последовательности landmarks.

    Args:
        landmarks_sequence: Список массивов landmarks
        min_intensity: Минимальная интенсивность улыбки (MAR threshold)
        min_frames_with_smile: Минимум кадров с улыбкой

    Returns:
        (success, smile_intensity, stats)
    """
    if len(landmarks_sequence) < 3:
        return False, 0.0, {"error": "Too few frames"}

    mar_values = []
    smile_frames = []

    for i, landmarks in enumerate(landmarks_sequence):
        try:
            # Комбинированная метрика: MAR + width/height ratio
            mar = calculate_mouth_aspect_ratio(landmarks)
            wh_ratio = calculate_mouth_width_height_ratio(landmarks)

            mar_values.append(mar)

            # Улыбка: высокий MAR И высокий w/h ratio
            smile_score = mar * 0.6 + (wh_ratio / 5.0) * 0.4

            if smile_score > min_intensity:
                smile_frames.append(i)

        except Exception as e:
            logger.warning(f"Failed to process frame {i}: {e}")
            mar_values.append(0.0)

    # Максимальная интенсивность
    max_intensity = max(mar_values) if mar_values else 0.0

    # Успех
    success = len(smile_frames) >= min_frames_with_smile

    stats = {
        "smile_frames": smile_frames,
        "max_mar": round(max_intensity, 3),
        "avg_mar": round(np.mean(mar_values), 3) if mar_values else 0.0,
        "frames_with_smile": len(smile_frames),
        "total_frames": len(landmarks_sequence),
    }

    logger.info(
        f"Smile detection: success={success}, intensity={max_intensity:.3f}, "
        f"frames_with_smile={len(smile_frames)}/{len(landmarks_sequence)}"
    )

    return success, max_intensity, stats


# ============================================================================
# Open Mouth Detection
# ============================================================================


def detect_open_mouth_in_sequence(
    landmarks_sequence: List[np.ndarray],
    min_mar: float = DEFAULT_OPEN_MOUTH_MAR_THRESHOLD,
    min_frames_with_open_mouth: int = 5,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Детекция открытого рта в последовательности landmarks.

    Args:
        landmarks_sequence: Список массивов landmarks
        min_mar: Минимальный MAR для открытого рта
        min_frames_with_open_mouth: Минимум кадров с открытым ртом

    Returns:
        (success, max_mar, stats)
    """
    if len(landmarks_sequence) < 3:
        return False, 0.0, {"error": "Too few frames"}

    mar_values = []
    open_frames = []

    for i, landmarks in enumerate(landmarks_sequence):
        try:
            mar = calculate_mouth_aspect_ratio(landmarks)
            mar_values.append(mar)

            if mar > min_mar:
                open_frames.append(i)

        except Exception as e:
            logger.warning(f"Failed to process frame {i}: {e}")
            mar_values.append(0.0)

    max_mar = max(mar_values) if mar_values else 0.0

    success = len(open_frames) >= min_frames_with_open_mouth

    stats = {
        "open_frames": open_frames,
        "max_mar": round(max_mar, 3),
        "avg_mar": round(np.mean(mar_values), 3) if mar_values else 0.0,
        "frames_with_open_mouth": len(open_frames),
        "total_frames": len(landmarks_sequence),
    }

    logger.info(
        f"Open mouth detection: success={success}, max_mar={max_mar:.3f}, "
        f"open_frames={len(open_frames)}/{len(landmarks_sequence)}"
    )

    return success, max_mar, stats
