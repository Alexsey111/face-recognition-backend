"""
Eye Blink Detection using Eye Aspect Ratio (EAR).
Based on: "Real-Time Eye Blink Detection using Facial Landmarks" by Soukupová and Čech (2016)
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Вычисление Eye Aspect Ratio (EAR).
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Где p1-p6 это координаты 6 точек глаза по порядку:
    p1, p4 - внешние углы (горизонтальная линия)
    p2, p3, p5, p6 - вертикальные точки
    
    Args:
        eye_landmarks: Массив формы (6, 2) с координатами глаза
        
    Returns:
        float: EAR значение (обычно 0.2-0.3 для открытого, <0.2 для закрытого глаза)
    """
    if len(eye_landmarks) != 6:
        raise ValueError(f"Expected 6 landmarks, got {len(eye_landmarks)}")
    
    # Вертикальные расстояния
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Горизонтальное расстояние
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)  # +epsilon для избежания деления на 0
    
    return ear


def extract_eye_landmarks_from_68(
    landmarks_68: np.ndarray,
    eye: str = "both"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Извлечение координат глаз из 68-point landmarks (dlib format).
    
    Индексы в 68-point landmarks:
    - Левый глаз: 36-41 (6 точек)
    - Правый глаз: 42-47 (6 точек)
    
    Args:
        landmarks_68: Массив формы (68, 2) с координатами лица
        eye: "left", "right", или "both"
        
    Returns:
        Tuple[left_eye, right_eye]: Координаты глаз формы (6, 2)
    """
    if landmarks_68.shape[0] != 68:
        raise ValueError(f"Expected 68 landmarks, got {landmarks_68.shape[0]}")
    
    left_eye = landmarks_68[36:42] if eye in ["left", "both"] else None
    right_eye = landmarks_68[42:48] if eye in ["right", "both"] else None
    
    return left_eye, right_eye


class BlinkDetector:
    """
    Детектор моргания на основе анализа последовательности кадров.
    
    Алгоритм:
    1. Вычисляем EAR для каждого кадра
    2. Детектируем последовательность: открыт → закрыт → открыт
    3. Проверяем длительность закрытия (должна быть в диапазоне 100-400ms)
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames_closed: int = 2,
        min_blink_duration_ms: float = 100.0,
        max_blink_duration_ms: float = 400.0,
        fps: float = 30.0,
    ):
        """
        Args:
            ear_threshold: Порог EAR для определения закрытого глаза
            consecutive_frames_closed: Минимум кадров с закрытым глазом
            min_blink_duration_ms: Минимальная длительность моргания (мс)
            max_blink_duration_ms: Максимальная длительность моргания (мс)
            fps: FPS видео/потока
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames_closed = consecutive_frames_closed
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = max_blink_duration_ms
        self.fps = fps
        
        # Состояние
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_frame = None
        
        # История EAR
        self.ear_history: List[float] = []
    
    def reset(self):
        """Сброс состояния детектора."""
        self.frame_counter = 0
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_start_frame = None
        self.ear_history.clear()
    
    def process_frame(self, eye_landmarks: np.ndarray) -> bool:
        """
        Обработка одного кадра.
        
        Args:
            eye_landmarks: Координаты глаза (6, 2)
            
        Returns:
            bool: True если обнаружено завершенное моргание
        """
        # Вычисляем EAR
        ear = calculate_ear(eye_landmarks)
        self.ear_history.append(ear)
        
        blink_detected = False
        
        # Проверяем закрыт ли глаз
        if ear < self.ear_threshold:
            self.frame_counter += 1
            
            # Начало моргания
            if not self.is_blinking:
                self.is_blinking = True
                self.blink_start_frame = len(self.ear_history) - 1
        else:
            # Глаз открыт
            if self.is_blinking:
                # Проверяем было ли достаточно кадров с закрытым глазом
                if self.frame_counter >= self.consecutive_frames_closed:
                    # Вычисляем длительность моргания
                    blink_duration_ms = (self.frame_counter / self.fps) * 1000
                    
                    # Проверяем диапазон длительности
                    if self.min_blink_duration_ms <= blink_duration_ms <= self.max_blink_duration_ms:
                        self.blink_counter += 1
                        blink_detected = True
                
                # Сбрасываем состояние
                self.is_blinking = False
                self.frame_counter = 0
                self.blink_start_frame = None
            else:
                self.frame_counter = 0
        
        return blink_detected
    
    def get_average_ear(self, window_size: int = 10) -> float:
        """Получение среднего EAR за последние N кадров."""
        if not self.ear_history:
            return 0.0
        
        recent = self.ear_history[-window_size:]
        return float(np.mean(recent))
    
    def get_stats(self) -> dict:
        """Получение статистики детектора."""
        return {
            "total_blinks": self.blink_counter,
            "current_ear": self.ear_history[-1] if self.ear_history else 0.0,
            "avg_ear": self.get_average_ear(),
            "is_currently_blinking": self.is_blinking,
            "frames_processed": len(self.ear_history),
        }


def detect_blinks_in_sequence(
    landmarks_sequence: List[np.ndarray],
    fps: float = 30.0,
    min_blinks: int = 1,
) -> Tuple[bool, int, dict]:
    """
    Детекция морганий в последовательности кадров.
    
    Args:
        landmarks_sequence: Список массивов с 68-point landmarks для каждого кадра
        fps: FPS видео
        min_blinks: Минимальное количество морганий для успеха
        
    Returns:
        Tuple[success, blink_count, stats]:
            - success: True если обнаружено достаточно морганий
            - blink_count: Количество обнаруженных морганий
            - stats: Детальная статистика
    """
    detector = BlinkDetector(fps=fps)
    blink_frames = []
    
    for frame_idx, landmarks_68 in enumerate(landmarks_sequence):
        # Извлекаем координаты обоих глаз
        left_eye, right_eye = extract_eye_landmarks_from_68(landmarks_68)
        
        # Проверяем оба глаза (используем среднее)
        if left_eye is not None and right_eye is not None:
            # Усредняем EAR для обоих глаз
            left_blink = detector.process_frame(left_eye)
            right_blink = detector.process_frame(right_eye)
            
            # Моргание засчитывается если хотя бы один глаз моргнул
            if left_blink or right_blink:
                blink_frames.append(frame_idx)
    
    stats = detector.get_stats()
    stats["blink_frames"] = blink_frames
    stats["fps"] = fps
    stats["total_frames"] = len(landmarks_sequence)
    
    success = stats["total_blinks"] >= min_blinks
    
    return success, stats["total_blinks"], stats


# Пример использования
if __name__ == "__main__":
    # Симуляция последовательности (для тестирования)
    import random
    
    # Генерируем фейковые landmarks
    def generate_fake_landmarks(eye_open: bool) -> np.ndarray:
        """Генерация фейковых landmarks для тестирования."""
        landmarks = np.random.rand(68, 2) * 100
        
        # Левый глаз (36-41)
        if eye_open:
            landmarks[37, 1] = 50  # Открытый глаз
            landmarks[38, 1] = 50
            landmarks[40, 1] = 50
            landmarks[41, 1] = 50
        else:
            landmarks[37, 1] = 55  # Закрытый глаз (ближе к горизонтали)
            landmarks[38, 1] = 55
            landmarks[40, 1] = 55
            landmarks[41, 1] = 55
        
        return landmarks
    
    # Симуляция последовательности: открыт-закрыт-открыт
    sequence = []
    for i in range(30):
        if 10 <= i <= 13:  # Кадры 10-13: закрыт (моргание)
            sequence.append(generate_fake_landmarks(eye_open=False))
        else:
            sequence.append(generate_fake_landmarks(eye_open=True))
    
    success, blink_count, stats = detect_blinks_in_sequence(sequence)
    
    print(f"Blink detection: {'SUCCESS' if success else 'FAILED'}")
    print(f"Blinks detected: {blink_count}")
    print(f"Stats: {stats}")
