"""
Video processing utilities for active liveness detection.
"""

import io
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from decord import VideoReader
    from decord import cpu as decord_cpu

    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


def extract_frames_from_video(
    video_data: bytes,
    max_frames: int = 30,
    target_fps: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Извлечение кадров из видео.

    Args:
        video_data: Байты видео файла
        max_frames: Максимальное количество кадров
        target_fps: Целевой FPS (если None, используется оригинальный)
        target_size: Целевой размер кадра (width, height)

    Returns:
        List[np.ndarray]: Список кадров в формате RGB
    """
    if HAS_DECORD:
        return _extract_frames_decord(video_data, max_frames, target_fps, target_size)
    else:
        return _extract_frames_opencv(video_data, max_frames, target_fps, target_size)


def _extract_frames_decord(
    video_data: bytes,
    max_frames: int,
    target_fps: Optional[float],
    target_size: Optional[Tuple[int, int]],
) -> List[np.ndarray]:
    """Извлечение кадров с помощью decord (быстрее)."""
    try:
        # Сохраняем во временный файл
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name

        # Загружаем видео
        vr = VideoReader(tmp_path, ctx=decord_cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        # Вычисляем индексы кадров для извлечения
        if target_fps and target_fps < fps:
            # Downsampling по FPS
            frame_step = int(fps / target_fps)
            frame_indices = list(range(0, total_frames, frame_step))[:max_frames]
        else:
            # Равномерное распределение по всему видео
            frame_indices = np.linspace(
                0, total_frames - 1, min(max_frames, total_frames), dtype=int
            )

        # Извлекаем кадры
        frames = vr.get_batch(frame_indices).asnumpy()

        # Resize если нужно
        if target_size:
            frames_resized = []
            for frame in frames:
                frame_resized = cv2.resize(
                    frame, target_size, interpolation=cv2.INTER_LINEAR
                )
                frames_resized.append(frame_resized)
            frames = np.array(frames_resized)

        # Удаляем временный файл
        import os

        os.unlink(tmp_path)

        logger.info(f"Extracted {len(frames)} frames using decord (FPS: {fps:.1f})")
        return list(frames)

    except Exception as e:
        logger.error(f"Failed to extract frames with decord: {e}")
        return _extract_frames_opencv(video_data, max_frames, target_fps, target_size)


def _extract_frames_opencv(
    video_data: bytes,
    max_frames: int,
    target_fps: Optional[float],
    target_size: Optional[Tuple[int, int]],
) -> List[np.ndarray]:
    """Извлечение кадров с помощью OpenCV (fallback)."""
    try:
        import tempfile

        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name

        # Открываем видео
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        # Получаем параметры видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Вычисляем индексы кадров
        if target_fps and target_fps < fps:
            frame_step = int(fps / target_fps)
        else:
            frame_step = max(1, total_frames // max_frames)

        frames = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                # Конвертируем BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize если нужно
                if target_size:
                    frame_rgb = cv2.resize(
                        frame_rgb, target_size, interpolation=cv2.INTER_LINEAR
                    )

                frames.append(frame_rgb)

            frame_idx += 1

        cap.release()

        # Удаляем временный файл
        import os

        os.unlink(tmp_path)

        logger.info(f"Extracted {len(frames)} frames using OpenCV (FPS: {fps:.1f})")
        return frames

    except Exception as e:
        logger.error(f"Failed to extract frames with OpenCV: {e}")
        raise


def calculate_video_quality(frames: List[np.ndarray]) -> dict:
    """
    Оценка качества видео для liveness detection.

    Args:
        frames: Список кадров

    Returns:
        dict: Метрики качества
    """
    if not frames:
        return {"quality_score": 0.0, "issues": ["no_frames"]}

    issues = []

    # 1. Проверка количества кадров
    frame_count = len(frames)
    if frame_count < 10:
        issues.append("insufficient_frames")

    # 2. Проверка разрешения
    heights = [frame.shape[0] for frame in frames]
    widths = [frame.shape[1] for frame in frames]
    avg_height, avg_width = np.mean(heights), np.mean(widths)

    if avg_width < 224 or avg_height < 224:
        issues.append("low_resolution")

    # 3. Проверка резкости (Laplacian variance)
    sharpness_scores = []
    for frame in frames[:10]:  # Проверяем первые 10 кадров
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_scores.append(laplacian_var)

    avg_sharpness = np.mean(sharpness_scores)
    if avg_sharpness < 100:
        issues.append("blurry")

    # 4. Проверка освещения
    brightness_scores = []
    for frame in frames[:10]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        brightness_scores.append(brightness)

    avg_brightness = np.mean(brightness_scores)
    if avg_brightness < 50:
        issues.append("too_dark")
    elif avg_brightness > 200:
        issues.append("overexposed")

    # 5. Проверка стабильности (motion blur)
    if len(frames) >= 2:
        motion_scores = []
        for i in range(min(5, len(frames) - 1)):
            diff = cv2.absdiff(
                cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY),
            )
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)

        avg_motion = np.mean(motion_scores)
        if avg_motion > 50:
            issues.append("excessive_motion")
        elif avg_motion < 1:
            issues.append("static_video")

    # Итоговая оценка качества (0-1)
    quality_penalties = {
        "insufficient_frames": 0.3,
        "low_resolution": 0.2,
        "blurry": 0.2,
        "too_dark": 0.15,
        "overexposed": 0.15,
        "excessive_motion": 0.1,
        "static_video": 0.1,
    }

    quality_score = 1.0
    for issue in issues:
        quality_score -= quality_penalties.get(issue, 0.1)

    quality_score = max(0.0, min(1.0, quality_score))

    return {
        "quality_score": quality_score,
        "frame_count": frame_count,
        "resolution": (int(avg_width), int(avg_height)),
        "sharpness": float(avg_sharpness),
        "brightness": float(avg_brightness),
        "motion": float(np.mean(motion_scores)) if len(frames) >= 2 else 0.0,
        "issues": issues,
    }


def stabilize_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Стабилизация последовательности кадров (удаление дрожания).

    Args:
        frames: Список кадров

    Returns:
        List[np.ndarray]: Стабилизированные кадры
    """
    if len(frames) < 2:
        return frames

    try:
        # Используем ORB для детекции ключевых точек
        orb = cv2.ORB_create()

        # Первый кадр - reference
        ref_frame = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        ref_kp, ref_desc = orb.detectAndCompute(ref_frame, None)

        stabilized = [frames[0]]

        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)

            # Matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(ref_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)

            # Вычисляем гомографию
            if len(matches) > 10:
                src_pts = np.float32(
                    [ref_kp[m.queryIdx].pt for m in matches[:10]]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches[:10]]).reshape(
                    -1, 1, 2
                )

                M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)

                if M is not None:
                    # Применяем трансформацию
                    h, w = frame.shape[:2]
                    frame_stabilized = cv2.warpAffine(frame, M, (w, h))
                    stabilized.append(frame_stabilized)
                else:
                    stabilized.append(frame)
            else:
                stabilized.append(frame)

        logger.info(f"Stabilized {len(frames)} frames")
        return stabilized

    except Exception as e:
        logger.warning(f"Frame stabilization failed: {e}")
        return frames
