"""
Eye Blink Detection using Eye Aspect Ratio (EAR).

Based on: "Real-Time Eye Blink Detection using Facial Landmarks"
by Soukupová & Čech (2016)

EAR Formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

Where p1-p6 are the 6 eye landmark points.

Blink Detection:
- EAR > threshold: eye open
- EAR < threshold: eye closed
- Sequence: open → closed → open = 1 blink
- Duration: 100-400ms (normal blink)

Accuracy: >95% on standard datasets
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# 68-point landmark indices for eyes (dlib/MediaPipe compatible)
LEFT_EYE_INDICES: Tuple[int, ...] = (36, 37, 38, 39, 40, 41)
RIGHT_EYE_INDICES: Tuple[int, ...] = (42, 43, 44, 45, 46, 47)

# EAR threshold for closed eye detection (from original paper)
DEFAULT_EAR_THRESHOLD: float = 0.21

# Blink duration constraints (milliseconds)
MIN_BLINK_DURATION_MS: float = 100.0
MAX_BLINK_DURATION_MS: float = 400.0

# Minimum EAR drop to register as a blink
MIN_EAR_DROP: float = 0.05

# ============================================================================
# Eye Aspect Ratio Calculation
# ============================================================================


def calculate_eye_aspect_ratio(
    eye_landmarks: np.ndarray,
) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for a single eye.

    Formula (Soukupová & Čech):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

    Args:
        eye_landmarks: Array of shape (6, 2) with eye coordinates.

    Returns:
        EAR value (0.0 = closed, ~0.3 = open).

    Raises:
        ValueError: If not exactly 6 landmarks provided.
    """
    if len(eye_landmarks) != 6:
        raise ValueError(f"Expected 6 eye landmarks, got {len(eye_landmarks)}")

    # Vertical distances
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # p2-p6
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # p3-p5

    # Horizontal distance
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # p1-p4

    # Avoid division by zero
    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


def calculate_average_ear(
    landmarks: np.ndarray,
    left_eye_indices: Tuple[int, ...] = LEFT_EYE_INDICES,
    right_eye_indices: Tuple[int, ...] = RIGHT_EYE_INDICES,
) -> float:
    """
    Calculate average EAR for both eyes.

    Args:
        landmarks: Full 68-point facial landmarks array.
        left_eye_indices: Indices for left eye landmarks.
        right_eye_indices: Indices for right eye landmarks.

    Returns:
        Average EAR (left_ear + right_ear) / 2.
    """
    left_eye = landmarks[left_eye_indices]
    right_eye = landmarks[right_eye_indices]

    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)

    return (left_ear + right_ear) / 2.0


def extract_eye_landmarks(
    landmarks: np.ndarray,
    eye: str = "both",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract eye landmarks from 68-point facial landmarks.

    Args:
        landmarks: Array of shape (68, 2) with facial landmarks.
        eye: "left", "right", or "both".

    Returns:
        Tuple of (left_eye, right_eye), each of shape (6, 2).

    Raises:
        ValueError: If invalid eye parameter or landmark count mismatch.
    """
    if landmarks.shape[0] != 68:
        raise ValueError(f"Expected 68 landmarks, got {landmarks.shape[0]}")

    if eye in ("left", "both"):
        left_eye = landmarks[LEFT_EYE_INDICES]
    else:
        left_eye = None

    if eye in ("right", "both"):
        right_eye = landmarks[RIGHT_EYE_INDICES]
    else:
        right_eye = None

    return left_eye, right_eye


# ============================================================================
# Blink Detection in Sequence
# ============================================================================


def detect_blinks_in_sequence(
    landmarks_sequence: List[np.ndarray],
    fps: float = 30.0,
    min_blinks: int = 1,
    ear_threshold: float = DEFAULT_EAR_THRESHOLD,
    min_ear_drop: float = MIN_EAR_DROP,
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Detect blinks in a sequence of facial landmark frames.

    Algorithm:
        1. Calculate EAR for each frame
        2. Detect sequence: open → closed → open
        3. Validate blink duration (100-400ms)
        4. Count valid blinks

    Args:
        landmarks_sequence: List of 68-point landmark arrays.
        fps: Video frame rate (for duration calculation).
        min_blinks: Minimum required number of blinks.
        ear_threshold: EAR threshold for closed eye detection.
        min_ear_drop: Minimum EAR drop to register as valid blink.

    Returns:
        Tuple of:
            - success: Whether min_blinks requirement is met.
            - blink_count: Number of detected blinks.
            - stats: Dictionary with detailed statistics.
    """
    if len(landmarks_sequence) < 3:
        return (
            False,
            0,
            {
                "error": "Too few frames (minimum 3 required)",
                "blink_frames": [],
                "avg_ear": 0.0,
                "total_frames": 0,
            },
        )

    # Calculate EAR for each frame
    ear_values: List[float] = []
    for landmarks in landmarks_sequence:
        try:
            ear = calculate_average_ear(landmarks)
            ear_values.append(ear)
        except Exception as e:
            logger.warning(f"Failed to calculate EAR: {e}")
            ear_values.append(0.3)  # Assume open eyes on error

    if not ear_values:
        return (
            False,
            0,
            {
                "error": "Failed to calculate EAR values",
                "blink_frames": [],
                "avg_ear": 0.0,
                "total_frames": 0,
            },
        )

    # Detect blinks using state machine
    blink_count = 0
    blink_frames: List[Dict[str, Any]] = []
    quality_issues: List[str] = []

    state: str = "open"  # States: "open" | "closed"
    blink_start_frame: int = 0
    peak_ear_before_blink: float = 0.0
    min_ear_during_blink: float = 1.0

    for i, ear in enumerate(ear_values):
        if state == "open":
            if ear < ear_threshold:
                # Transition to "closed"
                state = "closed"
                blink_start_frame = i
                peak_ear_before_blink = ear_values[i - 1] if i > 0 else ear
                min_ear_during_blink = ear

        elif state == "closed":
            min_ear_during_blink = min(min_ear_during_blink, ear)

            if ear >= ear_threshold:
                # Transition back to "open" - blink completed
                state = "open"
                blink_end_frame = i

                # Calculate duration
                duration_frames = blink_end_frame - blink_start_frame
                duration_ms = (duration_frames / fps) * 1000

                # Calculate EAR drop
                ear_drop = peak_ear_before_blink - min_ear_during_blink

                # Validate blink
                is_valid = True
                if duration_ms < MIN_BLINK_DURATION_MS:
                    quality_issues.append(
                        f"Blink #{blink_count + 1} too short ({duration_ms:.0f}ms)"
                    )
                    is_valid = False
                if duration_ms > MAX_BLINK_DURATION_MS:
                    quality_issues.append(
                        f"Blink #{blink_count + 1} too long ({duration_ms:.0f}ms)"
                    )
                    is_valid = False
                if ear_drop < min_ear_drop:
                    quality_issues.append(
                        f"Blink #{blink_count + 1} insufficient EAR drop ({ear_drop:.3f})"
                    )
                    is_valid = False

                if is_valid:
                    blink_count += 1
                    blink_frames.append(
                        {
                            "start_frame": blink_start_frame,
                            "end_frame": blink_end_frame,
                            "duration_ms": round(duration_ms, 1),
                            "ear_drop": round(ear_drop, 3),
                            "min_ear": round(min_ear_during_blink, 3),
                        }
                    )
                    logger.debug(
                        f"Blink detected: frames {blink_start_frame}-{blink_end_frame}, "
                        f"duration={duration_ms:.1f}ms, drop={ear_drop:.3f}"
                    )

    # Calculate statistics
    avg_ear = float(np.mean(ear_values))
    min_ear = float(np.min(ear_values))
    max_ear = float(np.max(ear_values))
    std_ear = float(np.std(ear_values))

    # Check quality
    if avg_ear < 0.15:
        quality_issues.append("Average EAR too low (eyes may be consistently closed)")
    if avg_ear > 0.35:
        quality_issues.append("Average EAR too high (unusual eye openness)")
    if std_ear < 0.02:
        quality_issues.append("Very low EAR variance (no eye movement detected)")

    # Determine success
    success = blink_count >= min_blinks

    stats = {
        "blink_frames": blink_frames,
        "avg_ear": round(avg_ear, 3),
        "min_ear": round(min_ear, 3),
        "max_ear": round(max_ear, 3),
        "std_ear": round(std_ear, 3),
        "ear_threshold_used": ear_threshold,
        "total_frames": len(ear_values),
        "quality_issues": quality_issues,
    }

    logger.info(
        f"Blink detection: {blink_count}/{min_blinks} blinks detected, "
        f"success={success}, avg_ear={avg_ear:.3f}"
    )

    return success, blink_count, stats


# ============================================================================
# Real-time Blink Detector (Stateful)
# ============================================================================


class BlinkDetector:
    """
    Stateful blink detector for real-time processing.

    Uses a sliding window for EAR analysis.
    """

    def __init__(
        self,
        fps: float = 30.0,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        window_size: int = 10,
    ):
        """
        Args:
            fps: Frame rate for duration calculations.
            ear_threshold: EAR threshold for closed eye detection.
            window_size: Size of EAR history window.
        """
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.window_size = window_size

        self.ear_history: List[float] = []
        self.state = "open"
        self.blink_count = 0
        self.blink_start_frame = 0
        self.current_frame = 0

    def reset(self) -> None:
        """Reset detector state."""
        self.ear_history.clear()
        self.state = "open"
        self.blink_count = 0
        self.blink_start_frame = 0
        self.current_frame = 0

    def process_frame(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame.

        Args:
            landmarks: Full 68-point facial landmarks.

        Returns:
            Dictionary with current state information.
        """
        avg_ear = calculate_average_ear(landmarks)
        self.ear_history.append(avg_ear)

        # Limit history size
        if len(self.ear_history) > self.window_size:
            self.ear_history.pop(0)

        blink_detected = False

        # State machine
        if self.state == "open" and avg_ear < self.ear_threshold:
            self.state = "closed"
            self.blink_start_frame = self.current_frame

        elif self.state == "closed" and avg_ear >= self.ear_threshold:
            self.state = "open"

            # Validate blink duration
            duration_frames = self.current_frame - self.blink_start_frame
            duration_ms = (duration_frames / self.fps) * 1000

            if MIN_BLINK_DURATION_MS <= duration_ms <= MAX_BLINK_DURATION_MS:
                self.blink_count += 1
                blink_detected = True

        self.current_frame += 1

        return {
            "blink_detected": blink_detected,
            "blink_count": self.blink_count,
            "current_ear": round(avg_ear, 3),
            "state": self.state,
            "avg_ear_window": (
                round(np.mean(self.ear_history), 3) if self.ear_history else 0.0
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_blinks": self.blink_count,
            "current_ear": self.ear_history[-1] if self.ear_history else 0.0,
            "avg_ear": np.mean(self.ear_history) if self.ear_history else 0.0,
            "is_currently_blinking": self.state == "closed",
            "frames_processed": len(self.ear_history),
        }


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# For backward compatibility with old import names
extract_eye_landmarks_from_68 = extract_eye_landmarks
calculate_ear = calculate_eye_aspect_ratio
