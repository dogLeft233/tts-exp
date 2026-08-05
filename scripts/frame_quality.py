"""Frame quality selection for TFG reference-image preprocessing.

Pure functions, detector-agnostic: callers inject ``extract_metrics`` so this
module stays testable without a face-detection dependency. The CLI
``scripts/38_prepare_frames.py`` wires the real (Haar/insightface) detector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FaceMetrics:
    yaw_deg: float
    face_area_px: int
    blur_score: float
    eye_open: float | None = None  # eye aspect ratio; None = unknown
    mouth_open_ratio: float | None = None


def compute_blur_score(gray: np.ndarray) -> float:
    """Laplacian variance: higher = sharper."""
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_score(m: FaceMetrics, max_yaw_deg: float = 30.0) -> float:
    """Higher = better reference frame.

    Monotone in sharpness and face size, decreasing in |yaw|. Uses log
    transforms so very large faces/sharps don't dominate the ranking.
    """
    yaw_penalty = abs(m.yaw_deg) / max_yaw_deg
    return math.log1p(m.blur_score) + math.log1p(m.face_area_px) - yaw_penalty


def is_acceptable(
    m: FaceMetrics,
    max_yaw_deg: float = 30.0,
    min_face_area_px: float = 20000.0,
) -> bool:
    """Quality gate: reject profile views and faces too small to drive TFG."""
    return abs(m.yaw_deg) <= max_yaw_deg and m.face_area_px >= min_face_area_px


def select_best_frame(
    frames: list,
    extract_metrics,
    max_yaw_deg: float = 30.0,
    min_face_area_px: float = 20000.0,
) -> tuple[int, FaceMetrics] | None:
    """Pick the best acceptable frame; None when no frame passes the gate.

    ``extract_metrics(frame) -> FaceMetrics | None`` (None = no face / undetected).
    """
    best_idx: int | None = None
    best_score: float = -1.0
    best_metrics: FaceMetrics | None = None
    for idx, frame in enumerate(frames):
        m = extract_metrics(frame)
        if m is None or not is_acceptable(m, max_yaw_deg, min_face_area_px):
            continue
        score = quality_score(m, max_yaw_deg)
        if best_idx is None or score > best_score:
            best_idx, best_score, best_metrics = idx, score, m
    if best_idx is None:
        return None
    assert best_metrics is not None
    return best_idx, best_metrics
