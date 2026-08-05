"""Face detectors for reference-frame quality selection.

Each detector maps a BGR frame to ``FaceMetrics | None``. Detectors are
pluggable so the quality-selection logic in ``frame_quality.py`` stays
dependency-free; these wrappers live behind the injection point
``extract_metrics_for_frame`` in ``scripts/38_prepare_frames.py``.
"""

from __future__ import annotations

import math

import numpy as np

from frame_quality import FaceMetrics, compute_blur_score


def _landmark2d(landmarks, idx, w, h):
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def _metrics_from_landmarks(pts: np.ndarray, w: int, h: int) -> FaceMetrics | None:
    """Pure geometry: 478x2 landmark array -> FaceMetrics.

    ``pts`` uses normalized x/y coordinates; indices follow the MediaPipe
    478-point face mesh convention.
    """
    lx, ly = pts[33] * (w, h)
    rx, ry = pts[263] * (w, h)
    nx, ny = pts[1] * (w, h)

    eye_dist = math.hypot(rx - lx, ry - ly)
    if eye_dist < 1e-6:
        return None

    nose_off_x = abs(nx - (lx + rx) / 2) / eye_dist
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * nose_off_x))))
    face_area_px = (eye_dist * 2.3) ** 2

    ulx, uly = pts[13] * (w, h)
    llx, lly = pts[14] * (w, h)
    mouth_open_ratio = math.hypot(llx - ulx, lly - uly) / eye_dist

    try:
        l_top = pts[159] * (w, h)
        l_bot = pts[145] * (w, h)
        r_top = pts[386] * (w, h)
        r_bot = pts[374] * (w, h)
        ear_left = math.hypot(l_bot[0] - l_top[0], l_bot[1] - l_top[1])
        ear_right = math.hypot(r_bot[0] - r_top[0], r_bot[1] - r_top[1])
        eye_open = 0.5 * (ear_left + ear_right) / eye_dist
    except IndexError:
        eye_open = None

    return FaceMetrics(
        yaw_deg=round(yaw, 2),
        face_area_px=int(round(face_area_px)),
        blur_score=0.0,
        eye_open=eye_open,
        mouth_open_ratio=round(mouth_open_ratio, 4),
    )


def mediapipe_extract_metrics(frame) -> FaceMetrics | None:
    """MediaPipe FaceLandmarker (Tasks API) detector.

    yaw is a 2D geometric approximation (nose offset vs inter-eye baseline):
    frontal ~0 deg, profile ~+-45-60 deg. face_area is estimated from the
    inter-eye distance scaled to a full-face bounding box.
    """
    import cv2
    import mediapipe as mp  # type: ignore[import-not-found]
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]
    landmarker = _get_landmarker(mp_python, vision)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    pts = np.array([(lm.x, lm.y) for lm in result.face_landmarks[0]], dtype=np.float64)
    metrics = _metrics_from_landmarks(pts, w, h)
    if metrics is None:
        return None
    return FaceMetrics(
        yaw_deg=metrics.yaw_deg,
        face_area_px=metrics.face_area_px,
        blur_score=compute_blur_score(gray),
        eye_open=metrics.eye_open,
        mouth_open_ratio=metrics.mouth_open_ratio,
    )


_LANDMARKER = None


def _get_landmarker(mp_python, vision, model_path: str = ""):
    """Lazily build the FaceLandmarker (module-level singleton)."""
    global _LANDMARKER
    if _LANDMARKER is None:
        import os
        from pathlib import Path

        if not model_path:
            candidates = [
                Path(os.environ.get("DITTO_AUX", "")) / "face_landmarker.task",
                Path("/root/autodl-tmp/checkpoints/ditto-talkinghead/ditto_pytorch/aux_models/face_landmarker.task"),
                Path("checkpoints/ditto_pytorch/aux_models/face_landmarker.task"),
            ]
            model_path = str(next((p for p in candidates if p.is_file()), ""))
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER
