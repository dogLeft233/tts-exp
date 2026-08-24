#!/usr/bin/env python3
"""Persistent visual mouth features for the LRS3 TTS visual audit."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

MOUTH_INDICES = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270,
    409, 287,
)
LEFT_EYE_INDEX = 33
RIGHT_EYE_INDEX = 263
NOSE_INDEX = 1
SCHEMA_VERSION = 1


def _validate_metadata_schema(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("visual sequence metadata must be a JSON object")
    version = metadata.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"unsupported visual sequence schema: {version!r}")
    return metadata


@dataclass(frozen=True)
class VisualSequence:
    """Per-frame canonical mouth geometry and extraction provenance."""

    landmarks: np.ndarray
    canonical_mouth: np.ndarray
    valid: np.ndarray
    mouth_open: np.ndarray
    mouth_width: np.ndarray
    mouth_height: np.ndarray
    mouth_area: np.ndarray
    timestamps_s: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        n = int(self.landmarks.shape[0])
        if self.landmarks.shape != (n, 478, 2):
            raise ValueError("landmarks must have shape (frames, 478, 2)")
        if self.canonical_mouth.shape != (n, len(MOUTH_INDICES), 2):
            raise ValueError("canonical_mouth shape is incompatible with MOUTH_INDICES")
        for name in ("valid", "mouth_open", "mouth_width", "mouth_height", "mouth_area", "timestamps_s"):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have one value per frame")
        if self.valid.dtype != np.bool_:
            raise ValueError("valid must be boolean")
        arrays = (self.landmarks, self.canonical_mouth, self.mouth_open, self.mouth_width,
                  self.mouth_height, self.mouth_area, self.timestamps_s)
        if not all(np.isfinite(value).all() for value in arrays if value.size):
            raise ValueError("visual features must be finite; invalid frames use valid=false and zeros")
        invalid = ~self.valid
        derived_arrays = (self.canonical_mouth, self.mouth_open, self.mouth_width,
                          self.mouth_height, self.mouth_area)
        if any(np.any(value[invalid] != 0.0) for value in derived_arrays if value.size and invalid.any()):
            raise ValueError("invalid frames must have zero-valued derived features")
        if n > 1 and np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("timestamps must be strictly increasing")


def _finite_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (478, 2):
        raise ValueError("landmarks must have shape (frames, 478, 2)")
    return np.isfinite(points).all(axis=2)


def canonicalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize each landmark frame by the inter-eye similarity transform."""
    points = np.asarray(landmarks, dtype=np.float64)
    valid_points = _finite_landmarks(points)
    output = np.zeros_like(points, dtype=np.float64)
    valid = np.zeros(points.shape[0], dtype=bool)
    for index, frame in enumerate(points):
        if not valid_points[index, [LEFT_EYE_INDEX, RIGHT_EYE_INDEX]].all():
            continue
        left = frame[LEFT_EYE_INDEX]
        right = frame[RIGHT_EYE_INDEX]
        center = 0.5 * (left + right)
        axis = right - left
        scale = float(np.linalg.norm(axis))
        if not math.isfinite(scale) or scale <= 1e-8:
            continue
        x_axis = axis / scale
        y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float64)
        finite = valid_points[index]
        centered = frame - center
        output[index, finite, 0] = centered[finite] @ x_axis / scale
        output[index, finite, 1] = centered[finite] @ y_axis / scale
        valid[index] = True
    return output, valid


def mouth_features(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Return canonical mouth geometry and scalar trajectories."""
    points = np.asarray(landmarks, dtype=np.float64)
    canonical, transform_valid = canonicalize_landmarks(points)
    finite = _finite_landmarks(points)
    required = np.asarray((LEFT_EYE_INDEX, RIGHT_EYE_INDEX, *MOUTH_INDICES), dtype=np.int64)
    valid = transform_valid & finite[:, required].all(axis=1)
    mouth = canonical[:, MOUTH_INDICES, :]
    open_value = np.zeros(points.shape[0], dtype=np.float64)
    width = np.zeros(points.shape[0], dtype=np.float64)
    height = np.zeros(points.shape[0], dtype=np.float64)
    area = np.zeros(points.shape[0], dtype=np.float64)
    if mouth.shape[1] > 0:
        x = mouth[:, :, 0]
        y = mouth[:, :, 1]
        width = x.max(axis=1) - x.min(axis=1)
        height = y.max(axis=1) - y.min(axis=1)
        area = width * height
        upper = canonical[:, 13, :]
        lower = canonical[:, 14, :]
        open_value = np.linalg.norm(lower - upper, axis=1)
    for array in (mouth, open_value, width, height, area):
        array[~valid] = 0.0
    return {
        "canonical_mouth": mouth,
        "valid": valid,
        "mouth_open": open_value,
        "mouth_width": width,
        "mouth_height": height,
        "mouth_area": area,
    }


def _fourcc_string(value: float) -> str | None:
    integer = int(value)
    if integer <= 0:
        return None
    chars = "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))
    return chars if all(32 <= ord(char) < 127 for char in chars) else None


def _landmarker(model_path: Path):
    import mediapipe as mp  # type: ignore[import-not-found]
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def extract_video_features(video_path: str | Path, landmarker_asset: str | Path) -> VisualSequence:
    """Sequentially decode one video and extract deterministic landmark features."""
    import cv2

    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = _fourcc_string(capture.get(cv2.CAP_PROP_FOURCC))
    if declared_frames <= 0:
        capture.release()
        raise ValueError(f"video has no reliable declared frame count: {path}")
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"invalid video metadata: {path}")

    mp, landmarker = _landmarker(Path(landmarker_asset).resolve())
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    decode_errors = 0
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                if declared_frames > 0 and frame_index != declared_frames:
                    raise ValueError(
                        f"decoder stopped at frame {frame_index}, expected {declared_frames}: {path}"
                    )
                break
            malformed = (
                frame is None
                or frame.ndim != 3
                or frame.shape[0] != height
                or frame.shape[1] != width
                or frame.shape[2] != 3
                or frame.dtype != np.uint8
            )
            if malformed:
                decode_errors += 1
                raise ValueError(f"malformed decoded frame {frame_index}: {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if result.face_landmarks:
                points = np.asarray([(point.x, point.y) for point in result.face_landmarks[0]], dtype=np.float64)
                if points.shape != (478, 2):
                    points = np.full((478, 2), np.nan, dtype=np.float64)
            else:
                points = np.full((478, 2), np.nan, dtype=np.float64)
            frames.append(points)
            timestamps.append(frame_index / fps)
            frame_index += 1
    finally:
        capture.release()
        landmarker.close()

    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    landmarks = np.stack(frames, axis=0)
    features = mouth_features(landmarks)
    landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "video_path": str(path),
        "declared_frame_count": declared_frames,
        "decoded_frame_count": len(frames),
        "read_frame_count": frame_index,
        "decode_errors": decode_errors,
        "fps": fps,
        "width": width,
        "height": height,
        "fourcc": fourcc,
        "landmarker_asset": str(Path(landmarker_asset).resolve()),
        "coordinate_system": "mediapipe_normalized_xy_then_eye_similarity_canonical",
        "landmark_confidence": "unavailable_from_face_landmarker_tasks_api",
        "mouth_indices": list(MOUTH_INDICES),
        "valid_fraction": float(features["valid"].mean()),
        "timestamps_monotonic": bool(np.all(np.diff(np.asarray(timestamps)) > 0)),
    }
    return VisualSequence(
        landmarks=landmarks,
        canonical_mouth=features["canonical_mouth"],
        valid=features["valid"],
        mouth_open=features["mouth_open"],
        mouth_width=features["mouth_width"],
        mouth_height=features["mouth_height"],
        mouth_area=features["mouth_area"],
        timestamps_s=np.asarray(timestamps, dtype=np.float64),
        metadata=metadata,
    )


def save_visual_sequence(path: str | Path, sequence: VisualSequence) -> None:
    _validate_metadata_schema(sequence.metadata)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        landmarks=sequence.landmarks,
        canonical_mouth=sequence.canonical_mouth,
        valid=sequence.valid,
        mouth_open=sequence.mouth_open,
        mouth_width=sequence.mouth_width,
        mouth_height=sequence.mouth_height,
        mouth_area=sequence.mouth_area,
        timestamps_s=sequence.timestamps_s,
        metadata_json=np.asarray(json.dumps(sequence.metadata, sort_keys=True), dtype=np.str_),
    )


def load_visual_sequence(path: str | Path) -> VisualSequence:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = _validate_metadata_schema(json.loads(str(data["metadata_json"].item())))
        raw_valid = np.asarray(data["valid"])
        if raw_valid.dtype != np.bool_:
            raise ValueError(f"valid array must be boolean, got {raw_valid.dtype}")
        return VisualSequence(
            landmarks=np.asarray(data["landmarks"], dtype=np.float64),
            canonical_mouth=np.asarray(data["canonical_mouth"], dtype=np.float64),
            valid=raw_valid,
            mouth_open=np.asarray(data["mouth_open"], dtype=np.float64),
            mouth_width=np.asarray(data["mouth_width"], dtype=np.float64),
            mouth_height=np.asarray(data["mouth_height"], dtype=np.float64),
            mouth_area=np.asarray(data["mouth_area"], dtype=np.float64),
            timestamps_s=np.asarray(data["timestamps_s"], dtype=np.float64),
            metadata=metadata,
        )
