"""Shared, model-free utilities for the fresh-source visual branch.

The module deliberately contains no model invocation and no SyncNet import.  It
is used by the runner for deterministic materialisation and by the independent
validator for small, separately implemented checks.  Keeping the feature file
contract here makes it possible to run the analysis on synthetic arrays in CI
without MediaPipe or a GPU.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.fresh_source_inputs.protocol import (
    ProtocolError,
    file_sha256,
    load_self_hashed,
    read_json,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FRAME_COUNT = 140
FPS = 25.0
FRAME_PERIOD_S = 1.0 / FPS
FRAME_TOLERANCE_S = 0.5 / FPS
J0 = np.arange(25, 115, dtype=np.int64)
SEEDS = (42, 43)
MODELS = ("wav2lip", "ditto")
MOUTH_SHAPE = (31, 2)
BOOTSTRAP_COUNT = 20_000
BOOTSTRAP_SEED = 20260910
PRACTICAL_THRESHOLD = 0.02
CALIBRATION_ERROR_THRESHOLD = 1e-6
MIN_CALIBRATION_SUPPORT = 0.90
MIN_OBSERVED_FRAMES = 81
BLIND_SEED = 20260911

DEFAULT_LANDMARKER_ASSET = REPO_ROOT / "checkpoints/mediapipe/face_landmarker.task"
LANDMARKER_OPTIONS = {
    "running_mode": "IMAGE",
    "num_faces": 1,
    "output_face_blendshapes": False,
    "output_facial_transformation_matrixes": False,
}


class VisualProtocolError(ProtocolError):
    """A frozen visual-branch contract violation."""


@dataclass(frozen=True)
class FeatureArray:
    """The minimal raw array contract consumed by the fixed-clock analysis."""

    mouth: np.ndarray
    valid: np.ndarray
    timestamps: np.ndarray
    metadata: dict[str, Any]
    landmarks: np.ndarray | None = None

    @property
    def frame_count(self) -> int:
        return int(self.mouth.shape[0])


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        raise VisualProtocolError(f"{name} contains non-finite values")
    return arr


def _metadata_from_npz(data: Any) -> dict[str, Any]:
    if "metadata_json" not in data.files:
        return {}
    try:
        metadata = json.loads(str(data["metadata_json"].item()))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VisualProtocolError("feature metadata_json is malformed") from exc
    if not isinstance(metadata, dict):
        raise VisualProtocolError("feature metadata must be a JSON object")
    return metadata


def validate_feature_arrays(
    mouth: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
    *,
    landmarks: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    path: Path | None = None,
) -> FeatureArray:
    """Validate a feature array without silently repairing missing rows."""

    prefix = f"{path}: " if path is not None else ""
    mouth = np.asarray(mouth, dtype=np.float64)
    valid_raw = np.asarray(valid)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if mouth.ndim != 3 or tuple(mouth.shape[1:]) != MOUTH_SHAPE:
        raise VisualProtocolError(f"{prefix}canonical_mouth must have shape (T,31,2)")
    if valid_raw.dtype != np.bool_ or valid_raw.shape != (mouth.shape[0],):
        raise VisualProtocolError(f"{prefix}valid must be boolean with shape (T,)")
    if timestamps.shape != valid_raw.shape:
        raise VisualProtocolError(f"{prefix}timestamps_s must have shape (T,)")
    _finite(mouth, "canonical_mouth")
    _finite(timestamps, "timestamps_s")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise VisualProtocolError(f"{prefix}timestamps_s must be strictly increasing")
    invalid = ~valid_raw
    if invalid.any() and np.any(mouth[invalid] != 0.0):
        raise VisualProtocolError(f"{prefix}invalid frames must contain zero mouth features")
    if landmarks is not None:
        landmarks = np.asarray(landmarks, dtype=np.float64)
        if landmarks.ndim != 3 or tuple(landmarks.shape[1:]) != (478, 2):
            raise VisualProtocolError(f"{prefix}landmarks must have shape (T,478,2)")
        if landmarks.shape[0] != mouth.shape[0]:
            raise VisualProtocolError(f"{prefix}landmarks frame count differs")
        _finite(landmarks, "landmarks")
    return FeatureArray(
        mouth=mouth,
        valid=np.asarray(valid_raw, dtype=bool),
        timestamps=timestamps,
        metadata=dict(metadata or {}),
        landmarks=landmarks,
    )


def load_feature(path: str | Path) -> FeatureArray:
    """Load only the raw, non-derived arrays used by B."""

    target = Path(path)
    try:
        with np.load(target, allow_pickle=False) as data:
            required = {"canonical_mouth", "valid", "timestamps_s"}
            if not required.issubset(data.files):
                raise VisualProtocolError(
                    f"feature is missing {sorted(required.difference(data.files))}: {target}"
                )
            landmarks = np.asarray(data["landmarks"], dtype=np.float64) if "landmarks" in data.files else None
            return validate_feature_arrays(
                data["canonical_mouth"],
                data["valid"],
                data["timestamps_s"],
                landmarks=landmarks,
                metadata=_metadata_from_npz(data),
                path=target,
            )
    except VisualProtocolError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise VisualProtocolError(f"cannot read feature: {target}") from exc


def save_feature(path: str | Path, feature: FeatureArray) -> str:
    """Write a deterministic feature NPZ and return its byte SHA-256."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    feature = validate_feature_arrays(
        feature.mouth,
        feature.valid,
        feature.timestamps,
        landmarks=feature.landmarks,
        metadata=feature.metadata,
        path=target,
    )
    payload: dict[str, Any] = {
        "canonical_mouth": feature.mouth,
        "valid": feature.valid,
        "timestamps_s": feature.timestamps,
        "metadata_json": np.asarray(
            json.dumps(feature.metadata, ensure_ascii=False, sort_keys=True), dtype=np.str_
        ),
    }
    if feature.landmarks is not None:
        payload["landmarks"] = feature.landmarks
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, target)
    return file_sha256(target)


def crop_feature(feature: Any, *, frame_count: int = FRAME_COUNT) -> FeatureArray:
    """Crop an existing VisualSequence/FeatureArray to the fixed 140-frame clock."""

    if isinstance(feature, FeatureArray):
        mouth = feature.mouth[:frame_count]
        valid = feature.valid[:frame_count]
        timestamps = feature.timestamps[:frame_count]
        landmarks = None if feature.landmarks is None else feature.landmarks[:frame_count]
        metadata = dict(feature.metadata)
    else:
        # This branch intentionally uses attribute access rather than importing
        # VisualSequence, so tests can pass a tiny fake sequence.
        mouth = np.asarray(feature.canonical_mouth)[:frame_count]
        valid = np.asarray(feature.valid)[:frame_count]
        timestamps = np.asarray(feature.timestamps_s)[:frame_count]
        landmarks = getattr(feature, "landmarks", None)
        if landmarks is not None:
            landmarks = np.asarray(landmarks)[:frame_count]
        metadata = dict(getattr(feature, "metadata", {}) or {})
    if len(mouth) < frame_count:
        raise VisualProtocolError(f"feature has only {len(mouth)} frames; need {frame_count}")
    metadata.setdefault("fixed_clock_frames", frame_count)
    metadata.setdefault("fps", FPS)
    return validate_feature_arrays(
        mouth, valid, timestamps, landmarks=landmarks, metadata=metadata
    )


def media_metadata(asset_path: str | Path) -> dict[str, Any]:
    """Return the pinned extractor provenance used in every feature row."""

    asset = Path(asset_path).resolve()
    try:
        version: str | None = importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "extractor": "scripts.experiments.lrs3_tts_visual_advantage.video_features",
        "extractor_api": ["create_landmarker", "extract_video_features"],
        "mediapipe_version": version,
        "landmarker_asset": str(asset),
        "landmarker_asset_sha256": file_sha256(asset) if asset.is_file() else None,
        "landmarker_options": dict(LANDMARKER_OPTIONS),
        "fixed_clock_frames": FRAME_COUNT,
        "fps": FPS,
        "canonical_feature": "eye_similarity_transform_then_31_point_mouth",
        "mouth_shape": list(MOUTH_SHAPE),
    }


def import_video_features():
    """Lazy import of the existing MediaPipe implementation.

    This function is kept separate so importing/validating the B package never
    imports cv2 or MediaPipe and therefore remains cheap and CPU-testable.
    """

    from scripts.experiments.lrs3_tts_visual_advantage.video_features import (
        create_landmarker,
        extract_video_features,
    )

    return create_landmarker, extract_video_features


def _centered_mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.shape[0] == 0:
        raise VisualProtocolError("cannot compare empty or differently shaped trajectories")
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - np.mean(left, axis=0, dtype=np.float64)
    right = right - np.mean(right, axis=0, dtype=np.float64)
    return float(np.mean((left - right) ** 2, dtype=np.float64))


def calibration_pair(
    feature: FeatureArray,
    indices: np.ndarray | Sequence[int] = J0,
    *,
    transform: str = "identity",
) -> dict[str, Any]:
    """Compute one real-only fixed-clock sensitivity condition."""

    feature = validate_feature_arrays(feature.mouth, feature.valid, feature.timestamps, metadata=feature.metadata)
    base = np.asarray(indices, dtype=np.int64)
    if np.any(base < 0) or np.any(base >= FRAME_COUNT):
        raise VisualProtocolError("calibration indices fall outside the fixed clock")
    if transform == "identity":
        other = base.copy()
    elif transform == "shift_plus5":
        other = base + 5
    elif transform == "shift_minus5":
        other = base - 5
    elif transform == "reverse":
        other = (FRAME_COUNT - 1) - base
    else:
        raise ValueError(f"unknown calibration transform: {transform}")
    in_range = (other >= 0) & (other < FRAME_COUNT)
    base = base[in_range]
    other = other[in_range]
    if base.size:
        # These are deliberately synthetic index transforms on one frozen
        # trajectory.  A +5/-5 or reversal pair is expected to have different
        # timestamps; PTS matching belongs to generated-vs-real support, not
        # to the real-only sensitivity calibration.
        support_mask = feature.valid[base] & feature.valid[other]
        base = base[support_mask]
        other = other[support_mask]
    count = int(base.size)
    coverage = float(count / len(indices)) if len(indices) else 0.0
    error = None
    if count:
        error = _centered_mse(feature.mouth[base], feature.mouth[other])
    return {
        "transform": transform,
        "valid_count": count,
        "support_total": len(indices),
        "coverage": coverage,
        "indices": base.tolist(),
        "other_indices": other.tolist(),
        "mse": error,
        "observed": bool(coverage >= MIN_CALIBRATION_SUPPORT),
    }


def calibrate_real_feature(feature: FeatureArray) -> dict[str, Any]:
    """Run the predeclared identity/shift/reversal gate on one real track."""

    conditions = {
        name: calibration_pair(feature, transform=name)
        for name in ("identity", "shift_plus5", "shift_minus5", "reverse")
    }
    identity = conditions["identity"]
    return {
        "identity_mse": identity["mse"],
        "identity_pass": bool(
            identity["observed"]
            and identity["mse"] is not None
            and identity["mse"] <= 1e-12
        ),
        "conditions": conditions,
    }


def run_calibration(
    group_features: Mapping[str, FeatureArray],
    groups: Sequence[str],
) -> dict[str, Any]:
    """Independent real-only calibration summary, retaining every group."""

    rows: list[dict[str, Any]] = []
    for group in groups:
        feature = group_features.get(str(group))
        if feature is None:
            rows.append(
                {
                    "source_group": str(group),
                    "observed": False,
                    "missing_reason": "real_feature_missing",
                    "identity_mse": None,
                    "conditions": {},
                }
            )
            continue
        values = calibrate_real_feature(feature)
        row = {"source_group": str(group), "observed": True, **values}
        rows.append(row)
    distortion_counts: dict[str, int] = {}
    distortion_observed: dict[str, int] = {}
    for transform in ("shift_plus5", "shift_minus5", "reverse"):
        observed_rows = [r for r in rows if r.get("conditions", {}).get(transform, {}).get("observed")]
        distortion_observed[transform] = len(observed_rows)
        distortion_counts[transform] = sum(
            float(r["conditions"][transform].get("mse") or 0.0) > CALIBRATION_ERROR_THRESHOLD
            for r in observed_rows
        )
    identity_values = [r.get("identity_mse") for r in rows if r.get("identity_mse") is not None]
    identity_pass = bool(rows and len(identity_values) == len(rows) and all(float(x) <= 1e-12 for x in identity_values))
    distortion_pass = all(distortion_counts.get(t, 0) >= 10 for t in ("shift_plus5", "shift_minus5", "reverse"))
    support_pass = all(
        distortion_observed.get(t, 0) == len(rows)
        for t in ("shift_plus5", "shift_minus5", "reverse")
    )
    calibrated = bool(len(rows) == 12 and identity_pass and distortion_pass and support_pass)
    return {
        "schema_version": 1,
        "protocol": {
            "fixed_clock_frames": FRAME_COUNT,
            "J0": J0.tolist(),
            "fps": FPS,
            "pair_tolerance_s": FRAME_TOLERANCE_S,
            "min_pair_coverage": MIN_CALIBRATION_SUPPORT,
            "identity_mse_max": 1e-12,
            "distortion_mse_min": CALIBRATION_ERROR_THRESHOLD,
            "required_distorted_groups": 10,
        },
        "groups": [str(g) for g in groups],
        "group_count": len(rows),
        "rows": rows,
        "identity_pass": identity_pass,
        "distortion_counts": distortion_counts,
        "distortion_observed": distortion_observed,
        "distortion_pass": distortion_pass,
        "support_pass": support_pass,
        "status": "METRIC_CALIBRATED" if calibrated else "METRIC_NOT_CALIBRATED",
        "calibrated": calibrated,
    }


def fixed_clock_support(
    features: Mapping[str, FeatureArray],
    names: Sequence[str],
    *,
    indices: np.ndarray | Sequence[int] = J0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact common frame indices; never interpolate or search."""

    if not names or "real" not in features:
        return np.asarray([], dtype=np.int64), {"missing": list(names)}
    base = np.asarray(indices, dtype=np.int64)
    support = np.ones(base.shape, dtype=bool)
    reasons: dict[str, int] = {}
    real = features["real"]
    for name in names:
        item = features.get(name)
        if item is None:
            support[:] = False
            reasons[name] = int(base.size)
            continue
        current = np.zeros(base.shape, dtype=bool)
        in_range = base < item.frame_count
        current[in_range] = item.valid[base[in_range]]
        timestamp_ok = np.zeros(base.shape, dtype=bool)
        timestamp_ok[in_range] = (
            np.abs(item.timestamps[base[in_range]] - real.timestamps[base[in_range]])
            <= FRAME_TOLERANCE_S + 1e-12
        )
        current &= timestamp_ok
        reasons[name] = int(np.sum(~current))
        support &= current
    # Real itself is also part of the support and must be finite/valid.
    in_range = base < real.frame_count
    own = np.zeros(base.shape, dtype=bool)
    own[in_range] = real.valid[base[in_range]]
    support &= own
    reasons["real"] = int(np.sum(~own))
    return base[support], {
        "requested_indices": base.tolist(),
        "indices": base[support].tolist(),
        "count": int(np.sum(support)),
        "requested_count": int(base.size),
        "coverage": float(np.sum(support) / base.size) if base.size else 0.0,
        "missing_by_arm": reasons,
        "timestamp_tolerance_s": FRAME_TOLERANCE_S,
    }


def _mse_components(real: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    real = np.asarray(real, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if real.shape != candidate.shape or real.ndim != 3 or real.shape[0] == 0:
        raise VisualProtocolError("MSE inputs must be non-empty and shape matched")
    mu_real = np.mean(real, axis=0, dtype=np.float64)
    mu_candidate = np.mean(candidate, axis=0, dtype=np.float64)
    centered_real = real - mu_real
    centered_candidate = candidate - mu_candidate
    total = float(np.mean((real - candidate) ** 2, dtype=np.float64))
    static = float(np.mean((mu_real - mu_candidate) ** 2, dtype=np.float64))
    dynamic = float(np.mean((centered_real - centered_candidate) ** 2, dtype=np.float64))
    reverse = float(np.mean((centered_real - centered_candidate[::-1]) ** 2, dtype=np.float64))
    identity_error = total - static - dynamic
    if abs(identity_error) > 1e-10 * max(1.0, total):
        raise VisualProtocolError("static/dynamic MSE decomposition identity failed")
    return {
        "e_total": total,
        "e_static": static,
        "e_dynamic": dynamic,
        "e_reverse": reverse,
        "identity_error": identity_error,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0 and numerator == 0.0:
        return 0.0
    return float(numerator / (denominator + 1e-12))


def group_decomposition(
    features: Mapping[str, FeatureArray],
    support: np.ndarray,
    *,
    natural_names: Sequence[str] = ("N42", "N43"),
    candidate_names: Sequence[str] = ("C42", "C43"),
) -> dict[str, Any]:
    """Decompose one model's two-seed arms on one fixed support."""

    real = features["real"].mouth[support]
    natural: list[dict[str, float]] = []
    candidate: list[dict[str, float]] = []
    for n_name, c_name in zip(natural_names, candidate_names, strict=True):
        natural.append(_mse_components(real, features[n_name].mouth[support]))
        candidate.append(_mse_components(real, features[c_name].mouth[support]))
    result: dict[str, Any] = {
        "support_count": int(support.size),
        "seed_values": {"natural": {}, "candidate": {}},
    }
    for label, values, names in (("natural", natural, natural_names), ("candidate", candidate, candidate_names)):
        for name, value in zip(names, values, strict=True):
            result["seed_values"][label][str(name)] = value
        for key in ("e_total", "e_static", "e_dynamic", "e_reverse", "identity_error"):
            result[f"{key}_{label}"] = float(np.mean([item[key] for item in values], dtype=np.float64))
    result["b_dynamic_seeds"] = [
        safe_ratio(n["e_dynamic"] - c["e_dynamic"], n["e_dynamic"] + c["e_dynamic"])
        for n, c in zip(natural, candidate, strict=True)
    ]
    result["q_natural_seeds"] = [safe_ratio(n["e_reverse"] - n["e_dynamic"], n["e_reverse"] + n["e_dynamic"]) for n in natural]
    result["q_candidate_seeds"] = [safe_ratio(c["e_reverse"] - c["e_dynamic"], c["e_reverse"] + c["e_dynamic"]) for c in candidate]
    result["b_dynamic"] = float(np.mean(result["b_dynamic_seeds"], dtype=np.float64))
    result["q_natural"] = float(np.mean(result["q_natural_seeds"], dtype=np.float64))
    result["q_candidate"] = float(np.mean(result["q_candidate_seeds"], dtype=np.float64))
    return result


def full_cohort_bounds(rows: Sequence[Mapping[str, Any]], groups: Sequence[str], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if row.get("observed") and row.get(metric) is not None]
    missing = len(groups) - len(values)
    total = float(np.sum(np.asarray(values, dtype=np.float64))) if values else 0.0
    denominator = len(groups)
    lower = float((total - missing) / denominator) if denominator else -1.0
    upper = float((total + missing) / denominator) if denominator else 1.0
    return {
        "group_count": denominator,
        "observed_group_count": len(values),
        "missing_group_count": missing,
        "sum_observed": total,
        "lower": lower,
        "upper": upper,
        "interpretation": "finite-cohort bounds; missing group values are bounded in [-1,1], not a CI",
    }


def bootstrap_metrics(rows: Sequence[Mapping[str, Any]], groups: Sequence[str], metrics: Sequence[str]) -> dict[str, Any]:
    """Shared deterministic group bootstrap (same draws for every metric)."""

    by_group: dict[str, list[Mapping[str, Any]]] = {str(g): [] for g in groups}
    for row in rows:
        if bool(row.get("observed")) and str(row.get("source_group")) in by_group:
            by_group[str(row["source_group"])].append(row)
    active = [str(g) for g in groups if by_group[str(g)]]
    values = np.asarray(
        [
            [float(np.mean([row[m] for row in by_group[group]], dtype=np.float64)) for m in metrics]
            for group in active
        ],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    if active:
        indices = rng.integers(0, len(active), size=(BOOTSTRAP_COUNT, len(active)), dtype=np.int64)
        draws = values[indices].mean(axis=1)
    else:
        indices = np.empty((BOOTSTRAP_COUNT, 0), dtype=np.int64)
        draws = np.empty((BOOTSTRAP_COUNT, len(metrics)), dtype=np.float64)
    result: dict[str, Any] = {
        "label": "observed_only",
        "observed_group_count": len(active),
        "observed_groups": active,
        "bootstrap_count": BOOTSTRAP_COUNT,
        "seed": BOOTSTRAP_SEED,
        "quantile_method": "linear",
    }
    for column, metric in enumerate(metrics):
        vals = values[:, column] if active else np.asarray([], dtype=np.float64)
        ci = [
            float(np.quantile(draws[:, column], 0.005, method="linear")),
            float(np.quantile(draws[:, column], 0.995, method="linear")),
        ] if active else [None, None]
        result[metric] = {
            "mean": float(np.mean(vals, dtype=np.float64)) if vals.size else None,
            "positive_groups": int(np.sum(vals > 0.0)) if vals.size else 0,
            "observed_group_count": len(active),
            "ci99": ci,
            "group_values": vals.tolist(),
        }
    return result


def make_contact_sheet(
    video_path: str | Path,
    feature: FeatureArray,
    output: str | Path,
    *,
    frames: Sequence[int] = (25, 70, 114),
) -> str | None:
    """Save a deterministic landmark overlay contact sheet when cv2 is available."""

    if feature.landmarks is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    images: list[np.ndarray] = []
    try:
        for frame_index in frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None or frame_index >= feature.frame_count:
                continue
            h, w = frame.shape[:2]
            points = feature.landmarks[int(frame_index)]
            for x, y in points:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (round(float(x) * w), round(float(y) * h)), 1, (0, 255, 0), -1)
            cv2.putText(frame, f"frame={frame_index}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            images.append(frame)
    finally:
        capture.release()
    if not images:
        return None
    height = max(int(image.shape[0]) for image in images)
    normalized = [cv2.resize(image, (height, height), interpolation=cv2.INTER_AREA) for image in images]
    sheet = np.concatenate(normalized, axis=1)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), sheet):
        return None
    return file_sha256(target)


def _read_video_frames(video_path: str | Path, start: int = 25, stop: int = 115) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VisualProtocolError(f"cannot open video for blind package: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for index in range(stop):
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start:
                frames.append(frame)
    finally:
        capture.release()
    if len(frames) != stop - start:
        raise VisualProtocolError(f"video has insufficient frames for blind interval: {video_path}")
    return frames


def _encode_silent_panels(panels: Sequence[Sequence[np.ndarray]], output: Path) -> str:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    # All panels go through the same resize and codec path.  The central R panel
    # is included in every trial, but there is intentionally no audio stream.
    normalized: list[np.ndarray] = []
    for triplet in panels:
        pieces = [cv2.resize(np.asarray(frame), (224, 224), interpolation=cv2.INTER_AREA) for frame in triplet]
        normalized.append(np.concatenate(pieces, axis=1))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (672, 224))
    if not writer.isOpened():
        raise VisualProtocolError(f"cannot create blind video: {output}")
    try:
        for frame in normalized:
            writer.write(frame)
    finally:
        writer.release()
    return file_sha256(output)


def _transform_frames(frames: Sequence[np.ndarray], transform: str) -> list[np.ndarray]:
    if transform == "identity":
        return [np.asarray(frame).copy() for frame in frames]
    if transform == "reverse":
        return [np.asarray(frame).copy() for frame in reversed(frames)]
    if transform == "shift_plus5":
        if len(frames) <= 5:
            raise VisualProtocolError("blind QC shift requires at least six frames")
        return [np.asarray(frames[min(index + 5, len(frames) - 1)]).copy() for index in range(len(frames))]
    raise ValueError(f"unknown blind transform: {transform}")


def _video_row_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """Normalise A's permissive videos manifest into (sample,model,arm)."""

    sample = row.get("sample_id") or row.get("id")
    model = row.get("model") or row.get("generator") or row.get("model_name")
    arm = row.get("arm") or row.get("condition") or row.get("audio_arm") or row.get("video_arm")
    if isinstance(row.get("key"), str) and (not sample or not model or not arm):
        parts = str(row["key"]).split("/")
        if len(parts) >= 3:
            sample, model, arm = parts[:3]
    if not sample or not model or not arm:
        return None
    seed = row.get("seed")
    repeat = row.get("repeat_index", row.get("repeat", 0))
    arm_text = str(arm).upper()
    if arm_text in {"N", "C", "DELAY"} and seed is not None:
        arm_text += str(int(seed))
    if (arm_text.endswith("_REPEAT") or str(repeat) not in {"0", "None", ""}) and arm_text in {"N42", "C42"}:
        arm_text += "_repeat"
    model_text = str(model).lower().replace("-", "_")
    if model_text in {"wav2lip", "wav2lip_v1", "wav2lip_gan"}:
        model_text = "wav2lip"
    elif model_text in {"ditto", "ditto_trt", "ditto_pytorch"}:
        model_text = "ditto"
    return str(sample), model_text, arm_text


def iter_video_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: Any = payload.get("rows", payload.get("videos", payload.get("cells", [])))
    if isinstance(values, Mapping):
        flattened: list[dict[str, Any]] = []
        for key, value in values.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("key", key)
                flattened.append(item)
        values = flattened
    if not isinstance(values, list):
        return []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def load_video_manifest(run_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        run_root / "run" / "A" / "videos.json",
        run_root / "run" / "A" / "videos_manifest.json",
        run_root / "run" / "replacement" / "videos.json",
        run_root / "videos.json",
    ]
    for path in candidates:
        if path.is_file():
            return load_self_hashed(path), path
    return None, None


def normalise_video_rows(payload: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_video_rows(payload):
        key = _video_row_key(row)
        if key is None:
            continue
        if key in result:
            raise VisualProtocolError(f"duplicate A video key: {key}")
        path = row.get("path") or row.get("video_path") or row.get("output") or row.get("media")
        item = dict(row)
        item["sample_id"], item["model"], item["arm"] = key
        item["path"] = str(path) if path else None
        result[key] = item
    return result


def verify_video_item(item: Mapping[str, Any]) -> tuple[bool, str | None]:
    path_value = item.get("path")
    if not path_value:
        return False, "path_missing"
    path = Path(str(path_value))
    if not path.is_file():
        return False, "file_missing"
    status = item.get("status")
    if status is not None and str(status).lower() not in {"complete", "pass", "ok"}:
        return False, "status_not_complete"
    expected_sha = item.get("sha256") or item.get("video_sha256") or item.get("path_sha256")
    if expected_sha and str(expected_sha) != file_sha256(path):
        return False, "sha256_mismatch"
    return True, None


def anonymise_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_json_checked(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return write_json(path, dict(payload))


__all__ = [
    "BLIND_SEED",
    "BOOTSTRAP_COUNT",
    "BOOTSTRAP_SEED",
    "CALIBRATION_ERROR_THRESHOLD",
    "DEFAULT_LANDMARKER_ASSET",
    "FPS",
    "FRAME_COUNT",
    "FRAME_PERIOD_S",
    "FRAME_TOLERANCE_S",
    "J0",
    "LANDMARKER_OPTIONS",
    "MIN_OBSERVED_FRAMES",
    "MODELS",
    "MOUTH_SHAPE",
    "PRACTICAL_THRESHOLD",
    "SEEDS",
    "FeatureArray",
    "ProtocolError",
    "VisualProtocolError",
    "_encode_silent_panels",
    "_mse_components",
    "_read_video_frames",
    "_transform_frames",
    "anonymise_name",
    "bootstrap_metrics",
    "calibrate_real_feature",
    "calibration_pair",
    "crop_feature",
    "fixed_clock_support",
    "full_cohort_bounds",
    "group_decomposition",
    "import_video_features",
    "iter_video_rows",
    "load_feature",
    "load_video_manifest",
    "make_contact_sheet",
    "media_metadata",
    "normalise_video_rows",
    "read_json",
    "run_calibration",
    "safe_ratio",
    "save_feature",
    "validate_feature_arrays",
    "verify_video_item",
    "write_json",
    "write_json_checked",
]
