#!/usr/bin/env python3
"""CPU-only independent mouth-motion analysis for the source-axis experiment.

The primary endpoint is speaker-equal separation between normalized mouth
aperture during open-vowel and bilabial MFA intervals. MFA intervals label the
fixed natural-audio clock; they are not treated as exact articulatory events.
Event matching is exploratory and uses one latency and one extrema detector
fitted on calibration speakers only.

Manifest schema (JSON):
  {"schema_version": 1, "rows": [{"id": ..., "speaker": ...,
    "phones": [{"label": "p", "start_s": 0.1, "end_s": 0.2,
                "usable": true}],
    "arms": {"baseline": {"video_path": ...}, ...}}]}

For calibration rows only the baseline arm is required. Extraction replaces
video_path with a saved feature_path. Analysis requires all eight frozen arms:
baseline, full_plus/full_minus, orth_plus/orth_minus, and random1/random2/random3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
PROTOCOL_ID = "tts_mechanism_visual_20260930_v2"
CONDITIONS = (
    "baseline", "full_plus", "full_minus", "orth_plus", "orth_minus",
    "random1", "random2", "random3",
)
CALIBRATION_CONDITIONS = ("baseline",)
FPS_REFERENCE = 25.0
EVENT_TOLERANCE_FRAMES = 10
MIN_CALIBRATION_CLIP_COVERAGE = 0.80
MIN_CALIBRATION_SPEAKER_COVERAGE = 0.80
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930
CALIBRATION_SEED = 20260930
SMOOTH_WIDTH = 3
MIN_EVENT_DISTANCE_FRAMES = 3
CALIBRATION_PROMINENCE_QUANTILE = 0.10

# These are the already-used IPA/MFA broad-class rules from
# mfa_linear_phone_gradient.phone_class, restricted to the two frozen classes
# in this endpoint. Diacritics are stripped by the same normalization rule.
BILABIAL_BASES = frozenset({"p", "b", "m"})
OPEN_VOWEL_BASES = frozenset({"a", "ɑ", "ɒ", "ɐ", "æ"})
STRIPPED_PHONE_MARKS = ("ː", "ʰ", "ʲ", "ʷ", "̪")

CONDITION_ALIASES = {
    "baseline": "baseline",
    "base": "baseline",
    "global_plus": "full_plus",
    "plus": "full_plus",
    "full_plus": "full_plus",
    "global_minus": "full_minus",
    "minus": "full_minus",
    "full_minus": "full_minus",
    "strict_orth": "orth_plus",
    "level_orth": "orth_plus",
    "level-orth": "orth_plus",
    "orth_plus": "orth_plus",
    "orth_minus": "orth_minus",
    "random1": "random1",
    "random2": "random2",
    "random3": "random3",
}


class VisualMechanismError(ValueError):
    """Invalid or incomplete visual-mechanism input."""


@dataclass(frozen=True)
class MouthTrack:
    timestamps_s: np.ndarray
    aperture: np.ndarray
    valid: np.ndarray
    mouth_width: np.ndarray
    landmarks: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        n = len(self.timestamps_s)
        if self.timestamps_s.shape != (n,) or self.aperture.shape != (n,):
            raise VisualMechanismError("timestamps and aperture must be 1-D and frame matched")
        if self.valid.shape != (n,) or self.valid.dtype != np.bool_:
            raise VisualMechanismError("valid must be a boolean value per frame")
        if self.mouth_width.shape != (n,):
            raise VisualMechanismError("mouth_width must have one value per frame")
        if self.landmarks.ndim != 3 or self.landmarks.shape[0] != n or self.landmarks.shape[1:] != (478, 2):
            raise VisualMechanismError("landmarks must have shape (frames, 478, 2)")
        arrays = (self.timestamps_s, self.aperture, self.mouth_width, self.landmarks)
        if not all(np.isfinite(x).all() for x in arrays):
            raise VisualMechanismError("track arrays must be finite")
        if n > 1 and np.any(np.diff(self.timestamps_s) <= 0):
            raise VisualMechanismError("timestamps must be strictly increasing")
        if np.any(self.aperture[~self.valid] != 0.0):
            raise VisualMechanismError("invalid-frame aperture must be zero")

    @property
    def fps(self) -> float:
        if len(self.timestamps_s) < 2:
            return float(self.metadata.get("fps", FPS_REFERENCE))
        return float(1.0 / np.median(np.diff(self.timestamps_s)))


def normalize_phone(label: Any) -> str:
    value = str(label or "").strip().lower()
    for mark in STRIPPED_PHONE_MARKS:
        value = value.replace(mark, "")
    return value


def phone_class(label: Any) -> str | None:
    normalized = normalize_phone(label)
    if normalized in BILABIAL_BASES:
        return "bilabial"
    if normalized in OPEN_VOWEL_BASES:
        return "open_vowel"
    return None


def _canonical_condition(name: Any) -> str:
    value = str(name).strip()
    return CONDITION_ALIASES.get(value, CONDITION_ALIASES.get(value.lower(), ""))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VisualMechanismError("manifest must be a JSON object")
    return value


def _manifest_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise VisualMechanismError(f"manifest schema_version must be {SCHEMA_VERSION}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise VisualMechanismError("manifest rows must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise VisualMechanismError("each manifest row must be an object")
        row = dict(raw)
        identifier = str(row.get("id", row.get("sample_id", "")))
        speaker = str(row.get("speaker", ""))
        if not identifier or not speaker:
            raise VisualMechanismError("each row needs id/sample_id and speaker")
        if identifier in seen:
            raise VisualMechanismError(f"duplicate sample id: {identifier}")
        seen.add(identifier)
        row["id"] = identifier
        row["speaker"] = speaker
        row["phones"] = _phone_spans(row)
        arms = row.get("arms")
        if not isinstance(arms, Mapping):
            raise VisualMechanismError(f"{identifier}: arms must be an object")
        normalized: dict[str, Any] = {}
        for key, arm in arms.items():
            condition = _canonical_condition(key)
            if not condition:
                continue
            if condition in normalized:
                raise VisualMechanismError(f"{identifier}: duplicate arm for {condition}")
            if isinstance(arm, Mapping):
                normalized[condition] = dict(arm)
            elif isinstance(arm, str):
                normalized[condition] = {"path": arm}
            else:
                raise VisualMechanismError(f"{identifier}/{condition}: arm must be an object")
        row["arms"] = normalized
        result.append(row)
    return result


def _phone_spans(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: Any = row.get("phones", row.get("phone_intervals", row.get("tokens")))
    if values is None and isinstance(row.get("arms"), Mapping):
        natural = row["arms"].get("N", row["arms"].get("natural", {}))
        if isinstance(natural, Mapping):
            values = natural.get("tokens", natural.get("phones"))
    if not isinstance(values, list):
        raise VisualMechanismError(f"{row.get('id', '<row>')}: phone spans are missing")
    result = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise VisualMechanismError("phone interval must be an object")
        label = raw.get("label", raw.get("phone", raw.get("token")))
        start = raw.get("start_s", raw.get("lo_s", raw.get("lo_decimal", raw.get("start"))))
        end = raw.get("end_s", raw.get("hi_s", raw.get("hi_decimal", raw.get("end"))))
        if label is None or start is None or end is None:
            raise VisualMechanismError(f"phone interval {index} needs label/start/end")
        try:
            lo, hi = float(start), float(end)
        except (TypeError, ValueError) as exc:
            raise VisualMechanismError(f"invalid phone interval {index}") from exc
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
            raise VisualMechanismError(f"invalid phone interval bounds at {index}")
        if raw.get("usable", True) is False or hi <= lo:
            continue
        result.append({"label": str(label), "start_s": lo, "end_s": hi})
    return result


def _load_track(path: str | Path, expected_sha256: str | None = None) -> MouthTrack:
    target = Path(path)
    if expected_sha256 and _sha256_file(target) != str(expected_sha256):
        raise VisualMechanismError(f"feature checksum mismatch: {target}")
    try:
        with np.load(target, allow_pickle=False) as data:
            required = {"timestamps_s", "valid", "mouth_open", "mouth_width", "landmarks"}
            if not required.issubset(data.files):
                raise VisualMechanismError(f"{target}: missing {sorted(required - set(data.files))}")
            metadata: dict[str, Any] = {}
            if "metadata_json" in data.files:
                metadata_value = json.loads(str(data["metadata_json"].item()))
                if isinstance(metadata_value, dict):
                    metadata = metadata_value
            return MouthTrack(
                timestamps_s=np.asarray(data["timestamps_s"], dtype=np.float64),
                aperture=np.asarray(data["mouth_open"], dtype=np.float64),
                valid=np.asarray(data["valid"], dtype=bool),
                mouth_width=np.asarray(data["mouth_width"], dtype=np.float64),
                landmarks=np.asarray(data["landmarks"], dtype=np.float64),
                metadata=metadata,
            )
    except VisualMechanismError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise VisualMechanismError(f"cannot load feature sequence: {target}") from exc


def _arm_feature_path(arm: Mapping[str, Any]) -> str | None:
    for key in ("feature_path", "features_path", "feature", "path"):
        value = arm.get(key)
        if value:
            return str(value)
    return None


def _arm_video_path(arm: Mapping[str, Any]) -> str | None:
    for key in ("video_path", "path", "video"):
        value = arm.get(key)
        if value:
            return str(value)
    return None


def extract_manifest(
    input_manifest: str | Path,
    output_manifest: str | Path,
    output_root: str | Path,
    landmarker_asset: str | Path,
) -> dict[str, Any]:
    """Extract the same MediaPipe mouth trajectory for every saved video on CPU."""
    asset = Path(landmarker_asset).resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"MediaPipe asset does not exist: {asset}")
    from scripts.experiments.lrs3_tts_visual_advantage.video_features import (
        create_landmarker,
        extract_video_features,
        save_visual_sequence,
    )
    import cv2
    import mediapipe as mp

    payload = _read_json(input_manifest)
    rows = _manifest_rows(payload)
    target_root = Path(output_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    handle = create_landmarker(asset)
    try:
        for row in rows:
            for condition, arm in row["arms"].items():
                video_path = _arm_video_path(arm)
                if not video_path:
                    existing = _arm_feature_path(arm)
                    if existing and Path(existing).is_file():
                        continue
                    raise VisualMechanismError(f"{row['id']}/{condition}: video_path is missing")
                video = Path(video_path).resolve()
                expected_video_sha = arm.get("video_sha256")
                video_sha = _sha256_file(video)
                if expected_video_sha and video_sha != str(expected_video_sha):
                    raise VisualMechanismError(f"video checksum mismatch: {video}")
                destination = target_root / row["id"] / f"{condition.replace('+', 'plus').replace('-', 'minus')}.npz"
                sequence = extract_video_features(video, asset, landmarker_handle=handle)
                save_visual_sequence(destination, sequence)
                arm["video_path"] = str(video)
                arm["video_sha256"] = video_sha
                arm["feature_path"] = str(destination)
                arm["feature_sha256"] = _sha256_file(destination)
                arm["frame_count"] = int(len(sequence.valid))
                arm["fps"] = float(sequence.metadata["fps"])
                arm["valid_fraction"] = float(sequence.valid.mean()) if len(sequence.valid) else 0.0
    finally:
        handle[1].close()
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "features_extracted",
        "extractor": "scripts.experiments.lrs3_tts_visual_advantage.video_features",
        "landmarker_asset": str(asset),
        "landmarker_asset_sha256": _sha256_file(asset),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mediapipe": str(mp.__version__),
            "opencv": str(cv2.__version__),
            "device": "CPU",
        },
        "producer_code_sha256": {
            "visual_analysis": _sha256_file(Path(__file__).resolve()),
            "video_features": _sha256_file(Path(extract_video_features.__code__.co_filename).resolve()),
        },
        "rows": rows,
    }
    _write_json(output_manifest, result)
    return result


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _row_tracks(row: Mapping[str, Any], required: Sequence[str]) -> dict[str, MouthTrack]:
    tracks: dict[str, MouthTrack] = {}
    for condition in required:
        arm = row["arms"].get(condition)
        if not isinstance(arm, Mapping):
            raise VisualMechanismError(f"{row['id']}: missing required arm {condition}")
        feature_path = _arm_feature_path(arm)
        if not feature_path:
            raise VisualMechanismError(f"{row['id']}/{condition}: feature_path is missing")
        tracks[condition] = _load_track(feature_path, arm.get("feature_sha256"))
    return tracks


def _available_row_tracks(
    row: Mapping[str, Any], required: Sequence[str]
) -> tuple[dict[str, MouthTrack], dict[str, str]]:
    """Load available arms while preserving unavailable rows in the denominator."""
    tracks: dict[str, MouthTrack] = {}
    missing: dict[str, str] = {}
    for condition in required:
        arm = row["arms"].get(condition)
        if not isinstance(arm, Mapping):
            missing[condition] = "arm_missing"
            continue
        feature_path = _arm_feature_path(arm)
        if not feature_path:
            missing[condition] = "feature_path_missing"
            continue
        target = Path(feature_path)
        if not target.is_file():
            missing[condition] = "feature_file_missing"
            continue
        try:
            tracks[condition] = _load_track(target, arm.get("feature_sha256"))
        except (OSError, VisualMechanismError) as exc:
            missing[condition] = f"feature_invalid:{type(exc).__name__}"
    return tracks, missing


def _common_clock(
    tracks: Mapping[str, MouthTrack],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Use the full baseline clock and one monotonic common-valid support."""
    if "baseline" not in tracks:
        raise VisualMechanismError("baseline track is required as the recipient clock")
    baseline = tracks["baseline"]
    base_times = baseline.timestamps_s
    if len(base_times) > 1:
        tolerance = 0.5 * float(np.median(np.diff(base_times))) + 1e-12
    else:
        tolerance = 0.5 / float(baseline.metadata.get("fps", FPS_REFERENCE))
    maps: dict[str, np.ndarray] = {}
    deltas: dict[str, list[float]] = {}
    for name, track in tracks.items():
        index_map = np.full(len(base_times), -1, dtype=np.int64)
        i = j = 0
        while i < len(base_times) and j < len(track.timestamps_s):
            delta = float(track.timestamps_s[j] - base_times[i])
            if abs(delta) <= tolerance:
                index_map[i] = j
                i += 1
                j += 1
            elif delta < -tolerance:
                j += 1
            else:
                i += 1
        maps[name] = index_map
        deltas[name] = [
            float(track.timestamps_s[k] - base_times[index])
            for index, k in enumerate(index_map)
            if k >= 0
        ]
    common = np.ones(len(base_times), dtype=bool)
    for name, track in tracks.items():
        index_map = maps[name]
        matched = index_map >= 0
        current = np.zeros(len(base_times), dtype=bool)
        current[matched] = track.valid[index_map[matched]]
        common &= current
    support = {
        "recipient_clock": "full baseline timestamps; no fixed-length crop",
        "clock_frames": int(len(base_times)),
        "valid_common_frames": int(common.sum()),
        "common_coverage": float(common.mean()) if len(common) else 0.0,
        "timestamp_tolerance_s": tolerance,
        "matched_frames_by_arm": {name: int(np.sum(index_map >= 0)) for name, index_map in maps.items()},
        "valid_frames_by_arm_on_clock": {},
        "max_abs_clock_delta_s_by_arm": {
            name: max((abs(x) for x in values), default=None) for name, values in deltas.items()
        },
    }
    for name, track in tracks.items():
        index_map = maps[name]
        matched = index_map >= 0
        support["valid_frames_by_arm_on_clock"][name] = int(np.sum(track.valid[index_map[matched]]))
    return common, maps, support


def _pose_proxies(track: MouthTrack, indices: np.ndarray) -> dict[str, np.ndarray]:
    landmarks = track.landmarks[indices]
    left = landmarks[:, 33, :]
    right = landmarks[:, 263, :]
    nose = landmarks[:, 1, :]
    eye_distance = np.linalg.norm(right - left, axis=1)
    eye_distance = np.maximum(eye_distance, 1e-12)
    center = 0.5 * (left + right)
    yaw_proxy = 2.0 * (nose[:, 0] - center[:, 0]) / eye_distance
    pitch_proxy = (nose[:, 1] - center[:, 1]) / eye_distance
    return {"yaw_proxy": yaw_proxy, "pitch_proxy": pitch_proxy}


def _summary_values(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"mean": None, "sd": None, "p10": None, "p90": None}
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=0)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _global_track_metrics(
    track: MouthTrack,
    index_map: np.ndarray,
    common: np.ndarray,
    base_times: np.ndarray,
) -> dict[str, Any]:
    base_indices = np.flatnonzero(common)
    current_indices = index_map[base_indices]
    aperture = track.aperture[current_indices]
    width = track.mouth_width[current_indices]
    pose = _pose_proxies(track, current_indices)
    # Velocity uses only adjacent frames on the shared recipient clock. Missing
    # frames create a gap; they are never bridged by interpolation.
    adjacent = (base_indices[1:] == base_indices[:-1] + 1) if len(base_indices) > 1 else np.asarray([], dtype=bool)
    if adjacent.any():
        current_pair = current_indices[:-1][adjacent]
        next_pair = current_indices[1:][adjacent]
        dt = base_times[base_indices[1:][adjacent]] - base_times[base_indices[:-1][adjacent]]
        velocity = np.abs(track.aperture[next_pair] - track.aperture[current_pair]) / dt
    else:
        velocity = np.asarray([], dtype=np.float64)
    return {
        "common_frame_count": int(len(base_indices)),
        "aperture": _summary_values(aperture),
        "mouth_width": _summary_values(width),
        "absolute_aperture_velocity_per_s": _summary_values(velocity),
        "yaw_proxy": _summary_values(pose["yaw_proxy"]),
        "pitch_proxy": _summary_values(pose["pitch_proxy"]),
        "valid_fraction_on_full_clock": float(len(base_indices) / len(base_times)) if len(base_times) else 0.0,
    }


def _phone_event_targets(phones: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    targets: dict[str, list[dict[str, Any]]] = {"minimum": [], "maximum": []}
    for phone in phones:
        category = phone_class(phone["label"])
        if category is None:
            continue
        kind = "minimum" if category == "bilabial" else "maximum"
        targets[kind].append({
            "category": category,
            "label": str(phone["label"]),
            "start_s": float(phone["start_s"]),
            "end_s": float(phone["end_s"]),
            "time_s": 0.5 * (float(phone["start_s"]) + float(phone["end_s"])),
        })
    return targets


def _phone_apertures(
    phones: Sequence[Mapping[str, Any]],
    track: MouthTrack,
    frame_indices: np.ndarray,
    latency_s: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    times = track.timestamps_s[frame_indices]
    aperture = track.aperture[frame_indices]
    token_rows: list[dict[str, Any]] = []
    expected = {"bilabial": 0, "open_vowel": 0}
    for phone in phones:
        category = phone_class(phone["label"])
        if category is None:
            continue
        expected[category] += 1
        start = float(phone["start_s"]) + latency_s
        end = float(phone["end_s"]) + latency_s
        mask = (times >= start) & (times < end)
        selected = aperture[mask]
        token_rows.append({
            "label": str(phone["label"]),
            "category": category,
            "start_s": float(phone["start_s"]),
            "end_s": float(phone["end_s"]),
            "requested_common_frames": int(mask.sum()),
            "observed_frames": int(selected.size),
            "observed": bool(selected.size),
            "aperture_mean": float(selected.mean()) if selected.size else None,
            "aperture_median": float(np.median(selected)) if selected.size else None,
        })
    return token_rows, expected


def _moving_average_segments(values: np.ndarray, valid: np.ndarray, width: int = SMOOTH_WIDTH) -> np.ndarray:
    if width < 1 or width % 2 == 0:
        raise VisualMechanismError("smoothing width must be a positive odd integer")
    out = np.zeros(len(values), dtype=np.float64)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return out
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    for segment in np.split(indices, cuts):
        if not len(segment):
            continue
        segment_values = values[segment]
        if width == 1 or len(segment_values) < width:
            out[segment] = segment_values
        else:
            out[segment] = np.convolve(segment_values, np.ones(width) / width, mode="same")
    return out


def _extrema_candidates(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    min_distance: int = MIN_EVENT_DISTANCE_FRAMES,
) -> dict[str, list[dict[str, float | int]]]:
    """Return local extrema and their symmetric, fixed-window prominence."""
    smooth = _moving_average_segments(np.asarray(values, dtype=np.float64), valid)
    radius = max(1, min_distance)
    raw: dict[str, list[dict[str, float | int]]] = {"minimum": [], "maximum": []}
    indices = np.flatnonzero(valid)
    if not len(indices):
        return raw
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    for segment in np.split(indices, cuts):
        if len(segment) < 2 * radius + 1:
            continue
        lo, hi = int(segment[0]), int(segment[-1]) + 1
        for index in range(lo + radius, hi - radius):
            if not valid[index - radius:index + radius + 1].all():
                continue
            left = smooth[index - radius:index]
            right = smooth[index + 1:index + radius + 1]
            value = float(smooth[index])
            if value >= float(np.max(left)) and value > float(np.max(right)):
                prominence = max(0.0, value - max(float(np.min(left)), float(np.min(right))))
                raw["maximum"].append({"index": index, "prominence": prominence, "value": value})
            if value <= float(np.min(left)) and value < float(np.min(right)):
                prominence = max(0.0, min(float(np.max(left)), float(np.max(right))) - value)
                raw["minimum"].append({"index": index, "prominence": prominence, "value": value})
    return raw


def _select_extrema(
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_prominence: float,
    min_distance: int = MIN_EVENT_DISTANCE_FRAMES,
) -> list[dict[str, float | int]]:
    eligible = [dict(x) for x in candidates if float(x["prominence"]) >= min_prominence]
    eligible.sort(key=lambda x: (-float(x["prominence"]), int(x["index"])))
    selected: list[dict[str, float | int]] = []
    for item in eligible:
        index = int(item["index"])
        if all(abs(index - int(old["index"])) >= min_distance for old in selected):
            selected.append(item)
    return sorted(selected, key=lambda x: int(x["index"]))


def _detect_events(
    track: MouthTrack,
    *,
    min_prominence: float,
    timestamp_offset_s: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    candidates = _extrema_candidates(track.aperture, track.valid)
    events: dict[str, list[dict[str, Any]]] = {"minimum": [], "maximum": []}
    for kind in events:
        selected = _select_extrema(candidates[kind], min_prominence=min_prominence)
        for item in selected:
            index = int(item["index"])
            events[kind].append({
                "index": index,
                "time_s": float(track.timestamps_s[index] + timestamp_offset_s),
                "value": float(item["value"]),
                "prominence": float(item["prominence"]),
            })
    return events


def _ordered_match(
    expected_times: Sequence[float],
    detected: Sequence[Mapping[str, Any]],
    *,
    tolerance_s: float,
) -> dict[str, Any]:
    """Monotonic one-to-one matching; each missed target costs one tolerance."""
    targets = np.asarray(sorted(float(x) for x in expected_times), dtype=np.float64)
    events = sorted((dict(x) for x in detected), key=lambda x: float(x["time_s"]))
    event_times = np.asarray([float(x["time_s"]) for x in events], dtype=np.float64)
    n, m = len(targets), len(events)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    action = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, :] = 0.0  # extra peaks are counted separately, not as missed phones
    action[0, 1:] = 2
    for i in range(1, n + 1):
        cost[i, 0] = i * tolerance_s
        action[i, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(cost[i - 1, j] + tolerance_s, 1), (cost[i, j - 1], 2)]
            delta = abs(float(targets[i - 1] - event_times[j - 1]))
            if delta <= tolerance_s + 1e-12:
                options.append((cost[i - 1, j - 1] + delta, 3))
            # Prefer a real match on equal costs, then a stable skip order.
            value, move = min(options, key=lambda x: (x[0], 0 if x[1] == 3 else x[1]))
            cost[i, j], action[i, j] = value, move
    matches: list[dict[str, Any]] = []
    missed: list[float] = []
    i, j = n, m
    while i or j:
        move = int(action[i, j])
        if move == 3:
            matches.append({
                "expected_s": float(targets[i - 1]),
                "detected_s": float(event_times[j - 1]),
                "signed_error_s": float(event_times[j - 1] - targets[i - 1]),
                "abs_error_s": float(abs(event_times[j - 1] - targets[i - 1])),
            })
            i -= 1
            j -= 1
        elif move == 1:
            missed.append(float(targets[i - 1]))
            i -= 1
        elif move == 2:
            j -= 1
        else:
            break
    matches.reverse()
    nmiss = len(missed)
    penalized = (sum(x["abs_error_s"] for x in matches) + nmiss * tolerance_s) / n if n else None
    signed = [x["signed_error_s"] for x in matches]
    return {
        "expected_count": n,
        "detected_count": m,
        "matched_count": len(matches),
        "missed_count": nmiss,
        "unmatched_detected_count": max(0, m - len(matches)),
        "match_fraction": float(len(matches) / n) if n else None,
        "penalized_abs_error_s": float(penalized) if penalized is not None else None,
        "mean_signed_error_s": float(np.mean(signed)) if signed else None,
        "median_signed_error_s": float(np.median(signed)) if signed else None,
        "matches": matches,
        "miss_penalty_s": tolerance_s,
    }


def _event_result(
    track: MouthTrack,
    phones: Sequence[Mapping[str, Any]],
    *,
    min_prominence: float,
    latency_s: float,
    tolerance_s: float,
    timestamp_offset_s: float = 0.0,
    aperture_override: np.ndarray | None = None,
) -> dict[str, Any]:
    if aperture_override is None:
        use_track = track
    else:
        override = np.asarray(aperture_override, dtype=np.float64)
        use_track = MouthTrack(
            timestamps_s=track.timestamps_s,
            aperture=override,
            valid=track.valid.copy(),
            mouth_width=track.mouth_width,
            landmarks=track.landmarks,
            metadata=track.metadata,
        )
    events = _detect_events(use_track, min_prominence=min_prominence, timestamp_offset_s=timestamp_offset_s)
    targets = _phone_event_targets(phones)
    output: dict[str, Any] = {}
    for kind in ("minimum", "maximum"):
        expected = [target["time_s"] + latency_s for target in targets[kind]]
        output[kind] = _ordered_match(expected, events[kind], tolerance_s=tolerance_s)
        output[kind]["event_times_s"] = [float(x["time_s"]) for x in events[kind]]
    return output


def _fit_prominence(rows: Sequence[dict[str, Any]], tracks: Mapping[str, MouthTrack]) -> float:
    values: list[float] = []
    for row in rows:
        track = tracks[row["id"]]
        candidates = _extrema_candidates(track.aperture, track.valid)
        values.extend(float(x["prominence"]) for items in candidates.values() for x in items if float(x["prominence"]) > 0.0)
    if not values:
        return 1e-8
    return max(1e-8, float(np.quantile(np.asarray(values, dtype=np.float64), CALIBRATION_PROMINENCE_QUANTILE)))


def _cal_clip_counts(
    rows: Sequence[dict[str, Any]],
    tracks: Mapping[str, MouthTrack],
    latency_s: float | None,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    by_clip: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = by_clip.setdefault(row["id"], {"bilabial": 0, "open_vowel": 0})
        track = tracks[row["id"]]
        offset = float(latency_s) if latency_s is not None else 0.0
        for phone in row["phones"]:
            category = phone_class(phone["label"])
            if category is None:
                continue
            in_interval = (track.timestamps_s >= float(phone["start_s"]) + offset) & (
                track.timestamps_s < float(phone["end_s"]) + offset
            )
            if np.any(in_interval & track.valid):
                counts[category] += 1
    minimum: dict[str, int] = {}
    support: dict[str, int] = {}
    for category in ("bilabial", "open_vowel"):
        values = np.asarray([row[category] for row in by_clip.values()], dtype=np.int64)
        threshold = 1
        max_count = int(values.max()) if values.size else 0
        for candidate in range(1, max_count + 1):
            if float(np.mean(values >= candidate)) >= MIN_CALIBRATION_CLIP_COVERAGE:
                threshold = candidate
            else:
                break
        minimum[category] = threshold
        support[category] = int(np.sum(values >= threshold))
    return by_clip, {"clip_count": len(by_clip), "min_tokens_per_clip_by_class": minimum,
                        "calibration_clips_meeting_min_by_class": support,
                        "required_calibration_clip_coverage": MIN_CALIBRATION_CLIP_COVERAGE}


def _fit_latency(
    rows: Sequence[dict[str, Any]],
    tracks: Mapping[str, MouthTrack],
    *,
    min_prominence: float,
) -> dict[str, Any]:
    tolerance_s = EVENT_TOLERANCE_FRAMES / FPS_REFERENCE
    signed: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    for row in rows:
        result = _event_result(
            tracks[row["id"]], row["phones"], min_prominence=min_prominence,
            latency_s=0.0, tolerance_s=tolerance_s,
        )
        matched = [
            float(match["signed_error_s"])
            for kind in ("minimum", "maximum")
            for match in result[kind]["matches"]
        ]
        signed.extend(matched)
        sample_rows.append({"id": row["id"], "speaker": row["speaker"], "matched_event_count": len(matched)})
    latency = float(np.median(np.asarray(signed, dtype=np.float64))) if signed else None
    speakers_with_matches = {
        item["speaker"] for item in sample_rows if int(item["matched_event_count"]) > 0
    }
    return {
        "latency_s": latency,
        "latency_frames_at_25fps": float(latency * FPS_REFERENCE) if latency is not None else None,
        "calibration_matched_event_count": len(signed),
        "calibration_speakers_with_matched_events": len(speakers_with_matches),
        "calibration_rows": sample_rows,
        "rule": "median signed event-minus-MFA-midpoint offset from calibration baseline tracks only",
    }


def _calibration_controls(
    rows: Sequence[dict[str, Any]],
    tracks: Mapping[str, MouthTrack],
    *,
    min_prominence: float,
    latency_s: float | None,
) -> dict[str, Any]:
    if latency_s is None:
        return {"status": "NOT_RUN_NO_CALIBRATED_LATENCY"}
    tolerance_s = EVENT_TOLERANCE_FRAMES / FPS_REFERENCE
    seed = np.random.Generator(np.random.PCG64(CALIBRATION_SEED))
    base_signed: list[float] = []
    plus_signed: list[float] = []
    minus_signed: list[float] = []
    base_match: list[float] = []
    shuffled_match: list[float] = []
    base_penalized: list[float] = []
    shuffled_penalized: list[float] = []
    static_event_counts: list[int] = []
    aperture_shuffle_exact: list[bool] = []
    for row in rows:
        track = tracks[row["id"]]
        base = _event_result(track, row["phones"], min_prominence=min_prominence,
                             latency_s=latency_s, tolerance_s=tolerance_s)
        plus = _event_result(track, row["phones"], min_prominence=min_prominence,
                             latency_s=latency_s, tolerance_s=tolerance_s,
                             timestamp_offset_s=5.0 / FPS_REFERENCE)
        minus = _event_result(track, row["phones"], min_prominence=min_prominence,
                              latency_s=latency_s, tolerance_s=tolerance_s,
                              timestamp_offset_s=-5.0 / FPS_REFERENCE)
        for result, destination in ((base, base_signed), (plus, plus_signed), (minus, minus_signed)):
            destination.extend(
                float(match["signed_error_s"])
                for kind in ("minimum", "maximum")
                for match in result[kind]["matches"]
            )
        expected = sum(base[kind]["expected_count"] for kind in ("minimum", "maximum"))
        matched = sum(base[kind]["matched_count"] for kind in ("minimum", "maximum"))
        base_match.append(float(matched / expected) if expected else 0.0)
        base_penalty = sum(
            float(base[kind]["penalized_abs_error_s"] or 0.0) * int(base[kind]["expected_count"])
            for kind in ("minimum", "maximum")
        )
        base_penalized.append(float(base_penalty / expected) if expected else tolerance_s)
        static = np.zeros_like(track.aperture)
        if track.valid.any():
            static[track.valid] = float(np.median(track.aperture[track.valid]))
        static_result = _event_result(track, row["phones"], min_prominence=min_prominence,
                                      latency_s=latency_s, tolerance_s=tolerance_s,
                                      aperture_override=static)
        static_event_counts.append(sum(static_result[kind]["detected_count"] for kind in ("minimum", "maximum")))
        shuffled = track.aperture.copy()
        valid_values = shuffled[track.valid].copy()
        shuffled[track.valid] = valid_values[seed.permutation(len(valid_values))]
        shuffled_result = _event_result(track, row["phones"], min_prominence=min_prominence,
                                        latency_s=latency_s, tolerance_s=tolerance_s,
                                        aperture_override=shuffled)
        shuffled_expected = sum(shuffled_result[kind]["expected_count"] for kind in ("minimum", "maximum"))
        shuffled_count = sum(shuffled_result[kind]["matched_count"] for kind in ("minimum", "maximum"))
        shuffled_match.append(float(shuffled_count / shuffled_expected) if shuffled_expected else 0.0)
        shuffled_penalty = sum(
            float(shuffled_result[kind]["penalized_abs_error_s"] or 0.0) * int(shuffled_result[kind]["expected_count"])
            for kind in ("minimum", "maximum")
        )
        shuffled_penalized.append(float(shuffled_penalty / shuffled_expected) if shuffled_expected else tolerance_s)
        aperture_shuffle_exact.append(bool(np.array_equal(np.sort(track.aperture), np.sort(shuffled))))

    med_base = float(np.median(base_signed)) if base_signed else None
    med_plus = float(np.median(plus_signed)) if plus_signed else None
    med_minus = float(np.median(minus_signed)) if minus_signed else None
    plus_shift = (med_plus - med_base) * FPS_REFERENCE if med_plus is not None and med_base is not None else None
    minus_shift = (med_minus - med_base) * FPS_REFERENCE if med_minus is not None and med_base is not None else None
    plus_ok = plus_shift is not None and abs(plus_shift - 5.0) <= 1.0
    minus_ok = minus_shift is not None and abs(minus_shift + 5.0) <= 1.0
    static_ok = bool(static_event_counts) and all(x == 0 for x in static_event_counts)
    shuffle_degraded = bool(
        base_penalized and shuffled_penalized
        and np.median(shuffled_penalized) > np.median(base_penalized) + 0.5 / FPS_REFERENCE
    )
    distribution_preserved = bool(aperture_shuffle_exact and all(aperture_shuffle_exact))
    passed = bool(plus_ok and minus_ok and static_ok and shuffle_degraded and distribution_preserved)
    return {
        "status": "EVENT_METRIC_CALIBRATED" if passed else "EVENT_METRIC_NOT_CALIBRATED",
        "known_shift_controls": {
            "plus5_median_signed_error_shift_frames": plus_shift,
            "minus5_median_signed_error_shift_frames": minus_shift,
            "plus5_sign_and_size_pass": bool(plus_ok),
            "minus5_sign_and_size_pass": bool(minus_ok),
            "expected_shift_frames": {"plus": 5, "minus": -5},
            "tolerance_frames": 1.0,
        },
        "static_repeat_control": {
            "detected_event_counts": static_event_counts,
            "pass": static_ok,
        },
        "frame_shuffle_control": {
            "baseline_match_fraction_by_clip": base_match,
            "shuffle_match_fraction_by_clip": shuffled_match,
            "baseline_penalized_abs_error_s_by_clip": base_penalized,
            "shuffle_penalized_abs_error_s_by_clip": shuffled_penalized,
            "median_penalized_error_increased": shuffle_degraded,
            "aperture_distribution_exactly_preserved": distribution_preserved,
            "pass": bool(shuffle_degraded and distribution_preserved),
        },
        "median_signed_error_frames": {
            "unshifted": med_base * FPS_REFERENCE if med_base is not None else None,
            "plus5": med_plus * FPS_REFERENCE if med_plus is not None else None,
            "minus5": med_minus * FPS_REFERENCE if med_minus is not None else None,
        },
    }


def calibrate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Freeze lag, event detector threshold, and phone count minima from CAL only."""
    payload = _read_json(manifest_path)
    rows = _manifest_rows(payload)
    if any("evaluation" in str(row.get("split", "")).lower() for row in rows):
        raise VisualMechanismError("calibration manifest contains evaluation rows")
    clips_per_speaker = {speaker: sum(row["speaker"] == speaker for row in rows) for speaker in {row["speaker"] for row in rows}}
    if any(count != 2 for count in clips_per_speaker.values()):
        raise VisualMechanismError("calibration requires exactly two manifest clips per speaker")
    tracks: dict[str, MouthTrack] = {}
    for row in rows:
        arm = row["arms"].get("baseline")
        if not isinstance(arm, Mapping) or not _arm_feature_path(arm):
            raise VisualMechanismError(f"{row['id']}: calibration needs a baseline feature")
        tracks[row["id"]] = _load_track(_arm_feature_path(arm), arm.get("feature_sha256"))
    if len({row["speaker"] for row in rows}) < 4:
        raise VisualMechanismError("calibration needs at least four speakers for a stable latency fit")
    prominence = _fit_prominence(rows, tracks)
    latency = _fit_latency(rows, tracks, min_prominence=prominence)
    _clip_counts, count_gate = _cal_clip_counts(rows, tracks, latency["latency_s"])
    controls = _calibration_controls(rows, tracks, min_prominence=prominence, latency_s=latency["latency_s"])
    count_support_pass = all(
        count_gate["calibration_clips_meeting_min_by_class"][key]
        >= math.ceil(MIN_CALIBRATION_CLIP_COVERAGE * count_gate["clip_count"])
        for key in ("bilabial", "open_vowel")
    )
    required_latency_speakers = math.ceil(MIN_CALIBRATION_SPEAKER_COVERAGE * len({row["speaker"] for row in rows}))
    latency_support_pass = bool(
        latency["latency_s"] is not None
        and latency["calibration_matched_event_count"] >= 10
        and latency["calibration_speakers_with_matched_events"] >= required_latency_speakers
    )
    shift_control_pass = bool(
        controls.get("known_shift_controls", {}).get("plus5_sign_and_size_pass")
        and controls.get("known_shift_controls", {}).get("minus5_sign_and_size_pass")
    )
    phone_symbols = {
        "normalization_removes": list(STRIPPED_PHONE_MARKS),
        "bilabial": sorted(BILABIAL_BASES),
        "open_vowel": sorted(OPEN_VOWEL_BASES),
        "source": "restricted, frozen IPA-to-articulatory mapping already used in mfa_linear_phone_gradient.py",
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "conditions": list(CONDITIONS),
        "clock": "full recipient natural-video timestamps; no fixed 140-frame crop; monotonic common valid support",
        "phone_primary": "per-clip equal-token class means and open-vowel minus bilabial separation; two clips equally averaged within speaker",
        "phone_inventory": phone_symbols,
        "phone_inventory_sha256": hashlib.sha256(json.dumps(phone_symbols, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "latency": latency,
        "min_prominence": prominence,
        "min_prominence_fit": "10th percentile of positive local-extremum prominences from CAL baseline tracks",
        "smoothing_width_frames": SMOOTH_WIDTH,
        "minimum_event_distance_frames": MIN_EVENT_DISTANCE_FRAMES,
        "event_tolerance_frames": EVENT_TOLERANCE_FRAMES,
        "phone_min_common_frames": 1,
        "min_phone_tokens_per_clip_by_class": count_gate["min_tokens_per_clip_by_class"],
        "phone_min_count_calibration": count_gate,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "speaker", "CI": [0.005, 0.995]},
        "primary_contrasts": ["orth_plus - baseline", "orth_plus - mean(random1, random2, random3)"],
        "event_timing_claim_scope": "exploratory only; event matching is not a primary endpoint",
        "pose_and_global_motion": "separate descriptive metrics; not included in phone separation",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "CALIBRATION_COMPLETE",
        "created_from_manifest": str(Path(manifest_path).resolve()),
        "calibration_rows": len(rows),
        "calibration_speakers": len({row["speaker"] for row in rows}),
        "protocol": protocol,
        "event_controls": controls,
        "phone_count_calibration_pass": bool(count_support_pass),
        "phone_count_calibration_supported": bool(count_support_pass),
        "event_metric_calibrated": controls.get("status") == "EVENT_METRIC_CALIBRATED",
        "latency_support_pass": latency_support_pass,
        "latency_support_gate": {
            "matched_events": latency["calibration_matched_event_count"],
            "minimum_matched_events": 10,
            "speakers_with_matched_events": latency["calibration_speakers_with_matched_events"],
            "minimum_speakers": required_latency_speakers,
        },
        "positive_phone_endpoint_eligible": bool(count_support_pass and latency_support_pass and shift_control_pass),
    }


def _speaker_primary_values(
    rows: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    latency_s = calibration.get("protocol", {}).get("latency", {}).get("latency_s")
    latency_s = float(latency_s) if latency_s is not None else 0.0
    minimum_counts = calibration.get("protocol", {}).get("min_phone_tokens_per_clip_by_class", {})
    minimum_counts = {
        "bilabial": int(minimum_counts.get("bilabial", 1)),
        "open_vowel": int(minimum_counts.get("open_vowel", 1)),
    }
    clip_rows: list[dict[str, Any]] = []
    speaker_clips: dict[str, list[dict[str, Any]]] = {}
    speaker_global: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    speaker_event: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        speaker = row["speaker"]
        speaker_clips.setdefault(speaker, [])
        speaker_global.setdefault(
            speaker,
            {
                name: {
                    metric: []
                    for metric in ("aperture", "mouth_width", "absolute_aperture_velocity_per_s", "yaw_proxy", "pitch_proxy")
                }
                for name in CONDITIONS
            },
        )
        speaker_event.setdefault(speaker, {name: {"signed": [], "penalized": [], "match_fraction": []} for name in CONDITIONS})
        expected = {"bilabial": 0, "open_vowel": 0}
        for phone in row["phones"]:
            category = phone_class(phone["label"])
            if category:
                expected[category] += 1

        tracks, missing_arms = _available_row_tracks(row, CONDITIONS)
        clock_support: dict[str, Any] | None = None
        common = np.zeros(0, dtype=bool)
        maps: dict[str, np.ndarray] = {}
        if "baseline" in tracks:
            common, maps, clock_support = _common_clock(tracks)
        all_arms_present = not missing_arms and set(tracks) == set(CONDITIONS)
        support = {
            "recipient_clock": "full baseline timestamps; no fixed-length crop",
            "clock_frames": clock_support["clock_frames"] if clock_support else None,
            "common_scope": "all_8_arms" if all_arms_present else "available_arms_only",
            "common_valid_frames": int(common.sum()) if common.size else 0,
            "common_coverage": float(common.mean()) if common.size else 0.0,
            "all_8_arm_common_valid_frames": int(common.sum()) if all_arms_present else None,
            "all_8_arm_common_coverage": float(common.mean()) if all_arms_present and common.size else None,
            "available_arms": [name for name in CONDITIONS if name in tracks],
            "valid_frames_by_arm_on_recipient_clock": (
                clock_support["valid_frames_by_arm_on_clock"] if clock_support else {}
            ),
        }
        if not all_arms_present:
            speaker_clips[speaker].append({"id": row["id"], "endpoint_complete": False, "conditions": {}})
            clip_rows.append({
                "id": row["id"], "speaker": speaker, "observed": False,
                "endpoint_complete": False, "missing_arms": missing_arms,
                "phone_expected_count": expected,
                "phone_observed_count_by_arm": {condition: None for condition in CONDITIONS},
                "support": support,
            })
            continue

        base_times = tracks["baseline"].timestamps_s
        common_indices = np.flatnonzero(common)
        global_by_arm = {
            name: _global_track_metrics(track, maps[name], common, base_times)
            for name, track in tracks.items()
        }
        token_values: dict[str, dict[str, list[float]]] = {
            name: {"bilabial": [], "open_vowel": []} for name in CONDITIONS
        }
        observed_counts: dict[str, dict[str, int]] = {}
        tokens_by_arm: dict[str, list[dict[str, Any]]] = {}
        phone_by_arm: dict[str, dict[str, float | None]] = {}
        for condition in CONDITIONS:
            matched_track = tracks[condition]
            frame_indices = maps[condition][common_indices]
            token_rows, _expected_here = _phone_apertures(row["phones"], matched_track, frame_indices, latency_s)
            tokens_by_arm[condition] = token_rows
            for token in token_rows:
                if token["observed"]:
                    token_values[condition][token["category"]].append(float(token["aperture_mean"]))
            observed_counts[condition] = {
                category: len(token_values[condition][category]) for category in expected
            }
            class_means: dict[str, float | None] = {}
            for category in ("bilabial", "open_vowel"):
                values = token_values[condition][category]
                class_means[category] = (
                    float(np.mean(values))
                    if len(values) >= minimum_counts[category]
                    else None
                )
            labial = class_means["bilabial"]
            vowel = class_means["open_vowel"]
            phone_by_arm[condition] = {
                "open_vowel_aperture": vowel,
                "bilabial_aperture": labial,
                "phone_separation": float(vowel - labial) if vowel is not None and labial is not None else None,
            }

        global_store = speaker_global[speaker]
        for condition in CONDITIONS:
            for metric in ("aperture", "mouth_width", "absolute_aperture_velocity_per_s", "yaw_proxy", "pitch_proxy"):
                global_store[condition].setdefault(metric, []).append(global_by_arm[condition][metric])

        event_store = speaker_event[speaker]
        tolerance_s = EVENT_TOLERANCE_FRAMES / FPS_REFERENCE
        for condition in CONDITIONS:
            event_result = _event_result(
                tracks[condition], row["phones"],
                min_prominence=float(calibration.get("protocol", {}).get("min_prominence", 1e-8)),
                latency_s=latency_s,
                tolerance_s=tolerance_s,
            )
            for kind in ("minimum", "maximum"):
                current = event_result[kind]
                if current["mean_signed_error_s"] is not None:
                    event_store[condition]["signed"].append(float(current["mean_signed_error_s"]))
                if current["penalized_abs_error_s"] is not None:
                    event_store[condition]["penalized"].append(float(current["penalized_abs_error_s"]))
                if current["match_fraction"] is not None:
                    event_store[condition]["match_fraction"].append(float(current["match_fraction"]))

        random_separations = [phone_by_arm[name]["phone_separation"] for name in ("random1", "random2", "random3")]
        random_mean = (
            float(np.mean([float(value) for value in random_separations]))
            if all(value is not None for value in random_separations) else None
        )
        clip_endpoint_complete = bool(
            common.any()
            and all(
                phone_by_arm[name]["phone_separation"] is not None
                for name in CONDITIONS
            )
        )
        clip_conditions = {condition: phone_by_arm[condition] for condition in CONDITIONS}
        clip_conditions["random_mean"] = {"phone_separation": random_mean}
        speaker_clips[speaker].append({
            "id": row["id"],
            "endpoint_complete": clip_endpoint_complete,
            "conditions": clip_conditions,
        })
        support.update({
            "common_valid_frames": int(common.sum()),
            "common_coverage": float(common.mean()) if common.size else 0.0,
            "all_8_arm_common_valid_frames": int(common.sum()),
            "all_8_arm_common_coverage": float(common.mean()) if common.size else 0.0,
        })
        clip_rows.append({
            "id": row["id"], "speaker": speaker, "observed": bool(common.any()),
            "endpoint_complete": clip_endpoint_complete,
            "missing_arms": {}, "support": support,
            "phone_expected_count": expected,
            "phone_observed_count_by_arm": observed_counts,
            "phone_tokens_by_arm": tokens_by_arm,
            "phone_class_means_and_separation_by_arm": phone_by_arm,
            "random_phone_separation_mean": random_mean,
            "global_by_arm": global_by_arm,
        })

    speaker_values: dict[str, dict[str, Any]] = {}
    for speaker in sorted(speaker_clips):
        clips = speaker_clips[speaker]
        speaker_supported = len(clips) == 2 and all(item["endpoint_complete"] for item in clips)
        result: dict[str, Any] = {condition: {} for condition in CONDITIONS}
        for condition in CONDITIONS:
            class_means = {
                category: (
                    float(np.mean([clip["conditions"][condition][key] for clip in clips]))
                    if speaker_supported else None
                )
                for category, key in (("open_vowel", "open_vowel_aperture"), ("bilabial", "bilabial_aperture"))
            }
            open_value, labial_value = class_means["open_vowel"], class_means["bilabial"]
            separations = [clip["conditions"][condition]["phone_separation"] for clip in clips] if speaker_supported else []
            result[condition] = {
                "open_vowel_aperture": open_value,
                "bilabial_aperture": labial_value,
                "phone_separation": float(np.mean(separations)) if speaker_supported else None,
            }
            for metric in ("aperture", "mouth_width", "absolute_aperture_velocity_per_s", "yaw_proxy", "pitch_proxy"):
                samples = [
                    item["mean"]
                    for item in speaker_global[speaker][condition][metric]
                    if item["mean"] is not None
                ]
                result[condition][f"global_{metric}_mean_across_clips"] = float(np.mean(samples)) if samples else None
                result[condition][f"global_{metric}_sd_across_clips"] = float(np.std(samples, ddof=0)) if samples else None
        random_clip_values = [clip["conditions"].get("random_mean", {}).get("phone_separation") for clip in clips]
        result["random_mean"] = {
            "phone_separation": float(np.mean(random_clip_values))
            if speaker_supported and all(value is not None for value in random_clip_values) else None,
        }
        expected = {
            category: sum(
                1 for row in rows if row["speaker"] == speaker
                for phone in row["phones"] if phone_class(phone["label"]) == category
            )
            for category in ("bilabial", "open_vowel")
        }
        observed_counts = {
            condition: {
                category: sum(
                    (clip_row["phone_observed_count_by_arm"].get(condition) or {}).get(category, 0) or 0
                    for clip_row in clip_rows if clip_row["speaker"] == speaker
                )
                for category in ("bilabial", "open_vowel")
            }
            for condition in CONDITIONS
        }
        result["support"] = {
            "expected_phone_tokens_by_class": expected,
            "observed_phone_tokens_by_arm": observed_counts,
            "minimum_tokens_per_clip_from_calibration": minimum_counts,
            "clips_with_complete_eight_arm_phone_support": sum(bool(item["endpoint_complete"]) for item in clips),
            "clips_expected": len(clips),
            "speaker_endpoint_supported": speaker_supported,
            "clip_weighting": "phone-class means and separation are computed per clip, then the two clips receive equal weight",
        }
        result["event_exploratory"] = {
            condition: {
                key: float(np.mean(values)) if values else None
                for key, values in speaker_event[speaker][condition].items()
            }
            for condition in CONDITIONS
        }
        speaker_values[speaker] = result
    return speaker_values, clip_rows


def _bootstrap_contrast(
    speaker_values: Mapping[str, Mapping[str, Any]],
    metric: str,
    left: str,
    right: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    names: list[str] = []
    values: list[float] = []
    for speaker, result in sorted(speaker_values.items()):
        a = result.get(left, {}).get(metric)
        b = result.get(right, {}).get(metric)
        if a is not None and b is not None:
            names.append(str(speaker))
            values.append(float(a) - float(b))
    if not values:
        return {
            "contrast": f"{left} - {right}", "metric": metric,
            "mean": None, "ci99": [None, None], "observed_speakers": 0,
            "expected_speakers": len(speaker_values), "missing_speakers": list(sorted(speaker_values)),
            "speaker_values": {},
        }
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(data), size=(draws, len(data)), endpoint=False)
    bootstrap_means = data[indices].mean(axis=1)
    all_names = set(speaker_values)
    return {
        "contrast": f"{left} - {right}",
        "metric": metric,
        "mean": float(data.mean()),
        "ci99": [float(np.quantile(bootstrap_means, 0.005, method="linear")),
                 float(np.quantile(bootstrap_means, 0.995, method="linear"))],
        "observed_speakers": len(names),
        "expected_speakers": len(speaker_values),
        "missing_speakers": sorted(all_names.difference(names)),
        "speaker_values": {name: float(value) for name, value in zip(names, values, strict=True)},
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "unit": "speaker; utterance/token/frame are nested",
    }


def analyze_manifest(manifest_path: str | Path, calibration: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    rows = _manifest_rows(payload)
    calibration_protocol = calibration.get("protocol", {})
    if calibration.get("protocol_id") != PROTOCOL_ID:
        raise VisualMechanismError("calibration protocol_id does not match this eight-arm visual analysis")
    if calibration_protocol.get("conditions") != list(CONDITIONS):
        raise VisualMechanismError("calibration condition list does not match the frozen eight-arm set")
    inventory = calibration_protocol.get("phone_inventory")
    inventory_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if calibration_protocol.get("phone_inventory_sha256") != inventory_hash:
        raise VisualMechanismError("calibration phone-inventory hash is missing or invalid")
    if any("cal" in str(row.get("split", "")).lower() for row in rows):
        raise VisualMechanismError("evaluation manifest contains calibration rows")
    speakers = {row["speaker"] for row in rows}
    if len(speakers) < 2:
        raise VisualMechanismError("evaluation needs at least two speakers")
    clip_counts = {speaker: sum(row["speaker"] == speaker for row in rows) for speaker in speakers}
    if any(count != 2 for count in clip_counts.values()):
        raise VisualMechanismError("evaluation requires exactly two manifest clips per speaker, including failed arms")
    speaker_values, clip_rows = _speaker_primary_values(rows, calibration)
    contrasts = {
        "orth_plus_minus_baseline": _bootstrap_contrast(speaker_values, "phone_separation", "orth_plus", "baseline"),
        "orth_plus_minus_random_mean": _bootstrap_contrast(speaker_values, "phone_separation", "orth_plus", "random_mean"),
        "orth_minus_minus_baseline": _bootstrap_contrast(speaker_values, "phone_separation", "orth_minus", "baseline"),
        "full_plus_minus_baseline": _bootstrap_contrast(speaker_values, "phone_separation", "full_plus", "baseline"),
        "full_minus_minus_baseline": _bootstrap_contrast(speaker_values, "phone_separation", "full_minus", "baseline"),
    }
    primary_eligible = bool(calibration.get("positive_phone_endpoint_eligible", False))
    primary_complete = all(
        contrasts[name]["observed_speakers"] == contrasts[name]["expected_speakers"]
        and contrasts[name]["expected_speakers"] > 0
        for name in ("orth_plus_minus_baseline", "orth_plus_minus_random_mean")
    )
    primary_positive = bool(
        primary_eligible and primary_complete
        and contrasts["orth_plus_minus_baseline"]["ci99"][0] is not None
        and contrasts["orth_plus_minus_random_mean"]["ci99"][0] is not None
        and contrasts["orth_plus_minus_baseline"]["ci99"][0] > 0.0
        and contrasts["orth_plus_minus_random_mean"]["ci99"][0] > 0.0
    )
    if not primary_eligible:
        primary_status = "DESCRIPTIVE_ONLY_CALIBRATION_GATE_FAILED"
    elif not primary_complete:
        primary_status = "INCOMPLETE_PAIRED_SPEAKER_SUPPORT"
    elif primary_positive:
        primary_status = "INDEPENDENT_PHONE_SEPARATION_CONFIRMED"
    else:
        primary_status = "PHONE_SEPARATION_NOT_CONFIRMED"
    event_calibrated = bool(calibration.get("event_metric_calibrated", False))
    clock_frames_expected = sum(
        int(item["support"]["clock_frames"])
        for item in clip_rows if item["support"]["clock_frames"] is not None
    )
    coverage_by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        arm_clips = [item for item in clip_rows if condition in item["support"]["available_arms"]]
        valid_frames = sum(
            int(item["support"]["valid_frames_by_arm_on_recipient_clock"].get(condition, 0))
            for item in clip_rows
        )
        coverage_by_condition[condition] = {
            "clips_expected": len(rows),
            "clips_with_feature_track": len(arm_clips),
            "valid_frames_observed_on_known_recipient_clocks": valid_frames,
            "recipient_clock_frames_known": clock_frames_expected,
            "valid_frame_coverage_on_known_clocks": (
                float(valid_frames / clock_frames_expected) if clock_frames_expected else None
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": "evaluation_analysis",
        "status": "COMPLETE",
        "created_from_manifest": str(Path(manifest_path).resolve()),
        "calibration_protocol_id": calibration.get("protocol_id"),
        "calibration_status": calibration.get("status"),
        "evaluation_manifest_sha256": _sha256_file(manifest_path),
        "calibration_event_gate": calibration.get("event_controls", {}).get("status"),
        "counts": {
            "clips_expected": len(rows),
            "speakers_expected": len(speakers),
            "clips_with_all_eight_arms": sum(not item["missing_arms"] for item in clip_rows),
            "clips_with_eight_arm_common_support": sum(
                item["support"]["all_8_arm_common_valid_frames"] is not None
                and item["support"]["all_8_arm_common_valid_frames"] > 0
                for item in clip_rows
            ),
            "phone_symbol_inventory": calibration.get("protocol", {}).get("phone_inventory"),
            "full_recipient_clock": True,
            "common_valid_support_across_all_eight_arms": True,
            "coverage_by_condition": coverage_by_condition,
        },
        "primary_endpoint": {
            "name": "open-vowel minus bilabial mean normalized mouth aperture",
            "unit": "per-clip class means and separation; two clips equally averaged within speaker; speakers equally weighted",
            "random_control_aggregation": "compute each random arm's clip phone separation, average three values per clip, then equally average the two clip means within speaker; never average trajectories",
            "contrast_requirements": ["orth_plus > baseline", "orth_plus > mean(random1, random2, random3)"],
            "status": primary_status,
            "positive_confirmation": primary_positive,
            "contrasts": contrasts,
            "missingness": {
                key: {"observed_speakers": val["observed_speakers"],
                      "expected_speakers": val["expected_speakers"],
                      "missing_speakers": val["missing_speakers"]}
                for key, val in contrasts.items()
            },
        },
        "descriptive_mouth_motion": {
            "metrics": ["overall aperture mean/sd/p10/p90", "absolute aperture velocity per second",
                        "mouth width", "2D nose/eye yaw proxy", "2D nose/eye pitch proxy"],
            "speaker_values_by_condition": {
                speaker: {condition: values[condition] for condition in CONDITIONS}
                for speaker, values in speaker_values.items()
            },
            "interpretation": "describes aperture level, movement magnitude and pose proxies; does not establish synchrony or perceived quality",
        },
        "event_timing_exploratory": {
            "calibrated": event_calibrated,
            "primary_endpoint": False,
            "claim_scope": "do not call a timing improvement unless calibration's fixed global detector and shift/repeat/shuffle controls pass",
            "speaker_values_by_condition": {
                speaker: values["event_exploratory"] for speaker, values in speaker_values.items()
            },
        },
        "support_by_clip": clip_rows,
        "speaker_values": speaker_values,
    }


def run_extract(args: argparse.Namespace) -> None:
    result = extract_manifest(args.manifest, args.output_manifest, args.output_root, args.asset)
    print(json.dumps({"status": result["stage"], "rows": len(result["rows"]), "output": str(args.output_manifest)}, ensure_ascii=False))


def run_calibrate(args: argparse.Namespace) -> None:
    result = calibrate_manifest(args.manifest)
    _write_json(args.output, result)
    print(json.dumps({"status": result["status"], "event_metric_calibrated": result["event_metric_calibrated"], "output": str(args.output)}, ensure_ascii=False))


def run_analyze(args: argparse.Namespace) -> None:
    calibration = _read_json(args.calibration)
    result = analyze_manifest(args.manifest, calibration)
    _write_json(args.output, result)
    print(json.dumps({"status": result["primary_endpoint"]["status"], "output": str(args.output)}, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract MediaPipe mouth features from saved videos on CPU")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output-manifest", required=True)
    extract.add_argument("--output-root", required=True)
    extract.add_argument("--asset", required=True, help="existing face_landmarker.task; no downloading")
    extract.set_defaults(func=run_extract)
    calibrate = subparsers.add_parser("calibrate", help="fit lag, detector, and phone support from CAL rows")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(func=run_calibrate)
    analyze = subparsers.add_parser("analyze", help="analyze locked evaluation rows with the frozen CAL artifact")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--calibration", required=True)
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(func=run_analyze)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
