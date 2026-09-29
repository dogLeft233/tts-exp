from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .common import ProtocolError, canonical_json_sha256, bytes_sha256


FPS = 25
SAMPLES_PER_FRAME = 640
MAX_DISPLACEMENT = 3.0
MIN_SPEED = 0.5
MAX_SPEED = 1.5
PROTECTED_FRAMES = 5
MAX_KNOTS = 16
MIN_KNOT_GAP = 12
KNOT_GRID = 0.25


def knot_positions(
    frame_count: int,
    valid_frame_count: int,
    *,
    max_knots: int = MAX_KNOTS,
    min_gap: int = MIN_KNOT_GAP,
    protected: int = PROTECTED_FRAMES,
) -> np.ndarray:
    if frame_count < 1 or not (1 <= valid_frame_count <= frame_count):
        raise ProtocolError("invalid full/valid frame counts")
    start = protected
    stop = valid_frame_count - protected - 1
    span = stop - start
    if span < min_gap:
        return np.zeros((0,), dtype=np.int64)
    count = min(int(max_knots), span // int(min_gap) + 1)
    positions = np.rint(np.linspace(start, stop, count, dtype=np.float64)).astype(np.int64)
    if np.unique(positions).size != positions.size or np.any(np.diff(positions) < min_gap):
        raise ProtocolError("could not construct an equally spaced knot grid with the required separation")
    return positions


def _validate_knot_values(values: Sequence[float], count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ProtocolError(f"expected {count} finite knot displacements")
    if np.any(np.abs(array) > MAX_DISPLACEMENT + 1e-12):
        raise ProtocolError("knot displacement exceeds 3 frames")
    grid_units = array / KNOT_GRID
    if np.any(np.abs(grid_units - np.rint(grid_units)) > 1e-10):
        raise ProtocolError("knot displacement is not on the 0.25-frame grid")
    return array


def build_map(
    frame_count: int,
    valid_frame_count: int,
    knot_deltas: Sequence[float] | None = None,
    *,
    positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    frame_count = int(frame_count)
    valid_frame_count = int(valid_frame_count)
    x = knot_positions(frame_count, valid_frame_count) if positions is None else np.asarray(positions, dtype=np.int64)
    if x.ndim != 1 or x.size > MAX_KNOTS:
        raise ProtocolError("knot positions must be a vector of at most 16 knots")
    if x.size and (x[0] < PROTECTED_FRAMES or x[-1] > valid_frame_count - PROTECTED_FRAMES - 1):
        raise ProtocolError("internal knot overlaps a protected boundary")
    if x.size > 1 and np.any(np.diff(x) < MIN_KNOT_GAP):
        raise ProtocolError("internal knots are less than 12 frames apart")
    if knot_deltas is None:
        values = np.zeros((x.size,), dtype=np.float64)
    else:
        values = _validate_knot_values(knot_deltas, x.size)

    indices = np.arange(frame_count, dtype=np.float64)
    delta = np.zeros((frame_count,), dtype=np.float64)
    if x.size:
        delta[:valid_frame_count] = np.interp(indices[:valid_frame_count], x.astype(np.float64), values)
    delta[:PROTECTED_FRAMES] = 0.0
    delta[max(PROTECTED_FRAMES, valid_frame_count - PROTECTED_FRAMES):] = 0.0
    delta[valid_frame_count:] = 0.0
    q = indices + delta
    result: dict[str, Any] = {
        "schema_version": 1,
        "frame_count": frame_count,
        "valid_frame_count": valid_frame_count,
        "knot_positions": [int(value) for value in x],
        "knot_deltas": [float(value) for value in values],
        "delta": [float(value) for value in delta],
        "q": [float(value) for value in q],
    }
    result["map_sha256"] = canonical_json_sha256({key: result[key] for key in ("frame_count", "valid_frame_count", "knot_positions", "knot_deltas", "delta", "q")})
    return result


def validate_map(mapping: Mapping[str, Any], *, frame_count: int | None = None, valid_frame_count: int | None = None) -> dict[str, Any]:
    total = int(mapping.get("frame_count", -1) if frame_count is None else frame_count)
    valid = int(mapping.get("valid_frame_count", -1) if valid_frame_count is None else valid_frame_count)
    positions = np.asarray(mapping.get("knot_positions", []), dtype=np.int64)
    values = np.asarray(mapping.get("knot_deltas", []), dtype=np.float64)
    rebuilt = build_map(total, valid, values, positions=positions)
    q = np.asarray(rebuilt["q"], dtype=np.float64)
    delta = np.asarray(rebuilt["delta"], dtype=np.float64)
    stored_q = np.asarray(mapping.get("q", []), dtype=np.float64)
    stored_delta = np.asarray(mapping.get("delta", []), dtype=np.float64)
    if stored_q.shape != q.shape or not np.array_equal(stored_q, q):
        raise ProtocolError("stored q does not match reconstruction from saved knots")
    if stored_delta.shape != delta.shape or not np.array_equal(stored_delta, delta):
        raise ProtocolError("stored displacement does not match saved knots")
    if np.any(np.abs(delta) > MAX_DISPLACEMENT + 1e-12):
        raise ProtocolError("map exceeds the 3-frame displacement bound")
    if np.any(delta[:PROTECTED_FRAMES] != 0.0) or np.any(delta[max(PROTECTED_FRAMES, valid - PROTECTED_FRAMES):] != 0.0):
        raise ProtocolError("map moves protected first/last/padded frames")
    if q.size > 1:
        speed = np.diff(q)
        if not np.isfinite(speed).all() or np.any(speed < MIN_SPEED - 1e-12) or np.any(speed > MAX_SPEED + 1e-12):
            raise ProtocolError("map violates monotone 0.5..1.5 frame speed")
        if np.any(np.abs(np.diff(speed)) > 0.5 + 1e-12):
            raise ProtocolError("map has an adjacent slope change above 0.5")
    if q[0] != 0.0 or q[-1] != total - 1 or np.any(q < 0.0) or np.any(q > total - 1):
        raise ProtocolError("map changed endpoints or reads outside the source video")
    return {
        "map_sha256": rebuilt["map_sha256"],
        "max_abs_displacement": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "max_speed_deviation": float(np.max(np.abs(np.diff(q) - 1.0))) if q.size > 1 else 0.0,
        "max_slope_change": float(np.max(np.abs(np.diff(np.diff(q))))) if q.size > 2 else 0.0,
        "regularization": regularization(delta),
        "blended_fraction": float(np.mean(np.abs(q - np.rint(q)) > 1e-12)) if q.size else 0.0,
    }


def regularization(delta: Sequence[float]) -> float:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ProtocolError("regularization requires finite one-dimensional displacements")
    displacement = float(np.mean(np.square(values / MAX_DISPLACEMENT))) if values.size else 0.0
    temporal = float(np.mean(np.square(np.diff(values) / 0.5))) if values.size > 1 else 0.0
    return displacement + temporal


def global_seed(
    frame_count: int,
    valid_frame_count: int,
    amount: int,
) -> dict[str, Any]:
    if int(amount) not in (-3, -2, -1, 1, 2, 3):
        raise ProtocolError("global seed amount must be one of -3,-2,-1,1,2,3")
    positions = knot_positions(frame_count, valid_frame_count)
    if positions.size == 0:
        return build_map(frame_count, valid_frame_count, [], positions=[])
    frame_index = np.arange(frame_count, dtype=np.float64)
    ramp_in = np.clip((frame_index - PROTECTED_FRAMES) / 10.0, 0.0, 1.0)
    ramp_out = np.clip((valid_frame_count - PROTECTED_FRAMES - 1 - frame_index) / 10.0, 0.0, 1.0)
    profile = float(amount) * np.minimum(ramp_in, ramp_out)
    values = np.rint(profile[positions] / KNOT_GRID) * KNOT_GRID
    return build_map(frame_count, valid_frame_count, values, positions=positions)


def render_map(source_frames: np.ndarray, mapping: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source_frames)
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ProtocolError(f"source frames must be uint8 BGR [F,H,W,3], got {source.shape} {source.dtype}")
    audit = validate_map(mapping, frame_count=source.shape[0])
    q = np.asarray(mapping["q"], dtype=np.float64)
    low = np.floor(q).astype(np.int64)
    high = np.ceil(q).astype(np.int64)
    weight = q - low.astype(np.float64)
    blended = (
        (1.0 - weight[:, None, None, None]) * source[low].astype(np.float64)
        + weight[:, None, None, None] * source[high].astype(np.float64)
    )
    if not np.isfinite(blended).all():
        raise ProtocolError("pixel interpolation produced non-finite values")
    output = np.floor(blended + 0.5).astype(np.uint8)
    audit.update({
        "source_frame_sha256": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in source],
        "output_frame_sha256": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in output],
        "pixel_arithmetic": "float64 adjacent-pixel blend; floor(value + 0.5) to uint8",
        "frame_count": int(output.shape[0]),
    })
    return output, audit


def render_nearest(source_frames: np.ndarray, mapping: Mapping[str, Any]) -> np.ndarray:
    source = np.asarray(source_frames)
    if source.ndim != 4 or source.dtype != np.uint8 or source.shape[-1] != 3:
        raise ProtocolError("nearest-neighbor source must be uint8 BGR frames")
    validate_map(mapping, frame_count=source.shape[0])
    q = np.asarray(mapping["q"], dtype=np.float64)
    indices = np.floor(q + 0.5).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= source.shape[0]):
        raise ProtocolError("nearest-neighbor source frame index is out of bounds")
    return source[indices].copy()


def mirror_map(mapping: Mapping[str, Any]) -> dict[str, Any]:
    values = -np.asarray(mapping.get("knot_deltas", []), dtype=np.float64)
    mirrored = build_map(
        int(mapping["frame_count"]),
        int(mapping["valid_frame_count"]),
        values,
        positions=mapping.get("knot_positions", []),
    )
    validate_map(mirrored)
    return mirrored


def frames_sha256(frames: np.ndarray) -> str:
    values = np.asarray(frames)
    if values.dtype != np.uint8:
        raise ProtocolError("frame hash expects uint8 pixels")
    return bytes_sha256(np.ascontiguousarray(values).tobytes())
