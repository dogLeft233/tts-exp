"""Canonical mouth-trajectory transfer metric for C.

The feature extractor itself remains B's pinned MediaPipe implementation.  C
owns only the fixed-clock comparison and can consume B's canonical-mouth NPZs
or arrays supplied by synthetic tests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    DELAY_FRAMES,
    MIN_VISUAL_ROWS,
    U_ROWS,
    TimingError,
    file_sha256,
)


def _coerce_trajectory(value: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite ``[frames, features]`` array and a valid-frame mask."""

    supplied_valid: Any = None
    if hasattr(value, "canonical_mouth"):
        supplied_valid = getattr(value, "valid", None)
        value = value.canonical_mouth
    elif isinstance(value, Mapping):
        supplied_valid = value.get("valid")
        for key in ("trajectory", "canonical_mouth", "mouth_features", "features", "mouth"):
            if key in value:
                value = value[key]
                break
        else:
            raise TimingError(f"{name} trajectory is missing")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim >= 2:
        array = array.reshape(array.shape[0], -1)
    else:
        raise TimingError(f"{name} trajectory has invalid shape: {array.shape}")
    if array.ndim != 2 or array.shape[0] <= max(U_ROWS) + DELAY_FRAMES or array.shape[1] <= 0:
        raise TimingError(f"{name} trajectory lacks fixed-clock support: {array.shape}")
    finite = np.isfinite(array).all(axis=1)
    if supplied_valid is None:
        valid = finite
    else:
        valid_array = np.asarray(supplied_valid)
        if valid_array.shape != (array.shape[0],) or valid_array.dtype != np.bool_:
            raise TimingError(f"{name} valid mask is malformed")
        valid = finite & valid_array
    output = np.asarray(array, dtype=np.float64).copy()
    output[~valid] = 0.0
    return output, valid


def load_mouth_trajectory(path: str | Path) -> dict[str, Any]:
    """Load a B feature artifact without importing its producer runner."""

    target = Path(path).resolve()
    if not target.is_file():
        raise TimingError(f"mouth feature artifact is missing: {target}")
    try:
        if target.suffix.lower() == ".npz":
            with np.load(target, allow_pickle=False) as data:
                keys = set(data.files)
                key = next((candidate for candidate in ("canonical_mouth", "trajectory", "features", "mouth") if candidate in keys), None)
                if key is None:
                    raise TimingError(f"mouth feature array is missing: {target}")
                trajectory = np.asarray(data[key])
                valid = np.asarray(data["valid"]) if "valid" in keys else None
                metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in keys else {}
        else:
            trajectory = np.load(target, allow_pickle=False)
            valid = None
            metadata = {}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise TimingError(f"mouth feature artifact is malformed: {target}") from exc
    result: dict[str, Any] = {"trajectory": trajectory, "metadata": metadata}
    if valid is not None:
        result["valid"] = valid
    output, output_valid = _coerce_trajectory(result, name=str(target))
    return {
        "trajectory": output,
        "valid": output_valid,
        "metadata": metadata,
        "path": str(target),
        "sha256": file_sha256(target),
    }


def save_mouth_trajectory(
    path: str | Path,
    trajectory: Any,
    *,
    valid: Sequence[bool] | np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(path).resolve()
    array, inferred_valid = _coerce_trajectory(
        {"trajectory": trajectory, "valid": np.asarray(valid, dtype=np.bool_) if valid is not None else None},
        name="mouth trajectory",
    )
    # _coerce_trajectory treats an explicit None as malformed.  The branch
    # uses this small correction to retain a convenient optional valid mask.
    if valid is None:
        inferred_valid = np.isfinite(np.asarray(trajectory, dtype=np.float64).reshape(np.asarray(trajectory).shape[0], -1)).all(axis=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("schema_version", 1)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            trajectory=array,
            canonical_mouth=array,
            valid=np.asarray(inferred_valid, dtype=np.bool_),
            metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True), dtype=np.str_),
        )
    temporary.replace(target)
    loaded = load_mouth_trajectory(target)
    if not np.array_equal(loaded["trajectory"], array) or not np.array_equal(loaded["valid"], inferred_valid):
        raise TimingError(f"mouth feature readback differs: {target}")
    return {"path": str(target), "sha256": file_sha256(target), "frames": int(array.shape[0]), "features": int(array.shape[1]), "valid_count": int(np.sum(inferred_valid))}


def center_sequence(sequence: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    selected = np.asarray(tuple(int(row) for row in rows), dtype=np.int64)
    if selected.size == 0 or selected.min() < 0 or selected.max() >= values.shape[0]:
        raise TimingError("visual center support is outside trajectory")
    centered = values[selected] - np.mean(values[selected], axis=0, keepdims=True, dtype=np.float64)
    return centered


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise TimingError(f"visual trajectory comparison shape differs: {left.shape}/{right.shape}")
    return float(np.mean(np.square(left - right), dtype=np.float64))


def visual_transfer_score(
    natural: Any,
    delayed: Any,
    *,
    rows: Sequence[int] = U_ROWS,
    compensation_frames: int = DELAY_FRAMES,
    min_rows: int = MIN_VISUAL_ROWS,
    strict: bool = False,
) -> dict[str, Any]:
    """Measure whether DELAY's mouth trajectory moves by exactly +5 frames."""

    if int(compensation_frames) != DELAY_FRAMES:
        raise TimingError("visual compensation is fixed at +5 frames")
    natural_values, natural_valid = _coerce_trajectory(natural, name="natural")
    delayed_values, delayed_valid = _coerce_trajectory(delayed, name="delay")
    if natural_values.shape[1] != delayed_values.shape[1]:
        raise TimingError("natural and DELAY trajectory dimensions differ")
    requested = np.asarray(tuple(int(row) for row in rows), dtype=np.int64)
    if requested.size == 0 or requested.min() < 0 or requested.max() + compensation_frames >= min(natural_values.shape[0], delayed_values.shape[0]):
        raise TimingError("visual support is outside the fixed clock")
    support = requested[
        natural_valid[requested]
        & natural_valid[requested + compensation_frames]
        & delayed_valid[requested]
        & delayed_valid[requested + compensation_frames]
    ]
    missing_rows = sorted({int(row) for row in requested} - {int(row) for row in support})
    if support.size < int(min_rows):
        result = {
            "status": "VISUAL_RESPONSE_UNRESOLVED",
            "valid": False,
            "rows": [int(row) for row in support],
            "requested_rows": [int(row) for row in requested],
            "missing_rows": missing_rows,
            "row_count": int(support.size),
            "min_rows": int(min_rows),
            "e_unc": None,
            "e_comp": None,
            "v": None,
            "compensation_frames": int(compensation_frames),
        }
        if strict:
            raise TimingError(f"visual response has only {support.size} valid rows")
        return result
    # Each compared sequence is centered on the same support independently;
    # no sequence is allowed to benefit from a different denominator.
    natural_unc = center_sequence(natural_values, support)
    delayed_unc = center_sequence(delayed_values, support)
    delayed_comp = center_sequence(delayed_values, support + compensation_frames)
    e_unc = _mse(natural_unc, delayed_unc)
    e_comp = _mse(natural_unc, delayed_comp)
    denominator = e_unc + e_comp + 1e-12
    v = 0.0 if e_unc == 0.0 and e_comp == 0.0 else float((e_unc - e_comp) / denominator)
    return {
        "status": "complete",
        "valid": True,
        "rows": [int(row) for row in support],
        "requested_rows": [int(row) for row in requested],
        "missing_rows": missing_rows,
        "row_count": int(support.size),
        "min_rows": int(min_rows),
        "e_unc": e_unc,
        "e_comp": e_comp,
        "v": v,
        "compensation_frames": int(compensation_frames),
    }


def compare_mouth_trajectories(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return visual_transfer_score(*args, **kwargs)


def mouth_transfer_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return visual_transfer_score(*args, **kwargs)


def synthesize_delayed_trajectory(
    trajectory: Any,
    *,
    shift_frames: int = DELAY_FRAMES,
) -> dict[str, Any]:
    """Build the calibration-only ``R_DELAY[t]=R[t-5]`` trajectory."""

    if int(shift_frames) != DELAY_FRAMES:
        raise TimingError("calibration shift is fixed at +5 frames")
    values, valid = _coerce_trajectory(trajectory, name="reference")
    shifted = np.zeros_like(values)
    shifted_valid = np.zeros(values.shape[0], dtype=np.bool_)
    shifted[shift_frames:] = values[:-shift_frames]
    shifted_valid[shift_frames:] = valid[:-shift_frames]
    return {"trajectory": shifted, "valid": shifted_valid, "shift_frames": int(shift_frames)}


def calibrate_visual_shift(
    trajectories: Mapping[str, Any] | Sequence[Any],
    *,
    shift_frames: int = DELAY_FRAMES,
    rows: Sequence[int] = U_ROWS,
    min_rows: int = MIN_VISUAL_ROWS,
) -> dict[str, Any]:
    """Calibrate the metric on synthetic +5-frame delayed trajectories."""

    if isinstance(trajectories, Mapping):
        items = [(str(group), value) for group, value in trajectories.items()]
    else:
        items = [(str(index), value) for index, value in enumerate(trajectories)]
    records: list[dict[str, Any]] = []
    for group, reference in items:
        shifted = synthesize_delayed_trajectory(reference, shift_frames=shift_frames)
        score = visual_transfer_score(reference, shifted, rows=rows, compensation_frames=shift_frames, min_rows=min_rows)
        records.append({"source_group": group, **score})
    valid_records = [record for record in records if record.get("valid")]
    passed = [
        record
        for record in valid_records
        if float(record["e_comp"]) <= 1e-12 and float(record["e_unc"]) > 1e-6
    ]
    return {
        "status": "complete",
        "records": records,
        "group_count": len(records),
        "observed_count": len(valid_records),
        "passed_count": len(passed),
        "required_groups": 12,
        "required_pass_count": 10,
        "shift_frames": int(shift_frames),
        "pass": bool(len(records) >= 12 and len(valid_records) == len(records) and len(passed) >= 10),
        "rule": "e_comp<=1e-12 and e_unc>1e-6 for at least 10/12 groups",
    }


def calibrate_shift(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return calibrate_visual_shift(*args, **kwargs)


def extract_mouth_trajectory(video_path: str | Path, landmarker_asset: str | Path) -> dict[str, Any]:
    """Use B's pinned extractor lazily, keeping C import-safe on CPU hosts."""

    try:
        from scripts.experiments.lrs3_tts_visual_advantage.video_features import (
            extract_video_features,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise TimingError("B visual feature extractor is unavailable") from exc
    sequence = extract_video_features(video_path, landmarker_asset)
    return {
        "trajectory": np.asarray(sequence.canonical_mouth),
        "valid": np.asarray(sequence.valid, dtype=np.bool_),
        "metadata": dict(sequence.metadata),
    }
