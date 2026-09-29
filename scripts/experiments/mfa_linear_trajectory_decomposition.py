"""Decompose MFA-linear trajectory gains using a frozen score run.

This module deliberately contains only CPU-side reanalysis.  It reads the
31-point SyncNet distance curves produced by the support30 ablation and never
imports the old scoring script (which would pull in torch and model code).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SPEC_ID = "mfa_linear_trajectory_decomposition_v1"
PROTOCOL_ID = "mfa_linear_trajectory_decomposition_v1"
SOURCE_PROTOCOL = "mfa_linear_trajectory_ablation_v2_support30"
SOURCE_EXPECTED_HASHES = {
    "inputs.json": "43af6d39d9bd4b59d74b63555e900a492a8cf7c695ed461c286a42b9c902404f",
    "scores.csv": "98dabb93faa2b1ee6e89eccbf3bed64ed7ab8f0e867b9f5815fd513234da9b03",
    "curves.json": "5483bf2047c76e676bb9f223bd9f06da1b114f38f3530c0d574776780bfa1c80",
    "scores_manifest.json": "cb6238fc6d5701498b065e3cc15038614e12b8d6816d1c7c34bf0dc114188b5a",
    "reuse_manifest.json": "b13e550cc4d29be33b881118df5834582f0f81d22b226a09a0d5be4c160d83b9",
}

LAGS = tuple(range(-15, 16))
VSHIFT = 15
FRAME_MS = 40.0
ARMS = ("N_RAW", "T_RAW", "N_100", "N_050", "N_000", "T_100", "T_050", "T_000")
R6 = ("N_100", "N_050", "N_000", "T_100", "T_050", "T_000")
REFERENCE_CELL = ("T_RAW", "T_RAW")
PRIMARY_CONTRASTS = ("N_dose", "T_dose", "source_interaction")
PRIMARY_METRICS = ("delta_C", "match_gain", "background_gain", "dominance")
DESCRIPTIVE_METRICS = (
    "C",
    "D",
    "B",
    "D_anchor",
    "C_anchor",
    "search_bonus",
    "O",
    "relative_margin",
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260916
BONFERRONI_TESTS = 12
BONFERRONI_ALPHA = 0.05 / BONFERRONI_TESTS
SUPPORT_THRESHOLDS = (30, 35, 40, 45, 50)
EXPECTED_SUPPORT_N = {30: 15, 35: 13, 40: 12, 45: 10, 50: 8}
ABS_TOL = 1e-6


class ProtocolError(RuntimeError):
    """A frozen input, identity, or calculation contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise ProtocolError(f"missing frozen input: {path}") from exc
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays while rejecting accidental non-finite data."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("attempted to serialize non-finite JSON value")
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _float(value: Any, field: str) -> float:
    if value is None or value == "":
        raise ProtocolError(f"missing numeric field {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid numeric field {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"non-finite numeric field {field}")
    return result


def _int(value: Any, field: str) -> int:
    if value is None or value == "":
        raise ProtocolError(f"missing integer field {field}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid integer field {field}: {value!r}") from exc
    return result


def _optional_float(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    return _float(value, field)


def _optional_int(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    return _int(value, field)


def expected_cells(records: Sequence[Mapping[str, Any]]) -> set[tuple[int, str, str, str]]:
    """Return the exact 15-cell-per-record matrix frozen by the spec."""

    result: set[tuple[int, str, str, str]] = set()
    for record in records:
        sample_id = _int(record["sample_id"], "sample_id")
        paired_key = str(record["paired_key"])
        for arm in ARMS:
            result.add((sample_id, paired_key, arm, arm))
        for arm in R6:
            result.add((sample_id, paired_key, arm, "N_RAW"))
        result.add((sample_id, paired_key, "N_RAW", "T_100"))
    return result


def _prepare_output(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ProtocolError("output directory must be disjoint from source run and its parents")
    if output.exists():
        if not output.is_dir():
            raise ProtocolError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise ProtocolError(f"output directory must be new or empty: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON input must be an object: {path}")
    return payload


def _validate_inputs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "complete":
        raise ProtocolError(f"inputs.json status is not complete: {payload.get('status')!r}")
    if payload.get("protocol") != SOURCE_PROTOCOL:
        raise ProtocolError(f"unexpected source protocol: {payload.get('protocol')!r}")
    if _int(payload.get("min_common_windows"), "inputs.min_common_windows") != 30:
        raise ProtocolError("source min_common_windows must be 30")
    records_raw = payload.get("records")
    if not isinstance(records_raw, list) or len(records_raw) != 15:
        raise ProtocolError("inputs.json must contain exactly 15 records")
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_keys: set[str] = set()
    for raw in records_raw:
        if not isinstance(raw, dict):
            raise ProtocolError("each inputs record must be an object")
        sample_id = _int(raw.get("sample_id"), "record.sample_id")
        paired_key = str(raw.get("paired_key", ""))
        speaker_id = str(raw.get("speaker_id", ""))
        if not paired_key:
            raise ProtocolError(f"sample {sample_id} has an empty paired_key")
        if sample_id in seen_ids or paired_key in seen_keys:
            raise ProtocolError("sample_id and paired_key must each be unique")
        if speaker_id != "S0765":
            raise ProtocolError(f"unexpected speaker for sample {sample_id}: {speaker_id!r}")
        seen_ids.add(sample_id)
        seen_keys.add(paired_key)
        records.append({"sample_id": sample_id, "paired_key": paired_key, "speaker_id": speaker_id})
    if seen_ids != set(range(1, 16)):
        raise ProtocolError(f"sample IDs must be 1..15, got {sorted(seen_ids)}")
    return records


def _verify_source_hashes(source: Path) -> dict[str, str]:
    actual = {name: sha256_file(source / name) for name in SOURCE_EXPECTED_HASHES}
    mismatches = {
        name: {"expected": SOURCE_EXPECTED_HASHES[name], "actual": actual[name]}
        for name in SOURCE_EXPECTED_HASHES
        if actual[name] != SOURCE_EXPECTED_HASHES[name]
    }
    if mismatches:
        raise ProtocolError("frozen source hash mismatch: " + json.dumps(mismatches, ensure_ascii=False, sort_keys=True))
    return actual


def _validate_manifest(manifest: Mapping[str, Any], name: str, records: Sequence[Mapping[str, Any]]) -> None:
    if manifest.get("status") != "complete":
        raise ProtocolError(f"{name} status must be complete")
    if name == "scores_manifest.json":
        if manifest.get("failures") != []:
            raise ProtocolError("scores_manifest failures must be []")
        for field, expected in (("sample_count", 15), ("expected_cells", 225), ("score_rows", 225), ("curve_rows", 225)):
            if _int(manifest.get(field), f"{name}.{field}") != expected:
                raise ProtocolError(f"{name}.{field} must be {expected}")
        protocol = manifest.get("scoring_protocol", {})
        if not isinstance(protocol, Mapping):
            raise ProtocolError("scores_manifest.scoring_protocol must be an object")
        if _int(protocol.get("vshift"), "scores_manifest.scoring_protocol.vshift") != 15 or _int(protocol.get("min_common_windows"), "scores_manifest.scoring_protocol.min_common_windows") != 30:
            raise ProtocolError("scores_manifest scoring protocol is not vshift=15/support30")
        if not manifest.get("syncnet_model_sha256"):
            raise ProtocolError("scores_manifest is missing SyncNet model hash")
    else:
        for field, expected in (("protocol", SOURCE_PROTOCOL), ("min_common_windows", 30), ("sample_count", 15)):
            if manifest.get(field) != expected:
                raise ProtocolError(f"{name}.{field} must be {expected!r}")


def _load_scores(path: Path, records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    required = {
        "status",
        "sample_id",
        "paired_key",
        "video_arm",
        "audio_arm",
        "C",
        "D",
        "curve_median",
        "k_star",
        "d_zero",
        "common_support_count",
        "fixed_k_n",
        "d_fixed_natural_lag",
        "score_box_sha256",
        "syncnet_model_sha256",
    }
    try:
        handle = path.open(newline="", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ProtocolError(f"missing score CSV: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ProtocolError(f"scores.csv missing fields: {missing}")
        cells: dict[tuple[int, str, str, str], dict[str, Any]] = {}
        model_hash = str(manifest.get("syncnet_model_sha256"))
        score_box_by_sample: dict[int, str] = {}
        for row_number, row in enumerate(reader, start=2):
            if row.get("status") != "complete":
                raise ProtocolError(f"scores.csv row {row_number} is not complete")
            sample_id = _int(row.get("sample_id"), f"scores row {row_number}.sample_id")
            key = (sample_id, str(row.get("paired_key", "")), str(row.get("video_arm", "")), str(row.get("audio_arm", "")))
            if key in cells:
                raise ProtocolError(f"duplicate score cell {key}")
            if key[2] not in ARMS or key[3] not in ARMS:
                raise ProtocolError(f"unknown score arm in {key}")
            values = {
                "C": _float(row.get("C"), f"scores row {row_number}.C"),
                "D": _float(row.get("D"), f"scores row {row_number}.D"),
                "B": _float(row.get("curve_median"), f"scores row {row_number}.curve_median"),
                "curve_median": _float(row.get("curve_median"), f"scores row {row_number}.curve_median"),
                "k_star": _int(row.get("k_star"), f"scores row {row_number}.k_star"),
                "d_zero": _float(row.get("d_zero"), f"scores row {row_number}.d_zero"),
                "common_support_count": _int(row.get("common_support_count"), f"scores row {row_number}.common_support_count"),
                "fixed_k_n": _optional_int(row.get("fixed_k_n"), f"scores row {row_number}.fixed_k_n"),
                "d_fixed_natural_lag": _optional_float(row.get("d_fixed_natural_lag"), f"scores row {row_number}.d_fixed_natural_lag"),
                "score_box_sha256": str(row.get("score_box_sha256", "")),
                "syncnet_model_sha256": str(row.get("syncnet_model_sha256", "")),
            }
            if values["syncnet_model_sha256"] != model_hash:
                raise ProtocolError(f"scores row {row_number} SyncNet hash disagrees with manifest")
            if values["common_support_count"] < 30:
                raise ProtocolError(f"scores row {row_number} support is below 30")
            box_hash = values["score_box_sha256"]
            if not box_hash:
                raise ProtocolError(f"scores row {row_number} has empty score_box_sha256")
            previous = score_box_by_sample.setdefault(sample_id, box_hash)
            if previous != box_hash:
                raise ProtocolError(f"sample {sample_id} has inconsistent score_box_sha256")
            values.update({"sample_id": sample_id, "paired_key": key[1], "video_arm": key[2], "audio_arm": key[3]})
            cells[key] = values
    expected = expected_cells(records)
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ProtocolError(f"scores.csv cell set mismatch; missing={missing[:3]}, extra={extra[:3]}")
    if len(cells) != 225:
        raise ProtocolError(f"scores.csv must contain 225 rows, got {len(cells)}")
    return cells


def _load_curves(path: Path, records: Sequence[Mapping[str, Any]], scores: Mapping[tuple[int, str, str, str], Mapping[str, Any]]) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    payload = _read_json(path)
    if tuple(payload.get("lags", ())) != LAGS:
        raise ProtocolError("curves.json top-level lags must be exactly -15..15")
    raw_curves = payload.get("curves")
    if not isinstance(raw_curves, list) or len(raw_curves) != 225:
        raise ProtocolError("curves.json must contain exactly 225 curves")
    curves: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for index, raw in enumerate(raw_curves):
        if not isinstance(raw, dict):
            raise ProtocolError(f"curve row {index} must be an object")
        sample_id = _int(raw.get("sample_id"), f"curve row {index}.sample_id")
        key = (sample_id, str(raw.get("paired_key", "")), str(raw.get("video_arm", "")), str(raw.get("audio_arm", "")))
        if key in curves:
            raise ProtocolError(f"duplicate curve cell {key}")
        if tuple(raw.get("lags", ())) != LAGS:
            raise ProtocolError(f"curve row {index} lags are not exactly -15..15")
        try:
            values = np.asarray(raw.get("values"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"curve row {index} values are invalid") from exc
        if values.shape != (31,) or not np.isfinite(values).all():
            raise ProtocolError(f"curve row {index} must have 31 finite values")
        support = _int(raw.get("common_support_count"), f"curve row {index}.common_support_count")
        if support < 30:
            raise ProtocolError(f"curve row {index} support is below 30")
        k_n = _optional_int(raw.get("k_n"), f"curve row {index}.k_n")
        if k_n is not None and k_n not in LAGS:
            raise ProtocolError(f"curve row {index} k_n must be one of the 31 lags when present")
        if key not in scores:
            raise ProtocolError(f"curve row {index} has no matching score cell: {key}")
        score = scores[key]
        if support != int(score["common_support_count"]):
            raise ProtocolError(f"support mismatch for cell {key}")
        metrics = _curve_metrics(LAGS, values)
        for field in ("C", "D", "curve_median", "d_zero"):
            if abs(float(metrics[field]) - float(score[field])) > ABS_TOL:
                raise ProtocolError(f"curve/score mismatch for {key} field {field}")
        if int(metrics["k_star"]) != int(score["k_star"]):
            raise ProtocolError(f"curve/score mismatch for {key} k_star")
        curves[key] = {"values": values, "lags": LAGS, "common_support_count": support, "k_n": k_n}
    expected = expected_cells(records)
    if set(curves) != expected:
        missing = sorted(expected - set(curves))
        extra = sorted(set(curves) - expected)
        raise ProtocolError(f"curves.json cell set mismatch; missing={missing[:3]}, extra={extra[:3]}")
    if len(curves) != len(scores):
        raise ProtocolError("scores and curves row counts differ")
    for record in records:
        sid = int(record["sample_id"])
        pk = str(record["paired_key"])
        supports = [
            int(row["common_support_count"])
            for (cell_sid, cell_pk, video, audio), row in curves.items()
            if cell_sid == sid and cell_pk == pk and (video, audio) != REFERENCE_CELL
        ]
        if len(supports) != 14 or len(set(supports)) != 1 or supports[0] < 30:
            raise ProtocolError(f"sample {sid} mechanism cells do not share one support >=30")
        reference_support = int(curves[(sid, pk, *REFERENCE_CELL)]["common_support_count"])
        if reference_support < 30:
            raise ProtocolError(f"sample {sid} T_RAW reference support is below 30")
    return curves


def _curve_metrics(lags: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    """Recompute C/D/B/k_star/d_zero in float64, with first-min tie breaking."""

    lag_tuple = tuple(int(lag) for lag in lags)
    if lag_tuple != LAGS:
        raise ProtocolError("curve lags must be exactly -15..15")
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (31,) or not np.isfinite(array).all():
        raise ProtocolError("SyncNet curve must contain 31 finite values")
    minimum = float(np.min(array))
    index = int(np.flatnonzero(array == minimum)[0])
    median = float(np.median(array))
    return {
        "C": float(median - minimum),
        "D": minimum,
        "B": median,
        "curve_median": median,
        "k_star": int(lag_tuple[index]),
        "d_zero": float(array[lag_tuple.index(0)]),
    }


def _lag_decomposition(values: Sequence[float], lags: Sequence[int], k_n: int) -> dict[str, Any]:
    """Compute fixed-natural-lag quantities for one curve."""

    lag_tuple = tuple(int(lag) for lag in lags)
    array = np.asarray(values, dtype=np.float64)
    if lag_tuple != LAGS or array.shape != (31,) or not np.isfinite(array).all():
        raise ProtocolError("invalid curve passed to lag decomposition")
    if int(k_n) not in lag_tuple:
        raise ProtocolError(f"natural anchor lag {k_n} is outside -15..15")
    anchor = float(array[lag_tuple.index(int(k_n))])
    mask = np.abs(np.asarray(lag_tuple, dtype=np.int64) - int(k_n)) > 2
    if int(mask.sum()) <= 0:
        raise ProtocolError("fixed far-background mask must contain at least one lag")
    far_background = float(np.median(array[mask]))
    relative = None if far_background <= 1e-12 else float((far_background - anchor) / far_background)
    return {
        "D_anchor": anchor,
        "C_anchor": float(np.median(array) - anchor),
        "search_bonus": float(anchor - float(np.min(array))),
        "O": far_background,
        "relative_margin": relative,
        "relative_margin_reason": "O<=1e-12" if relative is None else "",
    }


def _decompose(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    """Return a same-direction C decomposition for one paired comparison."""

    delta = float(candidate["C"] - baseline["C"])
    match = float(baseline["D"] - candidate["D"])
    background = float(candidate["B"] - baseline["B"])
    dominance = float(background - match)
    if abs(delta - (match + background)) > ABS_TOL:
        raise ProtocolError("delta_C != match_gain + background_gain")
    return {
        "delta_C": delta,
        "match_gain": match,
        "background_gain": background,
        "dominance": dominance,
    }


def _bootstrap_indices(draws: int = BOOTSTRAP_DRAWS, n: int = 15, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def _bootstrap_summary(values: Sequence[float], indices: np.ndarray, *, corrected: bool = False) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ProtocolError("bootstrap values must be a finite non-empty vector")
    if indices.ndim != 2 or indices.shape[1] != array.size or np.any(indices < 0) or np.any(indices >= array.size):
        raise ProtocolError("bootstrap index matrix does not match values")
    means = array[indices].mean(axis=1)
    alpha = BONFERRONI_ALPHA if corrected else 0.05
    interval = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    result: dict[str, Any] = {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive_count": int(np.count_nonzero(array > 0)),
        "n": int(array.size),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975], method="linear")],
    }
    if corrected:
        result["ci_bonferroni"] = [float(interval[0]), float(interval[1])]
        result["ci_bonferroni_coverage"] = float(1.0 - BONFERRONI_ALPHA)
        result["ci_bonferroni_alpha_each"] = float(BONFERRONI_ALPHA)
    return result


def _direction(value: float | None) -> str:
    if value is None:
        return "no_eligible_samples"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _summary(values: Sequence[float], indices: np.ndarray) -> dict[str, Any]:
    return _bootstrap_summary(values, indices, corrected=False)


def _optional_summary(values: Sequence[float | None], indices: np.ndarray) -> dict[str, Any]:
    """Summarize an optional diagnostic without turning missing values into NaN."""

    raw = list(values)
    valid = [float(value) for value in raw if value is not None]
    valid_n = len(valid)
    missing_n = len(raw) - valid_n
    if missing_n:
        return {
            "mean": None,
            "median": None,
            "positive_count": None,
            "n": len(raw),
            "valid_n": valid_n,
            "missing_n": missing_n,
            "ci95": [None, None],
        }
    result = _summary(valid, indices)
    result["valid_n"] = valid_n
    result["missing_n"] = 0
    return result


def _metric_summary(values: Sequence[float | None], metric: str, indices: np.ndarray) -> dict[str, Any]:
    if metric == "relative_margin":
        return _optional_summary(values, indices)
    return _summary(values, indices)


def _optional_linear(
    values: Mapping[tuple[str, str], Sequence[float | None]],
    terms: Mapping[tuple[str, str], float],
) -> list[float | None]:
    """Apply a linear contrast while preserving undefined optional entries."""

    if not terms:
        raise ProtocolError("optional linear contrast has no terms")
    first = next(iter(terms))
    if first not in values:
        raise ProtocolError(f"missing cell for optional linear contrast: {first}")
    n = len(values[first])
    result: list[float | None] = []
    for index in range(n):
        total = 0.0
        missing = False
        for key, coefficient in terms.items():
            if key not in values:
                raise ProtocolError(f"missing cell for optional linear contrast: {key}")
            vector = values[key]
            if len(vector) != n:
                raise ProtocolError("optional linear contrast vectors have different lengths")
            value = vector[index]
            if value is None:
                missing = True
                break
            total += float(coefficient) * float(value)
        result.append(None if missing else total)
    return result


def _optional_difference(candidate: Sequence[float | None], baseline: Sequence[float | None]) -> list[float | None]:
    if len(candidate) != len(baseline):
        raise ProtocolError("optional difference vectors have different lengths")
    return [
        None if left is None or right is None else float(left) - float(right)
        for left, right in zip(candidate, baseline, strict=True)
    ]


def _linear(values: Mapping[tuple[str, str], np.ndarray], terms: Mapping[tuple[str, str], float]) -> np.ndarray:
    result: np.ndarray | None = None
    for key, coefficient in terms.items():
        if key not in values:
            raise ProtocolError(f"missing cell for linear contrast: {key}")
        contribution = float(coefficient) * np.asarray(values[key], dtype=np.float64)
        result = contribution.copy() if result is None else result + contribution
    if result is None or not np.isfinite(result).all():
        raise ProtocolError("linear contrast is empty or non-finite")
    return result


def _condition_key(view: str, source: str, keep: str) -> tuple[str, str]:
    if view not in ("own", "natural"):
        raise ProtocolError(f"invalid view {view}")
    if source not in ("N", "T") or keep not in ("000", "050", "100"):
        raise ProtocolError(f"invalid condition {view}/{source}/{keep}")
    arm = f"{source}_{keep}"
    return (arm, arm) if view == "own" else (arm, "N_RAW")


def _attach_vector_decomposition(values: dict[str, Any]) -> dict[str, Any]:
    """Attach the shared C/D/B decomposition to vector-valued effects."""

    c = np.asarray(values["C"], dtype=np.float64)
    d = np.asarray(values["D"], dtype=np.float64)
    b = np.asarray(values["B"], dtype=np.float64)
    if c.ndim != 1 or d.shape != c.shape or b.shape != c.shape or not np.isfinite(c).all() or not np.isfinite(d).all() or not np.isfinite(b).all():
        raise ProtocolError("vector decomposition inputs must be finite, equal-length vectors")
    decomposed = [
        _decompose(
            {"C": float(c[index]), "D": float(d[index]), "B": float(b[index])},
            {"C": 0.0, "D": 0.0, "B": 0.0},
        )
        for index in range(c.size)
    ]
    values["match_gain"] = np.asarray([item["match_gain"] for item in decomposed], dtype=np.float64)
    values["background_gain"] = np.asarray([item["background_gain"] for item in decomposed], dtype=np.float64)
    values["delta_C"] = np.asarray([item["delta_C"] for item in decomposed], dtype=np.float64)
    values["dominance"] = np.asarray([item["dominance"] for item in decomposed], dtype=np.float64)
    values["match_anchor"] = -values["D_anchor"]
    values["search_change"] = values["search_bonus"]
    return values


def _effect_for_terms(
    metric_vectors: Mapping[str, Mapping[tuple[str, str], Sequence[float | None]]],
    terms: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    values = {
        metric: _linear(metric_vectors[metric], terms)
        for metric in ("C", "D", "B", "D_anchor", "search_bonus", "O", "lag_delta")
    }
    values["relative_margin"] = _optional_linear(metric_vectors["relative_margin"], terms)
    values["C_anchor"] = _linear(metric_vectors["C_anchor"], terms)
    return _attach_vector_decomposition(values)


def _make_primary_terms() -> dict[str, dict[tuple[str, str], float]]:
    n100 = _condition_key("own", "N", "100")
    n000 = _condition_key("own", "N", "000")
    t100 = _condition_key("own", "T", "100")
    t000 = _condition_key("own", "T", "000")
    return {
        "N_dose": {n100: 1.0, n000: -1.0},
        "T_dose": {t100: 1.0, t000: -1.0},
        "source_interaction": {t100: 1.0, t000: -1.0, n100: -1.0, n000: 1.0},
    }


def _build_cell_metrics(
    records: Sequence[Mapping[str, Any]],
    scores: Mapping[tuple[int, str, str, str], Mapping[str, Any]],
    curves: Mapping[tuple[int, str, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[tuple[str, str], dict[str, Any]]], list[dict[str, Any]]]:
    record_by_id = {int(record["sample_id"]): record for record in records}
    natural_lag: dict[int, int] = {}
    for sample_id, record in record_by_id.items():
        key = (sample_id, str(record["paired_key"]), "N_RAW", "N_RAW")
        natural_lag[sample_id] = int(_curve_metrics(curves[key]["lags"], curves[key]["values"])["k_star"])
        baseline_k_n = curves[key]["k_n"]
        if baseline_k_n is not None and int(baseline_k_n) != natural_lag[sample_id]:
            raise ProtocolError(f"natural baseline k_n must equal its k_star for sample {sample_id}")
    rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    by_sample: dict[int, dict[tuple[str, str], dict[str, Any]]] = {sid: {} for sid in record_by_id}
    for key in sorted(curves):
        sample_id, paired_key, video_arm, audio_arm = key
        score = scores[key]
        curve = curves[key]
        base = _curve_metrics(curve["lags"], curve["values"])
        k_star = int(base["k_star"])
        if k_star != int(score["k_star"]):
            raise ProtocolError(f"curve k_star/score k_star mismatch for {key}")
        if audio_arm == "N_RAW":
            if score["fixed_k_n"] is None or score["d_fixed_natural_lag"] is None:
                raise ProtocolError(f"fixed natural lag fields are required for {key}")
            if int(score["fixed_k_n"]) != natural_lag[sample_id]:
                raise ProtocolError(f"fixed_k_n mismatch for {key}")
            if curve["k_n"] is not None and int(curve["k_n"]) != natural_lag[sample_id]:
                raise ProtocolError(f"curve k_n mismatch for fixed natural audio cell {key}")
            expected_fixed = float(curve["values"][LAGS.index(natural_lag[sample_id])])
            if abs(float(score["d_fixed_natural_lag"]) - expected_fixed) > ABS_TOL:
                raise ProtocolError(f"d_fixed_natural_lag mismatch for {key}")
        if audio_arm != "N_RAW" and (score["fixed_k_n"] is not None or score["d_fixed_natural_lag"] is not None):
            raise ProtocolError(f"non-N_RAW cell unexpectedly has fixed natural lag fields: {key}")
        mechanism = (video_arm, audio_arm) != REFERENCE_CELL
        lag_extra = _lag_decomposition(curve["values"], curve["lags"], natural_lag[sample_id])
        identity_error = abs(float(base["C"]) - (float(lag_extra["C_anchor"]) + float(lag_extra["search_bonus"])))
        if mechanism and identity_error > ABS_TOL:
            raise ProtocolError(f"fixed-lag identity failed for {key}")
        if mechanism:
            identity_rows.append({"sample_id": sample_id, "cell": [video_arm, audio_arm], "identity_error": identity_error})
        if mechanism and float(lag_extra["search_bonus"]) < -ABS_TOL:
            raise ProtocolError(f"search_bonus is negative for {key}")
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "paired_key": paired_key,
            "video_arm": video_arm,
            "audio_arm": audio_arm,
            "reference_only": "true" if not mechanism else "false",
            "common_support_count": int(curve["common_support_count"]),
            "k_n": curve["k_n"] if curve["k_n"] is not None else "",
            "C": float(base["C"]),
            "D": float(base["D"]),
            "B": float(base["B"]),
            "curve_median": float(base["B"]),
            "k_star": k_star,
            "d_zero": float(base["d_zero"]),
            "fixed_k_n": score["fixed_k_n"],
            "d_fixed_natural_lag": score["d_fixed_natural_lag"],
            "k_N": natural_lag[sample_id],
            "k_N_ms": natural_lag[sample_id] * FRAME_MS,
        }
        if mechanism:
            row.update(lag_extra)
            row["lag_delta"] = int(k_star - natural_lag[sample_id])
            row["lag_delta_ms"] = float((k_star - natural_lag[sample_id]) * FRAME_MS)
        else:
            for field in ("D_anchor", "C_anchor", "search_bonus", "O", "relative_margin", "relative_margin_reason", "lag_delta", "lag_delta_ms"):
                row[field] = ""
        rows.append(row)
        if mechanism:
            by_sample[sample_id][(video_arm, audio_arm)] = row
    for sample_id, entries in by_sample.items():
        if len(entries) != 14:
            raise ProtocolError(f"sample {sample_id} has {len(entries)} mechanism cells, expected 14")
    return rows, by_sample, identity_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            row: dict[str, Any] = {}
            for field in fieldnames:
                value = raw.get(field, "")
                if value is None:
                    row[field] = ""
                elif isinstance(value, float):
                    if not math.isfinite(value):
                        raise ProtocolError(f"non-finite CSV value in {field}")
                    row[field] = repr(value)
                else:
                    row[field] = value
            writer.writerow(row)


def _write_cell_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "sample_id",
        "paired_key",
        "video_arm",
        "audio_arm",
        "reference_only",
        "common_support_count",
        "k_n",
        "C",
        "D",
        "B",
        "curve_median",
        "k_star",
        "d_zero",
        "fixed_k_n",
        "d_fixed_natural_lag",
        "k_N",
        "k_N_ms",
        "D_anchor",
        "C_anchor",
        "search_bonus",
        "O",
        "relative_margin",
        "relative_margin_reason",
        "lag_delta",
        "lag_delta_ms",
    )
    _write_csv(path, rows, fields)


def _primary_effect_rows(
    records: Sequence[Mapping[str, Any]],
    by_sample: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sample_ids = [int(record["sample_id"]) for record in records]
    metric_vectors: dict[str, Mapping[tuple[str, str], Sequence[float | None]]] = {}
    for metric in ("C", "D", "B", "D_anchor", "C_anchor", "search_bonus", "O", "lag_delta"):
        metric_vectors[metric] = {
            key: np.asarray([float(by_sample[sid][key][metric]) for sid in sample_ids], dtype=np.float64)
            for key in by_sample[sample_ids[0]]
        }
    metric_vectors["relative_margin"] = {
        key: [
            None if by_sample[sid][key]["relative_margin"] is None else float(by_sample[sid][key]["relative_margin"])
            for sid in sample_ids
        ]
        for key in by_sample[sample_ids[0]]
    }
    all_effects: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for contrast in PRIMARY_CONTRASTS:
        effects = _effect_for_terms(metric_vectors, _make_primary_terms()[contrast])
        all_effects[contrast] = effects
        for index, record in enumerate(records):
            row = {
                "sample_id": int(record["sample_id"]),
                "paired_key": str(record["paired_key"]),
                "family": "primary",
                "view": "own",
                "contrast": contrast,
                "delta_C": float(effects["delta_C"][index]),
                "match_gain": float(effects["match_gain"][index]),
                "background_gain": float(effects["background_gain"][index]),
                "dominance": float(effects["dominance"][index]),
                "match_anchor": float(effects["match_anchor"][index]),
                "search_change": float(effects["search_change"][index]),
                "O_change": float(effects["O"][index]),
                "relative_margin_change": (
                    "" if effects["relative_margin"][index] is None else float(effects["relative_margin"][index])
                ),
                "lag_delta_effect": float(effects["lag_delta"][index]),
                "lag_delta_candidate": "",
                "lag_delta_baseline": "",
            }
            terms = _make_primary_terms()[contrast]
            positive = [key for key, coefficient in terms.items() if coefficient > 0]
            negative = [key for key, coefficient in terms.items() if coefficient < 0]
            if len(positive) == 1 and len(negative) == 1:
                row["lag_delta_candidate"] = float(by_sample[int(record["sample_id"])][positive[0]]["lag_delta"])
                row["lag_delta_baseline"] = float(by_sample[int(record["sample_id"])][negative[0]]["lag_delta"])
            if abs(row["delta_C"] - (row["match_gain"] + row["background_gain"])) > ABS_TOL:
                raise ProtocolError(f"paired effect identity failed for {contrast} sample {record['sample_id']}")
            if abs(row["delta_C"] - (row["match_anchor"] + row["search_change"] + row["background_gain"])) > ABS_TOL:
                raise ProtocolError(f"fixed-lag paired identity failed for {contrast} sample {record['sample_id']}")
            rows.append(row)
    return rows, all_effects


def _write_paired_effects(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "sample_id",
        "paired_key",
        "family",
        "view",
        "contrast",
        "delta_C",
        "match_gain",
        "background_gain",
        "dominance",
        "match_anchor",
        "search_change",
        "O_change",
        "relative_margin_change",
        "lag_delta_effect",
        "lag_delta_candidate",
        "lag_delta_baseline",
    )
    _write_csv(path, rows, fields)


def _stats_for_effects(effects: Mapping[str, Mapping[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for contrast in PRIMARY_CONTRASTS:
        result[contrast] = {}
        for metric in PRIMARY_METRICS:
            result[contrast][metric] = _bootstrap_summary(effects[contrast][metric], indices, corrected=True)
    return result


def _condition_vectors(
    by_sample: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]],
    view: str,
    source: str,
    keep: str,
) -> dict[str, Any]:
    key = _condition_key(view, source, keep)
    sample_ids = list(by_sample)
    result = {}
    for metric in DESCRIPTIVE_METRICS + ("lag_delta",):
        if metric == "relative_margin":
            result[metric] = [
                None if by_sample[sid][key][metric] is None else float(by_sample[sid][key][metric])
                for sid in sample_ids
            ]
        else:
            result[metric] = np.asarray([float(by_sample[sid][key][metric]) for sid in sample_ids], dtype=np.float64)
    return result


def _effect_from_condition_vectors(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in candidate:
        if metric == "relative_margin":
            out[metric] = _optional_difference(candidate[metric], baseline[metric])
        else:
            out[metric] = np.asarray(candidate[metric] - baseline[metric], dtype=np.float64)
    return _attach_vector_decomposition(out)


def _summarize_effect(effect: Mapping[str, Any], indices: np.ndarray, metrics: Sequence[str]) -> dict[str, Any]:
    return {metric: _metric_summary(effect[metric], metric, indices) for metric in metrics}


def _make_dose_analysis(by_sample: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]], indices: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for view in ("own", "natural"):
        output[view] = {}
        for source in ("N", "T"):
            conditions = {keep: _condition_vectors(by_sample, view, source, keep) for keep in ("000", "050", "100")}
            output[view][source] = {
                "conditions": {
                    keep: {metric: _metric_summary(values[metric], metric, indices) for metric in DESCRIPTIVE_METRICS}
                    for keep, values in conditions.items()
                },
                "segments": {},
            }
            for label, left, right in (("050_minus_000", "050", "000"), ("100_minus_050", "100", "050"), ("100_minus_000", "100", "000")):
                effect = _effect_from_condition_vectors(conditions[left], conditions[right])
                output[view][source]["segments"][label] = {
                    "effects": _summarize_effect(effect, indices, ("delta_C", "match_gain", "background_gain", "dominance", "match_anchor", "search_change", "O", "relative_margin")),
                    "per_sample_C_delta": [float(x) for x in effect["delta_C"]],
                    "both_segments_C_positive_count": None,
                    "mean_C_monotonic": bool(float(np.mean(effect["delta_C"])) > 0),
                }
            first = _effect_from_condition_vectors(conditions["050"], conditions["000"])["delta_C"]
            second = _effect_from_condition_vectors(conditions["100"], conditions["050"])["delta_C"]
            output[view][source]["segments"]["100_minus_000"]["both_segments_C_positive_count"] = int(np.count_nonzero((first > 0) & (second > 0)))
            output[view][source]["segments"]["100_minus_000"]["per_sample_C_segments"] = {
                "050_minus_000": [float(x) for x in first],
                "100_minus_050": [float(x) for x in second],
            }
            output[view][source]["lag_delta_nonzero_counts"] = {
                keep: int(np.count_nonzero(conditions[keep]["lag_delta"] != 0)) for keep in ("000", "050", "100")
            }
    # The fixed-natural-track dose interaction is listed explicitly as an aux result.
    t100 = _condition_vectors(by_sample, "natural", "T", "100")
    t000 = _condition_vectors(by_sample, "natural", "T", "000")
    n100 = _condition_vectors(by_sample, "natural", "N", "100")
    n000 = _condition_vectors(by_sample, "natural", "N", "000")
    t_effect = _effect_from_condition_vectors(t100, t000)
    n_effect = _effect_from_condition_vectors(n100, n000)
    interaction = {
        metric: (
            _optional_difference(t_effect[metric], n_effect[metric])
            if metric == "relative_margin"
            else t_effect[metric] - n_effect[metric]
        )
        for metric in t_effect
    }
    output["natural"]["dose_interaction_T_minus_N"] = _summarize_effect(interaction, indices, ("delta_C", "match_gain", "background_gain", "dominance", "match_anchor", "search_change", "O", "relative_margin"))
    return output


def _source_differences(by_sample: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]], indices: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for view in ("own", "natural"):
        raw = {
            metric: np.asarray(
                [float(by_sample[sid][("N_RAW", "N_RAW")][metric]) for sid in by_sample],
                dtype=np.float64,
            )
            for metric in DESCRIPTIVE_METRICS + ("lag_delta",)
            if metric != "relative_margin"
        }
        raw["relative_margin"] = [
            None
            if by_sample[sid][("N_RAW", "N_RAW")]["relative_margin"] is None
            else float(by_sample[sid][("N_RAW", "N_RAW")]["relative_margin"])
            for sid in by_sample
        ]
        n100 = _condition_vectors(by_sample, view, "N", "100")
        t100 = _condition_vectors(by_sample, view, "T", "100")
        effects = {
            "N_100_minus_N_RAW": _effect_from_condition_vectors(n100, raw),
            "T_100_minus_N_100": _effect_from_condition_vectors(t100, n100),
            "T_100_minus_N_RAW": _effect_from_condition_vectors(t100, raw),
        }
        first, second, total = (effects["N_100_minus_N_RAW"], effects["T_100_minus_N_100"], effects["T_100_minus_N_RAW"])
        identity_errors = {metric: float(np.max(np.abs(first[metric] + second[metric] - total[metric]))) for metric in ("delta_C", "match_gain", "background_gain", "dominance")}
        if max(identity_errors.values()) > ABS_TOL:
            raise ProtocolError(f"source difference decomposition failed in {view}: {identity_errors}")
        output[view] = {
            name: _summarize_effect(effect, indices, ("delta_C", "match_gain", "background_gain", "dominance", "match_anchor", "search_change", "O", "relative_margin"))
            for name, effect in effects.items()
        }
        output[view]["identity_max_error"] = identity_errors
    return output


def _two_by_two(by_sample: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]], indices: np.ndarray) -> dict[str, Any]:
    sample_ids = list(by_sample)
    keys = {
        "Q00": ("N_RAW", "N_RAW"),
        "Q10": ("T_100", "N_RAW"),
        "Q01": ("N_RAW", "T_100"),
        "Q11": ("T_100", "T_100"),
    }
    output: dict[str, Any] = {"cells": {name: list(key) for name, key in keys.items()}, "metrics": {}}
    for metric in ("C", "-D", "B"):
        def q(name: str, metric_name: str = metric) -> np.ndarray:
            key = keys[name]
            raw_metric = "D" if metric_name == "-D" else metric_name
            values = np.asarray([float(by_sample[sid][key][raw_metric]) for sid in sample_ids], dtype=np.float64)
            return -values if metric_name == "-D" else values

        q00, q10, q01, q11 = (q(name) for name in ("Q00", "Q10", "Q01", "Q11"))
        effects = {
            "video_effect_at_N": q10 - q00,
            "audio_effect_at_N": q01 - q00,
            "av_interaction": q11 - q10 - q01 + q00,
            "native_total": q11 - q00,
            "audio_effect_at_T": q11 - q10,
            "video_effect_at_T": q11 - q01,
        }
        identity = effects["native_total"] - effects["video_effect_at_N"] - effects["audio_effect_at_N"] - effects["av_interaction"]
        identity_error = float(np.max(np.abs(identity)))
        if identity_error > ABS_TOL:
            raise ProtocolError(f"2x2 identity failed for {metric}")
        entry: dict[str, Any] = {
            "effects": {name: _summary(value, indices) for name, value in effects.items()},
            "identity_max_error": identity_error,
        }
        if metric == "C":
            entry["decomposition"] = {}
            for name in effects:
                # Build the corresponding C contrast directly, then use -L(D)/L(B).
                contrast_terms = {
                    "video_effect_at_N": {"Q10": 1.0, "Q00": -1.0},
                    "audio_effect_at_N": {"Q01": 1.0, "Q00": -1.0},
                    "av_interaction": {"Q11": 1.0, "Q10": -1.0, "Q01": -1.0, "Q00": 1.0},
                    "native_total": {"Q11": 1.0, "Q00": -1.0},
                    "audio_effect_at_T": {"Q11": 1.0, "Q10": -1.0},
                    "video_effect_at_T": {"Q11": 1.0, "Q01": -1.0},
                }[name]
                cvecs = {qname: np.asarray([float(by_sample[sid][keys[qname]]["C"]) for sid in sample_ids]) for qname in keys}
                dvecs = {qname: np.asarray([float(by_sample[sid][keys[qname]]["D"]) for sid in sample_ids]) for qname in keys}
                bvecs = {qname: np.asarray([float(by_sample[sid][keys[qname]]["B"]) for sid in sample_ids]) for qname in keys}
                c_line = sum(coef * cvecs[qname] for qname, coef in contrast_terms.items())
                match = -sum(coef * dvecs[qname] for qname, coef in contrast_terms.items())
                background = sum(coef * bvecs[qname] for qname, coef in contrast_terms.items())
                error = float(np.max(np.abs(c_line - match - background)))
                if error > ABS_TOL:
                    raise ProtocolError(f"2x2 C decomposition failed for {name}")
                entry["decomposition"][name] = {
                    "match_gain": _summary(match, indices),
                    "background_gain": _summary(background, indices),
                    "dominance": _summary(background - match, indices),
                    "identity_max_error": error,
                }
        output["metrics"][metric] = entry
    return output


def _sensitivity(by_sample: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]], indices: np.ndarray) -> list[dict[str, Any]]:
    del indices  # The spec requires means and directions only for this table.
    sample_ids = list(by_sample)
    supports = {sid: int(next(iter(by_sample[sid].values()))["common_support_count"]) for sid in sample_ids}
    terms = _make_primary_terms()
    table: list[dict[str, Any]] = []
    for threshold in SUPPORT_THRESHOLDS:
        eligible = [sid for sid in sample_ids if supports[sid] >= threshold]
        row: dict[str, Any] = {
            "threshold": threshold,
            "expected_n": EXPECTED_SUPPORT_N[threshold],
            "n": len(eligible),
            "sample_ids": eligible,
            "support_by_sample": {str(sid): supports[sid] for sid in eligible},
            "contrasts": {},
        }
        for contrast, contrast_terms in terms.items():
            effects: dict[str, list[float]] = {metric: [] for metric in PRIMARY_METRICS}
            for sid in eligible:
                entries = by_sample[sid]
                candidate = _linear({key: np.asarray([float(entries[key]["C"])]) for key in entries}, contrast_terms)[0]
                d_effect = _linear({key: np.asarray([float(entries[key]["D"])]) for key in entries}, contrast_terms)[0]
                b_effect = _linear({key: np.asarray([float(entries[key]["B"])]) for key in entries}, contrast_terms)[0]
                effects["delta_C"].append(float(candidate))
                effects["match_gain"].append(float(-d_effect))
                effects["background_gain"].append(float(b_effect))
                effects["dominance"].append(float(b_effect + d_effect))
            row["contrasts"][contrast] = {
                metric: {"mean": float(np.mean(values)) if values else None, "direction": _direction(float(np.mean(values)) if values else None)}
                for metric, values in effects.items()
            }
        table.append(row)
    return table


def _plot(by_sample: Mapping[int, Mapping[tuple[str, str], Mapping[str, Any]]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keeps = ("000", "050", "100")
    x = np.asarray([0.0, 0.5, 1.0])
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for row_index, view in enumerate(("own", "natural")):
        for col_index, metric in enumerate(("C", "D", "B")):
            axis = axes[row_index][col_index]
            for source, color in (("N", "tab:blue"), ("T", "tab:orange")):
                means = [float(np.mean(_condition_vectors(by_sample, view, source, keep)[metric])) for keep in keeps]
                axis.plot(x, means, marker="o", label=source, color=color)
            axis.set_title(f"{view}: {metric}")
            axis.set_xlabel("keep")
            axis.set_ylabel(metric + (" (lower is better)" if metric == "D" else ""))
            axis.set_xticks(x)
            axis.grid(alpha=0.25)
            if col_index == 0:
                axis.legend(frameon=False)
    figure.suptitle("MFA-linear trajectory retention (per-utterance metrics before aggregation)")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _interpretation(analysis: Mapping[str, Any]) -> list[str]:
    """Apply the predeclared interpretation rules without inventing tests."""

    primary = analysis["primary"]["comparisons"]
    t_dose = primary["T_dose"]
    n_dose = primary["N_dose"]
    interaction = primary["source_interaction"]
    lines: list[str] = []
    t_delta_ci = t_dose["delta_C"]["ci_bonferroni"]
    t_match_ci = t_dose["match_gain"]["ci_bonferroni"]
    t_background_ci = t_dose["background_gain"]["ci_bonferroni"]
    t_dominance_ci = t_dose["dominance"]["ci_bonferroni"]
    if t_delta_ci[0] > 0:
        lines.append("T_dose 的 delta_C Bonferroni 区间下界高于 0：在当前条件下，保留轨迹提高原生 Sync-C。")
        if t_match_ci[1] < 0:
            lines.append("match_gain 的校正区间上界低于 0，最佳距离项方向为抵消；总增分由背景项承担，不能把两项都称为改善。")
        elif t_background_ci[1] < 0:
            lines.append("background_gain 的校正区间上界低于 0，背景项方向为抵消；总增分由最佳距离项承担，不能把两项都称为改善。")
        elif t_background_ci[0] > 0 and t_dominance_ci[0] > 0:
            lines.append("background_gain 与 dominance 的校正区间下界也高于 0，背景项的数值贡献大于最佳距离项。")
        elif t_match_ci[0] > 0 and t_dominance_ci[1] < 0:
            lines.append("match_gain 的校正区间下界高于 0 且 dominance 上界低于 0，最佳距离项的数值贡献更大。")
        elif t_match_ci[0] > 0 and t_background_ci[0] > 0:
            lines.append("两个组件的校正区间都在 0 以上，但 dominance 未排除 0，不能区分谁的贡献更大。")
        else:
            lines.append("至少一个组件的校正区间跨过 0，因此总增分不能包装成两个组件都改善。")
    elif t_delta_ci[1] < 0:
        lines.append("T_dose 的 delta_C Bonferroni 区间上界低于 0，当前条件下保留轨迹对应原生 Sync-C 的稳定负向变化。")
        if t_match_ci[1] < 0:
            lines.append("match_gain 的校正区间上界低于 0，最佳距离项对总变化形成明确抵消。")
        if t_background_ci[1] < 0:
            lines.append("background_gain 的校正区间上界低于 0，背景项对总变化形成明确抵消。")
    else:
        lines.append("T_dose 的 delta_C 校正区间跨过 0，尚未建立当前条件下稳定的原生 Sync-C 总增益。")
        if t_match_ci[1] < 0:
            lines.append("match_gain 的校正区间上界低于 0，最佳距离项对总变化形成明确抵消。")
        if t_background_ci[1] < 0:
            lines.append("background_gain 的校正区间上界低于 0，背景项对总变化形成明确抵消。")

    if n_dose["delta_C"]["ci_bonferroni"][0] > 0 and t_dose["delta_C"]["ci_bonferroni"][0] > 0 and interaction["delta_C"]["ci_bonferroni"][0] <= 0 <= interaction["delta_C"]["ci_bonferroni"][1]:
        lines.append("N_dose 与 T_dose 都为正而 source_interaction 区间跨 0，结果更支持一般轨迹线索，尚未识别 TTS 特异性。")
    elif interaction["delta_C"]["ci_bonferroni"][0] > 0:
        lines.append("source_interaction 的 delta_C 校正区间下界高于 0，TTS 来源对该操作更敏感；其中仍可能包含声码器域敏感度。")
    elif interaction["delta_C"]["ci_bonferroni"][1] < 0:
        lines.append("source_interaction 的 delta_C 校正区间上界低于 0，TTS 来源对该操作的响应低于自然来源；其中仍可能包含声码器域敏感度。")
    else:
        lines.append("source_interaction 的 delta_C 校正区间跨 0，不能声称 TTS 来源对该操作更敏感。")

    gown = analysis["descriptive"]["source_differences"]["own"]["T_100_minus_N_RAW"]["delta_C"]
    if gown["ci95"][0] <= 0 <= gown["ci95"][1]:
        lines.append("G_own（T_100−N_RAW）均值为正但普通 95% CI 跨 0，本轮阳性参照不确定。")
    natural_t = analysis["dose"]["natural"]["T"]["segments"]["100_minus_000"]["effects"]["delta_C"]
    if t_dose["delta_C"]["ci_bonferroni"][0] > 0 and natural_t["ci95"][0] <= 0 <= natural_t["ci95"][1]:
        lines.append("原生视角改善而固定自然音轨视角未排除 0，原生增分尚未显示自然音轨兼容性转移。")
    elif t_dose["delta_C"]["ci_bonferroni"][0] > 0 and natural_t["ci95"][0] > 0:
        lines.append("固定自然音轨视角的 T_dose 普通 95% CI 也在 0 以上；这仍是历史分数响应，不能当作真实同步真值。")
    lines.append("background_gain>0 只说明曲线中位数上移；须结合 O、relative_margin 与 C_anchor，不能单独称为错位配对可区分性。")
    return lines


def _write_report(analysis: Mapping[str, Any], output: Path) -> None:
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    lines = [
        "# MFA-linear 轨迹增益的匹配项与背景项分解",
        "",
        "本报告是固定 support30 历史运行的 CPU-only 探索性再分析：S0765 的 15 条 utterance、225 个 cell、每条 31 个 lag。所有主区间来自同一组 20,000 次 utterance bootstrap；12 个主统计量使用 Bonferroni 校正区间（覆盖率 99.583333%，alpha_each=0.05/12）。",
        "",
        "## 主分析",
        "",
        "`match_gain = D_baseline - D_candidate`，`background_gain = B_candidate - B_baseline`，两者相加等于 `delta_C`。D 的原始方向是越低越好。下表先给均值，再给普通 95% CI 和校正 CI；数值保留三位小数，完整精度在 analysis.json。",
        "",
        "| 对比 | 指标 | 均值 | 普通95% CI | Bonferroni CI | 阳性条数/n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for contrast in PRIMARY_CONTRASTS:
        for metric in PRIMARY_METRICS:
            stats = analysis["primary"]["comparisons"][contrast][metric]
            lines.append(
                f"| {contrast} | {metric} | {fmt(stats['mean'])} | [{fmt(stats['ci95'][0])}, {fmt(stats['ci95'][1])}] | [{fmt(stats['ci_bonferroni'][0])}, {fmt(stats['ci_bonferroni'][1])}] | {stats['positive_count']}/{stats['n']} |"
            )
    lines += [
        "",
        "主判读只使用校正区间。`source_interaction` 是逐条先做 T_dose−N_dose 后再 bootstrap 的配对交互，不能由两个独立区间相减得到。",
        "",
    ]
    lines += [
        "",
        "## 辅助结果（descriptive/unadjusted）",
        "",
        "### 完整剂量与固定 lag",
        "",
        "| 视角/来源/keep | C | D | B | D_anchor | C_anchor | search_bonus | O | relative_margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in ("own", "natural"):
        for source in ("N", "T"):
            for keep in ("000", "050", "100"):
                stats = analysis["dose"][view][source]["conditions"][keep]
                lines.append(
                    f"| {view}/{source}/{keep} | {fmt(stats['C']['mean'])} | {fmt(stats['D']['mean'])} | {fmt(stats['B']['mean'])} | {fmt(stats['D_anchor']['mean'])} | {fmt(stats['C_anchor']['mean'])} | {fmt(stats['search_bonus']['mean'])} | {fmt(stats['O']['mean'])} | {fmt(stats['relative_margin']['mean'])} |"
                )
    lines += [
        "",
        "| 视角/来源/区段 | delta_C | match_gain | background_gain | dominance | 两段均正的 C 条数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view in ("own", "natural"):
        for source in ("N", "T"):
            for segment in ("050_minus_000", "100_minus_050", "100_minus_000"):
                entry = analysis["dose"][view][source]["segments"][segment]
                effects = entry["effects"]
                both = entry.get("both_segments_C_positive_count", "")
                lines.append(
                    f"| {view}/{source}/{segment} | {fmt(effects['delta_C']['mean'])} | {fmt(effects['match_gain']['mean'])} | {fmt(effects['background_gain']['mean'])} | {fmt(effects['dominance']['mean'])} | {both} |"
                )
    lines += [
        "",
        "### 来源差与 2×2",
        "",
        "| 视角/对比 | delta_C | match_gain | background_gain | dominance |",
        "|---|---:|---:|---:|---:|",
    ]
    for view in ("own", "natural"):
        for contrast in ("N_100_minus_N_RAW", "T_100_minus_N_100", "T_100_minus_N_RAW"):
            stats = analysis["descriptive"]["source_differences"][view][contrast]
            lines.append(
                f"| {view}/{contrast} | {fmt(stats['delta_C']['mean'])} | {fmt(stats['match_gain']['mean'])} | {fmt(stats['background_gain']['mean'])} | {fmt(stats['dominance']['mean'])} |"
            )
    lines += [
        "",
        "| 2×2 metric/effect | mean | ordinary 95% CI |",
        "|---|---:|---:|",
    ]
    for metric in ("C", "-D", "B"):
        for effect, stats in analysis["two_by_two"]["metrics"][metric]["effects"].items():
            lines.append(f"| {metric}/{effect} | {fmt(stats['mean'])} | [{fmt(stats['ci95'][0])}, {fmt(stats['ci95'][1])}] |")
    lines += [
        "",
        "### 支持阈值敏感性",
        "",
        "| threshold | n | contrast | delta_C | match_gain | background_gain | dominance |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in analysis["sensitivity"]:
        for contrast in PRIMARY_CONTRASTS:
            stats = row["contrasts"][contrast]
            lines.append(
                f"| {row['threshold']} | {row['n']} | {contrast} | {fmt(stats['delta_C']['mean'])} ({stats['delta_C']['direction']}) | {fmt(stats['match_gain']['mean'])} ({stats['match_gain']['direction']}) | {fmt(stats['background_gain']['mean'])} ({stats['background_gain']['direction']}) | {fmt(stats['dominance']['mean'])} ({stats['dominance']['direction']}) |"
            )
    lines += [
        "",
        "图见 `decomposition.png`；图中先对每条样本计算 C/D/B，再取样本均值。",
        "",
        "支持阈值敏感性只筛选 14 个机制 cell 的共同支持数，阈值为 30/35/40/45/50，预期样本数为 15/13/12/10/8；不重评分、不截曲线、不把 T_RAW 参考 cell 纳入。",
        "",
        "### 预定判读",
        "",
    ]
    lines.extend(f"- {item}" for item in analysis["interpretation"])
    lines += [
        "",
        "## 边界",
        "",
        "这是单说话人、历史已见样本的探索性代数分解。T_100 表示历史 MFA-linear 条件特征来源，不是原始 TTS 波形。本分析没有真实嘴型真值、人工同步标签或音质因果控制，因此不能宣称真实口型更准确、证明 SyncNet 被欺骗，或解释所有 TTS 优势。背景中位数也不是错误配对真值，远侧 O 不能直接叫负样本准确率；relative_margin 仅是尺度诊断。",
        "",
        "源 run 的五个冻结文件在分析前后均通过 SHA-256 校验，源目录保持只读。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validation_success(
    source_hashes_before: Mapping[str, str],
    source_hashes_after: Mapping[str, str],
    scores: Mapping[Any, Any],
    curves: Mapping[Any, Any],
    cell_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    def add(name: str, expected: Any, actual: Any, passed: bool = True) -> None:
        checks.append({"name": name, "pass": bool(passed), "expected": expected, "actual": actual})

    add("source_hashes_before", dict(SOURCE_EXPECTED_HASHES), dict(source_hashes_before), dict(source_hashes_before) == dict(SOURCE_EXPECTED_HASHES))
    add("source_hashes_after_unchanged", dict(source_hashes_before), dict(source_hashes_after), dict(source_hashes_before) == dict(source_hashes_after))
    add("score_cell_count", 225, len(scores), len(scores) == 225)
    add("curve_cell_count", 225, len(curves), len(curves) == 225)
    add("cell_metrics_rows", 225, len(cell_rows), len(cell_rows) == 225)
    add("mechanism_cell_rows", 210, sum(row["reference_only"] == "false" for row in cell_rows), sum(row["reference_only"] == "false" for row in cell_rows) == 210)
    add("reference_only_rows", 15, sum(row["reference_only"] == "true" for row in cell_rows), sum(row["reference_only"] == "true" for row in cell_rows) == 15)
    add("primary_paired_rows", 45, len(paired_rows), len(paired_rows) == 45)
    add("primary_comparison_count", 12, sum(len(analysis["primary"]["comparisons"][name]) for name in PRIMARY_CONTRASTS), sum(len(analysis["primary"]["comparisons"][name]) for name in PRIMARY_CONTRASTS) == 12)
    identity_error = float(analysis["validation_summary"]["max_identity_error"])
    add("max_identity_error", "<=1e-6", identity_error, identity_error <= ABS_TOL)
    add("primary_values_finite", True, bool(analysis["validation_summary"]["primary_values_finite"]), bool(analysis["validation_summary"]["primary_values_finite"]))
    return checks


def _run(source_arg: str, output_arg: str) -> int:
    source = Path(source_arg).resolve()
    output = Path(output_arg).resolve()
    checks: list[dict[str, Any]] = []
    prepared = False
    try:
        _prepare_output(source, output)
        prepared = True
        source_hashes_before = _verify_source_hashes(source)
        source_inputs = _read_json(source / "inputs.json")
        records = _validate_inputs(source_inputs)
        scores_manifest = _read_json(source / "scores_manifest.json")
        reuse_manifest = _read_json(source / "reuse_manifest.json")
        _validate_manifest(scores_manifest, "scores_manifest.json", records)
        _validate_manifest(reuse_manifest, "reuse_manifest.json", records)
        scores = _load_scores(source / "scores.csv", records, scores_manifest)
        curves = _load_curves(source / "curves.json", records, scores)
        cell_rows, by_sample, cell_identity_rows = _build_cell_metrics(records, scores, curves)
        paired_rows, effects = _primary_effect_rows(records, by_sample)
        indices = _bootstrap_indices()
        primary = _stats_for_effects(effects, indices)
        dose = _make_dose_analysis(by_sample, indices)
        source_differences = _source_differences(by_sample, indices)
        two_by_two = _two_by_two(by_sample, indices)
        sensitivity = _sensitivity(by_sample, indices)
        finite_primary = all(
            math.isfinite(float(stats[field]))
            for contrast in PRIMARY_CONTRASTS
            for metric in PRIMARY_METRICS
            for stats in (primary[contrast][metric],)
            for field in ("mean", "median")
        )
        max_identity_error = 0.0
        for row in paired_rows:
            max_identity_error = max(
                max_identity_error,
                abs(float(row["delta_C"]) - float(row["match_gain"]) - float(row["background_gain"])),
                abs(float(row["delta_C"]) - float(row["match_anchor"]) - float(row["search_change"]) - float(row["background_gain"])),
            )
        if cell_identity_rows:
            max_identity_error = max(
                max_identity_error,
                max(float(row["identity_error"]) for row in cell_identity_rows),
            )
        analysis: dict[str, Any] = {
            "status": "complete",
            "spec": SPEC_ID,
            "protocol": PROTOCOL_ID,
            "source_run": str(source),
            "sample_ids": [int(record["sample_id"]) for record in records],
            "sample_count": len(records),
            "score_cell_count": len(scores),
            "mechanism_cell_count": 210,
            "reference_only_cell_count": 15,
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "draws": BOOTSTRAP_DRAWS,
                "unit": "utterance",
                "same_index_matrix_for_all": True,
                "bonferroni_tests": BONFERRONI_TESTS,
                "bonferroni_alpha_each": BONFERRONI_ALPHA,
                "bonferroni_coverage": 1.0 - BONFERRONI_ALPHA,
                "quantile_method": "linear",
            },
            "primary": {
                "correction": "Bonferroni over 12 contrast×metric statistics",
                "comparisons": primary,
            },
            "descriptive": {
                "label": "descriptive/unadjusted",
                "source_differences": source_differences,
                "primary_auxiliary": {
                    contrast: {
                        metric: _metric_summary(effects[contrast][metric], metric, indices)
                        for metric in ("match_anchor", "search_change", "O", "relative_margin", "lag_delta")
                    }
                    for contrast in PRIMARY_CONTRASTS
                },
            },
            "dose": dose,
            "two_by_two": two_by_two,
            "sensitivity": sensitivity,
            "limitations": [
                "单说话人 S0765、15 条历史已见 utterance，bootstrap 只表示该队列的重采样不确定性。",
                "T_100 是历史 MFA-linear 条件特征来源，不是原始 TTS 波形；轨迹作用与声码器域适配不能完全分离。",
                "C/D/B 的分解是 SyncNet 距离曲线的数值恒等式，不提供真实嘴型真值、人工同步或评价器偏差的因果识别。",
                "B 是全 31 lag 的中位数，O 只是固定自然 lag 远侧背景，均不是真值负配对分布。",
            ],
            "validation_summary": {"max_identity_error": max_identity_error, "primary_values_finite": finite_primary},
        }
        analysis["interpretation"] = _interpretation(analysis)
        source_hashes_after = {name: sha256_file(source / name) for name in SOURCE_EXPECTED_HASHES}
        checks = _validation_success(source_hashes_before, source_hashes_after, scores, curves, cell_rows, paired_rows, analysis)
        if not all(bool(item["pass"]) for item in checks):
            raise ProtocolError("post-calculation validation failed")
        script_hash = sha256_file(Path(__file__).resolve())
        inputs_out = {
            "status": "complete",
            "spec": SPEC_ID,
            "protocol": PROTOCOL_ID,
            "source_protocol": SOURCE_PROTOCOL,
            "source_run": str(source_arg),
            "source_run_resolved": str(source),
            "source_files": {
                name: {
                    "expected_sha256": SOURCE_EXPECTED_HASHES[name],
                    "actual_sha256_before": source_hashes_before[name],
                    "actual_sha256_after": source_hashes_after[name],
                }
                for name in SOURCE_EXPECTED_HASHES
            },
            "records": records,
            "parameters": {
                "lags": list(LAGS),
                "vshift": VSHIFT,
                "frame_ms": FRAME_MS,
                "min_common_windows": 30,
                "reference_cell": list(REFERENCE_CELL),
                "mechanism_cells_per_sample": 14,
                "support_thresholds": list(SUPPORT_THRESHOLDS),
            },
            "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "n": 15, "bonferroni_tests": BONFERRONI_TESTS},
            "script_sha256": script_hash,
            "git_commit": _git_commit(),
            "numpy_version": np.__version__,
        }
        _write_cell_metrics(output / "cell_metrics.csv", cell_rows)
        _write_paired_effects(output / "paired_effects.csv", paired_rows)
        _plot(by_sample, output / "decomposition.png")
        _write_report(analysis, output)
        write_json(output / "inputs.json", inputs_out)
        write_json(output / "analysis.json", analysis)
        write_json(output / "validation.json", {"status": "complete", "valid": True, "checks": checks})
        return 0
    except Exception as exc:  # noqa: BLE001 - the CLI must mark every run-time failure incomplete.
        # Once _prepare_output succeeds, the directory belongs to this run.
        # Always replace any partial success marker with an explicit incomplete
        # status; a non-empty directory rejected by _prepare_output is never
        # touched.
        if prepared and output.exists() and output.is_dir():
            try:
                report = output / "report.md"
                if report.exists():
                    report.unlink()
                write_json(output / "analysis.json", {"status": "incomplete", "spec": SPEC_ID, "protocol": PROTOCOL_ID, "error": str(exc)})
                write_json(
                    output / "validation.json",
                    {
                        "status": "incomplete",
                        "valid": False,
                        "checks": checks,
                        "artifacts_present": sorted(item.name for item in output.iterdir()),
                        "error": str(exc),
                    },
                )
            except (OSError, ProtocolError):
                pass
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="frozen support30 run directory")
    parser.add_argument("--output-dir", required=True, help="new or empty output directory")
    args = parser.parse_args(argv)
    return _run(args.source_run, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
