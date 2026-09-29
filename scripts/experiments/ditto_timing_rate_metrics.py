"""Pure NumPy protocol math for ``ditto_timing_rate_v1``.

This module intentionally has no torch, ffmpeg, or repository imports.  The
runner owns media/model I/O; this file owns the support, lag, rank, and
bootstrap rules so they can be tested on small synthetic arrays and reused by
the independent checker only through reimplemented logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

K_MAX = 8
LAGS = tuple(range(-K_MAX, K_MAX + 1))
DELTAS = (-3, -2, -1, 1, 2, 3)
PHASE_COUNT = 200
GUARD_LEFT = 12
GUARD_RIGHT = 17
BOOTSTRAP_SEED = 20_260_918
BOOTSTRAP_DRAWS = 20_000


def _embeddings(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [F,D], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def safe_rows(frame_count: int, lo: int, hi: int) -> np.ndarray:
    """Rows whose complete conservative temporal support lies in ``[lo, hi)``."""

    F = int(frame_count)
    left = max(0, int(lo))
    right = min(F, int(hi))
    rows = [i for i in range(left, right) if i - GUARD_LEFT >= left and i + GUARD_RIGHT <= right]
    return np.asarray(rows, dtype=np.int64)


def support_for_halves(frame_count: int) -> dict[str, Any]:
    F = int(frame_count)
    boundary = F // 2
    halves = {
        "H0": {"lo": 0, "hi": boundary, "rows": safe_rows(F, 0, boundary)},
        "H1": {"lo": boundary, "hi": F, "rows": safe_rows(F, boundary, F)},
    }
    return {
        "frame_count": F,
        "boundary": boundary,
        "halves": {
            key: {"lo": int(value["lo"]), "hi": int(value["hi"]), "rows": [int(x) for x in value["rows"]]}
            for key, value in halves.items()
        },
        "complete": all(len(value["rows"]) >= 10 for value in halves.values()),
    }


def build_common_phases(frame_counts: Mapping[str, int], *, phase_count: int = PHASE_COUNT) -> dict[str, Any]:
    """Build the fixed relative-progress phase support shared by all six cells."""

    if not frame_counts:
        raise ValueError("at least one cell is required")
    support = {key: support_for_halves(int(F)) for key, F in frame_counts.items()}
    phase_rows: list[dict[str, Any]] = []
    for phase_index in range(int(phase_count)):
        phase = (phase_index + 0.5) / float(phase_count)
        half = "H0" if phase_index < phase_count // 2 else "H1"
        indices: dict[str, int] = {}
        valid = True
        for key, item in support.items():
            F = int(item["frame_count"])
            index = int(np.floor(phase * F))
            expected = item["halves"][half]
            if not (expected["lo"] <= index < expected["hi"] and index in expected["rows"]):
                valid = False
                break
            indices[key] = index
        if valid:
            phase_rows.append({"phase_index": phase_index, "phase": phase, "half": half, "indices": indices})
    counts = {
        key: {
            "H0": len({row["indices"][key] for row in phase_rows if row["half"] == "H0"}),
            "H1": len({row["indices"][key] for row in phase_rows if row["half"] == "H1"}),
        }
        for key in frame_counts
    }
    complete = bool(phase_rows) and all(value["H0"] >= 10 and value["H1"] >= 10 for value in counts.values())
    return {
        "phase_count": int(phase_count),
        "records": phase_rows,
        "distinct_index_counts": counts,
        "support": support,
        "complete": complete,
    }


def _distance_rows(visual: np.ndarray, audio: np.ndarray, rows: np.ndarray, lag: int) -> np.ndarray:
    v = _embeddings(visual, name="visual")
    a = _embeddings(audio, name="audio")
    q = np.asarray(rows, dtype=np.int64).reshape(-1)
    indices = q + int(lag)
    if np.any(q < 0) or np.any(q >= v.shape[0]) or np.any(indices < 0) or np.any(indices >= a.shape[0]):
        raise ValueError("distance support is out of bounds")
    return np.linalg.norm(v[q].astype(np.float64) - a[indices].astype(np.float64), axis=1)


def calibrate_lag(visual: np.ndarray, audio: np.ndarray, rows: Sequence[int], lags: Sequence[int] = LAGS) -> dict[str, Any]:
    """Choose one lag from the supplied rows; exact ties use |k| then k."""

    q = np.asarray(rows, dtype=np.int64).reshape(-1)
    if q.size == 0:
        raise ValueError("lag calibration needs at least one row")
    objectives: dict[int, float] = {}
    for lag in lags:
        objectives[int(lag)] = float(np.mean(_distance_rows(visual, audio, q, int(lag))))
    ordered = sorted(objectives, key=lambda lag: (objectives[lag], abs(lag), lag))
    best = int(ordered[0])
    return {
        "k0": best,
        "objective": objectives[best],
        "objectives": {str(key): value for key, value in objectives.items()},
        "rows": [int(x) for x in q],
        "boundary": abs(best) == K_MAX,
    }


def _win(distance_shift: float, distance_zero: float) -> float:
    if distance_shift > distance_zero:
        return 1.0
    if distance_shift < distance_zero:
        return 0.0
    return 0.5


def rank_cell(
    visual: np.ndarray,
    audio: np.ndarray,
    phase_records: Sequence[Mapping[str, Any]],
    key: str,
    calibrations: Mapping[str, Mapping[str, Any]],
    *,
    deltas: Sequence[int] = DELTAS,
) -> dict[str, Any]:
    """Score local audio mismatches using the opposite-half calibrated lag."""

    v = _embeddings(visual, name="visual")
    a = _embeddings(audio, name="audio")
    by_half: dict[str, list[float]] = {"H0": [], "H1": []}
    zero_by_half: dict[str, list[float]] = {"H0": [], "H1": []}
    rows: list[dict[str, Any]] = []
    for record in phase_records:
        half = str(record["half"])
        index = int(record["indices"][key])
        k0 = int(calibrations[half]["k0"])
        zero_index = index + k0
        if not (0 <= index < v.shape[0] and 0 <= zero_index < a.shape[0]):
            raise ValueError("rank row is out of bounds")
        d0 = float(np.linalg.norm(v[index].astype(np.float64) - a[zero_index].astype(np.float64)))
        outcomes: dict[str, float] = {}
        for delta in deltas:
            shifted = zero_index + int(delta)
            if not 0 <= shifted < a.shape[0]:
                raise ValueError("local mismatch row is out of bounds")
            outcomes[str(int(delta))] = _win(float(np.linalg.norm(v[index].astype(np.float64) - a[shifted].astype(np.float64))), d0)
        value = float(np.mean(list(outcomes.values())))
        by_half[half].append(value)
        zero_index_unadjusted = index
        d_zero = float(np.linalg.norm(v[index].astype(np.float64) - a[zero_index_unadjusted].astype(np.float64)))
        zero_outcomes = []
        for delta in deltas:
            shifted = zero_index_unadjusted + int(delta)
            if not 0 <= shifted < a.shape[0]:
                raise ValueError("zero-lag mismatch row is out of bounds")
            zero_outcomes.append(_win(float(np.linalg.norm(v[index].astype(np.float64) - a[shifted].astype(np.float64))), d_zero))
        zero_by_half[half].append(float(np.mean(zero_outcomes)))
        rows.append({"phase_index": int(record["phase_index"]), "phase": float(record["phase"]), "half": half, "i": index, "k0": k0, "d0": d0, "wins": outcomes, "value": value})
    if not by_half["H0"] or not by_half["H1"]:
        raise ValueError("both halves need common phase rows")
    R0 = float(np.mean(by_half["H0"]))
    R1 = float(np.mean(by_half["H1"]))
    Z0 = float(np.mean(zero_by_half["H0"]))
    Z1 = float(np.mean(zero_by_half["H1"]))
    return {
        "R": 0.5 * (R0 + R1),
        "R_zero": 0.5 * (Z0 + Z1),
        "half_means": {"H0": R0, "H1": R1},
        "zero_half_means": {"H0": Z0, "H1": Z1},
        "phase_counts": {"H0": len(by_half["H0"]), "H1": len(by_half["H1"])},
        "rows": rows,
    }


def official_c_interior(matrix: np.ndarray) -> dict[str, Any]:
    """Compute the descriptive C/B/D curve on rows with all 31 lags valid."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or not np.isfinite(value).all():
        raise ValueError(f"official distance matrix must be [F,31], got {value.shape}")
    rows = np.arange(15, value.shape[0] - 15, dtype=np.int64)
    if rows.size < 1:
        return {"status": "INSUFFICIENT_ROWS", "rows": [], "C": None, "B": None, "D": None, "curve": []}
    curve = np.mean(value[rows], axis=0)
    B = float(np.median(curve))
    D = float(np.min(curve))
    return {"status": "COMPLETE", "rows": [int(x) for x in rows], "C": B - D, "B": B, "D": D, "curve": [float(x) for x in curve]}


def bootstrap_summary(values: Sequence[float], *, primary: bool = False, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return {"status": "NOT_ESTIMABLE", "n": int(array.size), "values": []}
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, array.size, size=(int(draws), array.size), endpoint=False)
    means = np.mean(array[indices], axis=1)
    lower, upper = (1.25, 98.75) if primary else (2.5, 97.5)
    ci = np.quantile(means, [lower / 100.0, upper / 100.0], method="linear")
    return {
        "status": "COMPLETE",
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95": [float(ci[0]), float(ci[1])],
        "ci_percentiles": [float(lower), float(upper)],
        "positive_count": int(np.sum(array > 0.0)),
        "zero_count": int(np.sum(array == 0.0)),
        "negative_count": int(np.sum(array < 0.0)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "values": [float(x) for x in array],
        "draws": int(draws),
        "seed": int(seed),
    }


def status_from_ci(summary: Mapping[str, Any]) -> str:
    if summary.get("status") != "COMPLETE":
        return "INCONCLUSIVE"
    lo, hi = [float(x) for x in summary["ci95"]]
    if lo > 0.0:
        return "POSITIVE"
    if hi < 0.0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def chance_status(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    summary = bootstrap_summary(np.asarray(values, dtype=np.float64) - 0.5, primary=False, seed=seed)
    if summary.get("status") != "COMPLETE":
        return {"status": "INCONCLUSIVE", "summary": summary}
    lo, _hi = summary["ci95"]
    return {"status": "ALL_ABOVE_CHANCE" if lo > 0.0 else "INCONCLUSIVE", "summary": summary}
