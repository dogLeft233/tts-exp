from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, file_sha256, finite_array

METRICS = ("b_dynamic", "q_natural", "q_candidate")
BOOTSTRAP_COUNT = 20_000
BOOTSTRAP_SEED = 20260910
PRACTICAL_THRESHOLD = 0.02


def _step(timestamps: np.ndarray, fallback: float) -> float:
    if timestamps.size <= 1:
        return float(fallback)
    delta = np.diff(timestamps)
    if not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ProtocolError("timestamps are not strictly increasing")
    return float(np.median(delta))


def load_feature(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"canonical_mouth", "valid", "timestamps_s"}
            if not required.issubset(data.files):
                raise ProtocolError(f"feature fields missing: {path}")
            mouth = np.asarray(data["canonical_mouth"], dtype=np.float64)
            valid = np.asarray(data["valid"], dtype=bool)
            timestamps = np.asarray(data["timestamps_s"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read feature: {path}") from exc
    if (
        mouth.ndim != 3
        or mouth.shape[1:] != (31, 2)
        or valid.shape != (mouth.shape[0],)
        or timestamps.shape != valid.shape
    ):
        raise ProtocolError(f"feature shape mismatch: {path}")
    finite_array(mouth, "canonical_mouth")
    finite_array(timestamps, "timestamps")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ProtocolError(f"timestamps are not strictly increasing: {path}")
    if np.any(~valid & (np.abs(mouth).sum(axis=(1, 2)) > 0)):
        raise ProtocolError(f"invalid feature rows are not zero: {path}")
    return {"mouth": mouth, "valid": valid, "timestamps": timestamps}


def exact_matches(
    g: Mapping[str, np.ndarray], x: Mapping[str, np.ndarray]
) -> list[tuple[int, int, bool]]:
    """Reproduce the parent's monotonic timestamp pairing."""
    gt = np.asarray(g["timestamps"], dtype=np.float64)
    xt = np.asarray(x["timestamps"], dtype=np.float64)
    gstep = _step(gt, 0.02)
    xstep = _step(xt, gstep if xt.size <= 1 else 0.02)
    tolerance = min(gstep, xstep) * 0.5
    gi = xi = 0
    result: list[tuple[int, int, bool]] = []
    while gi < gt.size and xi < xt.size:
        delta = float(xt[xi] - gt[gi])
        if abs(delta) <= tolerance:
            result.append((gi, xi, bool(g["valid"][gi] and x["valid"][xi])))
            gi += 1
            xi += 1
        elif delta < -tolerance:
            xi += 1
        else:
            gi += 1
    return result


def exact_metric(
    g: Mapping[str, np.ndarray], x: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    matches = exact_matches(g, x)
    distances = [
        float(np.sqrt(np.mean((g["mouth"][i] - x["mouth"][j]) ** 2, dtype=np.float64)))
        for i, j, ok in matches
        if ok
    ]
    denominator = max(len(g["valid"]), len(x["valid"]))
    return {
        "distance": float(np.median(np.asarray(distances, dtype=np.float64)))
        if distances
        else None,
        "coverage": float(len(distances) / denominator) if denominator else 0.0,
        "matched_count": len(distances),
        "match_count": len(matches),
    }


def _old_eligibility(
    record: Mapping[str, Any],
    exact: Mapping[str, Mapping[str, Any]],
    features: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[bool, dict[str, bool]]:
    fractions = {
        arm: float(np.mean(features[arm]["valid"]))
        for arm in ("real", "natural", "candidate")
    }
    checks = {
        "real_valid_fraction": fractions["real"] >= config.VALID_FRACTION_MIN,
        "natural_valid_fraction": fractions["natural"] >= config.VALID_FRACTION_MIN,
        "candidate_valid_fraction": fractions["candidate"] >= config.VALID_FRACTION_MIN,
        "natural_coverage": float(exact["natural"]["coverage"]) >= config.COVERAGE_MIN,
        "candidate_coverage": float(exact["candidate"]["coverage"])
        >= config.COVERAGE_MIN,
        "primary_finite": exact["natural"]["distance"] is not None
        and exact["candidate"]["distance"] is not None,
    }
    return bool(all(checks.values())), checks


def _common_support(
    features: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gn = {
        i: j for i, j, ok in exact_matches(features["real"], features["natural"]) if ok
    }
    gm = {
        i: j
        for i, j, ok in exact_matches(features["real"], features["candidate"])
        if ok
    }
    common_g = sorted(set(gn).intersection(gm))
    return (
        np.asarray(common_g, dtype=np.int64),
        np.asarray([gn[i] for i in common_g], dtype=np.int64),
        np.asarray([gm[i] for i in common_g], dtype=np.int64),
    )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + 1e-12)) if denominator else 0.0


def _decomposition(
    features: Mapping[str, Mapping[str, np.ndarray]],
    gi: np.ndarray,
    ni: np.ndarray,
    mi: np.ndarray,
) -> dict[str, float]:
    g = features["real"]["mouth"][gi].astype(np.float64, copy=False)
    n = features["natural"]["mouth"][ni].astype(np.float64, copy=False)
    m = features["candidate"]["mouth"][mi].astype(np.float64, copy=False)
    mu_g, mu_n, mu_m = (np.mean(x, axis=0, dtype=np.float64) for x in (g, n, m))
    gc, nc, mc = g - mu_g, n - mu_n, m - mu_m
    total_n = float(np.mean((g - n) ** 2, dtype=np.float64))
    total_m = float(np.mean((g - m) ** 2, dtype=np.float64))
    static_n = float(np.mean((mu_g - mu_n) ** 2, dtype=np.float64))
    static_m = float(np.mean((mu_g - mu_m) ** 2, dtype=np.float64))
    dynamic_n = float(np.mean((gc - nc) ** 2, dtype=np.float64))
    dynamic_m = float(np.mean((gc - mc) ** 2, dtype=np.float64))
    reverse_n = float(np.mean((gc - nc[::-1]) ** 2, dtype=np.float64))
    reverse_m = float(np.mean((gc - mc[::-1]) ** 2, dtype=np.float64))
    return {
        "e_total_natural": total_n,
        "e_total_candidate": total_m,
        "e_static_natural": static_n,
        "e_static_candidate": static_m,
        "e_dynamic_natural": dynamic_n,
        "e_dynamic_candidate": dynamic_m,
        "e_reverse_natural": reverse_n,
        "e_reverse_candidate": reverse_m,
        "b_dynamic": _ratio(dynamic_n - dynamic_m, dynamic_n + dynamic_m),
        "q_natural": _ratio(reverse_n - dynamic_n, reverse_n + dynamic_n),
        "q_candidate": _ratio(reverse_m - dynamic_m, reverse_m + dynamic_m),
        "identity_error_natural": total_n - static_n - dynamic_n,
        "identity_error_candidate": total_m - static_m - dynamic_m,
    }


def analyze_record(record: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        arm: Path(str(record["feature_paths"][arm]))
        for arm in ("real", "natural", "candidate")
    }
    features = {arm: load_feature(path) for arm, path in paths.items()}
    exact = {
        arm: exact_metric(features["real"], features[arm])
        for arm in ("natural", "candidate")
    }
    old_eligible, checks = _old_eligibility(record, exact, features)
    gi, ni, mi = _common_support(features)
    denominator = max(*(len(features[arm]["valid"]) for arm in paths))
    common_coverage = float(gi.size / denominator) if denominator else 0.0
    observed = bool(
        bool(record.get("eligibility", {}).get("eligible", False))
        and old_eligible
        and common_coverage >= config.COVERAGE_MIN
        and gi.size > 0
    )
    missing_reasons: list[str] = []
    if (
        not bool(record.get("eligibility", {}).get("eligible", False))
        or not old_eligible
    ):
        missing_reasons.append("parent_eligibility_false")
    if common_coverage < config.COVERAGE_MIN:
        missing_reasons.append("common_coverage_below_0_85")
    if gi.size == 0:
        missing_reasons.append("empty_common_support")
    row: dict[str, Any] = {
        "sample_id": str(record["sample_id"]),
        "source_group": str(record["source_group"]),
        "parent_eligibility": bool(
            record.get("eligibility", {}).get("eligible", False)
        ),
        "recomputed_eligibility": old_eligible,
        "eligibility_checks": checks,
        "exact_time": exact,
        "common_support": {
            "real_indices": gi.tolist(),
            "natural_indices": ni.tolist(),
            "candidate_indices": mi.tolist(),
            "count": int(gi.size),
            "coverage": common_coverage,
        },
        "observed": observed,
        "missing_reasons": missing_reasons,
        "feature_hashes": {arm: file_sha256(path) for arm, path in paths.items()},
    }
    if gi.size:
        row.update(_decomposition(features, gi, ni, mi))
        for total_key, error_key in (
            ("e_total_natural", "identity_error_natural"),
            ("e_total_candidate", "identity_error_candidate"),
        ):
            if abs(row[error_key]) > 1e-10 * max(1.0, row[total_key]):
                raise ProtocolError(
                    f"MSE decomposition identity failed: {record['sample_id']}"
                )
        jtimes = features["real"]["timestamps"][gi]
        row["max_original_time_gap_s"] = (
            float(np.max(np.diff(jtimes))) if jtimes.size > 1 else 0.0
        )
        # The historical H endpoint was the median distance on common support.
        # Keep exact_time separate so the old bound can be reproduced byte-for-byte.
        row["d_n"] = float(
            np.median(
                np.sqrt(
                    np.mean(
                        (
                            features["real"]["mouth"][gi]
                            - features["natural"]["mouth"][ni]
                        )
                        ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
        )
        row["d_m"] = float(
            np.median(
                np.sqrt(
                    np.mean(
                        (
                            features["real"]["mouth"][gi]
                            - features["candidate"]["mouth"][mi]
                        )
                        ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                )
            )
        )
        row["b"] = (
            _ratio(float(row["d_n"] - row["d_m"]), float(row["d_n"] + row["d_m"]))
            if observed and row["d_n"] is not None and row["d_m"] is not None
            else None
        )
    else:
        for key in (
            "d_n",
            "d_m",
            "b",
            *METRICS,
            "e_total_natural",
            "e_total_candidate",
            "e_static_natural",
            "e_static_candidate",
            "e_dynamic_natural",
            "e_dynamic_candidate",
            "e_reverse_natural",
            "e_reverse_candidate",
            "identity_error_natural",
            "identity_error_candidate",
        ):
            row[key] = None
        row["max_original_time_gap_s"] = 0.0
    if not observed:
        for key in METRICS:
            row[key] = None
    return row


def group_bounds(
    rows: list[dict[str, Any]], groups: list[str], metric: str = "b"
) -> tuple[list[dict[str, Any]], float, float]:
    by_group: dict[str, list[float]] = {group: [] for group in groups}
    for row in rows:
        value = row.get(metric)
        if row["observed"] and value is not None:
            by_group[str(row["source_group"])].append(float(value))
    result: list[dict[str, Any]] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    for group in groups:
        observed = by_group[group]
        n = sum(1 for row in rows if str(row["source_group"]) == group)
        missing = n - len(observed)
        total = (
            float(np.sum(np.asarray(observed, dtype=np.float64))) if observed else 0.0
        )
        lower = float((total - missing) / n) if n else -1.0
        upper = float((total + missing) / n) if n else 1.0
        result.append(
            {
                "source_group": group,
                "n": n,
                "observed": len(observed),
                "missing": missing,
                "sum_observed": total,
                "L_g": lower,
                "U_g": upper,
            }
        )
        lower_values.append(lower)
        upper_values.append(upper)
    return result, float(np.mean(lower_values)), float(np.mean(upper_values))


def _observed_statistics(
    rows: list[dict[str, Any]], groups: list[str]
) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {g: [] for g in groups}
    for row in rows:
        if row["observed"]:
            by_group[str(row["source_group"])].append(row)
    active = [g for g in groups if by_group[g]]
    group_values = np.asarray(
        [
            [
                float(np.mean([r[m] for r in by_group[g]], dtype=np.float64))
                for m in METRICS
            ]
            for g in active
        ],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = (
        rng.integers(
            0, len(active), size=(BOOTSTRAP_COUNT, len(active)), dtype=np.int64
        )
        if active
        else np.empty((BOOTSTRAP_COUNT, 0), dtype=np.int64)
    )
    samples = (
        group_values[indices].mean(axis=1)
        if active
        else np.zeros((BOOTSTRAP_COUNT, len(METRICS)), dtype=np.float64)
    )
    output: dict[str, Any] = {
        "label": "observed_subset",
        "group_count": len(active),
        "record_count": sum(len(by_group[g]) for g in active),
        "excluded_groups": [g for g in groups if g not in active],
        "bootstrap_count": BOOTSTRAP_COUNT,
        "seed": BOOTSTRAP_SEED,
        "quantile_method": "linear",
        "groups": [
            {
                "source_group": g,
                "n": len(by_group[g]),
                **{
                    m: float(np.mean([r[m] for r in by_group[g]], dtype=np.float64))
                    for m in METRICS
                },
            }
            for g in active
        ],
    }
    for i, metric in enumerate(METRICS):
        values = group_values[:, i] if active else np.asarray([], dtype=np.float64)
        boot = samples[:, i]
        ci = (
            [
                float(np.quantile(boot, 0.005, method="linear")),
                float(np.quantile(boot, 0.995, method="linear")),
            ]
            if active
            else [None, None]
        )
        output[metric] = {
            "mean": float(np.mean(values)) if values.size else None,
            "positive_groups": int(np.sum(values > 0)),
            "total_groups": len(active),
            "ci99": ci,
            "passes": bool(
                active
                and ci[0] > 0
                and np.mean(values) > PRACTICAL_THRESHOLD
                and np.sum(values > 0) >= 17
            ),
        }
    return output


def analyze_records(
    records: list[dict[str, Any]],
    groups: list[str],
    split: Mapping[str, list[str]],
    history_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [analyze_record(record) for record in records]
    old_groups, old_lower, old_upper = group_bounds(rows, groups, "b")
    bounds: dict[str, Any] = {}
    group_outputs: dict[str, Any] = {}
    for metric in METRICS:
        gr, lower, upper = group_bounds(rows, groups, metric)
        group_outputs[metric] = gr
        bounds[metric] = {
            "L": lower,
            "U": upper,
            "interpretation": "all 23 groups equally weighted; missing values bounded in [-1,1]",
        }
    observed = _observed_statistics(rows, groups)
    timing_pass = bool(
        observed["q_natural"]["passes"] and observed["q_candidate"]["passes"]
    )
    dynamic_pass = bool(observed["b_dynamic"]["passes"])
    all_lower_positive = all(bounds[m]["L"] > 1e-12 for m in METRICS)
    if not all(row["feature_hashes"] for row in rows):
        decision = "BLOCKED"
    elif not timing_pass:
        decision = "TIMING_SPECIFICITY_NOT_ESTABLISHED"
    elif not dynamic_pass:
        decision = "NO_OBSERVED_DYNAMIC_ADVANTAGE_ESTABLISHED"
    elif all_lower_positive:
        decision = "CACHED_DYNAMIC_SIGNAL_ROBUST_TO_MISSINGNESS"
    else:
        decision = "OBSERVED_DYNAMIC_SIGNAL_REQUIRES_NEW_COHORT"
    legacy_reproduction: dict[str, Any] = {
        "observed_count": sum(bool(r["observed"]) for r in rows),
        "record_count": len(rows),
        "group_count": len(groups),
        "L": old_lower,
        "U": old_upper,
        "group_rows": old_groups,
    }
    if history_analysis is not None:
        legacy_reproduction["history_observed_count"] = int(
            history_analysis.get("observed_count", -1)
        )
        legacy_reproduction["history_L"] = float(
            history_analysis.get("bounds", {}).get("L", np.nan)
        )
        legacy_reproduction["history_U"] = float(
            history_analysis.get("bounds", {}).get("U", np.nan)
        )
        legacy_reproduction["max_abs_LU_error"] = max(
            abs(old_lower - legacy_reproduction["history_L"]),
            abs(old_upper - legacy_reproduction["history_U"]),
        )
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["missing_reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "schema_version": 2,
        "record_count": len(rows),
        "group_count": len(groups),
        "observed_count": sum(bool(row["observed"]) for row in rows),
        "missing_reason_counts": counts,
        "records": rows,
        "groups": group_outputs,
        "bounds": bounds,
        "observed_subset": observed,
        "legacy_reproduction": legacy_reproduction,
        "diagnostic_decision": decision,
        "engineering_decision": "GO",
        "parent_gate_repaired": False,
        "stage02_authorized": False,
        "replacement_confirmed": False,
        "waveform_head_authorized": False,
        "generalization_established": False,
        "model_calls": 0,
        "new_videos": 0,
        "new_scores": 0,
    }
