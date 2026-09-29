"""Pure statistics and protocol helpers for the Ditto-50 linkage experiment.

This module deliberately contains no model or filesystem code.  Keeping the
pairing and inference-free statistics here makes the producer and the
independent validator easier to audit.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


BOOTSTRAP_SEED = 20260918
PERMUTATION_SEED = 20260919
CORRELATION_BOOTSTRAP_SEED = 20260920
BOOTSTRAP_DRAWS = 20_000
MIN_JOINT_PAIRS = 40
VIEWS = ("native", "frozen", "reversed", "matched", "matched_frozen")


def freeze_decoys(records: Sequence[Mapping[str, Any]], count: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Freeze five unique, length-matched decoys for every valid target.

    Duplicate normalised texts are represented by the smallest numeric id.
    The candidate pool is formed before any video/model result is inspected.
    """

    canonical: Dict[str, Tuple[int, str, Optional[List[int]]]] = {}
    for item in records:
        sid = int(item["id"])
        text = str(item.get("normalized_text", ""))
        tokens = item.get("target_token_ids", item.get("token_ids"))
        token_values = None if tokens is None else [int(x) for x in tokens]
        previous = canonical.get(text)
        if previous is None or sid < previous[0]:
            canonical[text] = (sid, text, token_values)
    prepared = sorted(canonical.values(), key=lambda row: row[0])
    output: Dict[str, List[Dict[str, Any]]] = {}
    for sid, text, tokens in prepared:
        if tokens is None:
            output[str(sid)] = []
            continue
        candidates = []
        for other_sid, other_text, other_tokens in prepared:
            if other_sid == sid or other_tokens is None:
                continue
            candidates.append((abs(len(other_text) - len(text)), other_sid, other_text, other_tokens))
        candidates.sort(key=lambda row: (row[0], row[1]))
        output[str(sid)] = [
            {"id": int(other_sid), "normalized_text": other_text, "token_ids": list(other_tokens)}
            for _, other_sid, other_text, other_tokens in candidates[: int(count)]
        ]
    # Include records whose text was a duplicate but which were not retained,
    # so callers never silently lose an audited id.
    for item in records:
        output.setdefault(str(int(item["id"])), [])
    return output


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rank(values: Sequence[float]) -> List[float]:
    indexed = sorted((float(value), index) for index, value in enumerate(values))
    result = [0.0] * len(indexed)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][0] == indexed[cursor][0]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            result[indexed[position][1]] = rank
        cursor = end
    return result


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x = x - float(x.mean())
    y = y - float(y.mean())
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-15:
        return None
    return float(np.dot(x, y) / denominator)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys):
        return None
    return pearson(_rank(xs), _rank(ys))


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, q, interpolation="linear"))


def bootstrap_mean(values: Sequence[float], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        return {"status": "NOT_ESTIMABLE", "count": int(len(array)), "reason": "empty_or_nonfinite"}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)), dtype=np.int64)
    means = array[indices].mean(axis=1, dtype=np.float64)
    return {
        "status": "COMPLETE",
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95": [_quantile(means, 0.025), _quantile(means, 0.975)],
        "positive_count": int(np.sum(array > 0.0)),
        "zero_count": int(np.sum(array == 0.0)),
        "negative_count": int(np.sum(array < 0.0)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "values": [float(value) for value in array],
        "draws": int(draws),
        "seed": int(seed),
    }


def _correlation_bootstrap(xs: Sequence[float], ys: Sequence[float], draws: int, seed: int) -> Dict[str, Any]:
    if len(xs) != len(ys) or len(xs) < 2:
        return {"status": "NOT_ESTIMABLE", "count": len(xs), "reason": "length_or_sample_count"}
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    observed = spearman(x.tolist(), y.tolist())
    if observed is None:
        return {"status": "NOT_ESTIMABLE", "count": len(x), "rho": None, "ci95": None, "reason": "constant_input"}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(x), size=(int(draws), len(x)), dtype=np.int64)
    values: List[float] = []
    for row in indices:
        value = spearman(x[row].tolist(), y[row].tolist())
        if value is not None and math.isfinite(value):
            values.append(float(value))
    result: Dict[str, Any] = {
        "status": "COMPLETE" if len(values) >= int(draws * 0.95) else "INSUFFICIENT_NONDEGENERATE_DRAWS",
        "count": int(len(x)),
        "rho": float(observed),
        "valid_draws": int(len(values)),
        "draws": int(draws),
        "seed": int(seed),
    }
    result["ci95"] = [_quantile(np.asarray(values), 0.025), _quantile(np.asarray(values), 0.975)] if values else None
    if result["status"] != "COMPLETE":
        result["reason"] = "too_many_degenerate_bootstrap_draws"
    return result


def linkage_statistics(rows: Sequence[Mapping[str, Any]], draws: int = BOOTSTRAP_DRAWS) -> Dict[str, Any]:
    """Compute the pre-registered linkage and 2x2 summaries on one J set."""

    ordered = sorted(rows, key=lambda row: int(row["id"]))
    metrics = ("g", "b", "gmatched", "r_natural", "r_tts", "delta_c", "delta_d", "delta_cer")
    summaries: Dict[str, Any] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in ordered if _finite(row.get(metric))]
        summaries[metric] = bootstrap_mean(values, draws=draws, seed=BOOTSTRAP_SEED)

    g = [float(row["g"]) for row in ordered]
    delta_c = [float(row["delta_c"]) for row in ordered]
    rho = spearman(g, delta_c)
    permutation: Dict[str, Any]
    if rho is None:
        permutation = {"status": "NOT_ESTIMABLE", "rho": None, "reason": "constant_input", "draws": int(draws), "seed": PERMUTATION_SEED}
    else:
        rng = np.random.default_rng(PERMUTATION_SEED)
        y = np.asarray(delta_c, dtype=np.float64)
        permutation_values = []
        for _ in range(int(draws)):
            shuffled = rng.permutation(y)
            value = spearman(g, shuffled.tolist())
            if value is not None:
                permutation_values.append(abs(float(value)))
        exceed = int(sum(value >= abs(float(rho)) - 1e-12 for value in permutation_values))
        permutation = {
            "status": "COMPLETE",
            "rho": float(rho),
            "exceed_count": exceed,
            "p_two_sided": float((1 + exceed) / (int(draws) + 1)),
            "draws": int(draws),
            "seed": PERMUTATION_SEED,
        }
    correlation = _correlation_bootstrap(g, delta_c, draws, CORRELATION_BOOTSTRAP_SEED)
    sensitivity: Dict[str, Any] = {}
    for left in ("b", "gmatched"):
        x = [float(row[left]) for row in ordered]
        sensitivity["spearman_{}_delta_c".format(left)] = spearman(x, delta_c)
    sensitivity["pearson_g_delta_c"] = pearson(g, delta_c)

    g_positive = [value > 0.0 for value in g]
    c_positive = [value > 0.0 for value in delta_c]
    table = {
        "rows": {"G>0": {"deltaC>0": 0, "deltaC<=0": 0}, "G<=0": {"deltaC>0": 0, "deltaC<=0": 0}},
        "n": len(ordered),
    }
    for gp, cp in zip(g_positive, c_positive):
        table["rows"]["G>0" if gp else "G<=0"]["deltaC>0" if cp else "deltaC<=0"] += 1
    n = len(ordered)
    g_count = int(sum(g_positive))
    c_count = int(sum(c_positive))
    both = int(sum(gp and cp for gp, cp in zip(g_positive, c_positive)))
    p_g = g_count / n if n else None
    p_c = c_count / n if n else None
    table.update({
        "g_positive_count": g_count,
        "delta_c_positive_count": c_count,
        "both_positive_count": both,
        "only_vsr_positive_count": int(sum(gp and not cp for gp, cp in zip(g_positive, c_positive))),
        "only_sync_positive_count": int(sum((not gp) and cp for gp, cp in zip(g_positive, c_positive))),
        "both_nonpositive_count": int(sum((not gp) and (not cp) for gp, cp in zip(g_positive, c_positive))),
        "expected_both_under_independence": float(n * p_g * p_c) if p_g is not None and p_c is not None else None,
        "p_g_positive_given_delta_c_positive": (both / c_count) if c_count else None,
        "p_g_positive_given_delta_c_nonpositive": ((g_count - both) / (n - c_count)) if n - c_count else None,
        "near_zero_g_count": int(sum(abs(value) <= 1e-5 for value in g)),
        "near_zero_delta_c_count": int(sum(abs(value) <= 0.001 for value in delta_c)),
    })

    return {
        "count": n,
        "ids": [int(row["id"]) for row in ordered],
        "metrics": summaries,
        "primary_association": {"spearman": permutation, "bootstrap": correlation},
        "sensitivity": sensitivity,
        "contingency": table,
    }


def visual_status(*, engineering_status: str, calibration_status: str, joint_count: int, g_summary: Mapping[str, Any], b_summary: Mapping[str, Any], matched_summary: Mapping[str, Any]) -> str:
    if engineering_status != "PASS":
        return "TECHNICAL_INVALID"
    if int(joint_count) < MIN_JOINT_PAIRS:
        return "INSUFFICIENT_JOINT_PAIRS"
    if calibration_status != "CALIBRATED_ON_JOINT_COHORT":
        return "INCONCLUSIVE_VSR_VALIDITY"
    g_mean = g_summary.get("mean")
    b_mean = b_summary.get("mean")
    gm_mean = matched_summary.get("mean")
    if g_mean is None or b_mean is None or gm_mean is None:
        return "INCONCLUSIVE_PATTERN"
    if g_mean > 0.0 and b_mean <= 0.0:
        return "CONTROL_DRIVEN_DIFFERENCE"
    if g_mean > 0.0 and gm_mean <= 0.0:
        return "DURATION_SENSITIVE"
    ci = g_summary.get("ci95") or [None, None]
    if ci[0] is not None and ci[0] > 0.0 and b_mean > 0.0 and gm_mean > 0.0:
        return "EXPLORATORY_VISUAL_CONTENT_SUPPORT"
    if ci[1] is not None and ci[1] < 0.0:
        return "EXPLORATORY_REVERSE_EFFECT"
    if ci[0] is not None and ci[0] <= 0.0 <= ci[1]:
        return "NO_CLEAR_PAIRED_GAIN"
    return "INCONCLUSIVE_PATTERN"


def association_status(engineering_status: str, joint_count: int, calibration_status: str, association: Mapping[str, Any]) -> str:
    if engineering_status != "PASS" or int(joint_count) < MIN_JOINT_PAIRS or calibration_status != "CALIBRATED_ON_JOINT_COHORT":
        return "NOT_INTERPRETABLE"
    rho = association.get("rho")
    ci = association.get("ci95")
    p = association.get("p_two_sided")
    if rho is None or ci is None:
        return "NOT_ESTIMABLE"
    if rho > 0 and p is not None and p < 0.05 and ci[0] > 0:
        return "EXPLORATORY_POSITIVE_ASSOCIATION"
    if rho < 0 and p is not None and p < 0.05 and ci[1] < 0:
        return "EXPLORATORY_NEGATIVE_ASSOCIATION"
    return "NO_CLEAR_ASSOCIATION"


def sync_status(delta_c_summary: Mapping[str, Any]) -> str:
    """Classify the paired Sync-C mean without mixing it with VSR validity.

    This is deliberately an exploratory, interval-only label.  It does not
    require VSR calibration because SyncNet is a separate measurement chain.
    """

    if delta_c_summary.get("status") != "COMPLETE":
        return "NOT_ESTIMABLE"
    ci = delta_c_summary.get("ci95")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return "NOT_ESTIMABLE"
    try:
        lower, upper = float(ci[0]), float(ci[1])
    except (TypeError, ValueError):
        return "NOT_ESTIMABLE"
    if not math.isfinite(lower) or not math.isfinite(upper):
        return "NOT_ESTIMABLE"
    if lower > 0.0:
        return "EXPLORATORY_SYNC_GAIN"
    if upper < 0.0:
        return "EXPLORATORY_REVERSE_SYNC_GAIN"
    return "NO_CLEAR_SYNC_GAIN"


def build_pair_rows(view_rows: Sequence[Mapping[str, Any]], sync_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Join complete VSR rows with complete SyncNet rows by numeric id."""

    by_key = {(int(row["id"]), str(row["condition"]), str(row["view"])): row for row in view_rows if row.get("status") == "COMPLETE" and row.get("qc_eligible") is True}
    sync_by_key = {(int(row["id"]), str(row["condition"])): row for row in sync_rows if row.get("status") == "COMPLETE" and row.get("track_qc") is True}
    output: List[Dict[str, Any]] = []
    ids = sorted({int(row["id"]) for row in view_rows} & {int(row["id"]) for row in sync_rows})
    for sid in ids:
        required = [(sid, condition, view) for condition in ("natural", "tts") for view in VIEWS]
        if any(key not in by_key for key in required) or any((sid, condition) not in sync_by_key for condition in ("natural", "tts")):
            continue
        n = {view: by_key[(sid, "natural", view)] for view in VIEWS}
        t = {view: by_key[(sid, "tts", view)] for view in VIEWS}
        ns, ts = sync_by_key[(sid, "natural")], sync_by_key[(sid, "tts")]
        output.append({
            "id": sid,
            "g": float((t["native"]["M"] - t["frozen"]["M"]) - (n["native"]["M"] - n["frozen"]["M"])),
            "b": float(t["native"]["M"] - n["native"]["M"]),
            "gmatched": float((t["matched"]["M"] - t["matched_frozen"]["M"]) - (n["matched"]["M"] - n["matched_frozen"]["M"])),
            "r_natural": float(n["native"]["M"] - n["reversed"]["M"]),
            "r_tts": float(t["native"]["M"] - t["reversed"]["M"]),
            "delta_c": float(ts["sync_c"] - ns["sync_c"]),
            "delta_d": float(ns["sync_d"] - ts["sync_d"]),
            "delta_cer": float(n["native"].get("greedy_cer", 0.0) - t["native"].get("greedy_cer", 0.0)),
            "natural_q": float(n["native"]["M"] - n["frozen"]["M"]),
            "tts_q": float(t["native"]["M"] - t["frozen"]["M"]),
            "natural_sync_c": float(ns["sync_c"]),
            "tts_sync_c": float(ts["sync_c"]),
            "natural_sync_d": float(ns["sync_d"]),
            "tts_sync_d": float(ts["sync_d"]),
        })
    return output
