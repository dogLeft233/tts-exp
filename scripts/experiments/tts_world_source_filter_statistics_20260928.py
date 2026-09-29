"""Draft, score-free statistics for the WORLD EX/SP paired experiment.

This module reads no experiment files and performs no scoring. The formal run
must bind its input schema, support, code hash, and reviewer receipt before use.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FACTORS = ("EX", "SP", "BOTH")
RAW_DOMAIN_CONDITIONS = ("ID", "ROUNDTRIP", *FACTORS)
ARMS = ("N", "T")
GEOMETRIES = ("raw", "unit")
POLICIES = ("guard20", "valid", "guard0", "guard15")
ENDPOINTS = (
    "C", "B", "D", "C_anchor", "D_anchor", "search_uplift", "best_lag", "offset"
)
DRAW_COUNT = 20_000
BOOTSTRAP_SEED = 20260926
REQUIRED_SHAM = ("ROUNDTRIP",)


def _number(value: Any, path: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite score: {path}")
    return result


def pair_terms(
    cells: Mapping[str, Any],
    geometry: str,
    policy: str,
    endpoint: str,
    sham_conditions: Sequence[str] = REQUIRED_SHAM,
) -> dict[str, float]:
    """Return all predetermined scalar contrasts for one same-text N/T pair."""
    if geometry not in GEOMETRIES or policy not in POLICIES or endpoint not in ENDPOINTS:
        raise ValueError("unknown scoring view or endpoint")

    def value(arm: str, condition: str, q: str | None = None) -> float:
        source = cells[arm][condition]
        if q is not None:
            source = source[q]
        path = f"{arm}/{condition}/{q or 'joint'}/{geometry}/{policy}/{endpoint}"
        return _number(source[geometry][policy][endpoint], path)

    native = {arm: value(arm, "RAW") for arm in ARMS}
    identity = {arm: value(arm, "ID") for arm in ARMS}
    result = {
        "N/RAW": native["N"],
        "T/RAW": native["T"],
        "N/ID": identity["N"],
        "T/ID": identity["T"],
        "native_gap": native["T"] - native["N"],
        "id_gap": identity["T"] - identity["N"],
        "bridge_gap": (identity["T"] - identity["N"]) - (native["T"] - native["N"]),
        "N/id_minus_raw": identity["N"] - native["N"],
        "T/id_minus_raw": identity["T"] - native["T"],
    }

    for sham in sham_conditions:
        if sham not in ("SELF_SHIFT", "ROUNDTRIP"):
            raise ValueError(f"unknown sham condition: {sham}")
        for arm in ARMS:
            observed = value(arm, sham)
            result[f"{arm}/{sham}"] = observed
            result[f"{arm}/{sham}_minus_id"] = observed - identity[arm]

    swapped_joint: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for factor in FACTORS:
            q00 = identity[arm]
            q10 = value(arm, factor, "q10")
            q01 = value(arm, factor, "q01")
            q11 = value(arm, factor, "q11")
            swapped_joint[arm][factor] = q11
            prefix = f"{arm}/{factor}"
            result[f"{prefix}/q10"] = q10
            result[f"{prefix}/q01"] = q01
            result[f"{prefix}/q11"] = q11
            result[f"{prefix}/G"] = q10 - q00
            result[f"{prefix}/E"] = q01 - q00
            result[f"{prefix}/I"] = q11 - q10 - q01 + q00
            result[f"{prefix}/total"] = q11 - q00

        for path in ("q10", "q01", "q11"):
            result[f"{arm}/EX_SP_interaction/{path}"] = (
                result[f"{arm}/BOTH/{path}"]
                - result[f"{arm}/EX/{path}"]
                - result[f"{arm}/SP/{path}"]
                + identity[arm]
            )

    for arm in ARMS:
        raw0 = native[arm]
        for condition in RAW_DOMAIN_CONDITIONS:
            def raw_cell(q: str) -> float:
                path = f"{arm}/raw_fourcell/{condition}/{q}/{geometry}/{policy}/{endpoint}"
                return _number(cells[arm]["raw_fourcell"][condition][q][geometry][policy][endpoint], path)

            q10, q01, q11 = (raw_cell(q) for q in ("q10", "q01", "q11"))
            joint = identity[arm] if condition == "ID" else (
                value(arm, "ROUNDTRIP") if condition == "ROUNDTRIP" else swapped_joint[arm][condition]
            )
            if abs(q11 - joint) > 1e-10:
                raise ValueError(f"RAW-anchor q11 must equal joint {arm}/{condition}")
            prefix = f"{arm}/RAW_ANCHOR/{condition}"
            result[f"{prefix}/q10"] = q10
            result[f"{prefix}/q01"] = q01
            result[f"{prefix}/q11"] = q11
            result[f"{prefix}/G"] = q10 - raw0
            result[f"{prefix}/E"] = q01 - raw0
            result[f"{prefix}/I"] = q11 - q10 - q01 + raw0
            result[f"{prefix}/total"] = q11 - raw0
            closure = result[f"{prefix}/G"] + result[f"{prefix}/E"] + result[f"{prefix}/I"]
            if abs(closure - result[f"{prefix}/total"]) > 1e-10:
                raise ValueError(f"RAW-anchor four-cell closure failed: {prefix}")

        for condition in RAW_DOMAIN_CONDITIONS[1:]:
            for effect in ("G", "E", "I", "total"):
                result[f"{arm}/RAW_ANCHOR/{condition}_minus_ID/{effect}"] = (
                    result[f"{arm}/RAW_ANCHOR/{condition}/{effect}"]
                    - result[f"{arm}/RAW_ANCHOR/ID/{effect}"]
                )

    for factor in FACTORS:
        natural = swapped_joint["N"][factor]
        tts = swapped_joint["T"][factor]
        delta_id = result["id_gap"]
        delta_swap = tts - natural
        result[f"{factor}/N_gain"] = natural - identity["N"]
        result[f"{factor}/T_loss"] = identity["T"] - tts
        result[f"{factor}/R_T"] = identity["T"] - natural
        result[f"{factor}/R_N"] = tts - identity["N"]
        result[f"{factor}/swap_gap"] = delta_swap
        result[f"{factor}/transfer_component"] = (delta_id - delta_swap) / 2

    for arm in ARMS:
        for factor in FACTORS:
            prefix = f"{arm}/{factor}"
            closure = result[f"{prefix}/G"] + result[f"{prefix}/E"] + result[f"{prefix}/I"]
            if abs(closure - result[f"{prefix}/total"]) > 1e-10:
                raise ValueError(f"four-cell closure failed: {prefix}")
    return result


def speaker_bootstrap(values: Sequence[float], speakers: Sequence[str]) -> dict[str, Any]:
    """Equal-speaker estimate and fixed 20k PCG64 percentile intervals."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) != len(speakers) or not len(data):
        raise ValueError("values and speaker labels must be nonempty and paired")
    if not np.isfinite(data).all():
        raise ValueError("non-finite contrast")
    groups: dict[str, list[float]] = defaultdict(list)
    for value, speaker in zip(data, speakers, strict=True):
        if not speaker:
            raise ValueError("empty speaker label")
        groups[str(speaker)].append(float(value))
    labels = sorted(groups)
    means = np.array([np.mean(groups[label]) for label in labels], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(labels), (DRAW_COUNT, len(labels)))
    draws = means[indices].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
        "ci99": np.quantile(draws, [0.005, 0.995], method="linear").tolist(),
        "n_pairs": len(data),
        "n_speakers": len(labels),
        "group_means": dict(zip(labels, means.tolist(), strict=True)),
        "draws": DRAW_COUNT,
        "seed": BOOTSTRAP_SEED,
        "rng": "PCG64",
        "quantile_method": "linear",
    }


def summarize_view(
    records: Sequence[Mapping[str, Any]],
    support_ids: Sequence[str],
    support_name: str,
    geometry: str,
    policy: str,
    endpoint: str,
    sham_conditions: Sequence[str] = REQUIRED_SHAM,
) -> dict[str, dict[str, Any]]:
    """Summarize a prespecified common support; never choose rows from scores."""
    by_id = {str(row["id"]): row for row in records}
    if len(by_id) != len(records) or len(set(support_ids)) != len(support_ids):
        raise ValueError("duplicate score or support ID")
    if not support_ids or any(sample_id not in by_id for sample_id in support_ids):
        raise ValueError("frozen support contains missing score records")
    support_hash = hashlib.sha256("\n".join(support_ids).encode()).hexdigest()
    rows = [by_id[sample_id] for sample_id in support_ids]
    speakers = [str(row["speaker"]) for row in rows]
    paired = [pair_terms(row["cells"], geometry, policy, endpoint, sham_conditions) for row in rows]
    expected = set(paired[0])
    if any(set(row) != expected for row in paired[1:]):
        raise ValueError("contrast keys differ across the frozen support")
    output = {}
    for name in sorted(expected):
        stat = speaker_bootstrap([row[name] for row in paired], speakers)
        stat.update({
            "support_name": support_name,
            "support_ids_sha256": support_hash,
            "geometry": geometry,
            "policy": policy,
            "endpoint": endpoint,
            "contrast": name,
        })
        output[name] = stat
    return output


def adjudicate_primary(
    summary: Mapping[str, Mapping[str, Any]], frozen_common71_ids: Sequence[str]
) -> dict[str, Any]:
    """Apply only the fixed raw/guard20/common71 C criteria to its summary."""
    if len(frozen_common71_ids) != 71 or len(set(frozen_common71_ids)) != 71:
        raise ValueError("primary requires the frozen 71 unique pair IDs")
    support_hash = hashlib.sha256("\n".join(frozen_common71_ids).encode()).hexdigest()
    for name, stat in summary.items():
        expected = {
            "support_name": "common71",
            "support_ids_sha256": support_hash,
            "geometry": "raw",
            "policy": "guard20",
            "endpoint": "C",
            "contrast": name,
            "n_pairs": 71,
            "n_speakers": 15,
        }
        if any(stat.get(key) != value for key, value in expected.items()):
            raise ValueError(f"primary support/view mismatch: {name}")
    bidirectional = {
        factor: all(summary[f"{factor}/{key}"]["ci99"][0] > 0 for key in ("N_gain", "T_loss"))
        for factor in FACTORS
    }

    def inside(name: str, limit: float) -> bool:
        low, high = summary[name]["ci95"]
        return low >= -limit and high <= limit

    return {
        "bidirectional_99": bidirectional,
        "both_residual_precision_95": {
            name: inside(f"BOTH/{name}", 0.05) for name in ("R_T", "R_N")
        },
        "native_bridge_precision_95": {
            "gap": inside("bridge_gap", 0.05),
            "N_arm": inside("N/id_minus_raw", 0.20),
            "T_arm": inside("T/id_minus_raw", 0.20),
        },
        "scope": "common71/raw/guard20/C only; historical-sample mechanism test",
    }
