"""Draft feature-to-endpoint scorer for the WORLD EX/SP experiment.

This module has no CLI and reads no evaluation files. The formal run must first
bind a producer input seal, old RAW exact bridge, common support, and reviewer
approval. The score functions use the same frozen distance and endpoint source
as the prior 80-clip scorer; exact parity with cloud FIXED is still pending.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from scripts.experiments import tts_fixed_generator_shift_holdout80_scores_20260927 as prior
from scripts.experiments.tts_world_source_filter_statistics_20260928 import (
    FACTORS,
    GEOMETRIES,
    POLICIES,
)


CORE_CONDITIONS = ("RAW", "ID", "ROUNDTRIP", *FACTORS)
RAW_DOMAIN_CONDITIONS = ("ID", "ROUNDTRIP", *FACTORS)


def _feature(value: Any, name: str, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 1024:
        raise ValueError(f"{name} must be FLOAT32 [frames,1024]")
    if len(array) < length or not np.isfinite(array).all():
        raise ValueError(f"{name} is short or non-finite")
    return array[:length]


def score_matrix(
    visual: np.ndarray,
    audio: np.ndarray,
    length: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return 31-lag curves and endpoints for every eligible frozen view."""
    if length <= 0:
        raise ValueError("nonpositive frozen recipient support")
    v = _feature(visual, "visual", length)
    a = _feature(audio, "audio", length)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for geometry in GEOMETRIES:
        vv, aa = (prior.F["unit"](v), prior.F["unit"](a)) if geometry == "unit" else (v, a)
        matrix = prior.F["matrix"](vv, aa)
        if matrix.shape != (length, 31) or not np.isfinite(matrix).all():
            raise ValueError("distance matrix has wrong shape or non-finite entries")
        result[geometry] = {}
        for policy in POLICIES:
            if policy == "guard20" and length <= 40:
                continue
            if policy == "guard15" and length <= 30:
                continue
            endpoint = prior.F["summarize"](matrix, policy)
            endpoint["search_uplift"] = endpoint["D_anchor"] - endpoint["D"]
            endpoint["offset"] = -endpoint["best_lag"]
            if len(endpoint["curve"]) != 31:
                raise ValueError("lag curve is not 31 points")
            if not np.isclose(endpoint["C"], endpoint["B"] - endpoint["D"], atol=1e-10, rtol=0):
                raise ValueError("C = B - D closure failed")
            if not np.isclose(
                endpoint["C_anchor"], endpoint["B"] - endpoint["D_anchor"], atol=1e-10, rtol=0
            ):
                raise ValueError("anchor closure failed")
            result[geometry][policy] = endpoint
    return result


def score_recipient(
    features: Mapping[str, Mapping[str, np.ndarray]],
    frozen_joint_length: int,
    *,
    score_self_shift: bool = False,
    cached_raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one recipient; each condition shares this recipient's own clock."""
    required = set(CORE_CONDITIONS)
    if score_self_shift:
        required.add("SELF_SHIFT")
    if not required.issubset(features):
        raise ValueError(f"missing frozen features: {sorted(required - set(features))}")
    for condition in required:
        if set(features[condition]) != {"V", "A"}:
            raise ValueError(f"{condition} must provide V and A embeddings")

    cells: dict[str, Any] = {
        "RAW": dict(cached_raw) if cached_raw is not None else score_matrix(
            features["RAW"]["V"], features["RAW"]["A"], frozen_joint_length
        ),
        "ID": score_matrix(features["ID"]["V"], features["ID"]["A"], frozen_joint_length),
        "ROUNDTRIP": score_matrix(
            features["ROUNDTRIP"]["V"], features["ROUNDTRIP"]["A"], frozen_joint_length
        ),
    }
    if score_self_shift:
        cells["SELF_SHIFT"] = score_matrix(
            features["SELF_SHIFT"]["V"],
            features["SELF_SHIFT"]["A"],
            frozen_joint_length,
        )
    for factor in FACTORS:
        cells[factor] = {
            "q10": score_matrix(features[factor]["V"], features["ID"]["A"], frozen_joint_length),
            "q01": score_matrix(features["ID"]["V"], features[factor]["A"], frozen_joint_length),
            "q11": score_matrix(features[factor]["V"], features[factor]["A"], frozen_joint_length),
        }
    cells["raw_fourcell"] = {}
    for condition in RAW_DOMAIN_CONDITIONS:
        joint = cells[condition]["q11"] if condition in FACTORS else cells[condition]
        cells["raw_fourcell"][condition] = {
            "q10": score_matrix(
                features[condition]["V"], features["RAW"]["A"], frozen_joint_length
            ),
            "q01": score_matrix(
                features["RAW"]["V"], features[condition]["A"], frozen_joint_length
            ),
            "q11": joint,
        }
    return cells


def exact_raw_bridge(
    new_raw: Mapping[str, Any],
    old_fixed_score: Mapping[str, Any],
) -> float:
    """Compare RAW to the sealed old FIXED__FIXED curve and core endpoints."""
    maximum = 0.0
    for geometry in GEOMETRIES:
        old_policies = old_fixed_score[geometry]["FIXED__FIXED"]["policies"]
        for policy, old in old_policies.items():
            if old is None:
                if policy in new_raw[geometry]:
                    raise ValueError(f"old {geometry}/{policy} is absent but new is present")
                continue
            new = new_raw[geometry][policy]
            for field in ("curve", "C", "B", "D", "C_anchor", "D_anchor", "best_lag", "offset"):
                difference = np.asarray(new[field]) - np.asarray(old[field])
                error = float(np.max(np.abs(difference)))
                maximum = max(maximum, error)
                if error != 0:
                    raise ValueError(f"RAW exact bridge failed: {geometry}/{policy}/{field}: {error}")
    return maximum
