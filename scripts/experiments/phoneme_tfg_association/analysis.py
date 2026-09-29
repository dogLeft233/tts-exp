"""Prespecified source-block analysis for the phoneme/TFG association."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import PAIR_TYPES, TTS_CONDITIONS, ProtocolError


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _feature_value(row: Mapping[str, Any], condition: str, name: str) -> float | None:
    arms = row.get("arms", {})
    arm = arms.get(condition, {}) if isinstance(arms, Mapping) else {}
    if name == "primary_silhouette":
        return _finite(arm.get("primary_silhouette"))
    if name == "xlsr_phoneme_silhouette":
        return _finite(arm.get("xlsr_layer10", {}).get("phoneme", {}).get("silhouette"))
    if name == "viseme_silhouette":
        return _finite(arm.get("hubert_layer6", {}).get("viseme", {}).get("silhouette"))
    if name == "log_fisher":
        value = _finite(arm.get("primary_fisher_ratio"))
        return None if value is None or value <= 0 else float(math.log(value))
    raise ProtocolError(f"unknown feature name: {name}")


def _score_index(score_rows: Sequence[Mapping[str, Any]], support: str) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for raw in score_rows:
        rows = raw.get("endpoints") if isinstance(raw.get("endpoints"), list) else [raw]
        for row in rows:
            if str(row.get("support")) != support:
                continue
            sid = str(row.get("sample_id"))
            condition = str(row.get("condition"))
            tfg = str(row.get("tfg", "wav2lip"))
            key = (sid, condition, tfg)
            if key in index:
                raise ProtocolError(f"duplicate endpoint row: {key}")
            index[key] = row
    return index


def build_block_differences(
    blocks: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    feature_name: str = "primary_silhouette",
    support: str = "EQUAL_COUNT",
    tfg: str = "wav2lip",
) -> list[dict[str, Any]]:
    """Build one independent row per source block, not one row per video."""
    features = {str(row["sample_id"]): row for row in feature_rows}
    scores = _score_index(score_rows, support)
    result: list[dict[str, Any]] = []
    for block in blocks:
        sid = str(block["sample_id"])
        pair = tuple(str(item) for item in block["pair"])
        if len(pair) != 2:
            raise ProtocolError(f"block pair must have two arms: {block.get('block_id')}")
        feature = features.get(sid)
        if feature is None:
            continue
        endpoint = {condition: scores.get((sid, condition, tfg)) for condition in ("natural", *pair)}
        if any(value is None for value in endpoint.values()):
            continue
        s_values = {condition: _feature_value(feature, condition, feature_name) for condition in ("natural", *pair)}
        c_values = {condition: _finite(endpoint[condition].get("sync_c")) for condition in ("natural", *pair)}
        if any(value is None for value in (*s_values.values(), *c_values.values())):
            continue
        natural_s = float(s_values["natural"])
        natural_c = float(c_values["natural"])
        a, b = pair
        duration_ratio_a = _finite(feature.get("duration_ratios_to_natural", {}).get(a))
        duration_ratio_b = _finite(feature.get("duration_ratios_to_natural", {}).get(b))
        duration_log_ratio = None
        if duration_ratio_a is not None and duration_ratio_b is not None and duration_ratio_a > 0.0 and duration_ratio_b > 0.0:
            duration_log_ratio = float(math.log(duration_ratio_b / duration_ratio_a))
        feature_arms = feature.get("arms", {})
        silence_a = _finite(feature_arms.get(a, {}).get("silence_fraction")) if isinstance(feature_arms, Mapping) else None
        silence_b = _finite(feature_arms.get(b, {}).get("silence_fraction")) if isinstance(feature_arms, Mapping) else None
        silence_fraction_delta = None if silence_a is None or silence_b is None else float(silence_b - silence_a)
        result.append({
            "block_id": str(block["block_id"]),
            "sample_id": sid,
            "source_group": str(block["source_group"]),
            "pair": f"{a}|{b}",
            "pair_arms": [a, b],
            "feature": feature_name,
            "endpoint": f"{tfg}_{support.lower()}_sync_c",
            "support": support,
            "s_natural": natural_s,
            "s_a": float(s_values[a]),
            "s_b": float(s_values[b]),
            "c_natural": natural_c,
            "c_a": float(c_values[a]),
            "c_b": float(c_values[b]),
            "d_natural": _finite(endpoint["natural"].get("sync_d")),
            "d_a": _finite(endpoint[a].get("sync_d")),
            "d_b": _finite(endpoint[b].get("sync_d")),
            "b_natural": _finite(endpoint["natural"].get("background_b")),
            "b_a": _finite(endpoint[a].get("background_b")),
            "b_b": _finite(endpoint[b].get("background_b")),
            "x_a": (float(s_values[a]) - natural_s) / 0.1,
            "x_b": (float(s_values[b]) - natural_s) / 0.1,
            "x": (float(s_values[b]) - float(s_values[a])) / 0.1,
            "y_a": float(c_values[a]) - natural_c,
            "y_b": float(c_values[b]) - natural_c,
            "y": float(c_values[b]) - float(c_values[a]),
            "gain_d": None if _finite(endpoint[a].get("sync_d")) is None or _finite(endpoint[b].get("sync_d")) is None else float(endpoint[a]["sync_d"]) - float(endpoint[b]["sync_d"]),
            "delta_b": None if _finite(endpoint[a].get("background_b")) is None or _finite(endpoint[b].get("background_b")) is None else float(endpoint[b]["background_b"]) - float(endpoint[a]["background_b"]),
            "duration_ratio_a": duration_ratio_a,
            "duration_ratio_b": duration_ratio_b,
            "duration_log_ratio": duration_log_ratio,
            "silence_fraction_delta": silence_fraction_delta,
            "complete": True,
        })
    return result


def _design(rows: Sequence[Mapping[str, Any]], pair_order: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    usable = [row for row in rows if row.get("complete", True) and _finite(row.get("x")) is not None and _finite(row.get("y")) is not None and str(row.get("pair")) in pair_order]
    if not usable:
        return np.empty((0, len(pair_order) + 1)), np.empty(0), list(pair_order), []
    x = np.asarray([float(row["x"]) for row in usable], dtype=np.float64)
    y = np.asarray([float(row["y"]) for row in usable], dtype=np.float64)
    pair_values = [str(row["pair"]) for row in usable]
    design = np.zeros((len(usable), len(pair_order) + 1), dtype=np.float64)
    for index, pair in enumerate(pair_values):
        design[index, pair_order.index(pair)] = 1.0
    design[:, -1] = x
    return design, y, list(pair_order), usable


def _hc3(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xtx_inv = np.linalg.pinv(design.T @ design, rcond=1e-12)
    beta = xtx_inv @ design.T @ y
    fitted = design @ beta
    residual = y - fitted
    hat = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    adjusted = residual / np.maximum(1.0 - hat, 1e-12)
    meat = design.T @ ((adjusted[:, None] ** 2) * design)
    covariance = xtx_inv @ meat @ xtx_inv
    return beta, residual, hat, covariance


def _t_pvalue(t_value: float, df: int) -> float:
    try:
        from scipy.stats import t as student_t

        return float(2.0 * student_t.sf(abs(t_value), max(int(df), 1)))
    except ImportError:  # pragma: no cover
        return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def _t_critical(df: int, alpha: float = 0.05) -> float:
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(1.0 - alpha / 2.0, max(int(df), 1)))
    except ImportError:  # pragma: no cover
        return 1.96


def _fit_beta(design: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    beta, _residual, _hat, covariance = _hc3(design, y)
    estimate = float(beta[-1])
    se = float(math.sqrt(max(float(covariance[-1, -1]), 0.0)))
    statistic = estimate / se if se > 0.0 else math.copysign(float("inf"), estimate) if estimate else 0.0
    return estimate, se, statistic


def _wild_pvalue(design: np.ndarray, y: np.ndarray, *, draws: int, seed: int) -> tuple[float | None, int]:
    if draws < 1:
        return None, 0
    pair_design = design[:, :-1]
    beta0, residual0, hat0, _cov0 = _hc3(pair_design, y)
    fitted0 = pair_design @ beta0
    adjusted = residual0 / np.sqrt(np.maximum(1.0 - hat0, 1e-12))
    _estimate, _se, observed_t = _fit_beta(design, y)
    if not math.isfinite(observed_t):
        return None, 0
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    exceedances = 0
    for _ in range(int(draws)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(y))
        simulated = fitted0 + signs * adjusted
        _estimate_star, _se_star, t_star = _fit_beta(design, simulated)
        if math.isfinite(t_star) and abs(t_star) >= abs(observed_t):
            exceedances += 1
    return float((exceedances + 1) / (int(draws) + 1)), int(draws)


def fit_primary(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_name: str = "hubert_layer6_phoneme_silhouette",
    endpoint_name: str = "wav2lip_interior_equal_count_sync_c",
    pair_order: Sequence[str] | None = None,
    min_complete_blocks: int = 30,
    min_pair_blocks: int = 4,
    wild_draws: int = 9999,
    seed: int = 20260921,
) -> dict[str, Any]:
    """Fit the one registered six-pair-intercept plus within-utterance slope."""
    order = list(pair_order or ["|".join(pair) for pair in PAIR_TYPES])
    design, y, order, usable = _design(rows, order)
    counts = Counter(str(row["pair"]) for row in usable)
    base = {
        "estimand": "within_utterance_pair_adjusted_silhouette_to_sync_c_difference",
        "feature": feature_name,
        "endpoint": endpoint_name,
        "formula": "y_i = pair_intercept[pair_i] + beta * ((S_ib-S_ia)/0.1) + error_i",
        "n_blocks": len(usable),
        "n_source_groups": len({str(row["source_group"]) for row in usable}),
        "pair_counts": dict(sorted(counts.items())),
        "draws": int(wild_draws),
        "seed": int(seed),
        "claim_boundary": "association only; phoneme separability was not randomized and no causal claim is identified",
    }
    if len(usable) < min_complete_blocks or any(counts.get(pair, 0) < min_pair_blocks for pair in order):
        return {**base, "scientific_status": "INSUFFICIENT_SUPPORT", "reason": "complete-block or pair support gate failed", "engineering_status": "READY"}
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        return {**base, "scientific_status": "INSUFFICIENT_SUPPORT", "reason": "design matrix is rank deficient", "design_rank": rank, "engineering_status": "READY"}
    pair_design = design[:, :-1]
    residualized_x = (np.eye(len(usable)) - pair_design @ np.linalg.pinv(pair_design)) @ design[:, -1]
    if float(np.var(residualized_x)) <= 1e-12:
        return {**base, "scientific_status": "INSUFFICIENT_SUPPORT", "reason": "within-pair predictor variance is too small", "design_rank": rank, "engineering_status": "READY"}
    beta, residual, leverage, covariance = _hc3(design, y)
    if float(np.max(leverage)) >= 0.5:
        return {**base, "scientific_status": "INSUFFICIENT_SUPPORT", "reason": "maximum leverage is at least 0.5", "design_rank": rank, "max_leverage": float(np.max(leverage)), "engineering_status": "READY"}
    estimate = float(beta[-1])
    se = float(math.sqrt(max(float(covariance[-1, -1]), 0.0)))
    df = len(usable) - rank
    t_value = estimate / se if se > 0.0 else math.copysign(float("inf"), estimate) if estimate else 0.0
    critical = _t_critical(df)
    ci = [estimate - critical * se, estimate + critical * se]
    p_t = _t_pvalue(t_value, df) if math.isfinite(t_value) else 0.0
    p_wild, actual_draws = _wild_pvalue(design, y, draws=wild_draws, seed=seed + 17)
    if estimate > 0 and ci[0] > 0 and p_wild is not None and p_wild < 0.05:
        status = "POSITIVE_WITHIN_PAIR_ASSOCIATION"
    elif estimate < 0 and ci[1] < 0 and p_wild is not None and p_wild < 0.05:
        status = "NEGATIVE_WITHIN_PAIR_ASSOCIATION"
    else:
        status = "INCONCLUSIVE"
    leave_one_out: list[dict[str, Any]] = []
    for index, row in enumerate(usable):
        keep = [other for j, other in enumerate(usable) if j != index]
        x_loo, y_loo, _order, usable_loo = _design(keep, order)
        if len(usable_loo) >= rank + 2 and np.linalg.matrix_rank(x_loo) == x_loo.shape[1]:
            beta_loo, _se_loo, _t_loo = _fit_beta(x_loo, y_loo)
            leave_one_out.append({"block_id": row["block_id"], "beta": float(beta_loo)})
    return {
        **base,
        "beta": estimate,
        "beta_units": "Sync-C per 0.1 within-utterance silhouette difference",
        "se_hc3": se,
        "ci95_hc3": ci,
        "p_hc3_t": p_t,
        "p_wild": p_wild,
        "wild_draws": actual_draws,
        "design_rank": rank,
        "residual_df": df,
        "condition_number": float(np.linalg.cond(design)),
        "max_leverage": float(np.max(leverage)),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "residualized_predictor_variance": float(np.var(residualized_x)),
        "leave_one_out": leave_one_out,
        "scientific_status": status,
        "engineering_status": "READY",
    }


def _adjusted_design(
    rows: Sequence[Mapping[str, Any]],
    pair_order: Sequence[str],
    covariate_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    usable = [
        row
        for row in rows
        if row.get("complete", True)
        and _finite(row.get("x")) is not None
        and _finite(row.get("y")) is not None
        and str(row.get("pair")) in pair_order
        and all(_finite(row.get(name)) is not None for name in covariate_names)
    ]
    width = len(pair_order) + 1 + len(covariate_names)
    if not usable:
        return np.empty((0, width)), np.empty(0), []
    design = np.zeros((len(usable), width), dtype=np.float64)
    y = np.asarray([float(row["y"]) for row in usable], dtype=np.float64)
    for index, row in enumerate(usable):
        design[index, pair_order.index(str(row["pair"]))] = 1.0
        design[index, len(pair_order)] = float(row["x"])
        for offset, name in enumerate(covariate_names, start=len(pair_order) + 1):
            design[index, offset] = float(row[name])
    return design, y, usable


def fit_duration_silence_adjusted(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_order: Sequence[str] | None = None,
    min_complete_blocks: int = 30,
    min_pair_blocks: int = 4,
) -> dict[str, Any]:
    """Fit the registered explanatory adjustment without a new p-value search."""
    order = list(pair_order or ["|".join(pair) for pair in PAIR_TYPES])
    candidates = ("duration_log_ratio", "silence_fraction_delta")
    availability = {
        name: sum(
            row.get("complete", True)
            and _finite(row.get("x")) is not None
            and _finite(row.get("y")) is not None
            and _finite(row.get(name)) is not None
            for row in rows
        )
        for name in candidates
    }
    covariates = [name for name in candidates if availability[name] >= min_complete_blocks]
    design, y, usable = _adjusted_design(rows, order, covariates)
    counts = Counter(str(row["pair"]) for row in usable)
    result: dict[str, Any] = {
        "estimand": "within_pair_silhouette_slope_adjusted_for_duration_and_silence",
        "formula": "y = pair_intercept + beta_x*x + beta_covariates*covariates + error",
        "covariates_requested": list(candidates),
        "covariates_used": covariates,
        "covariate_availability": availability,
        "n_blocks": len(usable),
        "pair_counts": dict(sorted(counts.items())),
        "claim_boundary": "explanatory sensitivity only; duration/silence may be confounders or mediators and no direct causal effect is identified",
        "p_value_search": False,
    }
    if not covariates:
        return {**result, "status": "NOT_RUN_MISSING_COVARIATES", "reason": "no registered adjustment covariate reached the complete-block gate"}
    if len(usable) < min_complete_blocks or any(counts.get(pair, 0) < min_pair_blocks for pair in order):
        return {**result, "status": "INSUFFICIENT_SUPPORT", "reason": "complete-block or pair support gate failed"}
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        return {**result, "status": "INSUFFICIENT_SUPPORT", "reason": "adjusted design matrix is rank deficient", "design_rank": rank}
    beta, residual, leverage, covariance = _hc3(design, y)
    if float(np.max(leverage)) >= 0.5:
        return {**result, "status": "INSUFFICIENT_SUPPORT", "reason": "maximum leverage is at least 0.5", "design_rank": rank, "max_leverage": float(np.max(leverage))}
    x_index = len(order)
    estimate = float(beta[x_index])
    se = float(math.sqrt(max(float(covariance[x_index, x_index]), 0.0)))
    critical = _t_critical(len(usable) - rank)
    return {
        **result,
        "status": "COMPLETE",
        "beta_x": estimate,
        "se_hc3_x": se,
        "ci95_hc3_x": [estimate - critical * se, estimate + critical * se],
        "covariate_coefficients": {name: float(beta[len(order) + 1 + offset]) for offset, name in enumerate(covariates)},
        "design_rank": rank,
        "residual_df": len(usable) - rank,
        "max_leverage": float(np.max(leverage)),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "condition_number": float(np.linalg.cond(design)),
    }


def sensitivity_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    duration_ratio_min: float = 0.5,
    duration_ratio_max: float = 1.5,
    min_complete_blocks: int = 30,
    min_pair_blocks: int = 4,
    pair_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run only the pre-registered explanatory sensitivity views."""
    order = list(pair_order or ["|".join(pair) for pair in PAIR_TYPES])
    complete = [row for row in rows if row.get("complete", True) and _finite(row.get("x")) is not None and _finite(row.get("y")) is not None]
    duration_subset = [
        row
        for row in complete
        if _finite(row.get("duration_ratio_a")) is not None
        and _finite(row.get("duration_ratio_b")) is not None
        and duration_ratio_min <= float(row["duration_ratio_a"]) <= duration_ratio_max
        and duration_ratio_min <= float(row["duration_ratio_b"]) <= duration_ratio_max
    ]
    subset_fit = fit_primary(
        duration_subset,
        pair_order=order,
        min_complete_blocks=min_complete_blocks,
        min_pair_blocks=min_pair_blocks,
        wild_draws=0,
    )
    leave_one_pair_out: dict[str, Any] = {}
    for omitted in order:
        remaining = [pair for pair in order if pair != omitted]
        kept = [row for row in complete if str(row.get("pair")) != omitted]
        leave_one_pair_out[omitted] = fit_primary(
            kept,
            pair_order=remaining,
            min_complete_blocks=max(min_complete_blocks - min_pair_blocks, 1),
            min_pair_blocks=min_pair_blocks,
            wild_draws=0,
        )
    leave_one_block_within_pair: dict[str, list[dict[str, Any]]] = {}
    for pair in order:
        pair_results: list[dict[str, Any]] = []
        pair_indices = [index for index, item in enumerate(complete) if str(item.get("pair")) == pair]
        for removed_index in pair_indices:
            row = complete[removed_index]
            kept = [item for index, item in enumerate(complete) if index != removed_index]
            fit = fit_primary(
                kept,
                pair_order=order,
                min_complete_blocks=max(min_complete_blocks - 1, 1),
                min_pair_blocks=max(min_pair_blocks - 1, 1),
                wild_draws=0,
            )
            pair_results.append({"removed_block_id": row.get("block_id"), "beta": fit.get("beta"), "status": fit.get("scientific_status"), "n_blocks": fit.get("n_blocks")})
        leave_one_block_within_pair[pair] = pair_results
    adjusted = fit_duration_silence_adjusted(
        complete,
        pair_order=order,
        min_complete_blocks=min_complete_blocks,
        min_pair_blocks=min_pair_blocks,
    )
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "not_for_primary_inference": True,
        "duration_ratio_bounds": [float(duration_ratio_min), float(duration_ratio_max)],
        "complete_blocks": len(complete),
        "duration_ratio_subset": {"n_blocks": len(duration_subset), "fit": subset_fit},
        "duration_silence_adjusted": adjusted,
        "leave_one_pair_out": leave_one_pair_out,
        "leave_one_block_within_pair": leave_one_block_within_pair,
        "interpretation": "Display alongside the primary estimate; do not select a preferred sensitivity by its p-value.",
    }


def _bootstrap_mean(values: Mapping[str, float], *, draws: int, seed: int) -> dict[str, Any]:
    labels = sorted(values)
    numbers = np.asarray([float(values[label]) for label in labels], dtype=np.float64)
    if not labels or not np.isfinite(numbers).all():
        return {"estimate": None, "ci95": [None, None], "n_groups": len(labels), "draws": 0, "seed": seed}
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(numbers), size=(draws, len(numbers)))
    boot = numbers[indices].mean(axis=1)
    return {"estimate": float(numbers.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "n_groups": len(labels), "draws": int(draws), "seed": int(seed), "groups": labels}


def summarize_gains(
    blocks: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    tfg: str = "wav2lip",
    support: str = "EQUAL_COUNT",
    draws: int = 10000,
    seed: int = 20260921,
) -> dict[str, Any]:
    scores = _score_index(score_rows, support)
    by_tts: dict[str, dict[str, float]] = {condition: {} for condition in TTS_CONDITIONS}
    components: dict[str, dict[str, dict[str, float]]] = {condition: {} for condition in TTS_CONDITIONS}
    for block in blocks:
        sid = str(block["sample_id"])
        natural = scores.get((sid, "natural", tfg))
        if natural is None or _finite(natural.get("sync_c")) is None:
            continue
        c_n = float(natural["sync_c"])
        b_n = _finite(natural.get("background_b"))
        if b_n is None:
            b_n = _finite(natural.get("background_b_hat"))
        for condition in block["tts_arms"]:
            tts = scores.get((sid, str(condition), tfg))
            if tts is None or _finite(tts.get("sync_c")) is None:
                continue
            c_t = float(tts["sync_c"])
            by_tts[str(condition)][str(block["source_group"])] = c_t - c_n
            b_t = _finite(tts.get("background_b"))
            if b_t is None:
                b_t = _finite(tts.get("background_b_hat"))
            components[str(condition)][str(block["source_group"])] = {
                "gain_c": c_t - c_n,
                "gain_d": None if _finite(tts.get("sync_d")) is None or _finite(natural.get("sync_d")) is None else float(natural["sync_d"]) - float(tts["sync_d"]),
                "gain_b": None if b_t is None or b_n is None else b_t - b_n,
            }
    summaries: dict[str, Any] = {}
    for index, (condition, values) in enumerate(sorted(by_tts.items())):
        summary = _bootstrap_mean(values, draws=draws, seed=seed + index * 101)
        numbers = list(values.values())
        summary["positive_count"] = int(sum(value > 0.0 for value in numbers))
        summary["positive_fraction"] = None if not numbers else float(summary["positive_count"] / len(numbers))
        summaries[condition] = summary
    source_mean = _bootstrap_mean(
        {
            group: float(np.mean([values[group] for values in by_tts.values() if group in values]))
            for group in sorted({group for values in by_tts.values() for group in values})
            if any(group in values for values in by_tts.values())
        },
        draws=draws,
        seed=seed + 909,
    )
    source_values = {
        group: float(np.mean([values[group] for values in by_tts.values() if group in values]))
        for group in sorted({group for values in by_tts.values() for group in values})
        if any(group in values for values in by_tts.values())
    }
    source_mean["positive_count"] = int(sum(value > 0.0 for value in source_values.values()))
    source_mean["positive_fraction"] = None if not source_values else float(source_mean["positive_count"] / len(source_values))
    summaries["source_mean_across_assigned_tts"] = source_mean
    return {"support": support, "tfg": tfg, "summaries": summaries, "group_gain_c": by_tts, "components": components}


def benjamini_hochberg(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    valid = sorted((float(value), key) for key, value in p_values.items() if value is not None and math.isfinite(float(value)))
    result = {key: None for key in p_values}
    running = 1.0
    for index in range(len(valid) - 1, -1, -1):
        value, key = valid[index]
        running = min(running, value * len(valid) / (index + 1))
        result[key] = float(min(1.0, running))
    return result


def secondary_associations(block_rows: Sequence[Mapping[str, Any]], *, wild_draws: int = 9999, seed: int = 20260921) -> dict[str, Any]:
    specs = {
        "xlsr_l10_silhouette_to_C": "xlsr_phoneme_silhouette",
        "hubert_l6_viseme_silhouette_to_C": "viseme_silhouette",
        "hubert_l6_log_fisher_to_C": "log_fisher",
    }
    # The current block rows carry only the primary feature.  Additional rows
    # are accepted by callers when those features are available; missingness is
    # explicit and never filled with the primary value.
    p_values: dict[str, float | None] = {}
    results: dict[str, Any] = {}
    for name, feature_name in specs.items():
        rows = [row for row in block_rows if row.get("secondary_feature_name") == feature_name and _finite(row.get("secondary_x")) is not None]
        transformed = [{**row, "x": row["secondary_x"]} for row in rows]
        fit = fit_primary(transformed, wild_draws=wild_draws, seed=seed + len(results) * 101)
        p_values[name] = fit.get("p_wild")
        results[name] = fit
    q_values = benjamini_hochberg(p_values)
    for name in results:
        results[name]["q_fdr"] = q_values[name]
    return results


def power_plan(blocks: Sequence[Mapping[str, Any]], feature_rows: Sequence[Mapping[str, Any]], *, seed: int = 20260921, simulations: int = 2000) -> dict[str, Any]:
    features = {str(row["sample_id"]): row for row in feature_rows}
    rows = []
    for block in blocks:
        feature = features.get(str(block["sample_id"]))
        if feature is None:
            continue
        a, b = block["pair"]
        s_a = _feature_value(feature, a, "primary_silhouette")
        s_b = _feature_value(feature, b, "primary_silhouette")
        if s_a is not None and s_b is not None:
            rows.append({"pair": "|".join((a, b)), "x": (s_b - s_a) / 0.1})
    pair_order = ["|".join(pair) for pair in PAIR_TYPES]
    design, _y, _order, usable = _design([{**row, "y": 0.0} for row in rows], pair_order)
    if not usable or np.linalg.matrix_rank(design) < design.shape[1]:
        return {"status": "INSUFFICIENT_SUPPORT", "n_blocks": len(usable), "simulations": simulations}
    rng = np.random.Generator(np.random.PCG64(seed))
    grid: list[dict[str, Any]] = []
    for sigma in (0.25, 0.5, 1.0):
        for theta in (0.0, 0.1, 0.3, 0.5):
            for noise in ("normal", "t5"):
                rejected = 0
                for _ in range(int(simulations)):
                    eps = rng.normal(size=len(usable)) if noise == "normal" else rng.standard_t(5, size=len(usable)) / math.sqrt(5 / 3)
                    y = theta * design[:, -1] + float(sigma) * eps
                    estimate, se, t_value = _fit_beta(design, y)
                    if _t_pvalue(t_value, len(usable) - design.shape[1]) < 0.05:
                        rejected += 1
                grid.append({"sigma": sigma, "theta": theta, "noise": noise, "power": rejected / int(simulations), "simulations": int(simulations)})
    return {"status": "complete", "n_blocks": len(usable), "simulations": simulations, "seed": seed, "grid": grid, "interpretation": "planning only; sigma/effect are unknown and not observed results"}


__all__ = [
    "build_block_differences",
    "fit_primary",
    "fit_duration_silence_adjusted",
    "sensitivity_analysis",
    "summarize_gains",
    "secondary_associations",
    "power_plan",
    "benjamini_hochberg",
]
