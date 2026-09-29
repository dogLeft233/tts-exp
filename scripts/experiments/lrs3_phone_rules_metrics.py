"""Pure metrics and frozen-probe helpers for the LRS3 phone-rules experiment.

This module deliberately has no audio, model, filesystem, or subprocess side
effects.  The runner and the independent checker both use the small numerical
primitives here, while all scientific decisions are made from serialized
inputs and are recorded in the run artifacts.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

SILENCE_LABELS = frozenset({"", "sil", "sp", "spn", "pau", "noise"})


def normalize_phone(label: object) -> str:
    """Normalize only Unicode form and surrounding whitespace.

    In particular, this function does not remove IPA length marks, aspiration,
    tie bars, diacritics, or tone marks.  The experiment's label inventory is
    therefore the inventory supplied by MFA, not a hand-written ARPABET map.
    """

    return unicodedata.normalize("NFC", str(label)).strip()


def is_speech_label(label: object, silence_labels: Iterable[str] = SILENCE_LABELS) -> bool:
    normalized = normalize_phone(label)
    lowered = normalized.lower()
    return normalized != "" and lowered not in {normalize_phone(x).lower() for x in silence_labels}


def _l2_normalize(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError(f"{name} has zero norm")
    return value / norm


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, q, interpolation="linear"))


def derive_frontend_geometry(
    conv_kernel: Sequence[int],
    conv_stride: Sequence[int],
    conv_dilation: Sequence[int] | None = None,
    conv_padding: Sequence[int] | None = None,
) -> dict[str, float | int | list[int]]:
    """Derive receptive-field size, jump, and first-frame center.

    The default is the no-padding frontend used by HuBERT/wav2vec2.  Padding
    can be supplied explicitly so an unexpected model configuration is visible
    in the protocol instead of silently inheriting a 10 ms offset.
    """

    kernels = [int(x) for x in conv_kernel]
    strides = [int(x) for x in conv_stride]
    if not kernels or len(kernels) != len(strides):
        raise ValueError("conv_kernel and conv_stride must have equal non-zero length")
    if any(x <= 0 for x in kernels + strides):
        raise ValueError("convolution kernels and strides must be positive")
    dilations = [1] * len(kernels) if conv_dilation is None else [int(x) for x in conv_dilation]
    paddings = [0] * len(kernels) if conv_padding is None else [int(x) for x in conv_padding]
    if len(dilations) != len(kernels) or len(paddings) != len(kernels):
        raise ValueError("frontend geometry arrays must have equal length")
    if any(x <= 0 for x in dilations) or any(x < 0 for x in paddings):
        raise ValueError("dilation must be positive and padding non-negative")

    receptive_field = 1
    jump = 1
    first_center = 0.0
    effective_kernels: list[int] = []
    for kernel, stride, dilation, padding in zip(kernels, strides, dilations, paddings):
        effective = (kernel - 1) * dilation + 1
        effective_kernels.append(effective)
        first_center += ((effective - 1) / 2.0 - padding) * jump
        receptive_field += (effective - 1) * jump
        jump *= stride
    return {
        "receptive_field_samples": int(receptive_field),
        "frame_stride_samples": int(jump),
        "first_center_samples": float(first_center),
        "effective_kernels": effective_kernels,
        "conv_kernel": kernels,
        "conv_stride": strides,
        "conv_dilation": dilations,
        "conv_padding": paddings,
    }


def derive_frame_times(
    num_frames: int,
    sample_rate: int,
    conv_kernel: Sequence[int],
    conv_stride: Sequence[int],
    conv_dilation: Sequence[int] | None = None,
    conv_padding: Sequence[int] | None = None,
) -> np.ndarray:
    """Return explicit frontend-center times for ``num_frames`` outputs."""

    if int(num_frames) < 0 or int(sample_rate) <= 0:
        raise ValueError("num_frames must be non-negative and sample_rate positive")
    geometry = derive_frontend_geometry(conv_kernel, conv_stride, conv_dilation, conv_padding)
    return (
        float(geometry["first_center_samples"])
        + np.arange(int(num_frames), dtype=np.float64) * float(geometry["frame_stride_samples"])
    ) / float(sample_rate)


def pool_phone_tokens(
    layer_embeddings: np.ndarray,
    frame_times: np.ndarray,
    tokens: Sequence[Mapping[str, Any]],
    *,
    sample_id: str | None = None,
    source_group: str | None = None,
    condition: str | None = None,
    silence_labels: Iterable[str] = SILENCE_LABELS,
) -> list[dict[str, Any]]:
    """Mean-pool a frame sequence into auditable token records.

    Frame ownership is half-open: ``start_s <= frame_time < end_s``.  Tokens
    without a frame are retained with ``valid=False`` and are counted in the
    coverage denominator by the caller.
    """

    features = np.asarray(layer_embeddings, dtype=np.float64)
    times = np.asarray(frame_times, dtype=np.float64)
    if features.ndim != 2 or times.ndim != 1 or features.shape[0] != times.size:
        raise ValueError("layer_embeddings and frame_times have incompatible shapes")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(times)):
        raise ValueError("features and frame_times must be finite")
    result: list[dict[str, Any]] = []
    for token_index, raw in enumerate(tokens):
        label = normalize_phone(raw.get("label", raw.get("token", "")))
        start = float(raw["start_s"])
        end = float(raw["end_s"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"invalid token span at index {token_index}")
        speech = bool(raw.get("speech", raw.get("is_speech", is_speech_label(label, silence_labels))))
        record: dict[str, Any] = {
            "token_id": f"{sample_id or ''}:{token_index}",
            "token_index": int(token_index),
            "label": label,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "speech": speech,
            "sample_id": sample_id,
            "source_group": source_group,
            "condition": condition,
            "frame_indices": [],
            "embedding": None,
            "valid": False,
            "reason": "non_speech" if not speech else "no_frame",
        }
        if speech:
            indices = np.flatnonzero((times >= start) & (times < end))
            record["frame_indices"] = indices.astype(int).tolist()
            if indices.size:
                pooled = features[indices].mean(axis=0)
                try:
                    normalized = _l2_normalize(pooled, name="pooled embedding")
                except ValueError as exc:
                    record["reason"] = "zero_or_nonfinite_embedding"
                    raise ValueError(f"{record['token_id']}: {exc}") from exc
                record["embedding"] = normalized.tolist()
                record["valid"] = True
                record["reason"] = "ok"
        result.append(record)
    return result


def _record_vector(record: Mapping[str, Any]) -> np.ndarray | None:
    if not bool(record.get("valid", record.get("embedding") is not None)):
        return None
    value = record.get("embedding")
    if value is None:
        return None
    return _l2_normalize(np.asarray(value, dtype=np.float64), name="token embedding")


def fit_reference_centroids(
    records: Sequence[Mapping[str, Any]],
    *,
    min_tokens: int = 20,
    min_groups: int = 3,
    conditions: tuple[str, str] = ("natural", "tts"),
) -> dict[str, Any]:
    """Fit one shared natural/TTS reference centroid per supported label.

    Tokens are averaged within ``(label, condition, source_group)`` first,
    then groups are equally weighted, and finally the two conditions receive
    equal weight.  This prevents long clips or a duplicated source group from
    dominating the probe.
    """

    if min_tokens <= 0 or min_groups <= 0:
        raise ValueError("support thresholds must be positive")
    by_label_condition_group: dict[str, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for raw in records:
        condition = str(raw.get("condition", ""))
        if condition not in conditions or not bool(raw.get("speech", True)):
            continue
        label = normalize_phone(raw.get("label", ""))
        group = str(raw.get("source_group", ""))
        if not label or not group:
            continue
        vector = _record_vector(raw)
        if vector is not None:
            by_label_condition_group[label][condition][group].append(vector)

    support: dict[str, Any] = {}
    centers: dict[str, list[float]] = {}
    for label in sorted(by_label_condition_group):
        condition_info: dict[str, Any] = {}
        eligible = True
        group_centers: dict[str, np.ndarray] = {}
        for condition in conditions:
            groups = by_label_condition_group[label].get(condition, {})
            token_count = sum(len(vectors) for vectors in groups.values())
            eligible_groups = sorted(groups)
            condition_info[condition] = {
                "token_count": int(token_count),
                "group_count": len(eligible_groups),
                "groups": eligible_groups,
            }
            if token_count < min_tokens or len(eligible_groups) < min_groups:
                eligible = False
                continue
            group_vectors: list[np.ndarray] = []
            for group in eligible_groups:
                group_vector = np.mean(np.stack(groups[group]), axis=0)
                group_centers[group] = _l2_normalize(group_vector, name=f"{label}/{condition}/{group}")
                group_vectors.append(group_centers[group])
            condition_info[condition]["centroid"] = _l2_normalize(
                np.mean(np.stack(group_vectors), axis=0), name=f"{label}/{condition}"
            ).tolist()
        support[label] = condition_info
        if not eligible:
            continue
        natural = np.asarray(condition_info[conditions[0]]["centroid"], dtype=np.float64)
        tts = np.asarray(condition_info[conditions[1]]["centroid"], dtype=np.float64)
        centers[label] = _l2_normalize((natural + tts) / 2.0, name=f"{label}/shared").tolist()

    return {
        "labels": sorted(centers),
        "centroids": centers,
        "support": support,
        "min_tokens": int(min_tokens),
        "min_groups": int(min_groups),
        "conditions": list(conditions),
        "fit_scope": "train_only_group_equal_then_condition_equal",
    }


def _predict(vector: np.ndarray, centroids: Mapping[str, Sequence[float]]) -> str:
    normalized = _l2_normalize(vector, name="prediction vector")
    labels = sorted(str(label) for label in centroids)
    if not labels:
        raise ValueError("cannot predict with an empty centroid set")
    matrix = np.stack([_l2_normalize(np.asarray(centroids[label]), name=f"centroid/{label}") for label in labels])
    scores = matrix @ normalized
    # labels are sorted, so argmax has deterministic NFC-label tie-breaking.
    return labels[int(np.argmax(scores))]


def score_pair(
    records: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    *,
    condition: str,
    pair_id: str,
    min_labels: int = 5,
    min_tokens: int = 10,
    min_coverage: float = 0.70,
) -> dict[str, Any]:
    """Score one pair using the already-frozen shared reference."""

    labels = {str(label) for label in reference.get("labels", [])}
    side = [r for r in records if str(r.get("condition")) == condition and bool(r.get("speech", True))]
    speech_count = len(side)
    valid = [r for r in side if str(r.get("label", "")) in labels and _record_vector(r) is not None]
    valid_labels = {str(r.get("label")) for r in valid}
    coverage = len(valid) / speech_count if speech_count else 0.0
    label_rows: list[dict[str, Any]] = []
    correct = 0
    for label in sorted(valid_labels):
        label_rows_for_pair = [r for r in valid if str(r.get("label")) == label]
        label_correct = 0
        for row in label_rows_for_pair:
            predicted = _predict(_record_vector(row), reference["centroids"])
            if predicted == label:
                label_correct += 1
            label_rows.append({
                "token_id": row.get("token_id"),
                "label": label,
                "prediction": predicted,
                "correct": predicted == label,
            })
        correct += label_correct
    accuracy = (
        float(np.mean([
            sum(1 for row in label_rows if row["label"] == label and row["correct"])
            / sum(1 for row in label_rows if row["label"] == label)
            for label in sorted(valid_labels)
        ]))
        if valid_labels
        else None
    )
    reasons: list[str] = []
    if len(valid_labels) < min_labels:
        reasons.append("COMMON_LABELS_BELOW_MIN")
    if len(valid) < min_tokens:
        reasons.append("VALID_TOKENS_BELOW_MIN")
    if coverage < min_coverage:
        reasons.append("SPEECH_TOKEN_COVERAGE_BELOW_MIN")
    return {
        "pair_id": pair_id,
        "condition": condition,
        "accuracy": accuracy,
        "speech_token_count": speech_count,
        "valid_token_count": len(valid),
        "coverage": coverage,
        "supported_labels": sorted(valid_labels),
        "label_count": len(valid_labels),
        "eligible": not reasons,
        "reason_codes": reasons,
        "predictions": label_rows,
        "correct_token_count": correct,
    }


def paired_bootstrap(
    group_effects: Mapping[str, float] | Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260920,
    draws: int = 10_000,
) -> dict[str, Any]:
    """Compute a deterministic source-group paired bootstrap CI."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    if isinstance(group_effects, Mapping):
        values_by_group = {str(k): float(v) for k, v in group_effects.items()}
    else:
        values_by_group = {}
        for row in group_effects:
            group = str(row["source_group"])
            if group in values_by_group:
                raise ValueError(f"duplicate source_group: {group}")
            values_by_group[group] = float(row["effect"])
    groups = sorted(values_by_group)
    values = np.asarray([values_by_group[group] for group in groups], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap effects must be finite and non-empty")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    sample_indices = rng.integers(0, values.size, size=(int(draws), values.size))
    boot = values[sample_indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": _quantile(boot, 0.025),
        "ci_high": _quantile(boot, 0.975),
        "n_groups": int(values.size),
        "groups": groups,
        "seed": int(seed),
        "draws": int(draws),
        "method": "source_group_paired_percentile_bootstrap",
    }


def _metric_value(value: Mapping[str, Any] | None, key: str = "estimate") -> float:
    if not value:
        return float("nan")
    raw = value.get(key)
    return float(raw) if raw is not None else float("nan")


def decide_stage_a(
    hubert: Mapping[str, Any] | None,
    xlsr: Mapping[str, Any] | None,
    *,
    support_ok: bool,
    threshold: float = 0.020,
) -> dict[str, Any]:
    """Apply the preregistered Stage A gate without silently coercing NaNs."""

    if not support_ok:
        return {"science_decision": "INSUFFICIENT_SUPPORT", "reason_codes": ["SUPPORT_OR_FEATURE_AUDIT_FAILED"]}
    h_est = _metric_value(hubert)
    h_low = _metric_value(hubert, "ci_low")
    x_est = _metric_value(xlsr)
    if not all(math.isfinite(x) for x in (h_est, h_low, x_est)):
        return {"science_decision": "NO_CLEAR_ADVANTAGE", "reason_codes": ["MISSING_OR_NONFINITE_STAGE_A_STATISTIC"]}
    reasons: list[str] = []
    if h_est < threshold:
        reasons.append("HUBERT_EFFECT_BELOW_THRESHOLD")
    if h_low <= 0.0:
        reasons.append("HUBERT_CI_CROSSES_ZERO")
    if x_est <= 0.0:
        reasons.append("XLSR_DIRECTIONAL_CHECK_FAILED")
    decision = "ADVANTAGE_SUPPORTED" if not reasons else "NO_CLEAR_ADVANTAGE"
    return {
        "science_decision": decision,
        "reason_codes": reasons,
        "hubert_threshold": threshold,
        "xlsr_is_directional_guard": True,
    }


def decide_stage_b(
    enhanced: Mapping[str, Any] | None,
    gain_control: Mapping[str, Any] | None,
    xlsr: Mapping[str, Any] | None,
    timing: Mapping[str, Any] | None,
    *,
    min_effect: float = 0.010,
) -> dict[str, Any]:
    """Apply the main Stage B gate and report secondary TTS-gap decisions."""

    reasons: list[str] = []
    if not enhanced or not gain_control or not xlsr:
        reasons.append("MISSING_STAGE_B_STATISTIC")
    else:
        if _metric_value(enhanced) < min_effect:
            reasons.append("ENHANCEMENT_EFFECT_BELOW_THRESHOLD")
        if _metric_value(enhanced, "ci_low") <= 0.0:
            reasons.append("ENHANCEMENT_CI_CROSSES_ZERO")
        if _metric_value(gain_control, "ci_low") <= 0.0:
            reasons.append("GAIN_ONLY_NOT_EXCLUDED")
        if _metric_value(xlsr) <= 0.0:
            reasons.append("XLSR_DIRECTIONAL_CHECK_FAILED")
    timing_ok = bool((timing or {}).get("timing_pass", False))
    if not timing_ok:
        reasons.append(str((timing or {}).get("status", "TIMING_UNVERIFIED")))
    return {
        "science_decision": "NATURAL_ENHANCEMENT_SUPPORTED" if not reasons else "NO_IMPROVEMENT",
        "reason_codes": reasons,
        "timing_required": True,
        "min_effect": min_effect,
    }


__all__ = [
    "SILENCE_LABELS",
    "decide_stage_a",
    "decide_stage_b",
    "derive_frame_times",
    "derive_frontend_geometry",
    "fit_reference_centroids",
    "is_speech_label",
    "normalize_phone",
    "paired_bootstrap",
    "pool_phone_tokens",
    "score_pair",
]
