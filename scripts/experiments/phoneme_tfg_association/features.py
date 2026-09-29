"""CPU reconstruction of per-utterance phoneme separability features."""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.lrs3_phone_rules_metrics import SILENCE_LABELS, normalize_phone
from .protocol import CONDITIONS, PROTOCOL_ID, ProtocolError, file_sha256, read_json, write_json, write_jsonl

TRANSFER_MODULE = None


def _transfer_helpers() -> Any:
    global TRANSFER_MODULE
    if TRANSFER_MODULE is None:
        from scripts.experiments import lrs3_english_phoneme_transfer as module

        TRANSFER_MODULE = module
    return TRANSFER_MODULE


def load_cached_tokens(run_dir: Path, model_key: str) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reuse the exact transfer-run vector/token/sample schema."""
    module = _transfer_helpers()
    return module._load_store(run_dir, model_key)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_speech_tokens(rows: Sequence[Mapping[str, Any]], level: str = "phoneme") -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for row in rows:
        if not bool(row.get("speech", False)) or not bool(row.get("valid", True)):
            continue
        if row.get("embedding") is None:
            continue
        vector = np.asarray(row["embedding"], dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 0.0:
            continue
        label = normalize_phone(row.get("label", "")) if level == "phoneme" else str(row.get("viseme", "other"))
        if not label or label.lower() in SILENCE_LABELS:
            continue
        result.append(row)
    return result


def _support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = _valid_speech_tokens(rows, "phoneme")
    counts = Counter(normalize_phone(row.get("label", "")) for row in valid)
    repeated = sum(count >= 2 for count in counts.values())
    return {
        "valid_token_count": len(valid),
        "label_count": len(counts),
        "repeated_label_count": int(repeated),
        "label_histogram": dict(sorted(counts.items())),
    }


def _metric_rows(tokens: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]], model_key: str, layer: int, level: str) -> dict[tuple[str, str], dict[str, Any]]:
    module = _transfer_helpers()
    return module._sample_metric_rows(tokens, samples, model_key, layer, level)


def _scalar_metrics(metric: Mapping[str, Any] | None) -> dict[str, float | None]:
    metric = metric or {}
    return {
        "silhouette": _finite(metric.get("silhouette")),
        "fisher_ratio": _finite(metric.get("fisher_ratio")),
        "intra_class_dist": _finite(metric.get("intra_class_dist")),
        "inter_class_dist": _finite(metric.get("inter_class_dist")),
    }


def compute_sample_features(
    run_dir: Path,
    audit_payload: Mapping[str, Any],
    output_dir: Path,
    *,
    min_duration_s: float = 3.0,
    min_valid_tokens: int = 20,
    min_labels: int = 5,
    min_repeated_labels: int = 2,
) -> dict[str, Any]:
    """Compute every arm's scalar features without re-running SSL or MFA."""
    audit_rows = {str(row["sample_id"]): row for row in audit_payload.get("rows", [])}
    stores: dict[str, tuple[dict[int, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    requested = {"hubert": [6], "xlsr": [10]}
    metric_cache: dict[tuple[str, int, str], dict[tuple[str, str], dict[str, Any]]] = {}
    for model_key, layers in requested.items():
        stores[model_key] = load_cached_tokens(run_dir.parent if run_dir.name == "00_protocol" else run_dir, model_key)
        _vectors, tokens, samples, _meta = stores[model_key]
        for layer in layers:
            for level in ("phoneme", "viseme"):
                metric_cache[(model_key, layer, level)] = _metric_rows(tokens, samples, model_key, layer, level)

    rows: list[dict[str, Any]] = []
    for sid in sorted(audit_rows):
        audit = audit_rows[sid]
        arms: dict[str, Any] = {}
        reasons: list[str] = []
        for condition in CONDITIONS:
            condition_status = audit.get("arms", {}).get(condition, {})
            duration = _finite(condition_status.get("duration_s"))
            arm_features: dict[str, Any] = {
                "condition": condition,
                "duration_s": duration,
                "status": "ready" if condition_status.get("status") == "ready" else "ineligible",
                "reasons": list(condition_status.get("reasons", [])),
                "audio": condition_status.get("audio"),
                "audio_sha256": condition_status.get("audio_sha256_actual"),
                "pcm_sha256": condition_status.get("pcm_sha256"),
                "peak_abs": _finite(condition_status.get("peak_abs")),
                "rms": _finite(condition_status.get("rms")),
                "silence_fraction": _finite(condition_status.get("silence_fraction")),
                "silence_threshold": _finite(condition_status.get("silence_threshold")),
                "textgrid": condition_status.get("textgrid"),
                "textgrid_sha256": condition_status.get("textgrid_sha256"),
            }
            # The cache is the authority for token labels/vector_index.  It is
            # intentionally not reconstructed from a corpus-level pooled score.
            for model_key, layers in requested.items():
                _vectors, tokens, _samples, meta = stores[model_key]
                layer = layers[0]
                sample_tokens = [
                    row for row in tokens
                    if str(row.get("condition")) == condition and str(row.get("sample_id")) == sid and int(row.get("layer", -1)) == layer
                ]
                support = _support(sample_tokens)
                level_metrics = {
                    level: _scalar_metrics(metric_cache[(model_key, layer, level)].get((condition, sid)))
                    for level in ("phoneme", "viseme")
                }
                arm_features[f"{model_key}_layer{layer}"] = {
                    "phoneme": level_metrics["phoneme"],
                    "viseme": level_metrics["viseme"],
                    "support": support,
                    "embedding_dimension": int(meta.get("hidden_size", meta.get("feature_dim", 0)) or 0),
                }
                if model_key == "hubert":
                    arm_features["valid_token_count"] = support["valid_token_count"]
                    arm_features["label_count"] = support["label_count"]
                    arm_features["repeated_label_count"] = support["repeated_label_count"]
                    arm_features["primary_silhouette"] = level_metrics["phoneme"]["silhouette"]
                    arm_features["primary_fisher_ratio"] = level_metrics["phoneme"]["fisher_ratio"]
            if duration is None or duration < min_duration_s:
                arm_features["status"] = "ineligible"
                arm_features["reasons"].append("duration_below_threshold")
            if int(arm_features.get("valid_token_count", 0)) < min_valid_tokens:
                arm_features["status"] = "ineligible"
                arm_features["reasons"].append("valid_token_count_below_threshold")
            if int(arm_features.get("label_count", 0)) < min_labels:
                arm_features["status"] = "ineligible"
                arm_features["reasons"].append("label_count_below_threshold")
            if int(arm_features.get("repeated_label_count", 0)) < min_repeated_labels:
                arm_features["status"] = "ineligible"
                arm_features["reasons"].append("repeated_label_count_below_threshold")
            if arm_features.get("primary_silhouette") is None:
                arm_features["status"] = "ineligible"
                arm_features["reasons"].append("primary_silhouette_nonfinite")
            arms[condition] = arm_features
            if arm_features["status"] != "ready":
                reasons.append(f"{condition}:{','.join(arm_features['reasons'])}")
        primary_values = [arms[condition].get("primary_silhouette") for condition in CONDITIONS]
        eligible = not reasons and all(value is not None and math.isfinite(float(value)) for value in primary_values)
        if not eligible and not reasons:
            reasons.append("primary_feature_nonfinite")
        rows.append({
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "sample_id": sid,
            "source_group": str(audit.get("source_group", "")),
            "transcript_sha256": audit.get("transcript_sha256"),
            "eligible": bool(eligible),
            "reasons": reasons,
            "arms": arms,
            "primary_feature": "hubert_layer6_phoneme_silhouette",
            "primary_values": {condition: arms[condition].get("primary_silhouette") for condition in CONDITIONS},
            "duration_ratios_to_natural": {
                condition: None if arms["natural"].get("duration_s") in (None, 0) or arms[condition].get("duration_s") is None else float(arms[condition]["duration_s"] / arms["natural"]["duration_s"])
                for condition in CONDITIONS[1:]
            },
        })
    eligible_count = sum(bool(row["eligible"]) for row in rows)
    eligible_groups = len({str(row["source_group"]) for row in rows if row["eligible"]})
    meta = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "source": "cached transfer run; no new MFA/SSL/logistic probe",
        "primary_metric": "sklearn.metrics.silhouette_score(metric='cosine') via transfer helper",
        "models": {model: {"layer": layers[0]} for model, layers in requested.items()},
        "conditions": list(CONDITIONS),
        "thresholds": {"min_duration_s": min_duration_s, "min_valid_tokens": min_valid_tokens, "min_labels": min_labels, "min_repeated_labels": min_repeated_labels},
        "sample_count": len(rows),
        "eligible_sample_count": eligible_count,
        "eligible_source_group_count": eligible_groups,
    }
    write_jsonl(output_dir / "features.jsonl", rows)
    write_json(output_dir / "feature_meta.json", meta)
    return {"meta": meta, "rows": rows}


__all__ = ["load_cached_tokens", "compute_sample_features"]
