"""Producer-side CPU analysis for the C timing-transfer branch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .common import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DELAY_FRAMES,
    PROTOCOL_ID,
    PROTOCOL_REVISION,
    U_ROWS,
    TimingError,
    assert_finite,
)
from .scoring import score_embeddings
from .stats import bootstrap_group_ci, bootstrap_indices, paired_model_difference
from .visual import visual_transfer_score


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _record_arrays(record: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    """Resolve direct synthetic arrays or bindings normalized by the runner."""

    natural_visual = _first(record, ("natural_visual", "visual_n", "N_visual", "natural_visual_embedding"))
    delay_visual = _first(record, ("delay_visual", "visual_delay", "DELAY_visual", "delay_visual_embedding"))
    natural_audio = _first(record, ("natural_audio_embedding", "natural_audio", "audio_n", "N_audio"))
    natural_trajectory = _first(record, ("natural_trajectory", "trajectory_n", "N_trajectory", "natural_features"))
    delay_trajectory = _first(record, ("delay_trajectory", "trajectory_delay", "DELAY_trajectory", "delay_features"))
    # A normalized record may keep all arrays under ``arrays``.
    arrays = record.get("arrays")
    if isinstance(arrays, Mapping):
        natural_visual = natural_visual if natural_visual is not None else _first(arrays, ("natural_visual", "visual_n", "N_visual"))
        delay_visual = delay_visual if delay_visual is not None else _first(arrays, ("delay_visual", "visual_delay", "DELAY_visual"))
        natural_audio = natural_audio if natural_audio is not None else _first(arrays, ("natural_audio", "audio_n", "N_audio"))
        natural_trajectory = natural_trajectory if natural_trajectory is not None else _first(arrays, ("natural_trajectory", "trajectory_n", "N_trajectory"))
        delay_trajectory = delay_trajectory if delay_trajectory is not None else _first(arrays, ("delay_trajectory", "trajectory_delay", "DELAY_trajectory"))
    return natural_visual, delay_visual, natural_audio, natural_trajectory, delay_trajectory


def _summary_or_missing(
    values: Mapping[str, float],
    labels: Sequence[str],
    indices: np.ndarray,
    *,
    metric: str,
) -> dict[str, Any]:
    missing = [label for label in labels if label not in values]
    result: dict[str, Any] = {
        "metric": metric,
        "mean": None,
        "ci99": None,
        "group_means": {label: float(values[label]) for label in sorted(values)},
        "group_labels": list(labels),
        "group_count": len(labels),
        "observed_group_count": len(values),
        "missing_group_count": len(missing),
        "missing_groups": missing,
        "group_positive_count": int(sum(float(value) > 0.0 for value in values.values())),
        "draws": int(indices.shape[0]),
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.PCG64",
        "quantile_method": "linear",
    }
    if not missing:
        summary = bootstrap_group_ci(values, indices=indices, level=0.99)
        result.update({"mean": summary["mean"], "ci99": summary["ci99"], "group_means": summary["group_means"]})
    else:
        # Do not silently convert missing groups into zeros.  An observed-only
        # interval is useful diagnostically, but its denominator is explicit.
        if values:
            observed_labels = sorted(values)
            observed_indices = bootstrap_indices(observed_labels, draws=int(indices.shape[0]), seed=BOOTSTRAP_SEED)[1]
            observed = bootstrap_group_ci(values, indices=observed_indices, level=0.99)
            result["observed_mean"] = observed["mean"]
            result["observed_ci99"] = observed["ci99"]
    return result


def _group_rows(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TimingError("model record is not an object")
        group = str(row.get("source_group", row.get("group", "")))
        if not group or group in result:
            raise TimingError(f"duplicate or missing source group in model records: {group!r}")
        result[group] = row
    outside = sorted(set(result) - set(labels))
    if outside:
        raise TimingError(f"model records contain groups outside frozen cohort: {outside}")
    return result


def _calibration_pass(calibration: Mapping[str, Any] | None) -> bool:
    """Accept B's ``calibrated/status`` schema and C's legacy ``pass`` key."""

    if not isinstance(calibration, Mapping):
        return False
    status = str(calibration.get("status", "")).upper()
    status_pass = status in {"METRIC_CALIBRATED", "CALIBRATED", "PASS", "GO", "COMPLETE"}
    explicit = []
    for key in ("pass", "calibrated"):
        if key in calibration:
            explicit.append(bool(calibration[key]))
    return bool(explicit and all(explicit) and (status_pass or any(explicit)) or not explicit and status_pass)


def analyze_model(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    groups: Sequence[str],
    bootstrap: np.ndarray,
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    labels = sorted({str(group) for group in groups})
    indexed = _group_rows(rows, labels)
    per_group: list[dict[str, Any]] = []
    offset_values: dict[str, float] = {}
    rescue_values: dict[str, float] = {}
    residual_values: dict[str, float] = {}
    visual_values: dict[str, float] = {}
    scorer_valid_groups: list[str] = []
    visual_valid_groups: list[str] = []
    for label in labels:
        source = indexed.get(label)
        if source is None:
            per_group.append({"source_group": label, "status": "missing", "scorer": {"status": "SCORER_RESPONSE_UNRESOLVED"}, "visual": {"status": "VISUAL_RESPONSE_UNRESOLVED"}})
            continue
        sample_id = str(source.get("sample_id", source.get("id", label)))
        item: dict[str, Any] = {"sample_id": sample_id, "source_group": label, "status": "complete"}
        fresh = bool(source.get("fresh_forward", source.get("new_forward", source.get("delay_fresh_forward", False))))
        if not fresh:
            item["status"] = "SCORER_INPUT_UNRESOLVED"
            item["scorer"] = {"status": "SCORER_INPUT_UNRESOLVED", "reason": "DELAY row is not marked as a fresh generator forward"}
            item["visual"] = {"status": "VISUAL_RESPONSE_UNRESOLVED", "reason": "paired DELAY forward is not bound"}
            per_group.append(item)
            continue
        if source.get("natural_control_valid", source.get("a_control_valid", True)) is False:
            item["status"] = "SCORER_CONTROL_FAILED"
            item["scorer"] = {"status": "SCORER_CONTROL_FAILED", "reason": "A natural N42 measurement control is not valid"}
            # Keep the visual route independent; it can still report an
            # unresolved/observed trajectory without allowing scorer passage.
            natural_visual, delay_visual, natural_audio, natural_trajectory, delay_trajectory = _record_arrays(source)
            if natural_trajectory is None or delay_trajectory is None:
                item["visual"] = {"status": "VISUAL_RESPONSE_UNRESOLVED", "reason": "canonical N42/DELAY mouth features are missing"}
            else:
                try:
                    visual = visual_transfer_score(natural_trajectory, delay_trajectory)
                    item["visual"] = visual
                    if visual.get("valid"):
                        visual_values[label] = float(visual["v"])
                        visual_valid_groups.append(label)
                except Exception as exc:  # noqa: BLE001
                    item["visual"] = {"status": "VISUAL_RESPONSE_UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}"}
            per_group.append(item)
            continue
        natural_visual, delay_visual, natural_audio, natural_trajectory, delay_trajectory = _record_arrays(source)
        if natural_visual is None or delay_visual is None or natural_audio is None:
            item["status"] = "SCORER_RESPONSE_UNRESOLVED"
            item["scorer"] = {"status": "SCORER_RESPONSE_UNRESOLVED", "reason": "N42/DELAY/N audio embeddings are missing"}
        else:
            try:
                scorer = score_embeddings(natural_visual, delay_visual, natural_audio)
                item["scorer"] = {
                    key: value
                    for key, value in scorer.items()
                    if key not in {"matrix_natural", "matrix_delay"}
                }
                item["scorer"]["status"] = "complete"
                offset_values[label] = float(scorer["offset_delta"])
                rescue_values[label] = float(scorer["rescue"])
                residual_values[label] = float(scorer["residual"])
                scorer_valid_groups.append(label)
            except Exception as exc:  # noqa: BLE001 - preserve missingness per group
                item["status"] = "SCORER_RESPONSE_UNRESOLVED"
                item["scorer"] = {"status": "SCORER_RESPONSE_UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}"}
        if natural_trajectory is None or delay_trajectory is None:
            item["visual"] = {"status": "VISUAL_RESPONSE_UNRESOLVED", "reason": "canonical N42/DELAY mouth features are missing"}
            if item["status"] == "complete":
                item["status"] = "VISUAL_RESPONSE_UNRESOLVED"
        else:
            try:
                visual = visual_transfer_score(natural_trajectory, delay_trajectory)
                item["visual"] = visual
                if visual.get("valid"):
                    visual_values[label] = float(visual["v"])
                    visual_valid_groups.append(label)
                elif item["status"] == "complete":
                    item["status"] = "VISUAL_RESPONSE_UNRESOLVED"
            except Exception as exc:  # noqa: BLE001 - retain group in denominator
                item["visual"] = {"status": "VISUAL_RESPONSE_UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}"}
                if item["status"] == "complete":
                    item["status"] = "VISUAL_RESPONSE_UNRESOLVED"
        per_group.append(item)

    offset_summary = _summary_or_missing(offset_values, labels, bootstrap, metric="offset_delta")
    rescue_summary = _summary_or_missing(rescue_values, labels, bootstrap, metric="rescue")
    residual_summary = _summary_or_missing(residual_values, labels, bootstrap, metric="residual")
    visual_summary = _summary_or_missing(visual_values, labels, bootstrap, metric="v")
    offset_pass_count = int(sum(abs(float(value) - DELAY_FRAMES) <= 1.0 for value in offset_values.values()))
    rescue_positive_count = int(sum(float(value) > 0.0 for value in rescue_values.values()))
    scorer_controls_valid = bool(len(scorer_valid_groups) == len(labels))
    scorer_ci = rescue_summary.get("ci99")
    scorer_ci_lower = float(scorer_ci[0]) if isinstance(scorer_ci, Sequence) and len(scorer_ci) == 2 else None
    scorer_gate = {
        "controls_valid": scorer_controls_valid,
        "all_groups_observed": len(scorer_valid_groups) == len(labels),
        "observed_group_count": len(scorer_valid_groups),
        "expected_group_count": len(labels),
        "offset_change_within_one_count": offset_pass_count,
        "offset_required_count": 10,
        "rescue_positive_count": rescue_positive_count,
        "rescue_required_count": 10,
        "rescue_ci99_lower": scorer_ci_lower,
        "pass": bool(scorer_controls_valid and offset_pass_count >= 10 and rescue_positive_count >= 10 and scorer_ci_lower is not None and scorer_ci_lower > 0.0),
    }
    calibration_pass = _calibration_pass(calibration)
    visual_ci = visual_summary.get("ci99")
    visual_ci_lower = float(visual_ci[0]) if isinstance(visual_ci, Sequence) and len(visual_ci) == 2 else None
    visual_mean = visual_summary.get("mean")
    visual_gate = {
        "calibration_pass": calibration_pass,
        "all_groups_observed": len(visual_valid_groups) == len(labels),
        "observed_group_count": len(visual_valid_groups),
        "expected_group_count": len(labels),
        "mean_v": visual_mean,
        "v_threshold": 0.02,
        "v_ci99_lower": visual_ci_lower,
        "positive_count": int(sum(float(value) > 0.0 for value in visual_values.values())),
        "positive_required_count": 10,
        "pass": bool(calibration_pass and len(visual_valid_groups) == len(labels) and visual_mean is not None and float(visual_mean) > 0.02 and visual_ci_lower is not None and visual_ci_lower > 0.0 and sum(float(value) > 0.0 for value in visual_values.values()) >= 10),
    }
    if len(visual_valid_groups) != len(labels):
        # Keep the explicit unresolved status visible; it is not a zero score.
        outcome = "VISUAL_RESPONSE_UNRESOLVED"
    elif not scorer_controls_valid or not calibration_pass:
        outcome = "CONTROL_FAILED"
    elif scorer_gate["pass"] and visual_gate["pass"]:
        outcome = "TIMING_TRANSFER_OBSERVED"
    elif scorer_gate["pass"] != visual_gate["pass"]:
        outcome = "SCORER_VISUAL_DISAGREEMENT"
    else:
        outcome = "RESPONSE_NOT_ESTABLISHED"
    return {
        "model": str(model),
        "groups": labels,
        "records": per_group,
        "scorer": {
            "offset_delta": offset_summary,
            "rescue": rescue_summary,
            "residual": residual_summary,
            "gate": scorer_gate,
        },
        "visual": {"v": visual_summary, "gate": visual_gate},
        "calibration": dict(calibration or {"pass": False, "status": "missing"}),
        "outcome": outcome,
        "status": outcome,
        "interpretation": "A failed transfer gate is a scope limitation; it is not evidence that the generator ignores audio.",
    }


def analyze_records(
    records_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    groups: Sequence[str] | None = None,
    calibration: Mapping[str, Any] | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze synthetic or manifest-normalized C records on fixed support."""

    if not records_by_model:
        raise TimingError("C analysis requires at least one model")
    labels = sorted({str(group) for group in (groups or [])})
    if not labels:
        labels = sorted({str(row.get("source_group", row.get("group", ""))) for rows in records_by_model.values() for row in rows if row.get("source_group", row.get("group"))})
    if not labels:
        raise TimingError("C analysis has no source groups")
    bootstrap_labels, indices = bootstrap_indices(labels, draws=draws, seed=seed)
    if bootstrap_labels != labels:
        raise TimingError("source-group order changed before bootstrap")
    models = {
        str(model): analyze_model(str(model), rows, groups=labels, bootstrap=indices, calibration=calibration)
        for model, rows in records_by_model.items()
    }
    paired: dict[str, Any] = {"status": "unresolved", "models": []}
    model_names = list(models)
    if len(model_names) >= 2:
        left, right = model_names[0], model_names[1]
        left_values = models[left]["visual"]["v"].get("group_means", {})
        right_values = models[right]["visual"]["v"].get("group_means", {})
        common = sorted(set(left_values) & set(right_values))
        if len(common) == len(labels):
            diff = paired_model_difference(left_values, right_values, indices=indices, level=0.99)
            paired = {
                "status": "complete",
                "left_model": left,
                "right_model": right,
                "definition": f"{left} - {right}",
                **diff,
                "ci99": diff.get("ci99"),
            }
        else:
            paired.update({"left_model": left, "right_model": right, "missing_groups": sorted(set(labels) - set(common)), "observed_group_count": len(common), "group_count": len(labels)})
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "status": "complete",
        "groups": labels,
        "group_count": len(labels),
        "u_rows": list(U_ROWS),
        "natural_lags": list(range(-15, 16)),
        "delay_lags": list(range(-20, 11)),
        "compensation_frames": DELAY_FRAMES,
        "bootstrap": {
            "seed": int(seed),
            "draws": int(draws),
            "rng": "numpy.PCG64",
            "quantile_method": "linear",
            "unit": "source_group",
            "labels": labels,
            "indices": indices.tolist(),
        },
        "models": models,
        "paired_model_difference": paired,
        "replacement_confirmed": False,
        "training_authorized": False,
        "generalization_established": False,
    }
    assert_finite({key: value for key, value in result.items() if key != "bootstrap"})
    return result


def analyze(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return analyze_records(*args, **kwargs)
