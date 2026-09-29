from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.experiments.fresh_source_timing"

from .audio import build_delay_pcm, read_pcm16_wav, verify_source_index_map
from .common import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DELAY_FRAMES,
    DELAY_SAMPLES,
    EXPECTED_GROUP_COUNT,
    PROTOCOL_ID,
    PROTOCOL_REVISION,
    TimingError,
    file_sha256,
    load_self_hashed,
    read_json,
    resolve_path,
    write_json,
)
from .protocol import load_a_cells, load_delay_inputs
from .scoring import score_embeddings
from .stats import bootstrap_group_ci, bootstrap_indices, paired_model_difference
from .visual import visual_transfer_score


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_self_hashed(path)
    except TimingError:
        return read_json(path)


def _close(actual: Any, expected: Any, *, tolerance: float = 1e-6) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        try:
            return bool(np.isfinite(float(actual)) and np.isfinite(float(expected)) and abs(float(actual) - float(expected)) <= tolerance)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return isinstance(actual, Sequence) and len(actual) == len(expected) and all(_close(a, e, tolerance=tolerance) for a, e in zip(actual, expected, strict=True))
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(key in actual and _close(actual[key], value, tolerance=tolerance) for key, value in expected.items())
    return actual == expected


def _verify_delay_audio(run_root: Path, delay_inputs: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(delay_inputs.get("status", "")) != "READY":
        # A blocked/delayed run intentionally has no scientific audio cells;
        # its empty group list is a safe terminal state, not a malformed DELAY
        # manifest.  The caller separately checks that no positive flags/cells
        # were written.
        return errors
    groups = delay_inputs.get("groups")
    if not isinstance(groups, list) or len(groups) != EXPECTED_GROUP_COUNT:
        return [f"delay group count is not {EXPECTED_GROUP_COUNT}"]
    if delay_inputs.get("delay_samples") != DELAY_SAMPLES or delay_inputs.get("delay_frames") != DELAY_FRAMES:
        errors.append("fixed delay parameters changed")
    seen: set[tuple[str, str]] = set()
    for item in groups:
        if not isinstance(item, Mapping):
            errors.append("delay group row is malformed")
            continue
        sample_id = str(item.get("sample_id", ""))
        group = str(item.get("source_group", ""))
        key = (group, sample_id)
        if key in seen:
            errors.append(f"duplicate delay group: {key}")
            continue
        seen.add(key)
        try:
            natural_item = item["natural"]
            delay_item = item["delay"]
            map_item = item["source_index_map"]
            natural_path = resolve_path(natural_item["path"])
            delay_path = resolve_path(delay_item["path"])
            map_path = resolve_path(map_item["path"])
            if file_sha256(natural_path) != natural_item.get("sha256"):
                raise TimingError("natural container hash mismatch")
            if file_sha256(delay_path) != delay_item.get("sha256"):
                raise TimingError("delay container hash mismatch")
            if file_sha256(map_path) != map_item.get("sha256"):
                raise TimingError("source-index map hash mismatch")
            natural_raw, natural_values, natural_meta = read_pcm16_wav(natural_path)
            delay_raw, delay_values, delay_meta = read_pcm16_wav(delay_path)
            if natural_meta["pcm_sha256"] != natural_item.get("pcm_sha256") or delay_meta["pcm_sha256"] != delay_item.get("pcm_sha256"):
                raise TimingError("PCM hash binding mismatch")
            if natural_values.size != delay_values.size or int(item.get("delay_samples", DELAY_SAMPLES)) != DELAY_SAMPLES:
                raise TimingError("DELAY length or shift binding changed")
            with map_path.open("rb") as handle:
                mapping = np.asarray(np.load(handle, allow_pickle=False), dtype="<i8")
            verify_source_index_map(mapping, length=natural_values.size, shift_samples=DELAY_SAMPLES)
            rebuilt, expected_map = build_delay_pcm(natural_values, shift_samples=DELAY_SAMPLES)
            if rebuilt != delay_raw or not np.array_equal(mapping, expected_map):
                raise TimingError("DELAY PCM/source-index map reconstruction differs")
            if bytes(natural_raw) == bytes(delay_raw) and natural_values.size > DELAY_SAMPLES:
                # This is not impossible for an all-zero source, so only record
                # the exact contract above; no amplitude heuristic is applied.
                pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{group}/{sample_id}: {type(exc).__name__}: {exc}")
    if len(seen) != len(groups):
        errors.append("delay group identity is not unique")
    return errors


def _bootstrap_summary(values: Mapping[str, float], labels: list[str], indices: np.ndarray) -> dict[str, Any] | None:
    if set(values) != set(labels):
        return None
    summary = bootstrap_group_ci(values, indices=indices, level=0.99)
    summary.update(
        {
            "metric": "independent",
            "observed_group_count": len(values),
            "missing_group_count": 0,
            "missing_groups": [],
        }
    )
    return summary


def _metric_section_close(actual: Any, expected: Any) -> bool:
    """Compare the signed producer summary to independent raw-array fields.

    The producer intentionally omits implementation-only fields such as the
    full ``ci`` alias and ``level``; scientific values and denominators remain
    mandatory and are compared here.
    """

    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    required = (
        "mean",
        "ci99",
        "group_means",
        "group_labels",
        "group_count",
        "observed_group_count",
        "missing_group_count",
        "missing_groups",
        "group_positive_count",
    )
    return all(key in actual and key in expected and _close(actual[key], expected[key], tolerance=1e-6) for key in required)


def _independent_model(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    labels: list[str],
    indices: np.ndarray,
    calibration: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    by_group = {str(row.get("source_group", "")): row for row in rows if isinstance(row, Mapping)}
    records: list[dict[str, Any]] = []
    offsets: dict[str, float] = {}
    rescues: dict[str, float] = {}
    visuals: dict[str, float] = {}
    residuals: dict[str, float] = {}
    scorer_valid = 0
    visual_valid = 0
    for label in labels:
        row = by_group.get(label)
        if row is None or not bool(row.get("fresh_forward", row.get("new_forward", False))):
            records.append({"source_group": label, "status": "missing", "scorer": {"status": "SCORER_INPUT_UNRESOLVED"}, "visual": {"status": "VISUAL_RESPONSE_UNRESOLVED"}})
            continue
        record: dict[str, Any] = {"sample_id": str(row.get("sample_id", label)), "source_group": label, "status": "complete"}
        if row.get("natural_control_valid", row.get("a_control_valid", True)) is False:
            record["status"] = "SCORER_CONTROL_FAILED"
            record["scorer"] = {"status": "SCORER_CONTROL_FAILED", "reason": "A natural N42 measurement control is not valid"}
            try:
                visual = visual_transfer_score(row["natural_trajectory"], row["delay_trajectory"])
                record["visual"] = visual
                if visual.get("valid"):
                    visuals[label] = float(visual["v"])
                    visual_valid += 1
            except Exception as exc:  # noqa: BLE001
                record["visual"] = {"status": "VISUAL_RESPONSE_UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            continue
        try:
            score = score_embeddings(row["natural_visual"], row["delay_visual"], row["natural_audio"])
            offsets[label] = float(score["offset_delta"])
            rescues[label] = float(score["rescue"])
            residuals[label] = float(score["residual"])
            scorer_valid += 1
            record["scorer"] = {key: value for key, value in score.items() if key not in {"matrix_natural", "matrix_delay"}}
            record["scorer"]["status"] = "complete"
        except Exception as exc:  # noqa: BLE001
            record["scorer"] = {"status": "SCORER_RESPONSE_UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}"}
            record["status"] = "SCORER_RESPONSE_UNRESOLVED"
        try:
            visual = visual_transfer_score(row["natural_trajectory"], row["delay_trajectory"])
            record["visual"] = visual
            if visual.get("valid"):
                visuals[label] = float(visual["v"])
                visual_valid += 1
            elif record["status"] == "complete":
                record["status"] = "VISUAL_RESPONSE_UNRESOLVED"
        except Exception as exc:  # noqa: BLE001
            record["visual"] = {"status": "VISUAL_RESPONSE_UNRESOLVED", "reason": f"{type(exc).__name__}: {exc}"}
            if record["status"] == "complete":
                record["status"] = "VISUAL_RESPONSE_UNRESOLVED"
        records.append(record)
    offset_summary = _bootstrap_summary(offsets, labels, indices)
    rescue_summary = _bootstrap_summary(rescues, labels, indices)
    visual_summary = _bootstrap_summary(visuals, labels, indices)
    residual_summary = _bootstrap_summary(residuals, labels, indices)
    rescue_ci_lower = float(rescue_summary["ci99"][0]) if rescue_summary else None
    visual_ci_lower = float(visual_summary["ci99"][0]) if visual_summary else None
    visual_mean = float(visual_summary["mean"]) if visual_summary else None
    scorer_gate = {
        "controls_valid": scorer_valid == len(labels),
        "all_groups_observed": scorer_valid == len(labels),
        "observed_group_count": scorer_valid,
        "expected_group_count": len(labels),
        "offset_change_within_one_count": sum(abs(value - DELAY_FRAMES) <= 1 for value in offsets.values()),
        "offset_required_count": 10,
        "rescue_positive_count": sum(value > 0.0 for value in rescues.values()),
        "rescue_required_count": 10,
        "rescue_ci99_lower": rescue_ci_lower,
        "pass": bool(scorer_valid == len(labels) and sum(abs(value - DELAY_FRAMES) <= 1 for value in offsets.values()) >= 10 and sum(value > 0.0 for value in rescues.values()) >= 10 and rescue_ci_lower is not None and rescue_ci_lower > 0.0),
    }
    calibration_pass = bool(calibration and calibration.get("pass", False))
    visual_gate = {
        "calibration_pass": calibration_pass,
        "all_groups_observed": visual_valid == len(labels),
        "observed_group_count": visual_valid,
        "expected_group_count": len(labels),
        "mean_v": visual_mean,
        "v_threshold": 0.02,
        "v_ci99_lower": visual_ci_lower,
        "positive_count": sum(value > 0.0 for value in visuals.values()),
        "positive_required_count": 10,
        "pass": bool(calibration_pass and visual_valid == len(labels) and visual_mean is not None and visual_mean > 0.02 and visual_ci_lower is not None and visual_ci_lower > 0.0 and sum(value > 0.0 for value in visuals.values()) >= 10),
    }
    if visual_valid != len(labels):
        outcome = "VISUAL_RESPONSE_UNRESOLVED"
    elif scorer_valid != len(labels) or not calibration_pass:
        outcome = "CONTROL_FAILED"
    elif scorer_gate["pass"] and visual_gate["pass"]:
        outcome = "TIMING_TRANSFER_OBSERVED"
    elif scorer_gate["pass"] != visual_gate["pass"]:
        outcome = "SCORER_VISUAL_DISAGREEMENT"
    else:
        outcome = "RESPONSE_NOT_ESTABLISHED"
    expected = {
        "model": model,
        "groups": labels,
        "records": records,
        "scorer": {
            "offset_delta": offset_summary,
            "rescue": rescue_summary,
            "residual": residual_summary,
            "gate": scorer_gate,
        },
        "visual": {"v": visual_summary, "gate": visual_gate},
        "outcome": outcome,
        "status": outcome,
    }
    return expected, visuals, {"rescue": rescues, "offset": offsets, "residual": residuals}


def _check_analysis(run_root: Path, delay_inputs: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if analysis.get("protocol_id") != PROTOCOL_ID or analysis.get("protocol_revision") != PROTOCOL_REVISION or analysis.get("status") != "complete":
        errors.append("analysis protocol/status is invalid")
        return errors
    labels = sorted(str(item.get("source_group")) for item in delay_inputs.get("groups", []) if isinstance(item, Mapping))
    if labels != sorted(str(value) for value in analysis.get("groups", [])) or len(labels) != EXPECTED_GROUP_COUNT:
        errors.append("analysis group denominator changed")
        return errors
    bootstrap_payload = analysis.get("bootstrap")
    _expected_labels, expected_indices = bootstrap_indices(labels, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    if not isinstance(bootstrap_payload, Mapping) or bootstrap_payload.get("seed") != BOOTSTRAP_SEED or bootstrap_payload.get("draws") != BOOTSTRAP_DRAWS or bootstrap_payload.get("rng") != "numpy.PCG64" or bootstrap_payload.get("quantile_method") != "linear":
        errors.append("bootstrap configuration changed")
    else:
        actual_indices = np.asarray(bootstrap_payload.get("indices"), dtype=np.int64)
        if actual_indices.shape != expected_indices.shape or not np.array_equal(actual_indices, expected_indices):
            errors.append("bootstrap indices changed")
    try:
        records_by_model = load_a_cells(run_root, delay_inputs)
    except Exception as exc:  # noqa: BLE001
        return errors + [f"A binding failed: {type(exc).__name__}: {exc}"]
    models_payload = analysis.get("models")
    if not isinstance(models_payload, Mapping):
        return errors + ["analysis models are missing"]
    calibration = _read_optional(run_root / "run" / "B" / "calibration.json") or {"pass": False}
    # A B calibration's signed pass is a prerequisite; the trajectory-level
    # recomputation is done by B and C does not use it to choose a model.
    calibration_status = str(calibration.get("status", "")).upper()
    normalized_calibration = {
        "pass": bool(
            calibration.get("pass", False)
            or (
                calibration.get("calibrated", False)
                and calibration_status in {"METRIC_CALIBRATED", "CALIBRATED", "PASS", "GO", "COMPLETE"}
            )
        )
    }
    feature_bindings = analysis.get("feature_bindings", {})
    if not isinstance(feature_bindings, Mapping):
        errors.append("C feature bindings are missing")
        feature_bindings = {}
    else:
        from .visual import load_mouth_trajectory

        for model, rows in records_by_model.items():
            model_bindings = feature_bindings.get(model, {})
            if not isinstance(model_bindings, Mapping):
                errors.append(f"C feature bindings are malformed: {model}")
                continue
            for row in rows:
                group = str(row.get("source_group", ""))
                bindings = model_bindings.get(group, {})
                if not isinstance(bindings, Mapping):
                    if row.get("natural_trajectory") is not None or row.get("delay_trajectory") is not None:
                        errors.append(f"C feature binding missing: {model}/{group}")
                    continue
                for key, array_key in (("natural_trajectory_binding", "natural_trajectory"), ("delay_trajectory_binding", "delay_trajectory")):
                    if row.get(array_key) is None:
                        continue
                    item = bindings.get(key)
                    if not isinstance(item, Mapping):
                        errors.append(f"C feature binding missing: {model}/{group}/{key}")
                        continue
                    try:
                        path = resolve_path(item.get("path"))
                        if file_sha256(path) != item.get("sha256"):
                            raise TimingError("feature hash differs")
                        loaded = load_mouth_trajectory(path)
                        expected_array = np.asarray(row[array_key], dtype=np.float64).reshape(np.asarray(row[array_key]).shape[0], -1)
                        if not np.array_equal(loaded["trajectory"], expected_array):
                            raise TimingError("feature values differ")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"C feature binding invalid: {model}/{group}/{key}: {type(exc).__name__}: {exc}")
    recomputed: dict[str, dict[str, Any]] = {}
    visual_values: dict[str, dict[str, float]] = {}
    for model, rows in records_by_model.items():
        expected, values, _other = _independent_model(model, rows, labels, expected_indices, normalized_calibration)
        recomputed[model] = expected
        visual_values[model] = values
        produced = models_payload.get(model)
        if not isinstance(produced, Mapping):
            errors.append(f"analysis model is missing: {model}")
            continue
        # Compare the independently reconstructed scientific fields while
        # allowing the producer to include explanatory metadata in records.
        for key in ("outcome", "status", "groups"):
            if produced.get(key) != expected.get(key):
                errors.append(f"analysis {model}.{key} differs")
        for section, metric_names in (("scorer", ("offset_delta", "rescue", "residual")), ("visual", ("v",))):
            produced_section = produced.get(section)
            expected_section = expected.get(section)
            if not isinstance(produced_section, Mapping) or not isinstance(expected_section, Mapping):
                errors.append(f"analysis {model}.{section} is malformed")
                continue
            for metric_name in metric_names:
                if not _metric_section_close(produced_section.get(metric_name), expected_section.get(metric_name)):
                    errors.append(f"analysis {model}.{section}.{metric_name} differs from raw-array reconstruction")
            if not _close(produced_section.get("gate"), expected_section.get("gate"), tolerance=1e-6):
                errors.append(f"analysis {model}.{section}.gate differs from raw-array reconstruction")
        produced_records = produced.get("records")
        expected_records = expected.get("records")
        if not isinstance(produced_records, list) or len(produced_records) != len(expected_records):
            errors.append(f"analysis {model} record denominator changed")
        else:
            for index, expected_record in enumerate(expected_records):
                if not _close(produced_records[index], expected_record, tolerance=1e-5):
                    errors.append(f"analysis {model} record {index} differs")
    if len(recomputed) >= 2:
        names = list(recomputed)
        left_values = visual_values[names[0]]
        right_values = visual_values[names[1]]
        paired = analysis.get("paired_model_difference")
        if set(left_values) == set(labels) and set(right_values) == set(labels):
            expected_pair = paired_model_difference(left_values, right_values, indices=expected_indices, level=0.99)
            if not isinstance(paired, Mapping) or paired.get("status") != "complete" or not _close(paired.get("ci99"), expected_pair.get("ci99"), tolerance=1e-6) or not _close(paired.get("mean"), expected_pair.get("mean"), tolerance=1e-6):
                errors.append("paired model difference differs")
    return errors


def validate_run(run_root: Path) -> dict[str, Any]:
    """Independently validate C artifacts; scientific negatives still return GO."""

    root = Path(run_root).resolve()
    errors: list[str] = []
    try:
        delay_inputs = load_delay_inputs(root)
    except Exception as exc:  # noqa: BLE001
        return {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "NO_GO", "integrity": "NO_GO", "errors": [f"delay_inputs: {type(exc).__name__}: {exc}"]}
    errors.extend(_verify_delay_audio(root, delay_inputs))
    control_validation = _read_optional(root / "run" / "A" / "control_validation.json")
    upstream_control_failed = bool(
        control_validation is not None
        and (
            str(control_validation.get("status", "")).lower() not in {"go", "complete", "valid", "pass"}
            or control_validation.get("controls_run") is False
        )
    )
    final = _read_optional(root / "run" / "C" / "final.json")
    analysis = _read_optional(root / "run" / "C" / "analysis.json")
    status = str(delay_inputs.get("status", ""))
    if status != "READY":
        # Blocked/delayed artifacts are valid only if they never claim cells or
        # a positive scientific terminal.  This preserves the pre-COHORT path.
        for payload_name, payload in (("analysis", analysis), ("final", final)):
            if payload is None:
                errors.append(f"{payload_name} is missing")
                continue
            if int(payload.get("scientific_cells", 0) or 0) != 0:
                errors.append(f"{payload_name} contains scientific cells")
            if payload.get("replacement_confirmed") is True or payload.get("training_authorized") is True or payload.get("generalization_established") is True:
                errors.append(f"{payload_name} contains a positive terminal flag")
        return {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "status": "GO" if not errors else "NO_GO",
            "integrity": "GO" if not errors else "NO_GO",
            "independent": True,
            "scientific_cells": 0,
            "errors": errors,
        }
    terminal_control_failed = (
        upstream_control_failed
        and isinstance(analysis, Mapping)
        and isinstance(final, Mapping)
        and str(analysis.get("status")) in {"CONTROL_FAILED", "BLOCKED"}
        and str(final.get("status")) == str(analysis.get("status"))
        and int(analysis.get("scientific_cells", 0) or 0) == 0
        and int(final.get("scientific_cells", 0) or 0) == 0
    )
    if terminal_control_failed:
        if final.get("analysis_sha256") != file_sha256(root / "run" / "C" / "analysis.json"):
            errors.append("final analysis hash binding differs")
        for key in ("replacement_confirmed", "training_authorized", "generalization_established"):
            if final.get(key) is not False:
                errors.append(f"final.{key} must remain false")
        return {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "status": "GO" if not errors else "NO_GO",
            "integrity": "GO" if not errors else "NO_GO",
            "independent": True,
            "scientific_status": str(final.get("status")),
            "scientific_cells": 0,
            "errors": errors,
        }
    if analysis is None:
        errors.append("analysis is missing")
    else:
        errors.extend(_check_analysis(root, delay_inputs, analysis))
    if final is None:
        errors.append("final.json is missing")
    else:
        for key in ("replacement_confirmed", "training_authorized", "generalization_established"):
            if final.get(key) is not False:
                errors.append(f"final.{key} must remain false")
        if analysis is not None and final.get("analysis_sha256") != file_sha256(root / "run" / "C" / "analysis.json"):
            errors.append("final analysis hash binding differs")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_revision": PROTOCOL_REVISION,
        "status": "GO" if not errors else "NO_GO",
        "integrity": "GO" if not errors else "NO_GO",
        "independent": True,
        "scientific_cells": int(
            sum(
                sum(1 for row in model.get("records", []) if isinstance(row, Mapping) and row.get("status") == "complete")
                for model in (analysis or {}).get("models", {}).values()
                if isinstance(model, Mapping)
            )
        ),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root.resolve())
    write_json(args.run_root.resolve() / "run" / "C" / "validation_independent.json", result)
    print(result)
    return 0 if result["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
