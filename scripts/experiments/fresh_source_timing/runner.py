from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.experiments.fresh_source_timing"

from .analysis import analyze_records
from .common import (
    DELAY_FRAMES,
    DELAY_SAMPLES,
    PROTOCOL_ID,
    PROTOCOL_REVISION,
    TimingError,
    file_sha256,
    load_self_hashed,
    read_json,
    write_json,
)
from .protocol import (
    build_delay_inputs,
    cohort_is_ready,
    input_is_frozen,
    load_a_cells,
    load_delay_inputs,
)


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_self_hashed(path)
    except TimingError:
        return read_json(path)


def _cohort(run_root: Path) -> dict[str, Any]:
    for path in (run_root / "cohort.json", run_root / "run" / "shared" / "cohort.json"):
        if path.is_file():
            return _read_optional(path) or {}
    raise TimingError(f"cohort.json is missing under {run_root}")


def _write_blocked_artifacts(run_root: Path, *, status: str, reason: Any, upstream: str | None = None) -> int:
    root = run_root / "run" / "C"
    reason_value = reason if isinstance(reason, list) else [str(reason)]
    # A READY delay manifest is immutable after C/prepare.  Preserve it when
    # a later A/GPU or visual binding is unavailable so the run can resume
    # without silently changing the frozen PCM denominator.
    delay_path = root / "delay_inputs.json"
    if not delay_path.is_file():
        write_json(
            delay_path,
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "protocol_revision": PROTOCOL_REVISION,
                "status": status,
                "groups": [],
                "group_count": 0,
                "delay_samples": DELAY_SAMPLES,
                "delay_frames": DELAY_FRAMES,
                "reason": reason_value,
                "scientific_cells": 0,
            },
        )
    write_json(
        root / "analysis.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "status": status,
            "models": {},
            "paired_model_difference": None,
            "groups": [],
            "scientific_cells": 0,
            "reason": reason_value,
        },
    )
    analysis_sha256 = file_sha256(root / "analysis.json")
    write_json(
        root / "validation.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "GO",
            "integrity": "GO",
            "upstream_status": upstream,
            "scientific_cells": 0,
            "errors": [],
        },
    )
    write_json(
        root / "final.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "status": status,
            "replacement_confirmed": False,
            "training_authorized": False,
            "generalization_established": False,
            "scientific_cells": 0,
            "human_status": "pending",
            "reason": reason_value,
            "analysis_sha256": analysis_sha256,
        },
    )
    print(f"C status={status}")
    return 0


def run_blocked(run_root: Path) -> int:
    """Preserve the original blocked path without creating scientific cells."""

    root = run_root.resolve()
    cohort = _cohort(root)
    return _write_blocked_artifacts(
        root,
        status="BLOCKED_UPSTREAM_NEW_SOURCE",
        reason=cohort.get("blockers", [f"cohort status is {cohort.get('status')}"]),
        upstream=str(cohort.get("status")),
    )


def run_delayed(run_root: Path, *, status: str = "DELAYED_UPSTREAM_A", reason: Any = "A seed42 N/DELAY cells are not complete") -> int:
    """Record a safe delayed state after P is ready but A is not complete."""

    root = run_root.resolve()
    cohort = _cohort(root)
    return _write_blocked_artifacts(root, status=status, reason=reason, upstream=str(cohort.get("status")))


def _input_payload(run_root: Path) -> dict[str, Any] | None:
    for path in (run_root / "inputs.json", run_root / "run" / "shared" / "inputs.json"):
        payload = _read_optional(path)
        if payload is not None:
            return payload
    return None


def prepare(run_root: Path) -> int:
    root = run_root.resolve()
    cohort = _cohort(root)
    if not cohort_is_ready(cohort):
        return run_blocked(root)
    inputs = _input_payload(root)
    if inputs is None or not input_is_frozen(inputs):
        return run_delayed(root, status="DELAYED_INPUTS_NOT_FROZEN", reason="COHORT_READY but inputs are not frozen")
    try:
        build_delay_inputs(root, inputs=inputs, cohort=cohort)
    except Exception as exc:  # noqa: BLE001 - terminal engineering state is explicit
        return _write_blocked_artifacts(root, status="BLOCKED_INPUT_BINDING", reason=f"{type(exc).__name__}: {exc}", upstream=str(cohort.get("status")))
    print(f"C delay inputs ready: {root / 'run/C/delay_inputs.json'}")
    return 0


def _load_calibration(root: Path) -> dict[str, Any]:
    path = root / "run" / "B" / "calibration.json"
    value = _read_optional(path)
    if value is None:
        return {"status": "missing", "pass": False}
    records = value.get("records")
    if isinstance(records, list) and records and all(isinstance(item, dict) and "trajectory" in item for item in records):
        from .visual import calibrate_visual_shift

        trajectories = {str(item.get("source_group", index)): item["trajectory"] for index, item in enumerate(records)}
        return calibrate_visual_shift(trajectories)
    return value


def _materialize_features(root: Path, records_by_model: dict[str, list[dict[str, Any]]]) -> None:
    """Copy normalized N42/DELAY trajectories into C-owned feature artifacts.

    A/B inputs remain immutable.  Saving a C-local copy gives the validator a
    stable provenance path and makes it clear that the visual route consumed
    the same N42/DELAY rows as the scorer route.
    """

    from .visual import save_mouth_trajectory

    feature_root = root / "run" / "C" / "features"
    for model, rows in records_by_model.items():
        for row in rows:
            sample_id = str(row.get("sample_id", row.get("source_group", "unknown")))
            for arm, key in (("N42", "natural_trajectory"), ("DELAY42", "delay_trajectory")):
                trajectory = row.get(key)
                if trajectory is None:
                    continue
                target = feature_root / f"{sample_id}__{model}__{arm}.npz"
                binding = save_mouth_trajectory(target, trajectory, metadata={"model": model, "sample_id": sample_id, "arm": arm, "source": "C"})
                row[f"{key}_binding"] = binding


def analyze_run(run_root: Path) -> int:
    root = run_root.resolve()
    try:
        delay_inputs = load_delay_inputs(root)
    except Exception as exc:  # noqa: BLE001
        return run_delayed(root, status="DELAYED_INPUTS_NOT_READY", reason=f"{type(exc).__name__}: {exc}")
    if str(delay_inputs.get("status")) != "READY":
        return run_delayed(root, status=str(delay_inputs.get("status", "DELAYED")), reason=delay_inputs.get("reason", "delay input stage is not ready"))
    a_final = _read_optional(root / "run" / "A" / "final.json")
    if a_final is not None and str(a_final.get("status", "")).lower() in {"blocked", "delayed", "blocked_upstream_new_source", "blocked_ditto"}:
        return run_delayed(root, status="DELAYED_UPSTREAM_A", reason=a_final.get("reason", "A final is blocked"))
    try:
        control_validation = _read_optional(root / "run" / "A" / "control_validation.json")
        if control_validation is not None and str(control_validation.get("status", "")).lower() not in {"go", "complete", "valid", "pass"}:
            raise TimingError("A natural measurement controls are not valid")
        if control_validation is not None and control_validation.get("controls_run") is False:
            raise TimingError("A natural measurement controls were not run")
        records_by_model = load_a_cells(root, delay_inputs)
        _materialize_features(root, records_by_model)
        groups = [str(item["source_group"]) for item in delay_inputs.get("groups", []) if isinstance(item, dict)]
        calibration = _load_calibration(root)
        analysis = analyze_records(records_by_model, groups=groups, calibration=calibration)
        analysis["feature_bindings"] = {
            model: {
                str(row.get("source_group")): {
                    key: row[key]
                    for key in ("natural_trajectory_binding", "delay_trajectory_binding")
                    if key in row
                }
                for row in rows
                if isinstance(row, dict)
            }
            for model, rows in records_by_model.items()
        }
        analysis["delay_inputs_sha256"] = file_sha256(root / "run" / "C" / "delay_inputs.json")
        analysis["a_final_sha256"] = file_sha256(root / "run" / "A" / "final.json") if (root / "run" / "A" / "final.json").is_file() else None
        write_json(root / "run" / "C" / "analysis.json", analysis)
        from .validate import validate_run

        validation = validate_run(root)
        write_json(root / "run" / "C" / "validation.json", validation)
        if validation.get("status") != "GO":
            _write_blocked_artifacts(root, status="CONTROL_FAILED", reason=validation.get("errors", ["independent validation failed"]), upstream="GO")
            return 1
        final = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "protocol_revision": PROTOCOL_REVISION,
            "status": "COMPLETE",
            "replacement_confirmed": False,
            "training_authorized": False,
            "generalization_established": False,
            "human_status": "pending",
            "scientific_cells": sum(len(model.get("records", [])) for model in analysis.get("models", {}).values()),
            "models": {name: model.get("outcome") for name, model in analysis.get("models", {}).items()},
            "analysis_sha256": file_sha256(root / "run" / "C" / "analysis.json"),
            "validation_sha256": file_sha256(root / "run" / "C" / "validation.json"),
            "delay_inputs_sha256": file_sha256(root / "run" / "C" / "delay_inputs.json"),
            "interpretation": "Timing transfer is bounded to the fixed +5-frame integer-delay probe; it does not establish replacement benefit or authorize training.",
        }
        write_json(root / "run" / "C" / "final.json", final)
        print(json.dumps(final, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        _write_blocked_artifacts(root, status="CONTROL_FAILED", reason=f"{type(exc).__name__}: {exc}", upstream="GO")
        return 1


def run(run_root: Path, stage: str = "prepare") -> int:
    root = Path(run_root).resolve()
    if stage == "prepare":
        return prepare(root)
    if stage == "analyze":
        return analyze_run(root)
    raise ValueError(f"unknown C stage: {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fresh-source integer-delay timing-transfer branch")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "analyze"), default="prepare")
    args = parser.parse_args(argv)
    return run(args.run_root, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
