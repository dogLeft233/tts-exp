#!/usr/bin/env python3
"""Materialize the already-fixed terminal step 120 as a V3 method candidate.

This is a metadata-only operation. It reads a completed fit receipt, checkpoint,
and training record; copies the latter two byte-for-byte; writes a fixed-step
selection record; and derives a METHOD_CANDIDATE CLI config from the pinned
zero-engineering config. It never opens Dev/Confirm audio or runs the head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_REL = "scripts/audio_enhancement/configs/v3_zero_engineering_attempt02.json"
TEMPLATE_BYTES = 3_231
TEMPLATE_SHA256 = "6c67d0c86de72ddff104151e779b0e0942763af187d1c6c0cf7638b99d60598a"
EXPECTED_SCHEMA = "cross_tfg_fit40_native_transition_head_v3_training_runner_attempt02"
CHECKPOINT_SCHEMA = "cross_tfg_v3_parameter_checkpoint_v1"
TRAINING_SCHEMA = "cross_tfg_v3_candidate_training_record_v1"
SELECTION_SCHEMA = "cross_tfg_v3_candidate_selection_record_v1"
HEAD_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3"
FIXED_STEP = 120


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def read_regular(path: Path, label: str, limit: int = 4 * 1024 * 1024) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} has a symlink or junction component: {cursor}")
    resolved = lexical.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular file: {path}")
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while it was being read")
    return raw


def read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def exact_repo_file(rel: str, expected_bytes: int, expected_sha: str, label: str) -> tuple[Path, bytes]:
    path = ROOT / Path(rel)
    raw = read_regular(path, label)
    if len(raw) != expected_bytes or sha(raw) != expected_sha:
        raise ValueError(f"{label} no longer matches its frozen byte/SHA pin")
    return path, raw


def raw_file_pin(path: Path, raw: bytes, relative_to: Path) -> dict[str, Any]:
    try:
        rel = path.resolve(strict=True).relative_to(relative_to.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError(f"candidate provenance path is outside repository: {path}") from exc
    if "\\" in rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"unsafe repository-relative provenance path: {rel}")
    return {"path": rel, "bytes": len(raw), "sha256": sha(raw)}


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_inputs(completion_path: Path, checkpoint_path: Path,
                    training_path: Path, runtime_source_pins_path: Path,
                    runner_source_path: Path, runner_expected_sha256: str) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes,
                                                  dict[str, Any], bytes, dict[str, Any], bytes,
                                                  dict[str, Any]]:
    runtime_pins, runtime_pins_raw = read_json(runtime_source_pins_path, "fit runtime source pins")
    runner_path = runner_source_path.resolve(strict=True)
    try:
        runner_rel = runner_path.relative_to(ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError("actual fit runner source must be within the repository") from exc
    runner_raw = read_regular(runner_path, "actual fit runner source", limit=8 * 1024 * 1024)
    if sha(runner_raw) != runner_expected_sha256:
        raise ValueError("actual fit runner source differs from the supplied raw SHA pin")
    runner_pin = runtime_pins.get("runner")
    if not isinstance(runner_pin, dict) or runner_pin.get("bytes") != len(runner_raw) or runner_pin.get("sha256") != sha(runner_raw):
        raise ValueError("runtime source pins do not bind the supplied actual fit runner source bytes")
    if runtime_pins.get("schema") != "cross_tfg_fit40_native_transition_runtime_pins_v1":
        raise ValueError("fit runtime source pins schema is not the sealed native runner schema")
    if runner_pin.get("bytes") != len(runner_raw) or runner_pin.get("sha256") != runner_expected_sha256.lower():
        raise ValueError("supplied runner pin differs from the runtime source-pins runner binding")
    template_path, template_raw = exact_repo_file(TEMPLATE_REL, TEMPLATE_BYTES, TEMPLATE_SHA256,
                                                   "engineering V3 template")
    completion, completion_raw = read_json(completion_path, "fit completion record")
    runtime_completion_pin = completion.get("runtime_source_pins")
    if (not isinstance(runtime_completion_pin, dict)
            or runtime_completion_pin.get("bytes") != len(runtime_pins_raw)
            or runtime_completion_pin.get("sha256") != sha(runtime_pins_raw)):
        raise ValueError("completion record does not bind the supplied runtime source pins file")
    checkpoint, checkpoint_raw = read_json(checkpoint_path, "fit checkpoint")
    training, training_raw = read_json(training_path, "fit training record")

    if completion.get("schema") != EXPECTED_SCHEMA or completion.get("status") != "COMPLETE_FIT40_V3_120_STEPS":
        raise ValueError("fit completion record is not the sealed successful V3 attempt")
    if completion.get("completed_optimizer_steps") != FIXED_STEP:
        raise ValueError("fit completion does not certify exactly 120 optimizer steps")
    if (completion.get("fit_rows"), completion.get("fit_speakers"),
            completion.get("zero_gate_rows_passed")) != (40, 20, 40):
        raise ValueError("fit completion roster or same-device zero-head gate is incomplete")
    if (completion.get("selection_record_created") is not False
            or completion.get("development_audio_opened") is not False
            or completion.get("confirmation_audio_opened") is not False):
        raise ValueError("fit completion does not certify the pre-Dev unselected boundary")

    for field, path, raw in (("checkpoint", checkpoint_path, checkpoint_raw),
                             ("training_record", training_path, training_raw)):
        pin = completion.get(field)
        if not isinstance(pin, dict) or pin.get("bytes") != len(raw) or pin.get("sha256") != sha(raw):
            raise ValueError(f"completion {field} pin does not match the supplied raw file")
        if not isinstance(pin.get("path"), str) or not pin["path"]:
            raise ValueError(f"completion {field} path is absent")

    template = json.loads(template_raw.decode("utf-8"))
    if template.get("schema") != "cross_tfg_v3_enhancement_cli_config_v1" or template.get("profile_role") != "ENGINEERING_ONLY":
        raise ValueError("pinned template is not the V3 engineering-only config")
    if template.get("candidate_provenance") is not None:
        raise ValueError("engineering template must not already carry candidate provenance")
    template_parameter = template.get("parameter", {})
    if any(float(v) != 0.0 for row in template_parameter.get("values", []) for v in row):
        raise ValueError("engineering template is not the fixed zero-theta config")

    implementation = template.get("implementation")
    if not isinstance(implementation, dict) or implementation.get("path") != "scripts/audio_enhancement/shared_recipient_transition_head_v3.py":
        raise ValueError("template does not pin the frozen V3 head implementation")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("head_schema") != HEAD_SCHEMA:
        raise ValueError("checkpoint schema/head does not match the frozen V3 head")
    if checkpoint.get("implementation") != implementation:
        raise ValueError("checkpoint implementation pin differs from the engineering config")
    constructor = template.get("constructor")
    if not isinstance(constructor, dict) or checkpoint.get("constructor_sha256") != sha(canonical(constructor)):
        raise ValueError("checkpoint constructor digest differs from the frozen constructor")
    parameter = checkpoint.get("parameter")
    if not isinstance(parameter, dict) or parameter.get("shape") != [3, 3] or parameter.get("dtype") != "float32":
        raise ValueError("checkpoint parameter is not the exact 3×3 float32 V3 parameter")
    values = parameter.get("values")
    if not isinstance(values, list) or len(values) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in values):
        raise ValueError("checkpoint parameter matrix is malformed")
    for row in values:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("checkpoint parameter must contain only finite JSON numbers")
            try:
                packed = struct.pack("<f", float(value))
            except (OverflowError, struct.error) as exc:
                raise ValueError("checkpoint parameter is outside finite float32 range") from exc
            as_float32 = struct.unpack("<f", packed)[0]
            if not math.isfinite(as_float32) or as_float32 != float(value):
                raise ValueError("checkpoint parameter values must be exactly representable as float32")
            if as_float32 == 0.0 and math.copysign(1.0, as_float32) != math.copysign(1.0, float(value)):
                raise ValueError("checkpoint parameter zero sign changes during float32 conversion")
    if checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED":
        raise ValueError("checkpoint was already selected or its selection state is unclear")
    training_bindings = checkpoint.get("training_artifact_bindings")
    if not isinstance(training_bindings, dict) or training_bindings.get("completed_optimizer_steps") != FIXED_STEP:
        raise ValueError("checkpoint training bindings do not certify terminal step 120")

    if training.get("schema") != TRAINING_SCHEMA or training.get("status") != "COMPLETE":
        raise ValueError("training record is not complete")
    if (training.get("completed_optimizer_steps"), training.get("planned_optimizer_steps"),
            training.get("fit_row_count"), training.get("fit_speaker_count")) != (120, 120, 40, 20):
        raise ValueError("training record differs from the fixed 120-step/40-row/20-speaker fit")
    if training.get("checkpoint_sha256") != sha(checkpoint_raw):
        raise ValueError("training record does not bind the exact checkpoint bytes")
    if training.get("head_implementation_sha256") != implementation.get("sha256"):
        raise ValueError("training record does not bind the pinned V3 implementation")
    gate = training.get("zero_head_gate")
    if not isinstance(gate, dict) or gate.get("status") != "PASS" or gate.get("row_count") != 40 or gate.get("same_cuda_device_as_rms_baseline") is not True:
        raise ValueError("training record same-device zero-head gate is not 40/40 PASS")
    if (training.get("development_audio_opened") is not False
            or training.get("confirmation_audio_opened") is not False
            or training.get("selection_record_created") is not False
            or training.get("selection_status") != "NOT_SELECTED"):
        raise ValueError("training record does not certify no Dev/Confirm or selection during fit")
    if checkpoint.get("training_artifact_bindings", {}).get("runtime_source_pins_sha256") != sha(runtime_pins_raw):
        raise ValueError("checkpoint does not bind the supplied raw runtime source pins")
    if training.get("runtime_source_pins_sha256") != sha(runtime_pins_raw):
        raise ValueError("training record does not bind the supplied raw runtime source pins")
    if checkpoint.get("constructor_sha256") != training.get("constructor_sha256"):
        raise ValueError("training record constructor pin differs from checkpoint")
    if checkpoint.get("training_artifact_bindings", {}).get("fit_source_manifest_sha256") != training.get("fit_source_manifest_sha256"):
        raise ValueError("training checkpoint/record fit roster binding differs")
    if checkpoint.get("training_artifact_bindings", {}).get("native_geometry_manifest_sha256") != training.get("native_geometry_manifest_sha256"):
        raise ValueError("training checkpoint/record geometry binding differs")

    return (completion, completion_raw, checkpoint, checkpoint_raw, training, training_raw,
            template, template_raw, {"runner_path": runner_path, "runner_rel": runner_rel,
                                     "runner_raw": runner_raw, "runtime_pins_path": runtime_source_pins_path.resolve(strict=True),
                                     "runtime_pins_raw": runtime_pins_raw, "runtime_pins": runtime_pins,
                                     "template_path": template_path, "template_raw": template_raw})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-record", type=Path, required=True)
    parser.add_argument("--runtime-source-pins", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--runner-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve(strict=True)
    if project_root != ROOT.resolve(strict=True):
        raise ValueError("project root differs from the script's frozen repository root")
    completion, completion_raw, checkpoint, checkpoint_raw, training, training_raw, template, template_raw, pins = validate_inputs(
        args.completion, args.checkpoint, args.training_record, args.runtime_source_pins,
        args.runner_source, args.runner_source_sha256
    )

    requested_out = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    out = Path(os.path.abspath(os.fspath(requested_out)))
    try:
        out_rel = out.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("output directory must be lexically inside the repository") from exc
    if not out_rel.parts or any(part in {"", ".", ".."} for part in out_rel.parts):
        raise ValueError("output directory must be a safe non-root repository path")
    cursor = project_root
    for part in out_rel.parts[:-1]:
        cursor = cursor / part
        is_junction = getattr(cursor, "is_junction", lambda: False)()
        if cursor.is_symlink() or is_junction:
            raise ValueError(f"output path has a symlink or junction component: {cursor}")
        if cursor.exists():
            if not cursor.is_dir():
                raise ValueError(f"output parent component is not a directory: {cursor}")
        else:
            cursor.mkdir()
    out = cursor / out_rel.parts[-1]
    if out.exists() or out.is_symlink() or getattr(out, "is_junction", lambda: False)():
        raise FileExistsError(f"refusing existing output directory: {out}")
    out.mkdir(exist_ok=False)

    checkpoint_out = out / "candidate_provenance" / "checkpoint.json"
    training_out = out / "candidate_provenance" / "training_record.json"
    write_exclusive(checkpoint_out, checkpoint_raw)
    write_exclusive(training_out, training_raw)
    checkpoint_pin = raw_file_pin(checkpoint_out, checkpoint_raw, project_root)
    training_pin = raw_file_pin(training_out, training_raw, project_root)

    selection = {
        "schema": SELECTION_SCHEMA,
        "status": "FROZEN",
        "selection_rule": "fixed_terminal_optimizer_step_120_no_development_selection",
        "selected_optimizer_step": FIXED_STEP,
        "checkpoint_sha256": sha(checkpoint_raw),
        "training_record_sha256": sha(training_raw),
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "weight_selection_using_development": False,
        "role": "fixed_terminal_fit_step_materialization_only",
    }
    selection_raw = json_bytes(selection)
    selection_out = out / "candidate_provenance" / "selection_record.json"
    write_exclusive(selection_out, selection_raw)
    selection_pin = raw_file_pin(selection_out, selection_raw, project_root)

    config = json.loads(template_raw.decode("utf-8"))
    config["profile_role"] = "METHOD_CANDIDATE"
    config["parameter"] = checkpoint["parameter"]
    config["candidate_provenance"] = {
        "checkpoint": checkpoint_pin,
        "training_record": training_pin,
        "selection_record": selection_pin,
    }
    config_raw = json_bytes(config)
    config_out = out / "v3_method_candidate_config.json"
    write_exclusive(config_out, config_raw)

    source_info = {
        "materializer": {"path": Path(__file__).resolve().relative_to(project_root).as_posix(),
                         "bytes": Path(__file__).stat().st_size, "sha256": sha(Path(__file__).read_bytes())},
        "fit_runner": {"runtime_record_path": pins["runtime_pins"].get("runner", {}).get("path"),
                       "repository_source_path": pins["runner_rel"], "bytes": len(pins["runner_raw"]),
                       "sha256": sha(pins["runner_raw"])},
        "fit_runtime_source_pins": {"path": str(pins["runtime_pins_path"]),
                                    "bytes": len(pins["runtime_pins_raw"]),
                                    "sha256": sha(pins["runtime_pins_raw"])},
        "engineering_template": {"path": TEMPLATE_REL, "bytes": len(template_raw), "sha256": sha(template_raw)},
        "completion": {"path": str(args.completion.resolve(strict=True)), "bytes": len(completion_raw), "sha256": sha(completion_raw)},
        "original_checkpoint": {"path": completion["checkpoint"]["path"], "bytes": len(checkpoint_raw), "sha256": sha(checkpoint_raw)},
        "original_training_record": {"path": completion["training_record"]["path"], "bytes": len(training_raw), "sha256": sha(training_raw)},
    }
    receipt = {
        "schema": "cross_tfg_v3_step120_method_candidate_materialization_v1",
        "status": "MATERIALIZED_FIXED_STEP120_METHOD_CANDIDATE",
        "profile_role": config["profile_role"],
        "optimizer_steps": FIXED_STEP,
        "fit_row_count": 40,
        "fit_speaker_count": 20,
        "zero_head_gate_rows_passed": 40,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "method_selected_using_development": False,
        "model_execution_performed": False,
        "candidate_config": {"path": config_out.relative_to(project_root).as_posix(), "bytes": len(config_raw), "sha256": sha(config_raw)},
        "candidate_provenance": {
            "checkpoint": checkpoint_pin,
            "training_record": training_pin,
            "selection_record": selection_pin,
        },
        "source_bindings": source_info,
    }
    receipt_raw = json_bytes(receipt)
    write_exclusive(out / "materialization_receipt.json", receipt_raw)
    print(json.dumps({"status": receipt["status"], "output_dir": str(out),
                      "candidate_config": receipt["candidate_config"],
                      "selection_record_sha256": selection_pin["sha256"],
                      "development_audio_opened": False, "confirmation_audio_opened": False},
                     sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
