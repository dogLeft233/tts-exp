"""Build a self-hashed score binding from an actual completed D0 Fit40 run.

This metadata-only utility imports only the standard library. It performs no
training, CUDA/model initialization, rendering, scoring, API call, or selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

BINDING_SCHEMA = "cross_tfg_fit40_post_projection_score_input_binding_v1"
COMPLETION_SCHEMA = "cross_tfg_fit40_native_transition_head_v3_d0_direction_gatedc_runner_attempt02"
COMPLETION_STATUS = "COMPLETE_FIT40_DIRECT_D0_DIRECTION_GATE_DC_120_STEPS"
SOURCE_RECEIPT_SCHEMA = "fit40_d0_gatedc_source_stage_receipt_attempt02_v1"
SOURCE_RECEIPT_STATUS = "PASS_SOURCE_CLOSURE_EXTRACTED_NO_MODEL_OR_GPU"
SOURCE_MEMBER_COUNT = 23
HEAD_RELATIVE_PATH = "scripts/audio_enhancement/shared_recipient_transition_head_v3_d0_gatedc_attempt01.py"
RUNNER_RELATIVE_PATH = "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_d0_direction_gatedc_attempt02.py"
HELPER_SHA256 = "27a9eae5b52faa218c4ff383e742d3833694e60812a4af79606ad8207545df56"
SCORER_SHA256 = "1b6bd4dd292dca2daa9125acb67f6782e477075e6b87a79c9512897747578d54"
SOURCE_RECEIPT_SHA256 = "d4a2a5d8e6b8c4edd9c57e13f53bee6d727d77b518bea1311ad8be87896c5161"
SOURCE_RECEIPT_BYTES = 9594


class BindingError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise BindingError(f"{label} must be an absolute nonsymlink path: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise BindingError(f"{label} is not the exact regular file: {path}")
    return resolved


def pin_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = require_regular(path, label)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}, raw


def verify_pin(value: Any, label: str) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise BindingError(f"{label} must contain exactly path/bytes/sha256")
    path = Path(value["path"])
    if not path.is_absolute() or type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise BindingError(f"{label} path/size is malformed")
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise BindingError(f"{label} digest is malformed")
    pin, raw = pin_file(path, label)
    if pin != value:
        raise BindingError(f"{label} raw bytes differ from the producer pin")
    return pin, raw


def read_pinned_json(value: Any, label: str) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    pin, raw = verify_pin(value, label)
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise BindingError(f"{label} must decode to an object")
    return pin, raw, parsed


def verify_self_hash(value: dict[str, Any], label: str) -> None:
    recorded = value.get("self_sha256")
    payload = {key: item for key, item in value.items() if key != "self_sha256"}
    if not isinstance(recorded, str) or sha(canonical(payload)) != recorded:
        raise BindingError(f"{label} self-hash mismatch")


def verify_source_stage(receipt_pin: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    if receipt_pin["bytes"] != SOURCE_RECEIPT_BYTES or receipt_pin["sha256"] != SOURCE_RECEIPT_SHA256:
        raise BindingError("source-stage receipt differs from the already verified source closure")
    _, _, receipt = read_pinned_json(receipt_pin, "source-stage receipt")
    verify_self_hash(receipt, "source-stage receipt")
    if receipt.get("schema") != SOURCE_RECEIPT_SCHEMA or receipt.get("status") != SOURCE_RECEIPT_STATUS:
        raise BindingError("source-stage receipt schema/status differs")
    scope = receipt.get("scope")
    if not isinstance(scope, dict) or any(scope.get(name) is not False for name in ("confirmation_audio_opened", "cuda_initialized", "development_audio_opened", "model_imported", "optimizer_updated", "render_or_score")):
        raise BindingError("source-stage receipt does not preserve no-model/no-GPU scope")
    members = receipt.get("members")
    if not isinstance(members, list) or len(members) != SOURCE_MEMBER_COUNT:
        raise BindingError("source-stage receipt must contain exactly 23 members")
    root = Path(str(receipt.get("destination", "")))
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise BindingError("source-stage destination is not a canonical directory")
    seen: set[str] = set()
    for i, member in enumerate(members):
        if not isinstance(member, dict):
            raise BindingError(f"source member {i} is malformed")
        rel_text = member.get("relative_path")
        if not isinstance(rel_text, str) or not rel_text or "\\" in rel_text:
            raise BindingError(f"source member {i} relative path is invalid")
        rel = PurePosixPath(rel_text)
        if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts) or rel_text in seen:
            raise BindingError(f"source member {i} has an unsafe or duplicate path")
        seen.add(rel_text)
        expected = root.joinpath(*rel.parts)
        if member.get("path") != str(expected):
            raise BindingError(f"source member {rel_text} path differs from the exact receipt destination")
        verify_pin({"path": str(expected), "bytes": member.get("bytes"), "sha256": member.get("sha256")}, f"source member {rel_text}")
    if HEAD_RELATIVE_PATH not in seen or RUNNER_RELATIVE_PATH not in seen:
        raise BindingError("source-stage receipt omits the exact D0 head or runner")
    disk_members = {item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file() and "__pycache__" not in item.parts}
    if disk_members != seen:
        raise BindingError(f"staged source tree differs from receipt closure (extra={sorted(disk_members-seen)}, missing={sorted(seen-disk_members)})")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise BindingError("staged source tree contains a symlink")
    return receipt, root


def artifact_from_runtime(runtime: dict[str, Any], name: str) -> dict[str, Any]:
    value = runtime.get(name)
    if not isinstance(value, dict):
        raise BindingError(f"runtime-source pins lack {name}")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    completion_path = Path(args.completion)
    completion_pin, completion_raw = pin_file(completion_path, "completion")
    try:
        completion = json.loads(completion_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingError("completion is not UTF-8 JSON") from exc
    if not isinstance(completion, dict) or completion.get("schema") != COMPLETION_SCHEMA or completion.get("status") != COMPLETION_STATUS:
        raise BindingError("actual completion schema/status is not complete Fit40 D0/GateDC")
    if (completion.get("fit_rows"), completion.get("fit_speakers"), completion.get("completed_optimizer_steps"), completion.get("zero_gate_rows_passed")) != (40, 20, 120, 40):
        raise BindingError("actual completion does not bind 40 rows/20 speakers/120 updates/40 zero-gate passes")
    if any(completion.get(key) is not False for key in ("selection_record_created", "development_audio_opened", "confirmation_audio_opened")):
        raise BindingError("completion does not preserve unselected Fit-only scope")

    artifact_names = ("checkpoint", "training_record", "runtime_source_pins", "baseline_arrays", "zero_gate_arrays", "strict_backward_qualification", "training_score_arrays", "training_steps", "fit40_gate_dc_control")
    artifacts: dict[str, dict[str, Any]] = {"completion": completion_pin}
    for name in artifact_names:
        pin, _ = verify_pin(completion.get(name), f"completion.{name}")
        artifacts[name] = pin
    _, _, runtime = read_pinned_json(artifacts["runtime_source_pins"], "runtime-source pins")
    for name in ("fit_source_manifest", "native_geometry_manifest", "head_config", "parent_plan"):
        pin, _ = verify_pin(artifact_from_runtime(runtime, name), f"runtime_source_pins.{name}")
        artifacts[name] = pin

    _, _, training = read_pinned_json(artifacts["training_record"], "training record")
    if training.get("schema") != "cross_tfg_v3_d0_direction_gatedc_training_record_v1" or training.get("status") != "COMPLETE" or training.get("completed_optimizer_steps") != 120 or training.get("planned_optimizer_steps") != 120 or training.get("fit_row_count") != 40 or training.get("fit_speaker_count") != 20 or training.get("selection_status") != "NOT_SELECTED" or any(training.get(key) is not False for key in ("selection_record_created", "development_audio_opened", "confirmation_audio_opened")):
        raise BindingError("training record does not match the complete unselected Fit40/Adam120 scope")
    if completion.get("training_record_raw_sha256") != artifacts["training_record"]["sha256"]:
        raise BindingError("completion training-record raw digest differs")

    checkpoint_pin, _, checkpoint = read_pinned_json(artifacts["checkpoint"], "checkpoint")
    parameter = checkpoint.get("parameter")
    if not isinstance(parameter, dict) or parameter.get("shape") != [3, 3] or parameter.get("dtype") != "float32":
        raise BindingError("checkpoint parameter is not the exact [3,3] float32 head value")
    expected_theta_sha = sha(canonical(parameter))
    runtime_modules = runtime.get("training_modules")
    if not isinstance(runtime_modules, dict) or not isinstance(runtime_modules.get("v3_head"), dict) or not isinstance(runtime.get("runner"), dict):
        raise BindingError("runtime-source pins lack exact head/runner provenance")
    head_source = runtime_modules["v3_head"]
    checkpoint_implementation = checkpoint.get("implementation")
    if not isinstance(checkpoint_implementation, dict) or checkpoint_implementation.get("path") != HEAD_RELATIVE_PATH or checkpoint_implementation.get("bytes") != head_source.get("bytes") or checkpoint_implementation.get("sha256") != head_source.get("sha256"):
        raise BindingError("checkpoint relative head path/bytes/SHA differ from runtime-source head pin")

    receipt_pin, _ = pin_file(Path(args.source_stage_receipt), "source-stage receipt")
    receipt, code_root = verify_source_stage(receipt_pin)
    if Path(args.code_root).resolve(strict=True) != code_root:
        raise BindingError("explicit code_root differs from the actual source-stage destination")
    fit_root = Path(args.fit_source_root)
    if not fit_root.is_absolute() or fit_root.is_symlink() or not fit_root.is_dir() or fit_root.resolve(strict=True) != fit_root:
        raise BindingError("fit_source_root must be an actual canonical directory")

    helper_pin, _ = pin_file(Path(args.post_projection_controls), "post-projection controls helper")
    score_pin, _ = pin_file(Path(args.score_adapter), "score adapter")
    if helper_pin["sha256"] != HELPER_SHA256 or score_pin["sha256"] != SCORER_SHA256:
        raise BindingError("support-file bytes differ from the source-reviewed helper/scorer pins")

    lineage = dict(artifacts)
    lineage["head_implementation"] = head_source
    lineage["runner"] = runtime["runner"]
    parent_plan = runtime.get("parent_plan")
    if not isinstance(parent_plan, dict) or not isinstance(parent_plan.get("payload_sha256"), str) or len(parent_plan["payload_sha256"]) != 64:
        raise BindingError("runtime-source pins lack the actual canonical parent-plan payload digest")
    binding = {
        "schema": BINDING_SCHEMA,
        "status": "ACTUAL_COMPLETION_BOUND_READY_FOR_FIT_ONLY_SCORE_DIAGNOSTIC",
        "scope": {"development_audio_opened": False, "confirmation_audio_opened": False, "selection_record_created": False, "api_called": False},
        "source_stage_receipt": receipt_pin,
        "code_root": str(code_root),
        "fit_source_root": str(fit_root),
        "artifacts": artifacts,
        "source_lineage_pins": lineage,
        "source_lineage_pins_sha256": sha(canonical(lineage)),
        "support_files": {"post_projection_controls": helper_pin, "score_adapter": score_pin},
        "parent_plan_payload_sha256": parent_plan["payload_sha256"],
        "expected_checkpoint_parameter_sha256": expected_theta_sha,
    }
    binding["self_sha256"] = sha(canonical(binding))
    output = Path(args.output)
    if not output.is_absolute() or output.is_symlink() or output.exists() or not output.parent.is_dir() or output.parent.resolve(strict=True) != output.parent:
        raise BindingError("binding output must be absent with an existing canonical parent")
    raw = json.dumps(binding, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    return {"schema": BINDING_SCHEMA, "status": binding["status"], "path": str(output.resolve(strict=True)), "bytes": len(raw), "sha256": sha(raw), "self_sha256": binding["self_sha256"], "expected_checkpoint_parameter_sha256": expected_theta_sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bind actual completed D0 Fit40 artifacts for a later score-only diagnostic; metadata only.")
    parser.add_argument("--completion", required=True)
    parser.add_argument("--source-stage-receipt", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--fit-source-root", required=True)
    parser.add_argument("--post-projection-controls", required=True)
    parser.add_argument("--score-adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(args), sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"schema": BINDING_SCHEMA, "status": "FAILED_BINDING_NOT_WRITTEN", "exception_type": type(exc).__name__, "error": str(exc)}, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
