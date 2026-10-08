#!/usr/bin/env python3
"""Read-only status snapshot for the detached Fit40 D0/GateDC attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import stat
import time
from typing import Any


PROJECT_ROOT = pathlib.Path("/root/tts-research-system-20261001")
CONTROL_ROOT = PROJECT_ROOT / "cross_tfg_audio_head_20261003/job_control/fit40_d0_gatedc_training_attempt02"
TASK_ROOT = pathlib.Path("/dev/shm/cross_tfg_audio_head_20261003/fit40_d0_gatedc_training_attempt02")
OUTPUT_ROOT = TASK_ROOT / "run"


class StatusError(RuntimeError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def file_pin(path: pathlib.Path, maximum: int = 2 * 1024**3) -> dict[str, Any]:
    if path.is_symlink():
        raise StatusError(f"symlink refused: {path}")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise StatusError(f"not a bounded regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise StatusError(f"file changed during read: {path}")
    return {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()}


def read_json(path: pathlib.Path, maximum: int = 8 * 1024**2) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    if not path.exists() and not path.is_symlink():
        return None, None, "ABSENT"
    pin = file_pin(path, maximum)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, pin, f"INVALID_JSON:{type(exc).__name__}"
    if not isinstance(value, dict):
        return None, pin, "NOT_OBJECT"
    return value, pin, None


def verify_self(obj: dict[str, Any], field: str = "self_sha256") -> bool:
    declared = obj.get(field)
    payload = dict(obj)
    payload.pop(field, None)
    return isinstance(declared, str) and declared == sha256(canonical(payload))


def pinned_artifact(pin: Any, *, expected_root: pathlib.Path, maximum: int = 2 * 1024**3) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
        raise StatusError("completion artifact pin has an unexpected schema")
    path = pathlib.Path(str(pin["path"]))
    try:
        path.resolve(strict=True).relative_to(expected_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StatusError("completion artifact path escapes the runner output root") from exc
    actual = file_pin(path, maximum)
    if actual["bytes"] != int(pin["bytes"]) or actual["sha256"] != str(pin["sha256"]).lower():
        raise StatusError(f"completion artifact differs from its recorded pin: {path.name}")
    raw = path.read_bytes()
    return raw, actual


def read_json_artifact(pin: Any, *, expected_root: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, actual = pinned_artifact(pin, expected_root=expected_root, maximum=16 * 1024**2)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise StatusError(f"completion JSON artifact is not an object: {actual['path']}")
    return value, actual


def proc_state(pid: int | None, expected_ticks: int | None) -> str:
    if not isinstance(pid, int) or pid <= 0:
        return "PID_UNAVAILABLE"
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = raw[raw.rfind(")") + 2:].split()
        ticks = int(tail[19])
        return "RUNNING_IDENTITY_MATCH" if expected_ticks is None or ticks == expected_ticks else "PID_REUSED_OR_START_TICK_MISMATCH"
    except FileNotFoundError:
        return "NOT_RUNNING"
    except (OSError, ValueError, IndexError):
        return "PROC_STATE_UNAVAILABLE"


def expected_modes_row_ids(row_ids: list[str]) -> list[str]:
    """Expected 80 ordered IDs: dynamic and gate_dc for each frozen fit row."""
    return [row_id for row_id in row_ids for _mode in ("dynamic", "gate_dc")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=pathlib.Path, default=CONTROL_ROOT)
    parser.add_argument("--task-root", type=pathlib.Path, default=TASK_ROOT)
    args = parser.parse_args()
    control = args.control_root
    task = args.task_root
    if control.is_symlink() or task.is_symlink() or not control.is_dir() or not task.is_dir():
        raise StatusError("job-control/task root absent or symlink")
    launch, launch_pin, launch_error = read_json(control / "output/launch_receipt.json")
    if launch_error or not isinstance(launch, dict) or not verify_self(launch):
        raise StatusError(f"launch receipt is not valid: {launch_error or 'self-hash mismatch'}")
    child, child_pin, child_error = read_json(control / "output/child_process.json")
    exit_record, exit_pin, exit_error = read_json(control / "output/process_exit.json")
    supervisor_failure, supervisor_failure_pin, supervisor_failure_error = read_json(control / "output/launch_failure.json")
    if isinstance(child, dict) and not verify_self(child):
        child_error = "SELF_HASH_MISMATCH"
    if isinstance(exit_record, dict) and not verify_self(exit_record):
        exit_error = "SELF_HASH_MISMATCH"
    if isinstance(supervisor_failure, dict) and not verify_self(supervisor_failure):
        supervisor_failure_error = "SELF_HASH_MISMATCH"
    supervisor = launch.get("supervisor", {})
    supervisor_state = proc_state(supervisor.get("pid"), supervisor.get("start_ticks"))
    runner_state = proc_state(child.get("pid"), child.get("start_ticks")) if isinstance(child, dict) else "CHILD_RECEIPT_" + str(child_error)
    lineage_errors = []
    if isinstance(child, dict):
        if child.get("manifest_raw_sha256") != launch.get("manifest", {}).get("sha256"):
            lineage_errors.append("child/launch manifest SHA mismatch")
        if child.get("argv") != launch.get("runner", {}).get("argv"):
            lineage_errors.append("child argv differs from launch receipt")
    if isinstance(exit_record, dict) and isinstance(child, dict):
        if exit_record.get("pid") != child.get("pid") or exit_record.get("start_ticks") != child.get("start_ticks"):
            lineage_errors.append("process-exit PID/start ticks differ from child receipt")
        if exit_record.get("manifest_raw_sha256") != child.get("manifest_raw_sha256"):
            lineage_errors.append("process-exit manifest SHA differs from child receipt")
    if isinstance(exit_record, dict) and child_pin is not None and isinstance(child, dict):
        # The launcher records the full write_json_exclusive() pin, including
        # self_sha256; file_pin() above intentionally reports only raw bytes/SHA.
        # Reconstruct the writer's exact four-field identity from the validated
        # child contents so the raw lineage check remains strict.
        expected_child_receipt_pin = {**child_pin, "self_sha256": child.get("self_sha256")}
        if exit_record.get("child_process_receipt") != expected_child_receipt_pin:
            lineage_errors.append("process-exit child-receipt pin differs")
    status_obj, status_pin, status_error = read_json(task / "run/status.json", 1024**2)
    completion, completion_pin, completion_error = read_json(task / "run/completion.json")
    failure, failure_pin, failure_error = read_json(task / "run/failure.json")
    terminal_pass = False
    pass_gates = {}
    if isinstance(completion, dict) and completion_error is None:
        dc = completion.get("fit40_gate_dc_control")
        pass_gates = {
            "completion_status": completion.get("status") == "COMPLETE_FIT40_DIRECT_D0_DIRECTION_GATE_DC_120_STEPS",
            "completed_optimizer_steps": completion.get("completed_optimizer_steps") == 120,
            "fit_rows": completion.get("fit_rows") == 40,
            "zero_gate_rows_passed": completion.get("zero_gate_rows_passed") == 40,
            "selection_record_created": completion.get("selection_record_created") is False,
            "development_audio_opened": completion.get("development_audio_opened") is False,
            "confirmation_audio_opened": completion.get("confirmation_audio_opened") is False,
            "no_lineage_errors": not lineage_errors,
            "exit_manifest_matches_launch": isinstance(exit_record, dict) and exit_record.get("manifest_raw_sha256") == launch.get("manifest", {}).get("sha256"),
            "exit_return_code_zero": isinstance(exit_record, dict) and exit_record.get("return_code") == 0,
        }
        try:
            strict_record, strict_pin = read_json_artifact(completion.get("strict_backward_qualification"), expected_root=task / "run")
            full_backward = strict_record.get("full_objective_backward")
            pass_gates.update({
                "strict_backward_status": strict_record.get("status") == "PASS_FULL_OBJECTIVE_BACKWARD_BEFORE_OPTIMIZER",
                "strict_backward_shape": isinstance(full_backward, dict) and full_backward.get("gradient_shape") == [3, 3],
                "strict_backward_nonzero_finite_signal": isinstance(full_backward, dict) and isinstance(full_backward.get("gradient_nonzero_scalar_count"), int) and full_backward.get("gradient_nonzero_scalar_count", 0) > 0 and full_backward.get("gradient_positive_signal") is True,
                "strict_backward_no_update": isinstance(full_backward, dict) and full_backward.get("optimizer_updates") == 0 and full_backward.get("v3_head_adam_created") is False,
                "strict_backward_native_ops": isinstance(full_backward, dict) and full_backward.get("all_three_syncnet_pool_backward_layers_exercised") is True and full_backward.get("native_crop_bilinear_backward_exercised") is True and full_backward.get("v3_reflection_pad2d_backward_exercised") is True,
            })
        except Exception as exc:
            strict_pin = None
            pass_gates["strict_backward_artifact_valid"] = False
            lineage_errors.append(f"strict backward artifact: {type(exc).__name__}")
        try:
            training_record, training_pin = read_json_artifact(completion.get("training_record"), expected_root=task / "run")
            pass_gates.update({
                "training_record_complete": training_record.get("schema") == "cross_tfg_v3_d0_direction_gatedc_training_record_v1" and training_record.get("status") == "COMPLETE",
                "training_record_steps": training_record.get("completed_optimizer_steps") == 120 and training_record.get("planned_optimizer_steps") == 120,
                "training_record_fit_rows": training_record.get("fit_row_count") == 40 and training_record.get("fit_speaker_count") == 20,
                "training_record_zero_gate": training_record.get("zero_head_gate", {}).get("status") == "PASS" and training_record.get("zero_head_gate", {}).get("row_count") == 40,
                "training_record_gate_dc": training_record.get("gate_dc_control", {}).get("status") == "COMPLETE_FIT40_SCORE_ONLY_SAME_THETA" and training_record.get("gate_dc_control", {}).get("branch_rows") == 80 and training_record.get("gate_dc_control", {}).get("optimizer_updates") == 0,
                "training_record_scope_closed": training_record.get("development_audio_opened") is False and training_record.get("confirmation_audio_opened") is False and training_record.get("selection_record_created") is False and training_record.get("selection_status") == "NOT_SELECTED",
            })
        except Exception as exc:
            training_pin = None
            pass_gates["training_record_artifact_valid"] = False
            lineage_errors.append(f"training record artifact: {type(exc).__name__}")
        try:
            gate_pin = completion.get("fit40_gate_dc_control")
            gate_raw, gate_actual_pin = pinned_artifact(gate_pin, expected_root=task / "run", maximum=32 * 1024**2)
            gate_rows = [json.loads(line) for line in gate_raw.splitlines() if line]
            expected_row_ids = [str(row["row_id"]) for row in launch.get("inputs", {}).get("fit_wavs", [])]
            observed_ids = [str(row.get("row_id")) for row in gate_rows]
            expected_modes = [mode for _ in expected_row_ids for mode in ("dynamic", "gate_dc")]
            pass_gates.update({
                "gate_dc_artifact_rows": len(gate_rows) == 80 and observed_ids == expected_modes_row_ids(expected_row_ids),
                "gate_dc_artifact_modes": [row.get("gate_mode") for row in gate_rows] == expected_modes,
                "gate_dc_artifact_scope": all(row.get("gate_dc_control_is_training_objective") is False for row in gate_rows),
                "gate_dc_same_theta": len({row.get("same_frozen_parameter_sha256") for row in gate_rows}) == 1 and bool(gate_rows[0].get("same_frozen_parameter_sha256")) if gate_rows else False,
                "gate_dc_row_ids": len(expected_row_ids) == 40 and all(row_id in set(expected_row_ids) for row_id in observed_ids),
            })
        except Exception as exc:
            gate_actual_pin = None
            pass_gates["gate_dc_artifact_valid"] = False
            lineage_errors.append(f"GateDC artifact: {type(exc).__name__}")
        terminal_pass = all(pass_gates.values()) and isinstance(exit_record, dict) and exit_record.get("return_code") == 0 and runner_state == "NOT_RUNNING"
    if terminal_pass:
        snapshot_status = "PASS_FIT40_D0_GATEDC_COMPLETE"
    elif isinstance(supervisor_failure, dict) or isinstance(failure, dict) or (isinstance(exit_record, dict) and exit_record.get("return_code") != 0):
        snapshot_status = "FAILED_OR_BLOCKED_ROOT_REVIEW_REQUIRED"
    elif runner_state == "RUNNING_IDENTITY_MATCH" or supervisor_state == "RUNNING_IDENTITY_MATCH":
        snapshot_status = "RUNNING_OBSERVATION_ONLY"
    elif isinstance(exit_record, dict):
        snapshot_status = "STOPPED_ROOT_REVIEW_REQUIRED"
    else:
        snapshot_status = "INCOMPLETE_ROOT_REVIEW_REQUIRED"
    snapshot = {
        "schema": "fit40_d0_gatedc_readonly_status_snapshot_v1",
        "status": snapshot_status,
        "observed_unix_seconds": time.time(),
        "launch_receipt": launch_pin,
        "supervisor_process_state": supervisor_state,
        "child_process_receipt": child_pin,
        "child_process_state": runner_state,
        "process_exit": exit_pin,
        "process_exit_error": exit_error,
        "lineage_errors": lineage_errors,
        "runner_status": status_obj,
        "runner_status_pin": status_pin,
        "runner_status_read_error": status_error,
        "completion": completion,
        "completion_pin": completion_pin,
        "completion_read_error": completion_error,
        "completion_pass_gates": pass_gates,
        "strict_backward_qualification_pin": strict_pin if isinstance(completion, dict) and completion_error is None else None,
        "training_record_pin": training_pin if isinstance(completion, dict) and completion_error is None else None,
        "gate_dc_artifact_pin": gate_actual_pin if isinstance(completion, dict) and completion_error is None else None,
        "failure": failure,
        "failure_pin": failure_pin,
        "failure_read_error": failure_error,
        "supervisor_failure": supervisor_failure,
        "supervisor_failure_pin": supervisor_failure_pin,
        "supervisor_failure_read_error": supervisor_failure_error,
        "automatic_replay": False,
        "cleanup_performed": False,
    }
    print(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"schema": "fit40_d0_gatedc_readonly_status_snapshot_v1", "status": "STATUS_READ_FAILED", "type": type(exc).__name__, "message": str(exc)[:1200]}, sort_keys=True, separators=(",", ":")), flush=True)
        raise SystemExit(2)
