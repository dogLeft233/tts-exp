#!/usr/bin/env python3
"""Read-only, bounded status for the actual detached candidate180 child and its outputs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path("/root/tts-research-system-20261001")
JOB = ROOT / "cross_tfg_audio_head_20261003/job_control/dev360_candidate180_detached_attempt12"
PREFLIGHT = ROOT / "cross_tfg_audio_head_20261003/job_control/candidate_v3_cli03_metadata_preflight_attempt04/output/candidate_runtime_metadata_preflight_attempt03.json"
PLAN = ROOT / "runs/cross_tfg_audio_head_20261003/candidate180_cli03_runtime_repair_attempt03/candidate_plan_attempt14.json"
ROWS = PLAN.parent / "candidate_planned_rows_attempt14.jsonl"
PLAN_MANIFEST = PLAN.parent / "candidate_plan_manifest_attempt15.json"
BINDING = ROOT / "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json"
CANDIDATE_REPAIR_BINDING = ROOT / "runs/cross_tfg_audio_head_20261003/candidate180_cli03_runtime_repair_attempt03/candidate_runtime_repair_binding_attempt10.json"
CONFIG = ROOT / "runs/cross_tfg_audio_head_20261003/candidate180_cli03_runtime_repair_attempt03/v3_method_candidate_config_attempt03.json"
CANDIDATE_NAMESPACE = Path("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cli03_runtime_repair_attempt03")
CANDIDATE_ROOT = CANDIDATE_NAMESPACE / "candidate180"
CONTROL_ROOT = Path("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360")
CONTROLS_EXECUTION = CONTROL_ROOT.parent / "controls360.execution.json"
CANDIDATE_RUNNER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt15.py"
CANDIDATE_DISPATCHER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt07.py"
WAV2LIP_PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
CANDIDATE_RUN_ID = "cross_tfg_audio_head_20261003_v3_fit40_dev180_cli03_runtime_repair_attempt03"
CONTROL_RUN_ID = "cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10"
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
PLAN_BYTES = 934267
PLAN_SHA256 = "1a6c8ae99c4b8a1141c6f65748d426ba88034c14badf5ecec497ffc11d62bc60"
ROWS_BYTES = 606042
ROWS_SHA256 = "d781268d02c7495fd35c3bf314848863d57673eeef044a0c5edb19f9e8797f48"
BINDING_BYTES = 568289
BINDING_SHA256 = "dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5"
PLAN_MANIFEST_BYTES = 3172
PLAN_MANIFEST_SHA256 = "cd0aa3e7bca14a7676dab7f82d621b93e1a173e586f04efcdbaf3794b022808c"
REPAIR_BINDING_BYTES = 8057
REPAIR_BINDING_SHA256 = "4d50fcb6e6fd7a6fcd6a4a59515f7e92c9cbb43250e22ac97994d5558bb84ba8"
REPAIR_BINDING_SELF_SHA256 = "7106e528983dd927a031a02934da9dfc9aafe7e7c8a07617f2c1ca3fd47faeec"
CONFIG_BYTES = 5240
CONFIG_SHA256 = "fd2ce3509c457e40e618818828f6040ab65641315d40efe8f48e5dc3c5c3d1df"
CPU_RECEIPT_SHA256 = "8c44b1c80a08e48465f25672ef6b990002653ff7d250ede7369be13dab3a3b09"
CPU_RECEIPT_BYTES = 45226
MAX_RECEIPT = 4 * 1024 * 1024
MAX_LEDGER = 64 * 1024 * 1024
MAX_ARTIFACT = 32 * 1024 * 1024
MAX_PREFLIGHT = 2 * 1024 * 1024
MAX_LOG_TAIL = 64 * 1024
MAX_LINE = 4 * 1024 * 1024


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: dict[str, Any], key: str) -> str:
    return sha_bytes(canonical({name: item for name, item in value.items() if name != key}))


def read_regular(path: Path, label: str, limit: int, expected_bytes: int | None = None,
                 expected_sha256: str | None = None) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular nonsymlink file: {path}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError(f"{label} changed while being read: {path}")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError(f"{label} byte count differs from its launch pin")
    if expected_sha256 is not None and sha_bytes(raw) != expected_sha256:
        raise ValueError(f"{label} raw SHA differs from its launch pin")
    return raw


def read_json(path: Path, label: str, limit: int, expected_bytes: int | None = None,
              expected_sha256: str | None = None) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label, limit, expected_bytes, expected_sha256)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def read_transition_json(path: Path, label: str, limit: int) -> tuple[dict[str, Any] | None, bytes | None, dict[str, Any]]:
    """Report a bounded partial JSON snapshot when a live producer is replacing a status file."""
    try:
        raw = read_regular(path, label, limit)
    except FileNotFoundError:
        return None, None, {"status": "NOT_YET_CREATED", "path": str(path)}
    try:
        value = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, raw, {"status": "PARTIAL_JSON_DURING_WRITE", "path": str(path),
                           "bytes": len(raw), "raw_sha256": sha_bytes(raw)}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw, {"status": "PARSED", "path": str(path),
                        "bytes": len(raw), "raw_sha256": sha_bytes(raw)}


def flag_value(argv: Any, flag: str) -> str:
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError("candidate execution command is not a string argv list")
    indices = [index for index, item in enumerate(argv) if item == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError(f"candidate inner command lacks one exact {flag} value")
    return argv[indices[0] + 1]


def expected_strict_startup_code() -> str:
    raw = read_regular(CANDIDATE_RUNNER, "pinned candidate15 source", 2 * 1024 * 1024,
                       49071, "438db8d3cd255b5996d3d0cfe12c5653a2f94d3b647931a8e3ad8dc059070486")
    tree = ast.parse(raw.decode("utf-8"), filename=str(CANDIDATE_RUNNER))
    fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "strict_startup_gate_code"), None)
    body = list(fn.body) if fn is not None else []
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if fn is None or len(body) != 1 or not isinstance(body[0], ast.Return):
        raise ValueError("pinned candidate15 strict-startup source shape changed")
    value = body[0].value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        raise ValueError("candidate15 strict-startup command is not one pinned literal")
    return value.value


def proc_identity(pid: int, expected_ticks: int) -> dict[str, Any]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return {"state": "EXITED_OR_NOT_FOUND", "observed_start_ticks": None, "matches_launch": False}
    close = raw.rfind(")")
    if close < 0:
        return {"state": "STAT_MALFORMED", "observed_start_ticks": None, "matches_launch": False}
    fields = raw[close + 2:].split()
    if len(fields) <= 19:
        return {"state": "STAT_MALFORMED", "observed_start_ticks": None, "matches_launch": False}
    state = fields[0]
    ticks = int(fields[19])
    if ticks != expected_ticks:
        label = "PID_REUSED_OR_START_TICK_MISMATCH"
    elif state in ("Z", "X", "x"):
        label = "EXITED_ZOMBIE"
    else:
        label = "RUNNING"
    return {"state": label, "linux_process_state": state,
            "observed_start_ticks": ticks, "matches_launch": ticks == expected_ticks}


def bounded_log_tail(path: Path, expected: Path, label: str) -> dict[str, Any]:
    if path != expected or path.parent != JOB or path.is_symlink():
        raise ValueError(f"{label} path differs from its exact launch receipt location")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"status": "NOT_YET_CREATED", "path": str(path)}
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file")
    with path.open("rb") as stream:
        offset = max(0, info.st_size - MAX_LOG_TAIL)
        stream.seek(offset)
        tail = stream.read(MAX_LOG_TAIL)
    if len(tail) != min(info.st_size, MAX_LOG_TAIL):
        raise ValueError(f"{label} changed during tail read")
    return {"status": "READ_ONLY_TAIL", "path": str(path), "bytes_at_stat": int(info.st_size),
            "offset_bytes": offset, "tail_bytes": len(tail), "tail_sha256": sha_bytes(tail),
            "tail_utf8_replacement": tail.decode("utf-8", errors="replace")}


def ledger_snapshot(path: Path, expected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"status": "NOT_YET_CREATED", "path": str(path)}
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError("candidate result ledger is not a regular nonsymlink file")
    if info.st_size > MAX_LEDGER:
        return {"status": "PRESENT_SIZE_BOUND_EXCEEDED", "path": str(path), "bytes_at_stat": int(info.st_size),
                "complete_readback": False}
    raw = read_regular(path, "candidate result ledger", MAX_LEDGER)
    rows: list[dict[str, Any]] = []
    malformed = 0
    for line in raw.splitlines():
        if not line.strip():
            malformed += 1
            continue
        if len(line) > MAX_LINE:
            malformed += 1
            continue
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            malformed += 1
            continue
        if not isinstance(item, dict):
            malformed += 1
            continue
        rows.append(item)
    ids = [row.get("row_id") for row in rows]
    expected_ids = [row.get("row_id") for row in expected_rows]
    prefix_ok = ids == expected_ids[:len(ids)] and len(ids) <= len(expected_ids)
    identity_fields = ("row_id", "sample_id", "speaker_id", "model_id", "seed", "method_id", "split")
    prefix_identity_ok = prefix_ok and all(
        all(row.get(field) == planned.get(field) for field in identity_fields)
        for planned, row in zip(expected_rows, rows))
    complete_identity_ok = prefix_identity_ok and len(rows) == len(expected_rows)
    status_counts = {status: sum(row.get("status") == status for row in rows)
                     for status in ("PASS", "FAILED", "INCOMPLETE")}
    model_status_counts = {
        model: dict(sorted(Counter(str(row.get("status", "MISSING")) for row in rows
                                   if row.get("model_id") == model).items()))
        for model in ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16",
                      "ditto_hubert_fixed_portrait_attempt18")}
    canonical_ledger = b"".join(canonical(row) + b"\n" for row in rows) == raw
    all_pass = (complete_identity_ok and malformed == 0 and canonical_ledger
                and status_counts == {"PASS": 180, "FAILED": 0, "INCOMPLETE": 0}
                and model_status_counts == {model: {"PASS": 60} for model in model_status_counts})
    return {"status": "BOUNDED_FULL_LEDGER_READ", "path": str(path), "bytes": len(raw),
            "raw_sha256": sha_bytes(raw), "result_rows": len(rows),
            "status_counts": status_counts, "model_status_counts": model_status_counts,
            "ordered_prefix_matches_frozen_plan": prefix_ok,
            "ordered_all_rows_match_frozen_plan": prefix_ok and len(ids) == len(expected_ids),
            "row_identity_prefix_matches_frozen_plan": prefix_identity_ok,
            "all_rows_match_frozen_identity": complete_identity_ok,
            "canonical_jsonl_bytes_match_parsed_rows": canonical_ledger,
            "all_rows_pass": all_pass,
            "malformed_lines": malformed,
            "tail_rows": [{"row_id": row.get("row_id"), "model_id": row.get("model_id"),
                           "method_id": row.get("method_id"), "status": row.get("status")}
                          for row in rows[-10:]]}


def artifact_snapshot(path: Path, label: str, limit: int = MAX_ARTIFACT) -> dict[str, Any]:
    try:
        raw = read_regular(path, label, limit)
    except FileNotFoundError:
        return {"status": "NOT_YET_CREATED", "path": str(path)}
    return {"status": "PRESENT", "path": str(path), "bytes": len(raw), "raw_sha256": sha_bytes(raw)}


def verify_inner_execution(command: Any, execution: dict[str, Any], tools: dict[str, Any],
                           plan_raw_sha: str, rows_raw_sha: str,
                           manifest_raw_sha: str, binding_raw_sha: str) -> dict[str, Any]:
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command) or len(command) < 4:
        raise ValueError("candidate execution command is malformed")
    if command[0] != tools.get("stagea_python"):
        raise ValueError("candidate inner startup interpreter differs from the sealed runtime binding")
    if command[1] != "-c" or len(command[2]) > 1024 * 1024 or command[2] != expected_strict_startup_code():
        raise ValueError("candidate inner command lacks the strict startup-gate Python -c form")
    if command[3] != str(CANDIDATE_DISPATCHER):
        raise ValueError("candidate inner argv does not enter the pinned dispatcher")
    if "--execute-candidate" in command:
        raise ValueError("outer-only execute-candidate flag unexpectedly appears in the inner dispatcher argv")
    required = {
        "--plan": str(PLAN), "--planned-rows": str(ROWS),
        "--expected-plan-raw-sha256": plan_raw_sha, "--output-root": str(CANDIDATE_ROOT),
        "--project-root": str(ROOT),
        "--candidate-repair-binding": str(CANDIDATE_REPAIR_BINDING),
        "--candidate-repair-binding-raw-sha256": REPAIR_BINDING_SHA256,
    }
    for flag, expected in required.items():
        if flag_value(command, flag) != expected:
            raise ValueError(f"candidate inner argv {flag} differs from the exact frozen input/output")
    if flag_value(command, "--wav2lip-python") != tools.get("wav2lip_python"):
        raise ValueError("candidate inner generator interpreter differs from the sealed runtime binding")
    if (execution.get("plan_raw_sha256") != plan_raw_sha
            or execution.get("plan_manifest_raw_sha256") != manifest_raw_sha
            or execution.get("planned_rows_raw_sha256") != rows_raw_sha
            or execution.get("runtime_binding_raw_sha256") != binding_raw_sha
            or execution.get("candidate_repair_binding_raw_sha256") != REPAIR_BINDING_SHA256
            or execution.get("candidate_repair_binding_self_sha256") != REPAIR_BINDING_SELF_SHA256
            or execution.get("candidate_output_namespace") != str(CANDIDATE_NAMESPACE)):
        raise ValueError("candidate execution record does not bind the exact plan/rows/manifest/runtime/repair bindings")
    return {"status": "INNER_ARGV_AND_INPUT_BINDINGS_MATCH", "interpreter": command[0],
            "startup_code_sha256": sha_bytes(command[2].encode("utf-8")),
            "dispatcher_path": command[3], "checked_flags": required}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--launch-receipt-bytes", required=True, type=int)
    parser.add_argument("--launch-receipt-raw-sha256", required=True)
    args = parser.parse_args()
    if args.launch_receipt_bytes <= 0 or args.launch_receipt_bytes > MAX_RECEIPT:
        raise ValueError("launch receipt byte pin is outside the read bound")
    if len(args.launch_receipt_raw_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.launch_receipt_raw_sha256):
        raise ValueError("launch receipt raw SHA must be 64 lowercase hex characters")
    launch_path = JOB / f"candidate180_{args.request_id}.launch_receipt.json"
    receipt, receipt_raw = read_json(launch_path, "candidate180 detached launch receipt", MAX_RECEIPT,
                                     args.launch_receipt_bytes, args.launch_receipt_raw_sha256)
    if (receipt.get("schema") != "cross_tfg_candidate180_detached_launch_receipt_v1"
            or receipt.get("status") != "STARTED_DETACHED"
            or receipt.get("stage_id") != "candidate180"
            or receipt.get("request_id") != args.request_id
            or receipt.get("run_id") != CANDIDATE_RUN_ID
            or receipt.get("controls_run_id") != CONTROL_RUN_ID
            or receipt.get("output_root") != str(CANDIDATE_ROOT)
            or receipt.get("receipt_sha256") != payload_sha(receipt, "receipt_sha256")
            or receipt.get("scope", {}).get("automatic_replay") is not False
            or receipt.get("scope", {}).get("confirm_opened") is not False
            or receipt.get("scope", {}).get("candidate_selection_performed") is not False
            or receipt.get("scope", {}).get("weights_tuned_or_changed") is not False
            or receipt.get("scope", {}).get("api_or_asr_called_by_launcher") is not False
            or receipt.get("scope", {}).get("models_or_scorers_invoked_by_launcher") is not False):
        raise ValueError("launch receipt schema, status, self-hash, output or closed-scope flags differ")
    control_parent = receipt.get("controls_parent", {})
    if (control_parent.get("run_id") != CONTROL_RUN_ID or control_parent.get("rows") != 360
            or control_parent.get("status_counts") != {"PASS": 360, "FAILED": 0, "INCOMPLETE": 0}
            or not isinstance(control_parent.get("results_raw_sha256"), str)
            or len(control_parent.get("results_raw_sha256", "")) != 64
            or not isinstance(control_parent.get("strict_attestation_raw_sha256"), str)
            or len(control_parent.get("strict_attestation_raw_sha256", "")) != 64
            or not isinstance(control_parent.get("execution_raw_sha256"), str)
            or len(control_parent.get("execution_raw_sha256", "")) != 64):
        raise ValueError("launch receipt does not bind the completed all-PASS controls360 parent")
    expected_source_pins = {
        "candidate15_guard": (49071, "438db8d3cd255b5996d3d0cfe12c5653a2f94d3b647931a8e3ad8dc059070486"),
        "candidate_v3_cli03": (43071, "2cf5fc0f73d7c8d5aabbbc457672e487de2f8a88e683976bb58698325a80fdfd"),
        "candidate_dispatcher07": (12824, "f6844b51183b6cf3b57f6a62d0b1b9bd70c3a0d496480fe8cad70313907c16e3"),
        "stagea_core06": (79475, "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"),
        "frontend09": (23249, "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"),
        "strict_bootstrap03": (9525, "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"),
    }
    if set(receipt.get("source_pins", {})) != set(expected_source_pins):
        raise ValueError("launch receipt does not list the exact six candidate execution source pins")
    for name, (size, digest) in expected_source_pins.items():
        if receipt["source_pins"][name] != {"bytes": size, "raw_sha256": digest}:
            raise ValueError(f"launch receipt source pin changed: {name}")
    plan_raw = read_regular(PLAN, "candidate plan", 4 * 1024 * 1024, PLAN_BYTES, PLAN_SHA256)
    rows_raw = read_regular(ROWS, "candidate ordered rows", MAX_LEDGER, ROWS_BYTES, ROWS_SHA256)
    binding_raw = read_regular(BINDING, "runtime binding", MAX_ARTIFACT, BINDING_BYTES, BINDING_SHA256)
    repair_raw = read_regular(CANDIDATE_REPAIR_BINDING, "candidate runtime repair binding", MAX_RECEIPT,
                              REPAIR_BINDING_BYTES, REPAIR_BINDING_SHA256)
    repair = json.loads(repair_raw.decode("utf-8"))
    config_raw = read_regular(CONFIG, "candidate CLI03 config", MAX_RECEIPT, CONFIG_BYTES, CONFIG_SHA256)
    config = json.loads(config_raw.decode("utf-8"))
    plan = json.loads(plan_raw.decode("utf-8"))
    rows = [json.loads(line.decode("utf-8")) for line in rows_raw.splitlines() if line.strip()]
    if (len(rows) != 180 or rows != plan.get("rows")
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or plan.get("run_id") != CANDIDATE_RUN_ID
            or len({row.get("row_id") for row in rows}) != 180
            or b"".join(canonical(row) + b"\n" for row in rows) != rows_raw
            or receipt.get("candidate_plan", {}).get("raw_sha256") != sha_bytes(plan_raw)
            or receipt.get("candidate_plan", {}).get("bytes") != len(plan_raw)
            or receipt.get("candidate_rows", {}).get("raw_sha256") != sha_bytes(rows_raw)
            or receipt.get("candidate_rows", {}).get("bytes") != len(rows_raw)
            or receipt.get("candidate_rows", {}).get("count") != 180
            or receipt.get("runtime_binding", {}).get("raw_sha256") != sha_bytes(binding_raw)
            or receipt.get("runtime_binding", {}).get("bytes") != len(binding_raw)
            or receipt.get("candidate_runtime_repair_binding", {}).get("path") != str(CANDIDATE_REPAIR_BINDING)
            or receipt.get("candidate_runtime_repair_binding", {}).get("bytes") != len(repair_raw)
            or receipt.get("candidate_runtime_repair_binding", {}).get("raw_sha256") != sha_bytes(repair_raw)
            or receipt.get("candidate_runtime_repair_binding", {}).get("repair_binding_sha256")
               != REPAIR_BINDING_SELF_SHA256
            or receipt.get("method_config", {}).get("path") != str(CONFIG)
            or receipt.get("method_config", {}).get("bytes") != len(config_raw)
            or receipt.get("method_config", {}).get("raw_sha256") != sha_bytes(config_raw)):
        raise ValueError("current candidate inputs no longer match the launch receipt")
    cpu_ref = repair.get("cpu_preflight_receipt", {})
    cpu_path = Path(cpu_ref.get("path", ""))
    cpu_raw = read_regular(cpu_path, "actual CLI03 CPU preflight receipt", MAX_RECEIPT,
                           CPU_RECEIPT_BYTES, CPU_RECEIPT_SHA256)
    cpu_receipt = json.loads(cpu_raw.decode("utf-8"))
    if (cpu_receipt.get("schema") != "cross_tfg_v3_candidate_cpu_preflight_v1"
            or cpu_receipt.get("status") != "PASS_CPU_PREFLIGHT"
            or cpu_receipt.get("receipt_sha256") != payload_sha(cpu_receipt, "receipt_sha256")):
        raise ValueError("actual CPU preflight receipt schema/status/self-hash differs")
    if (repair.get("schema") != "cross_tfg_v3_candidate_runtime_repair_binding_v1"
            or repair.get("status") != "PASS_CPU_PREFLIGHT_CANDIDATE_RUNTIME_BINDING_ONLY"
            or repair.get("run_id") != CANDIDATE_RUN_ID
            or repair.get("repair_binding_sha256") != REPAIR_BINDING_SELF_SHA256
            or repair.get("repair_binding_sha256") != payload_sha(repair, "repair_binding_sha256")
            or repair.get("runtime_binding_raw_sha256") != BINDING_SHA256
            or repair.get("candidate_config", {}).get("raw_sha256") != CONFIG_SHA256
            or repair.get("candidate_config", {}).get("bytes") != CONFIG_BYTES
            or cpu_ref.get("path") != "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/candidate_v3_cpu_preflight_attempt03/output/cpu_preflight_receipt_attempt03.json"
            or cpu_ref.get("bytes") != CPU_RECEIPT_BYTES
            or cpu_ref.get("raw_sha256") != CPU_RECEIPT_SHA256
            or repair.get("candidate_output_namespace") != str(CANDIDATE_NAMESPACE)
            or repair.get("candidate_output_leaf") != str(CANDIDATE_ROOT)
            or config.get("source_pins", {}).get("cli", {}).get("sha256") != "2cf5fc0f73d7c8d5aabbbc457672e487de2f8a88e683976bb58698325a80fdfd"):
        raise ValueError("candidate runtime repair binding or CPU/config lineage differs")
    command = receipt.get("command")
    outer_expected = [WAV2LIP_PYTHON, "-B", str(CANDIDATE_RUNNER), "--project-root", str(ROOT),
                      "--candidate-plan", str(PLAN), "--candidate-plan-raw-sha256", sha_bytes(plan_raw),
                      "--planned-rows", str(ROWS), "--planned-rows-raw-sha256", sha_bytes(rows_raw),
                      "--plan-manifest", str(PLAN_MANIFEST),
                      "--plan-manifest-raw-sha256", receipt.get("candidate_plan_manifest", {}).get("raw_sha256", ""),
                      "--runtime-binding", str(BINDING), "--runtime-binding-raw-sha256", sha_bytes(binding_raw),
                      "--candidate-repair-binding", str(CANDIDATE_REPAIR_BINDING),
                      "--candidate-repair-binding-raw-sha256", REPAIR_BINDING_SHA256,
                      "--execute-candidate"]
    if command != outer_expected:
        raise ValueError("launch receipt outer argv differs from the exact guarded candidate15 invocation")

    plan_ref = receipt.get("candidate_plan", {})
    rows_ref = receipt.get("candidate_rows", {})
    manifest_ref = receipt.get("candidate_plan_manifest", {})
    binding_ref = receipt.get("runtime_binding", {})
    if (plan_ref.get("path") != str(PLAN) or rows_ref.get("path") != str(ROWS)
            or manifest_ref.get("path") != str(PLAN_MANIFEST)
            or binding_ref.get("path") != str(BINDING)):
        raise ValueError("launch receipt input paths differ from the sealed local plan")
    if not JOB.is_dir() or JOB.is_symlink():
        raise ValueError("candidate detached control directory is missing or unsafe")
    stdout_path = JOB / f"candidate180_{args.request_id}.stdout.log"
    stderr_path = JOB / f"candidate180_{args.request_id}.stderr.log"
    if receipt.get("stdout_path") != str(stdout_path) or receipt.get("stderr_path") != str(stderr_path):
        raise ValueError("launch receipt log paths differ from the exact request-scoped control files")

    pid = receipt.get("pid")
    ticks = receipt.get("start_ticks")
    if type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0:
        raise ValueError("launch receipt process identity is malformed")
    process = proc_identity(pid, ticks)
    namespace = CANDIDATE_NAMESPACE
    control_root = CONTROL_ROOT
    if namespace == CONTROL_ROOT.parent or namespace.parent != CONTROL_ROOT.parent.parent:
        raise ValueError("candidate namespace is not a distinct sibling of the frozen controls namespace")
    allowed_namespace_entries = {
        CANDIDATE_ROOT,
        namespace / "candidate180.execution.json",
        namespace / "candidate180.candidate_startup_policy_receipt.json",
    }
    if namespace.is_symlink() or not namespace.is_dir() or any(path not in allowed_namespace_entries
                                                               for path in namespace.iterdir()):
        raise ValueError("candidate namespace contains an unexpected sibling or is not a real directory")
    if control_root.is_symlink() or not control_root.is_dir():
        raise ValueError("completed controls360 parent is missing or unsafe")
    controls_execution_raw = read_regular(CONTROLS_EXECUTION, "completed controls360 execution record",
                                          8 * 1024 * 1024,
                                          expected_sha256=control_parent["execution_raw_sha256"])
    controls_execution = json.loads(controls_execution_raw.decode("utf-8"))
    if (controls_execution.get("schema") != "cross_tfg_v3_samehost_dev360_preflight_v1"
            or controls_execution.get("status") != "PASS_METADATA_ONLY"
            or controls_execution.get("execution_output_root") != str(CONTROL_ROOT)
            or controls_execution.get("child_returncode") != 0
            or controls_execution.get("confirm_audio_opened") is not False
            or controls_execution.get("plan_raw_sha256") != control_parent.get("plan_raw_sha256")
            or controls_execution.get("plan_manifest_raw_sha256") != control_parent.get("plan_manifest_raw_sha256")
            or controls_execution.get("runtime_binding_raw_sha256") != BINDING_SHA256):
        raise ValueError("controls360 execution sibling differs from the fully validated parent receipt")
    candidate_root_present = CANDIDATE_ROOT.exists() or CANDIDATE_ROOT.is_symlink()
    if candidate_root_present and (CANDIDATE_ROOT.is_symlink() or not CANDIDATE_ROOT.is_dir()):
        raise ValueError("candidate output root is not a real directory")

    binding = json.loads(binding_raw.decode("utf-8"))
    if (binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1"
            or binding.get("status") != "PASS"
            or binding.get("binding_sha256") != payload_sha(binding, "binding_sha256")
            or binding.get("gpu_uuid") != GPU_UUID):
        raise ValueError("current runtime binding self-hash/status/GPU differs")
    tools = binding.get("execution_bindings", {})
    manifest_path = PLAN_MANIFEST
    manifest_raw = read_regular(manifest_path, "candidate plan manifest", 4 * 1024 * 1024,
                                manifest_ref.get("bytes"), manifest_ref.get("raw_sha256"))
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if (manifest.get("status") != "FROZEN"
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
            or manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or manifest.get("planned_rows_sha256") != sha_bytes(rows_raw)
            or receipt.get("candidate_plan_manifest", {}).get("raw_sha256") != sha_bytes(manifest_raw)
            or receipt.get("candidate_plan_manifest", {}).get("bytes") != len(manifest_raw)):
        raise ValueError("candidate plan manifest self-hash/input lineage differs")
    if receipt.get("candidate_metadata_preflight_readback", {}).get("path") != str(PREFLIGHT):
        raise ValueError("actual candidate metadata-preflight readback path differs")
    initial_ref = receipt["candidate_metadata_preflight_readback"]
    initial_path = Path(initial_ref["path"])
    initial_raw = read_regular(initial_path, "candidate metadata preflight actual readback", 2 * 1024 * 1024,
                               initial_ref.get("bytes"), initial_ref.get("raw_sha256"))
    initial_metadata = json.loads(initial_raw.decode("utf-8"))
    if (initial_metadata.get("status") != "PASS_METADATA_ONLY"
            or initial_metadata.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or initial_metadata.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)
            or initial_metadata.get("candidate_repair_binding_raw_sha256") != REPAIR_BINDING_SHA256
            or initial_metadata.get("execution_performed") is not False):
        raise ValueError("actual candidate metadata preflight does not bind frozen inputs")
    execution_path = CANDIDATE_ROOT.parent / "candidate180.execution.json"
    execution: dict[str, Any] | None = None
    execution_snapshot: dict[str, Any] = {"status": "NOT_YET_CREATED", "path": str(execution_path)}
    startup_snapshot: dict[str, Any] = {"status": "NOT_YET_CREATED"}
    inner_argv: dict[str, Any] | None = None
    if execution_path.exists() or execution_path.is_symlink():
        execution, execution_raw, execution_read = read_transition_json(
            execution_path, "candidate execution record", MAX_ARTIFACT)
        if execution is None:
            execution_snapshot = execution_read
        else:
            assert execution_raw is not None
            if (execution.get("schema") != "cross_tfg_v3_step120_candidate_execution_v1"
                    or execution.get("gpu_uuid") != binding.get("gpu_uuid")
                    or execution.get("status") not in ("RUNNING", "STRICT_POLICY_RECEIPTS_COMPLETE",
                                                           "INCOMPLETE_STRICT_POLICY_RECEIPTS")):
                raise ValueError("candidate execution record has an unknown schema/status/GPU identity")
            if ((execution.get("status") == "RUNNING") != (execution.get("child_returncode") is None)
                    or not isinstance(execution.get("started_unix"), (int, float))
                    or execution.get("started_unix", 0) <= 0
                    or (execution.get("status") == "RUNNING" and execution.get("finished_unix") is not None)
                    or (execution.get("status") != "RUNNING" and not isinstance(execution.get("finished_unix"), (int, float)))):
                raise ValueError("candidate execution record status/exit timestamps are inconsistent")
            inner_argv = verify_inner_execution(execution.get("command"), execution, tools,
                                                sha_bytes(plan_raw), sha_bytes(rows_raw),
                                                manifest_ref.get("raw_sha256", ""), sha_bytes(binding_raw))
            execution_snapshot = {"status": execution.get("status"), "path": str(execution_path),
                                  "bytes": len(execution_raw), "raw_sha256": sha_bytes(execution_raw),
                                  "child_returncode": execution.get("child_returncode"),
                                  "started_unix": execution.get("started_unix"),
                                  "finished_unix": execution.get("finished_unix"),
                                  "inner_argv": inner_argv}
            startup_ref = execution.get("startup_policy_gate_receipt")
            if isinstance(startup_ref, dict):
                startup_path = Path(startup_ref.get("path", ""))
                startup, startup_raw = read_json(startup_path, "candidate startup-policy receipt", MAX_RECEIPT,
                                                 startup_ref.get("bytes"), startup_ref.get("raw_sha256"))
                if (startup_path != CANDIDATE_ROOT.parent / "candidate180.candidate_startup_policy_receipt.json"
                        or startup.get("schema") != "cross_tfg_step120_candidate_startup_policy_gate_v1"
                        or startup.get("status") != "PASS"
                        or startup.get("bootstrap_source_sha256") != "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
                        or startup.get("python_executable") != binding.get("python_environments", {}).get("stagea", {}).get("path")
                        or startup.get("policy") != binding.get("strict_determinism_policy")
                        or startup.get("receipt_sha256") != startup_ref.get("receipt_sha256")
                        or startup.get("receipt_sha256") != payload_sha(startup, "receipt_sha256")):
                    raise ValueError("candidate startup-policy receipt differs from execution lineage")
                startup_snapshot = {"status": startup.get("status"), "path": str(startup_path),
                                    "bytes": len(startup_raw), "raw_sha256": sha_bytes(startup_raw),
                                    "receipt_sha256": startup.get("receipt_sha256"),
                                    "pid": startup.get("pid")}

    status_path = CANDIDATE_ROOT / "control/run_status.json"
    status_snapshot: dict[str, Any] = {"status": "NOT_YET_CREATED", "path": str(status_path)}
    run_status: dict[str, Any] | None = None
    if status_path.exists() or status_path.is_symlink():
        run_status, status_raw, status_read = read_transition_json(
            status_path, "candidate Stage-A run status", MAX_RECEIPT)
        if run_status is None:
            status_snapshot = status_read
        else:
            assert status_raw is not None
            if (run_status.get("schema_version") != "cross_tfg_audio_head_stagea_v1"
                or run_status.get("run_id") != CANDIDATE_RUN_ID
                or run_status.get("plan_path") != str(PLAN)
                or run_status.get("plan_raw_sha256") != sha_bytes(plan_raw)
                or run_status.get("planned_rows_path") != str(ROWS)
                or run_status.get("planned_rows_sha256") != sha_bytes(rows_raw)
                or run_status.get("planned_rows") != 180):
                raise ValueError("candidate run_status does not bind the exact frozen 180-row input")
            if (run_status.get("status") not in ("RUNNING", "COMPLETE", "COMPLETE_WITH_FAILURES")
                or type(run_status.get("completed_rows")) is not int
                or not 0 <= run_status["completed_rows"] <= 180
                or (run_status.get("status") != "RUNNING"
                    and not isinstance(run_status.get("status_counts"), dict))):
                raise ValueError("candidate run_status has impossible status/count values")
            status_snapshot = {"status": run_status.get("status"), "path": str(status_path),
                               "bytes": len(status_raw), "raw_sha256": sha_bytes(status_raw),
                               "completed_rows": run_status.get("completed_rows"),
                               "planned_rows": run_status.get("planned_rows"),
                               "status_counts": run_status.get("status_counts"),
                               "current_row_id": run_status.get("current_row_id"),
                               "last_row_status": run_status.get("last_row_status"),
                               "started_unix": run_status.get("started_unix"),
                               "pid": run_status.get("pid"),
                               "repeat_qc_status": run_status.get("repeat_qc_status"),
                               "ended_unix": run_status.get("ended_unix")}

    final_results_path = CANDIDATE_ROOT / "results.jsonl"
    partial_results_path = CANDIDATE_ROOT / "results.partial.jsonl"
    if final_results_path.exists() or final_results_path.is_symlink():
        result_path = final_results_path
        result_ledger_role = "final_results"
    elif partial_results_path.exists() or partial_results_path.is_symlink():
        result_path = partial_results_path
        result_ledger_role = "in_progress_partial_results"
    else:
        result_path = final_results_path
        result_ledger_role = "not_yet_created"
    result = ledger_snapshot(result_path, rows) if candidate_root_present else {
        "status": "OUTPUT_ROOT_NOT_YET_CREATED", "path": str(result_path)}
    result["ledger_role"] = result_ledger_role
    manifest_path = CANDIDATE_ROOT / "result_manifest.json"
    result_manifest, manifest_raw, manifest_read = read_transition_json(
        manifest_path, "candidate result manifest", MAX_ARTIFACT)
    result_manifest_snapshot = manifest_read
    if result_manifest is not None:
        assert manifest_raw is not None
        if (result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1"
                or result_manifest.get("run_id") != CANDIDATE_RUN_ID
                or result_manifest.get("planned_rows") != 180
                or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")):
            raise ValueError("candidate result manifest schema/identity/self-hash differs")
        if result.get("status") == "BOUNDED_FULL_LEDGER_READ" and (
                result_manifest.get("results_jsonl_sha256") != result.get("raw_sha256")
                or result_manifest.get("result_rows") != result.get("result_rows")):
            raise ValueError("candidate result manifest does not bind the actual result ledger")
        if result.get("status") == "BOUNDED_FULL_LEDGER_READ" and result.get("result_rows") == 180:
            if (result_manifest.get("status_counts") != result.get("status_counts")
                    or result_manifest.get("repeat_qc_status") != "NOT_RUN_SUBRUN"
                    or result_manifest.get("generation_repeat_qc_rows") != 0
                    or result_manifest.get("generation_repeat_qc_jsonl_sha256") != sha_bytes(b"")):
                raise ValueError("complete candidate manifest counts/repeat marker differ from actual rows")
        result_manifest_snapshot.update({"manifest_status": result_manifest.get("status"),
                                         "planned_rows": result_manifest.get("planned_rows"),
                                         "result_rows": result_manifest.get("result_rows"),
                                         "status_counts": result_manifest.get("status_counts"),
                                         "repeat_qc_status": result_manifest.get("repeat_qc_status"),
                                         "results_jsonl_sha256": result_manifest.get("results_jsonl_sha256")})
        if run_status is not None and run_status.get("status") != "RUNNING" and (
                run_status.get("result_manifest_path") != str(manifest_path.resolve())
                or run_status.get("result_manifest_sha256") != sha_bytes(manifest_raw)
                or run_status.get("status_counts") != result_manifest.get("status_counts")
                or run_status.get("repeat_qc_status") != result_manifest.get("repeat_qc_status")):
            raise ValueError("candidate run_status does not point to the exact final result manifest")

    attestation_path = CANDIDATE_ROOT / "runtime_determinism_attestation.json"
    attestation, attestation_raw, attestation_read = read_transition_json(
        attestation_path, "candidate strict determinism attestation", MAX_ARTIFACT)
    attestation_snapshot = attestation_read
    if attestation is not None:
        assert attestation_raw is not None
        if (attestation.get("schema") != "cross_tfg_v3_samehost_renderer_determinism_attestation_v1"
                or attestation.get("status") != "PASS"
                or attestation.get("plan_raw_sha256") != sha_bytes(plan_raw)
                or attestation.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)
                or attestation.get("gpu_uuid") != binding.get("gpu_uuid")
                or attestation.get("qualified_policy_reference_sha256") != "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
                or attestation.get("strict_bootstrap_source_sha256") != "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
                or attestation.get("row_count") != 180
                or attestation.get("native_renderer_row_receipt_count") != 180
                or attestation.get("confirm_audio_read") is not False
                or attestation.get("confirm_data_opened") is not False
                or attestation.get("per_backend_repeat_qualification_receipts") != binding.get("backend_determinism_qualifications")
                or attestation.get("attestation_sha256") != payload_sha(attestation, "attestation_sha256")):
            raise ValueError("candidate strict-attestation schema/self-hash/row/runtime identity differs")
        if execution is None:
            attestation_snapshot["execution_record_status"] = "NOT_YET_READABLE"
        elif execution.get("status") != "STRICT_POLICY_RECEIPTS_COMPLETE":
            attestation_snapshot["execution_record_status"] = "FINALIZATION_IN_PROGRESS"
        attestation_ref = execution.get("strict_determinism_attestation") if execution is not None else None
        if execution is not None and execution.get("status") == "STRICT_POLICY_RECEIPTS_COMPLETE" and (
                not isinstance(attestation_ref, dict) or attestation_ref.get("path") != str(attestation_path.resolve())
                or attestation_ref.get("bytes") != len(attestation_raw)
                or attestation_ref.get("raw_sha256") != sha_bytes(attestation_raw)
                or attestation_ref.get("attestation_sha256") != attestation.get("attestation_sha256")):
            raise ValueError("candidate strict-attestation bytes differ from the execution record reference")
        attestation_snapshot.update({"status": attestation.get("status"), "row_count": attestation.get("row_count"),
                                     "native_renderer_row_receipt_count": attestation.get("native_renderer_row_receipt_count"),
                                     "attestation_sha256": attestation.get("attestation_sha256")})

    fresh_ref = receipt.get("fresh_candidate_metadata_preflight", {})
    fresh_path = Path(fresh_ref.get("path", ""))
    expected_fresh_path = JOB / f"candidate180_{args.request_id}.fresh_metadata_preflight.json"
    if fresh_path != expected_fresh_path:
        raise ValueError("fresh metadata preflight path differs from its request-scoped artifact")
    fresh_raw = read_regular(fresh_path, "fresh metadata preflight readback", MAX_PREFLIGHT,
                             fresh_ref.get("bytes"), fresh_ref.get("raw_sha256"))
    fresh_preflight = json.loads(fresh_raw.decode("utf-8"))
    if (fresh_preflight.get("status") != "PASS_METADATA_ONLY"
            or fresh_preflight.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or fresh_preflight.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)
            or fresh_preflight.get("candidate_repair_binding_raw_sha256") != REPAIR_BINDING_SHA256
            or fresh_preflight.get("execution_performed") is not False
            or fresh_preflight.get("gpu_uuid") != binding.get("gpu_uuid")
            or fresh_ref.get("gpu_uuid") != binding.get("gpu_uuid")):
        raise ValueError("fresh candidate metadata preflight no longer matches its launch receipt")

    initial_ref = receipt.get("candidate_metadata_preflight_readback", {})
    if (initial_ref.get("gpu_uuid") != binding.get("gpu_uuid")
            or receipt.get("runtime_binding", {}).get("gpu_uuid") != binding.get("gpu_uuid")
            or receipt.get("runtime_binding", {}).get("policy_sha256") != binding.get("determinism_policy_sha256")):
        raise ValueError("launch receipt GPU/policy metadata differs from the frozen runtime binding")

    repeat_path = CANDIDATE_ROOT / "generation_repeat_qc.jsonl"
    repeat_snapshot = artifact_snapshot(repeat_path, "candidate planned empty repeat-QC artifact", 1024 * 1024)
    if repeat_snapshot.get("status") == "PRESENT":
        repeat_raw = read_regular(repeat_path, "candidate planned empty repeat-QC artifact", 1024 * 1024)
        if repeat_raw != b"":
            raise ValueError("candidate run created repeat-QC rows although this frozen subrun declares none")
        repeat_snapshot["empty_raw_sha256"] = sha_bytes(repeat_raw)
    if result_manifest_snapshot.get("manifest_status") is not None:
        if (result_manifest_snapshot.get("repeat_qc_status") != "NOT_RUN_SUBRUN"
                or repeat_snapshot.get("status") != "PRESENT"
                or repeat_snapshot.get("bytes") != 0):
            raise ValueError("candidate result manifest/repeat artifact differs from the frozen no-repeat subrun contract")

    output = {
        "schema": "cross_tfg_candidate180_detached_status_snapshot_v1",
        "status": "READ_ONLY_SNAPSHOT",
        "launch_receipt": {"path": str(launch_path), "bytes": len(receipt_raw),
                           "raw_sha256": sha_bytes(receipt_raw), "receipt_sha256": receipt["receipt_sha256"],
                           "request_id": args.request_id, "pid": pid, "start_ticks": ticks},
        "process": process,
        "scope": {"candidate_run_id": CANDIDATE_RUN_ID, "controls_run_id": CONTROL_RUN_ID,
                  "candidate_rows": 180, "confirm_opened": False, "candidate_selection_performed": False,
                  "automatic_replay": False},
        "output_root": {"path": str(CANDIDATE_ROOT), "present": candidate_root_present},
        "controls_execution_record": {"path": str(CONTROLS_EXECUTION), "bytes": len(controls_execution_raw),
                                       "raw_sha256": sha_bytes(controls_execution_raw),
                                       "status": controls_execution.get("status")},
        "logs": {"stdout": bounded_log_tail(Path(receipt["stdout_path"]), stdout_path, "stdout log"),
                 "stderr": bounded_log_tail(Path(receipt["stderr_path"]), stderr_path, "stderr log")},
        "execution_record": execution_snapshot,
        "startup_policy_receipt": startup_snapshot,
        "run_status": status_snapshot,
        "results": result,
        "result_manifest": result_manifest_snapshot,
        "generation_repeat_qc": repeat_snapshot,
        "strict_determinism_attestation": attestation_snapshot,
        "fresh_candidate_metadata_preflight": {"path": str(fresh_path), "bytes": len(fresh_raw),
                                                "raw_sha256": sha_bytes(fresh_raw),
                                                "status": fresh_preflight.get("status"),
                                                "gpu_uuid": fresh_preflight.get("gpu_uuid")},
    }
    output["candidate180_strict_artifact_admission"] = bool(
        output["results"].get("ordered_all_rows_match_frozen_plan") is True
        and output["results"].get("all_rows_match_frozen_identity") is True
        and output["results"].get("malformed_lines") == 0
        and output["results"].get("canonical_jsonl_bytes_match_parsed_rows") is True
        and output["results"].get("status_counts") == {"PASS": 180, "FAILED": 0, "INCOMPLETE": 0}
        and output["results"].get("model_status_counts") == {
            model: {"PASS": 60} for model in ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16",
                                               "ditto_hubert_fixed_portrait_attempt18")}
        and output["results"].get("all_rows_pass") is True
        and output["result_manifest"].get("manifest_status") == "PASS"
        and output["run_status"].get("status") == "COMPLETE"
        and output["run_status"].get("completed_rows") == 180
        and output["run_status"].get("planned_rows") == 180
        and output["run_status"].get("status_counts") == {"PASS": 180, "FAILED": 0, "INCOMPLETE": 0}
        and output["execution_record"].get("status") == "STRICT_POLICY_RECEIPTS_COMPLETE"
        and output["execution_record"].get("child_returncode") == 0
        and output["strict_determinism_attestation"].get("status") == "PASS"
        and output["strict_determinism_attestation"].get("row_count") == 180
        and output["strict_determinism_attestation"].get("native_renderer_row_receipt_count") == 180
        and control_parent.get("status_counts") == {"PASS": 360, "FAILED": 0, "INCOMPLETE": 0})
    print(json.dumps(output, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"schema": "cross_tfg_candidate180_detached_status_snapshot_v1",
                          "status": "READ_ONLY_SNAPSHOT_FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)[:1200]}, sort_keys=True))
        raise SystemExit(2)
