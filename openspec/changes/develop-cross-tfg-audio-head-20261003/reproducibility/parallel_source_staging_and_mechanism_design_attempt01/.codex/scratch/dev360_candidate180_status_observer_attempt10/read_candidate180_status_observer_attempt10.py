#!/usr/bin/env python3
"""Read-only 60-second observer for one already-started candidate180 job.

This program never starts or replays the candidate. It requires the actual
detached launch-receipt bytes as runtime input, stages only the pinned status
reader in a separate create-only job-control directory, and stops at the first
terminal/error/full-denominator observation for root review. Bridge credentials
are inherited from the private runtime environment and are never read or saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
POLL_ROOT = HERE / "polls"
BOOTSTRAP_ROOT = HERE / "bootstrap"
CLIENT = PROJECT / ".codex/scratch/client.py"
CLIENT_PIN = (2706, "d0859d2e76cb59d1d4ec7c7a6e0c391e8ff28eababe5ab2058fb158f4866066f")
STATUS_WORKER = PROJECT / ".codex/scratch/dev360_candidate180_detached_launch_attempt12/read_candidate180_detached_status_attempt13.py"
STATUS_WORKER_PIN = (51011, "982e2172571786d524f6f1dd5e95ee5605e1f47f29ad6fcbdd3e8987e72a484c")
LAUNCHER = PROJECT / ".codex/scratch/dev360_candidate180_detached_launch_attempt12/launch_candidate180_detached_attempt12.py"
LAUNCHER_PIN = (59387, "3307c890e6ee20eee7c0bcaf1e18d560adbf83f72fe282092c8f5e4df82ef688")
PLAN = PROJECT / ".codex/scratch/candidate_v3_cli03_runtime_repair_attempt03/metadata_output_attempt03/candidate_plan_attempt14.json"
PLAN_PIN = (934267, "1a6c8ae99c4b8a1141c6f65748d426ba88034c14badf5ecec497ffc11d62bc60")
ROWS = PROJECT / ".codex/scratch/candidate_v3_cli03_runtime_repair_attempt03/metadata_output_attempt03/candidate_planned_rows_attempt14.jsonl"
ROWS_PIN = (606042, "d781268d02c7495fd35c3bf314848863d57673eeef044a0c5edb19f9e8797f48")
MANIFEST = PROJECT / ".codex/scratch/candidate_v3_cli03_runtime_repair_attempt03/metadata_output_attempt03/candidate_plan_manifest_attempt15.json"
MANIFEST_PIN = (3172, "cd0aa3e7bca14a7676dab7f82d621b93e1a173e586f04efcdbaf3794b022808c")
BINDING = PROJECT / ".codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/runtime_binding_attempt09.json"
BINDING_PIN = (568289, "dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5")
CORE = PROJECT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py"
CORE_PIN = (79475, "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483")
CANDIDATE_GUARD = PROJECT / "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt15.py"
CANDIDATE_GUARD_PIN = (49071, "438db8d3cd255b5996d3d0cfe12c5653a2f94d3b647931a8e3ad8dc059070486")
DISPATCHER = PROJECT / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt07.py"
DISPATCHER_PIN = (12824, "f6844b51183b6cf3b57f6a62d0b1b9bd70c3a0d496480fe8cad70313907c16e3")
FRONTEND = PROJECT / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt09.py"
FRONTEND_PIN = (23249, "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df")
BOOTSTRAP = PROJECT / "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03/sitecustomize.py"
BOOTSTRAP_PIN = (9525, "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65")
CONTROLS_RUNNER = PROJECT / "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_samehost_dev360_controls_attempt18.py"
CONTROLS_RUNNER_PIN = (78172, "30381ee8deb6d0e7880379fc91a042e1e2bb3cca375dd05423319259fd9e8756")

REMOTE_ROOT = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/dev360_candidate180_status_observer_attempt10"
REMOTE_WORKER = REMOTE_ROOT + "/workers/read_candidate180_detached_status_attempt13.py"
CANDIDATE_JOB = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/dev360_candidate180_detached_attempt12"
CANDIDATE_RUN_ID = "cross_tfg_audio_head_20261003_v3_fit40_dev180_cli03_runtime_repair_attempt03"
CONTROLS_RUN_ID = "cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10"
CONTROLS_PLAN_SHA256 = "8a592cc54dc981bec5e2398108e6873190a54c7c1ed2bb7901cd31aad6cc86eb"
CONTROLS_PLAN_MANIFEST_SHA256 = "ab07ee9bed69d067390b9ffae597200916838f8cca6bdb7ddf376b5d5febc214"
CONTROLS_ROWS_SHA256 = "45f1d3130c33c37a513db7b22f76d4107bd821f370a7b3ae2d5ae853894b27c8"
OUTPUT_NAMESPACE = "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cli03_runtime_repair_attempt03"
CANDIDATE_OUTPUT = OUTPUT_NAMESPACE + "/candidate180"
CONTROLS_OUTPUT = "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360"
CONTROL_EXECUTION = "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360.execution.json"
REMOTE_PLAN = "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/candidate180_cli03_runtime_repair_attempt03/candidate_plan_attempt14.json"
REMOTE_ROWS = "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/candidate180_cli03_runtime_repair_attempt03/candidate_planned_rows_attempt14.jsonl"
REMOTE_MANIFEST = "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/candidate180_cli03_runtime_repair_attempt03/candidate_plan_manifest_attempt15.json"
REMOTE_BINDING = "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json"
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
METHOD_ID = "shared_recipient_transition_head_v3_fit40_step120"
MODEL_IDS = ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18")

INTERVAL_SECONDS = 60
CLIENT_TIMEOUT_SECONDS = 120
PROCESS_TIMEOUT_SECONDS = 130
MAX_CLIENT_RESPONSE = 16 * 1024 * 1024
MAX_REMOTE_STDOUT = 8 * 1024 * 1024
MAX_RECEIPT = 4 * 1024 * 1024
MAX_STATUS_LEDGER = 64 * 1024 * 1024


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_hash(value: dict[str, Any], field: str) -> str:
    return sha(canonical({key: item for key, item in value.items() if key != field}))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def pin_file(path: Path, expected: tuple[int, str], label: str, limit: int = 16 * 1024 * 1024) -> bytes:
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_size != expected[0] or info.st_size > limit:
        raise RuntimeError(f"{label} is not the pinned regular file")
    raw = path.read_bytes()
    if len(raw) != expected[0] or sha(raw) != expected[1]:
        raise RuntimeError(f"{label} byte/SHA pin mismatch")
    return raw


def read_json(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def verify_plan_inputs() -> tuple[bytes, bytes, bytes, bytes, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan_raw = pin_file(PLAN, PLAN_PIN, "candidate plan", 4 * 1024 * 1024)
    rows_raw = pin_file(ROWS, ROWS_PIN, "candidate rows", MAX_STATUS_LEDGER)
    manifest_raw = pin_file(MANIFEST, MANIFEST_PIN, "candidate metadata manifest")
    binding_raw = pin_file(BINDING, BINDING_PIN, "runtime binding", 2 * 1024 * 1024)
    plan = read_json(plan_raw, "candidate plan")
    manifest = read_json(manifest_raw, "candidate plan manifest")
    binding = read_json(binding_raw, "runtime binding")
    rows: list[dict[str, Any]] = []
    for line in rows_raw.splitlines():
        if not line:
            raise ValueError("candidate planned rows contain a blank line")
        row = read_json(line, "planned row")
        rows.append(row)
    if (plan.get("plan_sha256") != payload_hash(plan, "plan_sha256")
            or plan.get("run_id") != CANDIDATE_RUN_ID
            or plan.get("status") != "FROZEN"
            or plan.get("rows") != rows
            or len(rows) != 180
            or b"".join(canonical(row) + b"\n" for row in rows) != rows_raw):
        raise ValueError("candidate plan/rows are not the exact canonical frozen 180-row input")
    ids = [row.get("row_id") for row in rows]
    model_counts = {model: sum(row.get("model_id") == model for row in rows) for model in MODEL_IDS}
    methods = {row.get("method_id") for row in rows}
    if (len(set(ids)) != 180 or model_counts != {model: 60 for model in MODEL_IDS}
            or methods != {METHOD_ID} or len({row.get("sample_id") for row in rows}) != 20):
        raise ValueError("candidate plan identity is not 180 ordered rows / three models by 60 / one method")
    if (manifest.get("schema") != "cross_tfg_v3_candidate_only_dev_plan_v1"
            or manifest.get("status") != "FROZEN"
            or manifest.get("run_id") != CANDIDATE_RUN_ID
            or manifest.get("plan_raw_sha256") != sha(plan_raw)
            or manifest.get("plan_sha256") != plan.get("plan_sha256")
            or manifest.get("planned_rows_sha256") != sha(rows_raw)
            or manifest.get("planned_rows") != 180
            or manifest.get("manifest_sha256") != payload_hash(manifest, "manifest_sha256")):
        raise ValueError("candidate manifest is not the self-hashed FROZEN metadata repair")
    if (binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1"
            or binding.get("status") != "PASS"
            or binding.get("gpu_uuid") != GPU_UUID):
        raise ValueError("runtime binding status/GPU identity differs from the launch contract")
    return plan_raw, rows_raw, manifest_raw, binding_raw, plan, binding, rows


def load_launch_receipt(path: Path, byte_count: int, raw_sha: str,
                        plan_raw: bytes, rows_raw: bytes, manifest_raw: bytes,
                        binding_raw: bytes) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("launch receipt must be a caller-supplied actual regular-file path")
    info = path.lstat()
    if byte_count <= 0 or byte_count > MAX_RECEIPT or info.st_size != byte_count:
        raise ValueError("actual launch-receipt byte input differs from the file")
    raw = path.read_bytes()
    if len(raw) != byte_count or sha(raw) != raw_sha:
        raise ValueError("actual launch-receipt raw SHA input differs from the file")
    receipt = read_json(raw, "actual candidate launch receipt")
    rid = receipt.get("request_id")
    if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", rid):
        raise ValueError("actual launch receipt has no safe request id")
    receipt_path = f"{CANDIDATE_JOB}/candidate180_{rid}.launch_receipt.json"
    if (receipt.get("schema") != "cross_tfg_candidate180_detached_launch_receipt_v1"
            or receipt.get("status") != "STARTED_DETACHED"
            or receipt.get("stage_id") != "candidate180"
            or receipt.get("run_id") != CANDIDATE_RUN_ID
            or receipt.get("controls_run_id") != CONTROLS_RUN_ID
            or receipt.get("launch_receipt_path") != receipt_path
            or receipt.get("pid", 0) <= 0 or receipt.get("start_ticks", 0) <= 0
            or receipt.get("output_root") != CANDIDATE_OUTPUT
            or receipt.get("receipt_sha256") != payload_hash(receipt, "receipt_sha256")):
        raise ValueError("actual launch receipt identity, status, output path, or self hash differs")
    refs = (("candidate_plan", REMOTE_PLAN, plan_raw), ("candidate_rows", REMOTE_ROWS, rows_raw),
            ("candidate_plan_manifest", REMOTE_MANIFEST, manifest_raw),
            ("runtime_binding", REMOTE_BINDING, binding_raw))
    for key, remote_path, expected_raw in refs:
        ref = receipt.get(key)
        if (not isinstance(ref, dict) or ref.get("path") != remote_path
                or ref.get("bytes") != len(expected_raw) or ref.get("raw_sha256") != sha(expected_raw)):
            raise ValueError(f"launch receipt does not bind actual pinned {key}")
    if receipt.get("candidate_rows", {}).get("count") != 180:
        raise ValueError("launch receipt candidate row count is not 180")
    runtime_ref = receipt.get("runtime_binding", {})
    if runtime_ref.get("gpu_uuid") != GPU_UUID:
        raise ValueError("launch receipt GPU does not match the qualified binding")
    controls = receipt.get("controls_parent")
    if (not isinstance(controls, dict) or controls.get("run_id") != CONTROLS_RUN_ID
            or controls.get("rows") != 360
            or controls.get("status_counts") != {"PASS": 360, "FAILED": 0, "INCOMPLETE": 0}
            or controls.get("plan_raw_sha256") != CONTROLS_PLAN_SHA256
            or controls.get("plan_manifest_raw_sha256") != CONTROLS_PLAN_MANIFEST_SHA256
            or controls.get("planned_rows_raw_sha256") != CONTROLS_ROWS_SHA256
            or not isinstance(controls.get("execution_raw_sha256"), str)
            or any(not isinstance(controls.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", controls[field])
                   for field in ("results_raw_sha256", "result_manifest_raw_sha256", "execution_raw_sha256",
                                 "strict_attestation_raw_sha256", "repeat_qc_raw_sha256"))
            or controls.get("repeat_qc_raw_sha256") != sha(b"")):
        raise ValueError("launch receipt does not bind the completed all-PASS controls360 parent")
    expected_qualifications = {
        "wav2lip_gan_bac9a81e": "c55b390b6d5189983a4a3e2d181d5ff7606cc191817f125a859800c3e09508d9",
        "musetalk_fixed_portrait_attempt16": "4053e74430e900037ca2f528e3678ad4375241e8e9c653367ea3de31ca94c86d",
        "ditto_hubert_fixed_portrait_attempt18": "4053e74430e900037ca2f528e3678ad4375241e8e9c653367ea3de31ca94c86d",
    }
    if controls.get("qualification_manifests") != expected_qualifications:
        raise ValueError("launch receipt native qualification manifest pins differ from the approved three-model set")
    fresh = receipt.get("fresh_candidate_metadata_preflight")
    expected_fresh_path = f"{CANDIDATE_JOB}/candidate180_{rid}.fresh_metadata_preflight.json"
    if (not isinstance(fresh, dict) or fresh.get("path") != expected_fresh_path
            or fresh.get("status") != "PASS_METADATA_ONLY" or fresh.get("gpu_uuid") != GPU_UUID
            or type(fresh.get("bytes")) is not int or fresh["bytes"] <= 0
            or not isinstance(fresh.get("raw_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", fresh["raw_sha256"])):
        raise ValueError("launch receipt fresh candidate metadata preflight reference is not an actual pinned PASS")
    for log_name in ("stdout", "stderr"):
        if receipt.get(f"{log_name}_path") != f"{CANDIDATE_JOB}/candidate180_{rid}.{log_name}.log":
            raise ValueError("launch receipt logs do not bind the request-scoped candidate job files")
    scope = receipt.get("scope")
    if (not isinstance(scope, dict) or scope.get("confirm_opened") is not False
            or scope.get("candidate_selection_performed") is not False
            or scope.get("automatic_replay") is not False):
        raise ValueError("launch receipt scope reports an unexpected action")
    return receipt, raw


def write_observation(directory: Path, obj: dict[str, Any]) -> None:
    body = dict(obj)
    body["observation_sha256"] = sha(canonical(body))
    create_only(directory / "observation.json", canonical(body) + b"\n")


def request_directory(parent: Path, label: str) -> tuple[str, Path]:
    poll_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + label + "_" + uuid.uuid4().hex[:8]
    directory = parent / poll_id
    directory.mkdir(parents=True, exist_ok=False)
    return poll_id, directory


def clean_envelope(raw: bytes) -> tuple[dict[str, Any], str]:
    value = read_json(raw, "bridge client response")
    trunc = value.get("truncated")
    valid = (value.get("ok") is True and value.get("code") == 0
             and value.get("timed_out") is False and value.get("incomplete_stream") is False
             and value.get("stream_error") in (None, "")
             and isinstance(trunc, dict) and trunc == {"stdout": False, "stderr": False}
             and value.get("stderr") == "" and isinstance(value.get("stdout"), str))
    return value, "CLEAN" if valid else "BRIDGE_ENVELOPE_NOT_CLEAN"


def clean_upload_response(raw: bytes, request: dict[str, Any],
                          expected: tuple[str, str, tuple[int, str]] | None) -> tuple[dict[str, Any] | None, str]:
    if expected is None:
        return None, "UPLOAD_EXPECTED_PIN_MISSING"
    expected_local, expected_remote, expected_pin = expected
    if (request.get("op") != "upload" or request.get("local") != expected_local
            or request.get("remote") != expected_remote):
        return None, "UPLOAD_REQUEST_IDENTITY_MISMATCH"
    try:
        value = read_json(raw, "bridge upload response")
    except (UnicodeError, json.JSONDecodeError, ValueError):
        return None, "UPLOAD_RESPONSE_NOT_JSON_OBJECT"
    required_fields = {"ok", "local", "remote", "size", "sha256"}
    if set(value) != required_fields:
        return value, "UPLOAD_RESPONSE_SCHEMA_MISMATCH"
    valid = (value.get("ok") is True
             and value.get("local") == expected_local
             and value.get("remote") == expected_remote
             and type(value.get("size")) is int and value.get("size") == expected_pin[0]
             and value.get("sha256") == expected_pin[1])
    return value, "CLEAN" if valid else "UPLOAD_RESPONSE_PIN_MISMATCH"


def call_bridge(label: str, request: dict[str, Any], parent: Path,
                parse_stdout_json: bool = True,
                expected_upload: tuple[str, str, tuple[int, str]] | None = None
                ) -> tuple[bool, str, dict[str, Any] | None, str]:
    call_id, directory = request_directory(parent, label)
    req_raw = canonical(request) + b"\n"
    request_path = directory / "request.json"
    response_path = directory / "response.json"
    create_only(request_path, req_raw)
    port = os.environ.get("BRIDGE_PORT_PRIVATE", "")
    if not port.isdecimal() or not os.environ.get("BRIDGE_TOKEN_PRIVATE"):
        create_only(response_path, b"")
        write_observation(directory, {"schema": "cross_tfg_candidate180_status_observer_call_v1",
            "status": "STOPPED_ROOT_REVIEW_REQUIRED", "call_id": call_id, "call": label,
            "request_path": str(request_path), "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
            "response_path": str(response_path), "response_bytes": 0, "response_raw_sha256": sha(b""),
            "reason": "PRIVATE_BRIDGE_RUNTIME_ENV_MISSING", "automatic_experiment_pass": False})
        return False, "PRIVATE_BRIDGE_RUNTIME_ENV_MISSING", None, call_id
    try:
        completed = subprocess.run([sys.executable, str(CLIENT), "--port", port, "--json-file", str(request_path),
                                    "--timeout", str(CLIENT_TIMEOUT_SECONDS)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=PROCESS_TIMEOUT_SECONDS, check=False)
        resp = completed.stdout[:MAX_CLIENT_RESPONSE]
        create_only(response_path, resp)
        if completed.returncode != 0 or completed.stderr or not resp or len(completed.stdout) > MAX_CLIENT_RESPONSE:
            write_observation(directory, {"schema": "cross_tfg_candidate180_status_observer_call_v1",
                "status": "STOPPED_ROOT_REVIEW_REQUIRED", "call_id": call_id, "call": label,
                "request_path": str(request_path), "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
                "response_path": str(response_path), "response_bytes": len(resp), "response_raw_sha256": sha(resp),
                "client_exit_code": completed.returncode, "client_stderr_bytes": len(completed.stderr),
                "client_stderr_sha256": sha(completed.stderr[:65536]), "reason": "BRIDGE_CLIENT_NONCLEAN_RESULT",
                "automatic_experiment_pass": False})
            return False, "BRIDGE_CLIENT_NONCLEAN_RESULT", None, call_id
        if request.get("op") == "upload":
            transfer, transfer_status = clean_upload_response(resp, request, expected_upload)
            write_observation(directory, {"schema": "cross_tfg_candidate180_status_observer_call_v1",
                "status": "CALL_COMPLETE" if transfer_status == "CLEAN" else "STOPPED_ROOT_REVIEW_REQUIRED",
                "call_id": call_id, "call": label, "request_path": str(request_path),
                "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
                "response_path": str(response_path), "response_bytes": len(resp), "response_raw_sha256": sha(resp),
                "reason": transfer_status,
                "upload_response": ({"ok": transfer.get("ok"), "local": transfer.get("local"),
                    "remote": transfer.get("remote"), "size": transfer.get("size"),
                    "sha256": transfer.get("sha256")} if transfer is not None else None),
                "automatic_experiment_pass": False})
            return transfer_status == "CLEAN", transfer_status, transfer, call_id
        if expected_upload is not None or parse_stdout_json is False:
            write_observation(directory, {"schema": "cross_tfg_candidate180_status_observer_call_v1",
                "status": "STOPPED_ROOT_REVIEW_REQUIRED", "call_id": call_id, "call": label,
                "request_path": str(request_path), "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
                "response_path": str(response_path), "response_bytes": len(resp), "response_raw_sha256": sha(resp),
                "reason": "EXEC_RESPONSE_MODE_INVALID", "automatic_experiment_pass": False})
            return False, "EXEC_RESPONSE_MODE_INVALID", None, call_id
        envelope, envelope_status = clean_envelope(resp)
        out = envelope.get("stdout") if isinstance(envelope, dict) else None
        parsed: dict[str, Any] | None = None
        reason = envelope_status
        if envelope_status == "CLEAN" and not parse_stdout_json:
            parsed = envelope
        elif envelope_status == "CLEAN" and isinstance(out, str):
            try:
                parsed = read_json(out.encode("utf-8"), "remote operation stdout")
                reason = "CLEAN"
            except (UnicodeError, json.JSONDecodeError, ValueError):
                reason = "REMOTE_STDOUT_NOT_JSON_OBJECT"
        write_observation(directory, {"schema": "cross_tfg_candidate180_status_observer_call_v1",
            "status": "CALL_COMPLETE" if reason == "CLEAN" else "STOPPED_ROOT_REVIEW_REQUIRED",
            "call_id": call_id, "call": label, "request_path": str(request_path),
            "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
            "response_path": str(response_path), "response_bytes": len(resp), "response_raw_sha256": sha(resp),
            "remote_code": envelope.get("code"), "remote_stdout_bytes": len(out.encode("utf-8")) if isinstance(out, str) else None,
            "remote_stderr_bytes": len(envelope.get("stderr", "")) if isinstance(envelope.get("stderr"), str) else None,
            "reason": reason, "automatic_experiment_pass": False})
        return reason == "CLEAN", reason, parsed, call_id
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout if isinstance(exc.stdout, bytes) else b""
        partial = partial[:MAX_CLIENT_RESPONSE]
        create_only(response_path, partial)
        write_observation(directory, {"schema": "cross_tfg_candidate180_status_observer_call_v1",
            "status": "STOPPED_ROOT_REVIEW_REQUIRED", "call_id": call_id, "call": label,
            "request_path": str(request_path), "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
            "response_path": str(response_path), "response_bytes": len(partial), "response_raw_sha256": sha(partial),
            "reason": "BRIDGE_CLIENT_TIMEOUT", "automatic_experiment_pass": False})
        return False, "BRIDGE_CLIENT_TIMEOUT", None, call_id
    except OSError as exc:
        create_only(response_path, b"")
        write_observation(directory, {"schema": "cross_tfg_candidate180_status_observer_call_v1",
            "status": "STOPPED_ROOT_REVIEW_REQUIRED", "call_id": call_id, "call": label,
            "request_path": str(request_path), "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
            "response_path": str(response_path), "response_bytes": 0, "response_raw_sha256": sha(b""),
            "reason": "BRIDGE_CLIENT_START_FAILED", "exception_type": type(exc).__name__,
            "automatic_experiment_pass": False})
        return False, "BRIDGE_CLIENT_START_FAILED", None, call_id


def validate_snapshot(snapshot: dict[str, Any], receipt: dict[str, Any], receipt_raw: bytes,
                      binding: dict[str, Any], rows: list[dict[str, Any]],
                      snapshot_raw_sha: str) -> tuple[bool, str, dict[str, Any]]:
    if snapshot.get("schema") != "cross_tfg_candidate180_detached_status_snapshot_v1":
        return False, "STATUS_SCHEMA_MISMATCH", {}
    if snapshot.get("status") != "READ_ONLY_SNAPSHOT":
        return False, "REMOTE_STATUS_WORKER_ERROR", {"reported_status": snapshot.get("status")}
    launch = snapshot.get("launch_receipt")
    if (not isinstance(launch, dict) or launch.get("path") != receipt.get("launch_receipt_path")
            or launch.get("bytes") != len(receipt_raw) or launch.get("raw_sha256") != sha(receipt_raw)
            or launch.get("receipt_sha256") != receipt.get("receipt_sha256")
            or launch.get("request_id") != receipt.get("request_id")
            or launch.get("pid") != receipt.get("pid") or launch.get("start_ticks") != receipt.get("start_ticks")):
        return False, "ACTUAL_LAUNCH_RECEIPT_BINDING_MISMATCH", {}
    if snapshot.get("scope") != {"candidate_run_id": CANDIDATE_RUN_ID, "controls_run_id": CONTROLS_RUN_ID,
            "candidate_rows": 180, "confirm_opened": False, "candidate_selection_performed": False,
            "automatic_replay": False}:
        return False, "SCOPE_OR_RUN_IDENTITY_MISMATCH", {}
    output_root = snapshot.get("output_root")
    if not isinstance(output_root, dict) or output_root.get("path") != CANDIDATE_OUTPUT:
        return False, "OUTPUT_ROOT_MISMATCH", {}
    controls = snapshot.get("controls_execution_record")
    parent = receipt.get("controls_parent", {})
    if (not isinstance(controls, dict) or controls.get("path") != CONTROL_EXECUTION
            or controls.get("raw_sha256") != parent.get("execution_raw_sha256")
            or controls.get("status") != "PASS_METADATA_ONLY"):
        return False, "CONTROLS_PARENT_LINEAGE_MISMATCH", {}
    process = snapshot.get("process")
    if (not isinstance(process, dict) or process.get("matches_launch") is not True
            or process.get("observed_start_ticks") != receipt.get("start_ticks")
            or process.get("state") not in ("RUNNING", "EXITED_ZOMBIE")):
        return False, "CANDIDATE_PROCESS_IDENTITY_CHANGED", {}
    if type(snapshot.get("candidate180_strict_artifact_admission")) is not bool:
        return False, "ADMISSION_FIELD_INVALID", {}
    results = snapshot.get("results")
    if not isinstance(results, dict):
        return False, "RESULTS_LEDGER_SNAPSHOT_MISSING", {}
    result_state = results.get("status")
    if result_state not in ("OUTPUT_ROOT_NOT_YET_CREATED", "BOUNDED_FULL_LEDGER_READ"):
        return False, "RESULTS_LEDGER_STATUS_INVALID", {"ledger_status": result_state}
    if result_state == "BOUNDED_FULL_LEDGER_READ":
        expected_result_paths = {CANDIDATE_OUTPUT + "/results.jsonl", CANDIDATE_OUTPUT + "/results.partial.jsonl"}
        count = results.get("result_rows")
        counts = results.get("status_counts")
        model_counts = results.get("model_status_counts")
        expected_ids = [row.get("row_id") for row in rows]
        if (type(count) is not int or not 0 <= count <= 180
                or type(results.get("malformed_lines")) is not int or results["malformed_lines"] != 0
                or results.get("path") not in expected_result_paths
                or not isinstance(results.get("raw_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", results.get("raw_sha256", ""))
                or not isinstance(counts, dict) or set(counts) != {"PASS", "FAILED", "INCOMPLETE"}
                or any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) != count
                or not isinstance(model_counts, dict) or set(model_counts) != set(MODEL_IDS)
                or results.get("row_identity_prefix_matches_frozen_plan") is not True
                or results.get("canonical_jsonl_bytes_match_parsed_rows") is not True
                or results.get("ordered_prefix_matches_frozen_plan") is not True):
            return False, "RESULTS_LEDGER_CONTENT_OR_COUNT_INVALID", {}
        merged = {status: 0 for status in ("PASS", "FAILED", "INCOMPLETE")}
        model_total = 0
        for model in MODEL_IDS:
            per_model = model_counts[model]
            if (not isinstance(per_model, dict) or any(k not in merged for k in per_model)
                    or any(type(v) is not int or v < 0 for v in per_model.values())):
                return False, "MODEL_STATUS_COUNTS_INVALID", {"model_id": model}
            total = sum(per_model.values())
            if total > 60:
                return False, "MODEL_STATUS_DENOMINATOR_EXCEEDED", {"model_id": model}
            model_total += total
            for status, value in per_model.items():
                merged[status] += value
        if model_total != count or merged != counts:
            return False, "MODEL_AND_GLOBAL_STATUS_COUNTS_DISAGREE", {}
        if results.get("ordered_all_rows_match_frozen_plan") is True and count != 180:
            return False, "FULL_ORDERED_LEDGER_COUNT_MISMATCH", {}
        if count == 180 and (results.get("ordered_all_rows_match_frozen_plan") is not True
                or results.get("all_rows_match_frozen_identity") is not True):
            return False, "FULL_DENOMINATOR_IDENTITY_MISMATCH", {}
    run_status = snapshot.get("run_status")
    if not isinstance(run_status, dict):
        return False, "RUN_STATUS_SNAPSHOT_MISSING", {}
    if run_status.get("status") not in ("NOT_YET_CREATED", "RUNNING", "COMPLETE", "COMPLETE_WITH_FAILURES"):
        return False, "RUN_STATUS_UNKNOWN", {}
    if run_status.get("status") not in ("NOT_YET_CREATED",):
        if (run_status.get("path") != CANDIDATE_OUTPUT + "/control/run_status.json"
                or run_status.get("planned_rows") != 180 or type(run_status.get("completed_rows")) is not int
                or not 0 <= run_status["completed_rows"] <= 180):
            return False, "RUN_STATUS_DENOMINATOR_INVALID", {}
        if run_status.get("status") == "RUNNING" and result_state == "BOUNDED_FULL_LEDGER_READ":
            if results["result_rows"] > run_status["completed_rows"] + 1:
                return False, "LEDGER_AHEAD_OF_RUN_STATUS", {}
        if run_status.get("status") != "RUNNING":
            return False, "RUN_STATUS_TERMINAL_ROOT_REVIEW_REQUIRED", {"run_status": run_status.get("status")}
    execution = snapshot.get("execution_record")
    if not isinstance(execution, dict):
        return False, "EXECUTION_RECORD_MISSING", {}
    if execution.get("status") not in ("NOT_YET_CREATED", "RUNNING", "STRICT_POLICY_RECEIPTS_COMPLETE", "INCOMPLETE_STRICT_POLICY_RECEIPTS"):
        return False, "EXECUTION_RECORD_STATUS_INVALID", {"status": execution.get("status")}
    if execution.get("status") not in ("NOT_YET_CREATED", "RUNNING"):
        return False, "EXECUTION_RECORD_TERMINAL_ROOT_REVIEW_REQUIRED", {"status": execution.get("status")}
    if (execution.get("status") == "RUNNING" and
            (not isinstance(execution.get("inner_argv"), dict)
             or execution["inner_argv"].get("status") != "INNER_ARGV_AND_INPUT_BINDINGS_MATCH")):
        return False, "INNER_COMMAND_BINDING_MISMATCH", {}
    manifest_snapshot = snapshot.get("result_manifest")
    if not isinstance(manifest_snapshot, dict) or manifest_snapshot.get("status") != "NOT_YET_CREATED":
        return False, "RESULT_MANIFEST_PRESENT_OR_TRANSITIONAL_ROOT_REVIEW_REQUIRED", {"manifest": manifest_snapshot}
    attestation_snapshot = snapshot.get("strict_determinism_attestation")
    if not isinstance(attestation_snapshot, dict) or attestation_snapshot.get("status") != "NOT_YET_CREATED":
        return False, "STRICT_ATTESTATION_PRESENT_OR_TRANSITIONAL_ROOT_REVIEW_REQUIRED", {"attestation": attestation_snapshot}
    startup = snapshot.get("startup_policy_receipt")
    if not isinstance(startup, dict) or startup.get("status") not in ("NOT_YET_CREATED", "PASS"):
        return False, "STARTUP_POLICY_RECEIPT_INVALID", {"startup": startup}
    fresh = snapshot.get("fresh_candidate_metadata_preflight")
    fresh_ref = receipt.get("fresh_candidate_metadata_preflight")
    if (not isinstance(fresh, dict) or not isinstance(fresh_ref, dict)
            or fresh.get("path") != fresh_ref.get("path")
            or fresh.get("bytes") != fresh_ref.get("bytes")
            or fresh.get("raw_sha256") != fresh_ref.get("raw_sha256")
            or fresh.get("status") != "PASS_METADATA_ONLY"
            or fresh.get("gpu_uuid") != GPU_UUID or fresh_ref.get("gpu_uuid") != GPU_UUID):
        return False, "FRESH_PREFLIGHT_LINEAGE_MISMATCH", {}
    if snapshot.get("candidate180_strict_artifact_admission") is True:
        return False, "CANDIDATE_ADMISSION_TRUE_ROOT_REVIEW_REQUIRED", {}
    if result_state == "BOUNDED_FULL_LEDGER_READ" and results.get("result_rows") == 180:
        return False, "FULL_PLANNED_DENOMINATOR_ROOT_REVIEW_REQUIRED", {
            "result_rows": 180, "status_counts": results.get("status_counts")}
    if process.get("state") != "RUNNING":
        return False, "PROCESS_NOT_RUNNING_ROOT_REVIEW_REQUIRED", {"process": process}
    detail = {"snapshot_stdout_raw_sha256": snapshot_raw_sha, "status": snapshot.get("status"),
              "request_id": receipt.get("request_id"), "pid": receipt.get("pid"),
              "start_ticks": receipt.get("start_ticks"), "planned_rows": 180,
              "result_rows": results.get("result_rows"), "status_counts": results.get("status_counts"),
              "model_status_counts": results.get("model_status_counts"), "malformed_lines": results.get("malformed_lines"),
              "run_status": run_status, "execution_record": execution,
              "candidate_admission_boolean": snapshot.get("candidate180_strict_artifact_admission"),
              "automatic_experiment_pass": False}
    return True, "RUNNING_OBSERVED", detail


def run_call(label: str, request: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str]:
    clean, reason, result, call_id = call_bridge(label, request, POLL_ROOT if label.startswith("poll") else BOOTSTRAP_ROOT)
    if not clean or result is None:
        return False, None, reason
    if result.get("status") not in ("PASS_EMPTY_STATUS_NAMESPACE_CREATED", "PASS_STATUS_WORKER_UPLOAD", "READ_ONLY_SNAPSHOT"):
        return False, result, f"REMOTE_OPERATION_STATUS_{result.get('status')}"
    return True, result, "CLEAN"


def bootstrap_remote_worker() -> bool:
    BOOTSTRAP_ROOT.mkdir(parents=True, exist_ok=True)
    create_command = ("set -eu; CONTROL='" + REMOTE_ROOT + "'; test ! -e \"$CONTROL\"; "
        "mkdir -m 700 -- \"$CONTROL\"; mkdir -m 700 -- \"$CONTROL/workers\"; "
        "printf '%s\\n' '{\"status\":\"PASS_EMPTY_STATUS_NAMESPACE_CREATED\"}'")
    ok, result, reason = run_call("bootstrap_namespace", {"op": "exec", "command": create_command, "timeout": 45})
    if not ok:
        record_stop("STATUS_WORKER_NAMESPACE_CREATE_FAILED", {"reason": reason, "response": result})
        return False
    request = {"op": "upload", "local": str(STATUS_WORKER), "remote": REMOTE_WORKER, "timeout": 90}
    ok, reason, result, _ = call_bridge("bootstrap_upload", request, BOOTSTRAP_ROOT,
        parse_stdout_json=False, expected_upload=(str(STATUS_WORKER), REMOTE_WORKER, STATUS_WORKER_PIN))
    if not ok:
        record_stop("STATUS_WORKER_UPLOAD_FAILED", {"reason": reason, "response": result})
        return False
    worker_size, worker_sha256 = STATUS_WORKER_PIN
    verify_command = ("set -eu; python3 -B -c 'import hashlib,json,os,stat; p=\"" + REMOTE_WORKER +
        "\"; s=os.lstat(p); d=open(p,\"rb\").read(2097153); good=stat.S_ISREG(s.st_mode) and not os.path.islink(p) "
        + "and s.st_size==" + str(worker_size) + " and len(d)==" + str(worker_size) 
        + " and hashlib.sha256(d).hexdigest()==\"" + worker_sha256 + "\"; "
        + "print(json.dumps({\"status\":\"PASS_STATUS_WORKER_UPLOAD\" if good else \"STATUS_WORKER_PIN_MISMATCH\",\"bytes\":len(d),\"sha256\":hashlib.sha256(d).hexdigest()},sort_keys=True)); "
        + "raise SystemExit(0 if good else 2)'" )
    ok, result, reason = run_call("bootstrap_upload_verify", {"op": "exec", "command": verify_command, "timeout": 45})
    if not ok:
        record_stop("STATUS_WORKER_REMOTE_PIN_VERIFY_FAILED", {"reason": reason, "response": result})
        return False
    return True


def poll_once(receipt: dict[str, Any], receipt_raw: bytes, rows: list[dict[str, Any]], binding: dict[str, Any]) -> tuple[bool, str]:
    rid = receipt["request_id"]
    command = ("set -eu; test -f '" + REMOTE_WORKER + "'; test ! -L '" + REMOTE_WORKER + "'; "
        "PYTHONDONTWRITEBYTECODE=1 python3 -B '" + REMOTE_WORKER + "' --request-id '" + rid + "' "
        "--launch-receipt-bytes '" + str(len(receipt_raw)) + "' --launch-receipt-raw-sha256 '" + sha(receipt_raw) + "'")
    poll_id, call_dir = request_directory(POLL_ROOT, "poll")
    request = {"op": "exec", "command": command, "timeout": 120}
    req_raw = canonical(request) + b"\n"
    req_path = call_dir / "request.json"
    response_path = call_dir / "response.json"
    observation_path = call_dir / "observation.json"
    create_only(req_path, req_raw)
    port = os.environ.get("BRIDGE_PORT_PRIVATE", "")
    if not port.isdecimal() or not os.environ.get("BRIDGE_TOKEN_PRIVATE"):
        create_only(response_path, b"")
        observation = {"schema": "cross_tfg_candidate180_status_observation_v1", "status": "STOPPED_ROOT_REVIEW_REQUIRED",
            "poll_id": poll_id, "reason": "PRIVATE_BRIDGE_RUNTIME_ENV_MISSING", "request_path": str(req_path),
            "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw), "response_path": str(response_path),
            "response_bytes": 0, "response_raw_sha256": sha(b""), "automatic_experiment_pass": False}
        write_observation(call_dir, observation)
        return False, "PRIVATE_BRIDGE_RUNTIME_ENV_MISSING"
    try:
        completed = subprocess.run([sys.executable, str(CLIENT), "--port", port, "--json-file", str(req_path),
                                     "--timeout", str(CLIENT_TIMEOUT_SECONDS)], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=PROCESS_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout if isinstance(exc.stdout, bytes) else b""
        partial = partial[:MAX_CLIENT_RESPONSE]
        create_only(response_path, partial)
        write_observation(call_dir, {"schema": "cross_tfg_candidate180_status_observation_v1",
            "status": "STOPPED_ROOT_REVIEW_REQUIRED", "poll_id": poll_id, "reason": "BRIDGE_CLIENT_TIMEOUT",
            "request_path": str(req_path), "request_bytes": len(req_raw), "request_raw_sha256": sha(req_raw),
            "response_path": str(response_path), "response_bytes": len(partial), "response_raw_sha256": sha(partial),
            "automatic_experiment_pass": False})
        return False, "BRIDGE_CLIENT_TIMEOUT"
    except OSError as exc:
        create_only(response_path, b"")
        write_observation(call_dir, {"schema": "cross_tfg_candidate180_status_observation_v1",
            "status": "STOPPED_ROOT_REVIEW_REQUIRED", "poll_id": poll_id, "reason": "BRIDGE_CLIENT_START_FAILED",
            "exception_type": type(exc).__name__, "request_path": str(req_path), "request_bytes": len(req_raw),
            "request_raw_sha256": sha(req_raw), "response_path": str(response_path),
            "response_bytes": 0, "response_raw_sha256": sha(b""), "automatic_experiment_pass": False})
        return False, "BRIDGE_CLIENT_START_FAILED"
    raw = completed.stdout[:MAX_CLIENT_RESPONSE]
    create_only(response_path, raw)
    base = {"schema": "cross_tfg_candidate180_status_observation_v1", "poll_id": poll_id,
            "observed_at_utc": utc_now(), "request_path": str(req_path), "request_bytes": len(req_raw),
            "request_raw_sha256": sha(req_raw), "response_path": str(response_path),
            "response_bytes": len(raw), "response_raw_sha256": sha(raw),
            "client_exit_code": completed.returncode, "client_stderr_bytes": len(completed.stderr),
            "client_stderr_sha256": sha(completed.stderr[:65536]), "automatic_experiment_pass": False}
    if completed.returncode != 0 or completed.stderr or not raw or len(completed.stdout) > MAX_CLIENT_RESPONSE:
        write_observation(call_dir, {**base, "status": "STOPPED_ROOT_REVIEW_REQUIRED", "reason": "BRIDGE_CLIENT_NONCLEAN_RESULT"})
        return False, "BRIDGE_CLIENT_NONCLEAN_RESULT"
    try:
        envelope = read_json(raw, "bridge response envelope")
    except Exception as exc:
        write_observation(call_dir, {**base, "status": "STOPPED_ROOT_REVIEW_REQUIRED", "reason": "BRIDGE_ENVELOPE_INVALID",
                                     "exception_type": type(exc).__name__})
        return False, "BRIDGE_ENVELOPE_INVALID"
    if (envelope.get("ok") is not True or envelope.get("code") != 0 or envelope.get("timed_out") is not False
            or envelope.get("incomplete_stream") is not False or envelope.get("stream_error") not in (None, "")
            or envelope.get("truncated") != {"stdout": False, "stderr": False} or envelope.get("stderr") != ""
            or not isinstance(envelope.get("stdout"), str) or not envelope.get("stdout")):
        write_observation(call_dir, {**base, "status": "STOPPED_ROOT_REVIEW_REQUIRED", "reason": "BRIDGE_ENVELOPE_NOT_CLEAN",
            "remote_code": envelope.get("code"), "remote_timed_out": envelope.get("timed_out"),
            "remote_incomplete_stream": envelope.get("incomplete_stream"), "remote_truncated": envelope.get("truncated")})
        return False, "BRIDGE_ENVELOPE_NOT_CLEAN"
    stdout = envelope["stdout"].encode("utf-8")
    if len(stdout) > MAX_REMOTE_STDOUT:
        write_observation(call_dir, {**base, "status": "STOPPED_ROOT_REVIEW_REQUIRED", "reason": "REMOTE_STATUS_STDOUT_OVER_BOUND",
                                     "remote_stdout_bytes": len(stdout), "remote_stdout_raw_sha256": sha(stdout)})
        return False, "REMOTE_STATUS_STDOUT_OVER_BOUND"
    try:
        snapshot = read_json(stdout, "candidate status worker snapshot")
        clean, reason, detail = validate_snapshot(snapshot, receipt, receipt_raw, binding, rows, sha(stdout))
    except Exception as exc:
        clean, reason, detail = False, "STATUS_VALIDATION_EXCEPTION", {"exception_type": type(exc).__name__, "error": str(exc)[:500]}
    write_observation(call_dir, {**base, "status": "RUNNING_OBSERVED" if clean else "STOPPED_ROOT_REVIEW_REQUIRED",
        "reason": reason, "snapshot_stdout_bytes": len(stdout), "snapshot_stdout_raw_sha256": sha(stdout),
        "launch_request_id": receipt["request_id"], "candidate_run_id": CANDIDATE_RUN_ID,
        "candidate_pid": receipt["pid"], "candidate_start_ticks": receipt["start_ticks"],
        "planned_rows": 180, "detail": detail, "root_consumption_required": not clean})
    return clean, reason


def record_stop(reason: str, detail: dict[str, Any] | None = None) -> None:
    stop = {"schema": "cross_tfg_candidate180_status_observer_stop_v1", "status": "STOPPED_ROOT_REVIEW_REQUIRED",
        "reason": reason, "stopped_at_utc": utc_now(), "planned_rows": 180,
        "candidate_run_id": CANDIDATE_RUN_ID, "automatic_experiment_pass": False,
        "root_full_readback_required": True, "detail": detail or {}}
    stop["stop_sha256"] = sha(canonical(stop))
    create_only(HERE / ("stop_" + str(int(time.time())) + "_" + uuid.uuid4().hex[:8] + ".json"), canonical(stop) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt-path", required=True)
    parser.add_argument("--launch-receipt-bytes", required=True, type=int)
    parser.add_argument("--launch-receipt-raw-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.launch_receipt_raw_sha256):
        raise ValueError("actual launch receipt SHA must be lowercase 64-character hex")
    pins = ((CLIENT, CLIENT_PIN, "bridge client"), (STATUS_WORKER, STATUS_WORKER_PIN, "candidate status worker"),
        (LAUNCHER, LAUNCHER_PIN, "candidate detached launcher"),
        (PLAN, PLAN_PIN, "candidate plan"), (ROWS, ROWS_PIN, "candidate ordered rows"),
        (MANIFEST, MANIFEST_PIN, "candidate repaired manifest"), (BINDING, BINDING_PIN, "runtime binding"),
        (CORE, CORE_PIN, "Stage-A core"), (CANDIDATE_GUARD, CANDIDATE_GUARD_PIN, "candidate guard"),
        (DISPATCHER, DISPATCHER_PIN, "candidate dispatcher"), (FRONTEND, FRONTEND_PIN, "TFG frontend"),
        (BOOTSTRAP, BOOTSTRAP_PIN, "strict bootstrap"), (CONTROLS_RUNNER, CONTROLS_RUNNER_PIN, "controls parent runner"))
    for path, pin, label in pins:
        pin_file(path, pin, label, 8 * 1024 * 1024)
    plan_raw, rows_raw, manifest_raw, binding_raw, plan, binding, rows = verify_plan_inputs()
    receipt_path = Path(args.launch_receipt_path)
    receipt, receipt_raw = load_launch_receipt(receipt_path, args.launch_receipt_bytes,
        args.launch_receipt_raw_sha256, plan_raw, rows_raw, manifest_raw, binding_raw)
    if (HERE / "observer.lock").exists():
        raise FileExistsError("observer lock already exists; never resume/replay an existing observer")
    HERE.mkdir(parents=True, exist_ok=True)
    create_only(HERE / "observer.lock", canonical({"pid": os.getpid(), "started_at_utc": utc_now(),
        "run_id": CANDIDATE_RUN_ID, "request_id": receipt["request_id"], "pid_candidate": receipt["pid"],
        "start_ticks": receipt["start_ticks"], "planned_rows": 180}) + b"\n")
    POLL_ROOT.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_ROOT.mkdir(parents=True, exist_ok=True)
    if not bootstrap_remote_worker():
        return 2
    while True:
        time.sleep(INTERVAL_SECONDS)
        clean, reason = poll_once(receipt, receipt_raw, rows, binding)
        if not clean:
            record_stop(reason, {"last_poll_directory": str(max(POLL_ROOT.iterdir(), key=lambda p: p.name))})
            return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        record_stop("OBSERVER_INTERRUPTED_BY_OPERATOR")
        raise SystemExit(130)
    except Exception as exc:
        record_stop("OBSERVER_SOURCE_OR_INPUT_ERROR", {"exception_type": type(exc).__name__, "error": str(exc)[:800]})
        raise SystemExit(2)
