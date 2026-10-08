#!/usr/bin/env python3
"""Create (but never dispatch) required-input-only Dev360/candidate180 request packets.

The source has two one-way phases. ``prepare-launch`` requires the actual
terminal controls inventory, six explicitly supplied reconstructed controls
artifacts, a source-observer terminal poll, and the actual candidate metadata
preflight/readback bundle (including its explicitly pinned request index). It
measures all dynamic byte counts and SHA-256 values from those files, then
writes ROOT-dispatcher JSON requests with exclusive creates.
``prepare-status`` requires the actual downloaded candidate launch receipt and
its download response before it can write one read-only status request. Neither
command invokes SSH, an API, a GPU, a model, or a dispatcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Any


ROOT_REMOTE = "/root/tts-research-system-20261001"
JOB_CONTROL = ROOT_REMOTE + "/cross_tfg_audio_head_20261003/job_control"
JOB = JOB_CONTROL + "/dev360_candidate180_detached_attempt08"
SOURCE_DIR = JOB_CONTROL + "/dev360_candidate180_source_packet_attempt02"
PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
POLICY_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
CONTROLS_RUN_ID = "cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10"
CANDIDATE_RUN_ID = "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt02"
CONTROLS_PLAN_SHA256 = "8a592cc54dc981bec5e2398108e6873190a54c7c1ed2bb7901cd31aad6cc86eb"
CONTROLS_PLAN_BYTES = 1781735
CONTROLS_ROWS_SHA256 = "45f1d3130c33c37a513db7b22f76d4107bd821f370a7b3ae2d5ae853894b27c8"
CONTROLS_ROWS_BYTES = 878184
CONTROLS_PLAN_MANIFEST_SHA256 = "ab07ee9bed69d067390b9ffae597200916838f8cca6bdb7ddf376b5d5febc214"
CONTROLS_PLAN_MANIFEST_BYTES = 1438
CONTROLS_LAUNCH_RECEIPT_SHA256 = "cd5e1fa2c37d615bb3427947fa1e9a2948af56dced57c5d3d1b6e10b404a02b5"
CONTROLS_LAUNCH_RECEIPT_BYTES = 5899
CONTROLS_LAUNCH_RECEIPT_LOCAL = ".codex/scratch/dev360_detached_status_attempt08/receipt_readback_attempt01/controls360_codex_controls360_attempt10_launch01.launch_receipt.json"
CONTROLS_LAUNCH_RECEIPT_REMOTE = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/dev360_phase2_recovery_attempt09/output/controls360_codex_controls360_attempt10_launch01.launch_receipt.json"
CONTROLS_INVENTORY_REMOTE = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/dev360_terminal_inventory_attempt02/output/terminal_inventory_attempt02.json"
RUNTIME_BINDING_SHA256 = "dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5"
RUNTIME_BINDING_BYTES = 568289
CANDIDATE_PLAN_SHA256 = "8615754443e653fc79203f108fa689205a77e7f6f234930b352204df08b98280"
CANDIDATE_PLAN_BYTES = 932236
CANDIDATE_ROWS_SHA256 = "c9dc6f9bf400aff7f2c25ee24528dfd3a038433c21e80e0177066b8ab3f8a19e"
CANDIDATE_ROWS_BYTES = 606042
CANDIDATE_MANIFEST_SHA256 = "d25985b666c59946a7acd1c948b834ea8628d0b9c8eef0442419818ea86c0192"
CANDIDATE_MANIFEST_BYTES = 2513
CANDIDATE_MANIFEST_SELF_SHA256 = "420d73cd5fb95692ded1c8bbd8cd82363ccb40169a064f21121be70f041909e5"
STRICT_BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
STRICT_BOOTSTRAP_BYTES = 9525
STRICT09_SHA256, STRICT09_BYTES = "4053e74430e900037ca2f528e3678ad4375241e8e9c653367ea3de31ca94c86d", 287148
STRICT12_SHA256, STRICT12_BYTES = "c55b390b6d5189983a4a3e2d181d5ff7606cc191817f125a859800c3e09508d9", 146401
EXPECTED_MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
EXPECTED_METHODS = ("identity", "rms_fixed")

LOCAL_FIXED = {
    "controls_plan": (".codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/controls_plan_attempt09.json", CONTROLS_PLAN_BYTES, CONTROLS_PLAN_SHA256),
    "controls_rows": (".codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/controls_planned_rows_attempt09.jsonl", CONTROLS_ROWS_BYTES, CONTROLS_ROWS_SHA256),
    "controls_plan_manifest": (".codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/controls_plan_manifest_attempt09.json", CONTROLS_PLAN_MANIFEST_BYTES, CONTROLS_PLAN_MANIFEST_SHA256),
    "runtime_binding": (".codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/runtime_binding_attempt09.json", RUNTIME_BINDING_BYTES, RUNTIME_BINDING_SHA256),
    "candidate_plan": (".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/candidate_rebound_plan_attempt13.json", CANDIDATE_PLAN_BYTES, CANDIDATE_PLAN_SHA256),
    "candidate_rows": (".codex/scratch/dev360_candidate180_cpu_prepare_attempt01/downloads_attempt02/candidate_rebound_rows_attempt12.jsonl", CANDIDATE_ROWS_BYTES, CANDIDATE_ROWS_SHA256),
    "candidate_manifest14": (".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/plan_manifest_metadata_repair_attempt14.json", CANDIDATE_MANIFEST_BYTES, CANDIDATE_MANIFEST_SHA256),
}

# Metadata-index identity is already sealed; its future q00/q01/q02 outputs stay required runtime inputs.
METADATA_PREFLIGHT_INDEX_REL = ".codex/scratch/dev360_candidate13_metadata_preflight_attempt04/request_index_attempt01.json"
METADATA_PREFLIGHT_INDEX_BYTES = 8115
METADATA_PREFLIGHT_INDEX_SHA256 = "96938177e8eb3593fd67d535e78cd71d64539bc37b07d5b62a5143cbce68a2ec"
CANDIDATE_PLAN_REMOTE = ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cpu_prepare_attempt01/rebound_plan_attempt12/candidate_rebound_plan_attempt13.json"
CANDIDATE_ROWS_REMOTE = ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cpu_prepare_attempt01/rebound_plan_attempt12/planned_rows.jsonl"
CANDIDATE_MANIFEST_REMOTE = ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cpu_prepare_attempt01/rebound_plan_attempt12/plan_manifest_metadata_repair_attempt14.json"
METADATA_PREFLIGHT_CONTROL_ROOT = JOB_CONTROL + "/dev360_candidate13_metadata_preflight_attempt04"
CANDIDATE_OUTPUT_ROOT = "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/candidate180"

CONTROL_ARTIFACTS = {
    "result_manifest": ("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360/result_manifest.json", "--result-manifest"),
    "results_ledger": ("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360/results.jsonl", "--results-ledger"),
    "run_status": ("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360/control/run_status.json", "--run-status"),
    "execution_record": ("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360.execution.json", "--execution-record"),
    "strict_attestation": ("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360/runtime_determinism_attestation.json", "--strict-attestation"),
    "repeat_qc_ledger": ("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360/generation_repeat_qc.jsonl", "--repeat-qc-ledger"),
}

INVENTORY_ARTIFACTS = {
    "result_manifest": ("result_manifest", CONTROL_ARTIFACTS["result_manifest"][0]),
    "results_ledger": ("results_ledger", CONTROL_ARTIFACTS["results_ledger"][0]),
    "run_status": ("run_status", CONTROL_ARTIFACTS["run_status"][0]),
    "execution_record": ("execution_record", CONTROL_ARTIFACTS["execution_record"][0]),
    "strict_attestation": ("strict_determinism_attestation", CONTROL_ARTIFACTS["strict_attestation"][0]),
    "repeat_qc_ledger": ("generation_repeat_qc_ledger", CONTROL_ARTIFACTS["repeat_qc_ledger"][0]),
}

INVENTORY_EXPECTED_SOURCE_PINS = {
    "runtime_binding": ("/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json", RUNTIME_BINDING_BYTES, RUNTIME_BINDING_SHA256),
    "plan": ("/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/plan.json", CONTROLS_PLAN_BYTES, CONTROLS_PLAN_SHA256),
    "planned_rows": ("/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/planned_rows.jsonl", CONTROLS_ROWS_BYTES, CONTROLS_ROWS_SHA256),
    "plan_manifest": ("/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/plan_manifest.json", CONTROLS_PLAN_MANIFEST_BYTES, CONTROLS_PLAN_MANIFEST_SHA256),
    "launch_receipt": (CONTROLS_LAUNCH_RECEIPT_REMOTE, CONTROLS_LAUNCH_RECEIPT_BYTES, CONTROLS_LAUNCH_RECEIPT_SHA256),
}

INVENTORY_SOURCE_PINS = {
    "inventory_worker02": (".codex/scratch/dev360_terminal_inventory_attempt02/terminal_inventory_and_chunk_planner_attempt02.py", 34744, "b03722b3039da1ff8f8f10cf07ce878f2615d4e2f087062faf192876e1ff4652"),
    "inventory_dispatch_index03": (".codex/scratch/dev360_terminal_inventory_attempt02/dispatch_index_attempt03.json", 12288, "9ea50ae0b7c03c68ab8db4dd4f62afc7822782f284e76074f830514bc4bb0e2b"),
    "inventory_packet_index02": (".codex/scratch/dev360_terminal_inventory_attempt02/terminal_inventory_packet_index_attempt02.json", 11760, "bcf008b6c600bd0a156cc8c1b8a4f28f90f3f1d672168cb44438971830053fb2"),
    "inventory_q00": (".codex/scratch/dev360_terminal_inventory_attempt02/requests_attempt02/q00_create_namespace.json", 486, "6e6b1cdef0b16b47c4bd2f1c854bba52f2295c22c0b7ee8314b85b6bef5df36f"),
    "inventory_q01": (".codex/scratch/dev360_terminal_inventory_attempt02/requests_attempt02/q01_upload_inventory_worker.json", 483, "8392d75ff829f514cb6b8b66f173e0ddc679c2d33d43c288ff92126f54c7f187"),
    "inventory_q02": (".codex/scratch/dev360_terminal_inventory_attempt02/requests_attempt02/q02_read_terminal_file_inventory.json", 361, "dd9760f48fe325213dca43a15866293d6fc88fc88bf64f8eb28aca286d770395"),
}

SOURCE_PINS = {
    "launcher08": (".codex/scratch/dev360_candidate180_detached_launch_attempt08/launch_candidate180_detached_attempt08.py", 49116, "7703866b8c7b73f1288fdf54c5b4028fc3b6268fd175366491ce18f2664e9d65", "launch_candidate180_detached_attempt08.py"),
    "status08": (".codex/scratch/dev360_candidate180_detached_launch_attempt08/read_candidate180_detached_status_attempt08.py", 46057, "b032ff16cbbee7356377f252e8785bf2e46db0ebfe828a0430f45070ee5f053d", "read_candidate180_detached_status_attempt08.py"),
}

OBSERVER_SOURCE_PINS = {
    "observer": (".codex/scratch/dev360_status_observer_attempt03/dev360_status_observer_attempt03.py", 22888, "1f61ee8027c0a5c3f11f66c18201acb846ffabb2974ea16a4b6f613a0e22d118"),
    "bridge_client": (".codex/scratch/client.py", 2706, "d0859d2e76cb59d1d4ec7c7a6e0c391e8ff28eababe5ab2058fb158f4866066f"),
    "status_worker": (".codex/scratch/dev360_detached_status_attempt08/read_dev360_detached_status_attempt08.py", 21199, "533dd688b752cd976882c374b478f5816cadbcce960fc870976d345f2f14a7f8"),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def self_hash_ok(value: dict[str, Any], field: str) -> bool:
    return value.get(field) == sha(canonical({key: item for key, item in value.items() if key != field}))


def read_regular(path: Path, label: str, limit: int) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > limit:
        raise ValueError(f"{label}: expected bounded regular non-symlink file: {path}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError(f"{label}: size changed while reading: {path}")
    return raw


def json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected JSON object")
    return value


def read_json(path: Path, label: str, limit: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label, limit)
    return json_bytes(raw, label), raw


def absolute(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve(strict=True)


def root_from_script() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "scripts/experiments/cross_tfg_audio_head_20261003").is_dir():
        raise RuntimeError("builder is not under the sealed tts-exp workspace")
    return root


def required_file(path_text: str, label: str, root: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required {label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"required {label} must be a regular non-symlink file: {path}")
    parent = path.parent
    while True:
        if parent.is_symlink():
            raise ValueError(f"required {label} has a symlink parent: {parent}")
        if parent.parent == parent:
            break
        parent = parent.parent
    return path.resolve(strict=True)


def fixed_input(root: Path, role: str) -> tuple[Path, bytes]:
    rel, expected_bytes, expected_sha = LOCAL_FIXED[role]
    path = root / rel
    raw = read_regular(path, role, max(expected_bytes + 1, 2 * 1024 * 1024))
    if len(raw) != expected_bytes or sha(raw) != expected_sha:
        raise ValueError(f"{role} differs from its fixed local raw pin")
    return path.resolve(strict=True), raw


def parse_jsonl(raw: bytes, label: str, expected: int) -> list[dict[str, Any]]:
    if not raw.endswith(b"\n"):
        raise ValueError(f"{label} lacks final newline")
    lines = raw.splitlines()
    if len(lines) != expected:
        raise ValueError(f"{label} row count differs from required {expected}")
    rows = [json_bytes(line, f"{label} line {i + 1}") for i, line in enumerate(lines)]
    if b"".join(canonical(row) + b"\n" for row in rows) != raw:
        raise ValueError(f"{label} is not exact canonical JSONL")
    return rows


def payload_sha(value: dict[str, Any], field: str) -> str:
    return sha(canonical({key: item for key, item in value.items() if key != field}))


def flag_value(argv: Any, flag: str) -> str:
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError("controls execution_command is not a string argv array")
    indices = [i for i, item in enumerate(argv) if item == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError(f"controls execution_command must contain exactly one {flag}")
    return argv[indices[0] + 1]


def validate_fixed_sources(root: Path) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    for role, (rel, expected_bytes, expected_sha, remote_name) in SOURCE_PINS.items():
        path = root / rel
        raw = read_regular(path, role, expected_bytes + 1)
        if len(raw) != expected_bytes or sha(raw) != expected_sha:
            raise ValueError(f"sealed source {role} changed")
        pins[role] = {"local": str(path.resolve(strict=True)), "remote": SOURCE_DIR + "/" + remote_name,
                      "bytes": len(raw), "raw_sha256": sha(raw)}
    for role, (rel, expected_bytes, expected_sha) in OBSERVER_SOURCE_PINS.items():
        raw = read_regular(root / rel, role, expected_bytes + 1)
        if len(raw) != expected_bytes or sha(raw) != expected_sha:
            raise ValueError(f"observer source pin {role} changed")
        pins["observer_source:" + role] = {"local": str((root / rel).resolve(strict=True)),
                                            "bytes": len(raw), "raw_sha256": sha(raw)}
    for role, (rel, expected_bytes, expected_sha) in INVENTORY_SOURCE_PINS.items():
        path = root / rel
        raw = read_regular(path, role, expected_bytes + 1)
        if len(raw) != expected_bytes or sha(raw) != expected_sha:
            raise ValueError(f"terminal inventory source pin {role} changed")
        pins["inventory_source:" + role] = {"local": str(path.resolve(strict=True)),
                                             "bytes": len(raw), "raw_sha256": sha(raw)}
    return pins


def validate_observer(root: Path, manifest_path: Path, observation_path: Path,
                      response_path: Path, launch_identity: dict[str, Any]) -> dict[str, Any]:
    manifest, manifest_raw = read_json(manifest_path, "source observer manifest", 2 * 1024 * 1024)
    observation, observation_raw = read_json(observation_path, "source observer terminal observation", 2 * 1024 * 1024)
    envelope, envelope_raw = read_json(response_path, "source observer terminal response", 20 * 1024 * 1024)
    if manifest.get("schema") != "cross_tfg_dev360_status_observer_manifest_v1" or not self_hash_ok(manifest, "manifest_sha256"):
        raise ValueError("observer manifest schema/self-hash mismatch")
    listed = {Path(item.get("path", "")).name: item for item in manifest.get("files", []) if isinstance(item, dict)}
    for role, (_, size, digest) in OBSERVER_SOURCE_PINS.items():
        basename = Path(OBSERVER_SOURCE_PINS[role][0]).name
        item = listed.get(basename)
        if not isinstance(item, dict) or item.get("bytes") != size or item.get("sha256") != digest:
            raise ValueError(f"observer manifest omits/mismatches pinned {role} source")
    clean_trunc = envelope.get("truncated")
    if (envelope.get("ok") is not True or type(envelope.get("code")) is not int or envelope.get("code") != 0
            or envelope.get("timed_out") is not False or envelope.get("incomplete_stream") is not False
            or envelope.get("stream_error") not in (None, "")
            or not isinstance(clean_trunc, dict) or clean_trunc.get("stdout") is not False
            or clean_trunc.get("stderr") is not False or not isinstance(envelope.get("stdout"), str)
            or not envelope["stdout"] or envelope.get("stderr") != ""):
        raise ValueError("observer terminal response is not a clean complete bridge envelope")
    snapshot = json_bytes(envelope["stdout"].encode("utf-8"), "observer terminal snapshot")
    if (snapshot.get("schema") != "cross_tfg_dev360_detached_status_snapshot_v1"
            or snapshot.get("status") != "TERMINAL_OR_IDENTITY_MISMATCH" or not self_hash_ok(snapshot, "snapshot_sha256")
            or snapshot.get("stage_id") != "controls360" or snapshot.get("run_id") != CONTROLS_RUN_ID
            or snapshot.get("pid") != launch_identity["launch_pid"]
            or snapshot.get("start_ticks") != launch_identity["launch_start_ticks"]
            or snapshot.get("plan_raw_sha256") != CONTROLS_PLAN_SHA256
            or snapshot.get("runtime_binding_raw_sha256") != RUNTIME_BINDING_SHA256):
        raise ValueError("observer did not return a terminal snapshot for the pinned controls process")
    if (observation.get("schema") != "cross_tfg_dev360_status_observer_poll_v1"
            or observation.get("status") != "STOPPED_ROOT_REVIEW_REQUIRED"
            or observation.get("validation_reason") != "STATUS_TERMINAL_OR_NONRUNNING"
            or observation.get("automatic_experiment_pass") is not False
            or observation.get("root_consumption_required") is not True
            or observation.get("run_id") != CONTROLS_RUN_ID
            or observation.get("pid") != launch_identity["launch_pid"]
            or observation.get("start_ticks") != launch_identity["launch_start_ticks"]
            or type(observation.get("client_exit_code")) is not int or observation.get("client_exit_code") != 0
            or type(observation.get("client_stderr_bytes")) is not int or observation.get("client_stderr_bytes") != 0
            or observation.get("response_truncated_at_cap") is not False
            or observation.get("response_raw_sha256") != sha(envelope_raw)
            or not self_hash_ok(observation, "observation_sha256")):
        raise ValueError("observer terminal observation does not bind the clean terminal response")
    summary = observation.get("detail", {}).get("snapshot", {})
    if (not isinstance(summary, dict) or summary.get("status") != snapshot.get("status")
            or summary.get("schema") != snapshot.get("schema")
            or summary.get("run_id") != snapshot.get("run_id")
            or summary.get("stage_id") != snapshot.get("stage_id")
            or summary.get("pid") != snapshot.get("pid")
            or summary.get("start_ticks") != snapshot.get("start_ticks")
            or summary.get("plan_raw_sha256") != snapshot.get("plan_raw_sha256")
            or summary.get("runtime_binding_raw_sha256") != snapshot.get("runtime_binding_raw_sha256")):
        raise ValueError("observer observation summary differs from returned terminal snapshot")
    process = snapshot.get("process")
    process_state = process.get("state") if isinstance(process, dict) else None
    if (process_state == "EXITED_OR_NOT_FOUND"):
        process_terminal = process.get("observed_start_ticks") is None and process.get("matches_launch") is False
    elif process_state == "EXITED_ZOMBIE":
        process_terminal = process.get("observed_start_ticks") == launch_identity["launch_start_ticks"] and process.get("matches_launch") is True
    else:
        process_terminal = False
    # The observer response stdout is the complete bounded status snapshot.
    # observation.detail.snapshot is a deliberately reduced projection without
    # run_status; use the validated complete snapshot for terminal state gates.
    run_snapshot = snapshot.get("run_status", {})
    manifest_snapshot = snapshot.get("result_manifest", {})
    execution_snapshot = snapshot.get("execution_record", {})
    if (not process_terminal or run_snapshot.get("status") != "COMPLETE"
            or run_snapshot.get("completed_rows") != 360 or run_snapshot.get("planned_rows") != 360
            or manifest_snapshot.get("status") != "PASS" or manifest_snapshot.get("planned_rows") != 360
            or manifest_snapshot.get("result_rows") != 360
            or execution_snapshot.get("status") != "CHILD_EXITED"
            or execution_snapshot.get("child_returncode") != 0):
        raise ValueError("observer response does not show the pinned process terminal state")
    return {"manifest": {"path": str(manifest_path), "bytes": len(manifest_raw), "raw_sha256": sha(manifest_raw)},
            "observation": {"path": str(observation_path), "bytes": len(observation_raw), "raw_sha256": sha(observation_raw)},
            "response": {"path": str(response_path), "bytes": len(envelope_raw), "raw_sha256": sha(envelope_raw)},
            "snapshot_status": snapshot.get("status"), "pid": snapshot["pid"], "start_ticks": snapshot["start_ticks"]}


def validate_controls(root: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    plan_path, plan_raw = fixed_input(root, "controls_plan")
    rows_path, rows_raw = fixed_input(root, "controls_rows")
    plan_manifest_path, plan_manifest_raw = fixed_input(root, "controls_plan_manifest")
    binding_path, binding_raw = fixed_input(root, "runtime_binding")
    candidate_plan_path, candidate_plan_raw = fixed_input(root, "candidate_plan")
    candidate_rows_path, candidate_rows_raw = fixed_input(root, "candidate_rows")
    candidate_manifest_path, candidate_manifest_raw = fixed_input(root, "candidate_manifest14")
    control_plan = json_bytes(plan_raw, "controls frozen plan")
    planned_rows = parse_jsonl(rows_raw, "controls frozen ordered rows", 360)
    plan_manifest = json_bytes(plan_manifest_raw, "controls frozen plan manifest")
    binding = json_bytes(binding_raw, "runtime binding")
    candidate_plan = json_bytes(candidate_plan_raw, "candidate plan")
    candidate_manifest = json_bytes(candidate_manifest_raw, "candidate manifest13")
    candidate_rows = parse_jsonl(candidate_rows_raw, "candidate frozen ordered rows", 180)
    if (control_plan.get("schema_version") != "cross-tfg-audio-eval-v1" or control_plan.get("status") != "FROZEN"
            or control_plan.get("run_id") != CONTROLS_RUN_ID or control_plan.get("plan_sha256") != payload_sha(control_plan, "plan_sha256")
            or control_plan.get("rows") != planned_rows or plan_manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_plan_v1"
            or plan_manifest.get("status") != "FROZEN" or plan_manifest.get("run_id") != CONTROLS_RUN_ID
            or plan_manifest.get("plan_raw_sha256") != sha(plan_raw) or plan_manifest.get("planned_rows_sha256") != sha(rows_raw)
            or plan_manifest.get("planned_rows") != 360 or not self_hash_ok(plan_manifest, "manifest_sha256")):
        raise ValueError("frozen controls plan/rows/manifest source pins do not close")
    if (tuple(item.get("model_id") for item in control_plan.get("models", [])) != EXPECTED_MODELS
            or tuple(item.get("method_id") for item in control_plan.get("methods", [])) != EXPECTED_METHODS):
        raise ValueError("controls model/method order differs from the frozen candidate contract")
    if (binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1" or binding.get("status") != "PASS"
            or binding.get("gpu_uuid") != GPU_UUID or binding.get("determinism_policy_sha256") != POLICY_SHA256
            or "cuda_visible_devices" in binding
            or binding.get("execution_bindings", {}).get("output_root") != "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10"
            or binding.get("confirm_audio_read") is not False or binding.get("confirm_effects_read") is not False):
        raise ValueError("runtime binding GPU/policy/output-root/closed-confirm pins do not match")
    if (candidate_plan.get("schema_version") != "cross-tfg-audio-eval-v1" or candidate_plan.get("status") != "FROZEN"
            or candidate_plan.get("run_id") != CANDIDATE_RUN_ID or candidate_plan.get("plan_sha256") != payload_sha(candidate_plan, "plan_sha256")
            or candidate_plan.get("plan_sha256") != "fee2718c0c4b18e5acabf3ca7d7103344bfb3859f3d8a3b93862c1ad4ce7d612"
            or candidate_plan.get("rows") != candidate_rows or len(candidate_rows) != 180
            or candidate_manifest.get("schema") != "cross_tfg_v3_candidate_only_dev_plan_v1"
            or candidate_manifest.get("status") != "FROZEN" or candidate_manifest.get("run_id") != CANDIDATE_RUN_ID
            or candidate_manifest.get("plan_raw_sha256") != sha(candidate_plan_raw)
            or candidate_manifest.get("planned_rows_sha256") != sha(candidate_rows_raw)
            or candidate_manifest.get("planned_rows") != 180 or not self_hash_ok(candidate_manifest, "manifest_sha256")
            or candidate_manifest.get("manifest_sha256") != CANDIDATE_MANIFEST_SELF_SHA256
            or candidate_manifest.get("fixed_optimizer_steps") != 120
            or candidate_manifest.get("confirm_audio_read") is not False
            or candidate_manifest.get("confirm_effects_read") is not False
            or candidate_manifest.get("candidate_runtime_qualification_performed") is not False
            or candidate_manifest.get("model_counts") != {model: 60 for model in EXPECTED_MODELS}):
        raise ValueError("candidate plan/rows/metadata-only FROZEN manifest13 binding differs")
    candidate_execution = candidate_plan.get("execution", {})
    candidate_method = candidate_plan.get("methods", [])
    candidate_models = candidate_plan.get("models", [])
    if (candidate_execution.get("samehost_runtime_binding", {}).get("raw_sha256") != RUNTIME_BINDING_SHA256
            or candidate_execution.get("candidate_only") is not True
            or candidate_execution.get("fixed_optimizer_steps") != 120
            or candidate_execution.get("confirm_audio_read") is not False
            or candidate_execution.get("confirm_effects_read") is not False
            or len(candidate_method) != 1 or candidate_method[0].get("method_id") != "shared_recipient_transition_head_v3_fit40_step120"
            or tuple(item.get("model_id") for item in candidate_models) != EXPECTED_MODELS
            or any(row.get("split") != "dev" or row.get("method_id") != "shared_recipient_transition_head_v3_fit40_step120" for row in candidate_rows)
            or {model: sum(row.get("model_id") == model for row in candidate_rows) for model in EXPECTED_MODELS}
               != {model: 60 for model in EXPECTED_MODELS}):
        raise ValueError("candidate plan does not bind the pinned runtime binding")

    args_by_role = {role: Path(getattr(args, role)).resolve(strict=True)
                    for role, attr in ((role, option) for role, (_, option) in CONTROL_ARTIFACTS.items())}
    raw_by_role: dict[str, bytes] = {}
    for role, path in args_by_role.items():
        raw_by_role[role] = read_regular(path, role, 128 * 1024 * 1024 if role == "results_ledger" else 32 * 1024 * 1024)
    results = parse_jsonl(raw_by_role["results_ledger"], "actual controls results ledger", 360)
    result_manifest = json_bytes(raw_by_role["result_manifest"], "actual controls result manifest")
    run_status = json_bytes(raw_by_role["run_status"], "actual controls run status")
    execution = json_bytes(raw_by_role["execution_record"], "actual controls execution record")
    attestation = json_bytes(raw_by_role["strict_attestation"], "actual controls strict attestation")
    expected_ids = [row.get("row_id") for row in planned_rows]
    statuses = {key: sum(row.get("status") == key for row in results) for key in ("PASS", "FAILED", "INCOMPLETE")}
    models = {model: sum(row.get("model_id") == model for row in results) for model in EXPECTED_MODELS}
    cells = {f"{model}|{method}": sum(row.get("model_id") == model and row.get("method_id") == method for row in results)
             for model in EXPECTED_MODELS for method in EXPECTED_METHODS}
    manifest_cells = [{"model_id": model, "method_id": method, "planned_rows": 60, "result_rows": 60,
                       "status_counts": {"PASS": 60, "FAILED": 0, "INCOMPLETE": 0}}
                      for model in EXPECTED_MODELS for method in EXPECTED_METHODS]
    if ([row.get("row_id") for row in results] != expected_ids or len(set(expected_ids)) != 360
            or any(row.get("split") != "dev" or row.get("status") != "PASS" for row in results)
            or any(any(row.get(key) != planned.get(key) for key in ("row_id", "sample_id", "speaker_id", "model_id", "seed", "method_id"))
                   for planned, row in zip(planned_rows, results))
            or statuses != {"PASS": 360, "FAILED": 0, "INCOMPLETE": 0}
            or models != {model: 120 for model in EXPECTED_MODELS}
            or cells != {f"{model}|{method}": 60 for model in EXPECTED_MODELS for method in EXPECTED_METHODS}):
        raise ValueError("actual controls rows are not the ordered 360/6x60 all-PASS ledger")
    repeat_raw = raw_by_role["repeat_qc_ledger"]
    if repeat_raw != b"":
        raise ValueError("frozen controls generation_repeat_qc is []: actual core06 output must be empty, NOT literal PASS")
    if (result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or result_manifest.get("run_id") != CONTROLS_RUN_ID
            or result_manifest.get("status") != "PASS" or result_manifest.get("planned_rows") != 360
            or result_manifest.get("result_rows") != 360 or result_manifest.get("plan_raw_sha256") != sha(plan_raw)
            or result_manifest.get("results_jsonl_sha256") != sha(raw_by_role["results_ledger"])
            or result_manifest.get("status_counts") != statuses or result_manifest.get("counts_by_model_method") != manifest_cells
            or result_manifest.get("failure_reasons") != []
            or result_manifest.get("repeat_qc_status") != "NOT_RUN_SUBRUN" or result_manifest.get("generation_repeat_qc_rows") != 0
            or result_manifest.get("generation_repeat_qc_jsonl_sha256") != sha(b"")
            or not self_hash_ok(result_manifest, "manifest_sha256")):
        raise ValueError("actual result manifest does not bind the ordered results and NOT_RUN_SUBRUN contract")
    expected_control_root = "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360"
    plan_remote = "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/plan.json"
    rows_remote = "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/planned_rows.jsonl"
    manifest_remote = "/root/tts-research-system-20261001/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/plan_manifest.json"
    if (run_status.get("schema_version") != "cross_tfg_audio_head_stagea_v1" or run_status.get("status") != "COMPLETE"
            or run_status.get("run_id") != CONTROLS_RUN_ID or run_status.get("plan_path") != plan_remote
            or run_status.get("plan_raw_sha256") != sha(plan_raw) or run_status.get("planned_rows_path") != rows_remote
            or run_status.get("planned_rows_sha256") != sha(rows_raw) or run_status.get("planned_rows") != 360
            or run_status.get("completed_rows") != 360 or run_status.get("status_counts") != statuses
            or run_status.get("repeat_qc_status") != "NOT_RUN_SUBRUN"
            or run_status.get("result_manifest_path") != expected_control_root + "/result_manifest.json"
            or run_status.get("result_manifest_sha256") != sha(raw_by_role["result_manifest"])
            or not isinstance(run_status.get("ended_unix"), (int, float))):
        raise ValueError("run_status is not the actual COMPLETE 360-row producer record")
    command = execution.get("execution_command")
    if (execution.get("schema") != "cross_tfg_v3_samehost_dev360_preflight_v1"
            or execution.get("status") != "PASS_METADATA_ONLY"
            or execution.get("plan_raw_sha256") != sha(plan_raw)
            or execution.get("plan_manifest_raw_sha256") != sha(plan_manifest_raw)
            or execution.get("runtime_binding_raw_sha256") != sha(binding_raw)
            or execution.get("execution_output_root") != expected_control_root
            or execution.get("child_returncode") != 0
            or execution.get("confirm_audio_opened") is not False or execution.get("development_audio_opened") is not False
            or execution.get("generated_media_opened") is not False or execution.get("model_execution_performed") is not False
            or control_plan.get("execution", {}).get("confirm_audio_read") is not False
            or control_plan.get("execution", {}).get("confirm_effects_read") is not False
            or flag_value(command, "--plan") != plan_remote or flag_value(command, "--planned-rows") != rows_remote
            or flag_value(command, "--expected-plan-raw-sha256") != sha(plan_raw)
            or flag_value(command, "--output-root") != expected_control_root):
        raise ValueError("actual execution record lacks PASS_METADATA_ONLY/child rc0 or producer argv lineage")
    started, finished = execution.get("execution_started_unix"), execution.get("execution_finished_unix")
    ended = run_status.get("ended_unix")
    if (not isinstance(started, (int, float)) or not isinstance(finished, (int, float))
            or not isinstance(ended, (int, float)) or finished < started or ended < started or finished < ended):
        raise ValueError("controls execution/run terminal times are missing or inconsistent")
    attestation_ref = execution.get("strict_determinism_attestation")
    if (not isinstance(attestation_ref, dict) or attestation_ref.get("path") != expected_control_root + "/runtime_determinism_attestation.json"
            or attestation_ref.get("bytes") != len(raw_by_role["strict_attestation"])
            or attestation_ref.get("raw_sha256") != sha(raw_by_role["strict_attestation"])
            or attestation_ref.get("attestation_sha256") != attestation.get("attestation_sha256")
            or attestation_ref.get("status") != "PASS"):
        raise ValueError("execution record strict attestation ref does not bind actual terminal file")
    if (attestation.get("schema") != "cross_tfg_v3_samehost_renderer_determinism_attestation_v1"
            or attestation.get("status") != "PASS" or attestation.get("gpu_uuid") != GPU_UUID
            or attestation.get("plan_raw_sha256") != sha(plan_raw)
            or attestation.get("runtime_binding_raw_sha256") != sha(binding_raw)
            or attestation.get("strict_bootstrap_source_sha256") != STRICT_BOOTSTRAP_SHA256
            or attestation.get("qualified_policy_reference_sha256") != POLICY_SHA256
            or attestation.get("policy_values") != binding.get("strict_determinism_policy")
            or attestation.get("row_count") != 360 or attestation.get("native_renderer_row_receipt_count") != 360
            or attestation.get("process_receipt_count") != 360
            or attestation.get("confirm_audio_read") is not False or attestation.get("confirm_data_opened") is not False
            or len(attestation.get("native_renderer_receipts_by_row", {})) != 360
            or len(attestation.get("process_receipts", [])) != 360
            or attestation.get("per_backend_repeat_qualification_receipts") != binding.get("backend_determinism_qualifications")
            or not self_hash_ok(attestation, "attestation_sha256")):
        raise ValueError("strict native renderer attestation is not the actual 360-row PASS receipt")
    native_by_row = attestation["native_renderer_receipts_by_row"]
    process_by_row = {item.get("row_id"): item for item in attestation["process_receipts"] if isinstance(item, dict)}
    if set(native_by_row) != set(expected_ids) or set(process_by_row) != set(expected_ids):
        raise ValueError("strict attestation row receipts do not cover the exact ordered controls row IDs")
    expected_native_entries = {}
    for model_id in EXPECTED_MODELS:
        backend = binding.get("backends", {}).get(model_id, {})
        generator = backend.get("generator", {}) if isinstance(backend, dict) else {}
        entry_field = "native_entry_path" if model_id == EXPECTED_MODELS[0] else "render_entry_path"
        native_entry = generator.get(entry_field) if isinstance(generator, dict) else None
        if not isinstance(native_entry, str) or not native_entry.startswith("/"):
            raise ValueError(f"runtime binding has no absolute native renderer entry for {model_id}")
        expected_native_entries[model_id] = native_entry
    strict_receipt_dir = expected_control_root + "/runtime_determinism_receipts"
    for planned in planned_rows:
        row_id, model_id = planned["row_id"], planned["model_id"]
        native, process = native_by_row[row_id], process_by_row[row_id]
        # The producer stores native bootstrap receipts in a row-ID keyed map;
        # each value is the compact {path,bytes,sha256,pid,entry_path} reference.
        # Row/model/backend identity therefore comes from the frozen map key and
        # plan, while the renderer entry is checked against the pinned binding.
        native_pid = native.get("pid") if isinstance(native, dict) else None
        process_pid = process.get("pid") if isinstance(process, dict) else None
        if (not isinstance(native, dict) or not isinstance(process, dict)
                or type(native_pid) is not int or native_pid <= 0
                or type(process_pid) is not int or process_pid <= 0
                or native.get("path") != f"{strict_receipt_dir}/{native_pid}.json"
                or process.get("path") != f"{strict_receipt_dir}/{process_pid}.json"
                or native.get("entry_path") != expected_native_entries[model_id]
                or process.get("row_id") != row_id or process.get("model_id") != model_id
                or process.get("entry_matches_backend") is not True
                or type(native.get("bytes")) is not int or native["bytes"] <= 0
                or type(process.get("bytes")) is not int or process["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(native.get("sha256", "")))
                or not re.fullmatch(r"[0-9a-f]{64}", str(process.get("sha256", "")))):
            raise ValueError(f"strict native/process attestation receipt is invalid for {row_id}")
    expected_qual = {EXPECTED_MODELS[0]: (STRICT12_BYTES, STRICT12_SHA256),
                     EXPECTED_MODELS[1]: (STRICT09_BYTES, STRICT09_SHA256),
                     EXPECTED_MODELS[2]: (STRICT09_BYTES, STRICT09_SHA256)}
    qual = binding.get("backend_determinism_qualifications", {})
    if set(qual) != set(EXPECTED_MODELS):
        raise ValueError("runtime binding lacks all three strict backend qualification entries")
    for model, (expected_bytes, expected_sha) in expected_qual.items():
        entry = qual.get(model, {})
        manifest = entry.get("lineage", {}).get("manifest", {}) if isinstance(entry, dict) else {}
        if (entry.get("status") != "PASS_EXACT_MATCH" or manifest.get("bytes") != expected_bytes
                or manifest.get("raw_sha256") != expected_sha):
            raise ValueError(f"runtime strict backend manifest pin mismatch for {model}")
    terminal_identity = {"run_id": CONTROLS_RUN_ID, "stage_id": "controls360",
                         "planned_rows": 360, "launch_pid": None, "launch_start_ticks": None}
    # PID/start ticks are filled from the independently observed launch receipt/index by caller below.
    return terminal_identity, {"run_id": CONTROLS_RUN_ID, "status_counts": statuses, "models": models, "cells": cells,
                               "controls_plan": {"path": str(plan_path), "bytes": len(plan_raw), "raw_sha256": sha(plan_raw)},
                               "controls_rows": {"path": str(rows_path), "bytes": len(rows_raw), "raw_sha256": sha(rows_raw)},
                               "controls_plan_manifest": {"path": str(plan_manifest_path), "bytes": len(plan_manifest_raw), "raw_sha256": sha(plan_manifest_raw)},
                               "runtime_binding": {"path": str(binding_path), "bytes": len(binding_raw), "raw_sha256": sha(binding_raw), "gpu_uuid": GPU_UUID, "policy_sha256": POLICY_SHA256},
                               "candidate_plan": {"path": str(candidate_plan_path), "bytes": len(candidate_plan_raw), "raw_sha256": sha(candidate_plan_raw)},
                               "candidate_rows": {"path": str(candidate_rows_path), "bytes": len(candidate_rows_raw), "raw_sha256": sha(candidate_rows_raw)},
                               "candidate_manifest14": {"path": str(candidate_manifest_path), "bytes": len(candidate_manifest_raw), "raw_sha256": sha(candidate_manifest_raw), "self_sha256": CANDIDATE_MANIFEST_SELF_SHA256},
                               "result_manifest": {"bytes": len(raw_by_role["result_manifest"]), "raw_sha256": sha(raw_by_role["result_manifest"])},
                               "results_ledger": {"bytes": len(raw_by_role["results_ledger"]), "raw_sha256": sha(raw_by_role["results_ledger"])},
                               "run_status": {"bytes": len(raw_by_role["run_status"]), "raw_sha256": sha(raw_by_role["run_status"])},
                               "execution_record": {"bytes": len(raw_by_role["execution_record"]), "raw_sha256": sha(raw_by_role["execution_record"]), "child_returncode": 0},
                               "strict_attestation": {"bytes": len(raw_by_role["strict_attestation"]), "raw_sha256": sha(raw_by_role["strict_attestation"]), "status": "PASS", "gpu_uuid": GPU_UUID, "policy_sha256": POLICY_SHA256},
                               "repeat_qc_ledger": {"bytes": 0, "raw_sha256": sha(b""), "status": "NOT_RUN_SUBRUN"},
                               "qualification_manifests": {model: digest for model, (_, digest) in {
                                   EXPECTED_MODELS[0]: (STRICT12_BYTES, STRICT12_SHA256),
                                   EXPECTED_MODELS[1]: (STRICT09_BYTES, STRICT09_SHA256),
                                   EXPECTED_MODELS[2]: (STRICT09_BYTES, STRICT09_SHA256),
                               }.items()}}, raw_by_role


def validate_terminal_inventory(path: Path, local_by_role: dict[str, Path],
                               expected_identity: dict[str, int]) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    inventory, raw = read_json(path, "actual controls terminal inventory", 8 * 1024 * 1024)
    if (inventory.get("schema") != "cross_tfg_dev360_terminal_file_inventory_v1"
            or inventory.get("status") != "TERMINAL_INVENTORY_READY"
            or not self_hash_ok(inventory, "inventory_sha256")
            or inventory.get("run_id") != CONTROLS_RUN_ID
            or inventory.get("stage_id") != "controls360"
            or inventory.get("plan_raw_sha256") != CONTROLS_PLAN_SHA256
            or inventory.get("runtime_binding_raw_sha256") != RUNTIME_BINDING_SHA256
            or inventory.get("physical_stage_leaf") != "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360"
            or inventory.get("execution_record_path") != CONTROL_ARTIFACTS["execution_record"][0]
            or inventory.get("all_rows_and_failures_preserved") is not True
            or inventory.get("automatic_science_pass_inference") is not False):
        raise ValueError("terminal inventory schema/self-hash/run/source binding failed")
    launch = inventory.get("launch_identity")
    if (not isinstance(launch, dict) or launch.get("pid") != expected_identity["launch_pid"]
            or launch.get("start_ticks") != expected_identity["launch_start_ticks"]
            or launch.get("process_state") not in {"EXITED", "PID_REUSED_OR_ORIGINAL_EXITED"}
            or (launch.get("process_state") == "EXITED" and launch.get("observed_process_start_ticks") is not None)
            or (launch.get("process_state") == "PID_REUSED_OR_ORIGINAL_EXITED"
                and launch.get("observed_process_start_ticks") in (None, launch.get("start_ticks")))):
        raise ValueError("terminal inventory does not prove the sealed controls process exited")
    source_pins = inventory.get("source_pins")
    source_pin_map = {item.get("role"): item for item in source_pins if isinstance(item, dict)} if isinstance(source_pins, list) else {}
    launch_pin = source_pin_map.get("launch_receipt")
    if set(source_pin_map) != set(INVENTORY_EXPECTED_SOURCE_PINS):
        raise ValueError("terminal inventory source-pin roles differ from the fixed plan/binding/receipt closure")
    for role, (remote, expected_bytes, expected_sha) in INVENTORY_EXPECTED_SOURCE_PINS.items():
        source_item = source_pin_map[role]
        if (source_item.get("path") != remote or source_item.get("bytes") != expected_bytes
                or source_item.get("raw_sha256") != expected_sha):
            raise ValueError(f"terminal inventory source pin mismatch for {role}")
    run_status = inventory.get("run_status")
    if (not isinstance(run_status, dict) or run_status.get("status") != "COMPLETE"
            or run_status.get("planned_rows") != 360 or run_status.get("completed_rows") != 360
            or run_status.get("status_counts") != {"PASS": 360, "FAILED": 0, "INCOMPLETE": 0}
            or run_status.get("repeat_qc_status") != "NOT_RUN_SUBRUN"):
        raise ValueError("terminal inventory run_status summary is not COMPLETE 360-row all-PASS")
    checks = inventory.get("checks")
    if not isinstance(checks, list) or not checks or any(
            not isinstance(item, dict) or item.get("status") != "PASS" for item in checks
    ) or inventory.get("errors") != []:
        raise ValueError("terminal inventory contains failed/missing integrity checks")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(INVENTORY_ARTIFACTS):
        raise ValueError("terminal inventory must contain exactly six pinned artifacts")
    inventory_by_name = {item.get("name"): item for item in artifacts if isinstance(item, dict)}
    expected_names = {name for name, _ in INVENTORY_ARTIFACTS.values()}
    if len(inventory_by_name) != len(artifacts) or set(inventory_by_name) != expected_names:
        raise ValueError("terminal inventory artifact names differ from the exact six-file allowlist")
    run_status_pin = inventory_by_name["run_status"]
    if (run_status.get("path") != run_status_pin.get("path")
            or run_status.get("bytes") != run_status_pin.get("bytes")
            or run_status.get("raw_sha256") != run_status_pin.get("raw_sha256")):
        raise ValueError("terminal inventory run_status summary differs from its pinned artifact entry")
    pins: dict[str, Any] = {}
    for role, (inventory_name, remote_path) in INVENTORY_ARTIFACTS.items():
        item = inventory_by_name[inventory_name]
        expected_local = local_by_role[role]
        artifact_raw = read_regular(expected_local, role,
                                    128 * 1024 * 1024 if role == "results_ledger" else 32 * 1024 * 1024)
        if (item.get("path") != remote_path or item.get("exists") is not True or item.get("status") != "PINNED"
                or item.get("bytes") != len(artifact_raw) or item.get("raw_sha256") != sha(artifact_raw)):
            raise ValueError(f"terminal inventory pin differs from actual reconstructed {role} bytes/path")
        pins[role] = {"inventory_name": inventory_name, "remote": remote_path,
                      "local": str(expected_local), "bytes": len(artifact_raw),
                      "raw_sha256": sha(artifact_raw)}
    summary = {"path": str(path), "bytes": len(raw), "raw_sha256": sha(raw),
            "inventory_sha256": inventory["inventory_sha256"], "run_id": inventory["run_id"],
            "stage_id": inventory["stage_id"], "pid": launch["pid"],
            "start_ticks": launch["start_ticks"], "process_state": launch["process_state"],
            "checks_passed": len(checks), "source_launch_receipt": launch_pin,
            "artifacts": pins}
    return summary, inventory, raw


def validate_control_launch_receipt(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    expected_path = (root_from_script() / CONTROLS_LAUNCH_RECEIPT_LOCAL).resolve(strict=True)
    actual_path = path.resolve(strict=True)
    receipt, raw = read_json(actual_path, "actual controls launch receipt", 2 * 1024 * 1024)
    if (actual_path != expected_path or len(raw) != CONTROLS_LAUNCH_RECEIPT_BYTES
            or sha(raw) != CONTROLS_LAUNCH_RECEIPT_SHA256
            or receipt.get("schema") != "cross_tfg_dev360_detached_launch_receipt_v1"
            or receipt.get("status") != "STARTED_DETACHED"
            or receipt.get("request_id") != "codex_controls360_attempt10_launch01"
            or receipt.get("run_id") != CONTROLS_RUN_ID or receipt.get("stage_id") != "controls360"
            or receipt.get("launch_receipt_path") != CONTROLS_LAUNCH_RECEIPT_REMOTE
            or receipt.get("output_root") != "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360"
            or receipt.get("plan_raw_sha256") != CONTROLS_PLAN_SHA256
            or receipt.get("plan_manifest_raw_sha256") != CONTROLS_PLAN_MANIFEST_SHA256
            or receipt.get("planned_rows_raw_sha256") != CONTROLS_ROWS_SHA256
            or receipt.get("runtime_binding_raw_sha256") != RUNTIME_BINDING_SHA256
            or not self_hash_ok(receipt, "receipt_sha256")
            or type(receipt.get("pid")) is not int or receipt["pid"] <= 0
            or type(receipt.get("start_ticks")) is not int or receipt["start_ticks"] <= 0):
        raise ValueError("actual controls launch receipt fails the pinned run/source/self-hash checks")
    identity = {"launch_pid": receipt["pid"], "launch_start_ticks": receipt["start_ticks"]}
    return identity, {"path": str(actual_path), "bytes": len(raw), "raw_sha256": sha(raw),
                      "receipt_sha256": receipt["receipt_sha256"], "pid": receipt["pid"],
                      "start_ticks": receipt["start_ticks"], "run_id": receipt["run_id"],
                      "stage_id": receipt["stage_id"]}


def validate_candidate_metadata(root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    _, binding_raw = fixed_input(root, "runtime_binding")
    _, plan_raw = fixed_input(root, "candidate_plan")
    binding = json_bytes(binding_raw, "runtime binding for candidate metadata")
    plan = json_bytes(plan_raw, "candidate plan for metadata policy binding")
    resources = binding.get("resources")
    min_gpu = resources.get("min_gpu_free_mib") if isinstance(resources, dict) else None
    min_ram = resources.get("min_effective_ram_available_bytes") if isinstance(resources, dict) else None
    reserve = resources.get("output_reserve_bytes") if isinstance(resources, dict) else None
    output_target = resources.get("output_mount_target") if isinstance(resources, dict) else None
    basis = resources.get("capacity_basis") if isinstance(resources, dict) else None
    stage_bytes = basis.get("stage_output_required_bytes", {}).get("candidate180") if isinstance(basis, dict) else None
    if (type(min_gpu) is not int or type(min_ram) is not int or type(reserve) is not int
            or type(stage_bytes) is not int or not isinstance(output_target, str)):
        raise ValueError("pinned runtime resource floors and candidate180 capacity shares are incomplete")
    execution = plan.get("execution", {})
    strict = execution.get("strict_renderer_determinism", {})
    runtime_ref = execution.get("samehost_runtime_binding", {})
    if (binding.get("gpu_uuid") != GPU_UUID or binding.get("determinism_policy_sha256") != POLICY_SHA256
            or runtime_ref.get("raw_sha256") != RUNTIME_BINDING_SHA256
            or runtime_ref.get("gpu_uuid") != GPU_UUID
            or runtime_ref.get("determinism_policy_sha256") != POLICY_SHA256
            or strict.get("qualified_policy_reference_sha256") != POLICY_SHA256
            or strict.get("policy_values") != binding.get("strict_determinism_policy")
            or strict.get("confirm_audio_read") is not False
            or execution.get("confirm_audio_read") is not False
            or execution.get("confirm_effects_read") is not False
            or metadata.get("status") != "PASS_METADATA_ONLY"
            or metadata.get("candidate_rows") != 180
            or metadata.get("plan_raw_sha256") != CANDIDATE_PLAN_SHA256
            or metadata.get("runtime_binding_raw_sha256") != RUNTIME_BINDING_SHA256
            or metadata.get("gpu_uuid") != GPU_UUID
            or metadata.get("execution_performed") is not False
            or metadata.get("confirm_audio_read") is not False):
        raise ValueError("candidate metadata result is not bound to the pinned GPU/strict-policy/metadata-only plan")
    if not isinstance(resources, dict):
        raise ValueError("pinned runtime binding lacks its resource admission fields")
    probe = metadata.get("live_resource_probe")
    if not isinstance(probe, dict):
        raise ValueError("candidate metadata result lacks the live resource probe")
    gpus = probe.get("gpus")
    selected = [item for item in gpus if isinstance(item, dict) and item.get("uuid") == GPU_UUID] if isinstance(gpus, list) else []
    if (len(selected) != 1 or selected[0].get("index") != 0
            or type(selected[0].get("memory_free_mib")) is not int
            or selected[0]["memory_free_mib"] < min_gpu):
        raise ValueError("candidate live probe does not show the pinned GPU on default visible index 0 above its free-memory floor")
    admission = probe.get("capacity_admission")
    required = stage_bytes + reserve
    if (not isinstance(admission, dict) or admission.get("basis_status") != "MEASURED_PILOT_WITH_EXPLICIT_SCENARIO_AND_LIVE_GATES"
            or admission.get("media_bytes_estimated") is not True or admission.get("stage") != "candidate180"
            or admission.get("remaining_scenario_bytes") != stage_bytes
            or admission.get("separate_policy_reserve_bytes") != reserve
            or admission.get("required_free_bytes") != required
            or admission.get("scenario_is_hard_bound") is not False):
        raise ValueError("candidate live probe lacks the pinned candidate180 capacity-scenario-plus-reserve admission")
    ram = probe.get("ram", {}).get("available_bytes") if isinstance(probe.get("ram"), dict) else None
    shm = probe.get("shm", {}).get("available_bytes") if isinstance(probe.get("shm"), dict) else None
    disk = probe.get("output_disk", {})
    if (type(ram) is not int or ram < min_ram
            or type(shm) is not int or shm < required
            or not isinstance(disk, dict) or disk.get("target") != output_target
            or type(disk.get("available_bytes")) is not int or disk["available_bytes"] < required):
        raise ValueError("candidate live probe does not meet the bound RAM/output/tmpfs capacity gate")
    return {"gpu_uuid": GPU_UUID, "gpu_index": 0, "gpu_free_mib": selected[0]["memory_free_mib"],
            "gpu_min_free_mib": min_gpu, "policy_sha256": POLICY_SHA256,
            "candidate180_required_free_bytes": required, "output_free_bytes": disk["available_bytes"],
            "shm_free_bytes": shm, "ram_available_bytes": ram}


def _metadata_request_table(root: Path, index: dict[str, Any], index_path: Path) -> dict[str, tuple[dict[str, Any], Path, bytes]]:
    requests = index.get("requests")
    expected = (
        ("00_create_candidate13_metadata_namespace", "exec"),
        ("01_run_candidate13_metadata_only_preflight", "exec"),
        ("02_download_candidate13_metadata_preflight", "download"),
    )
    if not isinstance(requests, list) or len(requests) != len(expected):
        raise ValueError("actual metadata request index must name exactly q00, q01, and q02")
    table: dict[str, tuple[dict[str, Any], Path, bytes]] = {}
    request_dir: Path | None = None
    for item, (name, op) in zip(requests, expected):
        if (not isinstance(item, dict) or item.get("name") != name or item.get("op") != op
                or type(item.get("bytes")) is not int or item.get("bytes") <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
            raise ValueError(f"metadata request index q entry has an unexpected name/op/pin: {name}")
        path = required_file(str(item.get("path", "")), f"metadata request {name}", root)
        raw = read_regular(path, f"metadata request {name}", 4 * 1024 * 1024)
        if len(raw) != item["bytes"] or sha(raw) != item["sha256"]:
            raise ValueError(f"metadata request {name} differs from the actual request-index byte/SHA pin")
        if request_dir is None:
            request_dir = path.parent.resolve(strict=True)
        elif path.parent.resolve(strict=True) != request_dir:
            raise ValueError("metadata q00/q01/q02 requests do not share one sealed requests directory")
        table[name] = (json_bytes(raw, name), path, raw)
    if request_dir is None or request_dir.parent.resolve(strict=True) != index_path.parent.resolve(strict=True):
        raise ValueError("metadata request files are not under the request-index attempt directory")
    return table


def _source_pin_from_plan(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    execution = plan.get("execution", {})
    closure = execution.get("samehost_dev360_source_closure", {})
    strict = execution.get("strict_renderer_determinism", {})
    refs = {
        "candidate13_guard": execution.get("samehost_candidate_guarded_launcher"),
        "candidate_dispatcher07": closure.get("step120_candidate_dispatcher"),
        "stagea_core06": closure.get("samehost_stagea_core"),
        "frontend_adapter09": closure.get("samehost_cross_frontend_adapter"),
    }
    out: dict[str, dict[str, Any]] = {}
    for role, ref in refs.items():
        if (not isinstance(ref, dict) or not isinstance(ref.get("path"), str)
                or type(ref.get("bytes")) is not int or ref["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("raw_sha256", "")))):
            raise ValueError(f"frozen candidate plan lacks a complete source pin for {role}")
        remote = ref["path"] if ref["path"].startswith("/") else ROOT_REMOTE + "/" + ref["path"]
        out[role] = {"path": ref["path"], "remote_path": remote,
                     "bytes": ref["bytes"], "raw_sha256": ref["raw_sha256"]}
    bootstrap = strict.get("bootstrap_source")
    if (not isinstance(bootstrap, dict) or not isinstance(bootstrap.get("path"), str)
            or type(bootstrap.get("bytes")) is not int
            or not re.fullmatch(r"[0-9a-f]{64}", str(bootstrap.get("raw_sha256", "")))):
        raise ValueError("frozen candidate plan lacks the strict bootstrap source pin")
    remote = bootstrap["path"] if bootstrap["path"].startswith("/") else ROOT_REMOTE + "/" + bootstrap["path"]
    out["strict_bootstrap03"] = {"path": bootstrap["path"], "remote_path": remote,
                                 "bytes": bootstrap["bytes"], "raw_sha256": bootstrap["raw_sha256"]}
    return out


def validate_metadata_bundle(root: Path, index_path: Path, supplied_index_bytes: int,
                             supplied_index_sha256: str, metadata_path: Path,
                             q00_response_path: Path, q01_response_path: Path,
                             q02_response_path: Path, plan_raw: bytes, rows_raw: bytes,
                             manifest_raw: bytes, binding_raw: bytes) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if METADATA_PREFLIGHT_INDEX_REL is None or any(value is None for value in (
            CANDIDATE_PLAN_REMOTE, CANDIDATE_ROWS_REMOTE, CANDIDATE_MANIFEST_REMOTE,
            METADATA_PREFLIGHT_CONTROL_ROOT, CANDIDATE_OUTPUT_ROOT)):
        raise ValueError("metadata preflight identity pins are not sealed into this builder")
    expected_index_path = (root / METADATA_PREFLIGHT_INDEX_REL).resolve(strict=True)
    if index_path.resolve(strict=True) != expected_index_path:
        raise ValueError("required metadata request index is not the sealed repaired-preflight index path")
    index_raw = read_regular(index_path, "actual metadata preflight request index", 4 * 1024 * 1024)
    if (type(supplied_index_bytes) is not int or supplied_index_bytes != len(index_raw)
            or supplied_index_bytes != METADATA_PREFLIGHT_INDEX_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", supplied_index_sha256)
            or supplied_index_sha256 != sha(index_raw)
            or supplied_index_sha256 != METADATA_PREFLIGHT_INDEX_SHA256):
        raise ValueError("required metadata request-index byte/SHA CLI pins must match the actual file")
    index = json_bytes(index_raw, "actual metadata request index")
    if (index.get("schema") != "cross_tfg_candidate13_metadata_preflight_attempt04_request_index_v1"
            or not self_hash_ok(index, "index_sha256")
            or index.get("candidate_run_id") != CANDIDATE_RUN_ID
            or index.get("controls_run_id") != CONTROLS_RUN_ID
            or index.get("candidate_cli_mode") != "metadata-only: --execute-candidate omitted"
            or index.get("control_root") != METADATA_PREFLIGHT_CONTROL_ROOT
            or index.get("candidate_output_leaf") != CANDIDATE_OUTPUT_ROOT
            or index.get("candidate_output_leaf_created") is not False
            or index.get("project_root") != ROOT_REMOTE
            or index.get("python_executable") != PYTHON):
        raise ValueError("metadata request index schema/run/control/output/source identity does not match the sealed preflight")
    expected_detached = {
        "candidate_output_leaf": CANDIDATE_OUTPUT_ROOT, "execute_flag": "--execute-candidate",
        "future_launch_request_created": False, "future_preflight_or_execution_hashes_filled": False,
        "status": "DEFERRED_UNTIL_CONTROLS_TERMINAL_AND_ACTUAL_PREFLIGHT_READBACK",
    }
    if index.get("detached_launch") != expected_detached:
        raise ValueError("metadata request index detached-launch fields are not deferred and future-hash-free")
    plan = json_bytes(plan_raw, "candidate plan for metadata request index")
    rows_ref = index.get("candidate_rows")
    plan_ref = index.get("candidate_plan")
    manifest_ref = index.get("candidate_manifest")
    binding_ref = index.get("runtime_binding")
    plan_manifest = json_bytes(manifest_raw, "candidate frozen manifest for metadata request index")
    expected_refs = (
        (plan_ref, CANDIDATE_PLAN_REMOTE, len(plan_raw), sha(plan_raw), plan.get("plan_sha256"), None),
        (rows_ref, CANDIDATE_ROWS_REMOTE, len(rows_raw), sha(rows_raw), None, None),
        (manifest_ref, CANDIDATE_MANIFEST_REMOTE, len(manifest_raw), sha(manifest_raw), plan_manifest.get("manifest_sha256"), "FROZEN"),
        (binding_ref, ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json",
         len(binding_raw), sha(binding_raw), None, None),
    )
    for ref, remote, size, digest, self_digest, status in expected_refs:
        if (not isinstance(ref, dict) or ref.get("path") != remote or ref.get("bytes") != size
                or ref.get("raw_sha256") != digest
                or (self_digest is not None and ref.get("self_sha256") != self_digest)
                or (status is not None and ref.get("status") != status)):
            raise ValueError("metadata request index candidate plan/rows/manifest/binding refs differ from pinned bytes")
    if binding_ref.get("gpu_uuid") != GPU_UUID:
        raise ValueError("metadata request index does not bind the sealed runtime-binding GPU")
    expected_source_pins = _source_pin_from_plan(plan)
    source_evidence = index.get("source_evidence")
    if not isinstance(source_evidence, dict):
        raise ValueError("metadata request index lacks source_evidence")
    for role in expected_source_pins:
        expected = expected_source_pins[role]
        local_expected = (root / expected["path"]).resolve(strict=True)
        local_raw = read_regular(local_expected, role, expected["bytes"] + 1)
        if len(local_raw) != expected["bytes"] or sha(local_raw) != expected["raw_sha256"]:
            raise ValueError(f"frozen source pin changed locally: {role}")
    expected_source_evidence = {
        "candidate13_guard": (expected_source_pins["candidate13_guard"]["bytes"], expected_source_pins["candidate13_guard"]["raw_sha256"],
                              (root / expected_source_pins["candidate13_guard"]["path"]).resolve(strict=True)),
        "candidate_dispatcher07": (expected_source_pins["candidate_dispatcher07"]["bytes"], expected_source_pins["candidate_dispatcher07"]["raw_sha256"],
                                   (root / expected_source_pins["candidate_dispatcher07"]["path"]).resolve(strict=True)),
        "stagea_core06": (expected_source_pins["stagea_core06"]["bytes"], expected_source_pins["stagea_core06"]["raw_sha256"],
                          (root / expected_source_pins["stagea_core06"]["path"]).resolve(strict=True)),
        "candidate_plan13": (len(plan_raw), sha(plan_raw), fixed_input(root, "candidate_plan")[0]),
        "candidate_rows12": (len(rows_raw), sha(rows_raw), fixed_input(root, "candidate_rows")[0]),
        "candidate_manifest14": (len(manifest_raw), sha(manifest_raw), fixed_input(root, "candidate_manifest14")[0]),
        "runtime_binding09": (len(binding_raw), sha(binding_raw), fixed_input(root, "runtime_binding")[0]),
        "baseline_controls_plan09": (CONTROLS_PLAN_BYTES, CONTROLS_PLAN_SHA256, fixed_input(root, "controls_plan")[0]),
        "generator_pin_repair_receipt": (5346, "803b1d6a2b4ef2ea690454292b4fd5f96fb2dd8f69cc661e571282e2e430e4df",
                                          (root / ".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/repair_receipt_attempt01.json").resolve(strict=True)),
        "remote_upload_readback_request_index": (7180, "a0dc5ff40a2388710207a30de0570b7f3342083bdf4bb5bcd81ac154e642d79b",
                                                 (root / ".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/request_index_attempt01.json").resolve(strict=True)),
    }
    if set(source_evidence) != set(expected_source_evidence):
        raise ValueError("metadata request index source_evidence roles differ from the sealed repaired-plan evidence")
    for evidence_key, (expected_size, expected_digest, local_expected) in expected_source_evidence.items():
        evidence = source_evidence.get(evidence_key)
        if not isinstance(evidence, dict):
            raise ValueError(f"metadata request index source evidence is missing: {evidence_key}")
        evidence_path = Path(str(evidence.get("path", ""))).resolve(strict=True)
        if (evidence_path != local_expected or evidence.get("bytes") != expected_size
                or evidence.get("sha256") != expected_digest):
            raise ValueError(f"metadata request index source_evidence differs from frozen plan: {evidence_key}")
        evidence_raw = read_regular(local_expected, "metadata index evidence " + evidence_key, expected_size + 1)
        if len(evidence_raw) != expected_size or sha(evidence_raw) != expected_digest:
            raise ValueError(f"metadata request index supporting evidence has changed: {evidence_key}")
    expected_scope = {
        "candidate_audio_opened_or_decoded": False, "candidate_output_leaf_created": False,
        "confirmation_read": False, "future_preflight_result_sha_guessed": False,
        "gpu_or_model_execution_performed": False,
        "preflight_dispatched": False, "remote_upload_or_readback_performed": False,
        "scores_or_selection_performed": False,
    }
    if index.get("scope") != expected_scope:
        raise ValueError("metadata request index scope is not the exact metadata-only/no-execution scope")
    expected_preflight = {"candidate_rows": 180, "candidate_runtime_qualification_performed": False,
                          "confirm_audio_read": False, "execution_performed": False,
                          "status": "PASS_METADATA_ONLY"}
    if index.get("expected_preflight_result") != expected_preflight:
        raise ValueError("metadata request index expected result differs from the frozen metadata-only contract")
    requests = _metadata_request_table(root, index, index_path)
    q00, _, _ = requests["00_create_candidate13_metadata_namespace"]
    q01, _, _ = requests["01_run_candidate13_metadata_only_preflight"]
    q02, _, _ = requests["02_download_candidate13_metadata_preflight"]
    q02_local = required_file(str(q02.get("local", "")), "metadata q02 local output path", root)
    if metadata_path.resolve(strict=True) != q02_local.resolve(strict=True):
        raise ValueError("metadata output path differs from the sealed q02 local readback reference")
    metadata_raw = read_regular(metadata_path, "actual candidate metadata preflight", 2 * 1024 * 1024)
    expected_remote = METADATA_PREFLIGHT_CONTROL_ROOT + "/output/" + metadata_path.name
    if index.get("candidate_metadata_output") != expected_remote:
        raise ValueError("metadata output remote path is outside the sealed preflight namespace")
    if (q02.get("op") != "download" or q02.get("remote") != expected_remote
            or q02_local.resolve(strict=True) != metadata_path.resolve(strict=True)):
        raise ValueError("metadata q02 request does not read back the exact preflight output")
    q02_response = download_response(q02_response_path, expected_remote, metadata_path, metadata_raw)
    q00_response = clean_exec_response(q00_response_path, "PASS_CANDIDATE13_METADATA_PREFLIGHT_NAMESPACE_CREATED")
    q01_response = clean_exec_response(q01_response_path, "PASS_METADATA_ONLY")
    q01_envelope, _ = read_json(q01_response_path, "metadata q01 response", 8 * 1024 * 1024)
    if q01_envelope["stdout"].encode("utf-8") != metadata_raw:
        raise ValueError("metadata q01 stdout does not exactly match the downloaded metadata bytes")
    metadata = json_bytes(metadata_raw, "actual candidate metadata output")
    if q01.get("op") != "exec" or not isinstance(q01.get("command"), str):
        raise ValueError("metadata q01 request is not an exec request")
    command_tokens = shlex.split(q01["command"], posix=True)
    def assignment(name: str) -> str:
        matches = [token[len(name) + 1:].rstrip(";") for token in command_tokens if token.startswith(name + "=")]
        if len(matches) != 1:
            raise ValueError(f"metadata q01 must contain exactly one {name} assignment")
        return matches[0]
    expected_guard = expected_source_pins["candidate13_guard"]["remote_path"]
    expected_values = {
        "--project-root": ROOT_REMOTE,
        "--candidate-plan": CANDIDATE_PLAN_REMOTE,
        "--candidate-plan-raw-sha256": sha(plan_raw),
        "--planned-rows": CANDIDATE_ROWS_REMOTE,
        "--planned-rows-raw-sha256": sha(rows_raw),
        "--plan-manifest": CANDIDATE_MANIFEST_REMOTE,
        "--plan-manifest-raw-sha256": sha(manifest_raw),
        "--runtime-binding": binding_ref["path"],
        "--runtime-binding-raw-sha256": sha(binding_raw),
    }
    try:
        cd_index = command_tokens.index("cd")
    except ValueError:
        cd_index = -1
    if (assignment("GUARD") != expected_guard
            or assignment("PYTHON") != PYTHON
            or assignment("CONTROL") != METADATA_PREFLIGHT_CONTROL_ROOT
            or assignment("CANDIDATE_LEAF") != CANDIDATE_OUTPUT_ROOT
            or assignment("OUTPUT") != expected_remote
            or cd_index < 0 or cd_index + 1 >= len(command_tokens)
            or command_tokens[cd_index + 1].rstrip(";") != ROOT_REMOTE
            or "$GUARD" not in command_tokens or "$PYTHON" not in command_tokens
            or "--execute-candidate" in command_tokens or "--confirm-audio-read" in command_tokens
            or "--confirm-effects-read" in command_tokens
            or q01["command"].count(expected_remote) < 1):
        raise ValueError("metadata q01 command is not the frozen guarded metadata-only invocation")
    for flag, value in expected_values.items():
        if flag_value(command_tokens, flag) != value:
            raise ValueError(f"metadata q01 command differs from frozen candidate identity: {flag}")
    if (q01["command"].count("--execute-candidate") or "rm " in q01["command"]
            or "sudo " in q01["command"] or "ssh " in q01["command"]):
        raise ValueError("metadata q01 command includes an execution/replay or unrelated shell action")
    # The q00 payload itself is pinned and must create only its isolated preflight namespace.
    if (q00.get("op") != "exec" or not isinstance(q00.get("command"), str)
            or METADATA_PREFLIGHT_CONTROL_ROOT not in q00["command"]
            or CANDIDATE_OUTPUT_ROOT not in q00["command"]
            or "mkdir" not in q00["command"] or "rm " in q00["command"]
            or "--execute-candidate" in q00["command"]):
        raise ValueError("metadata q00 command is not a create-only preflight namespace request")
    q00_envelope, _ = read_json(q00_response_path, "metadata q00 response", 8 * 1024 * 1024)
    q00_output = json_bytes(q00_envelope["stdout"].encode("utf-8"), "metadata q00 stdout")
    if (q00_output.get("status") != "PASS_CANDIDATE13_METADATA_PREFLIGHT_NAMESPACE_CREATED"
            or q00_output.get("candidate_run_id") != CANDIDATE_RUN_ID
            or q00_output.get("candidate_plan_raw_sha256") != sha(plan_raw)
            or q00_output.get("runtime_binding_raw_sha256") != sha(binding_raw)
            or q00_output.get("candidate_media_output_leaf_created") is not False
            or q00_output.get("candidate_execution_started") is not False):
        raise ValueError("metadata q00 stdout does not bind the exact candidate plan and no-execution state")
    metadata_admission = validate_candidate_metadata(root, metadata)
    metadata_detail = {"path": str(metadata_path), "bytes": len(metadata_raw), "raw_sha256": sha(metadata_raw),
                       "remote": expected_remote, "status": metadata["status"],
                       "plan_raw_sha256": metadata["plan_raw_sha256"],
                       "runtime_binding_raw_sha256": metadata["runtime_binding_raw_sha256"],
                       "gpu_uuid": metadata["gpu_uuid"], "execution_performed": False,
                       "confirm_audio_read": False, "live_resource_admission": metadata_admission,
                       "request_index": {"path": str(index_path), "bytes": len(index_raw),
                                         "raw_sha256": sha(index_raw), "index_sha256": index.get("index_sha256")},
                       "requests": {name: {"path": str(item[1]), "bytes": len(item[2]),
                                            "raw_sha256": sha(item[2]), "op": item[0].get("op")}
                                    for name, item in requests.items()},
                       "namespace_response": q00_response, "metadata_exec_response": q01_response,
                       "readback_response": q02_response,
                       "manifest_pin_evidence_boundary": "The metadata object does not emit the plan-manifest raw SHA; the frozen plan/manifest bytes and manifest14 lineage are separately pinned, then checked again by the launcher."}
    return index, metadata_detail, metadata


def validate_inventory_readback(exec_response_path: Path, download_response_path: Path,
                                inventory_path: Path, inventory: dict[str, Any],
                                inventory_raw: bytes) -> dict[str, Any]:
    exec_detail = clean_exec_response(exec_response_path, "TERMINAL_INVENTORY_READY")
    exec_envelope, _ = read_json(exec_response_path, "terminal inventory exec response", 8 * 1024 * 1024)
    returned = json_bytes(exec_envelope["stdout"].encode("utf-8"), "terminal inventory exec stdout")
    if (returned.get("schema") != inventory.get("schema")
            or returned.get("status") != "TERMINAL_INVENTORY_READY"
            or returned.get("inventory_path") != CONTROLS_INVENTORY_REMOTE
            or returned.get("inventory_bytes") != len(inventory_raw)
            or returned.get("inventory_raw_sha256") != sha(inventory_raw)
            or returned.get("inventory_sha256") != inventory.get("inventory_sha256")
            or returned.get("artifact_count") != 6 or returned.get("present_count") != 6
            or returned.get("row_count") != 360 or returned.get("checks_fail") != 0
            or returned.get("download_dispatch_allowed") is not True):
        raise ValueError("actual inventory q02 exec response does not bind the downloaded ready inventory")
    download_detail = download_response(download_response_path, CONTROLS_INVENTORY_REMOTE,
                                       inventory_path, inventory_raw)
    return {"exec_response": exec_detail, "download_response": download_detail,
            "inventory_raw_sha256": sha(inventory_raw), "inventory_sha256": inventory.get("inventory_sha256")}


def clean_exec_response(path: Path, expected_phrase: str | None = None) -> dict[str, Any]:
    value, raw = read_json(path, "metadata dispatcher exec response", 8 * 1024 * 1024)
    trunc = value.get("truncated")
    if (value.get("ok") is not True or type(value.get("code")) is not int or value.get("code") != 0
            or value.get("timed_out") is not False or value.get("incomplete_stream") is not False
            or value.get("stream_error") not in (None, "")
            or not isinstance(trunc, dict) or trunc.get("stdout") is not False or trunc.get("stderr") is not False
            or not isinstance(value.get("stdout"), str) or not value["stdout"] or value.get("stderr") != ""):
        raise ValueError("metadata command response is not a clean complete dispatcher envelope")
    if expected_phrase and expected_phrase not in value["stdout"]:
        raise ValueError(f"metadata response stdout does not contain {expected_phrase}")
    return {"path": str(path), "bytes": len(raw), "raw_sha256": sha(raw), "code": 0,
            "stdout_sha256": sha(value["stdout"].encode("utf-8"))}


def download_response(path: Path, expected_remote: str, expected_local: Path,
                      actual_raw: bytes) -> dict[str, Any]:
    value, raw = read_json(path, "metadata readback download response", 1024 * 1024)
    if (value.get("ok") is not True or value.get("remote") != expected_remote
            or Path(value.get("local", "")).resolve(strict=True) != expected_local.resolve(strict=True)
            or type(value.get("size")) is not int or value.get("size") != len(actual_raw)
            or value.get("sha256") != sha(actual_raw)):
        raise ValueError("metadata download response does not bind actual downloaded bytes/path")
    return {"path": str(path), "bytes": len(raw), "raw_sha256": sha(raw), "download_bytes": len(actual_raw),
            "download_raw_sha256": sha(actual_raw), "remote": expected_remote}


def create_exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def create_output_dir(root: Path, path_text: str) -> Path:
    requested = Path(path_text).expanduser()
    if not requested.is_absolute():
        requested = root / requested
    out = requested.resolve(strict=False)
    scratch = (root / ".codex/scratch").resolve(strict=True)
    if not out.is_relative_to(scratch):
        raise ValueError("request packet output must be a new directory under .codex/scratch")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"create-only output already exists: {out}")
    if not out.parent.is_dir() or out.parent.is_symlink():
        raise FileNotFoundError("request packet parent must be an existing real directory")
    out.mkdir(mode=0o700, parents=False)
    return out


def dump_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def remote_check(path: str, size: int, digest: str) -> str:
    q = shlex.quote(path)
    return f"test -f {q}; test ! -L {q}; test \"$(stat -c %s {q})\" = {shlex.quote(str(size))}; test \"$(sha256sum {q} | cut -d ' ' -f1)\" = {shlex.quote(digest)}"


def phase_a(args: argparse.Namespace) -> int:
    root = root_from_script()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.request_id):
        raise ValueError("request id must be a path-safe 1–80 character ASCII token")
    pins = validate_fixed_sources(root)
    actual_local = {role: required_file(getattr(args, role), role, root)
                    for role, (_, option) in CONTROL_ARTIFACTS.items()}
    # Obtain PID/start ticks from the separately pinned actual producer receipt.
    controls_launch_receipt_path = required_file(args.controls_launch_receipt, "controls launch receipt", root)
    terminal_inventory_path = required_file(args.terminal_inventory, "controls terminal inventory", root)
    terminal_inventory_exec_response_path = required_file(args.terminal_inventory_exec_response, "inventory exec response", root)
    terminal_inventory_download_response_path = required_file(args.terminal_inventory_download_response, "inventory download response", root)
    observer_manifest_path = required_file(args.observer_manifest, "observer manifest", root)
    observer_observation_path = required_file(args.observer_observation, "observer observation", root)
    observer_response_path = required_file(args.observer_response, "observer response", root)
    identity, launch_receipt_detail = validate_control_launch_receipt(controls_launch_receipt_path)
    actual_inventory, inventory_value, inventory_raw = validate_terminal_inventory(
        terminal_inventory_path, actual_local, identity)
    inventory_readback = validate_inventory_readback(
        terminal_inventory_exec_response_path, terminal_inventory_download_response_path,
        terminal_inventory_path, inventory_value, inventory_raw)
    # The source observer is an independent terminal-process observation.
    obs_manifest, obs_manifest_raw = read_json(observer_manifest_path, "observer manifest", 2 * 1024 * 1024)
    obs_value, obs_raw = read_json(observer_observation_path, "observer observation", 2 * 1024 * 1024)
    resp_value, resp_raw = read_json(observer_response_path, "observer response", 20 * 1024 * 1024)
    if not isinstance(resp_value.get("stdout"), str):
        raise ValueError("observer response stdout missing")
    observed_snapshot = json_bytes(resp_value["stdout"].encode("utf-8"), "observer terminal snapshot")
    identity_audit = validate_observer(root, observer_manifest_path,
                                       observer_observation_path, observer_response_path, identity)
    if sha(obs_manifest_raw) != identity_audit["manifest"]["raw_sha256"] or sha(obs_raw) != identity_audit["observation"]["raw_sha256"] or sha(resp_raw) != identity_audit["response"]["raw_sha256"]:
        raise ValueError("observer evidence changed during validation")
    if observed_snapshot.get("pid") != identity["launch_pid"] or observed_snapshot.get("start_ticks") != identity["launch_start_ticks"]:
        raise ValueError("observer process PID/start ticks disagree with actual controls launch receipt")

    controls_identity, controls_detail, control_raw = validate_controls(root, args)
    controls_identity["launch_pid"] = identity["launch_pid"]
    controls_identity["launch_start_ticks"] = identity["launch_start_ticks"]
    controls_summary = {**controls_identity, **controls_detail}
    controls_parent = {
        "run_id": CONTROLS_RUN_ID, "rows": 360,
        "status_counts": controls_detail["status_counts"],
        "plan_raw_sha256": controls_detail["controls_plan"]["raw_sha256"],
        "plan_manifest_raw_sha256": controls_detail["controls_plan_manifest"]["raw_sha256"],
        "planned_rows_raw_sha256": controls_detail["controls_rows"]["raw_sha256"],
        "results_raw_sha256": controls_detail["results_ledger"]["raw_sha256"],
        "result_manifest_raw_sha256": controls_detail["result_manifest"]["raw_sha256"],
        "execution_raw_sha256": controls_detail["execution_record"]["raw_sha256"],
        "strict_attestation_raw_sha256": controls_detail["strict_attestation"]["raw_sha256"],
        "repeat_qc_raw_sha256": controls_detail["repeat_qc_ledger"]["raw_sha256"],
        "qualification_manifests": controls_detail["qualification_manifests"],
    }
    metadata_index_path = required_file(args.metadata_request_index, "actual metadata request index", root)
    metadata_path = required_file(args.metadata_output, "actual downloaded candidate metadata preflight", root)
    metadata_q00_response_path = required_file(args.metadata_q00_response, "metadata q00 response", root)
    metadata_q01_response_path = required_file(args.metadata_q01_response, "metadata q01 response", root)
    metadata_q02_response_path = required_file(args.metadata_q02_response, "metadata q02 response", root)
    _, candidate_plan_raw = fixed_input(root, "candidate_plan")
    _, candidate_rows_raw = fixed_input(root, "candidate_rows")
    _, candidate_manifest_raw = fixed_input(root, "candidate_manifest14")
    _, runtime_binding_raw = fixed_input(root, "runtime_binding")
    candidate_source_pins = _source_pin_from_plan(json_bytes(candidate_plan_raw, "candidate plan source closure"))
    metadata_index, metadata_detail, metadata = validate_metadata_bundle(
        root, metadata_index_path, args.metadata_request_index_bytes,
        args.metadata_request_index_raw_sha256, metadata_path,
        metadata_q00_response_path, metadata_q01_response_path,
        metadata_q02_response_path, candidate_plan_raw, candidate_rows_raw,
        candidate_manifest_raw, runtime_binding_raw)
    metadata_raw = read_regular(metadata_path, "actual candidate metadata preflight", 2 * 1024 * 1024)
    metadata_result_remote = metadata_index["candidate_metadata_output"]

    # The inventory, producer launch receipt, observer, reconstructed producer
    # outputs, and metadata are copied to a distinct source packet directory.
    evidence: list[dict[str, Any]] = []
    aux_files = [
        ("terminal_inventory", terminal_inventory_path),
        ("terminal_inventory_exec_response", terminal_inventory_exec_response_path),
        ("terminal_inventory_download_response", terminal_inventory_download_response_path),
        ("controls_launch_receipt", controls_launch_receipt_path),
        ("observer_manifest", observer_manifest_path),
        ("observer_observation", observer_observation_path),
        ("observer_response", observer_response_path),
        ("candidate_metadata_request_index", metadata_index_path),
        ("candidate_metadata_preflight", metadata_path),
        ("candidate_metadata_q00_response", metadata_q00_response_path),
        ("candidate_metadata_q01_response", metadata_q01_response_path),
        ("candidate_metadata_q02_response", metadata_q02_response_path),
    ]
    for item in metadata_index["requests"]:
        aux_files.append(("candidate_metadata_request_" + item["name"],
                          required_file(item["path"], "sealed metadata request " + item["name"], root)))
    for role, local in aux_files:
        raw = read_regular(local, role, 32 * 1024 * 1024)
        extension = local.suffix if local.suffix else ".bin"
        evidence.append({"role": role, "local": str(local), "remote": SOURCE_DIR + "/evidence/" + role + extension,
                         "bytes": len(raw), "raw_sha256": sha(raw)})
    for role, pin in pins.items():
        if role.startswith(("observer_source:", "inventory_source:")):
            evidence.append({"role": role.replace(":", "_"), **pin,
                             "remote": SOURCE_DIR + "/evidence/source/" + Path(pin["local"]).name})
    evidence.extend([{"role": role, **pin} for role, pin in pins.items() if role in SOURCE_PINS])

    # The terminal result itself must have no failures; launch07 will perform its
    # own full terminal/strict-backend validation again on the remote namespace.
    local_by_role = actual_local
    request_files: dict[str, dict[str, Any]] = {}
    mkdir_cmd = ("set -euo pipefail; test -d " + shlex.quote(JOB_CONTROL) + "; test ! -L " + shlex.quote(JOB_CONTROL)
                 + "; test ! -e " + shlex.quote(SOURCE_DIR) + "; test ! -L " + shlex.quote(SOURCE_DIR)
                 + "; mkdir -m 700 " + shlex.quote(SOURCE_DIR) + "; mkdir -m 700 " + shlex.quote(SOURCE_DIR + "/evidence")
                 + "; mkdir -m 700 " + shlex.quote(SOURCE_DIR + "/evidence/source")
                 + "; test -d " + shlex.quote(SOURCE_DIR) + "; test -d " + shlex.quote(SOURCE_DIR + "/evidence")
                 + "; printf '%s\\n' '{\"status\":\"SOURCE_PACKET_CREATED\"}'")
    request_files["00_create_source_packet"] = {"op": "exec", "timeout": 60, "command": mkdir_cmd}
    for i, item in enumerate(evidence, 1):
        if item["bytes"] > 8 * 1024 * 1024:
            raise ValueError(f"source evidence upload exceeds the dispatcher's 8 MiB request cap: {item['role']}")
        request_files[f"{i:02d}_upload_{item['role']}"] = {"op": "upload", "timeout": 120,
            "local": item["local"], "remote": item["remote"],
            "expected_bytes": item["bytes"], "expected_sha256": item["raw_sha256"]}
    source_checks = []
    for item in evidence:
        source_checks.append(remote_check(item["remote"], item["bytes"], item["raw_sha256"]))
    request_files["30_verify_source_packet"] = {"op": "exec", "timeout": 120,
        "command": "set -euo pipefail; " + "; ".join(source_checks)
                    + "; printf '%s\\n' '{\"status\":\"PASS_SOURCE_PACKET_EXACT_BYTES\"}'"}

    checks = []
    for item in evidence:
        if item["remote"].startswith(SOURCE_DIR + "/"):
            checks.append(remote_check(item["remote"], item["bytes"], item["raw_sha256"]))
    for role, values in LOCAL_FIXED.items():
        rel, n, digest = values
        remote = {
            "controls_plan": ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/plan.json",
            "controls_rows": ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/planned_rows.jsonl",
            "controls_plan_manifest": ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/plan_manifest.json",
            "runtime_binding": ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json",
            "candidate_plan": CANDIDATE_PLAN_REMOTE,
            "candidate_rows": CANDIDATE_ROWS_REMOTE,
            "candidate_manifest14": CANDIDATE_MANIFEST_REMOTE,
        }.get(role)
        if remote:
            checks.append(remote_check(remote, n, digest))
    checks.extend(remote_check(remote, n, digest) for remote, n, digest in (
        (CONTROLS_LAUNCH_RECEIPT_REMOTE, CONTROLS_LAUNCH_RECEIPT_BYTES, CONTROLS_LAUNCH_RECEIPT_SHA256),
        (ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01/v3_method_candidate_config.json", 5240, "aba0dc744b76cfd9f73db547faa80508a4a46437432ef78feaf5b7bfb51f3666"),
        (ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01/materialization_receipt.json", 3965, "c64bfb0d80471c6d2764b6921a993a7c556b9511d4cba283f41eba002cdf8ebd"),
        (candidate_source_pins["candidate13_guard"]["remote_path"],
         candidate_source_pins["candidate13_guard"]["bytes"],
         candidate_source_pins["candidate13_guard"]["raw_sha256"]),
        (ROOT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt07.py", 12824, "f6844b51183b6cf3b57f6a62d0b1b9bd70c3a0d496480fe8cad70313907c16e3"),
        (ROOT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py", 79475, "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"),
        (ROOT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt09.py", 23249, "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"),
        (ROOT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03/sitecustomize.py", 9525, STRICT_BOOTSTRAP_SHA256),
        (ROOT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_v3_samehost_dev360_controls_attempt18.py", 78172, "30381ee8deb6d0e7880379fc91a042e1e2bb3cca375dd05423319259fd9e8756"),
        (ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt09/pilot_manifest_final.json", STRICT09_BYTES, STRICT09_SHA256),
        (ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt12/pilot_manifest_final.json", STRICT12_BYTES, STRICT12_SHA256),
    ))
    for role, item in {**{f"terminal_{k}": v for k, v in validate_terminal_file_pins(actual_local).items()}}.items():
        checks.append(remote_check(item["remote"], item["bytes"], item["raw_sha256"]))
    checks.append(remote_check(metadata_result_remote, len(metadata_raw), sha(metadata_raw)))
    launcher_remote = SOURCE_DIR + "/" + SOURCE_PINS["launcher08"][3]
    checks.append(remote_check(launcher_remote, SOURCE_PINS["launcher08"][1], SOURCE_PINS["launcher08"][2]))
    job_q = shlex.quote(JOB)
    source_q = shlex.quote(launcher_remote)
    launch_command = ("set -euo pipefail; " + "; ".join(checks) + "; test ! -e " + job_q + "; test ! -L " + job_q
        + "; mkdir -m 700 " + job_q + "; test -d " + job_q + "; test ! -L " + job_q
        + "; test -z \"$(find " + job_q + " -mindepth 1 -maxdepth 1 -print -quit)\""
        + "; test ! -e " + shlex.quote(str(CANDIDATE_OUTPUT_ROOT))
        + "; test ! -L " + shlex.quote(str(CANDIDATE_OUTPUT_ROOT))
        + "; PYTHONDONTWRITEBYTECODE=1 " + shlex.quote(PYTHON) + " -B " + source_q
        + " --request-id " + shlex.quote(args.request_id)
        + " --candidate-preflight-bytes " + shlex.quote(str(len(metadata_raw)))
        + " --candidate-preflight-raw-sha256 " + shlex.quote(sha(metadata_raw)))
    request_files["40_launch_candidate180_once"] = {"op": "exec", "timeout": 2400, "command": launch_command}

    out = create_output_dir(root, args.output_dir)
    reqdir = out / "requests"
    reqdir.mkdir(mode=0o700)
    request_manifest = []
    for name, request in request_files.items():
        raw = dump_json(request)
        path = reqdir / (name + ".json")
        create_exclusive(path, raw)
        request_manifest.append({"name": name, "path": str(path), "op": request["op"],
                                 "bytes": len(raw), "sha256": sha(raw), "raw_sha256": sha(raw)})
    dispatcher_source_evidence = {
        item["role"]: {"path": item["local"], "bytes": item["bytes"], "sha256": item["raw_sha256"]}
        for item in evidence
    }
    for role in LOCAL_FIXED:
        fixed_path, fixed_raw = fixed_input(root, role)
        dispatcher_source_evidence["fixed_" + role] = {
            "path": str(fixed_path), "bytes": len(fixed_raw), "sha256": sha(fixed_raw)}
    for role, terminal_path in actual_local.items():
        terminal_raw = read_regular(terminal_path, "dispatcher terminal source " + role,
                                    128 * 1024 * 1024 if role == "results_ledger" else 32 * 1024 * 1024)
        dispatcher_source_evidence["terminal_" + role] = {
            "path": str(terminal_path), "bytes": len(terminal_raw), "sha256": sha(terminal_raw)}
    index = {"schema": "cross_tfg_candidate180_required_input_dispatch_packet_v1", "phase": "A_SOURCE_UPLOAD_AND_SINGLE_LAUNCH",
        "status": "PREPARED_REQUESTS_ONLY_NOT_DISPATCHED", "request_id": args.request_id,
        "candidate_run_id": CANDIDATE_RUN_ID, "controls_parent_run_id": CONTROLS_RUN_ID,
        "job_namespace": JOB, "source_packet_namespace": SOURCE_DIR,
        "controls_launch_receipt": launch_receipt_detail,
        "controls_terminal_inventory": actual_inventory,
        "controls_terminal_inventory_readback": inventory_readback,
        "observer_terminal": identity_audit,
        "controls_terminal_summary": controls_summary, "controls_parent": controls_parent,
        "candidate_metadata_preflight": metadata_detail,
        "source_evidence": dispatcher_source_evidence,
        "source_and_static_pins": pins, "uploaded_evidence": evidence, "requests": request_manifest,
        "execution_policy": {"builder_executes_requests": False, "ssh_or_api_called": False, "gpu_or_model_called": False,
            "automatic_retry": False, "launch_is_create_only_single_request": True,
            "candidate_preflight_bytes_sha_derived_from_actual_file": True,
            "controls_process_identity_source": "actual pinned controls launch receipt, matched against terminal inventory and observer response",
            "controls_terminal_inventory_exec_and_download_responses_required": True,
            "six_terminal_artifact_local_paths_are_explicit_inputs": True,
            "obsolete_fixed_direct_readback_index_used": False,
            "cuda_visible_devices_source_field": "runtime_binding_attempt09.json omits top-level cuda_visible_devices; frozen launcher08 resolves binding.get('cuda_visible_devices', '0') to literal '0'",
            "source_dir_is_distinct_from_empty_job07": True,
            "phase_b_receipt_bytes_sha256": "REQUIRED_FROM_ACTUAL_LAUNCH_RECEIPT; no value prefilled"}}
    index_path = out / "request_index_attempt02.json"
    index["index_sha256"] = sha(canonical(index))
    create_exclusive(index_path, dump_json(index))
    return 0


def validate_terminal_file_pins(local_by_role: dict[str, Path]) -> dict[str, dict[str, Any]]:
    pinned = {}
    for role, path in local_by_role.items():
        raw = read_regular(path, role, 128 * 1024 * 1024 if role == "results_ledger" else 32 * 1024 * 1024)
        pinned[role] = {"remote": CONTROL_ARTIFACTS[role][0], "bytes": len(raw), "raw_sha256": sha(raw)}
    return pinned


def validate_candidate_receipt(root: Path, receipt: dict[str, Any], request_id: str,
                               phase_a: dict[str, Any]) -> None:
    if any(value is None for value in (CANDIDATE_PLAN_REMOTE, CANDIDATE_ROWS_REMOTE,
                                       CANDIDATE_MANIFEST_REMOTE, CANDIDATE_OUTPUT_ROOT)):
        raise ValueError("candidate receipt identity pins are not sealed into this builder")
    expected_receipt_path = JOB + "/candidate180_" + request_id + ".launch_receipt.json"
    expected_stdout = JOB + "/candidate180_" + request_id + ".stdout.log"
    expected_stderr = JOB + "/candidate180_" + request_id + ".stderr.log"
    expected_fresh = JOB + "/candidate180_" + request_id + ".fresh_metadata_preflight.json"
    if (receipt.get("schema") != "cross_tfg_candidate180_detached_launch_receipt_v1"
            or receipt.get("status") != "STARTED_DETACHED" or receipt.get("stage_id") != "candidate180"
            or receipt.get("request_id") != request_id or receipt.get("run_id") != CANDIDATE_RUN_ID
            or receipt.get("controls_run_id") != CONTROLS_RUN_ID
            or receipt.get("launch_receipt_path") != expected_receipt_path
            or receipt.get("stdout_path") != expected_stdout or receipt.get("stderr_path") != expected_stderr
            or receipt.get("output_root") != CANDIDATE_OUTPUT_ROOT
            or not self_hash_ok(receipt, "receipt_sha256")
            or type(receipt.get("pid")) is not int or receipt["pid"] <= 0
            or type(receipt.get("start_ticks")) is not int or receipt["start_ticks"] <= 0):
        raise ValueError("candidate receipt stage/run/actual paths/output root/identity/self-hash mismatch")
    plan_path, plan_raw = fixed_input(root, "candidate_plan")
    rows_path, rows_raw = fixed_input(root, "candidate_rows")
    manifest_path, manifest_raw = fixed_input(root, "candidate_manifest14")
    binding_path, binding_raw = fixed_input(root, "runtime_binding")
    plan = json_bytes(plan_raw, "candidate plan for launch receipt")
    candidate_rows = parse_jsonl(rows_raw, "candidate rows for launch receipt", 180)
    manifest = json_bytes(manifest_raw, "candidate manifest for launch receipt")
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("run_id") != CANDIDATE_RUN_ID
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or plan.get("plan_sha256") != "fee2718c0c4b18e5acabf3ca7d7103344bfb3859f3d8a3b93862c1ad4ce7d612"
            or plan.get("rows") != candidate_rows or len(candidate_rows) != 180
            or manifest.get("schema") != "cross_tfg_v3_candidate_only_dev_plan_v1"
            or manifest.get("status") != "FROZEN" or manifest.get("run_id") != CANDIDATE_RUN_ID
            or manifest.get("plan_raw_sha256") != sha(plan_raw)
            or manifest.get("planned_rows_sha256") != sha(rows_raw)
            or manifest.get("planned_rows") != 180
            or manifest.get("manifest_sha256") != CANDIDATE_MANIFEST_SELF_SHA256
            or not self_hash_ok(manifest, "manifest_sha256")):
        raise ValueError("candidate plan/rows/manifest closure is not the pinned 180-row FROZEN contract")
    binding_remote = ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json"
    expected_ref_fields = (
        ("candidate_plan", CANDIDATE_PLAN_REMOTE, plan_raw, {}),
        ("candidate_rows", CANDIDATE_ROWS_REMOTE, rows_raw, {"count": 180}),
        ("candidate_plan_manifest", CANDIDATE_MANIFEST_REMOTE, manifest_raw, {}),
        ("runtime_binding", binding_remote, binding_raw,
         {"gpu_uuid": GPU_UUID, "policy_sha256": POLICY_SHA256}),
    )
    for key, remote, raw, extra in expected_ref_fields:
        ref = receipt.get(key)
        if (not isinstance(ref, dict) or ref.get("path") != remote or ref.get("bytes") != len(raw)
                or ref.get("raw_sha256") != sha(raw)
                or any(ref.get(field) != value for field, value in extra.items())):
            raise ValueError(f"candidate receipt {key} path/bytes/raw-SHA/identity reference mismatch")
    expected_fixed_refs = {
        "method_config": (ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01/v3_method_candidate_config.json",
                          5240, "aba0dc744b76cfd9f73db547faa80508a4a46437432ef78feaf5b7bfb51f3666"),
        "materialization_receipt": (ROOT_REMOTE + "/runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01/materialization_receipt.json",
                                    3965, "c64bfb0d80471c6d2764b6921a993a7c556b9511d4cba283f41eba002cdf8ebd"),
    }
    for key, (remote, size, digest) in expected_fixed_refs.items():
        ref = receipt.get(key)
        if (not isinstance(ref, dict) or ref.get("path") != remote
                or ref.get("bytes") != size or ref.get("raw_sha256") != digest):
            raise ValueError(f"candidate receipt fixed source reference mismatch: {key}")
    metadata = phase_a.get("candidate_metadata_preflight", {})
    metadata_readback = receipt.get("candidate_metadata_preflight_readback")
    if (not isinstance(metadata_readback, dict)
            or metadata_readback != {"path": metadata.get("remote"), "bytes": metadata.get("bytes"),
                                    "raw_sha256": metadata.get("raw_sha256"), "status": "PASS_METADATA_ONLY",
                                    "gpu_uuid": GPU_UUID}):
        raise ValueError("candidate receipt metadata readback differs exactly from Phase A's actual metadata bundle")
    fresh = receipt.get("fresh_candidate_metadata_preflight")
    if (not isinstance(fresh, dict) or fresh.get("path") != expected_fresh
            or type(fresh.get("bytes")) is not int or not 0 < fresh["bytes"] <= 2 * 1024 * 1024
            or not re.fullmatch(r"[0-9a-f]{64}", str(fresh.get("raw_sha256", "")))
            or fresh.get("status") != "PASS_METADATA_ONLY" or fresh.get("gpu_uuid") != GPU_UUID
            or not isinstance(fresh.get("live_resource_probe"), dict)):
        raise ValueError("candidate receipt lacks an actual bounded fresh preflight path/bytes/SHA/GPU/live probe")
    fresh_metadata = {"status": fresh["status"], "candidate_rows": 180,
                      "plan_raw_sha256": sha(plan_raw), "runtime_binding_raw_sha256": sha(binding_raw),
                      "gpu_uuid": fresh["gpu_uuid"], "execution_performed": False,
                      "confirm_audio_read": False, "live_resource_probe": fresh["live_resource_probe"]}
    validate_candidate_metadata(root, fresh_metadata)
    expected_source_pins = _source_pin_from_plan(plan)
    expected_receipt_sources = {
        "candidate13_guard": expected_source_pins["candidate13_guard"],
        "candidate_dispatcher07": expected_source_pins["candidate_dispatcher07"],
        "stagea_core06": expected_source_pins["stagea_core06"],
        "frontend09": expected_source_pins["frontend_adapter09"],
        "strict_bootstrap03": {
            "bytes": STRICT_BOOTSTRAP_BYTES, "raw_sha256": STRICT_BOOTSTRAP_SHA256,
        },
    }
    actual_sources = receipt.get("source_pins")
    if not isinstance(actual_sources, dict) or set(actual_sources) != set(expected_receipt_sources):
        raise ValueError("candidate receipt source_pins roles differ from frozen plan source closure")
    for role, expected in expected_receipt_sources.items():
        actual = actual_sources[role]
        if (not isinstance(actual, dict) or actual.get("bytes") != expected["bytes"]
                or actual.get("raw_sha256") != expected["raw_sha256"]):
            raise ValueError(f"candidate receipt source pin differs from the frozen plan: {role}")
    expected_scope = {"fixed_optimizer_steps": 120, "candidate_rows": 180,
                      "confirm_opened": False, "candidate_selection_performed": False,
                      "weights_tuned_or_changed": False, "api_or_asr_called_by_launcher": False,
                      "models_or_scorers_invoked_by_launcher": False, "automatic_replay": False}
    if receipt.get("scope") != expected_scope:
        raise ValueError("candidate receipt scope differs from the frozen candidate-only launch contract")
    command = receipt.get("command")
    expected_command = [PYTHON, "-B", expected_source_pins["candidate13_guard"]["remote_path"],
                        "--project-root", ROOT_REMOTE,
                        "--candidate-plan", CANDIDATE_PLAN_REMOTE,
                        "--candidate-plan-raw-sha256", sha(plan_raw),
                        "--planned-rows", CANDIDATE_ROWS_REMOTE,
                        "--planned-rows-raw-sha256", sha(rows_raw),
                        "--plan-manifest", CANDIDATE_MANIFEST_REMOTE,
                        "--plan-manifest-raw-sha256", sha(manifest_raw),
                        "--runtime-binding", binding_remote,
                        "--runtime-binding-raw-sha256", sha(binding_raw),
                        "--execute-candidate"]
    if command != expected_command:
        raise ValueError("candidate receipt actual argv differs from the exact pinned execution contract")
    parent = receipt.get("controls_parent")
    if parent != phase_a.get("controls_parent"):
        raise ValueError("candidate receipt controls_parent is not an exact match to Phase A's validated parent summary")


def phase_b(args: argparse.Namespace) -> int:
    root = root_from_script()
    pins = validate_fixed_sources(root)
    request_id = args.request_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", request_id):
        raise ValueError("request id must be a path-safe 1–80 character ASCII token")
    receipt_path = required_file(args.launch_receipt, "candidate launch receipt", root)
    receipt_raw = read_regular(receipt_path, "actual candidate180 detached launch receipt", 4 * 1024 * 1024)
    receipt = json_bytes(receipt_raw, "actual candidate launch receipt")
    actual_receipt_sha = sha(receipt_raw)
    if (args.launch_receipt_bytes != len(receipt_raw)
            or not re.fullmatch(r"[0-9a-f]{64}", args.launch_receipt_raw_sha256)
            or args.launch_receipt_raw_sha256 != actual_receipt_sha):
        raise ValueError("required Phase B receipt byte/SHA inputs must match the actual downloaded receipt")
    receipt_remote = JOB + "/candidate180_" + request_id + ".launch_receipt.json"
    response_path = required_file(args.launch_receipt_response, "candidate launch receipt download response", root)
    response, response_raw = read_json(response_path, "actual launch receipt download response", 1024 * 1024)
    if (response.get("ok") is not True or response.get("remote") != receipt_remote
            or Path(response.get("local", "")).resolve(strict=True) != receipt_path
            or response.get("size") != len(receipt_raw) or response.get("sha256") != actual_receipt_sha):
        raise ValueError("download response does not pin the actual launch receipt path/bytes")
    phase_a_path = required_file(args.phase_a_index, "Phase A request index", root)
    phase_a, phase_a_raw = read_json(phase_a_path, "Phase A request index", 8 * 1024 * 1024)
    if (phase_a.get("schema") != "cross_tfg_candidate180_required_input_dispatch_packet_v1"
            or phase_a.get("phase") != "A_SOURCE_UPLOAD_AND_SINGLE_LAUNCH"
            or phase_a.get("request_id") != request_id
            or phase_a.get("status") != "PREPARED_REQUESTS_ONLY_NOT_DISPATCHED"
            or not self_hash_ok(phase_a, "index_sha256")):
        raise ValueError("Phase B request id/parent source index does not match the actual receipt")
    expected_parent = phase_a.get("controls_parent")
    controls_summary = phase_a.get("controls_terminal_summary", {})
    if (not isinstance(expected_parent, dict) or not isinstance(controls_summary, dict)
            or expected_parent != {
                "run_id": CONTROLS_RUN_ID, "rows": 360,
                "status_counts": {"PASS": 360, "FAILED": 0, "INCOMPLETE": 0},
                "plan_raw_sha256": controls_summary.get("controls_plan", {}).get("raw_sha256"),
                "plan_manifest_raw_sha256": controls_summary.get("controls_plan_manifest", {}).get("raw_sha256"),
                "planned_rows_raw_sha256": controls_summary.get("controls_rows", {}).get("raw_sha256"),
                "results_raw_sha256": controls_summary.get("results_ledger", {}).get("raw_sha256"),
                "result_manifest_raw_sha256": controls_summary.get("result_manifest", {}).get("raw_sha256"),
                "execution_raw_sha256": controls_summary.get("execution_record", {}).get("raw_sha256"),
                "strict_attestation_raw_sha256": controls_summary.get("strict_attestation", {}).get("raw_sha256"),
                "repeat_qc_raw_sha256": controls_summary.get("repeat_qc_ledger", {}).get("raw_sha256"),
                "qualification_manifests": controls_summary.get("qualification_manifests"),
            }):
        raise ValueError("Phase A controls_parent summary is not an exact derivative of its validated terminal pins")
    validate_candidate_receipt(root, receipt, request_id, phase_a)
    status_script = SOURCE_DIR + "/" + SOURCE_PINS["status08"][3]
    command = ("set -euo pipefail; " + remote_check(receipt_remote, len(receipt_raw), sha(receipt_raw)) + "; "
        + remote_check(status_script, SOURCE_PINS["status08"][1], SOURCE_PINS["status08"][2]) + "; PYTHONDONTWRITEBYTECODE=1 "
        + shlex.quote(PYTHON) + " -B " + shlex.quote(status_script) + " --request-id " + shlex.quote(request_id)
        + " --launch-receipt-bytes " + shlex.quote(str(len(receipt_raw)))
        + " --launch-receipt-raw-sha256 " + shlex.quote(actual_receipt_sha))
    out = create_output_dir(root, args.output_dir)
    reqdir = out / "requests"
    reqdir.mkdir(mode=0o700)
    request = {"op": "exec", "timeout": 180, "command": command}
    request_raw = dump_json(request)
    request_path = reqdir / "00_read_candidate180_status_once.json"
    create_exclusive(request_path, request_raw)
    dispatcher_source_evidence = {
        "phase_a_index": {"path": str(phase_a_path), "bytes": len(phase_a_raw), "sha256": sha(phase_a_raw)},
        "launch_receipt": {"path": str(receipt_path), "bytes": len(receipt_raw), "sha256": sha(receipt_raw)},
        "receipt_download_response": {"path": str(response_path), "bytes": len(response_raw), "sha256": sha(response_raw)},
    }
    for role, pin in pins.items():
        if isinstance(pin, dict) and "local" in pin and "bytes" in pin and "raw_sha256" in pin:
            dispatcher_source_evidence[role] = {
                "path": pin["local"], "bytes": pin["bytes"], "sha256": pin["raw_sha256"]}
    index = {"schema": "cross_tfg_candidate180_required_input_dispatch_packet_v1", "phase": "B_READ_ONLY_STATUS_AFTER_ACTUAL_RECEIPT",
        "status": "PREPARED_REQUESTS_ONLY_NOT_DISPATCHED", "request_id": request_id,
        "phase_a_index": {"path": str(phase_a_path), "bytes": len(phase_a_raw), "raw_sha256": sha(phase_a_raw)},
        "launch_receipt": {"path": str(receipt_path), "bytes": len(receipt_raw), "raw_sha256": sha(receipt_raw),
                           "receipt_self_sha256": receipt["receipt_sha256"], "remote": receipt_remote},
        "receipt_download_response": {"path": str(response_path), "bytes": len(response_raw), "raw_sha256": sha(response_raw)},
        "status_source": {"remote": status_script, "local": SOURCE_PINS["status08"][0],
                          "bytes": SOURCE_PINS["status08"][1], "raw_sha256": SOURCE_PINS["status08"][2]},
        "source_evidence": dispatcher_source_evidence,
        "source_and_static_pins": pins,
        "request": {"path": str(request_path), "bytes": len(request_raw), "sha256": sha(request_raw), "raw_sha256": sha(request_raw)},
        "requests": [{"name": request_path.stem, "path": str(request_path), "op": "exec",
                       "bytes": len(request_raw), "sha256": sha(request_raw), "raw_sha256": sha(request_raw)}],
        "execution_policy": {"builder_executes_request": False, "read_only_status_once": True,
                             "automatic_retry": False, "launch_receipt_size_and_sha_are_actual_inputs": True}}
    index["index_sha256"] = sha(canonical(index))
    create_exclusive(out / "request_index_attempt02.json", dump_json(index))
    return 0


def add_control_args(parser: argparse.ArgumentParser) -> None:
    for role, (_, option) in CONTROL_ARTIFACTS.items():
        parser.add_argument(option, dest=role, required=True, help=f"required actual downloaded {role} terminal artifact")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    launch = sub.add_parser("prepare-launch", help="validate actual controls/observer/metadata and write Phase A requests only")
    launch.add_argument("--output-dir", required=True)
    launch.add_argument("--request-id", required=True)
    launch.add_argument("--terminal-inventory", required=True,
                        help="required actual terminal inventory JSON from inventory/chunks attempt02")
    launch.add_argument("--terminal-inventory-exec-response", required=True,
                        help="required clean q02 inventory worker response with actual inventory pins")
    launch.add_argument("--terminal-inventory-download-response", required=True,
                        help="required actual response binding downloaded inventory path/bytes/SHA")
    launch.add_argument("--controls-launch-receipt", required=True,
                        help="required actual controls detached-launch receipt used to bind PID/start ticks")
    launch.add_argument("--observer-manifest", required=True)
    launch.add_argument("--observer-observation", required=True)
    launch.add_argument("--observer-response", required=True)
    launch.add_argument("--metadata-request-index", required=True,
                        help="required actual sealed metadata-only preflight request index")
    launch.add_argument("--metadata-request-index-bytes", required=True, type=int,
                        help="required actual byte length of that downloaded local request index")
    launch.add_argument("--metadata-request-index-raw-sha256", required=True,
                        help="required actual raw SHA-256 of that request index")
    launch.add_argument("--metadata-output", required=True,
                        help="required actual downloaded metadata-only candidate preflight JSON")
    launch.add_argument("--metadata-q00-response", required=True)
    launch.add_argument("--metadata-q01-response", required=True)
    launch.add_argument("--metadata-q02-response", required=True)
    add_control_args(launch)
    launch.set_defaults(func=phase_a)
    status = sub.add_parser("prepare-status", help="write Phase B status request only from an actual downloaded launch receipt")
    status.add_argument("--output-dir", required=True)
    status.add_argument("--request-id", required=True)
    status.add_argument("--launch-receipt", required=True)
    status.add_argument("--launch-receipt-response", required=True)
    status.add_argument("--phase-a-index", required=True)
    status.add_argument("--launch-receipt-bytes", required=True, type=int)
    status.add_argument("--launch-receipt-raw-sha256", required=True)
    status.set_defaults(func=phase_b)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "PACKET_NOT_CREATED", "phase": getattr(locals().get("args"), "phase", None),
                          "error_type": type(exc).__name__, "error": str(exc)[:1000],
                          "automatic_retry": False}, sort_keys=True))
        raise SystemExit(2)
