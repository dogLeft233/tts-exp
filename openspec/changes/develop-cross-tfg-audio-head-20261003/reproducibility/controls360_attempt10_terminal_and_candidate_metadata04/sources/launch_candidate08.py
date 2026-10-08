#!/usr/bin/env python3
"""Validate the completed controls360 parent and fresh candidate metadata, then start frozen candidate180 detached."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path("/root/tts-research-system-20261001")
JOB = ROOT / "cross_tfg_audio_head_20261003/job_control/dev360_candidate180_detached_attempt08"
PREFLIGHT_DIR = ROOT / "cross_tfg_audio_head_20261003/job_control/dev360_candidate13_metadata_preflight_attempt04"
CANDIDATE_PREFLIGHT = PREFLIGHT_DIR / "output/candidate13_metadata_preflight_attempt04.json"
PLAN = ROOT / "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cpu_prepare_attempt01/rebound_plan_attempt12/candidate_rebound_plan_attempt13.json"
ROWS = PLAN.parent / "planned_rows.jsonl"
PLAN_MANIFEST = PLAN.parent / "plan_manifest_metadata_repair_attempt14.json"
MATERIAL_DIR = ROOT / "runs/cross_tfg_audio_head_20261003/v3_step120_candidate_materialization_fit120_attempt11_attempt01"
METHOD_CONFIG = MATERIAL_DIR / "v3_method_candidate_config.json"
MATERIAL_RECEIPT = MATERIAL_DIR / "materialization_receipt.json"
BINDING = ROOT / "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json"
CONTROL_PLAN = ROOT / "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls_plan_attempt09/plan.json"
CONTROL_ROWS = CONTROL_PLAN.parent / "planned_rows.jsonl"
CONTROL_MANIFEST = CONTROL_PLAN.parent / "plan_manifest.json"
CONTROL_ROOT = Path("/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10/controls360")
NAMESPACE = CONTROL_ROOT.parent
CANDIDATE_ROOT = NAMESPACE / "candidate180"
CONTROLS_EXECUTION = CONTROL_ROOT.parent / "controls360.execution.json"
CANDIDATE_RUNNER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt13.py"
CANDIDATE_DISPATCHER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt07.py"
STAGEA_CORE = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py"
FRONTEND_ADAPTER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt09.py"
CONTROLS_RUNNER = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_samehost_dev360_controls_attempt18.py"
CONTROLS_RUNNER_SHA256 = "30381ee8deb6d0e7880379fc91a042e1e2bb3cca375dd05423319259fd9e8756"
CONTROLS_RUNNER_BYTES = 78172
STRICT_BOOTSTRAP = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03/sitecustomize.py"
STRICT_BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
STRICT_BOOTSTRAP_BYTES = 9525
WAV2LIP_PYTHON = Path("/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
RUNTIME_SHA256 = "dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5"
RUNTIME_BYTES = 568289
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
POLICY_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
CANDIDATE_RUN_ID = "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt02"
CONTROLS_RUN_ID = "cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10"
CANDIDATE_PLAN_SHA256 = "8615754443e653fc79203f108fa689205a77e7f6f234930b352204df08b98280"
CANDIDATE_PLAN_BYTES = 932236
CANDIDATE_ROWS_SHA256 = "c9dc6f9bf400aff7f2c25ee24528dfd3a038433c21e80e0177066b8ab3f8a19e"
CANDIDATE_ROWS_BYTES = 606042
CANDIDATE_MANIFEST_SHA256 = "d25985b666c59946a7acd1c948b834ea8628d0b9c8eef0442419818ea86c0192"
CANDIDATE_MANIFEST_BYTES = 2513
CONFIG_SHA256 = "aba0dc744b76cfd9f73db547faa80508a4a46437432ef78feaf5b7bfb51f3666"
CONFIG_BYTES = 5240
MATERIAL_RECEIPT_SHA256 = "c64bfb0d80471c6d2764b6921a993a7c556b9511d4cba283f41eba002cdf8ebd"
MATERIAL_RECEIPT_BYTES = 3965
CONTROL_PLAN_SHA256 = "8a592cc54dc981bec5e2398108e6873190a54c7c1ed2bb7901cd31aad6cc86eb"
CONTROL_PLAN_BYTES = 1781735
CONTROL_ROWS_SHA256 = "45f1d3130c33c37a513db7b22f76d4107bd821f370a7b3ae2d5ae853894b27c8"
CONTROL_ROWS_BYTES = 878184
CONTROL_MANIFEST_SHA256 = "ab07ee9bed69d067390b9ffae597200916838f8cca6bdb7ddf376b5d5febc214"
CONTROL_MANIFEST_BYTES = 1438
STRICT09_MANIFEST_SHA256 = "4053e74430e900037ca2f528e3678ad4375241e8e9c653367ea3de31ca94c86d"
STRICT09_MANIFEST_BYTES = 287148
STRICT12_MANIFEST_SHA256 = "c55b390b6d5189983a4a3e2d181d5ff7606cc191817f125a859800c3e09508d9"
STRICT12_MANIFEST_BYTES = 146401
EXPECTED_MODELS = ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18")
EXPECTED_METHOD = "shared_recipient_transition_head_v3_fit40_step120"
MAX_JSON = 32 * 1024 * 1024
MAX_ROWS = 16 * 1024 * 1024
MAX_PREFLIGHT = 2 * 1024 * 1024
SOURCE_PINS = {
    "candidate13_guard": (CANDIDATE_RUNNER, 35309,
                          "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"),
    "candidate_dispatcher07": (CANDIDATE_DISPATCHER, 12824,
                               "f6844b51183b6cf3b57f6a62d0b1b9bd70c3a0d496480fe8cad70313907c16e3"),
    "stagea_core06": (STAGEA_CORE, 79475,
                      "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"),
    "frontend09": (FRONTEND_ADAPTER, 23249,
                   "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"),
    "strict_bootstrap03": (STRICT_BOOTSTRAP, STRICT_BOOTSTRAP_BYTES, STRICT_BOOTSTRAP_SHA256),
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: dict[str, Any], key: str) -> str:
    return sha_bytes(canonical({name: item for name, item in value.items() if name != key}))


def read_regular(path: Path, label: str, limit: int = MAX_JSON) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular nonsymlink file: {path}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError(f"{label} changed while being read: {path}")
    return raw


def read_pinned(path: Path, label: str, size: int, digest: str, limit: int = MAX_JSON) -> bytes:
    raw = read_regular(path, label, limit)
    if len(raw) != size or sha_bytes(raw) != digest:
        raise ValueError(f"{label} differs from the sealed byte/SHA pin")
    return raw


def write_exclusive(path: Path, raw: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha_bytes(raw)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise ValueError("process stat record is malformed")
    fields = raw[close + 2:].split()
    if len(fields) <= 19:
        raise ValueError("process stat record lacks starttime")
    result = int(fields[19])
    if result <= 0:
        raise ValueError("process start tick is invalid")
    return result


def child_matches(proc: subprocess.Popen[Any], pid: int, ticks: int | None) -> bool:
    if proc.pid != pid or ticks is None:
        return False
    try:
        return proc_start_ticks(pid) == ticks and os.getpgid(pid) == pid
    except (OSError, ValueError, ProcessLookupError):
        return False


def stop_unfinalized(proc: subprocess.Popen[Any], pid: int, ticks: int | None) -> dict[str, Any]:
    if proc.poll() is not None:
        return {"action": "child_already_exited", "returncode": proc.returncode}
    if not child_matches(proc, pid, ticks):
        return {"action": "not_signalled_identity_unverified", "pid": pid, "start_ticks": ticks}
    try:
        os.killpg(pid, signal.SIGTERM)
        try:
            return {"action": "verified_process_group_SIGTERM", "returncode": proc.wait(timeout=8.0)}
        except subprocess.TimeoutExpired:
            if not child_matches(proc, pid, ticks):
                return {"action": "process_identity_changed_after_SIGTERM", "pid": pid, "start_ticks": ticks}
            os.killpg(pid, signal.SIGKILL)
            return {"action": "verified_process_group_SIGKILL_after_timeout", "returncode": proc.wait(timeout=8.0)}
    except (OSError, ProcessLookupError) as exc:
        return {"action": "cleanup_signal_failed", "error_type": type(exc).__name__}


def clean_child_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in tuple(env):
        upper = name.upper()
        if upper.startswith("CROSS_TFG_") or any(token in upper for token in
                ("API_KEY", "TOKEN", "PASSWORD", "PASSWD", "SECRET", "CREDENTIAL")):
            env.pop(name, None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def load_controls18() -> Any:
    raw = read_pinned(CONTROLS_RUNNER, "controls18 native qualification validator",
                      CONTROLS_RUNNER_BYTES, CONTROLS_RUNNER_SHA256, 2 * 1024 * 1024)
    spec = importlib.util.spec_from_file_location("candidate_launch_controls18_validator", CONTROLS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned controls18 qualification validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if sha_bytes(raw) != CONTROLS_RUNNER_SHA256:
        raise ValueError("controls18 source changed while loading")
    return module


def flag_value(argv: Any, flag: str) -> str:
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ValueError("producer command is not a string argv list")
    indices = [index for index, item in enumerate(argv) if item == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError(f"producer command lacks one exact {flag} value")
    return argv[indices[0] + 1]


def verify_static_candidate_inputs() -> tuple[dict[str, Any], bytes, dict[str, Any], bytes,
                                               list[dict[str, Any]], bytes, dict[str, Any], bytes]:
    plan_raw = read_pinned(PLAN, "candidate plan", CANDIDATE_PLAN_BYTES, CANDIDATE_PLAN_SHA256, 4 * 1024 * 1024)
    rows_raw = read_pinned(ROWS, "candidate ordered rows", CANDIDATE_ROWS_BYTES, CANDIDATE_ROWS_SHA256, MAX_ROWS)
    manifest_raw = read_pinned(PLAN_MANIFEST, "candidate plan manifest", CANDIDATE_MANIFEST_BYTES,
                               CANDIDATE_MANIFEST_SHA256, 4 * 1024 * 1024)
    binding_raw = read_pinned(BINDING, "runtime binding", RUNTIME_BYTES, RUNTIME_SHA256, 8 * 1024 * 1024)
    config_raw = read_pinned(METHOD_CONFIG, "step120 method config", CONFIG_BYTES, CONFIG_SHA256, 4 * 1024 * 1024)
    material_raw = read_pinned(MATERIAL_RECEIPT, "step120 materialization receipt", MATERIAL_RECEIPT_BYTES,
                               MATERIAL_RECEIPT_SHA256, 4 * 1024 * 1024)
    plan = json.loads(plan_raw.decode("utf-8"))
    manifest = json.loads(manifest_raw.decode("utf-8"))
    binding = json.loads(binding_raw.decode("utf-8"))
    rows = [json.loads(line.decode("utf-8")) for line in rows_raw.splitlines() if line.strip()]
    config = json.loads(config_raw.decode("utf-8"))
    material = json.loads(material_raw.decode("utf-8"))
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("run_id") != CANDIDATE_RUN_ID or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or plan.get("plan_sha256") != "00a5f585ae6d0f19a005a4efc971d0367dcc115f607305ef6777c11e85f0a306"):
        raise ValueError("candidate plan schema, frozen ID, or self-hash differs")
    if (len(rows) != 180 or rows != plan.get("rows")
            or len({row.get("row_id") for row in rows}) != 180
            or b"".join(canonical(row) + b"\n" for row in rows) != rows_raw):
        raise ValueError("candidate ordered row ledger is not exact canonical 180-row data")
    if (manifest.get("schema") != "cross_tfg_v3_candidate_only_dev_plan_v1"
            or manifest.get("status") != "FROZEN" or manifest.get("run_id") != CANDIDATE_RUN_ID
            or manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or manifest.get("planned_rows_sha256") != sha_bytes(rows_raw)
            or manifest.get("planned_rows") != 180
            or manifest.get("candidate_config_raw_sha256") != CONFIG_SHA256
            or manifest.get("materialization_receipt_raw_sha256") != MATERIAL_RECEIPT_SHA256
            or manifest.get("candidate_checkpoint_sha256")
               != "8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8"
            or manifest.get("candidate_runtime_qualification_performed") is not False
            or manifest.get("samehost_controls_plan_raw_sha256") != CONTROL_PLAN_SHA256
            or manifest.get("samehost_controls_plan_manifest_raw_sha256") != CONTROL_MANIFEST_SHA256
            or manifest.get("samehost_runtime_binding_raw_sha256") != RUNTIME_SHA256
            or manifest.get("method_counts") != {EXPECTED_METHOD: 180}
            or manifest.get("model_counts") != {model_id: 60 for model_id in EXPECTED_MODELS}
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")):
        raise ValueError("candidate plan manifest does not bind the exact plan and rows")
    execution = plan.get("execution", {})
    methods = plan.get("methods", [])
    models = plan.get("models", [])
    if (execution.get("candidate_only") is not True or execution.get("fixed_optimizer_steps") != 120
            or execution.get("confirm_audio_read") is not False or execution.get("confirm_effects_read") is not False
            or execution.get("development_weights_selected") is not False
            or execution.get("candidate_runtime_qualification_performed") is not False
            or execution.get("candidate_config_raw_sha256") != CONFIG_SHA256
            or execution.get("materialization_receipt_raw_sha256") != MATERIAL_RECEIPT_SHA256
            or execution.get("candidate_checkpoint_sha256")
               != "8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8"
            or execution.get("candidate_training_record_sha256")
               != "c2ed183ae5894bd3c42dc56c0242539f1f8290df1230d7b1aa4df8b7584a7e67"
            or execution.get("candidate_selection_record_sha256")
               != "db4ede1f315909b8e8d410d4ab519ee8745035233dc910333a2b409c0a2328fc"
            or execution.get("samehost_runtime_binding", {}).get("raw_sha256") != RUNTIME_SHA256
            or len(methods) != 1 or methods[0].get("method_id") != EXPECTED_METHOD
            or tuple(model.get("model_id") for model in models) != EXPECTED_MODELS):
        raise ValueError("candidate plan does not preserve the fixed step120, closed-Confirm, three-backend scope")
    if (any(row.get("split") != "dev" or row.get("method_id") != EXPECTED_METHOD for row in rows)
            or {model_id: sum(row.get("model_id") == model_id for row in rows) for model_id in EXPECTED_MODELS}
               != {model_id: 60 for model_id in EXPECTED_MODELS}):
        raise ValueError("candidate ledger is not the exact 60-per-model development-only method scope")
    runtime_ref = execution.get("samehost_runtime_binding", {})
    if (runtime_ref.get("path") != str(BINDING) or runtime_ref.get("bytes") != RUNTIME_BYTES
            or runtime_ref.get("raw_sha256") != RUNTIME_SHA256 or runtime_ref.get("gpu_uuid") != GPU_UUID
            or runtime_ref.get("determinism_policy_sha256") != POLICY_SHA256):
        raise ValueError("candidate plan runtime binding reference differs from the exact sealed runtime")
    if (manifest.get("fixed_optimizer_steps") != 120 or manifest.get("confirm_audio_read") is not False
            or manifest.get("confirm_effects_read") is not False
            or manifest.get("development_weights_selected") is not False
            or config.get("schema") != "cross_tfg_v3_enhancement_cli_config_v1"
            or config.get("candidate_provenance", {}).get("checkpoint", {}).get("sha256")
               != "8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8"
            or material.get("schema") != "cross_tfg_v3_step120_method_candidate_materialization_v1"
            or material.get("status") != "MATERIALIZED_FIXED_STEP120_METHOD_CANDIDATE"
            or material.get("candidate_config", {}).get("sha256") != CONFIG_SHA256
            or material.get("candidate_config", {}).get("path") != str(METHOD_CONFIG.relative_to(ROOT))
            or material.get("candidate_provenance", {}).get("checkpoint", {}).get("sha256")
               != "8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8"
            or material.get("optimizer_steps") != 120
            or material.get("model_execution_performed") is not False
            or material.get("development_audio_opened") is not False
            or material.get("confirmation_audio_opened") is not False
            or material.get("method_selected_using_development") is not False):
        raise ValueError("method materialization/config scope or checkpoint pin differs")
    if (binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1" or binding.get("status") != "PASS"
            or binding.get("binding_sha256") != payload_sha(binding, "binding_sha256")
            or binding.get("gpu_uuid") != GPU_UUID or binding.get("determinism_policy_sha256") != POLICY_SHA256
            or binding.get("execution_bindings", {}).get("output_root") != str(NAMESPACE)):
        raise ValueError("runtime binding schema/status/self-hash/GPU/output namespace differs")
    closure = execution.get("samehost_dev360_source_closure", {})
    refs = plan.get("source_reference_plans", {})
    plan_refs = {
        "candidate13_guard": (execution.get("samehost_candidate_guarded_launcher"),
                              refs.get("samehost_candidate_guarded_launcher")),
        "candidate_dispatcher07": (closure.get("step120_candidate_dispatcher"),
                                   refs.get("v3_candidate_dispatcher")),
        "stagea_core06": (closure.get("samehost_stagea_core"), refs.get("stagea_core")),
        "frontend09": (closure.get("samehost_cross_frontend_adapter"),
                       refs.get("cross_frontend_adapter_runner")),
        "strict_bootstrap03": (closure.get("samehost_strict_determinism_bootstrap"),
                               refs.get("samehost_strict_renderer_policy", {}).get("bootstrap_source")),
    }
    for name, (path, expected_bytes, expected_sha) in SOURCE_PINS.items():
        raw = read_pinned(path, f"{name} source", expected_bytes, expected_sha, 2 * 1024 * 1024)
        for ref in plan_refs[name]:
            if (not isinstance(ref, dict) or ref.get("path") != str(path.relative_to(ROOT))
                    or ref.get("bytes") != expected_bytes or ref.get("raw_sha256") != expected_sha):
                raise ValueError(f"{name} source pin differs between source closure and candidate plan")
        if sha_bytes(raw) != expected_sha:
            raise ValueError(f"{name} source changed while being verified")
    return plan, plan_raw, manifest, manifest_raw, rows, rows_raw, binding, binding_raw


def verify_controls_terminal(plan: dict[str, Any], binding: dict[str, Any], binding_raw: bytes) -> dict[str, Any]:
    control_plan_raw = read_pinned(CONTROL_PLAN, "controls360 frozen plan", CONTROL_PLAN_BYTES,
                                   CONTROL_PLAN_SHA256, 8 * 1024 * 1024)
    control_rows_raw = read_pinned(CONTROL_ROWS, "controls360 frozen rows", CONTROL_ROWS_BYTES,
                                   CONTROL_ROWS_SHA256, MAX_ROWS)
    control_manifest_raw = read_pinned(CONTROL_MANIFEST, "controls360 frozen plan manifest", CONTROL_MANIFEST_BYTES,
                                       CONTROL_MANIFEST_SHA256, 4 * 1024 * 1024)
    control_plan = json.loads(control_plan_raw.decode("utf-8"))
    control_rows = [json.loads(line.decode("utf-8")) for line in control_rows_raw.splitlines() if line.strip()]
    plan_manifest = json.loads(control_manifest_raw.decode("utf-8"))
    if (control_plan.get("schema_version") != "cross-tfg-audio-eval-v1" or control_plan.get("status") != "FROZEN"
            or control_plan.get("run_id") != CONTROLS_RUN_ID
            or control_plan.get("plan_sha256") != payload_sha(control_plan, "plan_sha256")
            or len(control_rows) != 360 or control_plan.get("rows") != control_rows
            or b"".join(canonical(row) + b"\n" for row in control_rows) != control_rows_raw
            or plan_manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_plan_v1"
            or plan_manifest.get("status") != "FROZEN"
            or plan_manifest.get("run_id") != CONTROLS_RUN_ID
            or plan_manifest.get("plan_raw_sha256") != sha_bytes(control_plan_raw)
            or plan_manifest.get("planned_rows_sha256") != sha_bytes(control_rows_raw)
            or plan_manifest.get("planned_rows") != 360
            or plan_manifest.get("manifest_sha256") != payload_sha(plan_manifest, "manifest_sha256")):
        raise ValueError("controls360 frozen plan/manifest/row ledger differs from attempt10 pins")
    controls_ref = plan.get("execution", {}).get("baseline_controls", {})
    if controls_ref.get("run_id") != CONTROLS_RUN_ID:
        raise ValueError("candidate plan does not name the exact controls360 parent run")
    if (controls_ref.get("plan_raw_sha256") != CONTROL_PLAN_SHA256
            or controls_ref.get("planned_rows_sha256") != CONTROL_ROWS_SHA256
            or control_plan.get("models") != plan.get("models")):
        raise ValueError("candidate baseline-controls lineage differs from exact parent model/plan pins")

    results_path = CONTROL_ROOT / "results.jsonl"
    manifest_path = CONTROL_ROOT / "result_manifest.json"
    status_path = CONTROL_ROOT / "control/run_status.json"
    repeat_path = CONTROL_ROOT / "generation_repeat_qc.jsonl"
    attestation_path = CONTROL_ROOT / "runtime_determinism_attestation.json"
    result_raw = read_regular(results_path, "completed controls360 results", 128 * 1024 * 1024)
    result_manifest_raw = read_regular(manifest_path, "completed controls360 result manifest", 8 * 1024 * 1024)
    status_raw = read_regular(status_path, "completed controls360 run status", 2 * 1024 * 1024)
    repeat_raw = read_regular(repeat_path, "controls360 frozen empty repeat artifact", 1024 * 1024)
    attestation_raw = read_regular(attestation_path, "controls360 strict renderer attestation", 16 * 1024 * 1024)
    execution_raw = read_regular(CONTROLS_EXECUTION, "controls360 execution record", 8 * 1024 * 1024)
    results = [json.loads(line.decode("utf-8")) for line in result_raw.splitlines() if line.strip()]
    result_manifest = json.loads(result_manifest_raw.decode("utf-8"))
    run_status = json.loads(status_raw.decode("utf-8"))
    execution_record = json.loads(execution_raw.decode("utf-8"))
    attestation = json.loads(attestation_raw.decode("utf-8"))
    counts = {status: sum(row.get("status") == status for row in results)
              for status in ("PASS", "FAILED", "INCOMPLETE")}
    per_model = {model_id: sum(row.get("model_id") == model_id for row in results) for model_id in EXPECTED_MODELS}
    if (len(results) != 360 or [row.get("row_id") for row in results] != [row.get("row_id") for row in control_rows]
            or len({row.get("row_id") for row in results}) != 360
            or b"".join(canonical(row) + b"\n" for row in results) != result_raw
            or any(row.get("split") != "dev" or row.get("status") != "PASS" for row in results)
            or any(any(row.get(key) != planned.get(key) for key in ("row_id", "sample_id", "speaker_id", "model_id", "seed", "method_id"))
                   for planned, row in zip(control_rows, results))
            or counts != {"PASS": 360, "FAILED": 0, "INCOMPLETE": 0}
            or per_model != {model_id: 120 for model_id in EXPECTED_MODELS}):
        raise ValueError("controls360 results are not the exact ordered all-PASS 360-row parent ledger")
    if (result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1"
            or result_manifest.get("run_id") != CONTROLS_RUN_ID
            or result_manifest.get("planned_rows") != 360 or result_manifest.get("result_rows") != 360
            or result_manifest.get("plan_raw_sha256") != sha_bytes(control_plan_raw)
            or result_manifest.get("results_jsonl_sha256") != sha_bytes(result_raw)
            or result_manifest.get("status_counts") != counts or result_manifest.get("status") != "PASS"
            or result_manifest.get("failure_reasons") != []
            or result_manifest.get("repeat_qc_status") != "NOT_RUN_SUBRUN"
            or result_manifest.get("generation_repeat_qc_rows") != 0
            or result_manifest.get("generation_repeat_qc_jsonl_sha256") != sha_bytes(b"")
            or repeat_raw != b"" or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")):
        raise ValueError("controls360 result manifest or preserved NOT_RUN_SUBRUN artifact differs")
    if (run_status.get("schema_version") != "cross_tfg_audio_head_stagea_v1"
            or run_status.get("status") != "COMPLETE" or run_status.get("run_id") != CONTROLS_RUN_ID
            or run_status.get("plan_path") != str(CONTROL_PLAN)
            or run_status.get("plan_raw_sha256") != sha_bytes(control_plan_raw)
            or run_status.get("planned_rows_path") != str(CONTROL_ROWS)
            or run_status.get("planned_rows_sha256") != sha_bytes(control_rows_raw)
            or run_status.get("completed_rows") != 360 or run_status.get("planned_rows") != 360
            or run_status.get("status_counts") != counts or run_status.get("repeat_qc_status") != "NOT_RUN_SUBRUN"
            or run_status.get("result_manifest_path") != str(manifest_path)
            or run_status.get("result_manifest_sha256") != sha_bytes(result_manifest_raw)
            or not run_status.get("ended_unix")):
        raise ValueError("controls360 run_status does not attest the exact completed parent ledger")
    command = execution_record.get("execution_command")
    if (execution_record.get("schema") != "cross_tfg_v3_samehost_dev360_preflight_v1"
            or execution_record.get("status") != "PASS_METADATA_ONLY"
            or execution_record.get("plan_raw_sha256") != sha_bytes(control_plan_raw)
            or execution_record.get("plan_manifest_raw_sha256") != sha_bytes(control_manifest_raw)
            or execution_record.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)
            or execution_record.get("execution_output_root") != str(CONTROL_ROOT)
            or execution_record.get("child_returncode") != 0
            or execution_record.get("confirm_audio_opened") is not False
            or execution_record.get("development_audio_opened") is not False
            or execution_record.get("generated_media_opened") is not False
            or execution_record.get("model_execution_performed") is not False
            or control_plan.get("execution", {}).get("confirm_audio_read") is not False
            or control_plan.get("execution", {}).get("confirm_effects_read") is not False
            or flag_value(command, "--plan") != str(CONTROL_PLAN)
            or flag_value(command, "--planned-rows") != str(CONTROL_ROWS)
            or flag_value(command, "--expected-plan-raw-sha256") != sha_bytes(control_plan_raw)
            or flag_value(command, "--output-root") != str(CONTROL_ROOT)):
        raise ValueError("controls360 PASS_METADATA_ONLY execution record does not bind the completed run and actual inner argv")
    attestation_ref = execution_record.get("strict_determinism_attestation")
    if (not isinstance(attestation_ref, dict)
            or attestation_ref.get("path") != str(attestation_path)
            or attestation_ref.get("bytes") != len(attestation_raw)
            or attestation_ref.get("raw_sha256") != sha_bytes(attestation_raw)
            or attestation_ref.get("status") != "PASS"
            or attestation_ref.get("attestation_sha256") != attestation.get("attestation_sha256")
            or attestation.get("schema") != "cross_tfg_v3_samehost_renderer_determinism_attestation_v1"
            or attestation.get("status") != "PASS"
            or attestation.get("plan_raw_sha256") != sha_bytes(control_plan_raw)
            or attestation.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)
            or attestation.get("gpu_uuid") != GPU_UUID
            or attestation.get("row_count") != 360
            or attestation.get("native_renderer_row_receipt_count") != 360
            or attestation.get("confirm_audio_read") is not False
            or attestation.get("confirm_data_opened") is not False
            or attestation.get("qualified_policy_reference_sha256") != POLICY_SHA256
            or attestation.get("strict_bootstrap_source_sha256") != STRICT_BOOTSTRAP_SHA256
            or attestation.get("attestation_sha256") != payload_sha(attestation, "attestation_sha256")):
        raise ValueError("controls360 strict renderer attestation is not the exact successful 360-row proof")
    validator = load_controls18()
    verified_qualifications = validator.validate_backend_determinism_qualifications(ROOT, binding)
    expected_qualifications = {
        "wav2lip_gan_bac9a81e": (STRICT12_MANIFEST_BYTES, STRICT12_MANIFEST_SHA256),
        "musetalk_fixed_portrait_attempt16": (STRICT09_MANIFEST_BYTES, STRICT09_MANIFEST_SHA256),
        "ditto_hubert_fixed_portrait_attempt18": (STRICT09_MANIFEST_BYTES, STRICT09_MANIFEST_SHA256),
    }
    if set(verified_qualifications) != set(EXPECTED_MODELS):
        raise ValueError("controls18 validator did not verify exactly the three native backend qualifications")
    for model_id, (expected_bytes, expected_sha) in expected_qualifications.items():
        ref = verified_qualifications[model_id].get("manifest", {})
        if ref.get("bytes") != expected_bytes or ref.get("sha256") != expected_sha:
            raise ValueError(f"native backend qualification manifest differs from its actual strict receipt: {model_id}")
    return {
        "run_id": CONTROLS_RUN_ID, "rows": 360, "status_counts": counts,
        "plan_raw_sha256": sha_bytes(control_plan_raw), "plan_manifest_raw_sha256": sha_bytes(control_manifest_raw),
        "planned_rows_raw_sha256": sha_bytes(control_rows_raw), "results_raw_sha256": sha_bytes(result_raw),
        "result_manifest_raw_sha256": sha_bytes(result_manifest_raw), "execution_raw_sha256": sha_bytes(execution_raw),
        "strict_attestation_raw_sha256": sha_bytes(attestation_raw), "repeat_qc_raw_sha256": sha_bytes(repeat_raw),
        "qualification_manifests": {key: value[1] for key, value in expected_qualifications.items()},
    }


def verify_metadata(path: Path, raw: bytes, expected_plan_sha: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if (value.get("status") != "PASS_METADATA_ONLY" or value.get("candidate_rows") != 180
            or value.get("plan_raw_sha256") != expected_plan_sha
            or value.get("runtime_binding_raw_sha256") != RUNTIME_SHA256
            or value.get("gpu_uuid") != GPU_UUID or value.get("execution_performed") is not False
            or value.get("confirm_audio_read") is not False
            or not isinstance(value.get("live_resource_probe"), dict)):
        raise ValueError(f"{path.name} is not a pinned PASS_METADATA_ONLY candidate preflight")
    return value


def candidate_cli_args(include_execute: bool) -> list[str]:
    argv = [str(CANDIDATE_RUNNER), "--project-root", str(ROOT),
            "--candidate-plan", str(PLAN), "--candidate-plan-raw-sha256", CANDIDATE_PLAN_SHA256,
            "--planned-rows", str(ROWS), "--planned-rows-raw-sha256", CANDIDATE_ROWS_SHA256,
            "--plan-manifest", str(PLAN_MANIFEST), "--plan-manifest-raw-sha256", CANDIDATE_MANIFEST_SHA256,
            "--runtime-binding", str(BINDING), "--runtime-binding-raw-sha256", RUNTIME_SHA256]
    if include_execute:
        argv.append("--execute-candidate")
    return argv


def assert_candidate_namespace(expected_controls_execution_sha256: str) -> None:
    if NAMESPACE.is_symlink() or not NAMESPACE.is_dir():
        raise FileExistsError("candidate180 output namespace is missing or unsafe")
    children = set(NAMESPACE.iterdir())
    allowed = {CONTROL_ROOT, CONTROLS_EXECUTION}
    if children != allowed:
        raise FileExistsError("candidate180 sibling namespace must contain only controls360 and its execution record")
    if CONTROL_ROOT.is_symlink() or not CONTROL_ROOT.is_dir():
        raise FileExistsError("completed controls360 sibling is missing or unsafe")
    execution_raw = read_regular(CONTROLS_EXECUTION, "completed controls360 execution record", 8 * 1024 * 1024)
    if sha_bytes(execution_raw) != expected_controls_execution_sha256:
        raise ValueError("controls360 execution record changed after completed-parent validation")
    for path in (CANDIDATE_ROOT, CANDIDATE_ROOT.parent / "candidate180.execution.json",
                 CANDIDATE_ROOT.parent / "candidate180.candidate_startup_policy_receipt.json"):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"candidate180 create-only output already exists: {path}")


def metadata_command() -> list[str]:
    return [str(WAV2LIP_PYTHON), "-B", *candidate_cli_args(False)]


def write_prestart_failure(request_id: str, error: BaseException, path: Path) -> int:
    failure = {
        "schema": "cross_tfg_candidate180_detached_launch_failure_v1",
        "status": "FAILED_BEFORE_START", "request_id": request_id,
        "run_id": CANDIDATE_RUN_ID, "output_root": str(CANDIDATE_ROOT),
        "failure_type": type(error).__name__, "failure": str(error)[:1200],
        "canonical_launch_receipt_absent": not (JOB / f"candidate180_{request_id}.launch_receipt.json").exists(),
        "child_started": False, "automatic_replay": False,
    }
    failure["failure_sha256"] = payload_sha(failure, "failure_sha256")
    raw = (json.dumps(failure, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    try:
        write_exclusive(path, raw)
        fsync_dir(JOB)
    except Exception as write_error:
        print(json.dumps({"status": "FAILED_BEFORE_START_FAILURE_RECEIPT_WRITE_FAILED",
                          "failure_type": type(error).__name__, "receipt_error": type(write_error).__name__,
                          "automatic_replay": False}, sort_keys=True))
        return 3
    print(json.dumps({"status": failure["status"], "failure_type": failure["failure_type"],
                      "failure_receipt_path": str(path), "child_started": False,
                      "automatic_replay": False}, sort_keys=True))
    return 3


def fresh_metadata_preflight(plan_raw_sha: str, binding_raw: bytes) -> tuple[bytes, dict[str, Any]]:
    env = clean_child_env()
    binding = json.loads(binding_raw.decode("utf-8"))
    env["CUDA_VISIBLE_DEVICES"] = str(binding.get("cuda_visible_devices", "0"))
    completed = subprocess.run(metadata_command(), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=1800)
    stdout = completed.stdout
    stderr = completed.stderr
    if len(stdout) > MAX_PREFLIGHT or len(stderr) > MAX_PREFLIGHT:
        raise ValueError("fresh candidate metadata preflight output exceeded its bounded limit")
    if completed.returncode != 0:
        raise RuntimeError(f"fresh candidate13 metadata preflight failed with rc={completed.returncode}; stderr_bytes={len(stderr)}")
    fresh = verify_metadata(Path("fresh_candidate13_metadata_preflight.json"), stdout, plan_raw_sha)
    return stdout, fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--candidate-preflight-bytes", required=True, type=int)
    parser.add_argument("--candidate-preflight-raw-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.request_id):
        raise ValueError("request_id must be a short path-safe identifier")
    if args.candidate_preflight_bytes <= 0 or args.candidate_preflight_bytes > MAX_PREFLIGHT:
        raise ValueError("candidate preflight byte pin is outside the source bound")
    if not re.fullmatch(r"[0-9a-f]{64}", args.candidate_preflight_raw_sha256):
        raise ValueError("candidate preflight raw SHA must be 64 lowercase hex characters")
    if ROOT.resolve(strict=True) != Path("/root/tts-research-system-20261001"):
        raise ValueError("project root differs from the frozen checkout")
    if not JOB.is_dir() or JOB.is_symlink():
        raise FileNotFoundError("fresh candidate180 detached control root must be created separately and be a real directory")
    if any(JOB.iterdir()):
        raise FileExistsError("fresh candidate180 detached control root must be empty")

    plan, plan_raw, candidate_manifest, candidate_manifest_raw, rows, rows_raw, binding, binding_raw = verify_static_candidate_inputs()
    if (not WAV2LIP_PYTHON.is_file() or WAV2LIP_PYTHON.is_symlink()
            or sha_bytes(read_pinned(STRICT_BOOTSTRAP, "strict startup bootstrap", STRICT_BOOTSTRAP_BYTES,
                                     STRICT_BOOTSTRAP_SHA256, 1024 * 1024)) != STRICT_BOOTSTRAP_SHA256):
        raise ValueError("candidate Python/bootstrap identity is missing or differs from the pinned strict environment")
    control = verify_controls_terminal(plan, binding, binding_raw)
    preflight_raw = read_pinned(CANDIDATE_PREFLIGHT, "actual downloaded candidate13 metadata preflight",
                                args.candidate_preflight_bytes, args.candidate_preflight_raw_sha256, MAX_PREFLIGHT)
    preflight = verify_metadata(CANDIDATE_PREFLIGHT, preflight_raw, sha_bytes(plan_raw))
    assert_candidate_namespace(control["execution_raw_sha256"])
    fresh_preflight_raw, fresh_preflight = fresh_metadata_preflight(sha_bytes(plan_raw), binding_raw)
    assert_candidate_namespace(control["execution_raw_sha256"])
    # The parent gate is repeated after live resource validation to ensure its terminal files remained identical.
    control_after_preflight = verify_controls_terminal(plan, binding, binding_raw)
    if control_after_preflight != control:
        raise ValueError("controls360 terminal lineage changed across fresh candidate capacity/GPU preflight")

    control_files = ("stdout.log", "stderr.log", "launch_receipt.json", "launch_receipt.pending.json",
                     "launch_receipt.unqualified.json", "launch_failure.json", "fresh_metadata_preflight.json")
    paths = {name: JOB / f"candidate180_{args.request_id}.{name}" for name in control_files}
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing candidate180 detached artifact: {path}")
    # Store and verify the exact pre-spawn metadata bytes before the GPU child can
    # start; the detached receipt will never point at a post-spawn best-effort file.
    fresh_metadata_sha = write_exclusive(paths["fresh_metadata_preflight.json"], fresh_preflight_raw)
    if fresh_metadata_sha != sha_bytes(fresh_preflight_raw):
        raise OSError("fresh metadata preflight write returned a mismatched digest")
    command = [str(WAV2LIP_PYTHON), "-B", *candidate_cli_args(True)]
    env = clean_child_env()
    env["CUDA_VISIBLE_DEVICES"] = str(binding.get("cuda_visible_devices", "0"))
    proc: subprocess.Popen[Any] | None = None
    try:
        with ExitStack() as stack:
            stdout_stream = stack.enter_context(os.fdopen(
                os.open(paths["stdout.log"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb", buffering=0))
            stderr_stream = stack.enter_context(os.fdopen(
                os.open(paths["stderr.log"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb", buffering=0))
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stdout_stream, stderr=stderr_stream, start_new_session=True,
                                    close_fds=True)
    except BaseException as exc:
        return write_prestart_failure(args.request_id, exc, paths["launch_failure.json"])
    if proc is None:
        return write_prestart_failure(args.request_id, RuntimeError("Popen did not return a process"),
                                      paths["launch_failure.json"])
    pid = int(proc.pid)
    start_ticks: int | None = None
    try:
        if pid <= 0:
            raise RuntimeError("candidate runner process has a nonpositive PID")
        start_ticks = proc_start_ticks(pid)
        if os.getpgid(pid) != pid:
            raise RuntimeError("candidate runner was not detached in its own process group")
        receipt = {
            "schema": "cross_tfg_candidate180_detached_launch_receipt_v1",
            "status": "STARTED_DETACHED", "stage_id": "candidate180", "request_id": args.request_id,
            "run_id": CANDIDATE_RUN_ID, "controls_run_id": CONTROLS_RUN_ID,
            "output_root": str(CANDIDATE_ROOT), "command": command,
            "pid": pid, "start_ticks": start_ticks,
            "stdout_path": str(paths["stdout.log"]), "stderr_path": str(paths["stderr.log"]),
            "launch_receipt_path": str(paths["launch_receipt.json"]),
            "candidate_plan": {"path": str(PLAN), "bytes": len(plan_raw), "raw_sha256": sha_bytes(plan_raw)},
            "candidate_rows": {"path": str(ROWS), "bytes": len(rows_raw), "raw_sha256": sha_bytes(rows_raw), "count": 180},
            "candidate_plan_manifest": {"path": str(PLAN_MANIFEST), "bytes": len(candidate_manifest_raw),
                                        "raw_sha256": sha_bytes(candidate_manifest_raw)},
            "method_config": {"path": str(METHOD_CONFIG), "bytes": CONFIG_BYTES, "raw_sha256": CONFIG_SHA256},
            "materialization_receipt": {"path": str(MATERIAL_RECEIPT), "bytes": MATERIAL_RECEIPT_BYTES,
                                         "raw_sha256": MATERIAL_RECEIPT_SHA256},
            "runtime_binding": {"path": str(BINDING), "bytes": len(binding_raw), "raw_sha256": sha_bytes(binding_raw),
                                "gpu_uuid": GPU_UUID, "policy_sha256": POLICY_SHA256},
            "controls_parent": control,
            "candidate_metadata_preflight_readback": {"path": str(CANDIDATE_PREFLIGHT), "bytes": len(preflight_raw),
                                                       "raw_sha256": sha_bytes(preflight_raw),
                                                       "status": preflight["status"],
                                                       "gpu_uuid": preflight["gpu_uuid"]},
            "fresh_candidate_metadata_preflight": {"path": str(paths["fresh_metadata_preflight.json"]),
                                                   "bytes": len(fresh_preflight_raw),
                                                   "raw_sha256": sha_bytes(fresh_preflight_raw),
                                                   "status": fresh_preflight["status"],
                                                   "gpu_uuid": fresh_preflight["gpu_uuid"],
                                                   "live_resource_probe": fresh_preflight["live_resource_probe"]},
            "source_pins": {"candidate13_guard": {"bytes": 35309,
                             "raw_sha256": "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"},
                            "candidate_dispatcher07": {"bytes": 12824,
                             "raw_sha256": "f6844b51183b6cf3b57f6a62d0b1b9bd70c3a0d496480fe8cad70313907c16e3"},
                            "stagea_core06": {"bytes": 79475,
                             "raw_sha256": "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"},
                            "frontend09": {"bytes": 23249,
                             "raw_sha256": "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"},
                            "strict_bootstrap03": {"bytes": STRICT_BOOTSTRAP_BYTES,
                             "raw_sha256": STRICT_BOOTSTRAP_SHA256}},
            "scope": {"fixed_optimizer_steps": 120, "candidate_rows": 180, "confirm_opened": False,
                      "candidate_selection_performed": False, "weights_tuned_or_changed": False,
                      "api_or_asr_called_by_launcher": False, "models_or_scorers_invoked_by_launcher": False,
                      "automatic_replay": False},
        }
        receipt["receipt_sha256"] = payload_sha(receipt, "receipt_sha256")
        receipt_raw = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        write_exclusive(paths["launch_receipt.pending.json"], receipt_raw)
        if read_regular(paths["launch_receipt.pending.json"], "pending candidate launch receipt") != receipt_raw:
            raise OSError("candidate launch receipt changed before publication")
        os.link(paths["launch_receipt.pending.json"], paths["launch_receipt.json"])
        fsync_dir(JOB)
        paths["launch_receipt.pending.json"].unlink()
        fsync_dir(JOB)
        print(json.dumps({"status": receipt["status"], "stage_id": receipt["stage_id"],
                          "run_id": receipt["run_id"], "controls_run_id": receipt["controls_run_id"],
                          "pid": pid, "start_ticks": start_ticks, "output_root": str(CANDIDATE_ROOT),
                          "launch_receipt_path": str(paths["launch_receipt.json"]),
                          "receipt_bytes": len(receipt_raw), "receipt_raw_sha256": sha_bytes(receipt_raw),
                          "receipt_sha256": receipt["receipt_sha256"],
                          "candidate_plan_raw_sha256": sha_bytes(plan_raw),
                          "controls_results_raw_sha256": control["results_raw_sha256"]},
                         sort_keys=True, ensure_ascii=False))
        return 0
    except BaseException as exc:
        cleanup = stop_unfinalized(proc, pid, start_ticks)
        for path, target in (
            (paths["launch_receipt.json"], paths["launch_receipt.unqualified.json"]),
            (paths["launch_receipt.pending.json"], JOB / f"candidate180_{args.request_id}.launch_receipt_pending.unqualified.json"),
        ):
            if path.exists() or path.is_symlink():
                if target.exists() or target.is_symlink():
                    continue
                os.replace(path, target)
        failure = {"schema": "cross_tfg_candidate180_detached_launch_failure_v1",
                   "status": "POSTSPAWN_FINALIZATION_FAILED", "request_id": args.request_id,
                   "run_id": CANDIDATE_RUN_ID, "pid": pid, "start_ticks": start_ticks,
                   "output_root": str(CANDIDATE_ROOT), "failure_type": type(exc).__name__,
                   "failure": str(exc)[:1200], "bounded_child_cleanup": cleanup,
                   "canonical_launch_receipt_absent": not paths["launch_receipt.json"].exists(),
                   "automatic_replay": False}
        failure["failure_sha256"] = payload_sha(failure, "failure_sha256")
        failure_raw = (json.dumps(failure, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        try:
            write_exclusive(paths["launch_failure.json"], failure_raw)
        except Exception:
            pass
        print(json.dumps({"status": failure["status"], "failure_type": failure["failure_type"],
                          "pid": pid, "failure_receipt_path": str(paths["launch_failure.json"]),
                          "canonical_launch_receipt_absent": failure["canonical_launch_receipt_absent"],
                          "automatic_replay": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAILED_BEFORE_LAUNCH", "error_type": type(exc).__name__,
                          "error": str(exc)[:800], "automatic_replay": False}, sort_keys=True))
        raise SystemExit(3)
