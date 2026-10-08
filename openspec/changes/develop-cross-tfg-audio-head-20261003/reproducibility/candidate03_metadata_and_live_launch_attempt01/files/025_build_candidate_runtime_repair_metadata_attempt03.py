#!/usr/bin/env python3
"""Build a CPU-preflight-gated candidate metadata revision; no media or runtime work."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
PLAN13 = ROOT / ".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/candidate_rebound_plan_attempt13.json"
MANIFEST14 = ROOT / ".codex/scratch/dev360_candidate_generator_pin_repair_attempt02/plan_manifest_metadata_repair_attempt14.json"
RUNTIME09 = ROOT / ".codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/runtime_binding_attempt09.json"
CPU03 = ROOT / ".codex/scratch/candidate_v3_cpu_preflight_attempt03/downloads_attempt02_revision02/cpu_preflight_receipt_attempt03.json"
CONFIG03 = ROOT / ".codex/scratch/candidate_v3_interop_thread_repair_attempt03/v3_method_candidate_config_attempt03.json"
GUARD15 = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt15.py"
OUT = ROOT / ".codex/scratch/candidate_v3_cli03_runtime_repair_attempt03/metadata_output_attempt03"

RUN_ID = "cross_tfg_audio_head_20261003_v3_fit40_dev180_cli03_runtime_repair_attempt03"
METHOD_ID = "shared_recipient_transition_head_v3_fit40_step120"
MODELS = ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18")
OUTPUT_NAMESPACE = "/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/candidate180_cli03_runtime_repair_attempt03"
OUTPUT_LEAF = OUTPUT_NAMESPACE + "/candidate180"
REMOTE_BASE = "/root/tts-research-system-20261001"
RUN_REL = "runs/cross_tfg_audio_head_20261003/candidate180_cli03_runtime_repair_attempt03"
PLAN13_RAW = "8615754443e653fc79203f108fa689205a77e7f6f234930b352204df08b98280"
PLAN13_SELF = "fee2718c0c4b18e5acabf3ca7d7103344bfb3859f3d8a3b93862c1ad4ce7d612"
MANIFEST14_RAW = "d25985b666c59946a7acd1c948b834ea8628d0b9c8eef0442419818ea86c0192"
MANIFEST14_SELF = "420d73cd5fb95692ded1c8bbd8cd82363ccb40169a064f21121be70f041909e5"
RUNTIME09_RAW = "dd07fa3d2b3b5940df25fd3521313f7b2c3f1c6b99494b78a688e1ce39e527c5"
CPU03_RAW = "8c44b1c80a08e48465f25672ef6b990002653ff7d250ede7369be13dab3a3b09"
CPU03_SELF = "66add979541447ec5f8a525576196fd34d9d25306a1c0126fa7957988e3d0486"
CPU_EXEC_REQUEST = ROOT / ".codex/scratch/candidate_v3_cpu_preflight_attempt03/requests_attempt02_revision02/q_run_strict_cpu_preflight_revision02.json"
CPU_EXEC_RESPONSE = ROOT / ".codex/scratch/candidate_v3_cpu_preflight_attempt03/requests_attempt02_revision02/q_run_strict_cpu_preflight_revision02.json.response.json"
CPU_DOWNLOAD_REQUEST = ROOT / ".codex/scratch/candidate_v3_cpu_preflight_attempt03/requests_attempt02_revision02/q_download_cpu_preflight_receipt.json"
CPU_DOWNLOAD_RESPONSE = ROOT / ".codex/scratch/candidate_v3_cpu_preflight_attempt03/requests_attempt02_revision02/q_download_cpu_preflight_receipt.json.response.json"
CPU_REMOTE_PATH = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/candidate_v3_cpu_preflight_attempt03/output/cpu_preflight_receipt_attempt03.json"
CONFIG03_RAW = "fd2ce3509c457e40e618818828f6040ab65641315d40efe8f48e5dc3c5c3d1df"
OLD_CONFIG_RAW = "aba0dc744b76cfd9f73db547faa80508a4a46437432ef78feaf5b7bfb51f3666"
OLD_CLI_RAW = "7a560332ce86a82ffede4254b015cabf5f10176e6f601bfc172a331f7bb885e6"
OLD_GUARD_RAW = "8c3b45f73d420b1dac6db0d5676eaf3343324cbb582d17cad7a3868d27e42708"
GUARD_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_step120_candidate_samehost_attempt15.py"
REPAIR_SCHEMA = "cross_tfg_v3_candidate_runtime_repair_binding_v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha(canonical({key: item for key, item in value.items() if key != field}))


def read_regular(path: Path, label: str, cap: int) -> bytes:
    path = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} crosses a symlink/junction: {cursor}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
        raise ValueError(f"{label} is not a bounded regular file")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return raw


def read_json(path: Path, label: str, expected: str | None = None,
              cap: int = 16 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label, cap)
    if expected is not None and sha(raw) != expected:
        raise ValueError(f"{label} raw SHA mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, raw


def row_context_sha(plan: Mapping[str, Any]) -> str:
    values = []
    for source in plan.get("rows", []):
        row = copy.deepcopy(source)
        row.pop("method_config_sha256", None)
        row.get("generation_settings", {}).pop("v3_cli_config_sha256", None)
        values.append(row)
    return sha(canonical(values))


def method_context_sha(plan: Mapping[str, Any]) -> str:
    methods = copy.deepcopy(plan.get("methods", []))
    for method in methods:
        method.pop("config_sha256", None)
        method.get("config", {}).pop("v3_cli_config_path", None)
        method.get("config", {}).pop("v3_cli_config_sha256", None)
    return sha(canonical(methods))


def verify_cpu_transport(cpu_raw: bytes) -> dict[str, Any]:
    exec_req, exec_req_raw = read_json(CPU_EXEC_REQUEST, "actual CPU03 execution request", cap=1024 * 1024)
    exec_resp, exec_resp_raw = read_json(CPU_EXEC_RESPONSE, "actual CPU03 execution response", cap=1024 * 1024)
    download_req, download_req_raw = read_json(CPU_DOWNLOAD_REQUEST, "actual CPU03 download request", cap=1024 * 1024)
    download_resp, download_resp_raw = read_json(CPU_DOWNLOAD_RESPONSE, "actual CPU03 download response", cap=1024 * 1024)
    stdout = exec_resp.get("stdout")
    try:
        exec_summary = json.loads(stdout) if isinstance(stdout, str) else None
    except json.JSONDecodeError as exc:
        raise ValueError("actual CPU03 execution stdout is not one JSON summary") from exc
    if (exec_req.get("op") != "exec" or exec_req.get("timeout") != 1800
            or not isinstance(exec_req.get("command"), str)
            or f"--receipt {CPU_REMOTE_PATH}" not in exec_req["command"]
            or exec_resp.get("ok") is not True or exec_resp.get("code") != 0
            or exec_resp.get("stderr") != "" or exec_resp.get("timed_out") is not False
            or exec_resp.get("incomplete_stream") is not False or exec_resp.get("stream_error") is not None
            or exec_resp.get("truncated") != {"stdout": False, "stderr": False}
            or not isinstance(exec_summary, dict)
            or exec_summary.get("schema") != "cross_tfg_v3_candidate_cpu_preflight_v1"
            or exec_summary.get("status") != "PASS_CPU_PREFLIGHT"
            or exec_summary.get("receipt_path") != CPU_REMOTE_PATH
            or exec_summary.get("receipt_bytes") != len(cpu_raw)
            or exec_summary.get("receipt_sha256") != CPU03_RAW
            or exec_summary.get("receipt_self_sha256") != CPU03_SELF):
        raise ValueError("actual CPU03 execution request/response does not cleanly attest the pinned receipt path and bytes")
    if (download_req.get("op") != "download" or download_req.get("remote") != CPU_REMOTE_PATH
            or download_req.get("local") != str(CPU03)
            or download_req.get("timeout") != 60
            or download_resp.get("ok") is not True
            or download_resp.get("remote") != CPU_REMOTE_PATH
            or download_resp.get("local") != str(CPU03)
            or download_resp.get("size") != len(cpu_raw)
            or download_resp.get("sha256") != CPU03_RAW
            or len(cpu_raw) != 45_226 or sha(cpu_raw) != CPU03_RAW):
        raise ValueError("actual CPU03 download response does not bind the pinned remote receipt to the reviewed local bytes")
    def ref(path: Path, raw: bytes) -> dict[str, Any]:
        return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "raw_sha256": sha(raw)}
    return {"schema": "cross_tfg_v3_candidate_cpu_preflight_transport_evidence_v1",
            "status": "PASS_ACTUAL_CLEAN_EXECUTION_AND_DOWNLOAD_ENVELOPES",
            "remote_receipt_path": CPU_REMOTE_PATH,
            "execution_request": ref(CPU_EXEC_REQUEST, exec_req_raw),
            "execution_response": ref(CPU_EXEC_RESPONSE, exec_resp_raw),
            "download_request": ref(CPU_DOWNLOAD_REQUEST, download_req_raw),
            "download_response": ref(CPU_DOWNLOAD_RESPONSE, download_resp_raw),
            "execution_envelope_clean": True, "download_envelope_matches_local_bytes": True}


def check_cpu_receipt(plan: Mapping[str, Any], cpu: Mapping[str, Any],
                      cpu_raw: bytes, config_raw: bytes) -> None:
    if (cpu.get("schema") != "cross_tfg_v3_candidate_cpu_preflight_v1"
            or cpu.get("status") != "PASS_CPU_PREFLIGHT"
            or cpu.get("receipt_sha256") != payload_sha(cpu, "receipt_sha256")
            or cpu.get("receipt_sha256") != CPU03_SELF
            or len(cpu_raw) != 45_226):
        raise ValueError("actual CPU03 receipt does not match its approved raw/self/status pin")
    inputs = cpu.get("inputs", {})
    if (inputs.get("plan", {}).get("sha256") != PLAN13_RAW
            or inputs.get("plan", {}).get("plan_sha256") != PLAN13_SELF
            or inputs.get("plan_manifest", {}).get("sha256") != MANIFEST14_RAW
            or inputs.get("plan_manifest", {}).get("manifest_sha256") != MANIFEST14_SELF
            or inputs.get("candidate_config_cli03", {}).get("sha256") != CONFIG03_RAW
            or inputs.get("candidate_config_cli03", {}).get("bytes") != len(config_raw)):
        raise ValueError("CPU03 did not use the pinned frozen plan13/manifest14/config03")
    scope, runtime = cpu.get("scope", {}), cpu.get("runtime", {})
    if (scope.get("candidate_audio_rows_completed") != 20
            or scope.get("candidate_audio_rows_expected") != 20
            or scope.get("same_profile_reference_rows_completed") != 1
            or scope.get("same_profile_reference_rows_expected") != 1
            or scope.get("candidate_full180_started") is not False
            or scope.get("candidate_selected") is not False
            or scope.get("cuda_initialized") is not False
            or scope.get("gpu_used") is not False
            or scope.get("video_rendered") is not False
            or scope.get("scorer_called") is not False
            or scope.get("confirm_audio_read") is not False
            or scope.get("confirm_effects_read") is not False
            or runtime.get("device") != "cpu"
            or runtime.get("rows_reused_one_torch_process") is not True
            or runtime.get("strict_process_count") != 1
            or runtime.get("torch_cuda_initialized_before_rows") is not False
            or runtime.get("bootstrap", {}).get("sha256") != "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"):
        raise ValueError("CPU03 scope/bootstrap/runtime is not a strict CPU-only preflight")
    planned = {}
    for row in plan["rows"]:
        if row.get("model_id") == MODELS[0]:
            planned[row["sample_id"]] = row
    actual_rows = [row for row in cpu.get("candidate_rows", []) if row.get("kind") == "candidate_dev"]
    refs = [row for row in cpu.get("candidate_rows", []) if row.get("kind") == "same_profile_reference"]
    if set(planned) != {row.get("sample_id") for row in actual_rows} or len(actual_rows) != 20 or len(refs) != 1:
        raise ValueError("CPU03 sample rows do not match the 20 unique planned Wav2Lip samples")
    for item in actual_rows:
        row = planned[item["sample_id"]]
        natural = row["natural_pcm"]
        if ((item.get("input", {}).get("pcm_float32_sha256"), item.get("input", {}).get("sample_count"),
             item.get("input", {}).get("sample_rate_hz"), item.get("input", {}).get("channels"),
             item.get("input", {}).get("sha256")) !=
                (natural.get("pcm_sha256"), natural.get("sample_count"), natural.get("sample_rate_hz"),
                 natural.get("channels"), natural.get("file_sha256"))
                or item.get("native_expected_frame_count") != row.get("wav2lip_native_frame_binding", {}).get("expected_inference_frame_count")
                or item.get("output_sample_count") != natural.get("sample_count")
                or item.get("output_sample_rate_hz") != natural.get("sample_rate_hz")
                or item.get("output_channels") != natural.get("channels")
                or item.get("strict_policy_preserved") is not True
                or item.get("cuda_initialized_after_row") is not False):
            raise ValueError(f"CPU03 row does not match the immutable natural PCM/native frame map: {item['sample_id']}")


def replace_value(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return new if value == old else value
    if isinstance(value, list):
        return [replace_value(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_value(item, old, new) for key, item in value.items()}
    return value


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    for path in (PLAN13, MANIFEST14, RUNTIME09, CPU03, CONFIG03, GUARD15):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(f"metadata output already exists: {OUT}")
    plan, plan_raw = read_json(PLAN13, "candidate predecessor plan13", PLAN13_RAW, 16 * 1024 * 1024)
    manifest, manifest_raw = read_json(MANIFEST14, "candidate predecessor manifest14", MANIFEST14_RAW)
    runtime, runtime_raw = read_json(RUNTIME09, "samehost runtime binding09", RUNTIME09_RAW, 2 * 1024 * 1024)
    cpu, cpu_raw = read_json(CPU03, "actual CLI03 CPU preflight receipt", CPU03_RAW, 2 * 1024 * 1024)
    config, config_raw = read_json(CONFIG03, "candidate method CLI03 config", CONFIG03_RAW, 2 * 1024 * 1024)
    guard_raw = read_regular(GUARD15, "candidate guarded launcher15", 2 * 1024 * 1024)
    if (plan.get("status") != "FROZEN" or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or plan.get("plan_sha256") != PLAN13_SELF or len(plan.get("rows", [])) != 180
            or manifest.get("status") != "FROZEN" or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
            or manifest.get("manifest_sha256") != MANIFEST14_SELF or manifest.get("plan_raw_sha256") != PLAN13_RAW
            or runtime.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1"
            or runtime.get("status") != "PASS" or runtime.get("binding_sha256") != payload_sha(runtime, "binding_sha256")
            or len(runtime_raw) != 568_289):
        raise ValueError("frozen predecessor plan/manifest or runtime binding failed exact identity/self checks")
    if sha(config_raw) != CONFIG03_RAW or config.get("source_pins", {}).get("cli", {}).get("sha256") != "2cf5fc0f73d7c8d5aabbbc457672e487de2f8a88e683976bb58698325a80fdfd":
        raise ValueError("CLI03 config and source pin are not the approved exact bytes")
    cli_pin = config["source_pins"]["cli"]
    cli_path = ROOT / cli_pin["path"]
    cli_raw = read_regular(cli_path, "CLI03 implementation", 2 * 1024 * 1024)
    if len(cli_raw) != cli_pin["bytes"] or sha(cli_raw) != cli_pin["sha256"]:
        raise ValueError("CLI03 implementation does not match config03 source pin")
    check_cpu_receipt(plan, cpu, cpu_raw, config_raw)
    cpu_transport = verify_cpu_transport(cpu_raw)
    if plan.get("run_id") != "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt02":
        raise ValueError("candidate predecessor plan run identity differs from the expected frozen attempt02")
    if plan.get("execution", {}).get("candidate_runtime_qualification_performed") is not False:
        raise ValueError("candidate predecessor already claims runtime qualification")
    if plan.get("execution", {}).get("samehost_runtime_binding", {}).get("raw_sha256") != RUNTIME09_RAW:
        raise ValueError("candidate predecessor does not reference actual runtime binding09")
    if plan.get("execution", {}).get("baseline_controls", {}).get("run_id") != "cross_tfg_audio_head_20261003_samehost_controls_dev360_attempt10":
        raise ValueError("candidate plan does not cite the exact controls360 attempt10 parent")

    config_rel = f"{RUN_REL}/v3_method_candidate_config_attempt03.json"
    config_remote = f"{REMOTE_BASE}/{config_rel}"
    plan_remote = f"{REMOTE_BASE}/{RUN_REL}/candidate_plan_attempt14.json"
    manifest_remote = f"{REMOTE_BASE}/{RUN_REL}/candidate_plan_manifest_attempt15.json"
    rows_remote = f"{REMOTE_BASE}/{RUN_REL}/candidate_planned_rows_attempt14.jsonl"
    repair_remote = f"{REMOTE_BASE}/{RUN_REL}/candidate_runtime_repair_binding_attempt10.json"
    cpu_ref_path = CPU_REMOTE_PATH
    original_namespace = runtime["execution_bindings"]["output_root"]
    controls_output_leaf = original_namespace.rstrip("/") + "/controls360"
    if (OUTPUT_NAMESPACE == original_namespace
            or OUTPUT_NAMESPACE.rstrip("/").rsplit("/", 1)[0] != original_namespace.rstrip("/").rsplit("/", 1)[0]):
        raise ValueError("fresh candidate output namespace must be a sibling of the frozen controls namespace")
    cpu_input_plan = cpu["inputs"]["plan"]
    cpu_input_manifest = cpu["inputs"]["plan_manifest"]
    row_context_digest = row_context_sha(plan)
    method_context_digest = method_context_sha(plan)

    repair = {
        "schema": REPAIR_SCHEMA,
        "status": "PASS_CPU_PREFLIGHT_CANDIDATE_RUNTIME_BINDING_ONLY",
        "run_id": RUN_ID,
        "predecessor_plan": {"path": cpu_input_plan["path"], "bytes": cpu_input_plan["bytes"],
                             "raw_sha256": cpu_input_plan["sha256"], "plan_sha256": cpu_input_plan["plan_sha256"]},
        "predecessor_plan_raw_sha256": PLAN13_RAW,
        "predecessor_plan_payload_sha256": PLAN13_SELF,
        "predecessor_manifest": {"path": cpu_input_manifest["path"], "bytes": cpu_input_manifest["bytes"],
                                 "raw_sha256": cpu_input_manifest["sha256"], "manifest_sha256": cpu_input_manifest["manifest_sha256"]},
        "predecessor_manifest_raw_sha256": MANIFEST14_RAW,
        "runtime_binding": {"path": f"{REMOTE_BASE}/runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_attempt09.json",
                             "bytes": len(runtime_raw), "raw_sha256": sha(runtime_raw),
                             "binding_sha256": runtime["binding_sha256"], "schema": runtime["schema"],
                             "status": runtime["status"], "gpu_uuid": runtime["gpu_uuid"]},
        "runtime_binding_raw_sha256": sha(runtime_raw),
        "controls_parent": {"run_id": plan["execution"]["baseline_controls"]["run_id"],
                             "plan": copy.deepcopy(plan["source_reference_plans"]["samehost_controls_dev360"]),
                             "output_leaf": controls_output_leaf,
                             "source_manifest_raw_sha256": manifest["samehost_controls_plan_manifest_raw_sha256"]},
        "cpu_preflight_receipt": {"path": cpu_ref_path, "bytes": len(cpu_raw), "raw_sha256": sha(cpu_raw),
                                  "schema": cpu["schema"], "status": cpu["status"]},
        "cpu_preflight_receipt_sha256": cpu["receipt_sha256"],
        "cpu_preflight_transport_evidence": cpu_transport,
        "cpu_preflight_scope": copy.deepcopy(cpu["scope"]),
        "candidate_config": {"path": config_remote, "project_relative_path": config_rel,
                             "bytes": len(config_raw), "raw_sha256": sha(config_raw)},
        "cli_source": {"project_relative_path": cli_pin["path"], "bytes": len(cli_raw), "raw_sha256": sha(cli_raw)},
        "candidate_method_id": METHOD_ID,
        "candidate_checkpoint_sha256": plan["execution"]["candidate_checkpoint_sha256"],
        "candidate_rows": 180,
        "candidate_rows_context_sha256": row_context_digest,
        "candidate_method_context_sha256": method_context_digest,
        "original_runtime_output_namespace": original_namespace,
        "controls_parent_output_leaf": controls_output_leaf,
        "candidate_output_namespace": OUTPUT_NAMESPACE,
        "candidate_output_leaf": OUTPUT_LEAF,
        "runtime_validation_overlay": {
            "schema": "cross_tfg_v3_candidate_runtime_validation_overlay_v1",
            "only_changed_field": "execution_bindings.output_root",
            "source_runtime_binding_raw_sha256": sha(runtime_raw),
            "source_binding_is_rewritten": False,
            "overlay_canonical_sha256_is_computed_by_guard_at_runtime": True,
        },
        "scope": {"candidate_generation_started": False, "candidate_gpu_qualification_performed": False,
                  "candidate_full180_started": False, "confirm_audio_read": False,
                  "confirm_effects_read": False, "status_meaning": "CPU preflight admission only; not a candidate runtime qualification"},
    }
    repair["repair_binding_sha256"] = payload_sha(repair, "repair_binding_sha256")
    repair_raw = pretty(repair)
    repair_sha = sha(repair_raw)

    candidate = copy.deepcopy(plan)
    candidate["run_id"] = RUN_ID
    method = candidate["methods"][0]
    method["config"]["v3_cli_config_path"] = config_rel
    method["config"]["v3_cli_config_sha256"] = sha(config_raw)
    method["config_sha256"] = sha(canonical(method["config"]))
    for row in candidate["rows"]:
        row["generation_settings"]["v3_cli_config_sha256"] = sha(config_raw)
        row["method_config_sha256"] = method["config_sha256"]
    candidate["execution"]["candidate_config_raw_sha256"] = sha(config_raw)
    candidate["execution"]["v3_cli_sha256"] = sha(cli_raw)
    candidate["execution"]["samehost_candidate_guarded_launcher"] = {
        "path": GUARD_REL, "bytes": len(guard_raw), "raw_sha256": sha(guard_raw)}
    candidate["execution"]["samehost_dev360_source_closure"]["step120_candidate_guarded_launcher"] = {
        "path": GUARD_REL, "bytes": len(guard_raw), "raw_sha256": sha(guard_raw)}
    candidate["execution"]["candidate_runtime_repair_binding"] = {
        "path": repair_remote, "bytes": len(repair_raw), "raw_sha256": repair_sha,
        "repair_binding_sha256": repair["repair_binding_sha256"], "schema": repair["schema"],
        "status": repair["status"]}
    candidate["execution"]["candidate_output_root"] = OUTPUT_LEAF
    candidate["execution"]["candidate_output_namespace"] = OUTPUT_NAMESPACE
    candidate["execution"]["candidate_runtime_qualification_performed"] = False
    candidate["source_reference_plans"]["samehost_candidate_guarded_launcher"] = copy.deepcopy(
        candidate["execution"]["samehost_candidate_guarded_launcher"])
    candidate["source_reference_plans"]["v3_cli"] = {
        "path": cli_pin["path"], "bytes": len(cli_raw), "raw_sha256": sha(cli_raw)}
    candidate["source_reference_plans"]["v3_cli_config"] = {
        "path": config_rel, "bytes": len(config_raw), "raw_sha256": sha(config_raw),
        "checkpoint_sha256": config["candidate_provenance"]["checkpoint"]["sha256"],
        "selection_record_sha256": config["candidate_provenance"]["selection_record"]["sha256"],
        "training_record_sha256": config["candidate_provenance"]["training_record"]["sha256"]}
    candidate["source_reference_plans"]["candidate_runtime_repair_binding"] = copy.deepcopy(
        candidate["execution"]["candidate_runtime_repair_binding"])
    candidate["source_reference_plans"]["candidate_output_root"] = {
        "namespace": OUTPUT_NAMESPACE, "leaf": OUTPUT_LEAF,
        "controls_parent_leaf": controls_output_leaf,
        "status": "PLANNED_FRESH_OUTPUT_NOT_CREATED"}
    if row_context_sha(candidate) != row_context_digest or method_context_sha(candidate) != method_context_digest:
        raise ValueError("candidate metadata edit changed frozen row/method context beyond CLI pin/config refs")
    candidate.pop("plan_sha256", None)
    candidate["plan_sha256"] = payload_sha(candidate, "plan_sha256")
    plan_raw_out = pretty(candidate)
    rows_raw_out = b"".join(canonical(row) + b"\n" for row in candidate["rows"])
    out_manifest = copy.deepcopy(manifest)
    out_manifest.update({
        "run_id": RUN_ID,
        "plan_sha256": candidate["plan_sha256"],
        "plan_raw_sha256": sha(plan_raw_out),
        "planned_rows_sha256": sha(rows_raw_out),
        "planned_rows": 180,
        "candidate_config_raw_sha256": sha(config_raw),
        "v3_cli_raw_sha256": sha(cli_raw),
        "candidate_runtime_repair_binding_raw_sha256": repair_sha,
        "candidate_cpu_preflight_receipt_raw_sha256": sha(cpu_raw),
        "candidate_runtime_qualification_performed": False,
        "candidate_output_namespace": OUTPUT_NAMESPACE,
        "candidate_output_root": OUTPUT_LEAF,
        "samehost_runtime_binding_raw_sha256": sha(runtime_raw),
        "status": "FROZEN",
    })
    out_manifest.pop("manifest_sha256", None)
    out_manifest["manifest_sha256"] = payload_sha(out_manifest, "manifest_sha256")
    manifest_raw_out = pretty(out_manifest)
    OUT.mkdir(parents=True, exist_ok=False)
    write_exclusive(OUT / "candidate_runtime_repair_binding_attempt10.json", repair_raw)
    write_exclusive(OUT / "candidate_plan_attempt14.json", plan_raw_out)
    write_exclusive(OUT / "candidate_planned_rows_attempt14.jsonl", rows_raw_out)
    write_exclusive(OUT / "candidate_plan_manifest_attempt15.json", manifest_raw_out)
    summary = {
        "schema": "cross_tfg_v3_candidate_runtime_repair_metadata_build_v1",
        "status": "PASS_METADATA_ONLY_CANDIDATE_PLAN_REBOUND_NOT_RUNTIME_QUALIFICATION",
        "run_id": RUN_ID, "planned_rows": 180, "models": list(MODELS),
        "source_inputs": {
            "predecessor_plan13": {"path": str(PLAN13), "bytes": len(plan_raw), "raw_sha256": sha(plan_raw), "plan_sha256": plan["plan_sha256"]},
            "predecessor_manifest14": {"path": str(MANIFEST14), "bytes": len(manifest_raw), "raw_sha256": sha(manifest_raw), "manifest_sha256": manifest["manifest_sha256"]},
            "runtime_binding09": {"path": str(RUNTIME09), "bytes": len(runtime_raw), "raw_sha256": sha(runtime_raw), "binding_sha256": runtime["binding_sha256"]},
            "cpu_preflight03": {"path": str(CPU03), "bytes": len(cpu_raw), "raw_sha256": sha(cpu_raw), "receipt_sha256": cpu["receipt_sha256"], "remote_path": CPU_REMOTE_PATH},
            "cpu_preflight_transport": cpu_transport,
            "candidate_config03": {"path": str(CONFIG03), "bytes": len(config_raw), "raw_sha256": sha(config_raw)},
            "candidate_guard15": {"path": str(GUARD15.relative_to(ROOT)), "bytes": len(guard_raw), "raw_sha256": sha(guard_raw)},
        },
        "outputs": {
            "repair_binding": {"path": repair_remote, "local_path": str(OUT / "candidate_runtime_repair_binding_attempt10.json"), "bytes": len(repair_raw), "raw_sha256": repair_sha, "self_sha256": repair["repair_binding_sha256"]},
            "plan": {"path": plan_remote, "local_path": str(OUT / "candidate_plan_attempt14.json"), "bytes": len(plan_raw_out), "raw_sha256": sha(plan_raw_out), "plan_sha256": candidate["plan_sha256"]},
            "rows": {"path": rows_remote, "local_path": str(OUT / "candidate_planned_rows_attempt14.jsonl"), "bytes": len(rows_raw_out), "raw_sha256": sha(rows_raw_out), "count": 180},
            "manifest": {"path": manifest_remote, "local_path": str(OUT / "candidate_plan_manifest_attempt15.json"), "bytes": len(manifest_raw_out), "raw_sha256": sha(manifest_raw_out), "manifest_sha256": out_manifest["manifest_sha256"]},
        },
        "scope": {"confirm_audio_read": False, "confirm_effects_read": False,
                  "candidate_generation_started": False, "gpu_used": False,
                  "plan_status": "FROZEN", "runtime_qualification_status": "NOT_PERFORMED"},
    }
    write_exclusive(OUT / "metadata_build_summary_attempt01.json", pretty(summary))
    print(json.dumps(summary["outputs"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
