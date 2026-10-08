#!/usr/bin/env python3
"""Guarded strict-policy launcher for the rebound same-host step-120 Dev180 slate.

The default path is metadata-only. Native generation starts only with
--execute-candidate after the exact plan, runtime receipt, backend repeats,
source closure, and live resource gates pass. This wrapper never reads Confirm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003"
GUARD_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_samehost_dev360_controls_attempt18.py"
BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03/sitecustomize.py"
CANDIDATE_DISPATCHER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt07.py"
EXPECTED_ROWS = 180
METHOD_CANDIDATE = "shared_recipient_transition_head_v3_fit40_step120"
W2L_NATIVE_ENTRY = "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/inference.py"
W2L_NATIVE_ENTRY_SHA256 = "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6"
W2L_RUNTIME_COMPAT_SHA256 = "462884a2534f73d93eb09cd05c87fefda677e3647366360fb81a9622fa7592ca"
STRICT_POLICY = {
    "pythonhashseed": "20261003", "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
    "numexpr_num_threads": "1", "torch_num_threads": 1, "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True, "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False, "torch_allow_tf32": False,
}


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def strict_startup_gate_code() -> str:
    """Check strict policy in the same Python process before dispatcher imports."""
    return r'''import hashlib,json,os,runpy,sys
from pathlib import Path
expected_env={"PYTHONHASHSEED":"20261003","CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
for key,value in expected_env.items():
    if os.environ.get(key)!=value: raise SystemExit("strict startup environment mismatch: "+key)
if os.environ.get("CROSS_TFG_STRICT_STARTUP_POLICY")!="1": raise SystemExit("strict startup flag missing")
import torch
if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1: raise SystemExit("strict Torch thread policy mismatch")
if not torch.are_deterministic_algorithms_enabled(): raise SystemExit("strict Torch deterministic algorithms are disabled")
if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False: raise SystemExit("strict cuDNN policy mismatch")
if torch.backends.cudnn.allow_tf32 is not False or torch.backends.cuda.matmul.allow_tf32 is not False: raise SystemExit("strict TF32 policy mismatch")
policy={"pythonhashseed":os.environ["PYTHONHASHSEED"],"cublas_workspace_config":os.environ["CUBLAS_WORKSPACE_CONFIG"],"omp_num_threads":os.environ["OMP_NUM_THREADS"],"mkl_num_threads":os.environ["MKL_NUM_THREADS"],"openblas_num_threads":os.environ["OPENBLAS_NUM_THREADS"],"numexpr_num_threads":os.environ["NUMEXPR_NUM_THREADS"],"torch_num_threads":torch.get_num_threads(),"torch_num_interop_threads":torch.get_num_interop_threads(),"torch_use_deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"torch_cudnn_deterministic":torch.backends.cudnn.deterministic,"torch_cudnn_benchmark":torch.backends.cudnn.benchmark,"torch_allow_tf32":bool(torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32)}
receipt={"schema":"cross_tfg_step120_candidate_startup_policy_gate_v1","status":"PASS","pid":os.getpid(),"python_executable":str(Path(sys.executable).resolve()),"bootstrap_source_sha256":"3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65","policy":policy}
raw_canonical=json.dumps(receipt,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
receipt["receipt_sha256"]=hashlib.sha256(raw_canonical).hexdigest()
receipt_path=Path(os.environ["CROSS_TFG_STARTUP_POLICY_RECEIPT"])
if not receipt_path.is_absolute() or receipt_path.exists() or receipt_path.is_symlink(): raise SystemExit("startup receipt target is unsafe or already exists")
raw=(json.dumps(receipt,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")
fd=os.open(receipt_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,"wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
dispatcher=sys.argv[1]
sys.argv=[dispatcher,*sys.argv[2:]]
runpy.run_path(dispatcher,run_name="__main__")
'''


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def load_guard(project_root: Path):
    path = project_root / GUARD_REL
    raw = path.read_bytes()
    closure = None
    spec = importlib.util.spec_from_file_location("samehost_dev360_guard_for_step120", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen same-host runtime guard")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, raw


def exact_project_source(project_root: Path, path_value: Any, expected_sha: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"{label} source reference is malformed")
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{label} source path escapes the repository") from exc
    raw = resolved.read_bytes()
    if sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} source SHA differs from the frozen plan")
    return resolved


def validate_candidate_plan(args: argparse.Namespace, project_root: Path, guard: Any,
                            binding_raw_sha: str) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, bytes]:
    plan, plan_raw = guard.read_json(args.candidate_plan, "step-120 candidate plan", args.candidate_plan_raw_sha256,
                                     32 * 1024 * 1024)
    manifest, manifest_raw = guard.read_json(args.plan_manifest, "step-120 plan manifest",
                                             args.plan_manifest_raw_sha256, 4 * 1024 * 1024)
    rows_raw = guard.safe_regular(args.planned_rows, "step-120 planned rows", 32 * 1024 * 1024)[1]
    rows = [json.loads(line.decode("utf-8")) for line in rows_raw.splitlines() if line.strip()]
    if (plan.get("status") != "FROZEN" or plan.get("schema_version") != "cross-tfg-audio-eval-v1"
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or len(plan.get("rows", [])) != EXPECTED_ROWS or rows != plan.get("rows")
            or sha_bytes(rows_raw) != args.planned_rows_raw_sha256.lower()
            or b"".join(canonical(row) + b"\n" for row in rows) != rows_raw):
        raise ValueError("candidate plan/row ledger raw binding, payload, or 180-row scope mismatch")
    if (manifest.get("schema") != "cross_tfg_v3_candidate_only_dev_plan_v1"
            or manifest.get("status") != "FROZEN"
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
            or manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or manifest.get("planned_rows_sha256") != sha_bytes(rows_raw)
            or manifest.get("planned_rows") != EXPECTED_ROWS):
        raise ValueError("candidate plan manifest does not bind the exact 180-row plan")
    if tuple(item.get("model_id") for item in plan.get("models", [])) != guard.MODELS:
        raise ValueError("candidate plan front-end roster differs from the qualified three-backend roster")
    methods = plan.get("methods", [])
    if len(methods) != 1 or methods[0].get("method_id") != METHOD_CANDIDATE:
        raise ValueError("candidate plan is not the fixed step-120 method slate")
    execution = plan.get("execution", {})
    if (execution.get("fixed_optimizer_steps") != 120
            or execution.get("candidate_checkpoint_sha256") is None
            or execution.get("confirm_audio_read") is not False
            or execution.get("confirm_effects_read") is not False):
        raise ValueError("candidate plan is not a closed-Confirm fixed step-120 plan")
    runtime_ref = execution.get("samehost_runtime_binding", {})
    strict_ref = execution.get("strict_renderer_determinism", {})
    qc_ref = execution.get("qc06_determinism_qualification", {})
    if (runtime_ref.get("raw_sha256") != binding_raw_sha
            or runtime_ref.get("gpu_uuid") != guard.GPU_UUID
            or strict_ref.get("policy_values") != guard.STRICT_BOOTSTRAP_POLICY
            or strict_ref.get("bootstrap_source", {}).get("path") != BOOTSTRAP_REL
            or strict_ref.get("bootstrap_source", {}).get("raw_sha256") != guard.STRICT_BOOTSTRAP_SHA256
            or strict_ref.get("qualified_policy_reference_sha256") != guard.QC06["determinism_policy_sha256"]
            or qc_ref.get("source_sha256") != guard.QC06["source_sha256"]):
        raise ValueError("candidate plan is not bound to the current strict renderer policy/QC06 source")
    candidate_guard = execution.get("samehost_candidate_guarded_launcher", {})
    candidate_guard_path = exact_project_source(project_root, candidate_guard.get("path"),
                                                candidate_guard.get("raw_sha256"), "candidate guarded launcher")
    if candidate_guard_path != (project_root / Path(__file__).resolve().relative_to(project_root)).resolve(strict=True):
        raise ValueError("candidate plan guarded-launcher pin does not match this source")
    closure = execution.get("samehost_dev360_source_closure")
    if not isinstance(closure, dict):
        raise ValueError("candidate plan lacks the exact same-host strict source closure")
    for entry in closure.values():
        exact_project_source(project_root, entry.get("path") if isinstance(entry, dict) else None,
                             entry.get("raw_sha256") if isinstance(entry, dict) else None,
                             "candidate strict source-closure entry")
    dispatcher = plan.get("source_reference_plans", {}).get("v3_candidate_dispatcher", {})
    dispatcher_path = exact_project_source(project_root, dispatcher.get("path"), dispatcher.get("raw_sha256"),
                                           "V3 candidate Stage-A dispatcher")
    if dispatcher_path != (project_root / CANDIDATE_DISPATCHER_REL).resolve(strict=True):
        raise ValueError("candidate Stage-A dispatcher path differs from the reviewed source")
    for key in ("stagea_core", "cross_frontend_adapter_runner", "v3_cli"):
        source = plan.get("source_reference_plans", {}).get(key, {})
        exact_project_source(project_root, source.get("path"), source.get("raw_sha256"), key)
    config_ref = plan.get("source_reference_plans", {}).get("v3_cli_config", {})
    config_path = exact_project_source(project_root, config_ref.get("path"), config_ref.get("raw_sha256"),
                                       "V3 candidate CLI config")
    method_config = methods[0].get("config", {})
    planned_config_path = Path(str(method_config.get("v3_cli_config_path", "")))
    if not planned_config_path.is_absolute():
        planned_config_path = project_root / planned_config_path
    if (planned_config_path.resolve(strict=True) != config_path
            or method_config.get("v3_cli_config_sha256") != config_ref.get("raw_sha256")):
        raise ValueError("candidate method config path/SHA differs from its plan source reference")
    return plan, plan_raw, manifest, manifest_raw, rows_raw


def strict_receipts(plan: Mapping[str, Any], binding: Mapping[str, Any], receipt_dir: Path,
                    plan_sha: str, binding_sha: str, guard: Any) -> dict[str, Any]:
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise ValueError("candidate renderer deterministic receipt directory is missing or unsafe")
    plan_rows = plan.get("rows")
    if not isinstance(plan_rows, list) or len(plan_rows) != EXPECTED_ROWS:
        raise ValueError("strict candidate receipt audit requires exactly 180 planned rows")
    expected = {row["row_id"]: row for row in plan_rows}
    exec_binding = binding["execution_bindings"]
    backend_map = binding["backends"]
    expected_entries: dict[str, tuple[str, str]] = {
        guard.MODELS[0]: (str(Path(backend_map[guard.MODELS[0]]["generator"]["native_entry_path"]).resolve(strict=True)),
                          str(Path(exec_binding["wav2lip_python"]).resolve(strict=True)),)
    }
    for model_id in guard.MODELS[1:]:
        generator = backend_map[model_id]["generator"]
        expected_entries[model_id] = (str(Path(generator["render_entry_path"]).resolve(strict=True)),
                                      str(Path(generator["python"]).resolve(strict=True)))
    files = sorted(receipt_dir.glob("*.json"))
    if not files:
        raise ValueError("candidate render run produced no strict bootstrap receipts")
    receipts: list[dict[str, Any]] = []
    matching_by_row: dict[str, list[dict[str, Any]]] = {row_id: [] for row_id in expected}
    for path in files:
        _, raw = guard.safe_regular(path, "candidate strict renderer receipt", 256 * 1024)
        record = json.loads(raw.decode("utf-8"))
        if (record.get("schema") != "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1"
                or record.get("status") != "PASS"
                or record.get("receipt_sha256") != payload_sha(record, "receipt_sha256")
                or record.get("bootstrap_source_sha256") != guard.STRICT_BOOTSTRAP_SHA256
                or record.get("policy") != guard.STRICT_BOOTSTRAP_POLICY
                or record.get("policy_payload_sha256") != sha_bytes(canonical(record.get("policy")))
                or record.get("cuda_available") is not True or record.get("pid") != int(path.stem)):
            raise ValueError(f"candidate renderer receipt failed strict policy/self-hash checks: {path.name}")
        row_id = record.get("row_id")
        if row_id not in expected:
            raise ValueError(f"candidate renderer receipt has an unknown row ID: {path.name}")
        row = expected[row_id]
        expected_entry, expected_python = expected_entries[row.get("model_id")]
        if (record.get("model_id") != row.get("model_id")
                or record.get("audio_frontend") != row.get("audio_frontend")
                or record.get("seed") != row.get("seed")
                or record.get("expected_backend_entry") != expected_entry
                or record.get("python_executable") != expected_python
                or record.get("entry_matches_backend") is not (record.get("entry_path") == expected_entry)):
            raise ValueError(f"candidate renderer receipt differs from exact row/backend binding: {row_id}")
        if record["entry_matches_backend"]:
            matching_by_row[row_id].append({"path": str(path.resolve(strict=True)), "bytes": len(raw),
                                            "sha256": sha_bytes(raw), "pid": record["pid"],
                                            "entry_path": expected_entry})
        receipts.append({"path": str(path.resolve(strict=True)), "bytes": len(raw),
                         "sha256": sha_bytes(raw), "row_id": row_id,
                         "model_id": row["model_id"], "pid": record["pid"],
                         "entry_matches_backend": record["entry_matches_backend"]})
    if any(len(items) != 1 for items in matching_by_row.values()):
        raise ValueError("candidate strict receipts do not contain exactly one native renderer receipt per row")
    result: dict[str, Any] = {
        "schema": "cross_tfg_v3_samehost_renderer_determinism_attestation_v1",
        "status": "PASS", "gpu_uuid": guard.GPU_UUID,
        "plan_raw_sha256": plan_sha, "runtime_binding_raw_sha256": binding_sha,
        "strict_bootstrap_source_sha256": guard.STRICT_BOOTSTRAP_SHA256,
        "policy_values": guard.STRICT_BOOTSTRAP_POLICY,
        "qualified_policy_reference_sha256": guard.QC06["determinism_policy_sha256"],
        "per_backend_repeat_qualification_receipts": binding.get("backend_determinism_qualifications"),
        "row_count": len(expected), "native_renderer_row_receipt_count": len(expected),
        "process_receipt_count": len(receipts),
        "native_renderer_receipts_by_row": {row_id: values[0] for row_id, values in sorted(matching_by_row.items())},
        "process_receipts": receipts,
        "confirm_audio_read": False, "confirm_data_opened": False,
    }
    result["attestation_sha256"] = payload_sha(result, "attestation_sha256")
    return result


def validate_w2l_result_evidence(output: Path, plan: Mapping[str, Any], binding: Mapping[str, Any],
                                 plan_sha: str, guard: Any) -> dict[str, Any]:
    """Verify that every successful candidate W2L row carries the actual native-policy/input receipts."""
    result_path = output / "results.jsonl"
    manifest_path = output / "result_manifest.json"
    manifest, manifest_raw = guard.read_json(manifest_path, "candidate result manifest", None, 8 * 1024 * 1024)
    _, results_raw = guard.safe_regular(result_path, "candidate result ledger", 128 * 1024 * 1024)
    rows = [json.loads(line.decode("utf-8")) for line in results_raw.splitlines() if line.strip()]
    expected_rows = plan.get("rows")
    if (not isinstance(expected_rows, list) or len(expected_rows) != EXPECTED_ROWS
            or len(rows) != EXPECTED_ROWS
            or [row.get("row_id") for row in rows] != [row.get("row_id") for row in expected_rows]
            or manifest.get("plan_raw_sha256") != plan_sha
            or manifest.get("planned_rows") != EXPECTED_ROWS
            or manifest.get("result_rows") != EXPECTED_ROWS
            or manifest.get("results_jsonl_sha256") != sha_bytes(results_raw)
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")):
        raise ValueError("candidate Wav2Lip receipt audit does not bind the exact full result ledger")
    backend = binding.get("backends", {}).get(guard.MODELS[0], {})
    generator = backend.get("generator", {})
    wrapper_sha = generator.get("runtime_compat_sitecustomize_sha256")
    native_path = generator.get("native_entry_path")
    native_sha = generator.get("native_entry_sha256")
    if (wrapper_sha != W2L_RUNTIME_COMPAT_SHA256
            or native_sha != W2L_NATIVE_ENTRY_SHA256
            or native_path != W2L_NATIVE_ENTRY):
        raise ValueError("candidate runtime binding lacks exact Wav2Lip native entry/wrapper pins")
    checked: list[dict[str, Any]] = []
    for planned, row in zip(expected_rows, rows):
        if row.get("model_id") != guard.MODELS[0]:
            continue
        if row.get("status") != "PASS":
            continue
        render = row.get("render", {})
        generation = render.get("generation", {}) if isinstance(render, dict) else {}
        audio = generation.get("generator_input_audio", {})
        if (generation.get("native_entry_path") != native_path
                or generation.get("native_entry_sha256") != native_sha
                or audio.get("path") != row.get("generator_pcm", {}).get("path")
                or audio.get("pcm_float32_sha256") != row.get("generator_pcm_sha256")
                or audio.get("sample_count") != planned.get("natural_pcm", {}).get("sample_count")
                or audio.get("sample_rate_hz") != 16000 or audio.get("channels") != 1):
            raise ValueError(f"candidate successful Wav2Lip row lacks exact transformed input/native-entry evidence: {row.get('row_id')}")
        expected_frames = planned.get("wav2lip_native_frame_binding", {}).get("expected_inference_frame_count")
        refs = generation.get("wav2lip_runtime_compatibility_receipts")
        if not isinstance(refs, dict) or set(refs) != {
                "constructor_policy", "actual_forward_policy", "prewrite_pixel_arrays"}:
            raise ValueError(f"candidate successful Wav2Lip row lacks native runtime receipts: {row.get('row_id')}")
        native_records: dict[str, Any] = {}
        for key, schema, expected_status in (
            ("constructor_policy", "cross_tfg_w2l_facealignment_constructor_policy_receipt_v1", "PASS_CONSTRUCTOR_RETURN_RESTORED"),
            ("actual_forward_policy", "cross_tfg_w2l_actual_forward_policy_receipt_v1", "PASS_ACTUAL_FORWARD_ENTRIES"),
            ("prewrite_pixel_arrays", "cross_tfg_w2l_prewrite_pixel_array_receipt_v1", "PASS"),
        ):
            ref = refs[key]
            if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
                raise ValueError(f"candidate Wav2Lip {key} reference is malformed")
            obj, raw = guard.read_json(Path(ref["path"]), f"candidate Wav2Lip {key}", ref.get("sha256"), 8 * 1024 * 1024)
            if (len(raw) != ref.get("bytes") or obj.get("schema") != schema
                    or obj.get("status") != expected_status
                    or (obj.get("row_id"), obj.get("model_id"), obj.get("seed")) !=
                       (planned.get("row_id"), guard.MODELS[0], planned.get("seed"))
                    or obj.get("runtime_compat_source_sha256") != wrapper_sha
                    or obj.get("strict_bootstrap_source_sha256") != guard.STRICT_BOOTSTRAP_SHA256
                    or obj.get("receipt_sha256") != payload_sha(obj, "receipt_sha256")):
                raise ValueError(f"candidate Wav2Lip {key} receipt failed exact row/source/self-hash checks")
            native_records[key] = {"path": ref["path"], "bytes": len(raw), "sha256": sha_bytes(raw)}
        prewrite_obj, _ = guard.read_json(Path(refs["prewrite_pixel_arrays"]["path"]),
                                          "candidate Wav2Lip prewrite array receipt",
                                          refs["prewrite_pixel_arrays"]["sha256"], 8 * 1024 * 1024)
        forward_obj, _ = guard.read_json(Path(refs["actual_forward_policy"]["path"]),
                                         "candidate Wav2Lip forward receipt",
                                         refs["actual_forward_policy"]["sha256"], 8 * 1024 * 1024)
        if (type(expected_frames) is not int or expected_frames <= 0
                or prewrite_obj.get("expected_frame_count") != expected_frames
                or prewrite_obj.get("frame_count") != expected_frames
                or len(prewrite_obj.get("frame_pixel_array_sha256s", [])) != expected_frames
                or prewrite_obj.get("ordered_frame_hashes_sha256") != sha_bytes(canonical(prewrite_obj.get("frame_pixel_array_sha256s")))
                or forward_obj.get("sfd_forward_count", 0) <= 0
                or forward_obj.get("wav2lip_generator_forward_count", 0) <= 0
                or forward_obj.get("policy_sha256") != "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
                or forward_obj.get("policy_payload_sha256") != "54d853a6b6ce09cc8c757297f9617b5ecf4e0d01458a0b2e33dcf0316a4417f5"
                or forward_obj.get("sfd_network_fallback_prohibited") is not True
                or not forward_obj.get("all_observed_actual_forward_entries_match_strict_policy")):
            raise ValueError(f"candidate Wav2Lip actual-forward/prewrite evidence differs from the bound frame count: {row.get('row_id')}")
        checked.append({"row_id": row["row_id"], "seed": row.get("seed"),
                        "expected_frame_count": expected_frames,
                        "native_receipts": native_records})
    return {"result_manifest_path": str(manifest_path.resolve(strict=True)),
            "result_manifest_bytes": len(manifest_raw), "result_manifest_sha256": sha_bytes(manifest_raw),
            "result_rows_sha256": sha_bytes(results_raw), "successful_w2l_rows_with_native_receipts": checked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--candidate-plan-raw-sha256", required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--planned-rows-raw-sha256", required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-raw-sha256", required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--runtime-binding-raw-sha256", required=True)
    parser.add_argument("--execute-candidate", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve(strict=True)
    if project_root != ROOT.resolve(strict=True):
        raise ValueError("candidate launcher project root differs from its pinned source repository")
    guard, guard_raw = load_guard(project_root)
    plan, plan_raw, manifest, manifest_raw, rows_raw = validate_candidate_plan(
        args, project_root, guard, args.runtime_binding_raw_sha256.lower())
    runtime, runtime_raw = guard.read_json(args.runtime_binding, "same-host runtime binding",
                                           args.runtime_binding_raw_sha256, 8 * 1024 * 1024)
    if sha_bytes(guard_raw) != plan["execution"]["samehost_dev360_source_closure"]["samehost_dev360_guarded_launcher"]["raw_sha256"]:
        raise ValueError("same-host runtime guard source differs from the candidate source closure")
    if runtime.get("status") != "PASS":
        raise ValueError("same-host runtime binding is not a PASS receipt")
    tools = runtime.get("execution_bindings", {})
    output = guard.subrun_output_root(runtime, "candidate180")
    helper = guard.load_resource_helper(project_root)
    live, identities = guard.validate_bindings(project_root, plan, runtime, helper, output,
                                                capacity_stage="candidate180")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("candidate Stage-A output root must be new or empty")
    if not args.execute_candidate:
        print(json.dumps({"status": "PASS_METADATA_ONLY", "candidate_rows": EXPECTED_ROWS,
                          "plan_raw_sha256": sha_bytes(plan_raw),
                          "runtime_binding_raw_sha256": sha_bytes(runtime_raw),
                          "gpu_uuid": runtime.get("gpu_uuid"), "live_resource_probe": live,
                          "execution_performed": False, "confirm_audio_read": False},
                         sort_keys=True, ensure_ascii=False))
        return 0

    receipt_dir = output / "runtime_determinism_receipts"
    output.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(mode=0o700)
    refs = plan.get("source_reference_plans", {})
    stagea_core = exact_project_source(project_root, refs.get("stagea_core", {}).get("path"),
                                       refs.get("stagea_core", {}).get("raw_sha256"), "Stage-A core")
    cross_runner = exact_project_source(project_root, refs.get("cross_frontend_adapter_runner", {}).get("path"),
                                        refs.get("cross_frontend_adapter_runner", {}).get("raw_sha256"),
                                        "cross-front-end adapter runner")
    cli = exact_project_source(project_root, refs.get("v3_cli", {}).get("path"),
                               refs.get("v3_cli", {}).get("raw_sha256"), "V3 candidate CLI")
    expected_hashes = runtime["expected_file_sha256"]
    dispatcher_command = [
        str(project_root / CANDIDATE_DISPATCHER_REL),
        "--stagea-core", str(stagea_core), "--cross-frontend-adapter-runner", str(cross_runner),
        "--v3-cli", str(cli), "--expected-v3-cli-sha256", refs["v3_cli"]["raw_sha256"],
        "--plan", str(args.candidate_plan.resolve(strict=True)),
        "--planned-rows", str(args.planned_rows.resolve(strict=True)),
        "--expected-plan-raw-sha256", sha_bytes(plan_raw),
        "--output-root", str(output), "--project-root", str(project_root),
        "--wav2lip-python", tools["wav2lip_python"], "--wav2lip-root", tools["wav2lip_root"],
        "--generator-child", tools["generator_child"],
        "--syncnet-root", tools["syncnet_root"], "--syncnet-python", tools["syncnet_python"],
        "--syncnet-checkpoint", tools["syncnet_checkpoint"], "--syncnet-evaluator", tools["syncnet_evaluator"],
        "--ffmpeg", tools["ffmpeg"], "--ffprobe", tools["ffprobe"],
        "--ffmpeg-sha256", expected_hashes["ffmpeg"],
        "--pcm16-lineage-sha256", expected_hashes["pcm16_lineage"],
        "--generator-timeout", str(runtime.get("timeouts", {}).get("generator_seconds", 900)),
        "--syncnet-timeout", str(runtime.get("timeouts", {}).get("syncnet_seconds", 1800)),
        "--ffmpeg-timeout", str(runtime.get("timeouts", {}).get("ffmpeg_seconds", 300)),
    ]
    startup_receipt_path = output.parent / f"{output.name}.candidate_startup_policy_receipt.json"
    if startup_receipt_path.exists() or startup_receipt_path.is_symlink():
        raise FileExistsError("candidate startup policy receipt already exists")
    command = [tools["stagea_python"], "-c", strict_startup_gate_code(), *dispatcher_command]
    bootstrap_dir = (project_root / BOOTSTRAP_REL).parent
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join((str(bootstrap_dir), str(project_root), env.get("PYTHONPATH", ""))),
        "CUDA_VISIBLE_DEVICES": str(runtime.get("cuda_visible_devices", "0")),
        "PYTHONHASHSEED": "20261003", "TFG_DETERMINISM_SEED": "20261003",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CROSS_TFG_STRICT_STARTUP_POLICY": "1",
        "CROSS_TFG_STARTUP_POLICY_RECEIPT": str(startup_receipt_path.resolve()),
        "CROSS_TFG_DETERMINISM_RECEIPT_DIR": str(receipt_dir.resolve()),
        "CROSS_TFG_RUNTIME_BINDING_PATH": str(args.runtime_binding.resolve(strict=True)),
        "CROSS_TFG_RUNTIME_BINDING_RAW_SHA256": sha_bytes(runtime_raw),
        "CROSS_TFG_CAPACITY_STAGE": "candidate180",
    })
    for key in tuple(env):
        if (key.startswith("CROSS_TFG_ROW_") or key in {
                "CROSS_TFG_MODEL_ID", "CROSS_TFG_AUDIO_FRONTEND", "CROSS_TFG_BACKEND_ENTRY",
                "CROSS_TFG_RENDER_SEED", "CROSS_TFG_EXPECTED_FRAME_COUNT", "CROSS_TFG_CONTROL_DIR",
                "CROSS_TFG_RUNTIME_COMPAT_PREFLIGHT", "CROSS_TFG_STRICT_DETERMINISM"}):
            env.pop(key, None)
    execution_record = output.parent / f"{output.name}.execution.json"
    if execution_record.exists() or execution_record.is_symlink():
        raise FileExistsError("candidate execution record already exists")
    record = {"schema": "cross_tfg_v3_step120_candidate_execution_v1", "status": "RUNNING",
              "plan_raw_sha256": sha_bytes(plan_raw), "plan_manifest_raw_sha256": sha_bytes(manifest_raw),
              "planned_rows_raw_sha256": sha_bytes(rows_raw), "runtime_binding_raw_sha256": sha_bytes(runtime_raw),
              "command": command, "started_unix": time.time(), "host": platform.node(),
              "gpu_uuid": runtime.get("gpu_uuid"), "confirm_audio_read": False}
    execution_record.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(execution_record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(record)); stream.flush(); os.fsync(stream.fileno())
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, env=env, check=False)
    record["child_returncode"] = completed.returncode
    record["finished_unix"] = time.time()
    try:
        startup_raw = guard.safe_regular(startup_receipt_path, "candidate startup policy gate receipt", 256 * 1024)[1]
        startup_receipt = json.loads(startup_raw.decode("utf-8"))
        if (startup_receipt.get("schema") != "cross_tfg_step120_candidate_startup_policy_gate_v1"
                or startup_receipt.get("status") != "PASS"
                or startup_receipt.get("bootstrap_source_sha256") != guard.STRICT_BOOTSTRAP_SHA256
                or startup_receipt.get("policy") != guard.STRICT_BOOTSTRAP_POLICY
                or startup_receipt.get("receipt_sha256") != payload_sha(startup_receipt, "receipt_sha256")):
            raise ValueError("actual candidate dispatcher process did not attest the strict startup policy")
        record["startup_policy_gate_receipt"] = {
            "path": str(startup_receipt_path.resolve(strict=True)), "bytes": len(startup_raw),
            "raw_sha256": sha_bytes(startup_raw), "receipt_sha256": startup_receipt["receipt_sha256"],
            "pid": startup_receipt["pid"],
        }
        deterministic = strict_receipts(plan, runtime, receipt_dir, sha_bytes(plan_raw), sha_bytes(runtime_raw), guard)
        w2l_evidence = validate_w2l_result_evidence(output, plan, runtime, sha_bytes(plan_raw), guard)
        det_path = output / "runtime_determinism_attestation.json"
        det_raw = pretty(deterministic)
        fd = os.open(det_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(det_raw); stream.flush(); os.fsync(stream.fileno())
        record["status"] = "STRICT_POLICY_RECEIPTS_COMPLETE"
        record["strict_determinism_attestation"] = {"path": str(det_path.resolve()), "bytes": len(det_raw),
                                                    "raw_sha256": sha_bytes(det_raw),
                                                    "attestation_sha256": deterministic["attestation_sha256"]}
        record["w2l_native_runtime_evidence"] = w2l_evidence
    except Exception as exc:
        record["status"] = "INCOMPLETE_STRICT_POLICY_RECEIPTS"
        record["strict_policy_error"] = repr(exc)
        fd = os.open(execution_record, os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(pretty(record)); stream.flush(); os.fsync(stream.fileno())
        return completed.returncode or 2
    fd = os.open(execution_record, os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(record)); stream.flush(); os.fsync(stream.fileno())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
