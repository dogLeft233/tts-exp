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
GUARD_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_v3_samehost_dev360_controls_attempt04.py"
BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt02/sitecustomize.py"
CANDIDATE_DISPATCHER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt01.py"
EXPECTED_ROWS = 180
METHOD_CANDIDATE = "shared_recipient_transition_head_v3_fit40_step120"
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
receipt={"schema":"cross_tfg_step120_candidate_startup_policy_gate_v1","status":"PASS","pid":os.getpid(),"python_executable":str(Path(sys.executable).resolve()),"bootstrap_source_sha256":"a2a56ad76a94f345e489bc186409958be201c2ad50bfe91c07a6f2902be59855","policy":policy}
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
        guard.MODELS[0]: (str(Path(exec_binding["generator_child"]).resolve(strict=True)),
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
    output = Path(tools.get("output_root", ""))
    if not output.is_absolute():
        raise ValueError("candidate execution binding lacks an absolute output root")
    helper = guard.load_resource_helper(project_root)
    live, identities = guard.validate_bindings(project_root, plan, runtime, helper, output)
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
    startup_receipt_path = output / "candidate_startup_policy_receipt.json"
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
    })
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
        det_path = output / "runtime_determinism_attestation.json"
        det_raw = pretty(deterministic)
        fd = os.open(det_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(det_raw); stream.flush(); os.fsync(stream.fileno())
        record["status"] = "STRICT_POLICY_RECEIPTS_COMPLETE"
        record["strict_determinism_attestation"] = {"path": str(det_path.resolve()), "bytes": len(det_raw),
                                                    "raw_sha256": sha_bytes(det_raw),
                                                    "attestation_sha256": deterministic["attestation_sha256"]}
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
