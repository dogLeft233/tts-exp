#!/usr/bin/env python3
"""CPU-only preflight for the frozen 540-row shared-profile dev execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return value


def run_version(python: str, expected_numpy: str) -> dict:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    code = (
        "import json,numpy,torch; "
        "initialized=torch.cuda.is_initialized(); available=torch.cuda.is_available(); "
        "print(json.dumps({'numpy':numpy.__version__,'torch':str(torch.__version__),"
        "'cuda_initialized':bool(initialized),'cuda_available':bool(available)},sort_keys=True,separators=(',',':')))"
    )
    proc = subprocess.run([python, "-B", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env, timeout=90, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"runtime import probe failed for {python}: {proc.stderr[-3000:]}")
    value = json.loads(proc.stdout.splitlines()[-1])
    if value.get("numpy") != expected_numpy or value.get("torch") != "2.5.1+cu124":
        raise RuntimeError(f"runtime version mismatch for {python}: {value}")
    if value.get("cuda_initialized") is not False:
        raise RuntimeError(f"CPU-only preflight initialized CUDA: {python}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--shm-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--planned-rows", required=True, type=Path)
    parser.add_argument("--bundle-manifest", required=True, type=Path)
    parser.add_argument("--execution-overlay-manifest", required=True, type=Path)
    parser.add_argument("--gpu-policy", required=True, type=Path)
    parser.add_argument("--gpu-wrapper", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True, type=Path)
    parser.add_argument("--dispatcher", required=True, type=Path)
    parser.add_argument("--stagea-core", required=True, type=Path)
    parser.add_argument("--adapter-runner", required=True, type=Path)
    parser.add_argument("--profile-source", required=True, type=Path)
    parser.add_argument("--generator-child", required=True, type=Path)
    parser.add_argument("--pcm16-lineage", required=True, type=Path)
    parser.add_argument("--syncnet-evaluator", required=True, type=Path)
    parser.add_argument("--syncnet-root", required=True, type=Path)
    parser.add_argument("--syncnet-python", required=True)
    parser.add_argument("--wav2lip-python", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-bundle-manifest-sha256", required=True)
    parser.add_argument("--expected-overlay-manifest-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-entrypoint-sha256", required=True)
    parser.add_argument("--expected-output-free-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    task = args.task_root.resolve(strict=True)
    shm = args.shm_root.resolve(strict=True)
    code_root = args.code_root.resolve(strict=True)
    plan_path = args.plan.resolve(strict=True)
    rows_path = args.planned_rows.resolve(strict=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("preflight must keep CUDA hidden")

    expected_files = {
        args.plan: args.expected_plan_sha256,
        args.planned_rows: args.expected_rows_sha256,
        args.bundle_manifest: args.expected_bundle_manifest_sha256,
        args.execution_overlay_manifest: args.expected_overlay_manifest_sha256,
        args.gpu_policy: args.expected_policy_sha256,
        args.entrypoint: args.expected_entrypoint_sha256,
    }
    for raw_path, expected in expected_files.items():
        path = raw_path.resolve(strict=True)
        if len(expected) != 64 or sha256_file(path) != expected.lower():
            raise RuntimeError(f"preflight file SHA mismatch: {path}")

    bundle = read_json(args.bundle_manifest)
    if bundle.get("schema") != "cross_tfg_profile_candidate_execution_bundle_v1":
        raise RuntimeError("unexpected execution bundle schema")
    if bundle.get("bundle_sha256") != args.expected_bundle_sha256.lower():
        raise RuntimeError("bundle manifest does not bind the expected archive SHA")
    for item in bundle.get("bundle_members", []):
        rel = PurePosixPath(str(item.get("path", "")))
        if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
            raise RuntimeError(f"unsafe bundle manifest path: {rel}")
        path = task.joinpath(*rel.parts).resolve(strict=True)
        if task not in path.parents or path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"extracted bundle member differs from manifest: {rel}")

    plan = read_json(plan_path)
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    payload_sha = hashlib.sha256(canonical(payload)).hexdigest()
    rows_raw = rows_path.read_bytes()
    rows = [json.loads(line) for line in rows_raw.decode("utf-8").splitlines() if line.strip()]
    if plan.get("status") != "FROZEN" or hashlib.sha256(canonical(payload)).hexdigest() != plan.get("plan_sha256"):
        raise RuntimeError("frozen plan canonical payload seal failed")
    if len(rows) != 540 or len(plan.get("rows", [])) != 540 or [x["row_id"] for x in rows] != [x["row_id"] for x in plan["rows"]]:
        raise RuntimeError("frozen plan/ordered row count mismatch")
    if len(plan.get("models", [])) != 3 or len(plan.get("methods", [])) != 3:
        raise RuntimeError("execution roster must remain three frontends by three selected profiles")
    if any(row.get("split") != "dev" for row in rows) or plan.get("generation_repeat_qc"):
        raise RuntimeError("execution rows/repeat scope differs from the frozen dev-only candidate plan")
    if plan.get("execution", {}).get("confirm_audio_read") is not False or plan.get("execution", {}).get("confirm_effects_read") is not False:
        raise RuntimeError("confirmation scope is not closed")
    if sha256_file(plan_path) != args.expected_plan_sha256.lower() or sha256_file(rows_path) != args.expected_rows_sha256.lower():
        raise RuntimeError("frozen plan or planned rows raw SHA mismatch")
    overlay = read_json(args.execution_overlay_manifest)
    if (overlay.get("schema") != "cross_tfg_execution_overlay_v1" or overlay.get("run_id") != plan.get("run_id")
            or overlay.get("plan_raw_sha256") != args.expected_plan_sha256.lower()
            or overlay.get("entrypoint_sha256") != args.expected_entrypoint_sha256.lower()):
        raise RuntimeError("execution overlay manifest is not bound to the frozen plan/entrypoint")
    policy = read_json(args.gpu_policy)
    if (policy.get("schema") != "cross_tfg_gpu_admission_policy_v1" or policy.get("fit_run_id") != plan.get("run_id")
            or policy.get("fit_plan_path") != str(plan_path) or policy.get("fit_plan_sha256") != args.expected_plan_sha256.lower()
            or policy.get("fit_worker_wrapper_path") != str(args.gpu_wrapper.resolve(strict=True))
            or policy.get("fit_worker_wrapper_sha256") != sha256_file(args.gpu_wrapper)):
        raise RuntimeError("GPU admission policy does not bind the exact candidate execution")
    if int(policy.get("minimum_free_mib", 0)) < 18432 or int(policy.get("stable_samples", 0)) < 2:
        raise RuntimeError("GPU admission policy is weaker than the frozen >=18 GiB/two-sample gate")
    if policy.get("competitors") != []:
        raise RuntimeError("unexpected unqualified GPU worker competitor declaration")

    source_expectations = {
        args.dispatcher: "dacda9729db79ec96c73c19402866b78fb3d695b2560ca6f285ce4ce0c056660",
        args.stagea_core: "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94",
        args.adapter_runner: "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc",
        args.profile_source: "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253",
        args.generator_child: "1c04011b79b38a4cff7a0235bc960acc12ee3f9aafc333bbf1633ab25eb3a29d",
        args.pcm16_lineage: "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0",
        args.syncnet_evaluator: "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b",
        args.gpu_wrapper: "411be16642d1e3c31576ec837b887d0e09303cdf3f95a2f5bdd48e5c683e9ef9",
    }
    for path, expected in source_expectations.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen source closure SHA mismatch: {path}")
    if not (args.syncnet_root.resolve(strict=True) / "run_pipeline.py").is_file():
        raise RuntimeError("pinned SyncNet source root is incomplete")
    if not Path(args.syncnet_python).is_file() or not Path(args.wav2lip_python).is_file():
        raise RuntimeError("pinned Python interpreter path is missing")
    if args.output_root.exists():
        raise RuntimeError("refusing an existing candidate output root")
    if not args.control_root.is_dir():
        raise RuntimeError("job-control root has not been prepared")

    free_bytes = shutil.disk_usage(shm).free
    if free_bytes < args.expected_output_free_bytes:
        raise RuntimeError(f"insufficient SHM output reserve: {free_bytes} < {args.expected_output_free_bytes}")
    runtime_w2l = run_version(args.wav2lip_python, "1.26.4")
    runtime_sync = run_version(args.syncnet_python, "2.2.6")
    import torch
    import numpy
    if torch.__version__ != "2.5.1+cu124" or numpy.__version__ != "1.26.4" or torch.cuda.is_initialized():
        raise RuntimeError("preflight process runtime differs from the pinned Wav2Lip environment")
    package_dir = code_root / "scripts/audio_enhancement"
    sys.path.insert(0, str(code_root))
    import scripts.audio_enhancement.shared_frequency_profile_v1 as profile
    if Path(profile.__file__).resolve() != (package_dir / "shared_frequency_profile_v1.py").resolve():
        raise RuntimeError("profile import resolved outside the staged source closure")

    gate_spec = importlib.util.spec_from_file_location("shared_profile_gpu_admission_preflight", args.gpu_wrapper.resolve(strict=True))
    if gate_spec is None or gate_spec.loader is None:
        raise RuntimeError("could not import the pinned GPU admission wrapper")
    gate_module = importlib.util.module_from_spec(gate_spec)
    sys.modules[gate_spec.name] = gate_module
    gate_spec.loader.exec_module(gate_module)
    gate = gate_module.Gate(
        policy_path=args.gpu_policy, policy_sha=args.expected_policy_sha256,
        plan_path=plan_path, plan_sha=args.expected_plan_sha256,
        entry_path=args.entrypoint, entry_sha=args.expected_entrypoint_sha256,
        core_path=args.stagea_core, core_sha=source_expectations[args.stagea_core],
        child_path=args.generator_child, child_sha=source_expectations[args.generator_child],
        sync_root=args.syncnet_root, sync_python=args.syncnet_python,
        evaluator_path=args.syncnet_evaluator, evaluator_sha=source_expectations[args.syncnet_evaluator],
        event_log=args.control_root / "preflight_attempt01/gpu_admission_events.jsonl",
    )
    gpu_apps = gate_module.gpu_apps()
    used, free = gate_module.memory()
    roots = gate.roots()
    event = {
        "schema": "cross_tfg_shared_profile_candidate_execution_preflight_v1",
        "status": "PREFLIGHT_PASS",
        "run_id": plan["run_id"], "plan_raw_sha256": args.expected_plan_sha256.lower(),
        "plan_payload_sha256": payload_sha, "planned_rows_sha256": args.expected_rows_sha256.lower(),
        "planned_rows": len(rows), "methods": [x["method_id"] for x in plan["methods"]],
        "models": [x["model_id"] for x in plan["models"]],
        "bundle_sha256": args.expected_bundle_sha256.lower(),
        "bundle_manifest_sha256": args.expected_bundle_manifest_sha256.lower(),
        "overlay_manifest_sha256": args.expected_overlay_manifest_sha256.lower(),
        "gpu_policy_sha256": args.expected_policy_sha256.lower(),
        "entrypoint_sha256": args.expected_entrypoint_sha256.lower(),
        "shm_free_bytes": free_bytes, "required_shm_free_bytes": args.expected_output_free_bytes,
        "runtime_wav2lip": runtime_w2l, "runtime_syncnet": runtime_sync,
        "gpu_used_mib": used, "gpu_free_mib": free, "visible_compute_apps": gpu_apps,
        "registered_process_roots": roots,
        "gpu_admission_applies_before": policy["admission_rule"],
        "confirm_audio_read": False, "confirm_effects_read": False,
        "model_forward_performed": False,
    }
    receipt_path = args.receipt.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(event) + b"\n"
    with receipt_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    event["receipt_path"] = str(receipt_path)
    event["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    event["receipt_bytes"] = len(raw)
    print(json.dumps(event, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
