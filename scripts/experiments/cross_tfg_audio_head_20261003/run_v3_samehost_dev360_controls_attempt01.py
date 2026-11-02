#!/usr/bin/env python3
"""Preflight and, only with an explicit switch, run the frozen same-host Dev360 controls.

Preflight verifies the exact plan/runtime receipt, current GPU and cgroup-v2/v1
resource identity, source/model/scorer files, and declared capacity gates. It
does not open audio or video. Execution requires a same-host preflight receipt
and --execute-controls; it delegates generation and scoring to the qualified
Stage-A core plus its existing cross-frontend adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
STAGEA_REL = BASE + "/run_stagea_dev_attempt03.py"
STAGEA_BYTES = 67_141
STAGEA_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
CROSS_ADAPTER_REL = BASE + "/run_cross_frontend_dev_attempt03.py"
CROSS_ADAPTER_BYTES = 7_505
CROSS_ADAPTER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
STAGEA_DISPATCHER_REL = BASE + "/run_cross_frontend_samehost_dev_attempt01.py"
STAGEA_DISPATCHER_BYTES = 2_724
STAGEA_DISPATCHER_SHA256 = "48345db1197dccf047c48061a95e9db505e72795e3daeb51fd98616afef63f59"
RESOURCE_HELPER_REL = BASE + "/v3_dev180_newhost_latebind_packet_attempt02.py"
RESOURCE_HELPER_BYTES = 62_830
RESOURCE_HELPER_SHA256 = "406f95e4a262fb62bb02bc67249e6a7bf211b803a014622e042329d772fdf7"
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
QC06 = {
    "preflight_raw_sha256": "c0dd743e3b9b250c88d8ffade8cdaa8c21c18411c40647264f9e9ee5779b3171",
    "terminal_raw_sha256": "a4fa225dc299f1b89c5cd8fc20b915df7f613c51b491280624895eef052f7269",
    "manifest_raw_sha256": "f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436",
    "determinism_policy_sha256": "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc",
    "source_sha256": "cf4d34832ee93bf5a4d2e4be349dad15469e423888c0a1c41bbbdf53f2a4726d",
}
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
METHODS = ("identity", "rms_fixed")
ATTESTATION_SCHEMA = "cross_tfg_v3_samehost_dev360_preflight_v1"
MAX_DEPENDENCY_BYTES = 16 * 1024 * 1024 * 1024


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def safe_regular(path: Path, label: str, limit: int = 16 * 1024 * 1024) -> tuple[Path, bytes]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    return lexical.resolve(strict=True), raw


def hash_regular(path: Path, label: str, expected_sha: str, expected_bytes: int | None = None) -> dict[str, Any]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"{label} traverses a symlink/junction: {cursor}")
    before = lexical.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0:
        raise ValueError(f"{label} is not a regular file: {lexical}")
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise ValueError(f"{label} byte count differs from frozen runtime metadata")
    digest = hashlib.sha256()
    with lexical.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    after = lexical.stat(follow_symlinks=False)
    actual = digest.hexdigest()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or actual != expected_sha.lower()):
        raise ValueError(f"{label} changed or differs from its SHA-256 pin: {lexical}")
    return {"path": str(lexical.resolve(strict=True)), "bytes": before.st_size, "sha256": actual}


def read_json(path: Path, label: str, expected_sha: str | None = None,
              limit: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    _, raw = safe_regular(path, label, limit)
    if expected_sha is not None and sha_bytes(raw) != expected_sha.lower():
        raise ValueError(f"{label} raw SHA mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def load_resource_helper(project_root: Path) -> Any:
    path = project_root / RESOURCE_HELPER_REL
    raw = path.read_bytes()
    if len(raw) != RESOURCE_HELPER_BYTES or sha_bytes(raw) != RESOURCE_HELPER_SHA256:
        raise ValueError("pinned cgroup-aware resource helper differs from the reviewed source")
    spec = importlib.util.spec_from_file_location("samehost_dev360_resource_helper_attempt02", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the cgroup-aware resource helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_plan(project_root: Path, plan_path: Path, plan_sha: str,
                  rows_path: Path, rows_sha: str, manifest_path: Path,
                  manifest_sha: str, runtime_sha: str) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    plan, plan_raw = read_json(plan_path, "frozen controls plan", plan_sha)
    manifest, manifest_raw = read_json(manifest_path, "controls plan manifest", manifest_sha, 4 * 1024 * 1024)
    _, rows_raw = safe_regular(rows_path, "controls planned rows", 32 * 1024 * 1024)
    rows = [json.loads(line.decode("utf-8")) for line in rows_raw.splitlines() if line.strip()]
    if (plan.get("status") != "FROZEN" or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")
            or plan.get("schema_version") != "cross-tfg-audio-eval-v1" or len(plan.get("rows", [])) != 360):
        raise ValueError("controls plan status/payload/schema/denominator mismatch")
    if rows != plan["rows"] or rows_sha.lower() != sha_bytes(rows_raw) or b"".join(canonical(row) + b"\n" for row in rows) != rows_raw:
        raise ValueError("planned row ledger does not byte-bind the exact frozen plan")
    if (manifest.get("schema") != "cross_tfg_v3_samehost_dev360_controls_plan_v1"
            or manifest.get("status") != "FROZEN"
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
            or manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or manifest.get("planned_rows_sha256") != sha_bytes(rows_raw)
            or manifest.get("planned_rows") != 360):
        raise ValueError("controls plan manifest does not bind this plan and 360 rows")
    binding = plan.get("execution", {}).get("samehost_runtime_binding", {})
    if binding.get("raw_sha256") != runtime_sha.lower():
        raise ValueError("frozen plan is not bound to the supplied current-host runtime receipt")
    if tuple(m.get("model_id") for m in plan.get("models", [])) != MODELS:
        raise ValueError("frozen model roster differs from the exact qualified W2L/MuseTalk/Ditto roster")
    if tuple(m.get("method_id") for m in plan.get("methods", [])) != METHODS:
        raise ValueError("frozen control method roster is not identity/RMS")
    if plan.get("execution", {}).get("confirm_audio_read") is not False or plan.get("execution", {}).get("confirm_effects_read") is not False:
        raise ValueError("frozen plan does not keep Confirm sealed")
    source_refs = plan.get("source_reference_plans", {})
    for key, rel, size, digest in (
        ("stagea_core", STAGEA_REL, STAGEA_BYTES, STAGEA_SHA256),
        ("cross_frontend_adapter", CROSS_ADAPTER_REL, CROSS_ADAPTER_BYTES, CROSS_ADAPTER_SHA256),
        ("samehost_stagea_runner", STAGEA_DISPATCHER_REL, STAGEA_DISPATCHER_BYTES, STAGEA_DISPATCHER_SHA256),
    ):
        reference = source_refs.get(key, {})
        if reference.get("path") != rel or reference.get("bytes") != size or reference.get("raw_sha256") != digest:
            raise ValueError(f"frozen plan source reference differs from {key}")
        hash_regular(project_root / rel, key, digest, size)
    expected_closure = {
        "samehost_dev360_plan_builder": BASE + "/build_v3_samehost_dev360_controls_plan_attempt02.py",
        "samehost_dev360_guarded_launcher": BASE + "/run_v3_samehost_dev360_controls_attempt01.py",
        "samehost_dev360_evidence_collector": BASE + "/collect_v3_samehost_dev360_controls_evidence_attempt01.py",
        "samehost_v3_baseline_reader_adapter": BASE + "/build_v3_candidate_samehost_controls_adapter_attempt01.py",
        "samehost_dynamic_540_joiner": BASE + "/join_v3_candidate_540_samehost_controls_attempt01.py",
        "step120_method_materializer": BASE + "/materialize_v3_step120_method_candidate_attempt02.py",
        "step120_samehost_plan_rebinder": BASE + "/rebind_v3_step120_candidate_plan_samehost_attempt01.py",
        "step120_latebind_packet": BASE + "/v3_dev180_newhost_latebind_packet_attempt02.py",
    }
    closure = plan.get("execution", {}).get("samehost_dev360_source_closure")
    if not isinstance(closure, dict) or set(closure) != set(expected_closure):
        raise ValueError("frozen plan does not contain the full same-host Dev360/step120 source closure")
    for key, relative in expected_closure.items():
        pin = closure[key]
        if not isinstance(pin, dict) or pin.get("path") != relative:
            raise ValueError(f"same-host source-closure path differs from its reviewed route: {key}")
        hash_regular(project_root / relative, key, str(pin.get("raw_sha256", "")), pin.get("bytes"))
    return plan, plan_raw, manifest, manifest_raw


def validate_bindings(project_root: Path, plan: Mapping[str, Any], binding: dict[str, Any],
                      helper: Any, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if binding.get("schema") != "cross_tfg_v3_samehost_runtime_binding_v1" or binding.get("status") != "PASS":
        raise ValueError("runtime binding must be a current-host PASS receipt")
    if binding.get("gpu_uuid") != GPU_UUID or binding.get("qc06_source_sha256") != QC06["source_sha256"]:
        raise ValueError("runtime binding GPU/QC06 source identity mismatch")
    if binding.get("determinism_policy_sha256") != QC06["determinism_policy_sha256"]:
        raise ValueError("runtime binding determinism policy differs from QC06")
    qc_receipts = binding.get("qc06_receipts", {})
    for key, digest in QC06.items():
        if qc_receipts.get(key) != digest:
            raise ValueError(f"runtime binding does not cite the exact QC06 receipt {key}")
    resources = binding.get("resources")
    if not isinstance(resources, dict) or resources.get("storage_gate_status") != "PASS":
        raise ValueError("runtime resource/capacity binding is incomplete or pending")
    if resources.get("gpu_uuid") != GPU_UUID:
        raise ValueError("runtime resource receipt is bound to another GPU")
    for key in ("min_gpu_free_mib", "min_effective_ram_available_bytes", "min_dev_shm_free_bytes",
                "output_required_bytes", "output_reserve_bytes", "output_free_bytes",
                "effective_ram_available_bytes", "dev_shm_free_bytes"):
        if type(resources.get(key)) is not int or resources[key] < 0:
            raise ValueError(f"runtime resource field {key} must be an explicit nonnegative integer")
    if resources["output_free_bytes"] < resources["output_required_bytes"] + resources["output_reserve_bytes"]:
        raise ValueError("runtime output capacity does not include required bytes plus reserve")
    basis = resources.get("capacity_basis")
    if not isinstance(basis, dict) or basis.get("status") != "MEASURED_OR_EXPLICITLY_BOUNDED":
        raise ValueError("capacity basis must be measured or explicitly bounded; unknown media bytes cannot pass")
    basis_sha = sha_bytes(canonical(basis))
    if resources.get("capacity_basis_sha256") != basis_sha:
        raise ValueError("capacity-basis payload and raw SHA do not match")
    cgroup = resources.get("cgroup")
    if not isinstance(cgroup, dict) or cgroup.get("version") not in ("v1", "v2"):
        raise ValueError("runtime cgroup identity is missing")

    exec_binding = binding.get("execution_bindings")
    if not isinstance(exec_binding, dict):
        raise ValueError("runtime receipt lacks explicit Stage-A tool and output bindings")
    required = ("output_root", "stagea_python", "syncnet_python", "generator_child",
                "syncnet_root", "syncnet_evaluator", "syncnet_checkpoint", "ffmpeg", "ffprobe",
                "pcm16_lineage", "wav2lip_python", "wav2lip_root")
    for key in required:
        value = exec_binding.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"runtime execution binding {key} must be an explicit absolute path")
    expected_hashes = binding.get("expected_file_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("runtime binding must contain exact tool-file SHA pins")
    tool_hashes = {
        "generator_child": plan.get("protocol", {}).get("generator", {}).get("child_script_sha256"),
        "syncnet_evaluator": plan.get("execution", {}).get("official_syncnet_evaluator_wrapper_sha256"),
        "syncnet_checkpoint": plan.get("protocol", {}).get("syncnet", {}).get("score_model_sha256"),
        "ffmpeg": plan.get("protocol", {}).get("syncnet", {}).get("primary_audio_input", {}).get("ffmpeg_sha256"),
        "pcm16_lineage": plan.get("execution", {}).get("primary_pcm16_lineage_code_sha256"),
    }
    for key, expected in tool_hashes.items():
        if not isinstance(expected, str) or expected_hashes.get(key) != expected:
            raise ValueError(f"execution tool pin does not match the frozen plan: {key}")
        hash_regular(Path(exec_binding[key]), key, expected)
    hash_regular(Path(exec_binding["syncnet_checkpoint"]), "SyncNet checkpoint",
                 expected_hashes["syncnet_checkpoint"])
    ffprobe_sha = expected_hashes.get("ffprobe")
    if not isinstance(ffprobe_sha, str) or len(ffprobe_sha) != 64:
        raise ValueError("runtime binding must pin the FFprobe executable SHA-256")
    hash_regular(Path(exec_binding["ffprobe"]), "FFprobe executable", ffprobe_sha)
    for key in ("stagea_python", "syncnet_python", "wav2lip_python"):
        if not Path(exec_binding[key]).is_file():
            raise FileNotFoundError(f"runtime Python interpreter is absent: {key}")
    for key in ("syncnet_root", "wav2lip_root"):
        if not Path(exec_binding[key]).is_dir():
            raise FileNotFoundError(f"runtime source/model directory is absent: {key}")

    backend_map = binding.get("backends")
    if not isinstance(backend_map, dict) or set(backend_map) != set(MODELS):
        raise ValueError("runtime backends must contain exactly the three frozen models")
    dependencies_by_path: dict[str, dict[str, Any]] = {}
    python_paths: dict[str, str] = {
        "stagea": exec_binding["stagea_python"],
        "syncnet": exec_binding["syncnet_python"],
    }
    backend_directories: dict[str, dict[str, Any]] = {}
    for model in plan["models"]:
        model_id = model["model_id"]
        backend = backend_map[model_id]
        if backend.get("model_binding") != {key: model.get(key) for key in (
                "model_id", "audio_frontend", "model_sha256", "preprocessing_sha256",
                "generator_audio_preprocessing_sha256", "score_model_sha256", "fps_num", "fps_den")}:
            raise ValueError(f"current runtime backend model/scorer identity changed: {model_id}")
        generator = backend.get("generator")
        if not isinstance(generator, dict) or generator.get("fps_num") != model.get("fps_num") or generator.get("fps_den") != model.get("fps_den"):
            raise ValueError(f"current runtime generator descriptor changed frame rate/model: {model_id}")
        generator_python = generator.get("python")
        if not isinstance(generator_python, str) or not Path(generator_python).is_absolute() or not Path(generator_python).is_file():
            raise ValueError(f"runtime generator Python is absent or not absolute: {model_id}")
        backend_key = {MODELS[0]: "wav2lip", MODELS[1]: "muse", MODELS[2]: "ditto"}[model_id]
        python_paths[backend_key] = generator_python
        if model_id == MODELS[0] and Path(generator_python).resolve(strict=True) != Path(exec_binding["wav2lip_python"]).resolve(strict=True):
            raise ValueError("Wav2Lip execution interpreter differs from the frozen backend descriptor")

        directory_values: dict[str, str] = {}
        for key in ("cwd", "repo", "model_source_root", "ditto_source_root", "model_code_root",
                    "model_workspace", "ditto_weights_root"):
            value = generator.get(key)
            if isinstance(value, str) and value:
                directory_values[key] = value
        for key in ("model_overlay_roots",):
            values = generator.get(key)
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, str) and value:
                        directory_values[f"{key}[{index}]"] = value
        env_overrides = generator.get("env", {})
        pythonpath = env_overrides.get("PYTHONPATH") if isinstance(env_overrides, Mapping) else None
        if isinstance(pythonpath, str):
            for index, value in enumerate(item for item in pythonpath.split(os.pathsep) if item):
                directory_values[f"PYTHONPATH[{index}]"] = value
        directory_records = {key: helper.directory_identity(Path(value), f"{model_id} {key}")
                             for key, value in sorted(directory_values.items())}
        backend_directories[model_id] = directory_records
        dependencies = backend.get("dependencies")
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError(f"runtime backend dependency inventory is empty: {model_id}")
        for item in dependencies:
            if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256", "role"}:
                raise ValueError(f"backend dependency receipt schema mismatch: {model_id}")
            path = Path(str(item["path"]))
            if not path.is_absolute() or type(item["bytes"]) is not int or item["bytes"] < 0 or not isinstance(item["role"], str):
                raise ValueError(f"backend dependency receipt has invalid path/size/role: {model_id}")
            previous = dependencies_by_path.get(str(path))
            identity = {"path": str(path), "bytes": item["bytes"], "sha256": item["sha256"]}
            if previous and (previous["bytes"], previous["sha256"]) != (identity["bytes"], identity["sha256"]):
                raise ValueError(f"one dependency path has conflicting pins: {path}")
            dependencies_by_path[str(path)] = identity
    expected_directories = binding.get("backend_directories")
    if expected_directories != backend_directories:
        raise ValueError("current backend directory path/device/inode/mtime snapshots differ from the runtime binding")
    expected_pythons = binding.get("python_environments")
    if not isinstance(expected_pythons, dict) or set(expected_pythons) != set(python_paths):
        raise ValueError("runtime receipt lacks exact Python environment identities for Stage-A, SyncNet, and all backends")
    actual_pythons = {key: helper.probe_python(value) for key, value in sorted(python_paths.items())}
    if expected_pythons != actual_pythons:
        raise ValueError("current Python executable/version/package identities differ from the runtime receipt")
    dependency_total = sum(item["bytes"] for item in dependencies_by_path.values())
    if dependency_total > MAX_DEPENDENCY_BYTES:
        raise ValueError("runtime backend dependency hash budget exceeds 16 GiB")
    verified_dependencies = []
    for item in sorted(dependencies_by_path.values(), key=lambda value: value["path"]):
        verified_dependencies.append(hash_regular(Path(item["path"]), "backend source/model dependency",
                                                  item["sha256"], item["bytes"]))

    thresholds = {
        "min_gpu_free_mib": resources["min_gpu_free_mib"],
        "min_ram_available_bytes": resources["min_effective_ram_available_bytes"],
        "min_shm_free_bytes": resources["min_dev_shm_free_bytes"],
        "min_output_free_bytes": resources["output_required_bytes"] + resources["output_reserve_bytes"],
    }
    current = helper.resource_snapshot(output_root, thresholds)
    gpu = next((item for item in current["gpus"] if item.get("uuid") == GPU_UUID), None)
    if gpu is None:
        raise ValueError("QC06 physical GPU UUID is absent from current nvidia-smi inventory")
    if current["ram"]["cgroup"].get("version") != cgroup.get("version"):
        raise ValueError("current cgroup v1/v2 resource controller differs from the bound runtime identity")
    stable = current["ram"]["cgroup"].get("identity", {})
    stable_expect = cgroup.get("identity")
    if not isinstance(stable_expect, dict):
        raise ValueError("runtime binding lacks the stable cgroup identity record")
    for key in ("version", "mount_target", "mount_root", "membership_path", "leaf_path", "finite_limit", "limit_bytes"):
        if stable.get(key) != stable_expect.get(key):
            raise ValueError(f"current cgroup identity differs from the bound resource receipt: {key}")
    if current["output_disk"].get("target") != resources.get("output_mount_target"):
        raise ValueError("current output mount target differs from the capacity receipt")
    if current["output_disk"].get("available_bytes", -1) < resources["output_required_bytes"] + resources["output_reserve_bytes"]:
        raise ValueError("current output mount cannot fit declared measured/bounded bytes plus reserve")
    return current, {"dependencies": verified_dependencies, "dependency_total_bytes": dependency_total,
                     "tool_file_pins": tool_hashes, "execution_bindings": exec_binding,
                     "python_environments": actual_pythons,
                     "backend_directories": backend_directories}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve(strict=True)
    if project_root != ROOT.resolve(strict=True):
        raise ValueError("project root differs from this source checkout")
    binding, binding_raw = read_json(args.runtime_binding, "same-host runtime binding", args.runtime_binding_raw_sha256, 8 * 1024 * 1024)
    plan, plan_raw, plan_manifest, plan_manifest_raw = validate_plan(
        project_root, args.plan, args.plan_raw_sha256, args.planned_rows, args.planned_rows_raw_sha256,
        args.plan_manifest, args.plan_manifest_raw_sha256, sha_bytes(binding_raw))
    if binding.get("status") != "PASS":
        raise ValueError("runtime binding must be a passing late-bound receipt")
    helper = load_resource_helper(project_root)
    resources, identities = validate_bindings(project_root, plan, binding, helper, Path(binding["execution_bindings"]["output_root"]))
    output_root = Path(binding["execution_bindings"]["output_root"])
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError("Stage-A output root must not exist or must be empty before execution")
    payload = {
        "schema": ATTESTATION_SCHEMA,
        "status": "PASS_METADATA_ONLY",
        "host": platform.node(),
        "created_unix": time.time(),
        "valid_for_seconds": 1800,
        "plan_raw_sha256": sha_bytes(plan_raw),
        "plan_manifest_raw_sha256": sha_bytes(plan_manifest_raw),
        "runtime_binding_raw_sha256": sha_bytes(binding_raw),
        "runtime_gpu_uuid": GPU_UUID,
        "runtime_resources": resources,
        "verified_dependency_files": identities["dependencies"],
        "verified_dependency_bytes": identities["dependency_total_bytes"],
        "verified_tool_pins": identities["tool_file_pins"],
        "verified_python_environments": identities["python_environments"],
        "verified_backend_directories": identities["backend_directories"],
        "execution_bindings": identities["execution_bindings"],
        "development_audio_opened": False,
        "confirm_audio_opened": False,
        "generated_media_opened": False,
        "model_execution_performed": False,
    }
    payload["attestation_sha256"] = payload_sha(payload, "attestation_sha256")
    attestation_path = Path(args.preflight_receipt)
    if attestation_path.exists() or attestation_path.is_symlink():
        raise FileExistsError(f"refusing existing preflight receipt: {attestation_path}")
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(attestation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(payload)); stream.flush(); os.fsync(stream.fileno())
    return {"status": payload["status"], "attestation_path": str(attestation_path.resolve()),
            "attestation_raw_sha256": sha_bytes(attestation_path.read_bytes()),
            "plan_raw_sha256": sha_bytes(plan_raw), "runtime_binding_raw_sha256": sha_bytes(binding_raw),
            "resource_snapshot": resources, "model_execution_performed": False}


def execute(args: argparse.Namespace) -> int:
    if not args.execute_controls:
        raise ValueError("execution requires --execute-controls")
    receipt_sha256 = args.preflight_receipt_raw_sha256
    if (not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in receipt_sha256)):
        raise ValueError("execution requires the exact 64-character preflight receipt raw SHA-256")
    receipt, receipt_raw = read_json(args.preflight_receipt, "same-host controls preflight receipt",
                                    receipt_sha256, 16 * 1024 * 1024)
    if (receipt.get("schema") != ATTESTATION_SCHEMA or receipt.get("status") != "PASS_METADATA_ONLY"
            or receipt.get("attestation_sha256") != payload_sha(receipt, "attestation_sha256")
            or receipt.get("host") != platform.node()
            or time.time() - float(receipt.get("created_unix", 0)) > float(receipt.get("valid_for_seconds", 0))):
        raise ValueError("preflight receipt is invalid, stale, or from another host")
    now = preflight_recheck(args, receipt)
    binding, binding_raw = read_json(args.runtime_binding, "same-host runtime binding", args.runtime_binding_raw_sha256, 8 * 1024 * 1024)
    plan, plan_raw, _manifest, _manifest_raw = validate_plan(
        Path(args.project_root).resolve(strict=True), args.plan, args.plan_raw_sha256,
        args.planned_rows, args.planned_rows_raw_sha256, args.plan_manifest,
        args.plan_manifest_raw_sha256, sha_bytes(binding_raw))
    if (receipt.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or receipt.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)
            or now.get("runtime_resources") is None):
        raise ValueError("preflight does not bind the exact current plan/runtime receipt")
    tools = binding["execution_bindings"]
    out = Path(tools["output_root"])
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Stage-A output root became nonempty after preflight")
    out.mkdir(parents=True, exist_ok=True)
    command = [
        tools["stagea_python"], str(Path(args.project_root) / STAGEA_DISPATCHER_REL),
        "--stagea-core", str(Path(args.project_root) / STAGEA_REL),
        "--cross-frontend-adapter-runner", str(Path(args.project_root) / CROSS_ADAPTER_REL),
        "--plan", str(args.plan.resolve(strict=True)),
        "--planned-rows", str(args.planned_rows.resolve(strict=True)),
        "--expected-plan-raw-sha256", sha_bytes(plan_raw),
        "--output-root", str(out), "--project-root", str(Path(args.project_root).resolve(strict=True)),
        "--wav2lip-python", tools["wav2lip_python"], "--wav2lip-root", tools["wav2lip_root"],
        "--generator-child", tools["generator_child"],
        "--syncnet-root", tools["syncnet_root"], "--syncnet-python", tools["syncnet_python"],
        "--syncnet-checkpoint", tools["syncnet_checkpoint"], "--syncnet-evaluator", tools["syncnet_evaluator"],
        "--ffmpeg", tools["ffmpeg"], "--ffprobe", tools["ffprobe"],
        "--ffmpeg-sha256", binding["expected_file_sha256"]["ffmpeg"],
        "--pcm16-lineage-sha256", binding["expected_file_sha256"]["pcm16_lineage"],
        "--generator-timeout", str(binding.get("timeouts", {}).get("generator_seconds", 900)),
        "--syncnet-timeout", str(binding.get("timeouts", {}).get("syncnet_seconds", 1800)),
        "--ffmpeg-timeout", str(binding.get("timeouts", {}).get("ffmpeg_seconds", 300)),
    ]
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(Path(args.project_root).resolve(strict=True)) + os.pathsep + env.get("PYTHONPATH", ""),
                "CUDA_VISIBLE_DEVICES": str(binding.get("cuda_visible_devices", "0")),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    receipt_copy = dict(receipt)
    receipt_copy["execution_started_unix"] = time.time()
    receipt_copy["execution_command"] = command
    receipt_copy["execution_output_root"] = str(out.resolve())
    execution_record = out.parent / f"{out.name}.execution.json"
    if execution_record.exists() or execution_record.is_symlink():
        raise FileExistsError(f"refusing existing execution record: {execution_record}")
    execution_record.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(execution_record, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(receipt_copy)); stream.flush(); os.fsync(stream.fileno())
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, env=env, check=False)
    receipt_copy["execution_finished_unix"] = time.time()
    receipt_copy["child_returncode"] = completed.returncode
    fd = os.open(execution_record, os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(pretty(receipt_copy)); stream.flush(); os.fsync(stream.fileno())
    return completed.returncode


def preflight_recheck(args: argparse.Namespace, previous: Mapping[str, Any]) -> dict[str, Any]:
    """Re-evaluate runtime gates without creating another receipt or touching media."""
    project_root = Path(args.project_root).resolve(strict=True)
    binding, binding_raw = read_json(args.runtime_binding, "same-host runtime binding", args.runtime_binding_raw_sha256, 8 * 1024 * 1024)
    plan, plan_raw, _manifest, _manifest_raw = validate_plan(
        project_root, args.plan, args.plan_raw_sha256, args.planned_rows, args.planned_rows_raw_sha256,
        args.plan_manifest, args.plan_manifest_raw_sha256, sha_bytes(binding_raw))
    helper = load_resource_helper(project_root)
    resources, _ids = validate_bindings(project_root, plan, binding, helper,
                                        Path(binding["execution_bindings"]["output_root"]))
    if (previous.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or previous.get("plan_manifest_raw_sha256") != sha_bytes(_manifest_raw)
            or previous.get("runtime_binding_raw_sha256") != sha_bytes(binding_raw)):
        raise ValueError("preflight receipt belongs to a different plan or runtime binding")
    old_resources = previous.get("runtime_resources", {})
    old_gpu = next((item for item in old_resources.get("gpus", []) if item.get("uuid") == GPU_UUID), None)
    new_gpu = next((item for item in resources.get("gpus", []) if item.get("uuid") == GPU_UUID), None)
    if old_gpu is None or new_gpu is None or old_gpu.get("uuid") != new_gpu.get("uuid"):
        raise ValueError("physical GPU UUID changed after preflight")
    old_cgroup = old_resources.get("ram", {}).get("cgroup", {}).get("identity", {})
    new_cgroup = resources.get("ram", {}).get("cgroup", {}).get("identity", {})
    if old_cgroup != new_cgroup:
        raise ValueError("cgroup resource-controller identity changed after preflight")
    if resources.get("output_disk", {}).get("available_bytes", -1) < binding["resources"]["output_required_bytes"] + binding["resources"]["output_reserve_bytes"]:
        raise ValueError("current output capacity fell below its measured/bounded requirement plus reserve")
    return {"runtime_resources": resources}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--execute-controls", action="store_true", help="explicitly authorize Dev360 rendering/scoring")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-raw-sha256", required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-manifest-raw-sha256", required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--planned-rows-raw-sha256", required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--runtime-binding-raw-sha256", required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt-raw-sha256")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        result = preflight(args)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
