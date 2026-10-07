#!/usr/bin/env python3
"""Fail-closed same-GPU admission and detached launch for fit40/V3/120-step training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "cross_tfg_fit40_transition_head_v3_attempt09_launch_receipt_v1"
BINDING_SCHEMA = "cross_tfg_fit40_transition_head_v3_late_binding_attempt09_v1"
MIN_CGROUP_MEMORY_AVAILABLE_BYTES = 8_589_934_592
EXPECTED_GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
EXPECTED_RUNNER_BYTES = 78238
EXPECTED_RUNNER_SHA256 = "cade46bf035a117f8ef8ca4bb8ce1b20c86c11c1b0ed111fbd87b8e1c8a9f24a"


class LaunchError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise LaunchError(f"refusing to overwrite launch receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def verify_pins(value: Any) -> int:
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if {"path", "bytes", "sha256"}.issubset(node) and isinstance(node.get("path"), str):
                path = Path(node["path"])
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    if path.is_symlink() or not path.is_file():
                        raise LaunchError(f"pinned launch input is missing or unsafe: {path}")
                    if path.stat().st_size != int(node["bytes"]) or sha_file(path) != str(node["sha256"]).lower():
                        raise LaunchError(f"pinned launch input changed: {path}")
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return len(seen)


def proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = raw.rfind(")")
    if end < 0:
        raise LaunchError(f"cannot parse process stat for pid {pid}")
    fields = raw[end + 2 :].split()
    if len(fields) <= 19:
        raise LaunchError(f"process stat is truncated for pid {pid}")
    return int(fields[19])


def gpu_snapshot(expected_uuid: str, minimum_free_mib: int) -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,index,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    selected = None
    for line in gpu.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise LaunchError("nvidia-smi returned an unparseable GPU row")
        if parts[0].lower() == expected_uuid.lower():
            selected = {
                "uuid": parts[0],
                "index": int(parts[1]),
                "memory_free_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
            }
    if selected is None:
        raise LaunchError("the geometry producer GPU UUID is absent")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    peers: list[str] = []
    for line in apps.stdout.splitlines():
        pid = line.strip()
        if pid and pid.lower() not in ("no running processes found", "n/a"):
            peers.append(pid)
    return {
        "gpu": selected,
        "compute_pids": peers,
        "free_ok": selected["memory_free_mib"] >= minimum_free_mib,
    }


def cgroup_memory_snapshot() -> dict[str, Any]:
    root = Path("/sys/fs/cgroup")
    maximum_path = root / "memory.max"
    current_path = root / "memory.current"
    if not maximum_path.is_file() or not current_path.is_file():
        return {"root": str(root), "memory_max_bytes": None, "memory_current_bytes": None, "available_bytes": None}
    maximum_raw = maximum_path.read_text(encoding="ascii").strip()
    current = int(current_path.read_text(encoding="ascii").strip())
    if maximum_raw == "max":
        available = None
        maximum = None
    else:
        maximum = int(maximum_raw)
        available = maximum - current
    return {
        "root": str(root),
        "memory_max_bytes": maximum,
        "memory_current_bytes": current,
        "available_bytes": available,
    }


def load_binding(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise LaunchError("launch binding is absent or unsafe")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256.lower():
        raise LaunchError("launch binding raw SHA-256 mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"launch binding is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError("launch binding must be a JSON object")
    return value, raw


def validate_binding(binding: dict[str, Any]) -> tuple[list[str], dict[str, str], Path, Path, dict[str, Any]]:
    if binding.get("schema") != BINDING_SCHEMA or binding.get("status") != "READY_FIT_ONLY_TRAINING_LAUNCH_BINDING":
        raise LaunchError("binding schema/status is not the expected fit40 training launch binding")
    if binding.get("execution_started") is not False or binding.get("optimizer_created_or_updated") is not False:
        raise LaunchError("binding already records execution or optimizer updates")
    scope = binding.get("scope")
    expected_scope = {
        "fit_rows": 40,
        "fit_speakers": 20,
        "canonical_endpoints": [0, 15],
        "optimizer_steps": 120,
        "pass_count": 3,
        "seed": 20261003,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "model_selection": False,
        "peak_ceiling": None,
        "determinism_policy": {
            "torch_use_deterministic_algorithms": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "cublas_workspace_config": ":4096:8",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "pythonhashseed": "20261003",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "openblas_num_threads": "1",
        },
    }
    if scope != expected_scope:
        raise LaunchError("fit scope/determinism differs from the frozen fit40/zero40/120-step attempt09 protocol")
    if binding.get("resource_gate_required_before_launch") != {
        "stable_samples": 2,
        "sample_interval_seconds": 5,
        "minimum_free_gpu_mib": 18432,
        "unknown_compute_pids": "wait; never terminate other processes",
        "minimum_shm_free_bytes": 4294967296,
        "same_gpu_uuid_as_geometry_producer": EXPECTED_GPU_UUID,
    }:
        raise LaunchError("resource policy differs from frozen attempt09 18 GiB/4 GiB/zero-peer admission")
    device = binding.get("device")
    if (not isinstance(device, dict) or device.get("uuid") != EXPECTED_GPU_UUID
            or device.get("index") != 0 or device.get("literal") != "cuda:0"):
        raise LaunchError("binding is not pinned to the actual QC06/producer GPU UUID and CUDA device")
    producer = binding.get("fit40_geometry_and_cache")
    identity = producer.get("runtime_gpu_identity") if isinstance(producer, dict) else None
    if not isinstance(identity, dict) or identity.get("uuid") != EXPECTED_GPU_UUID or identity.get("device_literal") != "cuda:0" or identity.get("device_index") != 0:
        raise LaunchError("actual producer runtime identity differs from the admitted attempt09 GPU")
    profile = identity.get("runtime_profile")
    strict_profile = {
        "seed": 20261003, "torch_num_threads": 1, "torch_num_interop_threads": 1,
        "torch_use_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
        "frame_rate": {"numerator": 25, "denominator": 1}, "source_sample_rate_hz": 16000,
        "python_version": "3.10.12", "torch_version": "2.5.1+cu124", "numpy_version": "1.26.4",
    }
    if not isinstance(profile, dict) or any(profile.get(k) != v for k, v in strict_profile.items()):
        raise LaunchError("actual producer runtime profile differs from strict QC06 deterministic/25-1 policy")
    argv = binding.get("argv")
    env = binding.get("environment")
    cwd = Path(str(binding.get("working_directory", "")))
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise LaunchError("bound argv is malformed")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise LaunchError("bound environment is malformed")
    exact_env = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "20261003", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if any(env.get(k) != v for k, v in exact_env.items()):
        raise LaunchError("bound runtime environment differs from strict attempt09 deterministic policy")
    if not env.get("CROSS_TFG_PSF_ROOT") or env.get("CUDA_VISIBLE_DEVICES") != str(profile.get("cuda_visible_devices")):
        raise LaunchError("bound environment does not match the producer runtime profile")
    runner = binding.get("runner")
    if (not isinstance(runner, dict) or int(runner.get("bytes", -1)) != EXPECTED_RUNNER_BYTES
            or str(runner.get("sha256", "")).lower() != EXPECTED_RUNNER_SHA256
            or len(argv) < 3 or argv[1] != "-B" or argv[2] != runner.get("path")):
        raise LaunchError("argv does not name the exact pinned attempt09 fit runner")
    if argv[0] != profile.get("python_executable_literal"):
        raise LaunchError("bound Python executable differs from the producer's qualified runtime")
    if not cwd.is_dir() or cwd.is_symlink():
        raise LaunchError("bound working directory is missing or unsafe")
    output = Path(str(binding.get("training_output_dir", "")))
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.is_symlink():
        raise LaunchError("fresh training output path is occupied or its parent is unsafe")
    pins = verify_pins(binding)
    return [str(x) for x in argv], {str(k): str(v) for k, v in env.items()}, cwd, output, {"pinned_file_count": pins, "uuid": str(device["uuid"]), "device": str(device["literal"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--wait-timeout-seconds", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--expected-preflight-sha256")
    args = parser.parse_args()
    if len(args.expected_binding_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.expected_binding_sha256.lower()):
        raise LaunchError("binding SHA must be a 64-character hexadecimal digest")
    binding, raw = load_binding(args.binding, args.expected_binding_sha256)
    argv, env, cwd, output_dir, summary = validate_binding(binding)
    if args.wait_timeout_seconds <= 0:
        raise LaunchError("wait timeout must be positive")
    if args.preflight_only:
        if args.preflight_receipt is None or args.expected_preflight_sha256 is not None:
            raise LaunchError("preflight mode requires a fresh preflight receipt path and no preflight SHA")
        for path in (args.preflight_receipt, args.launch_receipt, args.stdout_log, args.stderr_log):
            if path.exists() or path.is_symlink():
                raise LaunchError(f"preflight/launch evidence path is already occupied: {path}")
        admission = binding["resource_gate_required_before_launch"]
        expected_uuid = str(binding["device"]["uuid"])
        minimum_free = int(admission["minimum_free_gpu_mib"])
        minimum_shm = int(admission["minimum_shm_free_bytes"])
        samples = []
        deadline = time.monotonic() + args.wait_timeout_seconds
        while time.monotonic() < deadline and len(samples) < 2:
            snap = gpu_snapshot(expected_uuid, minimum_free)
            shm_free = shutil.disk_usage("/dev/shm").free
            cgroup = cgroup_memory_snapshot()
            cgroup_ok = cgroup["available_bytes"] is not None and int(cgroup["available_bytes"]) >= MIN_CGROUP_MEMORY_AVAILABLE_BYTES
            good = bool(snap["free_ok"]) and not snap["compute_pids"] and shm_free >= minimum_shm and cgroup_ok
            sample = {"observed_unix_seconds": time.time(), **snap, "shm_free_bytes": shm_free,
                      "cgroup_memory": cgroup, "cgroup_memory_available_ok": cgroup_ok, "pass": good}
            if not good:
                raise LaunchError("GPU/SHM/cgroup preflight admission failed")
            if samples and sample["observed_unix_seconds"] - samples[-1]["observed_unix_seconds"] < 5.0:
                raise LaunchError("preflight samples are closer than the frozen five seconds")
            samples.append(sample)
            if len(samples) < 2:
                time.sleep(5)
        if len(samples) != 2:
            raise LaunchError("preflight did not complete two stable resource samples")
        preflight = {
            "schema": "cross_tfg_fit40_transition_head_v3_attempt09_preflight_v1",
            "status": "PASS_BINDING_AND_RESOURCE_PREFLIGHT_ONLY",
            "binding": {"path": str(args.binding.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "launcher": {"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size, "sha256": sha_file(Path(__file__).resolve())},
            "device": binding["device"], "resource_samples": samples,
            "minimum_cgroup_memory_available_bytes": MIN_CGROUP_MEMORY_AVAILABLE_BYTES,
            "training_output_path_absent": not output_dir.exists(),
            "optimizer_updated_by_preflight": False, "training_started_by_preflight": False,
            "observed_unix_seconds": time.time(),
        }
        preflight_pin = write_once(args.preflight_receipt, preflight)
        print(json.dumps({"status": preflight["status"], "preflight_receipt": preflight_pin,
                          "binding_sha256": preflight["binding"]["sha256"]}, sort_keys=True))
        return 0

    if args.expected_preflight_sha256 is None or len(args.expected_preflight_sha256) != 64:
        raise LaunchError("launch requires the actual preflight receipt SHA-256")
    if args.preflight_receipt is None or args.preflight_receipt.is_symlink() or not args.preflight_receipt.is_file():
        raise LaunchError("actual preflight receipt is absent or unsafe")
    preflight_raw = args.preflight_receipt.read_bytes()
    if hashlib.sha256(preflight_raw).hexdigest() != args.expected_preflight_sha256.lower():
        raise LaunchError("preflight raw SHA-256 mismatch")
    preflight_value = json.loads(preflight_raw.decode("utf-8"))
    if (preflight_value.get("schema") != "cross_tfg_fit40_transition_head_v3_attempt09_preflight_v1"
            or preflight_value.get("status") != "PASS_BINDING_AND_RESOURCE_PREFLIGHT_ONLY"
            or preflight_value.get("binding", {}).get("sha256") != hashlib.sha256(raw).hexdigest()
            or preflight_value.get("device", {}).get("uuid") != EXPECTED_GPU_UUID
            or preflight_value.get("training_output_path_absent") is not True):
        raise LaunchError("preflight receipt does not bind this actual binding/GPU/output path")
    for path in (args.launch_receipt, args.stdout_log, args.stderr_log):
        if path.exists() or path.is_symlink():
            raise LaunchError(f"launch evidence path is already occupied: {path}")

    admission = binding["resource_gate_required_before_launch"]
    expected_uuid = str(binding["device"]["uuid"])
    minimum_free = int(admission["minimum_free_gpu_mib"])
    minimum_shm = int(admission["minimum_shm_free_bytes"])
    stable: list[dict[str, Any]] = []
    start = time.monotonic()
    deadline = start + args.wait_timeout_seconds
    while time.monotonic() < deadline:
        snapshot = gpu_snapshot(expected_uuid, minimum_free)
        shm_free = shutil.disk_usage("/dev/shm").free
        cgroup_memory = cgroup_memory_snapshot()
        cgroup_ok = (
            cgroup_memory["available_bytes"] is not None
            and int(cgroup_memory["available_bytes"]) >= MIN_CGROUP_MEMORY_AVAILABLE_BYTES
        )
        good = bool(snapshot["free_ok"]) and not snapshot["compute_pids"] and shm_free >= minimum_shm and cgroup_ok
        current = {
            "observed_unix_seconds": time.time(),
            **snapshot,
            "shm_free_bytes": shm_free,
            "cgroup_memory": cgroup_memory,
            "cgroup_memory_available_ok": cgroup_ok,
            "pass": good,
        }
        if not good:
            stable = []
        elif not stable:
            stable = [current]
        elif current["observed_unix_seconds"] - stable[-1]["observed_unix_seconds"] >= 5.0:
            stable.append(current)
        else:
            stable = [current]
        if len(stable) >= 2:
            break
        time.sleep(5)
    if len(stable) < 2:
        raise LaunchError("GPU/SHM admission did not reach two stable 5-second samples")

    # Recheck all input seals and resource admission immediately before launching.
    binding2, raw2 = load_binding(args.binding, args.expected_binding_sha256)
    if raw2 != raw:
        raise LaunchError("launch binding changed after admission")
    summary["pinned_file_count"] = verify_pins(binding2)
    last = gpu_snapshot(expected_uuid, minimum_free)
    shm_free = shutil.disk_usage("/dev/shm").free
    cgroup_memory = cgroup_memory_snapshot()
    cgroup_ok = (
        cgroup_memory["available_bytes"] is not None
        and int(cgroup_memory["available_bytes"]) >= MIN_CGROUP_MEMORY_AVAILABLE_BYTES
    )
    if not last["free_ok"] or last["compute_pids"] or shm_free < minimum_shm or not cgroup_ok:
        raise LaunchError("GPU/SHM/cgroup-memory admission was lost before launch")

    child_env = os.environ.copy()
    child_env.update(env)
    args.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with args.stdout_log.open("xb") as stdout, args.stderr_log.open("xb") as stderr:
        child = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
    ticks = proc_start_ticks(child.pid)
    receipt = {
        "schema": SCHEMA,
        "status": "STARTED",
        "binding": {"path": str(args.binding.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "preflight": {"path": str(args.preflight_receipt.resolve()), "bytes": len(preflight_raw), "sha256": hashlib.sha256(preflight_raw).hexdigest()},
        "launcher": {"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size, "sha256": sha_file(Path(__file__).resolve())},
        "child": {"pid": child.pid, "start_ticks": ticks, "argv": argv, "working_directory": str(cwd)},
        "verified_unique_pinned_files": summary["pinned_file_count"],
        "resource_admission_samples": stable,
        "minimum_cgroup_memory_available_bytes": MIN_CGROUP_MEMORY_AVAILABLE_BYTES,
        "final_resource_sample": {"gpu": last["gpu"], "shm_free_bytes": shm_free, "cgroup_memory": cgroup_memory},
        "stdout_log": str(args.stdout_log.resolve()),
        "stderr_log": str(args.stderr_log.resolve()),
        "development_audio_opened_by_launcher": False,
        "confirmation_audio_opened_by_launcher": False,
        "optimizer_updated_by_launcher": False,
        "launched_unix_seconds": time.time(),
    }
    receipt_pin = write_once(args.launch_receipt, receipt)
    print(json.dumps({"status": "STARTED", "pid": child.pid, "start_ticks": ticks, "launch_receipt": receipt_pin}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "LAUNCH_FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
