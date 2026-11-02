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


SCHEMA = "cross_tfg_fit40_transition_head_v3_launch_receipt_v1"
BINDING_SCHEMA = "cross_tfg_fit40_native_transition_head_v3_late_binding_v1"


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
    }
    if scope != expected_scope:
        raise LaunchError("fit scope differs from the frozen 40-row, 120-step training scope")
    if binding.get("resource_gate_required_before_launch") != {
        "stable_samples": 2,
        "sample_interval_seconds": 5,
        "minimum_free_gpu_mib": 18432,
        "unknown_compute_pids": "wait; never terminate other processes",
        "minimum_shm_free_bytes": 4294967296,
        "same_uuid_as_geometry_producer": binding.get("device", {}).get("uuid"),
    }:
        raise LaunchError("resource policy differs from frozen 18 GiB/4 GiB/zero-peer admission")
    device = binding.get("device")
    if not isinstance(device, dict) or not device.get("uuid") or not str(device.get("literal", "")).startswith("cuda"):
        raise LaunchError("bound training device and UUID are missing")
    argv = binding.get("argv")
    env = binding.get("environment")
    cwd = Path(str(binding.get("working_directory", "")))
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise LaunchError("bound argv is malformed")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise LaunchError("bound environment is malformed")
    runner = binding.get("runner")
    if not isinstance(runner, dict) or len(argv) < 3 or argv[1] != "-B" or argv[2] != runner.get("path"):
        raise LaunchError("argv does not name the pinned fit runner")
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
    args = parser.parse_args()
    if len(args.expected_binding_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.expected_binding_sha256.lower()):
        raise LaunchError("binding SHA must be a 64-character hexadecimal digest")
    binding, raw = load_binding(args.binding, args.expected_binding_sha256)
    argv, env, cwd, output_dir, summary = validate_binding(binding)
    if args.wait_timeout_seconds <= 0:
        raise LaunchError("wait timeout must be positive")
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
        good = bool(snapshot["free_ok"]) and not snapshot["compute_pids"] and shm_free >= minimum_shm
        current = {"observed_unix_seconds": time.time(), **snapshot, "shm_free_bytes": shm_free, "pass": good}
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
    if not last["free_ok"] or last["compute_pids"] or shm_free < minimum_shm:
        raise LaunchError("GPU/SHM admission was lost before launch")

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
        "child": {"pid": child.pid, "start_ticks": ticks, "argv": argv, "working_directory": str(cwd)},
        "verified_unique_pinned_files": summary["pinned_file_count"],
        "resource_admission_samples": stable,
        "final_resource_sample": {"gpu": last["gpu"], "shm_free_bytes": shm_free},
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
