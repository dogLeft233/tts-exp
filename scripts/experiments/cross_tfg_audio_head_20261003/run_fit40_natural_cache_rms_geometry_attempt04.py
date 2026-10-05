#!/usr/bin/env python3
"""Fail-closed admission and supervisor for the fit40 native RMS geometry run.

This standard-library wrapper records structured preflight and per-row terminal
state while the pinned producer performs the original RMS/full512/cache work.
It performs no model math itself and never retries or deletes partial evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


class RunnerError(RuntimeError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RunnerError(f"expected JSON object at {path}")
    return raw, value


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return {"path": str(path), "bytes": len(raw), "sha256": sha_bytes(raw)}


def file_pin(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink():
        raise RunnerError(f"pinned input is symlinked: {path}")
    resolved = path.resolve(strict=True)
    st = resolved.stat()
    if not resolved.is_file():
        raise RunnerError(f"pinned input is not a regular nonsymlink file: {resolved}")
    digest = sha_file(resolved)
    if st.st_size != int(expected["bytes"]) or digest != str(expected["sha256"]).lower():
        raise RunnerError(f"pinned input mismatch at {resolved}: bytes={st.st_size} sha256={digest}")
    return {"path": str(resolved), "bytes": st.st_size, "sha256": digest}


def atomic_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "ab") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def cgroup_available_bytes() -> tuple[int | None, dict[str, Any]]:
    candidates = [Path("/sys/fs/cgroup")]
    for root in candidates:
        max_path = root / "memory.max"
        current_path = root / "memory.current"
        if max_path.is_file() and current_path.is_file():
            max_raw = max_path.read_text(encoding="ascii").strip()
            current = int(current_path.read_text(encoding="ascii").strip())
            if max_raw == "max":
                return None, {"root": str(root), "memory_max": None, "memory_current": current}
            maximum = int(max_raw)
            return maximum - current, {"root": str(root), "memory_max": maximum, "memory_current": current}
    return None, {"root": None, "memory_max": None, "memory_current": None}


def nvidia_snapshot(expected_index: int, min_free_mib: int, *, allowed_pids: set[int]) -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=12,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=12,
    )
    if gpu.returncode != 0 or apps.returncode != 0:
        raise RunnerError(f"nvidia-smi failed: gpu={gpu.stderr[-1000:]} apps={apps.stderr[-1000:]}")
    rows: list[dict[str, Any]] = []
    for line in gpu.stdout.splitlines():
        fields = [field.strip() for field in next(csv.reader([line]))]
        if len(fields) != 5:
            raise RunnerError(f"malformed nvidia-smi GPU row: {line!r}")
        rows.append({
            "index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_total_mib": int(fields[3]),
            "memory_total_bytes": int(fields[3]) * 1024 * 1024,
            "memory_free_mib": int(fields[4]),
            "memory_free_bytes": int(fields[4]) * 1024 * 1024,
        })
    chosen = next((row for row in rows if row["index"] == expected_index), None)
    if chosen is None:
        raise RunnerError(f"nvidia-smi did not report expected GPU index {expected_index}")
    app_rows = []
    for line in apps.stdout.splitlines():
        line = line.strip()
        if not line or "No running processes found" in line:
            continue
        fields = [field.strip() for field in next(csv.reader([line]))]
        if len(fields) < 2:
            raise RunnerError(f"malformed nvidia-smi compute-app row: {line!r}")
        try:
            pid = int(fields[0])
        except ValueError as exc:
            raise RunnerError(f"nvidia-smi compute-app PID is not an integer: {line!r}") from exc
        app_rows.append({"pid": pid, "process_name": fields[1], "raw": line})
    other = [row for row in app_rows if row["pid"] not in allowed_pids]
    return {
        "observed_unix_seconds": time.time(),
        "gpu_rows": rows,
        "selected_gpu": chosen,
        "compute_apps": app_rows,
        "other_compute_apps": other,
        "minimum_free_mib": min_free_mib,
        "status": "PASS" if chosen["memory_free_mib"] >= min_free_mib and not other else "BLOCKED",
    }


def stable_gpu_admission(config: dict[str, Any], control: Path, *, own_pid: int | None = None) -> list[dict[str, Any]]:
    policy = config["resource_policy"]
    allowed = set() if own_pid is None else {int(own_pid)}
    samples = int(policy["gpu_admission_samples"])
    spacing = float(policy["gpu_sample_spacing_seconds"])
    expected_index = int(policy["gpu_index"])
    floor = int(policy["minimum_gpu_free_mib"])
    found: list[dict[str, Any]] = []
    for index in range(samples):
        snap = nvidia_snapshot(expected_index, floor, allowed_pids=allowed)
        snap["sample_index"] = index + 1
        snap["allowed_process_pids"] = sorted(allowed)
        atomic_line(control / "gpu_admission_events.jsonl", snap)
        if snap["status"] != "PASS":
            raise RunnerError(f"GPU admission blocked at sample {index + 1}: {snap}")
        if found and snap["selected_gpu"]["uuid"] != found[0]["selected_gpu"]["uuid"]:
            raise RunnerError("GPU UUID changed between stable admission samples")
        found.append(snap)
        if index + 1 < samples:
            time.sleep(spacing)
    return found


def validate_terminal(config: dict[str, Any]) -> dict[str, Any]:
    terminal = config["source_bindings"]["shared540_terminal_discovery"]
    binding_path = Path(terminal["terminal_binding_path"])
    binding_raw = binding_path.read_bytes()
    if sha_bytes(binding_raw) != terminal["terminal_binding_raw_sha256"]:
        raise RunnerError("shared540 terminal-binding file hash changed")
    checks = [
        ("source_status_path", "source_status_raw_sha256"),
        ("results_path", "results_raw_sha256"),
        ("result_manifest_path", "result_manifest_raw_sha256"),
    ]
    pins = []
    for path_key, sha_key in checks:
        path = Path(terminal[path_key])
        actual = sha_file(path)
        if actual != terminal[sha_key]:
            raise RunnerError(f"shared540 terminal source changed: {path}")
        pins.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
    if terminal.get("status") != "TERMINAL_SEALED" or terminal.get("producer_status") != "COMPLETE":
        raise RunnerError("configured shared540 discovery is not terminal complete")
    if terminal.get("planned_rows") != 540 or terminal.get("result_rows") != 540:
        raise RunnerError("configured shared540 terminal row counts changed")
    if terminal.get("result_status_counts") != {"FAILED": 0, "INCOMPLETE": 0, "PASS": 540}:
        raise RunnerError("configured shared540 terminal status counts changed")
    return {
        "schema": "cross_tfg_fit40_shared540_terminal_runtime_recheck_v1",
        "status": "PASS_TERMINAL_SOURCE_REHASH",
        "binding_path": str(binding_path),
        "binding_bytes": len(binding_raw),
        "binding_sha256": sha_bytes(binding_raw),
        "rechecked_files": pins,
        "run_id": terminal["run_id"],
        "planned_rows": 540,
        "result_rows": 540,
        "status_counts": terminal["result_status_counts"],
    }


def load_expected_rows(config: dict[str, Any]) -> list[str]:
    path = Path(config["remote_paths"]["staged_fit40_manifest_path"])
    raw, manifest = read_json(path)
    source_binding = config["source_bindings"]["fit40_manifest"]
    if len(raw) != int(source_binding["bytes"]) or sha_bytes(raw) != source_binding["sha256"]:
        raise RunnerError("staged fit40 manifest differs from the sealed local source manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise RunnerError("staged fit40 manifest does not contain exactly 40 rows")
    row_ids = [str(row["row_id"]) for row in rows]
    if len(set(row_ids)) != 40:
        raise RunnerError("staged fit40 manifest contains duplicate row IDs")
    expected = {f"S{speaker:04d}_j{slot:02d}_" for speaker in range(52, 72) for slot in (0, 15)}
    if {row.split("_", 2)[0] + "_" + row.split("_", 2)[1] + "_" for row in row_ids} != expected:
        raise RunnerError("staged fit40 row IDs differ from the frozen S0052–S0071 × j=0/15 roster")
    if manifest.get("scope", {}).get("dev_audio_read") is not False or manifest.get("scope", {}).get("confirm_audio_read") is not False:
        raise RunnerError("fit40 source manifest scope does not explicitly exclude dev and confirmation")
    return row_ids


def check_local_code_bundle(config: dict[str, Any], code_root: Path) -> list[dict[str, Any]]:
    manifest_path = Path(config["remote_paths"]["payload_manifest_path"])
    raw, manifest = read_json(manifest_path)
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise RunnerError("payload manifest has no sealed members")
    checked = []
    names = set()
    for item in rows:
        name = str(item["archive_path"])
        if name in names or not name.startswith("payload/") or ".." in Path(name).parts:
            raise RunnerError(f"duplicate or unsafe payload member path: {name}")
        names.add(name)
        path = code_root / Path(name).relative_to("payload")
        pin = file_pin(path, {"bytes": item["bytes"], "sha256": item["sha256"]})
        checked.append({"archive_path": name, **pin})
    return checked


def tree_inventory(root: Path, limit: int) -> dict[str, Any]:
    total = 0
    count = 0
    if not root.exists():
        return {"regular_file_count": 0, "logical_bytes": 0}
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in dirs:
            path = current_path / name
            if path.is_symlink():
                raise RunnerError(f"output contains a symlinked directory: {path}")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise RunnerError(f"output contains a symlink or nonregular file: {path}")
            total += path.stat().st_size
            count += 1
            if total > limit:
                raise RunnerError(f"producer output exceeds configured maximum bytes: {total}>{limit}")
    return {"regular_file_count": count, "logical_bytes": total}


def process_start_ticks(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    close = raw.rfind(")")
    fields = raw[close + 2 :].split() if close >= 0 else []
    return fields[19] if len(fields) > 19 else None


def row_ledger(expected: list[str], passed: set[str], failed: str | None = None) -> list[dict[str, str]]:
    result = []
    for row_id in expected:
        if row_id in passed:
            status = "PASS_ROW_GEOMETRY"
        elif row_id == failed:
            status = "FAILED_ROW_GEOMETRY"
        else:
            status = "NOT_RUN"
        result.append({"row_id": row_id, "status": status})
    return result


def get_progress(out: Path, expected: list[str]) -> tuple[set[str], str | None]:
    passed: set[str] = set()
    active = None
    p = out / "progress.json"
    if p.is_file():
        try:
            _, progress = read_json(p)
            ids = [str(value) for value in progress.get("row_ids", [])]
            if len(set(ids)) != len(ids) or not set(ids).issubset(set(expected)):
                raise RunnerError("producer progress has duplicate or unrecognized row IDs")
            passed = set(ids)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Atomic writer may be replacing the file; the next poll rechecks.
            pass
    failure_path = out / "failure.json"
    if failure_path.is_file():
        try:
            _, failure = read_json(failure_path)
            candidate = failure.get("active_row_id")
            if candidate is not None and str(candidate) in expected:
                active = str(candidate)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return passed, active


def write_state(control: Path, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    return atomic_json(control / "execution_state.json", {
        "schema": "cross_tfg_fit40_native_rms_geometry_execution_state_v1",
        "status": status,
        "updated_unix_seconds": time.time(),
        **payload,
    })


def perform(config_path: Path, expected_config_sha256: str, launch: bool) -> int:
    config_raw = config_path.read_bytes()
    if sha_bytes(config_raw) != expected_config_sha256.lower():
        raise RunnerError("fit40 config SHA does not match the launcher pin")
    config = json.loads(config_raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "cross_tfg_fit40_native_rms_geometry_execution_config_v2":
        raise RunnerError("unsupported fit40 execution config schema")
    control = Path(config["remote_paths"]["control_root"]).resolve(strict=True)
    code_root = Path(config["remote_paths"]["code_root"]).resolve(strict=True)
    out = Path(config["remote_paths"]["output_root"])
    if out.exists() or out.is_symlink():
        raise RunnerError(f"refusing to reuse fit40 output path: {out}")
    payload_members = check_local_code_bundle(config, code_root)
    rows = load_expected_rows(config)
    terminal_receipt = validate_terminal(config)
    policies = config["resource_policy"]
    cgroup_free, cgroup_record = cgroup_available_bytes()
    if cgroup_free is None or cgroup_free < int(policies["minimum_cgroup_memory_available_bytes"]):
        raise RunnerError(f"cgroup memory headroom is below the configured floor: {cgroup_record}")
    shm_free = shutil.disk_usage("/dev/shm").free
    required_shm = int(policies["max_expected_output_bytes"]) + int(policies["minimum_shm_free_after_run_reserve_bytes"])
    if shm_free < required_shm:
        raise RunnerError(f"/dev/shm free space cannot cover output bound plus reserve: {shm_free}<{required_shm}")
    gpu_samples = stable_gpu_admission(config, control)
    uuid = str(gpu_samples[0]["selected_gpu"]["uuid"])
    common = {
        "config_path": str(config_path),
        "config_bytes": len(config_raw),
        "config_sha256": sha_bytes(config_raw),
        "row_ids_expected": rows,
        "row_ledger": row_ledger(rows, set()),
        "source_bundle_members": payload_members,
        "shared540_terminal_recheck": terminal_receipt,
        "gpu_uuid": uuid,
        "gpu_admission_samples": gpu_samples,
        "cgroup_memory": cgroup_record,
        "shm_free_before_bytes": shm_free,
        "shm_required_bytes": required_shm,
    }
    preflight_pin = write_state(control, "PASS_PREFLIGHT_SOURCE_GPU_STORAGE", common)
    if not launch:
        print(json.dumps({"status": "PASS_PREFLIGHT_SOURCE_GPU_STORAGE", "execution_state": preflight_pin, **common}, sort_keys=True))
        return 0
    if out.exists() or out.is_symlink():
        raise RunnerError(f"fit40 output appeared after preflight; refusing launch: {out}")

    lock_path = control / "launcher.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(lock_fd, "w", encoding="ascii") as lock:
        lock.write(f"pid={os.getpid()}\n")
        lock.write(f"start_ticks={process_start_ticks(os.getpid()) or ''}\n")
        lock.flush()
        os.fsync(lock.fileno())

    env = os.environ.copy()
    env.update({
        "CROSS_TFG_PSF_ROOT": str(config["runtime"]["python_speech_features_root"]),
        "CROSS_TFG_FIT40_EXPECTED_GPU_UUID": uuid,
        "CROSS_TFG_FIT40_GPU_INDEX": str(policies["gpu_index"]),
        "CROSS_TFG_FIT40_MIN_GPU_FREE_MIB": str(policies["minimum_gpu_free_mib"]),
        "CROSS_TFG_FIT40_GPU_ADMISSION_LOG": str(control / "producer_gpu_admission_events.jsonl"),
        "CUBLAS_WORKSPACE_CONFIG": str(config["runtime"]["cublas_workspace_config"]),
        "OMP_NUM_THREADS": str(config["runtime"]["omp_num_threads"]),
        "MKL_NUM_THREADS": str(config["runtime"]["mkl_num_threads"]),
        "OPENBLAS_NUM_THREADS": str(config["runtime"]["openblas_num_threads"]),
        "PYTHONHASHSEED": str(config["runtime"]["pythonhashseed"]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    # The literal CUDA device is exposed again only in the child process.
    env["CUDA_VISIBLE_DEVICES"] = str(config["runtime"].get("cuda_visible_devices", ""))
    # Do not allow an inherited setting to replace the reviewed per-run seed/config.
    env.pop("PYTHONPATH", None)
    payload_config = code_root / config["remote_paths"]["config_relative_path"]
    producer = code_root / config["remote_paths"]["producer_relative_path"]
    plan = code_root / config["remote_paths"]["plan_relative_path"]
    parity = code_root / config["remote_paths"]["parity_helper_relative_path"]
    clock = code_root / config["remote_paths"]["clock_helper_relative_path"]
    command = [
        str(config["runtime"]["python_executable_literal"]), "-B", str(producer),
        "--repo-root", str(code_root),
        "--source-manifest", str(config["remote_paths"]["staged_fit40_manifest_path"]),
        "--source-root", str(config["remote_paths"]["staged_fit40_input_root"]),
        "--v2-plan", str(plan),
        "--parity-helper", str(parity),
        "--clock-helper", str(clock),
        "--primary-face", str(config["runtime"]["primary_face_path"]),
        "--ffmpeg", str(config["runtime"]["ffmpeg_path"]),
        "--ffmpeg-sha256", str(config["runtime"]["ffmpeg_sha256"]),
        "--out", str(out),
        "--device", str(config["runtime"]["device"]),
        "--seed", str(config["runtime"]["seed"]),
        "--minimum-free-bytes", str(policies["producer_per_row_minimum_free_bytes"]),
    ]
    log_out = control / "producer.stdout.log"
    log_err = control / "producer.stderr.log"
    started = time.time()
    with log_out.open("xb") as stdout_handle, log_err.open("xb") as stderr_handle:
        child = subprocess.Popen(command, cwd=code_root, env=env, stdout=stdout_handle, stderr=stderr_handle, close_fds=True)
        child_ticks = process_start_ticks(child.pid)
        running = {
            **common,
            "row_ledger": row_ledger(rows, set()),
            "child_pid": child.pid,
            "child_start_ticks": child_ticks,
            "child_command": command,
            "child_cwd": str(code_root),
            "child_python_executable": str(config["runtime"]["python_executable_literal"]),
            "started_unix_seconds": started,
            "producer_output_root": str(out),
        }
        write_state(control, "RUNNING_FIT40_NATIVE_RMS_GEOMETRY", running)
        deadline = started + float(policies["max_wall_seconds"])
        child_killed_reason: str | None = None
        while child.poll() is None:
            passed, failed_row = get_progress(out, rows)
            free_now = shutil.disk_usage("/dev/shm").free
            try:
                output_size = tree_inventory(out, int(policies["max_expected_output_bytes"]))["logical_bytes"]
            except RunnerError as exc:
                child_killed_reason = f"output_bound_or_tree_guard:{exc}"
                child.terminate()
                break
            write_state(control, "RUNNING_FIT40_NATIVE_RMS_GEOMETRY", {
                **running,
                "row_ledger": row_ledger(rows, passed, failed_row),
                "completed_rows": len(passed),
                "active_failed_row_id": failed_row,
                "elapsed_seconds": time.time() - started,
                "shm_free_bytes": free_now,
                "output_bytes": output_size,
            })
            if free_now < int(policies["minimum_shm_free_after_run_reserve_bytes"]):
                child_killed_reason = f"shm_reserve_breached:{free_now}"
                child.terminate()
                break
            if time.time() >= deadline:
                child_killed_reason = "wall_timeout"
                child.terminate()
                break
            time.sleep(2.0)
        try:
            return_code = child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            return_code = child.wait()
            child_killed_reason = child_killed_reason or "child_did_not_stop_after_terminate"
    passed, failed_row = get_progress(out, rows)
    final_state_path = out / "worker_state.json"
    final_state = None
    if final_state_path.is_file():
        try:
            _, final_state = read_json(final_state_path)
        except Exception:
            final_state = None
    output_inventory = tree_inventory(out, int(policies["max_expected_output_bytes"]))
    geometry_path = out / "fit40_native_rms_geometry_manifest.json"
    cache_path = out / "natural_audio_embedding_cache" / "cache_manifest.json"
    status = "FAILED_CHILD_OR_RESOURCE_GUARD"
    geometry_pin = None
    cache_pin = None
    geometry: dict[str, Any] | None = None
    if return_code == 0 and child_killed_reason is None and isinstance(final_state, dict):
        if final_state.get("status") == "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY" and geometry_path.is_file() and cache_path.is_file():
            _, geometry = read_json(geometry_path)
            _, cache_manifest = read_json(cache_path)
            saved_ids = [str(row.get("row_id")) for row in geometry.get("rows", [])]
            identity = geometry.get("runtime_gpu_identity")
            selected_gpu = gpu_samples[0]["selected_gpu"]
            producer_member_name = "payload/" + str(config["remote_paths"]["producer_relative_path"])
            producer_member = next((item for item in payload_members if item.get("archive_path") == producer_member_name), None)
            producer_binding = geometry.get("source_pins", {}).get("geometry_producer")
            profile = identity.get("runtime_profile") if isinstance(identity, dict) else None
            expected_policy = config["runtime"]["batch_policy"]
            identity_matches = bool(
                isinstance(identity, dict)
                and isinstance(profile, dict)
                and producer_member is not None
                and isinstance(producer_binding, dict)
                and int(identity.get("device_index", -1)) == int(policies["gpu_index"])
                and str(identity.get("uuid")) == str(selected_gpu["uuid"]) == uuid
                and str(identity.get("device_name")) == str(selected_gpu["name"])
                and int(identity.get("memory_total_bytes", -1)) == int(identity.get("torch_total_memory_bytes", -2))
                and str(identity.get("memory_total_source")) == "torch.cuda.get_device_properties(device_index).total_memory"
                and int(identity.get("memory_total_mib", -1)) == int(identity.get("torch_total_memory_bytes", -2)) // (1024 * 1024)
                and int(identity.get("nvidia_smi_memory_total_mib", -1)) == int(selected_gpu["memory_total_mib"])
                and int(identity.get("nvidia_smi_memory_total_bytes_reported", -1)) == int(selected_gpu["memory_total_bytes"])
                and int(identity.get("memory_total_reporting_difference_bytes_torch_minus_nvidia_smi", -10**30))
                    == int(identity.get("torch_total_memory_bytes", 0)) - int(identity.get("nvidia_smi_memory_total_bytes_reported", 0))
                and str(identity.get("device_literal")) == str(config["runtime"]["device"])
                and int(profile.get("seed", -1)) == int(config["runtime"]["seed"])
                and str(profile.get("python_executable_literal")) == str(config["runtime"]["python_executable_literal"])
                and str(profile.get("python_version")) == str(config["runtime"]["python_version"])
                and str(profile.get("torch_version")) == str(config["runtime"]["torch_version"])
                and str(profile.get("numpy_version")) == str(config["runtime"]["numpy_version"])
                and int(profile.get("torch_num_threads", -1)) == int(config["runtime"]["torch_num_threads"])
                and int(profile.get("torch_num_interop_threads", -1)) == int(config["runtime"]["torch_num_interop_threads"])
                and str(profile.get("cublas_workspace_config")) == str(config["runtime"]["cublas_workspace_config"])
                and str(profile.get("cuda_visible_devices")) == str(config["runtime"]["cuda_visible_devices"])
                and profile.get("cuda_matmul_allow_tf32") is config["runtime"]["cuda_matmul_allow_tf32"]
                and profile.get("cudnn_allow_tf32") is config["runtime"]["cudnn_allow_tf32"]
                and profile.get("cudnn_benchmark") is config["runtime"]["cudnn_benchmark"]
                and profile.get("cudnn_deterministic") is config["runtime"]["cudnn_deterministic"]
                and profile.get("batch_policy") == expected_policy
                and profile.get("frame_rate") == {"numerator": 25, "denominator": 1}
                and str(profile.get("wav2lip_git_head")) == str(config["runtime"]["wav2lip_git_head"])
                and str(profile.get("wav2lip_checkpoint_sha256")) == str(config["runtime"]["wav2lip_checkpoint_sha256"])
                and str(profile.get("syncnet_git_head")) == str(config["runtime"]["syncnet_git_head"])
                and str(profile.get("syncnet_checkpoint_sha256")) == str(config["runtime"]["syncnet_checkpoint_sha256"])
                and str(profile.get("fixed_portrait_sha256")) == str(config["runtime"]["primary_face_sha256"])
                and isinstance(geometry.get("runtime_gpu_after_model_load_admission"), dict)
                and geometry["runtime_gpu_after_model_load_admission"].get("status") == "PASS"
                and geometry["runtime_gpu_after_model_load_admission"].get("selected_gpu", {}).get("uuid") == uuid
                and int(geometry["runtime_gpu_after_model_load_admission"].get("selected_gpu", {}).get("index", -1)) == int(policies["gpu_index"])
                and str(producer_binding.get("sha256", "")) == producer_member.get("sha256")
                and int(producer_binding.get("bytes", -1)) == int(producer_member.get("bytes", -2))
            )
            row_identity_matches = bool(
                isinstance(identity, dict)
                and len(geometry.get("rows", [])) == 40
                and all(row.get("runtime_gpu_identity") == identity for row in geometry["rows"])
                and all(
                    isinstance(row.get("runtime_gpu_row_admission"), dict)
                    and row["runtime_gpu_row_admission"].get("status") == "PASS"
                    and row["runtime_gpu_row_admission"].get("selected_gpu", {}).get("uuid") == uuid
                    and int(row["runtime_gpu_row_admission"].get("selected_gpu", {}).get("index", -1)) == int(policies["gpu_index"])
                    and not row["runtime_gpu_row_admission"].get("other_compute_apps")
                    for row in geometry["rows"]
                )
            )
            if (len(saved_ids) == 40 and set(saved_ids) == set(rows) and len(set(saved_ids)) == 40
                    and len(cache_manifest.get("rows", [])) == 40 and identity_matches and row_identity_matches):
                geometry_pin = {"path": str(geometry_path), "bytes": geometry_path.stat().st_size, "sha256": sha_file(geometry_path)}
                cache_pin = {"path": str(cache_path), "bytes": cache_path.stat().st_size, "sha256": sha_file(cache_path)}
                passed = set(saved_ids)
                failed_row = None
                status = "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY"
            else:
                status = "FAILED_RUNTIME_IDENTITY_OR_ROW_COHORT_BINDING"
    if return_code != 0 and status == "FAILED_CHILD_OR_RESOURCE_GUARD" and child_killed_reason is None:
        status = "FAILED_CHILD_PREPARATION_NOT_SCIENTIFIC_EFFECT"
    receipt = {
        "schema": "cross_tfg_fit40_native_rms_geometry_execution_receipt_v1",
        "status": status,
        "child_return_code": return_code,
        "child_killed_reason": child_killed_reason,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "elapsed_seconds": time.time() - started,
        "child_pid": child.pid,
        "child_start_ticks": child_ticks,
        "gpu_uuid": uuid,
        "gpu_index": int(policies["gpu_index"]),
        "gpu_name": gpu_samples[0]["selected_gpu"]["name"],
        "gpu_memory_total_bytes": gpu_samples[0]["selected_gpu"]["memory_total_bytes"],
        "gpu_memory_total_mib": gpu_samples[0]["selected_gpu"]["memory_total_mib"],
        "gpu_admission_samples": gpu_samples,
        "producer_gpu_admission_log": {"path": str(control / "producer_gpu_admission_events.jsonl"), "exists": (control / "producer_gpu_admission_events.jsonl").is_file()},
        "shared540_terminal_recheck": terminal_receipt,
        "row_ledger": row_ledger(rows, passed, failed_row),
        "completed_rows": len(passed),
        "expected_rows": 40,
        "final_producer_state": final_state,
        "runtime_gpu_identity": (geometry.get("runtime_gpu_identity") if isinstance(geometry, dict) else None),
        "geometry_manifest": geometry_pin,
        "natural_audio_cache_manifest": cache_pin,
        "output_inventory": output_inventory,
        "producer_stdout": {"path": str(log_out), "bytes": log_out.stat().st_size, "sha256": sha_file(log_out)},
        "producer_stderr": {"path": str(log_err), "bytes": log_err.stat().st_size, "sha256": sha_file(log_err)},
        "preflight_state": preflight_pin,
        "model_score_computed": False,
        "training_performed": False,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
    }
    receipt_pin = atomic_json(control / "producer_execution_receipt_attempt04.json", receipt)
    write_state(control, status, {"execution_receipt": receipt_pin, **receipt})
    print(json.dumps({"status": status, "execution_receipt": receipt_pin, "completed_rows": len(passed), "expected_rows": 40}, sort_keys=True))
    return 0 if status == "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    try:
        return perform(args.config, args.expected_config_sha256, args.launch)
    except Exception as exc:
        # Preflight errors carry the entire frozen roster as NOT_RUN. If a
        # launcher lock already exists, preserve the last durable row ledger
        # instead of relabeling a post-launch supervisor error as preflight.
        control = Path(os.environ.get("CROSS_TFG_FIT40_CONTROL_ROOT", ".")).resolve()
        expected_row_ids: list[str] = []
        try:
            config_value = json.loads(args.config.read_text(encoding="utf-8"))
            configured = config_value.get("scope", {}).get("expected_row_ids", [])
            if isinstance(configured, list) and all(isinstance(value, str) for value in configured):
                expected_row_ids = configured
        except Exception:
            expected_row_ids = []
        launched = (control / "launcher.lock").is_file()
        failure_status = "FAILED_SUPERVISOR_EXCEPTION_AFTER_LAUNCH" if launched else "FAILED_PREFLIGHT_NOT_SCIENTIFIC_EFFECT"
        ledger = [{"row_id": row_id, "status": "NOT_RUN"} for row_id in expected_row_ids]
        if launched:
            try:
                state = json.loads((control / "execution_state.json").read_text(encoding="utf-8"))
                saved_ledger = state.get("row_ledger")
                if isinstance(saved_ledger, list):
                    ledger = saved_ledger
            except Exception:
                pass
        failure = {
            "schema": "cross_tfg_fit40_native_rms_geometry_preflight_failure_v1",
            "status": failure_status,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc()[-8000:],
            "observed_unix_seconds": time.time(),
            "expected_rows": len(expected_row_ids),
            "completed_rows": sum(1 for row in ledger if row.get("status") == "PASS_ROW_GEOMETRY"),
            "row_ledger": ledger,
            "model_score_computed": False,
            "training_performed": False,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
        }
        try:
            atomic_json(control / "preflight_failure_attempt04.json", failure)
        except Exception:
            pass
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
