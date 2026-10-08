#!/usr/bin/env python3
"""Single-use, source-pinned detached launcher for the Fit40 D0/GateDC run.

This file is only prepared locally. It is not an experiment worker and does
not import PyTorch or any model package. On dispatch it checks frozen inputs,
source hashes, the 40 source WAVs, current GPU identity/idle state and SHM
reserve, then starts the pinned runner once in a detached supervisor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
from typing import Any


PROJECT_ROOT = pathlib.Path("/root/tts-research-system-20261001")
CONTROL_ROOT = PROJECT_ROOT / "cross_tfg_audio_head_20261003/job_control/fit40_d0_gatedc_training_attempt02"
TASK_ROOT = pathlib.Path("/dev/shm/cross_tfg_audio_head_20261003/fit40_d0_gatedc_training_attempt02")
OUTPUT_ROOT = TASK_ROOT / "run"
MANIFEST_PATH = CONTROL_ROOT / "inputs/launch_manifest_attempt02.json"
LAUNCH_RECEIPT = CONTROL_ROOT / "output/launch_receipt.json"
LAUNCH_LOCK = CONTROL_ROOT / "output/launch.lock"
FAILURE_RECEIPT = CONTROL_ROOT / "output/launch_failure.json"
SUPERVISOR_LOG = CONTROL_ROOT / "logs/supervisor.log"
RUN_LOG = CONTROL_ROOT / "logs/fit_runner_attempt02.stdout_stderr.log"
CHILD_RECEIPT = CONTROL_ROOT / "output/child_process.json"
EXIT_RECEIPT = CONTROL_ROOT / "output/process_exit.json"
PYTHON = pathlib.Path("/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
CODE_ROOT = TASK_ROOT / "code"
STAGE_RECEIPT = CONTROL_ROOT / "output/source_stage_receipt_attempt02.json"
MIN_SHM_FREE_BYTES = 5 * 1024**3
MIN_GPU_FREE_MIB = 18_432


class LaunchError(RuntimeError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def self_hash(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256(canonical(payload))


def write_json_exclusive(path: pathlib.Path, value: dict[str, Any], digest_field: str | None = None) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise LaunchError(f"refusing to replace existing receipt: {path}")
    payload = dict(value)
    if digest_field is not None:
        payload[digest_field] = self_hash(payload, digest_field)
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "bytes": len(raw), "sha256": sha256(raw), "self_sha256": payload.get(digest_field) if digest_field else None}


def read_regular(path: pathlib.Path, *, expected_bytes: int | None = None, expected_sha: str | None = None, maximum: int = 2 * 1024**3) -> bytes:
    if path.is_symlink():
        raise LaunchError(f"symlink refused: {path}")
    try:
        before = path.lstat()
    except OSError as exc:
        raise LaunchError(f"cannot stat {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise LaunchError(f"not a bounded regular file: {path}")
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise LaunchError(f"byte count mismatch: {path}")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if before.st_size <= 8 * 1024**2 else None
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise LaunchError(f"file changed during read: {path}")
    if expected_sha is not None and digest.hexdigest() != expected_sha.lower():
        raise LaunchError(f"SHA-256 mismatch: {path}")
    if chunks is None:
        return b""
    return b"".join(chunks)


def verify_pin(pin: dict[str, Any]) -> dict[str, Any]:
    if set(pin) != {"role", "path", "bytes", "sha256"}:
        raise LaunchError("manifest file pin has an unexpected schema")
    path = pathlib.Path(pin["path"])
    read_regular(path, expected_bytes=int(pin["bytes"]), expected_sha=str(pin["sha256"]))
    return {"role": pin["role"], "path": str(path), "bytes": int(pin["bytes"]), "sha256": str(pin["sha256"])}


def load_pinned_json(pin: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(pathlib.Path(pin["path"]), expected_bytes=int(pin["bytes"]), expected_sha=str(pin["sha256"]), maximum=8 * 1024**2)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise LaunchError(f"expected JSON object: {pin['path']}")
    return value, raw


def check_payload_hash(obj: dict[str, Any], key: str, expected: str) -> None:
    payload = {name: value for name, value in obj.items() if name != key}
    if obj.get(key) != expected or sha256(canonical(payload)) != expected:
        raise LaunchError(f"canonical {key} mismatch")


def verify_manifest_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    verified = []
    for pin in manifest.get("file_pins", []):
        verified.append(verify_pin(pin))
    roles = {row["role"] for row in verified}
    if len(roles) != len(verified):
        raise LaunchError("duplicate file-pin role")
    return verified


def verify_source_stage(manifest: dict[str, Any]) -> dict[str, Any]:
    bundle = manifest.get("source_bundle")
    if not isinstance(bundle, dict):
        raise LaunchError("launch manifest lacks its source-bundle binding")
    receipt_path = pathlib.Path(str(bundle.get("stage_receipt_path", "")))
    if receipt_path != STAGE_RECEIPT or receipt_path.is_symlink():
        raise LaunchError("source-stage receipt path differs from the attempt02 control namespace")
    raw = read_regular(receipt_path, maximum=2 * 1024**2)
    receipt = json.loads(raw)
    if not isinstance(receipt, dict) or receipt.get("schema") != "fit40_d0_gatedc_source_stage_receipt_attempt02_v1" or receipt.get("status") != "PASS_SOURCE_CLOSURE_EXTRACTED_NO_MODEL_OR_GPU":
        raise LaunchError("source-stage receipt status/schema differs")
    if receipt.get("self_sha256") != self_hash(receipt, "self_sha256"):
        raise LaunchError("source-stage receipt self hash mismatch")
    if receipt.get("destination") != str(CODE_ROOT.resolve(strict=True)):
        raise LaunchError("source-stage receipt extraction root differs")
    if receipt.get("archive") != bundle.get("archive") or receipt.get("manifest") != bundle.get("manifest"):
        raise LaunchError("source-stage receipt does not bind exact archive/manifest pins")
    expected = bundle.get("members")
    actual = receipt.get("members")
    if not isinstance(expected, list) or not isinstance(actual, list) or len(expected) != len(actual):
        raise LaunchError("source-stage member denominator differs")
    expected_by_path = {str(x.get("relative_path")): x for x in expected if isinstance(x, dict)}
    actual_by_path = {str(x.get("relative_path")): x for x in actual if isinstance(x, dict)}
    if len(expected_by_path) != len(expected) or len(actual_by_path) != len(actual) or set(expected_by_path) != set(actual_by_path):
        raise LaunchError("source-stage member paths differ from the sealed bundle")
    for rel, pin in expected_by_path.items():
        row = actual_by_path[rel]
        target = CODE_ROOT / rel
        if (int(row.get("bytes", -1)), str(row.get("sha256", "")).lower(), str(row.get("path", ""))) != (int(pin["bytes"]), str(pin["sha256"]).lower(), str(target)):
            raise LaunchError(f"source-stage receipt member identity differs: {rel}")
        read_regular(target, expected_bytes=int(pin["bytes"]), expected_sha=str(pin["sha256"]))
    return {"path": str(receipt_path), "bytes": len(raw), "sha256": sha256(raw), "self_sha256": receipt["self_sha256"], "members": len(actual)}


def verify_data_inputs(manifest: dict[str, Any]) -> dict[str, Any]:
    inputs = manifest["inputs"]
    parent, parent_raw = load_pinned_json(inputs["parent_plan"])
    check_payload_hash(parent, "plan_payload_sha256", inputs["parent_plan"]["payload_sha256"])
    source, source_raw = load_pinned_json(inputs["source_manifest"])
    if source.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1" or source.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise LaunchError("fit-source manifest status/schema differs")
    scope = source.get("scope", {})
    if any(scope.get(k) is not False for k in ("candidate_transform_executed", "confirm_audio_read", "dev_audio_read", "model_forward", "render", "score")):
        raise LaunchError("source-manifest scope is not source-only/Fit-only")
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise LaunchError("source manifest must contain exactly 40 Fit rows")
    geometry, geometry_raw = load_pinned_json(inputs["geometry_manifest"])
    if geometry.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1" or geometry.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY":
        raise LaunchError("native geometry status/schema differs")
    check_payload_hash(geometry, "payload_sha256", inputs["geometry_manifest"]["payload_sha256"])
    if int(geometry.get("fit_rows", -1)) != 40 or geometry.get("method_id") != "rms_fixed":
        raise LaunchError("geometry is not exactly the 40-row RMS baseline")
    if (int(geometry.get("source_clock_fps_num", -1)), int(geometry.get("source_clock_fps_den", -1))) != (25, 1):
        raise LaunchError("geometry source clock is not frozen 25/1")
    if geometry.get("audio_clock_policy") != "unchanged_original_natural_pcm_mfcc_queries":
        raise LaunchError("geometry audio clock changed")
    if geometry.get("source_pins", {}).get("source_manifest", {}).get("sha256") != inputs["source_manifest"]["sha256"]:
        raise LaunchError("geometry does not bind the exact Fit source manifest")
    if geometry.get("source_pins", {}).get("v2_runtime_plan", {}).get("sha256") != inputs["parent_plan"]["sha256"]:
        raise LaunchError("geometry does not bind the exact parent V2 plan")
    geom_rows = geometry.get("rows")
    row_ids = [str(r.get("row_id")) for r in rows]
    geom_ids = [str(r.get("row_id")) for r in geom_rows] if isinstance(geom_rows, list) else []
    if len(geom_ids) != 40 or row_ids != geom_ids or len(set(row_ids)) != 40:
        raise LaunchError("source and geometry rows do not match the frozen ordered 40-row roster")
    expected_pairs = [(f"S{speaker:04d}", j) for speaker in range(52, 72) for j in (0, 15)]
    if [(str(r.get("speaker_id")), int(r.get("canonical_j", -1))) for r in rows] != expected_pairs:
        raise LaunchError("source rows do not match the frozen S0052-S0071, j00/j15 cohort")
    source_root = pathlib.Path(manifest["source_root"])
    if source_root.is_symlink() or not source_root.is_dir():
        raise LaunchError("Fit source root is absent or a symlink")
    root_resolved = source_root.resolve(strict=True)
    verified_wavs = []
    for row in rows:
        rel = pathlib.PurePosixPath(str(row.get("transfer_relative_path", "")))
        if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
            raise LaunchError("unsafe Fit WAV relative path")
        wav = source_root.joinpath(*rel.parts)
        try:
            resolved = wav.resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise LaunchError(f"Fit WAV escapes/does not exist under source root: {rel}") from exc
        pin = read_regular(wav, expected_bytes=int(row["transfer_file_bytes"]), expected_sha=str(row["transfer_file_sha256"]))
        if row.get("channels") != 1 or row.get("sample_rate_hz") != 16000 or row.get("source_encoding") != "PCM_16":
            raise LaunchError(f"Fit source WAV metadata is not mono 16-kHz PCM16: {row.get('row_id')}")
        verified_wavs.append({"row_id": row["row_id"], "bytes": int(row["transfer_file_bytes"]), "sha256": str(row["transfer_file_sha256"])})
    cache_pin = geometry.get("audio_cache")
    if not isinstance(cache_pin, dict):
        raise LaunchError("geometry lacks its exact natural audio-cache manifest pin")
    cache_manifest, cache_raw = load_pinned_json({"path": cache_pin["path"], "bytes": cache_pin["bytes"], "sha256": cache_pin["sha256"]})
    if cache_manifest.get("schema") != "cross_tfg_syncnet_full_audio_embedding_cache_manifest_v1" or cache_manifest.get("row_count") != 40:
        raise LaunchError("source audio cache is not the frozen full 40-row V2 cache")
    check_payload_hash(cache_manifest, "manifest_sha256", str(cache_manifest.get("manifest_sha256", "")))
    if {str(r.get("row_id")) for r in cache_manifest.get("rows", [])} != set(row_ids):
        raise LaunchError("natural audio-cache row identities differ from Fit40 geometry")
    head_cfg, head_raw = load_pinned_json(inputs["head_config"])
    if sha256(head_raw) != manifest["inputs"]["head_config"]["sha256"]:
        raise LaunchError("head config raw SHA mismatch")
    return {
        "parent_plan": {"bytes": len(parent_raw), "sha256": sha256(parent_raw), "payload_sha256": parent["plan_payload_sha256"]},
        "source_manifest": {"bytes": len(source_raw), "sha256": sha256(source_raw), "rows": 40, "status": source["status"]},
        "geometry_manifest": {"bytes": len(geometry_raw), "sha256": sha256(geometry_raw), "payload_sha256": geometry["payload_sha256"], "rows": 40, "source_clock_fps": [25, 1]},
        "audio_cache_manifest": {"bytes": len(cache_raw), "sha256": sha256(cache_raw), "rows": 40},
        "head_config": {"bytes": len(head_raw), "sha256": sha256(head_raw)},
        "fit_wavs": verified_wavs,
    }


def gpu_snapshot(expected: dict[str, Any]) -> dict[str, Any]:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        rows = [line.strip().split(",") for line in query.stdout.splitlines() if line.strip()]
        selected = [row for row in rows if len(row) == 5 and row[0].strip() == "0"]
        if len(selected) != 1:
            raise LaunchError("nvidia-smi did not report exactly one physical index-0 GPU row")
        row = selected[0]
        uuid, name = row[1].strip(), row[2].strip()
        free_mib, total_mib = int(row[3].strip()), int(row[4].strip())
        if uuid != expected["uuid"] or name != expected["device_name"] or total_mib != expected["nvidia_smi_memory_total_mib"]:
            raise LaunchError("current physical GPU identity differs from the completed geometry producer")
        if free_mib < MIN_GPU_FREE_MIB:
            raise LaunchError(f"index-0 GPU free memory below sealed 18-GiB reserve: {free_mib} MiB")
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        app_lines = [x.strip() for x in apps.stdout.splitlines() if x.strip() and "no running processes" not in x.lower()]
        if app_lines:
            raise LaunchError(f"refusing to overlap an existing GPU compute process; observed PID rows: {app_lines[:8]}")
        return {"index": 0, "uuid": uuid, "name": name, "free_mib": free_mib, "total_mib": total_mib, "compute_process_rows": 0}
    except subprocess.TimeoutExpired as exc:
        raise LaunchError("nvidia-smi GPU preflight timed out") from exc


def proc_start_ticks(pid: int) -> int | None:
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = raw[raw.rfind(")") + 2:].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def runner_argv(manifest: dict[str, Any]) -> list[str]:
    runner = manifest["runner"]
    i = manifest["inputs"]
    return [
        str(PYTHON), "-B", str(CODE_ROOT / runner["relative_path"]),
        "--parent-plan", str(i["parent_plan"]["path"]),
        "--expected-parent-plan-sha256", str(i["parent_plan"]["sha256"]),
        "--expected-parent-plan-payload-sha256", str(i["parent_plan"]["payload_sha256"]),
        "--source-manifest", str(i["source_manifest"]["path"]),
        "--expected-source-manifest-sha256", str(i["source_manifest"]["sha256"]),
        "--source-root", str(manifest["source_root"]),
        "--geometry-manifest", str(i["geometry_manifest"]["path"]),
        "--expected-geometry-manifest-sha256", str(i["geometry_manifest"]["sha256"]),
        "--head-config", str(i["head_config"]["path"]),
        "--expected-head-config-sha256", str(i["head_config"]["sha256"]),
        "--output-dir", str(OUTPUT_ROOT),
        "--device", "cuda:0",
    ]


def clean_child_environment(manifest: dict[str, Any]) -> dict[str, str]:
    allow = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "TMPDIR") if os.environ.get(key)}
    allow["HOME"] = "/root"
    allow.update({
        "CROSS_TFG_PSF_ROOT": manifest["runtime"]["cross_tfg_psf_root"],
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "20261003",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return allow


def run_supervisor(manifest: dict[str, Any], manifest_raw_sha: str) -> int:
    proc: subprocess.Popen | None = None
    try:
        if sys.executable != str(PYTHON):
            raise LaunchError("supervisor interpreter is not the literal qualified Python")
        verify_manifest_files(manifest)
        source_stage_pin = verify_source_stage(manifest)
        input_pins = verify_data_inputs(manifest)
        gpu = gpu_snapshot(manifest["gpu_identity"])
        shm_free = shutil.disk_usage("/dev/shm").free
        if shm_free < MIN_SHM_FREE_BYTES:
            raise LaunchError(f"/dev/shm free bytes below the sealed 5-GiB reserve: {shm_free}")
        argv = runner_argv(manifest)
        child_env = clean_child_environment(manifest)
        started = time.time()
        log_handle = RUN_LOG.open("ab")
        try:
            proc = subprocess.Popen(argv, cwd=str(CODE_ROOT), env=child_env, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=False, close_fds=True)
        finally:
            log_handle.close()
        child = {
            "schema": "fit40_d0_gatedc_child_process_v1", "status": "STARTED",
            "pid": int(proc.pid), "start_ticks": proc_start_ticks(proc.pid),
            "argv": argv, "cwd": str(CODE_ROOT), "output_root": str(OUTPUT_ROOT),
            "started_unix_seconds": started, "manifest_raw_sha256": manifest_raw_sha,
            "gpu_snapshot": gpu, "shm_free_bytes_before_runner": int(shm_free),
        }
        child_pin = write_json_exclusive(CHILD_RECEIPT, child, "self_sha256")
        code = proc.wait()
        finished = time.time()
        exit_payload = {
            "schema": "fit40_d0_gatedc_process_exit_v1", "status": "EXITED",
            "return_code": int(code), "pid": int(proc.pid), "start_ticks": child["start_ticks"],
            "started_unix_seconds": started, "finished_unix_seconds": finished,
            "manifest_raw_sha256": manifest_raw_sha, "child_process_receipt": child_pin,
            "runner_status_path": str(OUTPUT_ROOT / "status.json"),
        }
        write_json_exclusive(EXIT_RECEIPT, exit_payload, "self_sha256")
        return int(code)
    except BaseException as exc:
        # A child receipt is part of the launch lineage. If writing it (or a
        # later supervisor receipt) fails after Popen, never leave a live
        # training child whose PID and command were not durably recorded.
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            except OSError:
                pass
        failure = {"schema": "fit40_d0_gatedc_supervisor_failure_v1", "status": "SUPERVISOR_FAILED", "type": type(exc).__name__, "message": str(exc)[:1600], "traceback": traceback.format_exc(limit=16), "timestamp_unix_seconds": time.time(), "manifest_raw_sha256": manifest_raw_sha}
        try:
            if not FAILURE_RECEIPT.exists():
                write_json_exclusive(FAILURE_RECEIPT, failure, "self_sha256")
        except Exception:
            pass
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    try:
        raw = read_regular(args.manifest, expected_sha=args.expected_manifest_sha256, maximum=1 * 1024**2)
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or manifest.get("schema") != "fit40_d0_gatedc_training_launch_manifest_v1":
            raise LaunchError("launch manifest schema mismatch")
        if manifest.get("self_sha256") != self_hash(manifest, "self_sha256"):
            raise LaunchError("launch manifest self hash mismatch")
        if args.supervise:
            return run_supervisor(manifest, sha256(raw))
        if sys.executable != str(PYTHON):
            raise LaunchError("launcher interpreter is not the literal qualified Python")
        for directory in (CONTROL_ROOT, CONTROL_ROOT / "output", CONTROL_ROOT / "logs", TASK_ROOT, CODE_ROOT):
            if directory.is_symlink() or not directory.is_dir():
                raise LaunchError(f"required fresh namespace directory is absent or unsafe: {directory}")
        if CODE_ROOT.is_symlink() or not CODE_ROOT.is_dir():
            raise LaunchError("staged source root is absent or unsafe")
        if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
            raise LaunchError("runner output root already exists; no resume/replay is supported")
        if LAUNCH_RECEIPT.exists() or LAUNCH_RECEIPT.is_symlink() or LAUNCH_LOCK.exists() or LAUNCH_LOCK.is_symlink():
            raise LaunchError("launch receipt/lock already exists; refusing a second launch")
        verify_manifest_files(manifest)
        source_stage_pin = verify_source_stage(manifest)
        input_pins = verify_data_inputs(manifest)
        gpu = gpu_snapshot(manifest["gpu_identity"])
        shm_free = shutil.disk_usage("/dev/shm").free
        if shm_free < MIN_SHM_FREE_BYTES:
            raise LaunchError(f"/dev/shm free bytes below the sealed 5-GiB reserve: {shm_free}")
        supervisor_env = clean_child_environment(manifest)
        lock = {"schema": "fit40_d0_gatedc_launch_lock_v1", "status": "RESERVED", "launcher_pid": os.getpid(), "manifest_raw_sha256": sha256(raw), "timestamp_unix_seconds": time.time()}
        write_json_exclusive(LAUNCH_LOCK, lock, "self_sha256")
        argv = [str(PYTHON), "-B", str(pathlib.Path(__file__).resolve()), "--manifest", str(args.manifest), "--expected-manifest-sha256", sha256(raw), "--supervise"]
        with SUPERVISOR_LOG.open("xb") as supervisor_log:
            proc = subprocess.Popen(argv, cwd=str(CODE_ROOT), env=supervisor_env, stdin=subprocess.DEVNULL, stdout=supervisor_log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        receipt = {
            "schema": "fit40_d0_gatedc_detached_launch_receipt_v1", "status": "STARTED_DETACHED",
            "manifest": {"path": str(args.manifest), "bytes": len(raw), "sha256": sha256(raw)},
            "supervisor": {"pid": int(proc.pid), "start_ticks": proc_start_ticks(proc.pid), "argv": argv, "cwd": str(CODE_ROOT), "started_unix_seconds": time.time()},
            "runner": {"path": str(CODE_ROOT / manifest["runner"]["relative_path"]), "argv": runner_argv(manifest), "output_root": str(OUTPUT_ROOT), "status_path": str(OUTPUT_ROOT / "status.json"), "completion_path": str(OUTPUT_ROOT / "completion.json"), "failure_path": str(OUTPUT_ROOT / "failure.json")},
            "inputs": input_pins, "source_pins_verified": len(manifest["file_pins"]), "source_stage_receipt": source_stage_pin, "staged_code_root": str(CODE_ROOT),
            "gpu_prelaunch_snapshot": gpu, "shm_free_bytes": int(shm_free), "minimum_shm_reserve_bytes": MIN_SHM_FREE_BYTES,
            "environment_allowlist": {k: v for k, v in supervisor_env.items() if k not in ("LD_LIBRARY_PATH", "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")},
            "environment_secrets_recorded": False, "network_or_api_requested": False,
            "qualification_only_flag": False, "optimizer_sequence": "same runner process: Fit40 baseline/zero gate -> strict full-objective backward gate -> V3-head Adam120 -> same-theta GateDC score-only control",
            "automatic_replay": False,
        }
        try:
            pin = write_json_exclusive(LAUNCH_RECEIPT, receipt, "self_sha256")
        except Exception:
            # If the durable launch receipt cannot be sealed, stop that exact
            # detached process group so no untracked training remains active.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise
        print(json.dumps({"schema": receipt["schema"], "status": receipt["status"], **pin, "supervisor_pid": proc.pid, "supervisor_start_ticks": receipt["supervisor"]["start_ticks"], "runner_status_path": receipt["runner"]["status_path"]}, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        payload = {"schema": "fit40_d0_gatedc_launch_failure_v1", "status": "LAUNCH_FAILED_PRESERVED", "type": type(exc).__name__, "message": str(exc)[:1600], "traceback": traceback.format_exc(limit=16), "timestamp_unix_seconds": time.time()}
        try:
            if CONTROL_ROOT.is_dir() and not FAILURE_RECEIPT.exists() and not FAILURE_RECEIPT.is_symlink():
                write_json_exclusive(FAILURE_RECEIPT, payload, "self_sha256")
        except Exception:
            pass
        print(json.dumps({"schema": payload["schema"], "status": payload["status"], "message": payload["message"]}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
