#!/usr/bin/env python3
"""Admit and detach the complete six-row three-backend engineering repeat qualification.

The root dispatcher must grant its exclusive GPU lane before invoking this
launcher. The launcher checks a fresh successful preflight receipt, starts the
strict pilot in a new session under GNU timeout, and seals a PID/start-tick and
log receipt. It does not itself render or score media.
"""
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
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
RUNNER_REL = BASE + "/run_v3_backend_strict_repeat_qualification_attempt09.py"
RUNNER_SHA256 = "2ba5c129c6ad38a5bd719f2224d0076afbd737e9da228ef92051d9f3b18a7e0d"
PREFLIGHT_REL = BASE + "/preflight_v3_backend_strict_repeat_subset_attempt09.py"
PREFLIGHT_SHA256 = "b4404eef40a28ad769b35774e97dc9c3105e0788106d786a181947265c66797c"
REQUESTED_MODELS = ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18")
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
MAX_RUNTIME_SECONDS = 21600
KILL_GRACE_SECONDS = 60
W2L_PYTHON_LAUNCH = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
W2L_PYTHON_PREFIX = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or unsafe: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError("Linux process stat lacks a closing comm delimiter")
    fields = raw[closing + 2:].split()
    if len(fields) < 20:
        raise ValueError("Linux process stat is truncated before starttime")
    return int(fields[19])


def write_exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def safe_control_dir(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"dispatch control directory is not a real directory: {path}")
    else:
        path.mkdir(parents=True, mode=0o700)
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("resolved dispatch control path is unsafe")
    return resolved


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve(strict=True)
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise ValueError("repository root must be a real directory")
    runner_path = repo_root / RUNNER_REL
    if runner_path.is_symlink() or not runner_path.is_file() or sha_file(runner_path) != RUNNER_SHA256:
        raise ValueError("pinned strict repeat runner is missing or has the wrong SHA")
    preflight_script = repo_root / PREFLIGHT_REL
    if (preflight_script.is_symlink() or not preflight_script.is_file()
            or sha_file(preflight_script) != PREFLIGHT_SHA256):
        raise ValueError("pinned read-only preflight source is missing or has the wrong SHA")
    preflight_path = args.preflight_receipt
    preflight, preflight_raw = read_object(preflight_path, "read-only preflight receipt")
    if (preflight.get("schema") != "cross_tfg_backend_strict_repeat_subset_preflight_v1"
            or preflight.get("status") != "PASS_PRECHECK_DISPATCH_ADMISSION_REQUIRED"
            or preflight.get("preflight_sha256") != payload_sha(preflight, "preflight_sha256")
            or preflight.get("performed") is not False
            or preflight.get("rendering_performed") is not False
            or preflight.get("syncnet_scoring_performed") is not False
            or preflight.get("api_calls_performed") is not False
            or preflight.get("model_forward_performed") is not False
            or preflight.get("preflight_script", {}).get("sha256") != PREFLIGHT_SHA256
            or preflight.get("preflight_script", {}).get("path") != str(preflight_script.resolve(strict=True))
            or preflight.get("runner", {}).get("sha256") != RUNNER_SHA256
            or preflight.get("requested_models") != list(REQUESTED_MODELS)
            or preflight.get("output_root") != str(args.output_root.resolve())
            or preflight.get("gpu_headroom_gate", {}).get("status") != "PASS"
            or preflight.get("gpu_headroom_gate", {}).get("expected_uuid") != GPU_UUID):
        raise ValueError("preflight receipt does not pass the exact Wav2Lip+Ditto dispatch contract")
    observed = float(preflight.get("resource_snapshot", {}).get("observed_unix", 0))
    if observed <= 0 or time.time() - observed > args.max_preflight_age_seconds:
        raise ValueError("preflight resource snapshot is absent or too old for dispatch")
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"strict repeat output root must be fresh and absent: {output_root}")

    control_dir = safe_control_dir(args.control_dir)
    log_path = control_dir / "pilot08_stdout_stderr.log"
    receipt_path = control_dir / "detached_launch_receipt.json"
    if any(path.exists() or path.is_symlink() for path in (log_path, receipt_path)):
        raise FileExistsError("detached log or launch receipt already exists")

    # Keep the venv entrypoint as argv[0]. Resolving it first can launch the
    # system Python target and lose the venv site-packages.
    probe = preflight.get("wav2lip_argv0_probe", {})
    python_text = probe.get("argv0")
    if python_text != W2L_PYTHON_LAUNCH:
        raise ValueError("preflight does not pin the expected Wav2Lip venv argv[0]")
    python_path = Path(python_text)
    if not python_path.is_absolute() or not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError("Wav2Lip venv argv[0] is missing or not executable")
    target_text = str(python_path.resolve(strict=True))
    if (target_text != probe.get("resolved_target")
            or target_text != str(Path(preflight.get("wav2lip_python_identity", {}).get("python", "")).resolve(strict=True))
            or probe.get("prefix") != W2L_PYTHON_PREFIX
            or probe.get("numpy") != preflight.get("wav2lip_python_identity", {}).get("numpy")
            or probe.get("torch") != preflight.get("wav2lip_python_identity", {}).get("torch")):
        raise ValueError("preflight Wav2Lip venv target/package identity changed before launch")
    timeout_path_text = shutil.which("timeout")
    if not timeout_path_text:
        raise FileNotFoundError("GNU timeout is unavailable; refusing an unbounded detached job")
    timeout_path = Path(timeout_path_text).resolve(strict=True)
    timeout_sha = sha_file(timeout_path)
    argv = [str(timeout_path), "--signal=TERM", f"--kill-after={KILL_GRACE_SECONDS}s",
            f"{MAX_RUNTIME_SECONDS}s", str(python_path), "-B", str(runner_path),
            "--backend", REQUESTED_MODELS[0], "--backend", REQUESTED_MODELS[1],
          "--backend", REQUESTED_MODELS[2],
            "--output-root", str(output_root), "--execute-strict-repeats"]
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    with log_path.open("xb") as log_stream:
        proc = subprocess.Popen(argv, cwd=repo_root, env=env,
                                stdin=subprocess.DEVNULL, stdout=log_stream,
                                stderr=subprocess.STDOUT, close_fds=True,
                                start_new_session=True)
        started_ticks = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                started_ticks = proc_start_ticks(proc.pid)
                break
            except (FileNotFoundError, ProcessLookupError):
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
        if started_ticks is None:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, 15)
                except ProcessLookupError:
                    pass
            raise RuntimeError("detached timeout process exited before its PID/start-tick identity was recorded")
        receipt: dict[str, Any] = {
            "schema": "cross_tfg_backend_strict_repeat_detached_launch_v1",
            "status": "RUNNING_DETACHED",
            "run_id": preflight.get("run_id"),
            "pid": proc.pid,
            "process_group_id": proc.pid,
            "process_start_ticks": started_ticks,
            "started_unix": time.time(),
            "start_new_session": True,
            "external_timeout": {"path": str(timeout_path), "sha256": timeout_sha,
                                 "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                                 "kill_grace_seconds": KILL_GRACE_SECONDS},
            "argv": argv,
            "python_launch_path": str(python_path),
            "python_resolved_target": target_text,
            "repo_root": str(repo_root),
            "runner": {"path": str(runner_path), "bytes": runner_path.stat().st_size,
                       "sha256": RUNNER_SHA256},
            "output_root": str(output_root),
            "preflight_receipt": {"path": str(preflight_path.resolve(strict=True)),
                                  "bytes": len(preflight_raw), "sha256": sha_bytes(preflight_raw),
                                  "preflight_sha256": preflight.get("preflight_sha256")},
            "requested_models": list(REQUESTED_MODELS),
            "planned_rows": 6, "requested_render_rows": 6,
            "gpu_uuid": GPU_UUID,
            "root_dispatcher_exclusive_gpu_admission_required": True,
            "stdout_stderr_log": str(log_path),
            "rendering_started_by_launcher": False,
            "syncnet_scoring_started_by_launcher": False,
            "api_calls_started_by_launcher": False,
            "dev_or_confirm_media_read": False,
        }
        receipt["launch_receipt_sha256"] = payload_sha(receipt, "launch_receipt_sha256")
        try:
            write_exclusive(receipt_path, (json.dumps(receipt, sort_keys=True, indent=2,
                                                     ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        except Exception:
            try:
                os.killpg(proc.pid, 15)
            except ProcessLookupError:
                pass
            raise
    return {"status": receipt["status"], "pid": proc.pid, "process_start_ticks": started_ticks,
            "process_group_id": proc.pid, "launch_receipt": str(receipt_path),
            "stdout_stderr_log": str(log_path), "output_root": str(output_root),
            "launch_receipt_sha256": receipt["launch_receipt_sha256"],
            "detached": True, "planned_rows": 6, "requested_render_rows": 6,
            "requested_models": list(REQUESTED_MODELS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--max-preflight-age-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.max_preflight_age_seconds <= 0:
        parser.error("max preflight age must be positive")
    try:
        value = run(args)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "DETACHED_LAUNCH_FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
