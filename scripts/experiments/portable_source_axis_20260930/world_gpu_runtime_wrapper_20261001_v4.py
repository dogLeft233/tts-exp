#!/usr/bin/env python3
"""Run the sealed WORLD module with durable five-flag forward telemetry.

The scientific runner stays byte-identical. This wrapper enforces the five
runtime flags and writes one hash-chained JSONL record for every global
``nn.Module`` forward pre-hook. It fsyncs the event stream and atomically
checkpoints counts every 32 calls, then seals the complete event stream after
the runner exits.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_FLAGS = {
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": False,
    "matmul_allow_tf32": False,
}
RUNNER_MODULE = "portable_source_axis_20260930.world_phone_transfer_gpu_evaluate_20261001"
CHECKPOINT_EVERY = 32


def _torch_state(torch: Any) -> dict[str, Any]:
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("runtime event write made no progress")
        view = view[written:]


def _write_atomic(path: Path, payload: dict[str, Any], serial: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{serial}.tmp")
    data = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: world_gpu_runtime_wrapper_20261001_v4.py <runner.py> <runner args...>")
    requested_runner = Path(sys.argv[1]).resolve(strict=True)
    runtime_path_text = os.environ.get("WORLD_RUNTIME_FLAGS_PATH")
    events_path_text = os.environ.get("WORLD_RUNTIME_FORWARD_EVENTS_PATH")
    if not runtime_path_text or not events_path_text:
        raise SystemExit("WORLD_RUNTIME_FLAGS_PATH and WORLD_RUNTIME_FORWARD_EVENTS_PATH are required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("CUBLAS_WORKSPACE_CONFIG must be set to :4096:8 before importing torch")

    runtime_path = Path(runtime_path_text).resolve()
    events_path = Path(events_path_text).resolve()
    if runtime_path == events_path:
        raise SystemExit("runtime sidecar and event stream must have distinct paths")
    if runtime_path.exists() or events_path.exists():
        raise SystemExit("refusing to overwrite existing runtime telemetry")
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event_fd = os.open(events_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    import torch

    before_policy = _torch_state(torch)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    after_policy = _torch_state(torch)
    if {key: after_policy[key] for key in EXPECTED_FLAGS} != EXPECTED_FLAGS:
        os.close(event_fd)
        raise SystemExit(f"runtime flag policy did not take effect: {after_policy}")

    module_spec = importlib.util.find_spec(RUNNER_MODULE)
    if module_spec is None or not module_spec.origin:
        os.close(event_fd)
        raise SystemExit(f"runner module could not be resolved: {RUNNER_MODULE}")
    module_origin = Path(module_spec.origin).resolve(strict=True)
    if module_origin != requested_runner:
        os.close(event_fd)
        raise SystemExit(f"runner module path mismatch: resolved={module_origin} requested={requested_runner}")

    wrapper_path = Path(__file__).resolve()
    runner_sha = hashlib.sha256(module_origin.read_bytes()).hexdigest()
    wrapper_sha = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    call_counts: Counter[str] = Counter()
    event_count = 0
    event_bytes = 0
    durable_event_count = 0
    chain_sha = "0" * 64
    mismatch_count = 0
    mismatch_examples: list[dict[str, Any]] = []
    checkpoint_serial = 0
    record: dict[str, Any] = {
        "schema_version": 4,
        "classification": "same-process five-flag pre-hook telemetry with hash-chained per-module events and durable checkpoints",
        "captured_at_utc_epoch": time.time(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "runner_module": RUNNER_MODULE,
        "runner_path": str(module_origin),
        "runner_sha256": runner_sha,
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": wrapper_sha,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "policy": EXPECTED_FLAGS,
        "torch_flags_before_policy": before_policy,
        "torch_flags_after_policy_before_forward": after_policy,
        "forward_event_log_path": str(events_path),
        "forward_event_checkpoint_interval": CHECKPOINT_EVERY,
        "nn_module_forward_call_count": 0,
        "durable_forward_event_count": 0,
        "forward_event_bytes": 0,
        "forward_event_chain_sha256": chain_sha,
        "forward_event_file_sha256": None,
        "nn_module_call_counts_by_type": {},
        "flag_mismatch_count": 0,
        "flag_mismatch_examples": [],
        "torch_flags_after_runner": None,
        "runner_exit_code": None,
        "runtime_forward_gate": "not_started",
    }

    def checkpoint(status: str) -> None:
        nonlocal checkpoint_serial, durable_event_count
        os.fsync(event_fd)
        durable_event_count = event_count
        checkpoint_serial += 1
        record.update({
            "nn_module_forward_call_count": event_count,
            "durable_forward_event_count": durable_event_count,
            "forward_event_bytes": event_bytes,
            "forward_event_chain_sha256": chain_sha,
            "nn_module_call_counts_by_type": dict(sorted(call_counts.items())),
            "flag_mismatch_count": mismatch_count,
            "flag_mismatch_examples": mismatch_examples,
            "runtime_forward_gate": status,
            "last_checkpoint_at_utc_epoch": time.time(),
        })
        _write_atomic(runtime_path, record, checkpoint_serial)

    _write_atomic(runtime_path, record, checkpoint_serial)

    def check_before_module_forward(module: Any, _inputs: Any) -> None:
        nonlocal event_count, event_bytes, chain_sha, mismatch_count
        state = _torch_state(torch)
        module_name = f"{type(module).__module__}.{type(module).__qualname__}"
        call_counts[module_name] += 1
        current = {key: state[key] for key in EXPECTED_FLAGS}
        event = {
            "seq": event_count + 1,
            "module": module_name,
            "flags_at_call": current,
            "monotonic_ns": time.monotonic_ns(),
        }
        event_hash = hashlib.sha256(bytes.fromhex(chain_sha) + _canonical(event)).hexdigest()
        line = _canonical({**event, "event_chain_sha256": event_hash}) + b"\n"
        _write_all(event_fd, line)
        event_count += 1
        event_bytes += len(line)
        chain_sha = event_hash
        if current != EXPECTED_FLAGS:
            mismatch_count += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append({"seq": event_count, "module": module_name, "flags_at_call": current})
        if event_count % CHECKPOINT_EVERY == 0:
            checkpoint("running" if mismatch_count == 0 else "fail")
        if current != EXPECTED_FLAGS:
            raise RuntimeError(f"WORLD runtime flags changed before module forward: {module_name} {current}")

    hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(check_before_module_forward)
    sys.argv = [str(module_origin), *sys.argv[2:]]
    exit_code = 1
    caught_system_exit: SystemExit | None = None
    caught_exception: BaseException | None = None
    try:
        runpy.run_module(RUNNER_MODULE, run_name="__main__", alter_sys=True)
        exit_code = 0
    except SystemExit as exc:
        caught_system_exit = exc
        exit_code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:
        caught_exception = exc
        exit_code = 1
    finally:
        hook_handle.remove()
        state_after = _torch_state(torch)
        if exit_code == 0 and (event_count == 0 or mismatch_count):
            exit_code = 55
        try:
            os.fsync(event_fd)
            durable_event_count = event_count
            os.close(event_fd)
            event_digest = hashlib.sha256()
            with events_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    event_digest.update(block)
            checkpoint_serial += 1
            record.update({
                "nn_module_forward_call_count": event_count,
                "durable_forward_event_count": durable_event_count,
                "forward_event_bytes": events_path.stat().st_size,
                "forward_event_chain_sha256": chain_sha,
                "forward_event_file_sha256": event_digest.hexdigest(),
                "nn_module_call_counts_by_type": dict(sorted(call_counts.items())),
                "flag_mismatch_count": mismatch_count,
                "flag_mismatch_examples": mismatch_examples,
                "torch_flags_after_runner": state_after,
                "runner_exit_code": exit_code,
                "runtime_forward_gate": "pass" if event_count > 0 and mismatch_count == 0 and exit_code == 0 else "fail",
                "finished_at_utc_epoch": time.time(),
            })
            _write_atomic(runtime_path, record, checkpoint_serial)
        except BaseException as telemetry_error:
            try:
                os.close(event_fd)
            except OSError:
                pass
            raise RuntimeError(f"failed to seal WORLD runtime telemetry: {telemetry_error}") from telemetry_error

    if caught_exception is not None:
        raise caught_exception
    if caught_system_exit is not None:
        raise SystemExit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
