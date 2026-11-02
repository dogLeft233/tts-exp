#!/usr/bin/env python3
"""Run the sealed WORLD module with five-flag forward checkpoints.

The scientific runner remains byte-identical. A same-process global
``nn.Module`` forward pre-hook checks all five runtime flags before every
module call and updates an in-memory hash chain and module counters. A compact
atomic sidecar is fsynced every five seconds, after each completed/failed
context, and at shutdown; no per-forward event file is written.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import signal
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
RUNNER_MODULE = "portable_source_axis_20260930.world_phone_shape_mean_dynamic_gpu_20261001_attempt08"
CHECKPOINT_INTERVAL_SECONDS = 5.0


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


def _write_atomic(path: Path, payload: dict[str, Any], serial: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{serial}.tmp")
    data = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("runtime checkpoint write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # Some filesystems do not support syncing directory descriptors.
        pass


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: world_gpu_runtime_wrapper_20261001_v5.py <runner.py> <runner args...>")
    requested_runner = Path(sys.argv[1]).resolve(strict=True)
    runtime_path_text = os.environ.get("WORLD_RUNTIME_FLAGS_PATH")
    if not runtime_path_text:
        raise SystemExit("WORLD_RUNTIME_FLAGS_PATH must point to the job-control sidecar")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("CUBLAS_WORKSPACE_CONFIG must be set to :4096:8 before importing torch")

    runtime_path = Path(runtime_path_text).resolve()
    if runtime_path.exists():
        raise SystemExit("refusing to overwrite existing runtime sidecar")
    runtime_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    before_policy = _torch_state(torch)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    after_policy = _torch_state(torch)
    if {key: after_policy[key] for key in EXPECTED_FLAGS} != EXPECTED_FLAGS:
        raise SystemExit(f"runtime flag policy did not take effect: {after_policy}")

    module_spec = importlib.util.find_spec(RUNNER_MODULE)
    if module_spec is None or not module_spec.origin:
        raise SystemExit(f"runner module could not be resolved: {RUNNER_MODULE}")
    module_origin = Path(module_spec.origin).resolve(strict=True)
    if module_origin != requested_runner:
        raise SystemExit(f"runner module path mismatch: resolved={module_origin} requested={requested_runner}")

    wrapper_path = Path(__file__).resolve()
    runner_sha = hashlib.sha256(module_origin.read_bytes()).hexdigest()
    wrapper_sha = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    call_counts: Counter[str] = Counter()
    forward_count = 0
    mismatch_count = 0
    mismatch_examples: list[dict[str, Any]] = []
    last_flags: dict[str, bool] | None = None
    chain_sha = "0" * 64
    checkpoint_serial = 0
    last_checkpoint_monotonic = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": 5,
        "classification": "same-process five-flag global nn.Module pre-hook; every forward checked, compact hash-chain checkpoints persisted",
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
        "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "checkpoint_triggers": ["elapsed_interval", "context_complete", "context_failed", "shutdown"],
        "nn_module_forward_call_count": 0,
        "nn_module_call_counts_by_type": {},
        "last_forward_flags": None,
        "hash_chain_initial": "0" * 64,
        "hash_chain_last": "0" * 64,
        "flag_mismatch_count": 0,
        "flag_mismatch_examples": [],
        "last_checkpoint_reason": "initialized_before_runner",
        "last_checkpoint_forward_count": 0,
        "last_checkpoint_at_utc_epoch": time.time(),
        "torch_flags_after_runner": None,
        "runner_exit_code": None,
        "controlled_stop_reason": None,
        "runtime_forward_gate": "not_started",
    }

    def persist_checkpoint(reason: str, *, final: bool = False, state_after: dict[str, Any] | None = None, exit_code: int | None = None) -> None:
        nonlocal checkpoint_serial, last_checkpoint_monotonic
        checkpoint_serial += 1
        record.update({
            "nn_module_forward_call_count": forward_count,
            "nn_module_call_counts_by_type": dict(sorted(call_counts.items())),
            "last_forward_flags": last_flags,
            "hash_chain_last": chain_sha,
            "flag_mismatch_count": mismatch_count,
            "flag_mismatch_examples": mismatch_examples,
            "last_checkpoint_reason": reason,
            "last_checkpoint_forward_count": forward_count,
            "last_checkpoint_at_utc_epoch": time.time(),
            "runtime_forward_gate": (
                "pass" if final and exit_code == 0 and forward_count > 0 and mismatch_count == 0
                else "fail" if final or mismatch_count
                else "running"
            ),
        })
        if state_after is not None:
            record["torch_flags_after_runner"] = state_after
        if exit_code is not None:
            record["runner_exit_code"] = exit_code
        _write_atomic(runtime_path, record, checkpoint_serial)
        last_checkpoint_monotonic = time.monotonic()

    persist_checkpoint("initialized_before_runner")

    def request_controlled_stop(_signum: int, _frame: Any) -> None:
        record["controlled_stop_reason"] = "external_disk_free_guard_SIGUSR1"
        raise SystemExit(74)

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_controlled_stop)

    def check_before_module_forward(module: Any, _inputs: Any) -> None:
        nonlocal forward_count, mismatch_count, chain_sha, last_flags
        state = _torch_state(torch)
        flags = {key: state[key] for key in EXPECTED_FLAGS}
        module_name = f"{type(module).__module__}.{type(module).__qualname__}"
        forward_count += 1
        call_counts[module_name] += 1
        last_flags = flags
        event = {"seq": forward_count, "module": module_name, "flags_at_call": flags}
        chain_sha = hashlib.sha256(bytes.fromhex(chain_sha) + _canonical(event)).hexdigest()
        if flags != EXPECTED_FLAGS:
            mismatch_count += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(event)
        now = time.monotonic()
        if flags != EXPECTED_FLAGS:
            persist_checkpoint("flag_mismatch", final=True, state_after=state, exit_code=56)
            raise RuntimeError(f"WORLD runtime flags changed before module forward: {event}")
        if now - last_checkpoint_monotonic >= CHECKPOINT_INTERVAL_SECONDS:
            persist_checkpoint("elapsed_interval")

    try:
        runner = importlib.import_module(RUNNER_MODULE)
    except BaseException:
        raise
    if Path(runner.__file__).resolve(strict=True) != module_origin:
        raise SystemExit("imported runner origin changed after module resolution")

    hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(check_before_module_forward)

    original_append_jsonl = runner._append_jsonl

    def append_jsonl_with_context_checkpoint(path: Path, value: Any) -> None:
        original_append_jsonl(path, value)
        if Path(path).name == "attempts.jsonl" and value.get("status") in {"complete", "failed"}:
            persist_checkpoint(f"context_{value.get('status')}")

    runner._append_jsonl = append_jsonl_with_context_checkpoint
    sys.argv = [str(module_origin), *sys.argv[2:]]
    exit_code = 1
    caught_system_exit: SystemExit | None = None
    caught_exception: BaseException | None = None
    try:
        result = runner.main()
        exit_code = int(result) if isinstance(result, int) else 0
    except SystemExit as exc:
        caught_system_exit = exc
        exit_code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:
        caught_exception = exc
        exit_code = 1
    finally:
        runner._append_jsonl = original_append_jsonl
        hook_handle.remove()
        state_after = _torch_state(torch)
        if exit_code == 0 and (forward_count == 0 or mismatch_count):
            exit_code = 55
        persist_checkpoint("shutdown", final=True, state_after=state_after, exit_code=exit_code)

    if caught_exception is not None:
        raise caught_exception
    if caught_system_exit is not None:
        raise SystemExit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
