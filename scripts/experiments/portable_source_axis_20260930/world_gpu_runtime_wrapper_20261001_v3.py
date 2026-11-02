#!/usr/bin/env python3
"""Run the sealed WORLD module with enforced five-flag provenance.

The launcher sets CUBLAS_WORKSPACE_CONFIG before Python starts. This wrapper
records the pre-policy state, enforces the frozen deterministic/TF32 flags,
validates that the module resolves to the exact runner file, and checks the
five flags before every ``nn.Module`` forward call in the same process.
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


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: world_gpu_runtime_wrapper_20261001_v3.py <runner.py> <runner args...>")
    requested_runner = Path(sys.argv[1]).resolve(strict=True)
    runtime_path_text = os.environ.get("WORLD_RUNTIME_FLAGS_PATH")
    if not runtime_path_text:
        raise SystemExit("WORLD_RUNTIME_FLAGS_PATH must point to the job-control sidecar")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("CUBLAS_WORKSPACE_CONFIG must be set to :4096:8 before importing torch")

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

    spec = importlib.util.find_spec(RUNNER_MODULE)
    if spec is None or not spec.origin:
        raise SystemExit(f"runner module could not be resolved: {RUNNER_MODULE}")
    module_origin = Path(spec.origin).resolve(strict=True)
    if module_origin != requested_runner:
        raise SystemExit(f"runner module path mismatch: resolved={module_origin} requested={requested_runner}")

    runtime_path = Path(runtime_path_text)
    wrapper_path = Path(__file__).resolve()
    runner_sha = hashlib.sha256(module_origin.read_bytes()).hexdigest()
    wrapper_sha = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    call_counts: Counter[str] = Counter()
    mismatch_events: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "schema_version": 3,
        "classification": "same-process runtime provenance; exact package module path and global nn.Module pre-hook",
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
        "nn_module_forward_call_count": 0,
        "nn_module_call_counts_by_type": {},
        "flag_mismatch_count": 0,
        "flag_mismatch_examples": [],
        "torch_flags_after_runner": None,
        "runner_exit_code": None,
        "runtime_forward_gate": "not_started",
    }
    _write_atomic(runtime_path, record)

    def check_before_module_forward(module: Any, _inputs: Any) -> None:
        state = _torch_state(torch)
        call_counts[f"{type(module).__module__}.{type(module).__qualname__}"] += 1
        current = {key: state[key] for key in EXPECTED_FLAGS}
        if current != EXPECTED_FLAGS:
            if len(mismatch_events) < 20:
                mismatch_events.append({"module": f"{type(module).__module__}.{type(module).__qualname__}", "flags": current})
            raise RuntimeError(f"WORLD forward runtime flags changed: {current}")

    hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(check_before_module_forward)
    sys.argv = [str(module_origin), *sys.argv[2:]]
    exit_code = 1
    caught_system_exit: SystemExit | None = None
    try:
        runpy.run_module(RUNNER_MODULE, run_name="__main__", alter_sys=True)
        exit_code = 0
    except SystemExit as exc:
        caught_system_exit = exc
        exit_code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        hook_handle.remove()
        state_after = _torch_state(torch)
        forward_calls = int(sum(call_counts.values()))
        if exit_code == 0 and (forward_calls == 0 or mismatch_events):
            exit_code = 55
        record.update({
            "nn_module_forward_call_count": forward_calls,
            "nn_module_call_counts_by_type": dict(sorted(call_counts.items())),
            "flag_mismatch_count": len(mismatch_events),
            "flag_mismatch_examples": mismatch_events,
            "torch_flags_after_runner": state_after,
            "runner_exit_code": exit_code,
            "runtime_forward_gate": "pass" if forward_calls > 0 and not mismatch_events and exit_code == 0 else "fail",
            "finished_at_utc_epoch": time.time(),
        })
        _write_atomic(runtime_path, record)
    if caught_system_exit is not None:
        raise SystemExit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
