#!/usr/bin/env python3
"""Capture WORLD runtime determinism/TF32 provenance, then run the sealed CLI.

The wrapper intentionally does not alter TF32 settings. The evaluator enforces
its frozen three deterministic flags; this sidecar records the two TF32 flags
and CUBLAS workspace setting from the same Python process that performs all
model forwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any


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
        raise SystemExit("usage: world_gpu_runtime_wrapper.py <runner.py> <runner args...>")
    runner_path = Path(sys.argv[1]).resolve(strict=True)
    runtime_path_text = os.environ.get("WORLD_RUNTIME_FLAGS_PATH")
    if not runtime_path_text:
        raise SystemExit("WORLD_RUNTIME_FLAGS_PATH must point to the job-control sidecar")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("CUBLAS_WORKSPACE_CONFIG must be set to :4096:8 before importing torch")

    import torch

    runtime_path = Path(runtime_path_text)
    wrapper_path = Path(__file__).resolve()
    runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    wrapper_sha = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    before = _torch_state(torch)
    record: dict[str, Any] = {
        "schema_version": 1,
        "classification": "runtime provenance from the same process as all WORLD forwards",
        "captured_at_utc_epoch": time.time(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "runner_path": str(runner_path),
        "runner_sha256": runner_sha,
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": wrapper_sha,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_policy": "preserve PyTorch runtime defaults; evaluator source does not assign either TF32 flag",
        "torch_flags_before_runner": before,
        "torch_flags_after_runner": None,
        "runner_exit_code": None,
    }
    _write_atomic(runtime_path, record)

    sys.argv = [str(runner_path), *sys.argv[2:]]
    exit_code = 1
    try:
        runpy.run_path(str(runner_path), run_name="__main__")
        exit_code = 0
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        raise
    finally:
        record["torch_flags_after_runner"] = _torch_state(torch)
        record["runner_exit_code"] = exit_code
        record["finished_at_utc_epoch"] = time.time()
        _write_atomic(runtime_path, record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
