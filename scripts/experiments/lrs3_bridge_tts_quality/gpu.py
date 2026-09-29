from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import config
from .common import ProtocolError


def _query(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"nvidia-smi failed with exit {result.returncode}: {result.stderr[-1000:]}")
    return result.stdout


def gpu_snapshot() -> dict[str, Any]:
    rows: list[dict[str, int]] = []
    output = _query([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise ProtocolError("nvidia-smi GPU output is malformed")
        index, utilization, memory_used, memory_total = (int(float(item)) for item in fields)
        rows.append(
            {
                "index": index,
                "utilization_gpu_percent": utilization,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
            }
        )
    if not rows:
        raise ProtocolError("nvidia-smi returned no GPUs")
    processes_output = _query([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    processes = [line.strip() for line in processes_output.splitlines() if line.strip()]
    return {"gpus": rows, "active_compute_processes": processes, "pid": os.getpid()}


def assert_gpu_idle() -> dict[str, Any]:
    snapshot = gpu_snapshot()
    if snapshot["active_compute_processes"]:
        raise ProtocolError(f"GPU has active compute processes: {snapshot['active_compute_processes']}")
    busy = [
        row for row in snapshot["gpus"] if int(row["utilization_gpu_percent"]) != 0
    ]
    if busy:
        raise ProtocolError(f"GPU is occupied: {busy}")
    return snapshot


@contextlib.contextmanager
def gpu_lease(stage: str) -> Iterator[dict[str, Any]]:
    """Serialize GPU work across experiment processes and record idle state.

    The lease is intentionally process-wide and non-reentrant.  A stage must
    release it before another stage can load a model, which keeps WavLM,
    Qwen-TTS, Wav2Lip, and SyncNet from sharing a 16 GiB card accidentally.
    """

    path = Path(config.GPU_LOCK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError(f"another experiment already holds the GPU lease: {path}") from exc
        before = assert_gpu_idle()
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"stage": stage, "pid": os.getpid(), "before": before}) + "\n")
        handle.flush()
        try:
            yield {"stage": stage, "pid": os.getpid(), "before": before}
        finally:
            # Do not turn a finished child process into a false success.  The
            # snapshot is evidence; a non-idle card is reported to the caller.
            after_error = None
            try:
                after = gpu_snapshot()
            except Exception as exc:  # noqa: BLE001 - cleanup cannot mask stage failure
                # Cleanup must never mask the exception raised by the stage
                # itself.  The missing post-snapshot is kept in the lease log.
                after = None
                after_error = f"{type(exc).__name__}: {exc}"
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"stage": stage, "pid": os.getpid(), "before": before, "after": after, "after_error": after_error}) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def release_torch_memory() -> dict[str, Any]:
    """Best-effort release after a model stage; never hides a stage error."""

    import gc

    gc.collect()
    result: dict[str, Any] = {"gc": True, "cuda_empty_cache": False}
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            result["cuda_empty_cache"] = True
    except Exception as exc:  # noqa: BLE001 - cleanup is best effort
        result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    return result
