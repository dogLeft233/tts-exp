from __future__ import annotations

import subprocess
from types import SimpleNamespace
from pathlib import Path

import scripts.experiments.mfa_linear_video_retiming.run as runner


def _config() -> dict[str, object]:
    return {"budget": {"minimum_free_gpu_mib": 5000,
                       "minimum_available_ram_bytes": 1,
                       "minimum_free_disk_bytes": 1}}


def _completed(args: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


def test_resource_preflight_distinguishes_transient_utilization_from_live_compute(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _path: SimpleNamespace(free=100))
    monkeypatch.setattr(runner, "_available_ram_bytes", lambda: 100)
    processes_output = {"value": ""}

    def fake_run(args, **_kwargs):
        if "--query-gpu=" in args[1]:
            return _completed(args, "0, 16000, 85\n")
        return _completed(args, processes_output["value"])

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    transient = runner._resource_preflight(_config(), tmp_path, require_gpu=True)
    assert transient["status"] == "RESOURCE_WAIT"
    assert transient["reason"] == "GPU_HIGH_UTILIZATION"

    processes_output["value"] = "GPU-uuid, 456, external-worker\n"
    external = runner._resource_preflight(_config(), tmp_path, require_gpu=True)
    assert external["status"] == "RESOURCE_WAIT"
    assert external["reason"] == "EXTERNAL_COMPUTE_PROCESS_PRESENT"
    assert external["compute_processes"] == ["GPU-uuid, 456, external-worker"]


def test_wait_for_resource_preflight_only_retries_transient_gpu_utilization(
    monkeypatch, tmp_path: Path,
) -> None:
    sequence = [
        {"status": "RESOURCE_WAIT", "reason": "GPU_HIGH_UTILIZATION"},
        {"status": "RESOURCE_WAIT", "reason": "GPU_HIGH_UTILIZATION"},
        {"status": "READY"},
    ]
    sleeps: list[float] = []

    def fake_preflight(*_args, **_kwargs):
        return sequence.pop(0)

    monkeypatch.setattr(runner, "_resource_preflight", fake_preflight)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    result = runner._wait_for_resource_preflight(
        _config(), tmp_path, require_gpu=True, max_wait_seconds=5.0, poll_interval_seconds=2.0,
    )

    assert result["status"] == "READY"
    assert result["cooldown_poll_count"] == 2
    assert sleeps == [2.0, 2.0]


def test_wait_for_resource_preflight_does_not_poll_live_external_compute(
    monkeypatch, tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(runner, "_resource_preflight", lambda *_args, **_kwargs: {
        "status": "RESOURCE_WAIT", "reason": "EXTERNAL_COMPUTE_PROCESS_PRESENT",
        "compute_processes": ["GPU-uuid, 456, external-worker"],
    })
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    result = runner._wait_for_resource_preflight(_config(), tmp_path, require_gpu=True)

    assert result["status"] == "RESOURCE_WAIT"
    assert result["reason"] == "EXTERNAL_COMPUTE_PROCESS_PRESENT"
    assert result["cooldown_poll_count"] == 0
    assert sleeps == []
