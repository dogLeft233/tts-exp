from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.experiments.lrs3_mfa_linear_replacement.run_stage01 import assert_gpu_ready


def test_gpu_gate_rejects_forbidden_hours_without_calling_nvidia_smi() -> None:
    called = False

    def runner(*args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0, stdout="0, 0, 0, 16384\n")

    with pytest.raises(RuntimeError, match="23:00-08:00"):
        assert_gpu_ready(local_hour=23, runner=runner)
    assert called is False


def test_gpu_gate_accepts_only_idle_gpu() -> None:
    result = assert_gpu_ready(local_hour=12, runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0, 0, 0, 16384\n"))
    assert result["gpus"][0]["memory_used_mib"] == 0
    with pytest.raises(RuntimeError, match="occupied"):
        assert_gpu_ready(local_hour=12, runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="0, 0, 10, 16384\n"))
