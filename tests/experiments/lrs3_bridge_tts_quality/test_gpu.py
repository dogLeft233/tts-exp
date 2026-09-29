from __future__ import annotations

from pathlib import Path

from scripts.experiments.lrs3_bridge_tts_quality import config, gpu


def test_gpu_lease_is_serial_and_records_before_after(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "GPU_LOCK_PATH", tmp_path / "gpu.lock")
    snapshot = {"gpus": [{"index": 0, "utilization_gpu_percent": 0, "memory_used_mib": 0, "memory_total_mib": 16384}], "active_compute_processes": [], "pid": 1}
    monkeypatch.setattr(gpu, "gpu_snapshot", lambda: snapshot)
    with gpu.gpu_lease("unit-test") as lease:
        assert lease["stage"] == "unit-test"
    assert (tmp_path / "gpu.lock").read_text(encoding="utf-8").count('"after"') == 1
