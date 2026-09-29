from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.tts_native_gain_attribution import common, generation
from scripts.experiments.tts_native_gain_attribution.leaptalk_adapter import (
    command_argv,
    discover,
)
from scripts.experiments.tts_native_gain_attribution.perception import _analyze_ratings
from scripts.experiments.tts_native_gain_attribution.validate import (
    _independent_distance_matrix,
)


def test_adapter_command_is_argv_safe_for_paths_with_spaces() -> None:
    command = command_argv(
        "python run.py --audio {audio} --output {output}",
        {"audio": "/tmp/with spaces/input.wav", "output": "/tmp/with spaces/out.mp4"},
    )
    assert command == [
        "python",
        "run.py",
        "--audio",
        "/tmp/with spaces/input.wav",
        "--output",
        "/tmp/with spaces/out.mp4",
    ]


def test_base_only_leaptalk_binding_stays_dependency_blocked(tmp_path: Path) -> None:
    config_path = tmp_path / "leaptalk.json"
    config_path.write_text('{"root": "' + str(tmp_path) + '", "command": ["python", "run.py"]}', encoding="utf-8")
    with pytest.raises(common.DependencyBlockedError):
        discover(config_path)


def test_resource_gate_keeps_gpu_and_disk_budgets_separate(monkeypatch) -> None:
    monkeypatch.setattr(common, "disk_snapshot", lambda: {"total_bytes": 1 << 40, "used_bytes": 1 << 30, "free_bytes": 1 << 39})
    gpu = {"available": True, "rows": [{"index": 0, "memory_free_mib": 16_384, "utilization_gpu_percent": 0}]}
    monkeypatch.setattr(common, "gpu_snapshot", lambda: gpu)
    monkeypatch.setattr(common, "gpu_compute_pids", list)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)
    result = common.resource_gate(gpu_peak_bytes=2 << 30, disk_temp_bytes=128, disk_persistent_bytes=256, require_gpu=True)
    assert result["gate"] == "PASS"
    assert result["gpu_peak_bytes"] == 2 << 30
    assert result["disk_temp_bytes"] == 128
    assert result["disk_persistent_bytes"] == 256


def test_validator_distance_recompute_is_sensitive_to_matrix_content() -> None:
    visual = np.zeros((4, 1024), dtype=np.float32)
    audio = np.zeros((4, 1024), dtype=np.float32)
    expected = _independent_distance_matrix(visual, audio)
    altered = expected.copy()
    altered[1, 15] += 0.25
    assert not np.allclose(expected, altered, atol=1e-4, rtol=1e-6)


def test_empty_human_ratings_are_explicitly_unassessed(tmp_path: Path) -> None:
    assert _analyze_ratings(tmp_path / "empty.csv", {"mapping": []}, kind="sync")["status"] == "PERCEPTION_NOT_ASSESSED"
    assert _analyze_ratings(tmp_path / "empty.csv", {"mapping": []}, kind="quality")["status"] == "QUALITY_NOT_ASSESSED"


def test_foreign_gpu_process_is_typed_resource_wait_and_logged(monkeypatch, tmp_path: Path) -> None:
    class FakeProcess:
        pid = 123
        returncode = -15

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(common, "executable_path", lambda _path, _name: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(common.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(common, "process_tree_pids", lambda _pid: {123})
    monkeypatch.setattr(common, "gpu_snapshot", lambda: {"available": True, "rows": []})
    monkeypatch.setattr(common, "gpu_compute_pids", lambda: [999])

    log_path = tmp_path / "cell" / "adapter.log"
    with pytest.raises(common.ResourceWaitError):
        common.run_monitored(["fake"], cwd=None, log_path=log_path, interval_seconds=0.1)
    resource_log = log_path.with_suffix(".log.resource.json")
    assert resource_log.is_file()
    assert '"status": "RESOURCE_WAIT"' in resource_log.read_text(encoding="utf-8")


def test_interrupted_generation_media_is_quarantined_for_retry(tmp_path: Path) -> None:
    output_dir = tmp_path / "cell"
    output_dir.mkdir()
    (output_dir / "video.mp4").write_bytes(b"partial-video")
    (output_dir / "response.json").write_bytes(b"partial-response")

    generation._quarantine_partial(output_dir)

    assert not (output_dir / "video.mp4").exists()
    assert not (output_dir / "response.json").exists()
    assert (output_dir / "attempts" / "attempt1" / "video.mp4").read_bytes() == b"partial-video"
    assert (output_dir / "attempts" / "attempt1" / "response.json").read_bytes() == b"partial-response"
