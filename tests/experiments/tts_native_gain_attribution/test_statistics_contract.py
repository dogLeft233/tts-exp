import json

import numpy as np

from scripts.experiments.tts_native_gain_attribution import config
from scripts.experiments.tts_native_gain_attribution.analysis import (
    curve_metrics,
    four_cell,
)
from scripts.experiments.tts_native_gain_attribution.common import (
    resource_gate,
    write_self_hashed_json,
)
from scripts.experiments.tts_native_gain_attribution.syncnet import (
    _control_result,
    _shift_pcm,
)
from scripts.experiments.tts_native_gain_attribution.validate import (
    _self,
    _verify_file_binding,
)


def test_curve_reduces_time_before_lag_and_uses_registered_offset() -> None:
    matrix = np.full((4, config.LAG_COUNT), 4.0, dtype=np.float32)
    matrix[:, config.VSHIFT] = 1.0
    metrics = curve_metrics(matrix, [0, 1, 2, 3])
    assert metrics["sync_d"] == 1.0
    assert metrics["background_b"] == 4.0
    assert metrics["sync_c"] == 3.0
    assert metrics["official_offset"] == 0


def test_four_cell_decomposition_identity() -> None:
    result = four_cell(2.0, 3.0, 4.0, 7.0)
    assert result["generation"] == 2.0
    assert result["evaluation"] == 1.0
    assert result["interaction"] == 2.0
    assert result["generation"] + result["evaluation"] + result["interaction"] == result["total"]


def test_pcm_shift_preserves_length_and_direction() -> None:
    values = np.arange(10, dtype=np.int16)
    assert _shift_pcm(values, 2).tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert _shift_pcm(values, -2).tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 0, 0]
    assert len(_shift_pcm(values, 2)) == len(values)


def test_delay_control_sign_and_identity_labels() -> None:
    baseline = np.full((60, config.LAG_COUNT), 5.0, dtype=np.float32)
    baseline[:, 15] = 1.0
    plus = np.full_like(baseline, 5.0)
    plus[:, 20] = 1.0
    minus = np.full_like(baseline, 5.0)
    minus[:, 10] = 1.0
    support = list(range(20, 40))
    assert _control_result(baseline, plus, shift_name="PLUS_200MS", support=support)["status"] == "DELAY_DETECTED"
    assert _control_result(baseline, minus, shift_name="MINUS_200MS", support=support)["status"] == "DELAY_DETECTED"
    assert _control_result(baseline, baseline.copy(), shift_name="IDENTITY", support=support)["status"] == "IDENTITY_PASS"


def test_validator_rejects_tampered_self_hashed_artifact(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    write_self_hashed_json(path, {"status": "COMPLETE"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "FALSE_COMPLETE"
    path.write_text(json.dumps(payload), encoding="utf-8")
    errors: list[str] = []
    assert _self(path, errors) == {}
    assert errors


def test_validator_rejects_file_hash_replacement(tmp_path) -> None:
    path = tmp_path / "bound.bin"
    path.write_bytes(b"replacement")
    errors: list[str] = []
    assert _verify_file_binding({"path": str(path), "sha256": "0" * 64}, "bound", errors) is None
    assert errors and "sha256" in errors[0]


def test_validator_rejects_missing_bound_file(tmp_path) -> None:
    errors: list[str] = []
    assert _verify_file_binding({"path": str(tmp_path / "missing.bin"), "sha256": "0" * 64}, "missing", errors) is None
    assert errors and "missing" in errors[0]


def test_resource_gate_reports_insufficient_disk_without_starting_gpu_work() -> None:
    result = resource_gate(estimated_persistent=1 << 60, estimated_temp=0, require_gpu=False)
    assert result["gate"] == "RESOURCE_WAIT"
    assert any("disk free" in item for item in result["errors"])
