import json
from pathlib import Path

import pytest

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation import config
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.common import (
    verify_self_hashed_json,
    write_self_hashed_json,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.final import (
    write_blocked_terminal,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.stats import (
    _compatibility,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.validate import (
    ValidationError,
    validate_stage04,
)


def _replacement_rows(gap_c: float = 0.0, gap_d: float = 0.0, offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{index}",
            "source_group": f"group-{index}",
            "gap_c": gap_c,
            "gap_d": gap_d,
            "av_offset": offset,
        }
        for index in range(config.EXPECTED_RECORD_COUNT)
    ]


def test_compatibility_requires_both_noninferiority_bounds_and_offsets() -> None:
    baseline = [0] * config.EXPECTED_RECORD_COUNT
    assert _compatibility(_replacement_rows(), baseline)["passes"] is True
    degraded = _replacement_rows(gap_c=-0.1001)
    assert _compatibility(degraded, baseline)["passes"] is False
    shifted = _replacement_rows(offset=2)
    assert _compatibility(shifted, baseline)["passes"] is False


def test_self_hashed_artifact_corruption_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_self_hashed_json(path, {"stage_id": "x", "protocol_id": config.PROTOCOL_ID, "value": 1})
    payload = verify_self_hashed_json(path)
    payload["value"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        verify_self_hashed_json(path)


def test_blocked_terminal_is_self_hashed_and_validated(tmp_path: Path) -> None:
    write_blocked_terminal(RuntimeError("test blocker"), tmp_path / "04_final")
    result = validate_stage04(tmp_path)
    assert result == {"stage": "04_final", "status": "valid", "scientific_decision": "BLOCKED"}


def test_stage04_validator_rejects_corrupted_terminal(tmp_path: Path) -> None:
    final_dir = tmp_path / "04_final"
    write_blocked_terminal(RuntimeError("test blocker"), final_dir)
    path = final_dir / "final.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"status": "blocked"', '"status": "corrupt"', 1), encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid artifact"):
        validate_stage04(tmp_path)
