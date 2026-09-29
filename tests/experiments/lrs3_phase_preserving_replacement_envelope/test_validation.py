import shutil
from pathlib import Path

import pytest

from scripts.experiments.lrs3_phase_preserving_replacement_envelope import config
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.common import (
    verify_self_hashed_json,
    write_self_hashed_json,
)
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.stats import (
    _compatibility,
)
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.validate import (
    ValidationError,
    validate_stage00,
    validate_stage01,
    validate_stage02,
    validate_stage03,
    validate_stage04,
    validate_stage05,
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
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        verify_self_hashed_json(path)


def test_stage00_validator_rejects_corrupted_artifact(tmp_path: Path) -> None:
    source = config.RUN_ROOT / "00_protocol"
    target = tmp_path / "00_protocol"
    shutil.copytree(source, target)
    cohort = target / "cohort.json"
    cohort.write_text(cohort.read_text(encoding="utf-8").replace('"record_count": 23', '"record_count": 22', 1), encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid artifact"):
        validate_stage00(tmp_path)


@pytest.mark.parametrize(
    ("relative", "artifact", "validator"),
    [
        ("01_candidates", "audio_manifest.json", validate_stage01),
        ("02_audio_diagnostics", "diagnostics.json", validate_stage02),
        ("03_videos", "videos_manifest.json", validate_stage03),
        ("04_scores", "scores_manifest.json", validate_stage04),
        ("05_final", "analysis.json", validate_stage05),
    ],
)
def test_each_downstream_validator_rejects_corrupted_artifact(
    tmp_path: Path,
    relative: str,
    artifact: str,
    validator: object,
) -> None:
    source = config.RUN_ROOT / relative
    target = tmp_path / relative
    shutil.copytree(source, target)
    path = target / artifact
    path.write_text(path.read_text(encoding="utf-8").replace('"status": "complete"', '"status": "corrupt"', 1), encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid artifact"):
        validator(tmp_path)
