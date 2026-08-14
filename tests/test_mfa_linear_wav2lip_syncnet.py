from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts/eval_mfa_linear_wav2lip_syncnet.py"
SPEC = importlib.util.spec_from_file_location("eval_mfa_linear", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _complete_summary() -> dict:
    scores = []
    for sample_id in range(1, 16):
        for arm, c, d in (
            ("natural_raw", 5.0, 8.0),
            ("raw_tts", 6.0, 7.0),
            ("mfa_linear", 5.5, 7.5),
        ):
            scores.append({
                "sample_id": sample_id, "paired_key": f"pair-{sample_id}", "arm": arm,
                "sync_c": c + sample_id / 100, "sync_d": d - sample_id / 100,
                "av_offset": -2,
            })
    return {"status": "complete", "failures": [], "scores": scores}


def test_analysis_has_correct_paired_directions_and_counts() -> None:
    result = MODULE.analyze(_complete_summary())
    natural = result["comparisons"]["mfa_linear_vs_natural_raw"]
    tts = result["comparisons"]["mfa_linear_vs_raw_tts"]
    assert natural["mean_delta_sync_c"] == pytest.approx(0.5)
    assert natural["mean_delta_sync_d"] == pytest.approx(-0.5)
    assert natural["sync_c_better_count"] == 15
    assert natural["sync_d_better_count"] == 15
    assert natural["joint_better_count"] == 15
    assert tts["mean_delta_sync_c"] == pytest.approx(-0.5)
    assert tts["mean_delta_sync_d"] == pytest.approx(0.5)
    assert tts["joint_better_count"] == 0


def test_analysis_bootstrap_is_deterministic() -> None:
    first = MODULE.analyze(_complete_summary())
    second = MODULE.analyze(_complete_summary())
    assert first["comparisons"]["mfa_linear_vs_natural_raw"]["bootstrap_95_ci_delta_sync_c"] == second["comparisons"]["mfa_linear_vs_natural_raw"]["bootstrap_95_ci_delta_sync_c"]


def test_analysis_rejects_incomplete_matrix() -> None:
    summary = _complete_summary()
    summary["scores"].pop()
    with pytest.raises(ValueError, match="45-score"):
        MODULE.analyze(summary)


def test_record_validator_rejects_missing_arm() -> None:
    records = []
    for sample_id in range(1, 16):
        for arm in MODULE.ARMS:
            records.append({
                "sample_id": sample_id, "paired_key": f"pair-{sample_id}", "arm": arm,
                "speaker_id": "S0765", "split": "valid", "face_sha256": "a" * 64,
            })
    records.pop()
    with pytest.raises(ValueError, match="45"):
        MODULE._validate_records(records)
