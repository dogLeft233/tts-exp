from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.eval_aishell1_n25_wav2lip_syncnet import (
    ARMS,
    analyze,
    cluster_bootstrap,
    validate_records,
)

REPO = Path(__file__).resolve().parents[1]
COHORT = REPO / "runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json"


def synthetic_records() -> list[dict[str, object]]:
    rows = []
    for speaker_index, speaker in enumerate(("S0765", "S0901", "S0906", "S0912", "S0913")):
        for utterance_index in range(5):
            key = f"{speaker}:pair{utterance_index}"
            for arm in ARMS:
                rows.append({
                    "sample_id": f"{speaker_index}_{utterance_index}",
                    "paired_key": key,
                    "speaker_id": speaker,
                    "split": "valid" if speaker == "S0765" else "train",
                    "transcript": "测试",
                    "arm": arm,
                    "face_sha256": "face",
                })
    return rows


def complete_summary() -> dict[str, object]:
    scores = []
    for row in synthetic_records():
        arm = str(row["arm"])
        base = {"natural_raw": 5.0, "raw_tts": 6.0, "mfa_linear": 7.0}[arm]
        scores.append({**row, "sync_c": base, "sync_d": 8.0 - (base - 5.0), "av_offset": -2})
    return {"status": "complete", "failures": [], "scores": scores}


def test_synthetic_records_are_exactly_five_by_five() -> None:
    rows = synthetic_records()
    validate_records(rows)
    assert len(rows) == 75


def test_validate_records_rejects_duplicate_composite_identity() -> None:
    rows = synthetic_records()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="75 records"):
        validate_records(rows)


def test_validate_records_rejects_heldout_speaker() -> None:
    rows = synthetic_records()
    rows[0]["speaker_id"] = "S0770"
    with pytest.raises(ValueError, match="heldout"):
        validate_records(rows)


def test_cluster_bootstrap_is_deterministic() -> None:
    values = {speaker: [float(index) for index in range(5)] for speaker in ("S0765", "S0901", "S0906", "S0912", "S0913")}
    assert cluster_bootstrap(values, 42) == cluster_bootstrap(values, 42)


def test_analysis_reports_speaker_summary_and_scope() -> None:
    result = analyze(complete_summary())
    assert result["analysis"] == "aishell1_mfa_linear_n25_clustered_exploratory"
    assert "5 speakers x 5" in result["scope"]
    comparison = result["comparisons"]["mfa_linear_vs_natural_raw"]
    assert comparison["n"] == 25
    assert comparison["speaker_count"] == 5
    assert set(comparison["speaker_summary"]) == {"S0765", "S0901", "S0906", "S0912", "S0913"}


def test_checked_in_cohort_has_expected_size() -> None:
    if not COHORT.is_file():
        pytest.skip("generated cohort is not available")
    payload = json.loads(COHORT.read_text(encoding="utf-8"))
    assert payload["counts"]["selected_pairs"] == 25
