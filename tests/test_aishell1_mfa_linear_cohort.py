from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.prepare_aishell1_mfa_linear_cohort import (
    COUNT,
    HELDOUT_SPEAKERS,
    PER_SPEAKER,
    SPEAKERS,
    canonical_sha256,
    select_cohort,
    validate_cohort,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json"
COHORT = REPO / "runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json"


def test_frozen_cohort_is_five_by_five_and_excludes_heldout() -> None:
    cohort = select_cohort(SOURCE)
    validate_cohort(cohort)
    assert len(cohort["records"]) == COUNT == 25
    assert cohort["cohort"]["speakers"] == list(SPEAKERS)
    assert cohort["cohort"]["per_speaker"] == PER_SPEAKER
    assert cohort["cohort"]["heldout_speakers"] == list(HELDOUT_SPEAKERS)
    assert cohort["cohort"]["excluded_speakers"] == ["S0770", "S0914", "S0915"]
    assert cohort["selection"]["speaker_counts"] == {speaker: 5 for speaker in SPEAKERS}
    assert all(row["speaker_id"] not in HELDOUT_SPEAKERS for row in cohort["records"])
    assert all(row["speaker_id"] in SPEAKERS for row in cohort["records"])


def test_selection_is_deterministic_and_hash_matches() -> None:
    first = select_cohort(SOURCE)
    second = select_cohort(SOURCE)
    assert first["selection"] == second["selection"]
    keys = [row["paired_key"] for row in first["records"]]
    assert canonical_sha256(keys) == first["selection"]["ordered_paired_keys_sha256"]
    for speaker in SPEAKERS:
        keys_for_speaker = [row["paired_key"] for row in first["records"] if row["speaker_id"] == speaker]
        assert keys_for_speaker == sorted(keys_for_speaker)


def test_selector_rejects_duplicate_condition(tmp_path: Path) -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(next(row for row in payload["records"] if row["speaker_id"] == "S0765" and row["condition"] == "natural"))
    payload["records"].append(duplicate)
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate natural"):
        select_cohort(source)


def test_validator_rejects_heldout_record() -> None:
    cohort = select_cohort(SOURCE)
    bad = copy.deepcopy(cohort)
    bad["records"][0]["speaker_id"] = "S0770"
    with pytest.raises(ValueError, match="forbidden speaker"):
        validate_cohort(bad)


def test_checked_in_cohort_matches_selector() -> None:
    if not COHORT.is_file():
        pytest.skip("generated cohort is not available")
    checked = json.loads(COHORT.read_text(encoding="utf-8"))
    assert checked == select_cohort(SOURCE)
