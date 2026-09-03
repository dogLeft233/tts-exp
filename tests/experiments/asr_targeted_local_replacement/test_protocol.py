from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.experiments.asr_targeted_local_replacement import protocol


ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "runs/lrs3_mfa_linear_replacement_20260824/00_protocol_lock_retry1/manifest.json"
PARENT_LOCK = PARENT.with_name("test_lock.json")


def test_frozen_inputs_are_24_fit_only_pairs():
    frozen = protocol.load_frozen_inputs(ROOT)
    assert len(frozen["samples"]) == 24
    assert len(frozen["asr_records"]) == 24
    assert frozen["sealed_splits_accessed"] is False
    assert {sample["split"] for sample in frozen["samples"]} == {"fit"}
    assert len({sample["source_group"] for sample in frozen["samples"]}) == 24


def test_parent_sealed_access_is_rejected():
    parent = json.loads(PARENT.read_text())
    lock = json.loads(PARENT_LOCK.read_text())
    changed = copy.deepcopy(parent)
    changed["media_access"]["test_media_opened"] = True
    with pytest.raises(ValueError, match="sealed splits"):
        protocol._validate_parent_lock(ROOT, changed, lock)


def test_tts_insertion_inside_donor_interval_is_not_clean():
    operations = [
        {"operation": "equal", "reference_index": 0},
        {"operation": "insertion", "reference_index": None},
        {"operation": "equal", "reference_index": 1},
    ]
    assert protocol._tts_interval_clean(operations, [0, 1]) is False


def test_score_fields_are_rejected_from_selection_projection():
    with pytest.raises(ValueError, match="score-bearing"):
        protocol.assert_score_blind_fields(["sample_id", "sync_c"])
