import numpy as np
import pytest

from scripts.experiments.masked_tts_trajectory_specificity.run import (
    _aggregate_contrast,
    _cohort_records,
    _eligible_donors,
    _reverse_inside_core,
    _select_donor,
    final_decision,
)
from scripts.experiments.masked_tts_reconstruction.config import SEEDS, WAVLM_DIM


def _mask(label: str, sample_id: str, group: str, canonical: int, mask_hash: str, start: int = 2, end: int = 6) -> dict:
    return {
        "label": label,
        "sample_id": sample_id,
        "source_group": group,
        "canonical_index": canonical,
        "mask_sha256": mask_hash,
        "tts_frame_start": start,
        "tts_frame_end": end,
    }


def test_same_phone_donors_are_sorted_and_identity_filtered() -> None:
    target = _mask("aa", "target", "target-group", 0, "0f" * 32)
    candidates = [
        _mask("aa", "z", "g2", 2, "22" * 32),
        _mask("aa", "a", "g1", 4, "11" * 32),
        _mask("bb", "b", "g0", 1, "33" * 32),
        _mask("aa", "target", "g3", 1, "44" * 32),
        _mask("aa", "same-group", "target-group", 1, "55" * 32),
        _mask("aa", "short", "g0", 1, "66" * 32, 0, 1),
    ]
    tts = {row["sample_id"]: np.zeros((8, WAVLM_DIM), dtype=np.float32) for row in candidates}
    eligible = _eligible_donors(target, candidates, tts)
    assert [(row["source_group"], row["sample_id"]) for row in eligible] == [("g1", "a"), ("g2", "z")]
    donor, index = _select_donor(target, eligible)
    assert index == int("0f" * 8, 16) % 2
    assert donor["sample_id"] == eligible[index]["sample_id"]
    assert _select_donor(target, eligible) == (donor, index)


def test_reversal_changes_only_the_target_core() -> None:
    values = np.zeros((96, WAVLM_DIM), dtype=np.float32)
    values[3] = 1.0
    values[4] = 2.0
    values[5] = 3.0
    reversed_values = _reverse_inside_core(values, 3, 6)
    assert np.array_equal(reversed_values[3:6, 0], np.array([3.0, 2.0, 1.0], dtype=np.float32))
    assert np.count_nonzero(reversed_values[:3]) == 0
    assert np.count_nonzero(reversed_values[6:]) == 0
    with pytest.raises(ValueError):
        _reverse_inside_core(np.ones_like(values), 3, 6)


def test_aggregation_uses_positive_gain_directions() -> None:
    records = []
    rows = []
    paired = {}
    for group_index in range(8):
        group = f"g{group_index}"
        for record_index in range(2):
            sid = f"{group}-r{record_index}"
            records.append({"sample_id": sid, "source_group": group})
            for seed in SEEDS:
                key = (sid, int(seed))
                paired[key] = {"sync_c": 4.0, "sync_d": 8.0}
                rows.append({"sample_id": sid, "seed": int(seed), "sync_c": 5.0, "sync_d": 7.0})
    c = _aggregate_contrast(records, rows, paired, lambda row, ref: row["sync_c"] - ref["sync_c"], "sync_c_gain")
    d = _aggregate_contrast(records, rows, paired, lambda row, ref: ref["sync_d"] - row["sync_d"], "sync_d_gain")
    assert c["summary"]["median"] == 1.0
    assert d["summary"]["median"] == 1.0
    assert c["summary"]["pass"] is True
    assert d["summary"]["pass"] is True


def test_incomplete_cohort_is_rejected() -> None:
    with pytest.raises(ValueError):
        _cohort_records({"status": "complete", "records": [], "groups": []})


def test_missing_donor_is_not_selectable() -> None:
    with pytest.raises(ValueError):
        _select_donor(_mask("aa", "target", "g0", 0, "0f" * 32), [])


    part_a = {
        "engineering_audit": "PAIRED_BEATS_NATURAL_REFERENCE",
        "bootstrap": {
            "centroid_modality_C": {"pass": True},
            "centroid_modality_D": {"pass": True},
        },
    }
    assert final_decision(None, part_a, {"trajectory_status": "NO_TRAJECTORY_SIGNAL"}, "PENDING_CONDITIONAL_TRAINING")["recommendation"] == "USE_SIMPLER_PHONE_CONTROL_PATH"
    part_a["bootstrap"]["centroid_modality_C"]["pass"] = False
    part_a["bootstrap"]["centroid_modality_D"]["pass"] = False
    assert final_decision(None, part_a, {"trajectory_status": "NO_TRAJECTORY_SIGNAL"}, "PENDING_CONDITIONAL_TRAINING")["recommendation"] == "RETAIN_PAIRED_MODALITY_PATH"
    assert final_decision(None, part_a, {"trajectory_status": "TFG_TRAJECTORY_SIGNAL"}, "SKIPPED_ALREADY_SENSITIVE")["recommendation"] == "CONFIRM_TRAJECTORY_MODEL_ON_NEW_RECORDS"
    assert final_decision(None, part_a, {"trajectory_status": "NO_TRAJECTORY_SIGNAL"}, "NOT_EVALUATED")["recommendation"] == "STOP_INVALID_EXPERIMENT"
