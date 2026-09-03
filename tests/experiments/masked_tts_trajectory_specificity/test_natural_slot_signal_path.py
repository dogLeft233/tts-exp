import numpy as np

from scripts.experiments.masked_tts_reconstruction.config import SEEDS, WAVLM_DIM
from scripts.experiments.masked_tts_reconstruction.features import phone_phase_linear
from scripts.experiments.masked_tts_trajectory_specificity.natural_slot_signal_path import (
    _natural_span,
    _neighbor_mismatch,
    _aggregate_named,
    final_decision,
)


def _records():
    return [
        {"sample_id": f"g{group}-r{record}", "source_group": f"g{group}"}
        for group in range(8)
        for record in range(2)
    ]


def test_mask_level_aggregation_keeps_same_seed_masks_separate():
    rows = []
    for group in range(8):
        for record in range(2):
            sid = f"g{group}-r{record}"
            for seed in SEEDS:
                rows.extend([
                    {"sample_id": sid, "source_group": f"g{group}", "seed": int(seed), "mask_sha256": "a" * 64, "gain": 1.0},
                    {"sample_id": sid, "source_group": f"g{group}", "seed": int(seed), "mask_sha256": "b" * 64, "gain": 3.0},
                ])
    result = _aggregate_named(_records(), rows, "gain", mask_level=True)
    assert result["summary"]["median"] == 2.0
    assert result["summary"]["positive_groups"] == 8


def test_natural_span_uses_bound_alignment_and_clips_frames():
    mask = {"sample_id": "s", "natural_phone_index": 0, "label": "aa"}
    alignment = {"records": [{"sample_id": "s", "natural_phones": [{"phone": "aa", "start": -0.1, "end": 0.1}]}]}
    assert _natural_span(mask, 10, alignment)[:2] == (0, 5)


def test_natural_slot_is_core_only_and_shape_matches():
    values = np.arange(8 * WAVLM_DIM, dtype=np.float32).reshape(8, WAVLM_DIM)
    aligned, _ = phone_phase_linear(values, 2, 6, 4)
    slot = np.zeros((96, WAVLM_DIM), dtype=np.float32)
    slot[20:24] = aligned
    assert slot.shape == (96, WAVLM_DIM)
    assert np.count_nonzero(slot[:20]) == 0
    assert np.count_nonzero(slot[24:]) == 0


def test_neighbor_mismatch_is_deterministic():
    alignment = {
        "s": {"tts_phones": [{"phone": "aa"}, {"phone": "bb"}, {"phone": "cc"}]},
        "t": {"tts_phones": [{"phone": "aa"}, {"phone": "bb"}, {"phone": "dd"}]},
    }
    target = {"sample_id": "s", "tts_phone_index": 1}
    candidate = {"sample_id": "t", "tts_phone_index": 1}
    assert _neighbor_mismatch(target, candidate, alignment) == 1


def test_final_decision_keeps_waveform_gate_closed():
    result = final_decision(
        None,
        {"status": "complete"},
        {"status": "complete"},
        {"status": "complete", "contrasts": {}},
        {"status": "INSUFFICIENT_MATCHED_DONOR_COVERAGE"},
    )
    assert result["recommendation"] == "REDESIGN_CONDITIONING_OBJECTIVE_BEFORE_MORE_DATA"
    assert result["waveform_decoder_gate"] == "CLOSED"
