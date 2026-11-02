from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/experiments/world_phone_transfer_20261001.py"
SPEC = importlib.util.spec_from_file_location("world_phone_transfer_20261001_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
world = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = world
SPEC.loader.exec_module(world)


def _phone(index: int, label: str, start: float, end: float) -> dict:
    return {
        "index": index,
        "raw_label": label,
        "normalized_label": world.normalize_phone(label),
        "start_s": start,
        "end_s": end,
        "in_frozen_inventory": True,
        "word_index": 0,
        "raw_start_s": start,
        "raw_end_s": end,
        "clamp_start_s": 0.0,
        "clamp_end_s": 0.0,
    }


def _grid(phones: list[dict], words: list[dict]) -> dict:
    return {
        "words": words,
        "phones": phones,
        "duration_s": 1.0,
        "word_tier_name": "words",
        "phone_tier_name": "phones",
    }


def test_physical_phone_normalizer_only_applies_nfc_and_outer_strip():
    assert world.normalize_phone("  a\u0308ːʰ  ") == "äːʰ"
    assert world.normalize_phone("Aː") == "Aː"


def test_lcs_uses_recipient_indices_then_donor_indices_for_ties():
    recipient = [
        {"index": 0, "normalized_label": "a"},
        {"index": 1, "normalized_label": "b"},
        {"index": 2, "normalized_label": "a"},
    ]
    donor = [
        {"index": 0, "normalized_label": "b"},
        {"index": 1, "normalized_label": "a"},
        {"index": 2, "normalized_label": "b"},
    ]
    assert world.deterministic_lcs(recipient, donor) == [(0, 1), (1, 2)]


def test_map_requires_identical_words_and_preserves_unmatched_silence():
    word = {"index": 1, "raw_label": "你", "normalized_label": "你", "start_s": 0.1, "end_s": 0.9}
    rec_phones = [_phone(0, "sil", 0.0, 0.1), _phone(1, "n", 0.1, 0.4), _phone(2, "i˨˩˦", 0.4, 0.9)]
    don_phones = [_phone(0, "sil", 0.0, 0.1), _phone(1, "n", 0.1, 0.5), _phone(2, "i˨˩˦", 0.5, 0.9)]
    # The leading silence is deliberately outside the nonempty word interval.
    rec = _grid(rec_phones, [word])
    don = _grid(don_phones, [word])
    result = world.map_phone_intervals(rec, don)
    assert result["matched_phone_count"] == 2
    assert result["unmatched_recipient_phones"][0]["reason"] == "silence_preserved"
    assert result["recipient_audio_duration_s"] == 1.0


def test_nearest_center_ties_choose_earlier_frame():
    assert world.nearest_index(np.asarray([0.0, 1.0]), 0.5) == 0


def test_rt_identity_envelope_match_stays_within_contract():
    x = np.sin(np.arange(6400, dtype=np.float32) * np.float32(0.031))
    x = (x * (world.FIXED_RMS / np.sqrt(np.mean(x.astype(np.float64) ** 2)))).astype(np.float32)
    times, envelope = world.local_rms_envelope(x, np)
    matched, audit, arrays = world.local_envelope_match(x, envelope, times, np)
    assert matched.dtype == np.float32
    assert np.max(np.abs(matched - x)) <= 1e-7
    assert audit["relative_rms_change_from_input"] <= 1e-6
    assert arrays["local_gain_db"].shape == x.shape


def test_parameter_transfer_maps_ap_and_preserves_sp_power():
    recipient = {
        "f0": np.asarray([120.0, 0.0]),
        "time": np.asarray([0.025, 0.075]),
        "sp": np.asarray([[1.0, 3.0], [4.0, 4.0]]),
        "ap": np.asarray([[0.1, 0.3], [0.2, 0.4]]),
        "frequency_hz": np.asarray([0.0, 8000.0]),
        "fft_size": 2,
    }
    donor = {
        "f0": np.asarray([220.0, 240.0]),
        "time": np.asarray([0.025, 0.075]),
        "sp": np.asarray([[3.0, 1.0], [1.0, 3.0]]),
        "ap": np.asarray([[0.8, 0.6], [0.4, 0.2]]),
        "frequency_hz": np.asarray([0.0, 8000.0]),
        "fft_size": 2,
    }
    mapping = {
        "matched_phones": [{
            "recipient_interval_s": [0.0, 0.1],
            "donor_interval_s": [0.0, 0.1],
            "recipient_phone_index": 0,
        }],
        "recipient_audio_duration_s": 0.1,
        "donor_audio_duration_s": 0.1,
    }

    arms, audit = world.transfer_parameters(recipient, donor, mapping, np)

    np.testing.assert_allclose(audit["detail"]["mapped_ap"], donor["ap"])
    np.testing.assert_allclose(arms["EX"]["f0"], donor["f0"])
    np.testing.assert_allclose(arms["F0"]["f0"], [220.0, 0.0])
    np.testing.assert_allclose(arms["AP"]["ap"], [[0.8, 0.6], [0.2, 0.4]])
    np.testing.assert_allclose(np.sum(arms["SP"]["sp"], axis=1), np.sum(recipient["sp"], axis=1))


def test_grid_pair_reads_decoded_counts_from_audio_metadata(tmp_path):
    grid_path = tmp_path / "pair.TextGrid"
    grid_path.write_text("\n".join([
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", "xmax = 1",
        "tiers? <exists>", "size = 2", "item []:", "    item [1]:", '        class = "IntervalTier"',
        '        name = "words"', "        xmin = 0", "        xmax = 1", "        intervals: size = 1",
        "        intervals [1]:", "            xmin = 0", "            xmax = 1", '            text = "你"',
        "    item [2]:", '        class = "IntervalTier"', '        name = "phones"', "        xmin = 0",
        "        xmax = 1", "        intervals: size = 2", "        intervals [1]:", "            xmin = 0",
        "            xmax = 0.45", '            text = "n"', "        intervals [2]:", "            xmin = 0.45",
        "            xmax = 1", '            text = "i˨˩˦"',
    ]), encoding="utf-8")
    record = {"conditions": {
        "natural": {"normalized_lab_text": "你", "expected_textgrid_path": str(grid_path)},
        "tts": {"normalized_lab_text": "你", "expected_textgrid_path": str(grid_path)},
    }}
    input_info = {"natural": {"decoded_samples": 16000}, "tts": {"decoded_samples": 32000}}
    phone_table = {"raw_to_normalized": {"n": "n", "i˨˩˦": "i˨˩˦"}}

    natural, tts = world._grid_pair(record, input_info, None, phone_table, "fixture")

    assert (natural["audio_samples"], natural["duration_s"]) == (16000, 1.0)
    assert (tts["audio_samples"], tts["duration_s"]) == (32000, 2.0)
    assert [item["normalized_label"] for item in natural["phones"]] == ["n", "i˨˩˦"]


def test_direction_names_match_donor_and_recipient_conditions():
    natural_grid, tts_grid = {"side": "N"}, {"side": "T"}
    specs = world._direction_specs(natural_grid, tts_grid)
    by_name = {name: (recipient, donor, rec_grid, don_grid) for name, recipient, donor, rec_grid, don_grid in specs}

    assert by_name["tts_to_natural"] == ("natural", "tts", natural_grid, tts_grid)
    assert by_name["natural_to_tts"] == ("tts", "natural", tts_grid, natural_grid)
