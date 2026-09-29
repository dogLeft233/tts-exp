from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/experiments"))

import tts_visual_timing_metrics as metrics  # noqa: E402
from tts_visual_timing import _expected_event_time, _half_up, build_audio_knots  # noqa: E402
from tts_visual_timing_fixtures import SYNTHETIC_LABEL, run_synthetic_fixtures  # noqa: E402


def _tokens(word_index: int, word: str, phones: tuple[str, ...], start: float) -> list[dict[str, object]]:
    result = []
    for index, phone in enumerate(phones):
        left = start + index * 0.1
        result.append({"index": index, "label": phone, "normalized": phone, "start_s": left, "end_s": left + 0.1, "silence": False, "unknown": False, "word_index": word_index, "word_normalized": word, "word_label": word})
    return result


def test_event_distance_identity_shift_and_empty_penalty():
    identity = metrics.event_distance([{"time_s": 0.5}], [{"time_s": 0.5}])
    assert identity["distance_ms"] == pytest.approx(0.0)
    shifted = metrics.event_distance([{"time_s": 0.5}], [{"time_s": 0.58}])
    assert shifted["distance_ms"] == pytest.approx(80.0)
    missing = metrics.event_distance([{"time_s": 0.5}], [])
    assert missing["distance_ms"] == pytest.approx(240.0)
    with pytest.raises(ValueError):
        metrics.event_distance([], [])


def test_local_warp_has_endpoints_and_correct_direction():
    mapping, meta = metrics.local_warp_indices(250, 25.0, 0.08)
    assert mapping[0] == 0 and mapping[-1] == 249
    assert all(right >= left for left, right in zip(mapping, mapping[1:]))
    # Positive delta reads a future source frame, so an interior event appears earlier.
    assert _expected_event_time(100, mapping, 25.0) < 100 / 25.0
    assert meta["positive_reads_future"] is True


def test_v2_local_warp_uses_fixed_wider_plateau_without_changing_endpoints():
    mapping, meta = metrics.local_warp_indices(250, 25.0, 0.16, protocol="tts_visual_timing_v2")
    assert mapping[0] == 0 and mapping[-1] == 249
    assert meta["protocol"] == "tts_visual_timing_v2"
    # The fixed v2 plateau is four frames at 160 ms, with no clipping.
    assert max(mapping[index] - index for index in range(50, 200)) == 4
    assert min(mapping[index] - index for index in range(50, 200)) >= 0


def test_continuous_support_blocks_keep_phone_slopes_and_split_real_gaps():
    segments = [
        {"n_start_s": 0.0, "n_end_s": 0.1, "t_start_s": 0.0, "t_end_s": 0.2, "slope": 2.0},
        {"n_start_s": 0.1, "n_end_s": 0.2, "t_start_s": 0.2, "t_end_s": 0.25, "slope": 0.5},
        {"n_start_s": 0.3, "n_end_s": 0.4, "t_start_s": 0.35, "t_end_s": 0.45, "slope": 1.0},
    ]
    blocks = metrics.continuous_support_blocks(segments)
    assert len(blocks) == 2
    assert blocks[0]["segment_indices"] == [0, 1]
    assert blocks[1]["segment_indices"] == [2]
    assert segments[0]["slope"] != segments[1]["slope"]


def test_audio_mapping_ignores_silence_without_dropping_unlabelled_speech():
    n = [{"label": "sil", "silence": True, "unknown": False, "start_s": 0.0, "end_s": 0.1, "word_index": None, "word_normalized": None}]
    n += _tokens(0, "go", ("g", "ow"), 0.1)
    t = [{"label": "sil", "silence": True, "unknown": False, "start_s": 0.0, "end_s": 0.2, "word_index": None, "word_normalized": None}]
    t += _tokens(0, "go", ("g", "ow"), 0.2)
    assert metrics.audio_time_map(n, t)["status"] == "COMPLETE"
    broken = [dict(item, word_index=None, word_normalized=None) for item in n[1:]]
    assert metrics.audio_time_map(broken, t)["status"] == "ALIGNMENT_UNSUPPORTED"


def test_calibration_direction_uses_observed_shift_and_rejects_missing_inverse():
    reference = [{"index": 50, "event_position_frames": 50.0, "time_s": 2.0}, {"index": 75, "event_position_frames": 75.0, "time_s": 3.0}]
    identity = list(range(100))
    future = [min(99, index + 2) for index in range(100)]
    result = metrics.evaluate_calibration_record(
        reference,
        {
            "REPEAT": {"events": reference, "mapping": identity, "status": "MEASURABLE", "delta_s": 0.0},
            "LOCAL_+80": {"events": [{"index": 48, "event_position_frames": 48.0, "time_s": 1.92}, {"index": 73, "event_position_frames": 73.0, "time_s": 2.92}], "mapping": future, "status": "MEASURABLE", "delta_s": 0.08},
            "LOCAL_-80": {"events": reference, "mapping": identity, "status": "MEASURABLE", "delta_s": -0.08},
            "LOCAL_+160": {"events": [{"index": 46, "event_position_frames": 46.0, "time_s": 1.84}, {"index": 71, "event_position_frames": 71.0, "time_s": 2.84}], "mapping": [min(99, index + 4) for index in range(100)], "status": "MEASURABLE", "delta_s": 0.16},
            "LOCAL_-160": {"events": reference, "mapping": identity, "status": "MEASURABLE", "delta_s": -0.16},
            "FROZEN": {"events": [], "mapping": [50] * 100, "status": "LOW_MOTION", "delta_s": None},
        },
        source_interval=(0.0, 10.0),
    )
    assert result["arms"]["LOCAL_+80"]["direction_ok"] is True
    assert result["arms"]["LOCAL_+80"]["expected_events"]


def test_pixel_coordinate_normalization_is_invariant_to_resolution():
    pixels = np.zeros((20, 478, 2), dtype=np.float64)
    for index in range(20):
        pixels[index, 33] = [80.0, 40.0]
        pixels[index, 263] = [240.0, 40.0]
        pixels[index, 13] = [150.0, 98.0 if index in (8, 9, 10) else 100.0]
        pixels[index, 14] = [150.0, 102.0 if index in (8, 9, 10) else 100.0]
        pixels[index, 61] = [100.0, 90.0]
        pixels[index, 291] = [220.0, 90.0]
    normalized = pixels.copy()
    normalized[:, :, 0] /= 320.0
    normalized[:, :, 1] /= 160.0
    valid = np.ones(20, dtype=bool)
    times = np.arange(20, dtype=np.float64) / 25.0
    pixel_result = metrics.aperture_events(pixels, valid, times, width=320, height=160, coordinate_system="pixel")
    normalized_result = metrics.aperture_events(normalized, valid, times, width=320, height=160)
    assert normalized_result["status"] == pixel_result["status"]
    assert normalized_result["events"] == pixel_result["events"]


def test_invalid_required_point_is_not_a_closed_mouth():
    landmarks = np.zeros((20, 478, 2), dtype=np.float64)
    landmarks[:, 33] = [0.1, 0.2]
    landmarks[:, 263] = [0.9, 0.2]
    landmarks[:, 13] = [0.5, 0.5]
    landmarks[:, 14] = [0.5, 0.5]
    landmarks[:, 61] = [0.3, 0.5]
    landmarks[:, 291] = [0.7, 0.5]
    landmarks[4, 13] = np.nan
    result = metrics.aperture_events(landmarks, np.ones(20, dtype=bool), np.arange(20) / 25.0, width=100, height=100)
    assert result["status"] in ("LOW_MOTION", "MEASURABLE")
    assert result["valid_fraction"] == pytest.approx(19 / 20)
    assert all(int(item["index"]) != 4 for item in result["events"])


def test_audio_mapping_requires_word_ordinal_and_does_not_cross_repeated_words():
    n = _tokens(0, "go", ("g", "ow", "silx"), 0.0) + _tokens(1, "go", ("g", "ow"), 0.4)
    t = _tokens(0, "go", ("g", "ow", "silx"), 0.0) + _tokens(1, "go", ("g",), 0.4)
    result = metrics.audio_time_map(n, t)
    assert result["status"] == "COMPLETE"
    assert result["matched_word_count"] == 2
    assert result["matched_phone_count"] == 4
    missing_words = [dict(item, word_index=None, word_normalized=None) for item in n]
    assert metrics.audio_time_map(missing_words, t)["status"] == "ALIGNMENT_UNSUPPORTED"


def test_half_up_and_local_mapping_do_not_silently_clamp():
    assert _half_up(1.5) == 2
    assert _half_up(1.499999) == 1
    mapping, _ = metrics.local_warp_indices(50, 25.0, -0.16)
    assert min(mapping) >= 0 and max(mapping) < 50


def test_audio_knots_include_full_duration_endpoints_and_half_up():
    n = _tokens(0, "go", ("g", "ow"), 0.0)
    t = _tokens(0, "go", ("g", "ow"), 0.0)
    result = build_audio_knots(n, t, n_duration_samples=16000, t_duration_samples=17600)
    assert result["status"] == "COMPLETE"
    assert result["source_knots"][0] == 0 and result["target_knots"][0] == 0
    assert result["source_knots"][-1] == 17600 and result["target_knots"][-1] == 16000


def test_bootstrap_indices_are_saved_and_reproducible():
    left = metrics.bootstrap_mean({"b": 2.0, "a": 1.0}, draws=32, seed=7, alpha=0.05)
    right = metrics.bootstrap_mean({"a": 1.0, "b": 2.0}, draws=32, seed=7, alpha=0.05)
    assert left["draw_indices"] == right["draw_indices"]
    assert left["ci"] == right["ci"]


def test_unentered_gate_branches_are_labeled_synthetic():
    cases = run_synthetic_fixtures()
    assert set(cases) == {"A_calibration_failed", "B_native_no_support", "Wav2Lip_native_unconfirmed", "identity_not_equivalent", "C_all_gates_pass"}
    assert all(item["label"] == SYNTHETIC_LABEL for item in cases.values())
    assert all(item["actual"] == item["expected"] for item in cases.values())
