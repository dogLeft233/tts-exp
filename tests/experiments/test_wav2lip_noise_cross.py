from __future__ import annotations

import numpy as np

from scripts.experiments.wav2lip_noise_cross_metrics import (
    bootstrap_group_summary,
    common_support,
    construct_audio_conditions,
    curve_metrics,
    delayed_audio,
    four_cell,
    matrix_from_embeddings,
)


def _audio(length: int, amplitude: float = 0.2) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / 16_000.0
    return np.rint(amplitude * np.sin(2 * np.pi * 220.0 * time) * 32767.0).astype(np.int16)


def test_audio_conditions_keep_two_clocks_and_target_snr() -> None:
    natural = _audio(32_000, 0.2)
    tts = _audio(33_000, 0.3)
    result = construct_audio_conditions(natural, tts, 151)
    assert result["conditions"]["N"]["raw"].tolist() == natural.tolist()
    assert result["conditions"]["T"]["raw"].tolist() == tts.tolist()
    assert result["metadata"]["g"] <= 1.0
    for source, expected in (("N", natural.size), ("T", tts.size)):
        item = result["conditions"][source]
        assert item["a0"].size == expected
        assert item["noise20"].size == expected
        assert 19.8 <= item["actual_snr_db"] <= 20.2


def test_common_headroom_scales_a0_but_does_not_change_raw() -> None:
    natural = np.full(20_000, 30_000, dtype=np.int16)
    tts = np.full(21_000, 32_000, dtype=np.int16)
    result = construct_audio_conditions(natural, tts, 152)
    assert result["metadata"]["g"] < 1.0
    assert np.array_equal(result["conditions"]["N"]["raw"], natural)
    assert not np.array_equal(result["conditions"]["N"]["a0"], natural)


def test_four_cell_and_time_first_curve() -> None:
    assert four_cell(5.0, 4.0, 4.5, 3.0)["evaluation"] == -1.0
    assert four_cell(5.0, 4.0, 4.5, 3.0)["generation"] == -0.5
    assert four_cell(5.0, 4.0, 4.5, 3.0)["interaction"] == -0.5
    matrix = np.full((30, 31), 2.0, dtype=np.float32)
    matrix[:, 15] = 1.0
    matrix[0, 0] = -10.0  # row-wise min would choose the wrong lag
    metrics = curve_metrics(matrix, list(range(1, 29)))
    assert metrics["min_index"] == 15


def test_common_support_uses_shortest_matrix_without_padding() -> None:
    matrices = [np.zeros((100, 31)), np.zeros((90, 31)), np.zeros((95, 31))]
    rows, status = common_support(matrices)
    assert status == "PASS"
    assert rows[0] == 15 and rows[-1] == 74 and len(rows) == 60
    short, status = common_support([np.zeros((50, 31))] * 5)
    assert short == list(range(15, 35))
    assert status == "INSUFFICIENT_SUPPORT"


def test_delay_preserves_sample_count_and_shifts_content() -> None:
    value = np.arange(8_000, dtype=np.int16)
    shifted = delayed_audio(value, 200)
    assert shifted.size == value.size
    assert np.all(shifted[:3_200] == 0)
    assert np.array_equal(shifted[3_200:], value[:-3_200])


def test_bootstrap_is_source_group_weighted_and_deterministic() -> None:
    rows = [
        {"source_group": "a", "value": 1.0},
        {"source_group": "a", "value": 3.0},
        {"source_group": "b", "value": 5.0},
    ]
    first = bootstrap_group_summary(rows, "value", draws=128)
    second = bootstrap_group_summary(rows, "value", draws=128)
    assert first == second
    assert first["mean"] == 3.5
    assert first["group_count"] == 2


def test_matrix_reference_shape_and_finite_values() -> None:
    visual = np.ones((40, 1024), dtype=np.float32)
    audio = np.ones((40, 1024), dtype=np.float32)
    matrix = matrix_from_embeddings(visual, audio)
    assert matrix.shape == (40, 31)
    assert np.isfinite(matrix).all()
