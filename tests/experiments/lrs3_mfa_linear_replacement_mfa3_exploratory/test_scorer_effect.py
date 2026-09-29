from __future__ import annotations

import pytest

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.scorer_effect import (
    analyze_scorer_effect,
)


def _row(index: int, *, group: str, natural_video_c: float = 5.0) -> dict[str, object]:
    return {
        "sample_id": f"sample-{index}",
        "source_group": group,
        "G_N_E_N": {"sync_c": natural_video_c, "sync_d": 8.0},
        "G_M_E_N": {"sync_c": natural_video_c + 1.0, "sync_d": 7.0},
        "G_N_E_M": {"sync_c": natural_video_c + 0.25, "sync_d": 7.75},
        "G_M_E_M": {"sync_c": natural_video_c + 1.5, "sync_d": 6.5},
    }


def test_four_cell_analysis_orients_sync_d_as_improvement() -> None:
    result = analyze_scorer_effect(
        [_row(0, group="g0"), _row(1, group="g1"), _row(2, group="g1")],
        bootstrap_seed=7,
    )
    natural = result["contrasts"]["visual_under_natural_audio"]
    assert result["record_count"] == 3
    assert result["source_group_count"] == 2
    assert natural["sync_c"]["record_mean"] == pytest.approx(1.0)
    assert natural["sync_d"]["record_mean"] == pytest.approx(1.0)
    assert natural["sync_d"]["group_mean"] == pytest.approx(1.0)


def test_scorer_visual_interaction_is_separate_from_cross_cell_difference() -> None:
    result = analyze_scorer_effect(
        [_row(0, group="g0"), _row(1, group="g1")],
        bootstrap_seed=11,
    )
    interaction = result["contrasts"]["scorer_visual_interaction"]
    assert interaction["sync_c"]["record_mean"] == pytest.approx(0.25)
    assert interaction["sync_d"]["record_mean"] == pytest.approx(0.25)


def test_audio_main_effect_averages_fixed_video_contrasts() -> None:
    result = analyze_scorer_effect([_row(0, group="g0"), _row(1, group="g1")])
    audio = result["contrasts"]["mfa_scoring_audio_main_effect"]
    assert audio["sync_c"]["record_mean"] == pytest.approx(0.375)
    assert audio["sync_d"]["record_mean"] == pytest.approx(0.375)


def test_four_cell_analysis_rejects_missing_cell() -> None:
    row = _row(0, group="g0")
    del row["G_N_E_M"]
    with pytest.raises(ValueError, match="missing score cell"):
        analyze_scorer_effect([row])
