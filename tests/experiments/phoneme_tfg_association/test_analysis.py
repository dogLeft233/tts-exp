from __future__ import annotations

import numpy as np

from scripts.experiments.phoneme_tfg_association.analysis import fit_primary, sensitivity_analysis


def _synthetic(beta: float, noise: float = 0.0):
    rng = np.random.default_rng(12)
    rows = []
    pairs = ["qwen_cloud|qwen_local", "qwen_cloud|index_tts2", "qwen_cloud|cosyvoice2", "qwen_local|index_tts2", "qwen_local|cosyvoice2", "index_tts2|cosyvoice2"]
    for pair_index, pair in enumerate(pairs):
        for repeat in range(6):
            x = (repeat - 2.5) * 0.35 + pair_index * 0.02
            rows.append({"block_id": f"{pair_index}-{repeat}", "source_group": f"g{pair_index}-{repeat}", "pair": pair, "x": x, "y": beta * x + noise * rng.normal(), "complete": True})
    return rows


def test_primary_regression_recovers_within_pair_slope():
    result = fit_primary(_synthetic(1.7, 0.01), wild_draws=99)
    assert result["scientific_status"] == "POSITIVE_WITHIN_PAIR_ASSOCIATION"
    assert abs(result["beta"] - 1.7) < 0.2
    assert result["design_rank"] == 7
    assert result["residual_df"] == 29


def test_weak_support_is_not_relabelled_as_null():
    result = fit_primary(_synthetic(1.0)[:12], wild_draws=19)
    assert result["scientific_status"] == "INSUFFICIENT_SUPPORT"
    assert "reason" in result


def test_registered_sensitivity_views_are_explicit_and_not_primary_inference():
    rows = []
    for index, row in enumerate(_synthetic(1.2)):
        duration_a = 0.85 + (index % 6) * 0.02
        duration_b = 1.15 - (index % 5) * 0.015
        rows.append({
            **row,
            "duration_ratio_a": duration_a,
            "duration_ratio_b": duration_b,
            "duration_log_ratio": np.log(duration_b / duration_a),
            "silence_fraction_delta": 0.005 + (index % 4) * 0.003,
        })
    result = sensitivity_analysis(rows, min_complete_blocks=30, min_pair_blocks=4)
    assert result["status"] == "COMPLETE"
    assert result["not_for_primary_inference"] is True
    assert result["duration_ratio_subset"]["n_blocks"] == 36
    assert result["duration_silence_adjusted"]["status"] == "COMPLETE"
    assert len(result["leave_one_pair_out"]) == 6
    assert sum(len(values) for values in result["leave_one_block_within_pair"].values()) == 36
