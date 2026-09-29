from __future__ import annotations

from scripts.experiments.phone_separability_mechanism.candidate import _select_dev


def _score(arm: str, effect: float, margin: float) -> dict:
    return {
        "arm": arm,
        "views": {"core": {"accuracy": {"estimate": effect, "ci_low": effect - 0.01, "n_groups": 6}, "margin": {"estimate": margin, "ci_low": margin - 0.001}}},
        "tts_baseline": {"accuracy": {"estimate": 0.10}},
    }


def test_candidate_gate_selects_only_from_dev_and_ranks_margin() -> None:
    result = _select_dev([_score("SPECTRAL_DRC", 0.08, 0.01), _score("EQ_MATCH", 0.07, 0.02)], {"candidate": {"selection_view": "core", "min_tts_fraction": 0.5, "selectable_arms": ["SPECTRAL_DRC", "EQ_MATCH"]}})
    assert result["status"] == "SELECTED"
    assert result["selected_arm"] == "EQ_MATCH"
    assert result["selection_split"] == "dev"
    assert result["e_seen_locked"] is True
