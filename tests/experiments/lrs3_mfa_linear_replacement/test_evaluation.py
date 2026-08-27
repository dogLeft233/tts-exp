from __future__ import annotations

import pytest

from scripts.experiments.lrs3_mfa_linear_replacement.run_evaluation import build_analysis
from scripts.experiments.lrs3_mfa_linear_replacement.protocol import canonical_sha256
from scripts.experiments.lrs3_mfa_linear_replacement.statistics import exact_sign_flip_pvalue

MODEL = "a" * 64
SOURCE = "b" * 64


def _rows(count: int = 24) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        cells = {
            "G_N_E_N": {"sync_c": 5.0, "sync_d": 8.0, "mux_sha256": f"{index:064x}"},
            "G_M_E_N": {"sync_c": 6.0, "sync_d": 7.0, "mux_sha256": f"{index + 24:064x}"},
            "G_N_E_M": {"sync_c": 5.5, "sync_d": 7.5, "mux_sha256": f"{index + 48:064x}"},
            "G_M_E_M": {"sync_c": 6.5, "sync_d": 6.5, "mux_sha256": f"{index + 72:064x}"},
        }
        rows.append({"sample_id": f"sample-{index}", "source_group": f"group-{index}", "syncnet_model_sha256": MODEL, "syncnet_source_sha256": SOURCE, **cells})
    return rows


def _expected_records(rows):
    return [{"sample_id": row["sample_id"], "source_group": row["source_group"]} for row in rows]


def test_exact_sign_flip_matches_small_bruteforce_oracle() -> None:
    result = exact_sign_flip_pvalue([1.0, 2.0])
    assert result["assignment_count"] == 4
    assert result["exceedance_count"] == 1
    assert result["p_value_one_sided"] == pytest.approx(0.25)
    assert exact_sign_flip_pvalue([1.0, 5e-13])["p_value_one_sided"] == pytest.approx(0.25)


def test_replacement_analysis_uses_natural_audio_authoritative_cells() -> None:
    rows = _rows()
    expected = _expected_records(rows)
    analysis = build_analysis(rows, model_sha256=MODEL, source_sha256=SOURCE, expected_records=expected, expected_cohort_sha256=canonical_sha256(expected), engineering_complete=True, bootstrap_draws=32)
    assert analysis["decision"] == "GO"
    assert analysis["co_primary"]["means"] == {"benefit_C": 1.0, "benefit_D": 1.0}
    assert analysis["benefits"]["joint_win_count"] == 24
    assert analysis["four_cell_means"]["G_M_E_N"]["sync_c"] == 6.0


def test_partial_authoritative_matrix_is_blocked_before_statistics() -> None:
    rows = _rows()
    rows[0].pop("G_M_E_N")
    expected = _expected_records(rows)
    with pytest.raises(ValueError, match="missing cell"):
        build_analysis(rows, model_sha256=MODEL, source_sha256=SOURCE, expected_records=expected, expected_cohort_sha256=canonical_sha256(expected), engineering_complete=False, bootstrap_draws=16)
