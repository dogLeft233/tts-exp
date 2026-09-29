import numpy as np

from scripts.experiments.lrs3_phase_preserving_replacement_envelope import config
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.diagnostics import (
    directional_metrics,
)
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.scoring import (
    expected_cells_for_sample,
)
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.stats import (
    cluster_bootstrap,
)


def test_directional_projection_and_orthogonal_residual() -> None:
    natural = np.zeros((2, 2), dtype=np.float64)
    mfa = np.asarray([[1.0, 0.0], [0.0, 0.0]])
    candidate = np.asarray([[0.5, 0.0], [0.0, 0.0]])
    result = directional_metrics(natural, mfa, candidate)
    assert result["progress"] == 0.5
    assert result["orthogonal_ratio"] == 0.0


def test_directional_projection_reports_orthogonal_residual() -> None:
    natural = np.zeros((2, 2), dtype=np.float64)
    mfa = np.asarray([[1.0, 0.0], [0.0, 0.0]])
    candidate = np.asarray([[0.5, 0.5], [0.0, 0.0]])
    result = directional_metrics(natural, mfa, candidate)
    assert result["progress"] == 0.5
    assert result["orthogonal_ratio"] > 0.0


def test_matrix_enumerator_has_one_baseline_and_two_cells_per_arm() -> None:
    cells = expected_cells_for_sample("sample")
    assert len(cells) == 13
    assert len({cell["cell"] for cell in cells}) == 13
    assert cells[0] == {"sample_id": "sample", "cell": "V_N/A_N", "video_arm": "N", "audio_arm": "N"}
    assert sum(cell["video_arm"] == "N" for cell in cells) == 1
    assert sum(cell["audio_arm"] == "N" for cell in cells) == 7


def test_group_bootstrap_is_repeatable_and_positive_direction_is_preserved() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    groups = ["g1", "g2", "g3", "g4"]
    first = cluster_bootstrap(values, groups)
    second = cluster_bootstrap(values, groups)
    assert first == second
    assert first["mean"] == 2.5
    assert first["draws"] == config.BOOTSTRAP_DRAWS
    assert first["source_group_count"] == 4


def test_bootstrap_rejects_non_finite_values() -> None:
    try:
        cluster_bootstrap([np.nan], ["g"])
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite value was accepted")
