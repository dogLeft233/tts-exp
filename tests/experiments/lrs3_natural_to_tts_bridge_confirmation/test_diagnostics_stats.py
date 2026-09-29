import numpy as np

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation import config
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.diagnostics import (
    directional_metrics,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.scoring import (
    expected_cells_for_sample,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.stats import (
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


def test_matrix_is_exactly_the_registered_six_cells() -> None:
    cells = expected_cells_for_sample("sample")
    assert len(cells) == 6
    assert len({cell["cell"] for cell in cells}) == 6
    assert cells == [
        {"sample_id": "sample", "cell": "V_N/A_N", "video_arm": "N", "audio_arm": "N"},
        {"sample_id": "sample", "cell": "V_N_REPEAT/A_N", "video_arm": "N_REPEAT", "audio_arm": "N"},
        {"sample_id": "sample", "cell": "V_LOCAL_SWAP/A_N", "video_arm": "LOCAL_SWAP", "audio_arm": "N"},
        {"sample_id": "sample", "cell": "V_LOCAL_SWAP/A_LOCAL_SWAP", "video_arm": "LOCAL_SWAP", "audio_arm": "LOCAL_SWAP"},
        {"sample_id": "sample", "cell": "V_BRIDGE_075/A_N", "video_arm": "BRIDGE_075", "audio_arm": "N"},
        {"sample_id": "sample", "cell": "V_BRIDGE_075/A_BRIDGE_075", "video_arm": "BRIDGE_075", "audio_arm": "BRIDGE_075"},
    ]


def test_group_bootstrap_is_repeatable_and_uses_registered_defaults() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    groups = ["g1", "g2", "g3", "g4"]
    first = cluster_bootstrap(values, groups)
    second = cluster_bootstrap(values, groups)
    assert first == second
    assert first["mean"] == 2.5
    assert first["draws"] == config.BOOTSTRAP_DRAWS
    assert first["seed"] == config.BOOTSTRAP_SEED
    assert first["source_group_count"] == 4


def test_bootstrap_rejects_non_finite_values() -> None:
    try:
        cluster_bootstrap([np.nan], ["g"])
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite value was accepted")
