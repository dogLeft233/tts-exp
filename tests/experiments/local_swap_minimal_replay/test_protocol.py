from pathlib import Path

from scripts.experiments.local_swap_minimal_replay import config


def test_fixed_cell_matrix_has_eight_cells_per_sample() -> None:
    assert len(config.CELL_SPECS) == 8
    assert len(config.SAMPLE_IDS) == 3
    assert config.EXPECTED_NEW_CELL_COUNT == len(config.CELL_SPECS) * len(config.SAMPLE_IDS)


def test_run_id_is_restricted_to_a_fresh_safe_suffix() -> None:
    assert config.run_root_for("20260913_test").name == "local_swap_minimal_replay_20260913_test"
    try:
        config.run_root_for("../unsafe")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe run id was accepted")
