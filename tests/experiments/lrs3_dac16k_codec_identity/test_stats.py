from __future__ import annotations

import numpy as np

from scripts.experiments.lrs3_dac16k_codec_identity import config
from scripts.experiments.lrs3_dac16k_codec_identity.stats import cluster_bootstrap


def test_matrix_has_nine_unique_ordered_cells() -> None:
    assert len(config.MATRIX_CELLS) == 9
    assert len(set(config.MATRIX_CELLS)) == 9
    assert config.MATRIX_CELLS[0] == "V_N/A_N"
    assert config.MATRIX_CELLS[-1] == "V_D/A_D"


def test_cluster_bootstrap_is_reproducible_and_grouped() -> None:
    values = [1.0, 2.0, 10.0]
    groups = ["a", "b", "b"]
    first = cluster_bootstrap(values, groups, seed=7, draws=200)
    second = cluster_bootstrap(values, groups, seed=7, draws=200)
    assert first == second
    assert first["source_group_count"] == 2
    assert first["record_count"] == 3
    assert first["ci95"][0] <= np.mean(values) <= first["ci95"][1]
