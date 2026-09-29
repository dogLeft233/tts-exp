import numpy as np

from scripts.experiments.local_swap_minimal_replay import config
from scripts.experiments.local_swap_minimal_replay.analysis import _segment_rows
from scripts.experiments.local_swap_minimal_replay.scoring import reconstruct_global


def test_global_reconstruction_uses_syncnet_offset_axis() -> None:
    matrix = np.ones((30, 31), dtype=np.float64)
    matrix[:, 18] = 0.25
    result = reconstruct_global(matrix)
    assert result["offset"] == -3
    assert result["sync_d"] == 0.25
    assert result["sync_c"] == 0.75
    assert result["offsets"][18] == -3


def test_local_support_excludes_boundary_receptive_fields() -> None:
    rows = _segment_rows(4 * 26 - config.WINDOW_FRAMES, 26, 1)
    assert rows[0] >= config.VSHIFT
    assert rows[-1] + config.WINDOW_FRAMES <= 52 - config.LOCAL_MARGIN_FRAMES
    assert len(rows) >= config.MIN_LOCAL_WINDOWS
