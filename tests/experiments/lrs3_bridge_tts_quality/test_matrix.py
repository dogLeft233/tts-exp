from __future__ import annotations

import numpy as np

from scripts.experiments.lrs3_bridge_tts_quality import config
from scripts.experiments.lrs3_bridge_tts_quality.scoring import parse_syncnet_log


def test_registered_matrix_counts_and_cells() -> None:
    assert len(config.ARMS) == 4
    assert len(config.REPEATS) == 2
    assert len(config.SCORE_CELLS) == 7
    assert config.EXPECTED_VIDEO_COUNT == 176
    assert config.EXPECTED_CELL_COUNT == 308
    cells = config.expected_score_cells(config.EXPECTED_SAMPLE_IDS[0])
    assert [row["cell"] for row in cells] == list(config.MATRIX_CELLS)
    assert cells[-1]["cell"] == "V_N/A_N_REV"


def test_syncnet_log_parser_keeps_unrounded_values_and_offset() -> None:
    path = __import__("pathlib").Path(__file__).with_name("_syncnet.log")
    path.write_text("Confidence:\t0.123456\nMin dist:\t1.234567\nAV offset:\t-2\n", encoding="utf-8")
    try:
        result = parse_syncnet_log(path)
    finally:
        path.unlink()
    assert np.isclose(result["sync_c"], 0.123456)
    assert np.isclose(result["sync_d"], 1.234567)
    assert result["av_offset"] == -2
