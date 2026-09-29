from __future__ import annotations

import numpy as np

from scripts.experiments.wav2lip_residual_local_response.support import chunk_columns, exposure_for, row_supports


def test_official_chunks_have_last_window_and_five_frame_union() -> None:
    chunks = chunk_columns()
    assert len(chunks) == 93
    assert np.array_equal(chunks[-1], np.arange(292, 308))
    supports = row_supports()
    assert supports[30][0] == 96
    assert supports[30][-1] == 123


def test_overlap_is_counted_once() -> None:
    result = exposure_for([96, 97, 98, 99, 100])
    first = next(row for row in result["rows"] if row["row"] == 30)
    assert first["overlap"] == [96, 97, 98, 99, 100]
    assert first["exposure"] == 5 / len(first["support"])


def test_zero_exposure_is_preserved() -> None:
    result = exposure_for([0, 1])
    assert all(row["exposure"] == 0.0 for row in result["rows"] if 30 <= row["row"] <= 57)
