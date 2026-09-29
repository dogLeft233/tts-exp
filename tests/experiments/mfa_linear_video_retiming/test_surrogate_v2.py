from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError
from scripts.experiments.mfa_linear_video_retiming.retime import build_map
from scripts.experiments.mfa_linear_video_retiming.surrogate import interpolate_visual, score_proxy


def test_identity_matches_direct_float32_distance() -> None:
    rng = np.random.default_rng(17)
    visual = rng.normal(size=(111, 1024)).astype(np.float32)
    audio = rng.normal(size=(111, 1024)).astype(np.float32)
    mapping = build_map(115, 111)
    rows = np.arange(15, 75)
    result = score_proxy(visual, audio, mapping, rows)
    direct = np.stack([np.sqrt(np.sum(np.square(visual[rows] - audio[rows + lag] + np.float32(1e-6), dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32)
                       for lag in range(-15, 16)], axis=1).mean(axis=0, dtype=np.float32)
    assert np.allclose(result["curve"], direct, atol=1e-4, rtol=0)
    assert result["score_kind"] == "proxy_embedding"


def test_centre_anchor_differs_from_start_on_nonuniform_map() -> None:
    values = np.arange(130 * 1024, dtype=np.float32).reshape(130, 1024) / 10000
    base = build_map(134, 130)
    deltas = np.asarray(base["knot_deltas"])
    deltas[2:5] = 0.25
    mapping = build_map(134, 130, deltas, positions=base["knot_positions"])
    rows = np.arange(15, 85)
    centre = interpolate_visual(values, mapping, rows, anchor="window_center")
    start = interpolate_visual(values, mapping, rows, anchor="window_start")
    assert not np.array_equal(centre, start)


def test_out_of_bounds_is_rejected_without_padding() -> None:
    visual = np.zeros((100, 1024), dtype=np.float32)
    audio = np.zeros_like(visual)
    with pytest.raises(ProtocolError, match="PROXY_SUPPORT_OUT_OF_BOUNDS"):
        score_proxy(visual, audio, build_map(104, 100), np.arange(0, 25))
