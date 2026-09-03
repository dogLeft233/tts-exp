import numpy as np
import pytest

from scripts.experiments.asr_sync_error_correlation.local_sync import (
    compute_local_scores,
    map_local_scores,
    parity_report,
    select_track,
    validate_paired_track_metadata,
)


def test_local_formula_and_mapping():
    dists = np.ones((20, 31), dtype=float)
    dists[:, 14] = np.linspace(0.1, 1.0, 20)
    scores = compute_local_scores(dists)
    assert scores["j_star"] == 14
    assert scores["av_offset_frames"] == 1
    assert scores["sync_d"] == pytest.approx(np.mean(dists[:, 14]), abs=1e-6)
    mapped = map_local_scores(scores, track_start_frame=10, track_end_frame=80, audio_duration_s=10)
    assert mapped["timestamps_s"][0] > 0.4
    assert "median_filter_edge" in mapped["excluded_counts"]


def test_track_tie_uses_lowest_index_and_no_track_fails():
    tracks = {4: {"frame": list(range(60))}, 2: {"frame": list(range(60))}}
    assert select_track(tracks)["selected_track_index"] == 2
    with pytest.raises(ValueError):
        select_track({1: {"frame": list(range(50))}})


def test_parity_and_pair_identity():
    dists = np.ones((20, 31), dtype=float)
    scores = compute_local_scores(dists)
    assert parity_report(scores, upstream_offset=15, upstream_confidence=scores["sync_c"])["passed"]
    with pytest.raises(ValueError):
        validate_paired_track_metadata({"selected_track_index": 1}, {"selected_track_index": 2})
