import sys

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing import config
from scripts.experiments.lrs3_real_video_local_timing.media import build_mapping
from scripts.experiments.lrs3_real_video_local_timing.scoring import (
    local_evidence,
    parse_syncnet_log,
    reconstruct_global,
)


def test_global_reconstruction_uses_the_official_column_offsets() -> None:
    matrix = np.ones((12, 31), dtype=np.float64)
    matrix[:, 18] = 0.25
    result = reconstruct_global(matrix)
    assert result["offset"] == -3
    assert result["sync_d"] == 0.25
    assert result["sync_c"] == 0.75
    assert result["offsets"][18] == -3


def test_local_masks_are_supported_rows_shared_by_the_three_arms() -> None:
    mappings = build_mapping(150)
    matrix = np.ones((145, 31), dtype=np.float64)
    evidence = local_evidence(matrix, mappings[config.WARP_ARM], 150, 150 * 640)
    assert evidence["supported_rows"][0] == config.VSHIFT
    assert evidence["supported_rows"][-1] == 145 - config.VSHIFT - 1
    assert len(evidence["PLUS"]["rows"]) >= config.MIN_LOCAL_ROWS
    assert len(evidence["MINUS"]["rows"]) >= config.MIN_LOCAL_ROWS
    assert all(row >= config.VSHIFT and row < 145 - config.VSHIFT for row in evidence["PLUS"]["rows"] + evidence["MINUS"]["rows"])


def test_official_calc_pdist_maps_late_video_read_to_minus_three_offset() -> None:
    import torch

    sys.path.insert(0, "third_party/syncnet_python")
    from SyncNetInstance import calc_pdist

    audio = torch.eye(80, dtype=torch.float32)
    visual = audio.roll(-3, dims=0)
    distances = calc_pdist(visual, audio, vshift=config.VSHIFT)
    row = distances[20]
    assert int(torch.argmin(row).item()) == config.VSHIFT + 3
    assert config.VSHIFT - int(torch.argmin(row).item()) == -3


def test_syncnet_log_requires_one_unambiguous_value_per_metric(tmp_path) -> None:
    path = tmp_path / "score.log"
    path.write_text("AV offset: \t-3\nMin dist: \t1.250\nConfidence: \t0.750\n", encoding="utf-8")
    assert parse_syncnet_log(path) == {"av_offset": -3, "sync_d": 1.25, "sync_c": 0.75}
