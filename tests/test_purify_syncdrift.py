"""Checks for the NAS SyncDrift video alignment port."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "purify_syncdrift.py"
SPEC = importlib.util.spec_from_file_location("purify_syncdrift_port", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_window_search_and_frame_mapping_recover_known_offset() -> None:
    rng = np.random.default_rng(19)
    audio = rng.normal(size=(150, 32)).astype(np.float32)
    lip = np.zeros_like(audio)
    lip[2:] = audio[:-2]
    crop = module.SyncNetCrop(Path("crop.avi"), 0, 149, 0)
    info = module.VideoInfo(25.0, 150, 224, 224, 6.0, 6.0)

    result = module.estimate_correction(
        module.Embeddings(audio, lip, crop), info,
        mode="window", window_sec=2.0, stride_sec=1.0,
        search_min_ms=-120.0, search_max_ms=120.0,
        median_filter_size=3, min_valid_feats=8, flat_score_epsilon=1e-4,
    )

    assert result.valid_window_ratio == 1.0
    assert np.all(result.frame_correction_ms == 80.0)
    np.testing.assert_array_equal(result.mapped_indices[:5], [2, 3, 4, 5, 6])
    assert result.mapped_indices[-1] == 149


def test_local_syncnet_log_parser_uses_official_scores() -> None:
    log = "AV offset: 2\nMin dist: 7.125\nConfidence: 4.500\n"
    assert module.parse_syncnet_scores(log) == (7.125, 4.5)
    assert module.build_parser().get_default("syncnet_root") == str(module.DEFAULT_SYNCNET_ROOT)
