from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/experiments/p1_gain_dose_response_20261016/full_context_distance_overlay.py"
SPEC = importlib.util.spec_from_file_location("p1_full_context_distance_overlay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


torch = pytest.importorskip("torch")


def test_full_context_uses_audio_tail_when_audio_is_longer_than_video() -> None:
    # At q=24 and lag=+15, the real audio index is 39. V has only 30 rows,
    # while A has 40; truncating A to min(A,V) would substitute right padding.
    visual = np.zeros((30, 2), dtype=np.float32)
    audio = np.zeros((40, 2), dtype=np.float32)
    visual[24] = np.array([1.0, 0.0], dtype=np.float32)
    audio[39] = np.array([1.0, 0.0], dtype=np.float32)

    actual = overlay.full_context_distances(audio, visual, [24])

    assert actual.shape == (1, 31)
    assert actual[0, 30] == pytest.approx(float(torch.nn.functional.pairwise_distance(
        torch.from_numpy(visual[24:25]), torch.from_numpy(audio[39:40]), eps=1e-6
    ).item()), abs=0.0)
    assert actual[0, 30] < 1e-5


def test_full_context_requires_all_real_lag_rows() -> None:
    audio = np.zeros((31, 2), dtype=np.float32)
    visual = np.zeros((20, 2), dtype=np.float32)
    with pytest.raises(overlay.OverlayError, match="lacks real"):
        overlay.full_context_distances(audio, visual, [16])


def test_full_context_metrics_use_fixed_anchor_not_new_argmin() -> None:
    curve = np.asarray([9.0] * 31, dtype=np.float64)
    curve[2] = 1.0
    curve[21] = 2.0
    metrics = overlay.curve_metrics(curve, anchor_index=21)
    assert metrics["argmin_index"] == 2
    assert metrics["anchor_lag_index"] == 21
    assert metrics["anchor_lag_frames"] == 6
    assert metrics["anchor_D"] == 2.0
    assert metrics["anchor_C"] == metrics["B"] - 2.0


def test_primary_fields_are_preserved_when_full_fields_change() -> None:
    original = {
        "speakers": [{
            "speaker_id": "S0102",
            "conditions": [{
                "condition_id": "RAW_R1",
                "status": "SCORED",
                "q_support": [20],
                "distance_path": "old-primary.npy",
                "distance_sha256": "a" * 64,
                "distance_shape": [1, 31],
                "curve31": [1.0] * 31,
                "lags": list(range(-15, 16)),
                "C": 0.0,
                "B": 1.0,
                "Dmin": 1.0,
                "D0": 1.0,
                "bestlag": 0,
                "argmin_index": 15,
                "offset_at_lag_boundary": False,
                "search_gain": 0.0,
                "anchor": {"D": 1.0, "C": 0.0},
                "anchor_lag_frames": 0,
                "audio_feature_path": "audio.npy",
                "audio_feature_sha256": "b" * 64,
                "feature_path": "video.npy",
                "feature_sha256": "c" * 64,
                "visual_sha256": "c" * 64,
                "audio_variant_sha256": "d" * 64,
                "video_sha256": "e" * 64,
                "audio_rms_dbfs": -20.0,
                "relative_dose_db": 0.0,
                "curve31_full": [2.0] * 31,
                "C_full": 0.0,
            }],
        }]
    }
    derived = copy.deepcopy(original)
    derived["speakers"][0]["conditions"][0]["curve31_full"] = [3.0] * 31
    derived["speakers"][0]["conditions"][0]["C_full"] = 1.0

    overlay._validate_primary_unchanged(original, derived)

    derived["speakers"][0]["conditions"][0]["C"] = 1.0
    with pytest.raises(overlay.OverlayError, match="non-full score fields changed"):
        overlay._validate_primary_unchanged(original, derived)
