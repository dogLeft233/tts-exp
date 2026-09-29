from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from experiments.mfa_linear_trajectory_ablation import (
    ARMS,
    RESYNTH_ARMS,
    ProtocolError,
    _cell_map,
    _comparison,
    _curve,
    _curve_metrics,
    _load_historical_conditioning,
    _occurrence_mask,
    _sensitivity_table,
    ablate_phone_dynamics,
)


def test_mean_preserving_residual_scaling_and_identity() -> None:
    values = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [10.0, 11.0]])
    occurrences = [{"eligible": True, "frame_indices": [0, 1, 2]}]
    np.testing.assert_array_equal(ablate_phone_dynamics(values, occurrences, 1.0), values)
    half = ablate_phone_dynamics(values, occurrences, 0.5)
    zero = ablate_phone_dynamics(values, occurrences, 0.0)
    np.testing.assert_allclose(half[:3].mean(dim=0), values[:3].mean(dim=0))
    np.testing.assert_allclose((half[:3] - values[:3].mean(dim=0)) * 2.0, values[:3] - values[:3].mean(dim=0))
    np.testing.assert_allclose(zero[:3], values[:3].mean(dim=0).expand(3, -1))
    np.testing.assert_array_equal(half[3], values[3])


def test_occurrences_are_separate_and_silence_single_frame_are_unchanged() -> None:
    tokens = [
        {"token": "sil", "start_s": 0.0, "end_s": 0.04, "is_silence": True},
        {"token": "a", "start_s": 0.04, "end_s": 0.12},
        {"token": "a", "start_s": 0.12, "end_s": 0.14},
        {"token": "sil", "start_s": 0.14, "end_s": 0.40, "is_silence": True},
    ]
    occurrences, metadata = _occurrence_mask(tokens, tokens, 20)
    assert metadata["eligible_occurrence_count"] == 1
    assert occurrences[1]["eligible"] is True
    assert occurrences[2]["eligible"] is False
    assert occurrences[0]["reason"] == "silence"
    assert occurrences[2]["reason"] == "fewer_than_two_frames"
    values = torch.arange(20 * 2, dtype=torch.float32).reshape(20, 2)
    ablated = ablate_phone_dynamics(values, occurrences, 0.0)
    for occurrence in (occurrences[0], occurrences[2], occurrences[3]):
        for index in occurrence["frame_indices"]:
            assert torch.equal(ablated[index], values[index])


def test_curve_metrics_uses_first_minimum_and_zero_lag() -> None:
    values = [4.0] * 31
    values[0] = 2.0
    values[15] = 3.0
    values[16] = 2.0
    metrics = _curve_metrics(list(range(-15, 16)), values)
    assert metrics["D"] == 2.0
    assert metrics["k_star"] == -15
    assert metrics["d_zero"] == 3.0
    assert metrics["C"] == pytest.approx(2.0)


def test_bootstrap_is_deterministic_and_utterance_based() -> None:
    values = np.asarray([-1.0, 0.0, 2.0, 3.0])
    indices = np.random.default_rng(20260916).integers(0, len(values), size=(100, len(values)))
    first = _comparison(values, indices)
    second = _comparison(values, indices)
    assert first == second
    assert first["n"] == 4
    assert first["values"] == values.tolist()


def test_historical_loader_reads_conditioning_key_only(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    conditioning = torch.ones(3, 4)
    output = torch.full((3, 4), 99.0)
    torch.save({"conditioning": conditioning, "output": output}, path)
    row = {"historical_feature_path": str(path), "historical_feature_shape": [3, 4], "historical_conditioning_shape": [3, 4], "historical_feature_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
    loaded = _load_historical_conditioning(row)
    assert torch.equal(loaded, conditioning)
    assert not torch.equal(loaded, output)


def test_protocol_cell_count_and_t_raw_is_native_only() -> None:
    sample_ids = [1]
    cells = {(sid, arm, arm): {"C": 1.0, "D": 2.0} for sid in sample_ids for arm in ARMS}
    cells.update({(sid, arm, "N_RAW"): {"C": 1.0, "D": 2.0} for sid in sample_ids for arm in RESYNTH_ARMS})
    cells[(1, "N_RAW", "T_100")] = {"C": 1.0, "D": 2.0}
    result = _cell_map([{"sample_id": sid, "video_arm": video, "audio_arm": audio, "C": row["C"], "D": row["D"]} for (sid, video, audio), row in cells.items()])
    assert len(result) == 15
    assert (1, "T_RAW", "N_RAW") not in result
    assert (1, "N_RAW", "T_100") in result


def test_curve_support_threshold_boundaries() -> None:
    visual = np.zeros((60, 4), dtype=np.float32)
    audio = np.zeros((61, 4), dtype=np.float32)
    support_30 = list(range(15, 45))
    support_31 = list(range(15, 46))
    assert _curve(visual, audio, support_30, min_common_windows=30)[2] == 30
    assert _curve(visual, audio, support_31, min_common_windows=30)[2] == 31
    with pytest.raises(ProtocolError):
        _curve(visual, audio, support_30[:-1], min_common_windows=30)
    with pytest.raises(ProtocolError):
        _curve(visual, audio, support_31, min_common_windows=50)


def test_support_sensitivity_filters_complete_non_t_raw_cells() -> None:
    sample_ids = [1, 2]
    cells = {}
    for sid, support in ((1, 31), (2, 49)):
        for arm in ARMS:
            cells[(sid, arm, arm)] = {"C": float(sid), "D": 10.0, "common_support_count": support}
        for arm in RESYNTH_ARMS:
            cells[(sid, arm, "N_RAW")] = {"C": float(sid), "D": 10.0, "common_support_count": support}
        cells[(sid, "N_RAW", "T_100")] = {"C": float(sid), "D": 10.0, "common_support_count": support}
    table = _sensitivity_table(cells, sample_ids, thresholds=(30, 35, 50))
    assert [(row["threshold"], row["n"], row["sample_ids"]) for row in table] == [(30, 2, [1, 2]), (35, 1, [2]), (50, 0, [])]
    assert table[0]["G_own"]["C"] == pytest.approx(0.0)
    assert table[2]["G_own"]["C"] is None
