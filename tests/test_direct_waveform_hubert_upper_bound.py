from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from direct_waveform_hubert_upper_bound import (  # noqa: E402
    _masked_feature_metrics,
    _optimize_direct_waveform,
)


class FakeEncoder(nn.Module):
    def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool) -> SimpleNamespace:
        assert output_hidden_states is True
        frames = waveform.shape[-1] // 320
        signal = waveform[:, : frames * 320].reshape(waveform.shape[0], frames, 320).mean(dim=-1)
        base = signal.unsqueeze(-1).expand(-1, -1, 768)
        return SimpleNamespace(hidden_states=tuple(base + float(index) for index in range(7)))


def test_masked_feature_metrics_excludes_unmatched_frames() -> None:
    predicted = torch.zeros(1, 2, 768)
    target = torch.ones_like(predicted)
    mask = torch.tensor([[True, False]])

    metrics = _masked_feature_metrics(predicted, target, mask)

    assert metrics["matched_frame_count"] == 1
    assert metrics["coverage"] == 0.5
    assert metrics["smooth_l1"] == 0.5


def test_direct_waveform_optimization_reduces_feature_gap() -> None:
    encoder = FakeEncoder()
    natural = torch.zeros(1, 1, 640)
    target = torch.full((1, 2, 768), 7.0)
    mask = torch.tensor([[True, False]])

    result = _optimize_direct_waveform(
        natural,
        target,
        mask,
        encoder,
        mode="unbounded",
        residual_scale=1.0,
        steps=8,
        learning_rate=0.1,
    )

    assert result["final"]["combined"] < result["initial"]["combined"]
    assert result["gap_reduction_fraction"] > 0
    assert result["final"]["clipped_sample_count"] == 0
    assert result["final"]["residual_rms"] > 0
