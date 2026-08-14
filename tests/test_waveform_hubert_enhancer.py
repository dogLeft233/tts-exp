#!/usr/bin/env python3
"""Small, dependency-light tests for the HuBERT waveform enhancer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from waveform_hubert_enhancer import (  # noqa: E402
    ConditionedResidualTCN,
    EncoderInterface,
    HuBERTWaveformEnhancer,
    WaveformEnhancer,
)


class FakeHuBERT(torch.nn.Module):
    def __init__(self, hidden_size: int = 768, stride: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stride = stride
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool = False):
        if waveform.ndim != 2:
            raise ValueError("fake waveform must be [B,N]")
        usable = waveform[..., : waveform.shape[-1] // self.stride * self.stride]
        frames = usable.reshape(waveform.shape[0], -1, self.stride).mean(-1, keepdim=True)
        hidden = frames.repeat(1, 1, self.hidden_size) * self.scale
        return type("Output", (), {"hidden_states": tuple(hidden for _ in range(7))})()


def make_model() -> HuBERTWaveformEnhancer:
    return HuBERTWaveformEnhancer(
        FakeHuBERT(),
        interface=EncoderInterface(frame_stride_samples=320),
        channels=8,
        dilations=(1, 2),
    )


def test_fresh_conditioned_model_is_identity() -> None:
    torch.manual_seed(0)
    model = make_model()
    x = torch.randn(1, 1, 640)
    output = model(x)
    assert torch.equal(output["waveform"], x)
    assert output["enhanced_features"].shape[-1] == 768


def test_conditioning_is_finite_and_model_has_bound() -> None:
    model = make_model()
    x = torch.randn(1, 1, 641)
    output = model(x)
    assert output["conditioning"].shape == (1, 64, 641)
    assert torch.isfinite(output["conditioning"]).all()
    assert float(output["residual"].abs().max()) <= 0.2 + 1e-6


def test_frozen_encoder_but_enhanced_feature_gradient_reaches_model() -> None:
    model = make_model()
    x = torch.randn(1, 1, 640)
    output = model(x)
    target = torch.zeros_like(output["enhanced_features"])
    loss = torch.nn.functional.smooth_l1_loss(output["enhanced_features"], target)
    loss.backward()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.waveform_model.parameters())
    assert any(p.grad is not None for p in model.condition_projection.parameters())


def test_numpy_wrapper_preserves_exact_length() -> None:
    wrapper = WaveformEnhancer(make_model(), chunk_seconds=0.08, stride_seconds=0.064)
    audio = np.sin(np.linspace(0, 3, 777, dtype=np.float32))
    result = wrapper.enhance(audio)
    assert result.shape == audio.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.array_equal(result, audio)


def test_bad_encoder_interface_rejected() -> None:
    with pytest.raises(ValueError, match="768-D"):
        HuBERTWaveformEnhancer(
            FakeHuBERT(hidden_size=1024),
            interface=EncoderInterface(hidden_size=1024),
            channels=8,
            dilations=(1,),
        )


def test_residual_tcn_requires_matching_conditioning() -> None:
    model = ConditionedResidualTCN(conditioning_dim=4, channels=8, dilations=(1,))
    with pytest.raises(ValueError, match="conditioning"):
        model(torch.zeros(1, 1, 10), torch.zeros(1, 3, 10))
