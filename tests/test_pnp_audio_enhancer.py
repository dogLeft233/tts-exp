"""Tests for the CPU-only waveform enhancer MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pnp_audio_enhancer import ResidualTCN, WaveformEnhancer, enhance


def test_fresh_model_is_identity() -> None:
    torch.manual_seed(0)
    model = ResidualTCN(channels=8, dilations=(1, 2, 4))
    x = torch.randn(2, 1, 257)
    with torch.no_grad():
        y = model(x)
    assert torch.equal(y, x)


def test_model_preserves_shape_and_residual_bound() -> None:
    model = ResidualTCN(channels=8, dilations=(1, 2), residual_scale=0.2)
    x = torch.randn(3, 1, 101)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.max(torch.abs(y - x)) <= 0.2 + 1e-6


def test_numpy_api_preserves_exact_length_and_dtype() -> None:
    wrapper = WaveformEnhancer(
        ResidualTCN(channels=8, dilations=(1, 2)),
        chunk_seconds=0.01,
        stride_seconds=0.005,
    )
    audio = np.sin(np.linspace(0, 4 * np.pi, 237, dtype=np.float64))
    output = wrapper.enhance(audio)
    assert output.shape == audio.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.array_equal(output, audio.astype(np.float32))


def test_short_input_reflect_padding() -> None:
    model = ResidualTCN(channels=8, dilations=(1, 2))
    output = WaveformEnhancer(
        model,
        chunk_seconds=0.1,
        stride_seconds=0.05,
    ).enhance(np.array([0.1, -0.2, 0.3], dtype=np.float32))
    assert output.shape == (3,)
    assert np.array_equal(output, np.array([0.1, -0.2, 0.3], dtype=np.float32))


def test_chunked_output_is_finite_and_exact_length() -> None:
    wrapper = WaveformEnhancer(
        ResidualTCN(channels=8, dilations=(1, 2)),
        chunk_seconds=0.02,
        stride_seconds=0.01,
    )
    audio = np.random.default_rng(2).normal(size=1000).astype(np.float32)
    output = wrapper.enhance(audio)
    assert output.shape == audio.shape
    assert np.isfinite(output).all()
    assert np.array_equal(output, audio)


def test_sample_rate_and_mono_validation() -> None:
    wrapper = WaveformEnhancer(ResidualTCN(channels=8, dilations=(1,)))
    with pytest.raises(ValueError, match="sample_rate"):
        wrapper.enhance(np.zeros(10, dtype=np.float32), sample_rate=8000)
    with pytest.raises(ValueError, match="mono"):
        wrapper.enhance(np.zeros((2, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        wrapper.enhance(np.array([0.0, np.nan], dtype=np.float32))


def test_convenience_function_is_identity_before_training() -> None:
    audio = np.linspace(-0.5, 0.5, 111, dtype=np.float32)
    output = enhance(
        audio,
        ResidualTCN(channels=8, dilations=(1, 2)),
        chunk_seconds=0.01,
        stride_seconds=0.005,
    )
    assert np.array_equal(output, audio)
