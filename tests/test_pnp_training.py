"""Tests for differentiable acoustic losses and CPU overfit training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pnp_audio_enhancer import ResidualTCN
from train_pnp_audio_enhancer import (
    LossWeights,
    acoustic_loss,
    log_mel_loss,
    multi_resolution_stft_loss,
    train_paired,
)


def test_stft_and_mel_losses_propagate_input_gradient() -> None:
    prediction = torch.randn(1, 1, 1024, requires_grad=True)
    target = torch.randn(1, 1, 1024)
    stft = multi_resolution_stft_loss(prediction, target, fft_sizes=(128, 256))
    mel = log_mel_loss(prediction, target, n_fft=128, n_mels=16)
    (stft + mel).backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(stft) > 0
    assert float(mel) > 0


def test_acoustic_loss_identity_is_zero_except_residual() -> None:
    waveform = torch.randn(1, 1, 1024)
    loss, terms = acoustic_loss(
        waveform,
        waveform,
        input_waveform=waveform,
        weights=LossWeights(),
    )
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)
    assert all(torch.allclose(value, torch.tensor(0.0), atol=1e-7) for value in terms.values())


def test_optional_frozen_ssl_teacher_keeps_student_gradient() -> None:
    class MeanEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return waveform[..., ::8] * self.scale

    encoder = MeanEncoder()
    prediction = torch.randn(1, 1, 1024, requires_grad=True)
    target = torch.randn(1, 1, 1024)
    loss, terms = acoustic_loss(prediction, target, ssl_encoder=encoder)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert encoder.scale.requires_grad is False
    assert "ssl" in terms
    assert float(terms["ssl"]) > 0


def test_cpu_training_writes_checkpoint_and_updates_model(tmp_path: Path) -> None:
    sr = 16_000
    natural = np.sin(np.linspace(0, 4 * np.pi, 1024, dtype=np.float32))
    target = np.roll(natural, 2).astype(np.float32)
    natural_path = tmp_path / "natural.wav"
    target_path = tmp_path / "target.wav"
    sf.write(natural_path, natural, sr)
    sf.write(target_path, target, sr)
    torch.manual_seed(0)
    model = ResidualTCN(channels=8, dilations=(1, 2))
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history = train_paired(
        [{"natural_path": str(natural_path), "audio_path": str(target_path)}],
        epochs=10,
        learning_rate=1e-4,
        model=model,
        checkpoint_path=tmp_path / "model.pt",
    )
    assert len(history) == 10
    assert history[-1]["loss"] < history[0]["loss"]
    assert (tmp_path / "model.pt").exists()
    assert any(not torch.equal(initial[key], value) for key, value in model.state_dict().items())
