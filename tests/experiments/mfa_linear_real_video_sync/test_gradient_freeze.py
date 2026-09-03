"""Identity, freeze, and end-to-end gradient contract tests."""
from __future__ import annotations

import torch
from torch import nn

from scripts.experiments.mfa_linear_real_video_sync.config import MODEL_RESIDUAL_SCALE, SEED
from scripts.experiments.mfa_linear_real_video_sync.model import (
    assert_identity,
    assert_syncnet_frozen,
    build_waveform_model,
    disposable_two_backward_smoke,
    freeze_syncnet,
    module_state_sha256,
    trainable_parameters,
)
from scripts.experiments.mfa_linear_real_video_sync.syncnet_loss import (
    audio_embeddings,
    official_syncnet_distance_curve,
)


class TinySyncNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio = nn.Sequential(nn.Flatten(), nn.Linear(13 * 20, 8), nn.BatchNorm1d(8))

    def forward_aud(self, value: torch.Tensor) -> torch.Tensor:
        return self.audio(value)


def test_fresh_residual_tcn_is_exact_identity_and_has_only_waveform_modules() -> None:
    model = build_waveform_model(seed=SEED)
    source = torch.randn(1, 1, 1024)
    assert_identity(model, source)
    assert model.residual_scale == MODEL_RESIDUAL_SCALE
    forbidden = (nn.MultiheadAttention, nn.Transformer, nn.Embedding, nn.LSTM, nn.GRU)
    assert not any(isinstance(module, forbidden) for module in model.modules())
    assert model.input_projection.in_channels == 1
    assert model.output_projection.out_channels == 1


def test_frozen_syncnet_keeps_state_and_live_input_gradient() -> None:
    syncnet = TinySyncNet()
    before = module_state_sha256(syncnet)
    freeze_syncnet(syncnet)
    assert_syncnet_frozen(syncnet)
    model = build_waveform_model(seed=SEED)
    optimizer_parameters = trainable_parameters(model, syncnet)
    assert optimizer_parameters
    assert all(not parameter.requires_grad for parameter in syncnet.parameters())

    waveform = (0.1 * torch.randn(1, 1, 4000)).requires_grad_()
    embedding = audio_embeddings(syncnet, waveform, video_frame_count=6)
    assert embedding.requires_grad
    visual = torch.randn_like(embedding)
    curve = official_syncnet_distance_curve(embedding, visual, vshift=1)
    curve.sum().backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert waveform.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in syncnet.parameters())
    assert module_state_sha256(syncnet) == before
    assert not syncnet.training


def test_disposable_two_backward_smoke_restarts_from_same_hash() -> None:
    source = 0.1 * torch.randn(1, 1, 512)
    target = source + 0.01 * torch.sin(torch.linspace(0, 20, source.shape[-1]))[None, None]

    def objective(model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
        return torch.mean((model(waveform) - target) ** 2)

    result = disposable_two_backward_smoke(seed=SEED, waveform=source, objective=objective)
    assert result["status"] == "GO"
    assert result["first_output_projection_gradient_norm"] > 0
    assert result["second_upstream_gradient_norm"] > 0
    assert result["initial_model_sha256"] == result["fresh_model_sha256"]
