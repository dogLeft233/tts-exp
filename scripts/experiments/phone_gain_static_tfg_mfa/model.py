"""Fixed architecture for A/B/C MFA-conditioned gain heads."""

from __future__ import annotations

from typing import Any


def _torch():
    import torch

    return torch


class ResidualBlock:
    pass


def build_model(*, vocab_size: int, embedding_dim: int = 16, audio_channels: int = 24, hidden_channels: int = 64, max_gain_db: float = 6.0, dilations: tuple[int, ...] = (1, 2, 4)):
    torch = _torch()
    nn = torch.nn

    class _Block(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, 3, padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, 3, padding=dilation, dilation=dilation),
                nn.GELU(),
            )

        def forward(self, value):
            return value + self.net(value)

    class MFAConditionedGainEnhancer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim))
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.embedding.weight[0].zero_() if vocab_size == 1 else None
            self.input = nn.Conv1d(audio_channels + embedding_dim + 3, hidden_channels, 5, padding=2)
            self.blocks = nn.ModuleList([_Block(int(dilation)) for dilation in dilations])
            self.output = nn.Conv1d(hidden_channels, audio_channels, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            self.max_gain_db = float(max_gain_db)
            self.embedding_dim = int(embedding_dim)
            self.audio_channels = int(audio_channels)

        def forward(self, audio_features, phone_ids=None, timing=None, *, mode: str = "MFA_PHONE_TIME"):
            if audio_features.ndim != 3 or audio_features.shape[1] != self.audio_channels:
                raise ValueError(f"audio_features must be [B,{self.audio_channels},T]")
            batch, _, frames = audio_features.shape
            if phone_ids is None:
                phone_ids = torch.zeros((batch, frames), dtype=torch.long, device=audio_features.device)
            if timing is None:
                timing = torch.zeros((batch, 3, frames), dtype=audio_features.dtype, device=audio_features.device)
            if phone_ids.shape != (batch, frames) or timing.shape != (batch, 3, frames):
                raise ValueError("conditioning tensor shapes do not match features")
            if torch.any(phone_ids < 0) or torch.any(phone_ids >= self.embedding.num_embeddings):
                raise ValueError("phone_ids contain an out-of-range vocabulary id")
            ids = phone_ids
            identity = self.embedding(ids).transpose(1, 2)
            special = (ids == self.embedding.num_embeddings - 2) | (ids == self.embedding.num_embeddings - 1)
            identity = torch.where(special.unsqueeze(1), torch.zeros_like(identity), identity)
            time_features = timing
            if mode == "AUDIO_FEATURES":
                identity = torch.zeros_like(identity)
                time_features = torch.zeros_like(time_features)
            elif mode == "BOUNDARY_TIME":
                identity = torch.zeros_like(identity)
            elif mode != "MFA_PHONE_TIME":
                raise ValueError(f"unknown model mode: {mode}")
            value = self.input(torch.cat([audio_features, identity, time_features], dim=1))
            for block in self.blocks:
                value = block(value)
            return self.max_gain_db * torch.tanh(self.output(value))

    model = MFAConditionedGainEnhancer()
    # SIL and UNK are appended by PhoneVocabulary.  The caller fixes their
    # exact IDs after construction; this helper keeps model construction pure.
    return model


def zero_initial_output(model: Any, features: Any, phone_ids: Any, timing: Any, mode: str) -> bool:
    import torch

    model.eval()
    with torch.no_grad():
        output = model(features, phone_ids, timing, mode=mode)
    return bool(float(output.abs().max().cpu()) == 0.0)


__all__ = ["build_model", "zero_initial_output"]
