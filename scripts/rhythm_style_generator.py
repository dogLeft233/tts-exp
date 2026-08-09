#!/usr/bin/env python3
"""Compact dual-input conditional waveform generator for the local MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GeneratorConfig:
    content_dim: int = 768
    rhythm_dim: int = 6
    style_input_dim: int = 84
    style_dim: int = 64
    content_proj_dim: int = 128
    rhythm_proj_dim: int = 32
    style_proj_dim: int = 32
    context_dim: int = 192
    context_heads: int = 4
    context_layers: int = 4
    context_ff_dim: int = 384
    channels: int = 64
    residual_scale: float = 0.2


class StyleProsodyEncoder(nn.Module):
    """Encode variable-length TTS mel/prosody frames into a joint style vector."""

    def __init__(self, input_dim: int = 84, style_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.projection = nn.Linear(128, style_dim)

    def forward(self, features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [B,T,{self.input_dim}]")
        encoded = self.net(features.transpose(1, 2)).transpose(1, 2)
        if mask is None:
            encoded_mask = torch.ones(
                encoded.shape[:2], dtype=torch.bool, device=encoded.device
            )
        else:
            if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("style mask must be bool with shape [B,T]")
            encoded_mask = F.interpolate(
                mask.float().unsqueeze(1), size=encoded.shape[1], mode="nearest"
            ).squeeze(1).bool()
        weights = encoded_mask.unsqueeze(-1).to(encoded.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weights).sum(dim=1) / count
        variance = ((encoded - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
        pooled = torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=-1)
        return self.projection(pooled)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.net(x))


class RhythmStyleGenerator(nn.Module):
    """Natural-content residual generator conditioned on TTS style and rhythm."""

    def __init__(self, config: GeneratorConfig = GeneratorConfig()) -> None:
        super().__init__()
        if config.context_dim != config.content_proj_dim + config.rhythm_proj_dim + config.style_proj_dim:
            raise ValueError("context_dim must equal the concatenated projection dimensions")
        if config.channels % 8:
            raise ValueError("channels must be divisible by 8")
        self.config = config
        self.style_encoder = StyleProsodyEncoder(config.style_input_dim, config.style_dim)
        self.content_projection = nn.Linear(config.content_dim, config.content_proj_dim)
        self.rhythm_projection = nn.Linear(config.rhythm_dim, config.rhythm_proj_dim)
        self.style_projection = nn.Linear(config.style_dim, config.style_proj_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.context_dim,
            nhead=config.context_heads,
            dim_feedforward=config.context_ff_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=config.context_layers)
        self.waveform_projection = nn.Conv1d(1, config.channels, kernel_size=1)
        self.context_projection = nn.Conv1d(config.context_dim, config.channels, kernel_size=1)
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(config.channels, dilation) for dilation in (1, 2, 4, 8, 16, 32, 64, 128))
        )
        self.output_projection = nn.Conv1d(config.channels, 1, kernel_size=1)
        self.content_head = nn.Linear(config.context_dim, config.content_dim)
        self.rhythm_head = nn.Linear(config.context_dim, config.rhythm_dim)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        natural_waveform: Tensor,
        natural_content: Tensor,
        rhythm: Tensor,
        style_features: Tensor,
        *,
        frame_mask: Optional[Tensor] = None,
        style_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if natural_waveform.ndim != 3 or natural_waveform.shape[1] != 1:
            raise ValueError("natural_waveform must have shape [B,1,N]")
        if natural_content.ndim != 3 or natural_content.shape[-1] != self.config.content_dim:
            raise ValueError("natural_content has an invalid shape")
        if rhythm.ndim != 3 or rhythm.shape[:2] != natural_content.shape[:2] or rhythm.shape[-1] != self.config.rhythm_dim:
            raise ValueError("rhythm must have shape [B,F,rhythm_dim]")
        if natural_content.shape[0] != natural_waveform.shape[0] or style_features.shape[0] != natural_waveform.shape[0]:
            raise ValueError("all inputs must have the same batch size")
        if frame_mask is None:
            frame_mask = torch.ones(
                natural_content.shape[:2], dtype=torch.bool, device=natural_content.device
            )
        if frame_mask.shape != natural_content.shape[:2] or frame_mask.dtype != torch.bool:
            raise ValueError("frame_mask must be bool with shape [B,F]")
        style = self.style_encoder(style_features, style_mask)
        context_input = torch.cat(
            [
                self.content_projection(natural_content),
                self.rhythm_projection(rhythm),
                self.style_projection(style).unsqueeze(1).expand(-1, natural_content.shape[1], -1),
            ],
            dim=-1,
        )
        context = self.context_encoder(context_input, src_key_padding_mask=~frame_mask)
        context = context * frame_mask.unsqueeze(-1).to(context.dtype)
        n_samples = natural_waveform.shape[-1]
        sample_context = F.interpolate(
            context.transpose(1, 2), size=n_samples, mode="linear", align_corners=False
        )
        hidden = self.waveform_projection(natural_waveform) + self.context_projection(sample_context)
        hidden = self.residual_blocks(hidden)
        residual = self.config.residual_scale * torch.tanh(self.output_projection(hidden))
        output = natural_waveform + residual
        return {
            "waveform": output,
            "residual": residual,
            "style": style,
            "context": context,
            "content_prediction": self.content_head(context),
            "rhythm_prediction": self.rhythm_head(context),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
