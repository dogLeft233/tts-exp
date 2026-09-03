"""The single compact masked-natural reconstruction network and loss."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .config import (
    CONTEXT_DILATIONS,
    HIDDEN_WIDTH,
    KERNEL_SIZE,
    MAX_PARAMETERS,
    TTS_DILATIONS,
    VELOCITY_WEIGHT,
    WAVLM_DIM,
)
from .protocol import validate_batch_fields


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, KERNEL_SIZE, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.norm(self.conv(x)))


class MaskedNaturalReconstructor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.natural_projection = nn.Conv1d(82, HIDDEN_WIDTH, 1)
        self.natural_blocks = nn.Sequential(*(ResidualTCNBlock(HIDDEN_WIDTH, d) for d in CONTEXT_DILATIONS))
        self.tts_projection = nn.Conv1d(WAVLM_DIM, HIDDEN_WIDTH, 1)
        self.tts_blocks = nn.Sequential(*(ResidualTCNBlock(HIDDEN_WIDTH, d) for d in TTS_DILATIONS))
        self.fusion_projection = nn.Conv1d(2 * HIDDEN_WIDTH, HIDDEN_WIDTH, 1)
        self.output_projection = nn.Conv1d(HIDDEN_WIDTH, 80, 1)
        if self.parameter_count >= MAX_PARAMETERS:
            raise ValueError(f"model has too many trainable parameters: {self.parameter_count}")

    def forward(
        self,
        natural_mel: Tensor,
        masked_support: Tensor,
        target_core: Tensor,
        tts_features: Tensor,
    ) -> Tensor:
        _validate_tensors(natural_mel, masked_support, target_core, tts_features)
        natural = torch.cat((natural_mel, masked_support, target_core), dim=-1).transpose(1, 2)
        tts = tts_features.transpose(1, 2)
        natural_hidden = self.natural_blocks(self.natural_projection(natural))
        tts_hidden = self.tts_blocks(self.tts_projection(tts))
        fused = torch.cat((natural_hidden, tts_hidden), dim=1)
        output = self.output_projection(torch.nn.functional.gelu(self.fusion_projection(fused)))
        output = output.transpose(1, 2)
        if not torch.isfinite(output).all():
            raise FloatingPointError("reconstructor output is not finite")
        return output

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "natural_input_dim": 82,
            "tts_input_dim": WAVLM_DIM,
            "hidden_width": HIDDEN_WIDTH,
            "kernel_size": KERNEL_SIZE,
            "context_dilations": list(CONTEXT_DILATIONS),
            "tts_dilations": list(TTS_DILATIONS),
            "parameter_count": self.parameter_count,
            "attention": False,
            "recurrence": False,
            "waveform_decoder": False,
        }


def _validate_tensors(natural_mel: Tensor, masked_support: Tensor, target_core: Tensor, tts_features: Tensor) -> None:
    if natural_mel.ndim != 3 or natural_mel.shape[1:] != (96, 80):
        raise ValueError(f"natural_mel must have shape [B,96,80], got {tuple(natural_mel.shape)}")
    if masked_support.shape != (natural_mel.shape[0], 96, 1) or target_core.shape != masked_support.shape:
        raise ValueError("mask channels must have shape [B,96,1]")
    if tts_features.shape != (natural_mel.shape[0], 96, WAVLM_DIM):
        raise ValueError(f"tts_features must have shape [B,96,{WAVLM_DIM}]")
    for value in (natural_mel, masked_support, target_core, tts_features):
        if not torch.isfinite(value).all():
            raise ValueError("model inputs must be finite")


def forward_batch(model: MaskedNaturalReconstructor, batch: Mapping[str, Tensor]) -> Tensor:
    validate_batch_fields(batch)
    return model(batch["natural_mel"], batch["masked_support"], batch["target_core"], batch["tts_features"])


def reconstruction_loss(prediction: Tensor, target: Tensor, target_core: Tensor) -> dict[str, Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (96, 80):
        raise ValueError("prediction and target must both have shape [B,96,80]")
    if target_core.shape != (prediction.shape[0], 96, 1):
        raise ValueError("target_core must have shape [B,96,1]")
    core = target_core[..., 0].to(dtype=torch.bool)
    if not core.any():
        raise ValueError("target core must contain at least one frame")
    frame_mask = core.unsqueeze(-1).expand_as(prediction)
    patch = torch.abs(prediction - target)[frame_mask].mean()
    endpoint_mask = core[:, 1:] & core[:, :-1]
    predicted_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    if endpoint_mask.any():
        velocity = torch.abs(predicted_delta - target_delta)[endpoint_mask.unsqueeze(-1).expand_as(predicted_delta)].mean()
    else:
        velocity = prediction.sum() * 0.0
    total = patch + VELOCITY_WEIGHT * velocity
    result = {"patch": patch, "velocity": velocity, "total": total}
    if not all(torch.isfinite(value) for value in result.values()):
        raise FloatingPointError("reconstruction loss is not finite")
    return result
