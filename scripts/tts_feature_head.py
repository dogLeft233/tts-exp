#!/usr/bin/env python3
"""Feature-domain residual adapter for natural-to-TTS representation experiments.

The module deliberately has no waveform, TFG, SyncNet, or Hugging Face dependency.
It consumes precomputed SSL features and produces an identity-initialized residual
correction.  Dataset and training policy live in ``train_tts_feature_head.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

DEFAULT_INPUT_DIM = 768
DEFAULT_HIDDEN_CHANNELS = 128
DEFAULT_DILATIONS = (1, 2, 4)
DEFAULT_RESIDUAL_SCALE = 0.25


def _check_features(features: Tensor, input_dim: int) -> None:
    if features.ndim != 3 or features.shape[-1] != input_dim:
        raise ValueError(
            f"features must have shape [batch, frames, {input_dim}], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[1] == 0:
        raise ValueError("features must contain at least one frame")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")


def _check_mask(mask: Tensor, features: Tensor) -> None:
    if mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError(
            "mask must have shape [batch, frames] matching features; "
            f"got {tuple(mask.shape)} for {tuple(features.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")


class FeatureResidualBlock(nn.Module):
    """A same-length temporal residual block over feature frames."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.net(x))


class ResidualFeatureAdapter(nn.Module):
    """Small, temporal, identity-initialized feature adapter.

    Parameters
    ----------
    input_dim:
        Dimensionality of the precomputed SSL feature vectors.
    hidden_channels:
        Width of the temporal convolutional body.
    dilations:
        Positive Conv1d dilations.  The convolutions are non-causal and preserve
        the frame count.
    residual_scale:
        Upper scale applied to the bounded residual.  The final projection is
        zero initialized, so a fresh adapter is exactly ``features``.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        dilations: tuple[int, ...] | list[int] = DEFAULT_DILATIONS,
        residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not dilations or any(int(d) <= 0 for d in dilations):
            raise ValueError("dilations must contain positive values")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.input_dim = int(input_dim)
        self.hidden_channels = int(hidden_channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.residual_scale = float(residual_scale)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Conv1d(self.input_dim, self.hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(FeatureResidualBlock(self.hidden_channels, d) for d in self.dilations)
        )
        self.output_projection = nn.Conv1d(self.hidden_channels, self.input_dim, 1)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return an exact-length feature correction.

        ``mask`` marks real frames.  Invalid/padded frames are returned unchanged
        and cannot contribute residuals, which makes variable-length evaluation
        safe without silently treating padding as speech.
        """
        _check_features(features, self.input_dim)
        if mask is not None:
            _check_mask(mask, features)
        stack_features = features
        if mask is not None:
            stack_features = features * mask.unsqueeze(-1).to(features.dtype)
        hidden = self.input_norm(stack_features).transpose(1, 2)
        hidden = self.input_projection(hidden)
        hidden = self.blocks(hidden)
        residual = self.output_projection(hidden).transpose(1, 2)
        residual = self.residual_scale * torch.tanh(residual)
        if mask is not None:
            residual = residual * mask.unsqueeze(-1).to(residual.dtype)
        output = features + residual
        if not torch.isfinite(output).all():
            raise FloatingPointError("adapter output contains NaN or Inf")
        return output

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config(self) -> dict[str, Any]:
        """Return a JSON-serializable model configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_channels": self.hidden_channels,
            "dilations": list(self.dilations),
            "residual_scale": self.residual_scale,
            "parameter_count": self.parameter_count,
            "identity_initialized": True,
        }


def feature_statistics(values: Tensor | np.ndarray, mask: Tensor | np.ndarray | None = None) -> dict[str, Any]:
    """Collect finite debug statistics without serializing feature arrays."""
    array: np.ndarray | None = None
    tensor: Tensor | None = None
    finite_array: np.ndarray | None = None
    finite_tensor: Tensor | None = None
    if isinstance(values, np.ndarray):
        array = values.astype(np.float64, copy=False)
        finite_array = np.isfinite(array)
        selected_array = array[finite_array]
        selected: np.ndarray | Tensor = selected_array
    else:
        tensor = values.detach().to(dtype=torch.float64)
        finite_tensor = torch.isfinite(tensor)
        selected_tensor = tensor[finite_tensor]
        selected = selected_tensor
    if mask is not None:
        if isinstance(mask, np.ndarray):
            if array is None or finite_array is None:
                raise TypeError("numpy mask requires numpy values")
            mask_array = mask.astype(bool, copy=False)
            if array.ndim == 3:
                expanded = np.broadcast_to(mask_array[..., None], array.shape)
                selected = array[expanded & finite_array]
            else:
                selected = array[mask_array & finite_array]
        else:
            if tensor is None or finite_tensor is None:
                raise TypeError("torch mask requires torch values")
            mask_tensor = mask.to(dtype=torch.bool)
            if tensor.ndim == 3:
                expanded = mask_tensor.unsqueeze(-1).expand_as(tensor)
                selected = tensor[expanded & finite_tensor]
            else:
                selected = tensor[mask_tensor & finite_tensor]
    count = int(np.prod(values.shape))
    finite_count = int(selected.numel() if isinstance(selected, Tensor) else selected.size)
    invalid_count = count - finite_count
    if finite_count == 0:
        return {
            "count": count,
            "finite_count": 0,
            "nan_count": 0,
            "inf_count": invalid_count,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    if isinstance(selected, Tensor):
        assert tensor is not None
        mean = float(selected.mean().item())
        std = float(selected.std(unbiased=False).item())
        minimum = float(selected.min().item())
        maximum = float(selected.max().item())
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
    else:
        assert array is not None
        mean = float(np.mean(selected))
        std = float(np.std(selected))
        minimum = float(np.min(selected))
        maximum = float(np.max(selected))
        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
    return {
        "count": count,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "invalid_count": invalid_count,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
    }


def parameter_statistics(model: nn.Module) -> dict[str, float | int]:
    """Return finite parameter and gradient norms for debug logs."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {"parameter_count": 0, "parameter_norm": 0.0, "gradient_norm": 0.0}
    parameter_norm_sq = sum((torch.sum(p.detach() ** 2) for p in params), torch.zeros((), device=params[0].device))
    parameter_norm = torch.sqrt(parameter_norm_sq)
    grad_terms = [torch.sum(p.grad.detach() ** 2) for p in params if p.grad is not None]
    gradient_norm_sq = sum((term for term in grad_terms), torch.zeros((), device=params[0].device))
    gradient_norm = torch.sqrt(gradient_norm_sq)
    result = {
        "parameter_count": int(sum(p.numel() for p in params)),
        "parameter_norm": float(parameter_norm.item()),
        "gradient_norm": float(gradient_norm.item()),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError("parameter debug statistics are not finite")
    return result


__all__ = [
    "DEFAULT_DILATIONS",
    "DEFAULT_HIDDEN_CHANNELS",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_RESIDUAL_SCALE",
    "FeatureResidualBlock",
    "ResidualFeatureAdapter",
    "feature_statistics",
    "parameter_statistics",
]
