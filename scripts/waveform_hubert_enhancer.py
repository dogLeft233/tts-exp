#!/usr/bin/env python3
"""Single-encoder HuBERT-conditioned residual waveform enhancer.

The local training encoder is HuggingFace HuBERT Base (768-D).  Its features
are used only to condition this waveform model and to provide a differentiable
perceptual loss.  The output is always a waveform; Ditto later re-encodes it
with its own native 1024-D frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

TARGET_SAMPLE_RATE = 16_000
DEFAULT_MODEL_ID = "facebook/hubert-base-ls960"
DEFAULT_LAYER = 6
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_FRAME_STRIDE = 320
DEFAULT_CONDITIONING_DIM = 64
DEFAULT_RESIDUAL_SCALE = 0.2


def _finite_tensor(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or infinite values")
    return value


def _extract_hidden_states(output: Any) -> tuple[Tensor, ...]:
    hidden = getattr(output, "hidden_states", None)
    if hidden is None and isinstance(output, dict):
        hidden = output.get("hidden_states")
    if hidden is None or not isinstance(hidden, (tuple, list)):
        raise ValueError("HuBERT output must expose hidden_states")
    return tuple(hidden)


class ResidualBlock(nn.Module):
    """Same-length non-causal residual temporal block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8")
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


class ConditionedResidualTCN(nn.Module):
    """Identity-initialized residual TCN conditioned by frame features."""

    def __init__(
        self,
        conditioning_dim: int = DEFAULT_CONDITIONING_DIM,
        channels: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        if conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be positive")
        if channels <= 0 or channels % 8:
            raise ValueError("channels must be a positive multiple of 8")
        if not dilations or any(int(d) <= 0 for d in dilations):
            raise ValueError("dilations must contain positive values")
        if not np.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        self.conditioning_dim = int(conditioning_dim)
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.residual_scale = float(residual_scale)
        self.input_projection = nn.Conv1d(1 + conditioning_dim, channels, 1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(channels, dilation) for dilation in self.dilations)
        )
        self.output_projection = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, waveform: Tensor, conditioning: Tensor) -> Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [B,1,N]")
        if conditioning.ndim != 3:
            raise ValueError("conditioning must have shape [B,C,N]")
        if conditioning.shape[0] != waveform.shape[0] or conditioning.shape[-1] != waveform.shape[-1]:
            raise ValueError("conditioning batch and sample dimensions must match waveform")
        if conditioning.shape[1] != self.conditioning_dim:
            raise ValueError("conditioning channel dimension does not match model")
        _finite_tensor(waveform, "waveform")
        _finite_tensor(conditioning, "conditioning")
        hidden = self.input_projection(torch.cat([waveform, conditioning], dim=1))
        hidden = self.blocks(hidden)
        residual = self.residual_scale * torch.tanh(self.output_projection(hidden))
        output = waveform + residual
        _finite_tensor(output, "enhanced waveform")
        if torch.max(torch.abs(residual.detach())) > self.residual_scale + 1e-5:
            raise FloatingPointError("residual exceeded configured bound")
        return output


@dataclass(frozen=True)
class EncoderInterface:
    """Auditable contract for the single local HuBERT teacher."""

    model_id: str = DEFAULT_MODEL_ID
    selected_layer: int = DEFAULT_LAYER
    content_layer: int = 2
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    frame_stride_samples: int = DEFAULT_FRAME_STRIDE
    sample_rate: int = TARGET_SAMPLE_RATE
    conditioning_dim: int = DEFAULT_CONDITIONING_DIM
    frozen: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "selected_layer": self.selected_layer,
            "content_layer": self.content_layer,
            "hidden_size": self.hidden_size,
            "frame_stride_samples": self.frame_stride_samples,
            "sample_rate": self.sample_rate,
            "conditioning_dim": self.conditioning_dim,
            "frozen": self.frozen,
            "ditto_native_interface": False,
        }


class HuBERTWaveformEnhancer(nn.Module):
    """Waveform enhancer using one frozen HuBERT module with shared weights.

    ``compute_enhanced_features=True`` deliberately retains the computation
    graph through the frozen encoder input.  Encoder parameters stay frozen,
    while HuBERT feature loss can still update the waveform enhancer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        interface: EncoderInterface = EncoderInterface(),
        channels: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        if not interface.frozen:
            raise ValueError("the HuBERT teacher must be frozen")
        if interface.hidden_size != DEFAULT_HIDDEN_SIZE:
            raise ValueError("this experiment requires 768-D HF HuBERT features")
        if interface.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("this experiment requires 16 kHz audio")
        if interface.frame_stride_samples != DEFAULT_FRAME_STRIDE:
            raise ValueError("unexpected HuBERT frame stride")
        if interface.model_id == "ditto_native" or interface.hidden_size == 1024:
            raise ValueError("Ditto native 1024-D features are not this trainer's encoder")
        self.encoder = encoder
        self.interface = interface
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.condition_projection = nn.Linear(interface.hidden_size, interface.conditioning_dim)
        self.waveform_model = ConditionedResidualTCN(
            conditioning_dim=interface.conditioning_dim,
            channels=channels,
            dilations=dilations,
            residual_scale=residual_scale,
        )

    def train(self, mode: bool = True) -> "HuBERTWaveformEnhancer":
        super().train(mode)
        # The teacher must remain in eval mode even while the enhancer trains.
        self.encoder.eval()
        return self

    def encode_layers(
        self,
        waveform: Tensor,
        layers: Sequence[int],
        *,
        retain_gradient: bool = False,
    ) -> dict[int, Tensor]:
        """Encode several HuBERT layers with exactly one encoder invocation."""
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [B,1,N]")
        if not layers:
            raise ValueError("layers must be non-empty")
        requested = tuple(int(layer) for layer in layers)
        if len(set(requested)) != len(requested) or any(layer < 0 for layer in requested):
            raise ValueError("layers must contain distinct non-negative indices")
        _finite_tensor(waveform, "encoder input")
        signal = waveform[:, 0, :]
        if retain_gradient:
            output = self.encoder(signal, output_hidden_states=True)
        else:
            with torch.no_grad():
                output = self.encoder(signal, output_hidden_states=True)
        hidden_states = _extract_hidden_states(output)
        result: dict[int, Tensor] = {}
        for layer in requested:
            if layer >= len(hidden_states):
                raise ValueError(f"HuBERT layer {layer} is unavailable")
            features = hidden_states[layer]
            if (
                features.ndim != 3
                or features.shape[0] != waveform.shape[0]
                or features.shape[-1] != self.interface.hidden_size
            ):
                raise ValueError("HuBERT hidden state must have shape [B,F,D]")
            result[layer] = _finite_tensor(features, f"HuBERT layer {layer} features")
        return result

    def _encode_layer(self, waveform: Tensor, layer: int, *, retain_gradient: bool) -> Tensor:
        return self.encode_layers(
            waveform,
            (int(layer),),
            retain_gradient=retain_gradient,
        )[int(layer)]

    def encode_layer(self, waveform: Tensor, layer: int, *, retain_gradient: bool = False) -> Tensor:
        """Encode a chosen frozen HuBERT layer with explicit gradient policy."""
        return self._encode_layer(waveform, int(layer), retain_gradient=retain_gradient)

    def _encode(self, waveform: Tensor, *, retain_gradient: bool) -> Tensor:
        return self._encode_layer(
            waveform, self.interface.selected_layer, retain_gradient=retain_gradient
        )

    def _conditioning(self, natural_features: Tensor, samples: int) -> Tensor:
        projected = self.condition_projection(natural_features)
        conditioning = F.interpolate(
            projected.transpose(1, 2), size=samples, mode="linear", align_corners=False
        )
        return _finite_tensor(conditioning, "waveform conditioning")

    def forward(
        self,
        natural_waveform: Tensor,
        *,
        compute_enhanced_features: bool = True,
    ) -> dict[str, Tensor]:
        if natural_waveform.ndim != 3 or natural_waveform.shape[1] != 1:
            raise ValueError("natural_waveform must have shape [B,1,N]")
        _finite_tensor(natural_waveform, "natural waveform")
        natural_features = self._encode(natural_waveform, retain_gradient=False)
        conditioning = self._conditioning(natural_features, natural_waveform.shape[-1])
        enhanced = self.waveform_model(natural_waveform, conditioning)
        result: dict[str, Tensor] = {
            "waveform": enhanced,
            "natural_features": natural_features.detach(),
            "natural_content_features": self.encode_layer(
                natural_waveform, self.interface.content_layer, retain_gradient=False
            ).detach(),
            "conditioning": conditioning,
            "residual": enhanced - natural_waveform,
        }
        if compute_enhanced_features:
            result["enhanced_features"] = self._encode(enhanced, retain_gradient=True)
            result["enhanced_content_features"] = self.encode_layer(
                enhanced, self.interface.content_layer, retain_gradient=True
            )
        return result

    def model_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "condition_projection": {
                "input_dim": self.interface.hidden_size,
                "output_dim": self.interface.conditioning_dim,
            },
            "waveform_model": {
                "class": self.waveform_model.__class__.__name__,
                "channels": self.waveform_model.channels,
                "dilations": list(self.waveform_model.dilations),
                "residual_scale": self.waveform_model.residual_scale,
            },
            "encoder_calls": {
                "natural_conditioning": "no_grad",
                "enhanced_feature_loss": "input_gradient_retained",
                "style_layer": self.interface.selected_layer,
                "content_layer": self.interface.content_layer,
            },
        }


def residual_stats(residual: Tensor) -> dict[str, float | bool]:
    """Return finite JSON-friendly diagnostics for a residual tensor."""
    _finite_tensor(residual, "residual")
    values = residual.detach().float()
    return {
        "rms": float(torch.sqrt(torch.mean(values.square())).cpu()),
        "mean_abs": float(torch.mean(values.abs()).cpu()),
        "peak": float(torch.max(values.abs()).cpu()),
        "finite": True,
    }


@dataclass
class WaveformEnhancer:
    """NumPy-facing exact-length inference wrapper."""

    model: HuBERTWaveformEnhancer
    chunk_seconds: float = 4.0
    stride_seconds: float = 2.0
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        self.chunk_seconds = float(self.chunk_seconds)
        self.stride_seconds = float(self.stride_seconds)
        if self.chunk_seconds < 0.064 or self.stride_seconds < 0.064:
            raise ValueError("chunk_seconds and stride_seconds must provide HuBERT context")
        if self.stride_seconds > self.chunk_seconds:
            raise ValueError("stride_seconds cannot exceed chunk_seconds")
        self.device = torch.device(self.device)
        self.model.to(self.device)
        self.model.eval()

    @property
    def chunk_samples(self) -> int:
        return max(1, int(round(self.chunk_seconds * TARGET_SAMPLE_RATE)))

    @property
    def stride_samples(self) -> int:
        return max(1, int(round(self.stride_seconds * TARGET_SAMPLE_RATE)))

    def enhance(self, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {TARGET_SAMPLE_RATE}")
        values = np.asarray(audio)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("audio must be a non-empty mono vector")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError("audio must contain finite numeric values")
        source = torch.from_numpy(values.astype(np.float32, copy=False)).to(self.device)
        with torch.no_grad():
            result = self._enhance_tensor(source)
        output = result.detach().cpu().numpy().astype(np.float32, copy=False)
        if output.shape != values.shape or not np.isfinite(output).all():
            raise FloatingPointError("enhancer returned an invalid output shape or value")
        return output

    def _enhance_tensor(self, waveform: Tensor) -> Tensor:
        n_samples = waveform.numel()
        chunk = self.chunk_samples
        if n_samples <= chunk:
            padded = _reflect_pad(waveform, chunk)
            output = self.model(padded[None, None, :], compute_enhanced_features=False)["waveform"]
            return output[0, 0, :n_samples]
        stride = self.stride_samples
        residual = torch.zeros(n_samples, device=waveform.device)
        weights = torch.zeros_like(residual)
        window = torch.hann_window(chunk, periodic=False, device=waveform.device).clamp_min(1e-3)
        starts = list(range(0, n_samples - chunk + 1, stride))
        final_start = n_samples - chunk
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            segment = waveform[start : start + chunk]
            output = self.model(segment[None, None, :], compute_enhanced_features=False)["waveform"][0, 0]
            residual[start : start + chunk] += (output - segment) * window
            weights[start : start + chunk] += window
        return waveform + residual / weights.clamp_min(1e-6)


def _reflect_pad(waveform: Tensor, target_length: int) -> Tensor:
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("waveform must be a non-empty vector")
    if waveform.numel() >= target_length:
        return waveform[:target_length]
    if waveform.numel() == 1:
        return waveform.repeat(target_length)
    parts = [waveform]
    remaining = target_length - waveform.numel()
    reflected = waveform.flip(0)
    while remaining:
        part = reflected[:remaining]
        parts.append(part)
        remaining -= part.numel()
        reflected = reflected.flip(0)
    return torch.cat(parts)[:target_length]


__all__ = [
    "DEFAULT_FRAME_STRIDE",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_LAYER",
    "DEFAULT_MODEL_ID",
    "TARGET_SAMPLE_RATE",
    "EncoderInterface",
    "ConditionedResidualTCN",
    "HuBERTWaveformEnhancer",
    "WaveformEnhancer",
    "residual_stats",
]
