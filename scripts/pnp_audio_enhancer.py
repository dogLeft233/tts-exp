#!/usr/bin/env python3
"""A small, waveform-only, identity-initialized residual enhancer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

TARGET_SAMPLE_RATE = 16_000
DEFAULT_RESIDUAL_SCALE = 0.2


class ResidualBlock(nn.Module):
    """A same-length dilated temporal residual block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.net(x))


class ResidualTCN(nn.Module):
    """Non-causal, exact-length residual temporal convolutional network.

    The output projection is initialized to zero, making a fresh model an
    exact identity map regardless of the random initialization of the body.
    """

    def __init__(
        self,
        channels: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 8 != 0:
            raise ValueError("channels must be a positive multiple of 8")
        if not dilations or any(d <= 0 for d in dilations):
            raise ValueError("dilations must contain positive values")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = float(residual_scale)
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(channels, int(dilation)) for dilation in dilations)
        )
        self.output_projection = nn.Conv1d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, waveform: Tensor) -> Tensor:
        """Enhance ``[batch, 1, samples]`` while preserving its shape."""
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [batch, 1, samples]")
        if waveform.shape[-1] == 0:
            raise ValueError("waveform must contain at least one sample")
        hidden = self.input_projection(waveform)
        hidden = self.blocks(hidden)
        residual = self.output_projection(hidden)
        return waveform + self.residual_scale * torch.tanh(residual)


@dataclass
class WaveformEnhancer:
    """NumPy-facing inference wrapper for :class:`ResidualTCN`."""

    model: ResidualTCN
    sample_rate: int = TARGET_SAMPLE_RATE
    chunk_seconds: float = 4.0
    stride_seconds: float = 2.0
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {TARGET_SAMPLE_RATE}")
        if self.chunk_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("chunk_seconds and stride_seconds must be positive")
        if self.stride_seconds > self.chunk_seconds:
            raise ValueError("stride_seconds cannot exceed chunk_seconds")
        self.device = torch.device(self.device)
        self.model.to(self.device)
        self.model.eval()

    @property
    def chunk_samples(self) -> int:
        return max(1, int(round(self.chunk_seconds * self.sample_rate)))

    @property
    def stride_samples(self) -> int:
        return max(1, int(round(self.stride_seconds * self.sample_rate)))

    def enhance(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """Enhance mono 16 kHz audio and return exactly the input length."""
        requested_sr = self.sample_rate if sample_rate is None else sample_rate
        if requested_sr != self.sample_rate:
            raise ValueError(f"sample_rate must be {self.sample_rate}")
        waveform = np.asarray(audio)
        if waveform.ndim != 1:
            raise ValueError("audio must be a mono one-dimensional array")
        if waveform.size == 0:
            raise ValueError("audio must contain at least one sample")
        if not np.issubdtype(waveform.dtype, np.number):
            raise TypeError("audio must be numeric")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("audio must contain only finite values")
        source = waveform.astype(np.float32, copy=False)
        with torch.no_grad():
            output = self._enhance_tensor(torch.from_numpy(source).to(self.device))
        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    def _enhance_tensor(self, waveform: Tensor) -> Tensor:
        n_samples = waveform.numel()
        chunk = self.chunk_samples
        if n_samples <= chunk:
            padded = _reflect_pad_to_length(waveform, chunk)
            enhanced = self.model(padded[None, None, :])[0, 0, :n_samples]
            return waveform + (enhanced - padded[:n_samples])

        stride = self.stride_samples
        residual = torch.zeros(n_samples, device=waveform.device, dtype=torch.float32)
        weights = torch.zeros_like(residual)
        window = torch.hann_window(chunk, periodic=False, device=waveform.device)
        window = window.clamp_min(1e-3)
        starts = list(range(0, max(1, n_samples - chunk + 1), stride))
        final_start = n_samples - chunk
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            end = start + chunk
            segment = waveform[start:end]
            if segment.numel() < chunk:
                segment = _reflect_pad_to_length(segment, chunk)
            enhanced = self.model(segment[None, None, :])[0, 0]
            residual[start:end] += (enhanced - segment) * window
            weights[start:end] += window
        return waveform + residual / weights.clamp_min(1e-6)


def _reflect_pad_to_length(waveform: Tensor, target_length: int) -> Tensor:
    """Pad a one-dimensional tensor by reflection, including very short input."""
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if waveform.numel() >= target_length:
        return waveform[:target_length]
    if waveform.numel() == 1:
        return waveform.repeat(target_length)
    pieces = [waveform]
    remaining = target_length - waveform.numel()
    reflected = waveform.flip(0)
    while remaining > 0:
        piece = reflected[:remaining]
        pieces.append(piece)
        remaining -= piece.numel()
        reflected = reflected.flip(0)
    return torch.cat(pieces, dim=0)[:target_length]


def enhance(
    audio: np.ndarray,
    model: ResidualTCN | None = None,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Convenience function for the public waveform-only API."""
    wrapper = WaveformEnhancer(
        model=model or ResidualTCN(),
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        stride_seconds=stride_seconds,
        device=device,
    )
    return wrapper.enhance(audio, sample_rate=sample_rate)
