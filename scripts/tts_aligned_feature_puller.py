#!/usr/bin/env python3
"""Bounded Stage-1 puller for raw-TTS-aligned HuBERT layer-6 goals.

This module is feature-domain only.  It accepts natural-speech HuBERT layers 2
and 6, produces a normalized layer-6 residual, and never consumes a weak
waveform target, phone label, MFA span, or TTS feature at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tts_feature_head import FeatureResidualBlock
from waveform_hubert_enhancer import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    TARGET_SAMPLE_RATE,
)

DEFAULT_CONTENT_LAYER = 2
DEFAULT_INPUT_DIM = DEFAULT_HIDDEN_SIZE
DEFAULT_HIDDEN_CHANNELS = 128
DEFAULT_DILATIONS = (1, 2, 4)
DEFAULT_BOUND_QUANTILE = 0.95
DEFAULT_STANDARD_DEVIATION_FLOOR = 1e-6


def _finite(value: Tensor, name: str) -> Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or infinite values")
    return value


def _check_features(value: Tensor, name: str, input_dim: int) -> None:
    if value.ndim != 3 or value.shape[-1] != input_dim:
        raise ValueError(
            f"{name} must have shape [batch, frames, {input_dim}], "
            f"got {tuple(value.shape)}"
        )
    if value.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    _finite(value, name)


def _check_mask(mask: Tensor, reference: Tensor, name: str = "padding_mask") -> None:
    if mask.ndim != 2 or tuple(mask.shape) != tuple(reference.shape[:2]):
        raise ValueError(
            f"{name} must have shape [batch, frames] matching features; "
            f"got {tuple(mask.shape)} for {tuple(reference.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")


def _check_inputs(
    natural_layer2: Tensor,
    natural_layer6: Tensor,
    padding_mask: Tensor,
    input_dim: int,
) -> None:
    _check_features(natural_layer2, "natural_layer2", input_dim)
    _check_features(natural_layer6, "natural_layer6", input_dim)
    if tuple(natural_layer2.shape[:2]) != tuple(natural_layer6.shape[:2]):
        raise ValueError("natural_layer2 and natural_layer6 must share batch and frame dimensions")
    _check_mask(padding_mask, natural_layer6)


def _selected_frames(value: Tensor, mask: Tensor, name: str) -> Tensor:
    _check_features(value, name, value.shape[-1])
    _check_mask(mask, value)
    selected = value[mask]
    if selected.numel() == 0:
        raise ValueError(f"{name} requires at least one selected frame")
    return selected


@dataclass(frozen=True)
class FeatureNormalizer:
    """Frozen per-dimension normalizer fitted only to natural train frames."""

    layer2_mean: Tensor
    layer2_std: Tensor
    layer6_mean: Tensor
    layer6_std: Tensor
    source_frame_count: int
    standard_deviation_floor: float = DEFAULT_STANDARD_DEVIATION_FLOOR

    def __post_init__(self) -> None:
        tensors = {
            "layer2_mean": self.layer2_mean,
            "layer2_std": self.layer2_std,
            "layer6_mean": self.layer6_mean,
            "layer6_std": self.layer6_std,
        }
        expected = (DEFAULT_INPUT_DIM,)
        for name, value in tensors.items():
            if not isinstance(value, Tensor) or tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            _finite(value, name)
        if torch.any(self.layer2_std <= 0) or torch.any(self.layer6_std <= 0):
            raise ValueError("normalizer standard deviations must be positive")
        if self.source_frame_count <= 0:
            raise ValueError("source_frame_count must be positive")
        if not np.isfinite(self.standard_deviation_floor) or self.standard_deviation_floor <= 0:
            raise ValueError("standard_deviation_floor must be finite and positive")

    @property
    def input_dim(self) -> int:
        return int(self.layer2_mean.numel())

    def normalize_layer2(self, value: Tensor) -> Tensor:
        _check_features(value, "layer2", self.input_dim)
        return (value - self.layer2_mean.to(value)) / self.layer2_std.to(value)

    def normalize_layer6(self, value: Tensor) -> Tensor:
        _check_features(value, "layer6", self.input_dim)
        return (value - self.layer6_mean.to(value)) / self.layer6_std.to(value)

    def denormalize_layer6(self, value: Tensor) -> Tensor:
        _check_features(value, "normalized_layer6", self.input_dim)
        return value * self.layer6_std.to(value) + self.layer6_mean.to(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "layer2_mean": self.layer2_mean.detach().cpu().tolist(),
            "layer2_std": self.layer2_std.detach().cpu().tolist(),
            "layer6_mean": self.layer6_mean.detach().cpu().tolist(),
            "layer6_std": self.layer6_std.detach().cpu().tolist(),
            "source_frame_count": int(self.source_frame_count),
            "standard_deviation_floor": float(self.standard_deviation_floor),
            "fit_scope": "natural_train_real_frames_only",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureNormalizer":
        required = {
            "input_dim",
            "layer2_mean",
            "layer2_std",
            "layer6_mean",
            "layer6_std",
            "source_frame_count",
            "standard_deviation_floor",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"normalizer is missing fields: {sorted(missing)}")
        if int(payload["input_dim"]) != DEFAULT_INPUT_DIM:
            raise ValueError("normalizer input_dim is incompatible with HF HuBERT Base")
        return cls(
            layer2_mean=torch.tensor(payload["layer2_mean"], dtype=torch.float32),
            layer2_std=torch.tensor(payload["layer2_std"], dtype=torch.float32),
            layer6_mean=torch.tensor(payload["layer6_mean"], dtype=torch.float32),
            layer6_std=torch.tensor(payload["layer6_std"], dtype=torch.float32),
            source_frame_count=int(payload["source_frame_count"]),
            standard_deviation_floor=float(payload["standard_deviation_floor"]),
        )


def fit_feature_normalizer(
    natural_layer2: Tensor,
    natural_layer6: Tensor,
    padding_mask: Tensor,
    *,
    standard_deviation_floor: float = DEFAULT_STANDARD_DEVIATION_FLOOR,
) -> FeatureNormalizer:
    """Fit normalizer statistics from natural train real frames only."""
    _check_inputs(natural_layer2, natural_layer6, padding_mask, DEFAULT_INPUT_DIM)
    if not np.isfinite(standard_deviation_floor) or standard_deviation_floor <= 0:
        raise ValueError("standard_deviation_floor must be finite and positive")
    layer2 = _selected_frames(natural_layer2, padding_mask, "natural_layer2")
    layer6 = _selected_frames(natural_layer6, padding_mask, "natural_layer6")
    layer2 = layer2.detach().to(dtype=torch.float64)
    layer6 = layer6.detach().to(dtype=torch.float64)
    floor = float(standard_deviation_floor)
    return FeatureNormalizer(
        layer2_mean=layer2.mean(dim=0).to(dtype=torch.float32),
        layer2_std=layer2.std(dim=0, unbiased=False).clamp_min(floor).to(dtype=torch.float32),
        layer6_mean=layer6.mean(dim=0).to(dtype=torch.float32),
        layer6_std=layer6.std(dim=0, unbiased=False).clamp_min(floor).to(dtype=torch.float32),
        source_frame_count=int(layer2.shape[0]),
        standard_deviation_floor=floor,
    )


@dataclass(frozen=True)
class ResidualBound:
    """A train-only normalized L6 displacement cap."""

    value: float
    quantile: float
    source_frame_count: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value < 0:
            raise ValueError("residual bound value must be finite and non-negative")
        if not np.isfinite(self.quantile) or not 0 < self.quantile <= 1:
            raise ValueError("residual bound quantile must be in (0, 1]")
        if self.source_frame_count <= 0:
            raise ValueError("residual bound source_frame_count must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_l6_l2_cap": float(self.value),
            "quantile": float(self.quantile),
            "source_frame_count": int(self.source_frame_count),
            "fit_scope": "aligned_natural_to_raw_tts_train_target_frames_only",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualBound":
        return cls(
            value=float(payload["normalized_l6_l2_cap"]),
            quantile=float(payload["quantile"]),
            source_frame_count=int(payload["source_frame_count"]),
        )


def fit_residual_bound(
    natural_layer6: Tensor,
    aligned_tts_layer6: Tensor,
    target_mask: Tensor,
    normalizer: FeatureNormalizer,
    *,
    quantile: float = DEFAULT_BOUND_QUANTILE,
) -> ResidualBound:
    """Fit the hard normalized residual bound from aligned train target frames."""
    _check_features(natural_layer6, "natural_layer6", normalizer.input_dim)
    _check_features(aligned_tts_layer6, "aligned_tts_layer6", normalizer.input_dim)
    if tuple(natural_layer6.shape) != tuple(aligned_tts_layer6.shape):
        raise ValueError("natural_layer6 and aligned_tts_layer6 must have identical shape")
    _check_mask(target_mask, natural_layer6, "target_mask")
    if not np.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    natural = normalizer.normalize_layer6(natural_layer6)
    target = normalizer.normalize_layer6(aligned_tts_layer6)
    displacement = torch.linalg.vector_norm((target - natural)[target_mask], dim=-1)
    if displacement.numel() == 0:
        raise ValueError("residual bound requires at least one aligned target frame")
    _finite(displacement, "train normalized L6 displacement")
    value = float(torch.quantile(displacement.detach().to(dtype=torch.float64), float(quantile)).item())
    return ResidualBound(value=value, quantile=float(quantile), source_frame_count=int(displacement.numel()))


def validate_hubert_interface(interface: Mapping[str, Any]) -> dict[str, Any]:
    """Reject every encoder interface other than local 16 kHz HF HuBERT Base."""
    required = {
        "model_id",
        "selected_layer",
        "content_layer",
        "hidden_size",
        "frame_stride_samples",
        "sample_rate",
        "frozen",
        "ditto_native_interface",
    }
    missing = required.difference(interface)
    if missing:
        raise ValueError(f"HuBERT interface is missing fields: {sorted(missing)}")
    normalized = dict(interface)
    if normalized["model_id"] != DEFAULT_MODEL_ID:
        raise ValueError("Stage 1 requires facebook/hubert-base-ls960")
    if int(normalized["selected_layer"]) != DEFAULT_LAYER:
        raise ValueError("Stage 1 requires HuBERT layer 6 as its target layer")
    if int(normalized["content_layer"]) != DEFAULT_CONTENT_LAYER:
        raise ValueError("Stage 1 requires HuBERT layer 2 as its content layer")
    if int(normalized["hidden_size"]) != DEFAULT_INPUT_DIM:
        raise ValueError("Stage 1 requires 768-D HF HuBERT features")
    if int(normalized["frame_stride_samples"]) != DEFAULT_FRAME_STRIDE:
        raise ValueError("Stage 1 requires a 320-sample HuBERT frame stride")
    if int(normalized["sample_rate"]) != TARGET_SAMPLE_RATE:
        raise ValueError("Stage 1 requires 16 kHz HuBERT input")
    if not bool(normalized["frozen"]):
        raise ValueError("Stage 1 requires a frozen HuBERT encoder")
    if bool(normalized["ditto_native_interface"]):
        raise ValueError("Ditto-native features cannot be used by Stage 1")
    return normalized


class AlignedTTSFeaturePuller(nn.Module):
    """Identity-initialized, bounded natural L2/L6-to-L6 residual puller."""

    def __init__(
        self,
        normalizer: FeatureNormalizer,
        residual_bound: ResidualBound,
        *,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if normalizer.input_dim != DEFAULT_INPUT_DIM:
            raise ValueError("Stage 1 requires 768-D normalizer statistics")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not dilations or any(int(value) <= 0 for value in dilations):
            raise ValueError("dilations must contain positive values")
        self.input_dim = DEFAULT_INPUT_DIM
        self.hidden_channels = int(hidden_channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.residual_bound = residual_bound
        self.register_buffer("layer2_mean", normalizer.layer2_mean.detach().clone())
        self.register_buffer("layer2_std", normalizer.layer2_std.detach().clone())
        self.register_buffer("layer6_mean", normalizer.layer6_mean.detach().clone())
        self.register_buffer("layer6_std", normalizer.layer6_std.detach().clone())
        self.normalizer_source_frame_count = int(normalizer.source_frame_count)
        self.standard_deviation_floor = float(normalizer.standard_deviation_floor)
        self.input_norm = nn.LayerNorm(2 * self.input_dim)
        self.input_projection = nn.Conv1d(2 * self.input_dim, self.hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(FeatureResidualBlock(self.hidden_channels, dilation) for dilation in self.dilations)
        )
        self.output_projection = nn.Conv1d(self.hidden_channels, self.input_dim, 1)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    @property
    def normalizer(self) -> FeatureNormalizer:
        return FeatureNormalizer(
            layer2_mean=self.layer2_mean.detach().clone(),
            layer2_std=self.layer2_std.detach().clone(),
            layer6_mean=self.layer6_mean.detach().clone(),
            layer6_std=self.layer6_std.detach().clone(),
            source_frame_count=self.normalizer_source_frame_count,
            standard_deviation_floor=self.standard_deviation_floor,
        )

    def _bounded_delta(self, raw_delta: Tensor) -> Tensor:
        _finite(raw_delta, "unnormalized delta")
        norms = torch.linalg.vector_norm(raw_delta, dim=-1, keepdim=True)
        cap = torch.as_tensor(self.residual_bound.value, dtype=raw_delta.dtype, device=raw_delta.device)
        scale = torch.clamp(cap / norms.clamp_min(torch.finfo(raw_delta.dtype).eps), max=1.0)
        return raw_delta * scale

    def forward(
        self,
        natural_layer2: Tensor,
        natural_layer6: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Return normalized, per-frame L2-capped layer-6 residuals."""
        _check_inputs(natural_layer2, natural_layer6, padding_mask, self.input_dim)
        frame_mask = padding_mask.unsqueeze(-1).to(dtype=natural_layer2.dtype)
        normalized_l2 = (natural_layer2 - self.layer2_mean) / self.layer2_std
        normalized_l6 = (natural_layer6 - self.layer6_mean) / self.layer6_std
        stacked = torch.cat([normalized_l2, normalized_l6], dim=-1) * frame_mask
        hidden = self.input_norm(stacked).transpose(1, 2)
        hidden = self.input_projection(hidden)
        hidden = self.blocks(hidden)
        raw_delta = self.output_projection(hidden).transpose(1, 2)
        delta = self._bounded_delta(raw_delta) * frame_mask
        _finite(delta, "bounded delta")
        norms = torch.linalg.vector_norm(delta, dim=-1)
        if torch.any(norms > self.residual_bound.value + 1e-5):
            raise FloatingPointError("delta exceeded its configured per-frame L2 bound")
        return delta

    def goal_layer6(self, natural_layer6: Tensor, delta_layer6: Tensor) -> Tensor:
        """Convert normalized delta to the raw HuBERT layer-6 goal exactly once."""
        _check_features(natural_layer6, "natural_layer6", self.input_dim)
        _check_features(delta_layer6, "delta_layer6", self.input_dim)
        if tuple(natural_layer6.shape) != tuple(delta_layer6.shape):
            raise ValueError("natural_layer6 and delta_layer6 must have identical shape")
        _finite(delta_layer6, "delta_layer6")
        goal = natural_layer6 + delta_layer6 * self.layer6_std.to(natural_layer6)
        return _finite(goal, "goal_layer6")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def model_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "input_layers": [DEFAULT_CONTENT_LAYER, DEFAULT_LAYER],
            "target_layer": DEFAULT_LAYER,
            "hidden_channels": self.hidden_channels,
            "dilations": list(self.dilations),
            "parameter_count": self.parameter_count,
            "identity_initialized": True,
            "residual_space": "normalized_hubert_layer6",
            "loss_reduction": "per_frame_feature_mean",
            "hard_bound": self.residual_bound.as_dict(),
            "normalizer": self.normalizer.as_dict(),
        }


@dataclass(frozen=True)
class Stage1LossWeights:
    pull_cosine: float = 1.0
    pull_smooth_l1: float = 1.0
    unmatched_zero_delta: float = 0.1
    delta_magnitude: float = 0.01
    delta_smoothness: float = 0.01
    smooth_l1_beta: float = 1.0

    def __post_init__(self) -> None:
        values = {
            "pull_cosine": self.pull_cosine,
            "pull_smooth_l1": self.pull_smooth_l1,
            "unmatched_zero_delta": self.unmatched_zero_delta,
            "delta_magnitude": self.delta_magnitude,
            "delta_smoothness": self.delta_smoothness,
            "smooth_l1_beta": self.smooth_l1_beta,
        }
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Stage 1 loss weights must be finite and non-negative")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "pull_cosine": float(self.pull_cosine),
            "pull_smooth_l1": float(self.pull_smooth_l1),
            "unmatched_zero_delta": float(self.unmatched_zero_delta),
            "delta_magnitude": float(self.delta_magnitude),
            "delta_smoothness": float(self.delta_smoothness),
            "smooth_l1_beta": float(self.smooth_l1_beta),
        }


def _masked_mean(value: Tensor, mask: Tensor, name: str) -> Tensor:
    if value.shape != mask.shape:
        raise ValueError(f"{name} and mask must have identical shape")
    count = int(mask.sum().item())
    if count == 0:
        return value.new_zeros(())
    return value[mask].mean()


def compute_stage1_loss(
    natural_layer6: Tensor,
    delta_layer6: Tensor,
    aligned_tts_layer6: Tensor,
    padding_mask: Tensor,
    target_mask: Tensor,
    normalizer: FeatureNormalizer,
    *,
    weights: Stage1LossWeights = Stage1LossWeights(),
) -> tuple[Tensor, dict[str, Tensor], dict[str, int | float]]:
    """Compute the direct aligned raw-TTS Stage-1 objective.

    ``aligned_tts_layer6`` is detached at the supervision boundary.  No weak
    waveform signal is accepted by this function.
    """
    _check_features(natural_layer6, "natural_layer6", normalizer.input_dim)
    _check_features(delta_layer6, "delta_layer6", normalizer.input_dim)
    _check_features(aligned_tts_layer6, "aligned_tts_layer6", normalizer.input_dim)
    if tuple(natural_layer6.shape) != tuple(delta_layer6.shape) or tuple(natural_layer6.shape) != tuple(aligned_tts_layer6.shape):
        raise ValueError("natural, delta, and aligned TTS layer-6 tensors must share shape")
    _check_mask(padding_mask, natural_layer6)
    _check_mask(target_mask, natural_layer6, "target_mask")
    if torch.any(target_mask & ~padding_mask):
        raise ValueError("target_mask must be a subset of padding_mask real frames")
    target_count = int(target_mask.sum().item())
    if target_count == 0:
        raise ValueError("Stage 1 loss requires at least one aligned target frame")
    natural_z = normalizer.normalize_layer6(natural_layer6)
    goal_z = natural_z + delta_layer6
    target_z = normalizer.normalize_layer6(aligned_tts_layer6).detach()
    pull_cosine = _masked_mean(
        1.0 - F.cosine_similarity(goal_z, target_z, dim=-1, eps=1e-8),
        target_mask,
        "pull cosine",
    )
    pull_smooth_l1 = _masked_mean(
        F.smooth_l1_loss(goal_z, target_z, beta=weights.smooth_l1_beta, reduction="none").mean(dim=-1),
        target_mask,
        "pull Smooth-L1",
    )
    unmatched_mask = padding_mask & ~target_mask
    delta_power = delta_layer6.square().mean(dim=-1)
    unmatched_zero_delta = _masked_mean(delta_power, unmatched_mask, "unmatched zero-delta")
    delta_magnitude = _masked_mean(delta_power, padding_mask, "delta magnitude")
    pair_mask = padding_mask[:, 1:] & padding_mask[:, :-1]
    variation = (delta_layer6[:, 1:] - delta_layer6[:, :-1]).square().mean(dim=-1)
    delta_smoothness = _masked_mean(variation, pair_mask, "delta smoothness")
    total = (
        weights.pull_cosine * pull_cosine
        + weights.pull_smooth_l1 * pull_smooth_l1
        + weights.unmatched_zero_delta * unmatched_zero_delta
        + weights.delta_magnitude * delta_magnitude
        + weights.delta_smoothness * delta_smoothness
    )
    _finite(total, "Stage 1 loss")
    terms = {
        "pull_cosine": pull_cosine,
        "pull_smooth_l1": pull_smooth_l1,
        "unmatched_zero_delta": unmatched_zero_delta,
        "delta_magnitude": delta_magnitude,
        "delta_smoothness": delta_smoothness,
        "total": total,
    }
    diagnostics: dict[str, int | float] = {
        "target_frames": target_count,
        "real_frames": int(padding_mask.sum().item()),
        "unmatched_real_frames": int(unmatched_mask.sum().item()),
        "consecutive_real_pairs": int(pair_mask.sum().item()),
        "target_coverage": float(target_count / max(1, int(padding_mask.sum().item()))),
    }
    return total, terms, diagnostics


def delta_statistics(delta_layer6: Tensor, padding_mask: Tensor) -> dict[str, float | int]:
    """Return finite train/valid diagnostics for real-frame normalized deltas."""
    _check_features(delta_layer6, "delta_layer6", DEFAULT_INPUT_DIM)
    _check_mask(padding_mask, delta_layer6)
    selected = delta_layer6.detach()[padding_mask]
    if selected.numel() == 0:
        raise ValueError("delta statistics require at least one real frame")
    norms = torch.linalg.vector_norm(selected, dim=-1).to(dtype=torch.float64)
    _finite(norms, "delta norms")
    return {
        "frame_count": int(norms.numel()),
        "rms": float(torch.sqrt(torch.mean(norms.square())).item()),
        "mean": float(norms.mean().item()),
        "p95": float(torch.quantile(norms, 0.95).item()),
        "max": float(norms.max().item()),
    }


__all__ = [
    "AlignedTTSFeaturePuller",
    "DEFAULT_BOUND_QUANTILE",
    "DEFAULT_CONTENT_LAYER",
    "DEFAULT_DILATIONS",
    "DEFAULT_HIDDEN_CHANNELS",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_STANDARD_DEVIATION_FLOOR",
    "FeatureNormalizer",
    "ResidualBound",
    "Stage1LossWeights",
    "compute_stage1_loss",
    "delta_statistics",
    "fit_feature_normalizer",
    "fit_residual_bound",
    "validate_hubert_interface",
]
