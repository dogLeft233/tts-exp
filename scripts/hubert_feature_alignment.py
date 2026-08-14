#!/usr/bin/env python3
"""Align raw TTS HuBERT features to a natural-audio frame grid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def frame_centers(
    frame_count: int,
    *,
    frame_stride_samples: int = 320,
    sample_rate: int = 16_000,
    device: torch.device | None = None,
) -> Tensor:
    """Return the centre time (seconds) of each encoder frame."""
    if frame_count <= 0 or frame_stride_samples <= 0 or sample_rate <= 0:
        raise ValueError("frame_count, frame_stride_samples, and sample_rate must be positive")
    return (
        torch.arange(frame_count, dtype=torch.float32, device=device)
        * (float(frame_stride_samples) / float(sample_rate))
        + float(frame_stride_samples) / (2.0 * float(sample_rate))
    )


def _span_value(span: Mapping[str, Any], name: str) -> float:
    value = span.get(name)
    if value is None:
        raise ValueError(f"span is missing {name}")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"span {name} must be finite")
    return result


def _valid_span(span: Mapping[str, Any], min_confidence: float) -> bool:
    try:
        start = _span_value(span, "natural_start_s")
        end = _span_value(span, "natural_end_s")
        tts_start = _span_value(span, "tts_start_s")
        tts_end = _span_value(span, "tts_end_s")
    except ValueError:
        return False
    if not (0.0 <= start < end and 0.0 <= tts_start < tts_end):
        return False
    return float(span.get("natural_confidence", 1.0)) >= min_confidence and float(
        span.get("tts_confidence", 1.0)
    ) >= min_confidence


def _interpolate(features: Tensor, times: Tensor, centers: Tensor) -> Tensor:
    """Linearly interpolate [F,D] features at monotonically arbitrary times."""
    if features.ndim != 2 or centers.ndim != 1 or features.shape[0] != centers.numel():
        raise ValueError("features and frame centers have incompatible shapes")
    if features.shape[0] == 1:
        return features.expand(times.numel(), -1)
    right = torch.searchsorted(centers, times).clamp(1, centers.numel() - 1)
    left = right - 1
    left_t = centers[left]
    right_t = centers[right]
    alpha = ((times - left_t) / (right_t - left_t).clamp_min(torch.finfo(times.dtype).eps)).unsqueeze(-1)
    return features[left] + alpha * (features[right] - features[left])


def align_tts_features_to_natural(
    natural_features: Tensor,
    tts_features: Tensor,
    spans: Sequence[Mapping[str, Any]],
    *,
    frame_stride_samples: int = 320,
    sample_rate: int = 16_000,
    min_confidence: float = 0.8,
    min_frames: int = 1,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Map TTS features through matched phone spans onto natural frame times.

    The returned target is only meaningful where ``mask`` is true.  Unmatched
    pauses are deliberately excluded rather than filled with a natural or zero
    feature, so they cannot create an accidental style-training signal.
    """
    if natural_features.ndim not in (2, 3) or tts_features.ndim not in (2, 3):
        raise ValueError("features must have shape [F,D] or [B,F,D]")
    if natural_features.ndim != tts_features.ndim:
        raise ValueError("natural and TTS feature ranks must match")
    batched = natural_features.ndim == 3
    natural_2d = natural_features[0] if batched else natural_features
    tts_2d = tts_features[0] if batched else tts_features
    if batched and (natural_features.shape[0] != 1 or tts_features.shape[0] != 1):
        raise ValueError("alignment currently accepts one item at a time")
    if natural_2d.shape[-1] != tts_2d.shape[-1]:
        raise ValueError("natural and TTS feature dimensions differ")
    if not torch.isfinite(natural_2d).all() or not torch.isfinite(tts_2d).all():
        raise FloatingPointError("feature tensors must be finite")
    natural_centers = frame_centers(
        natural_2d.shape[0], frame_stride_samples=frame_stride_samples,
        sample_rate=sample_rate, device=natural_2d.device,
    )
    tts_centers = frame_centers(
        tts_2d.shape[0], frame_stride_samples=frame_stride_samples,
        sample_rate=sample_rate, device=tts_2d.device,
    )
    if tts_centers.device != natural_centers.device:
        tts_2d = tts_2d.to(natural_centers.device)
        tts_centers = tts_centers.to(natural_centers.device)
    aligned = torch.zeros_like(natural_2d)
    mask = torch.zeros(natural_2d.shape[0], dtype=torch.bool, device=natural_2d.device)
    used_spans = 0
    for span in spans:
        if not isinstance(span, Mapping) or not _valid_span(span, min_confidence):
            continue
        start = _span_value(span, "natural_start_s")
        end = _span_value(span, "natural_end_s")
        tts_start = _span_value(span, "tts_start_s")
        tts_end = _span_value(span, "tts_end_s")
        current = (natural_centers >= start) & (natural_centers < end)
        if not current.any():
            continue
        natural_times = natural_centers[current]
        ratio = (natural_times - start) / (end - start)
        mapped_times = tts_start + ratio * (tts_end - tts_start)
        # The span itself is the authoritative time range. Interpolation may
        # touch the nearest encoder frame at a boundary but never extrapolates.
        mapped_times = mapped_times.clamp(tts_centers[0], tts_centers[-1])
        aligned[current] = _interpolate(tts_2d, mapped_times, tts_centers)
        mask[current] = True
        used_spans += 1
    matched = int(mask.sum().item())
    if matched < min_frames:
        raise ValueError(f"only {matched} valid HuBERT style frames; minimum is {min_frames}")
    result = aligned.unsqueeze(0) if batched else aligned
    stats = {
        "natural_frames": int(natural_2d.shape[0]),
        "tts_frames": int(tts_2d.shape[0]),
        "matched_frames": matched,
        "coverage": float(matched / natural_2d.shape[0]),
        "used_spans": used_spans,
        "min_confidence": float(min_confidence),
    }
    return result.detach(), mask, stats


def _masked_mean(values: Tensor, mask: Tensor, *, min_frames: int = 1) -> Tensor:
    if values.ndim != 2 or mask.ndim != 1 or values.shape[0] != mask.numel():
        raise ValueError("values and mask must have shapes [F,D] and [F]")
    count = int(mask.sum().item())
    if count < min_frames:
        raise ValueError(f"only {count} valid frames; minimum is {min_frames}")
    result = values[mask].mean()
    if not torch.isfinite(result):
        raise FloatingPointError("masked feature loss is non-finite")
    return result


def masked_tts_feature_losses(
    predicted: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    natural: Tensor | None = None,
    min_frames: int = 1,
) -> dict[str, Tensor]:
    """Return normalized style, cosine, and optional direction losses."""
    if predicted.ndim == 3:
        if predicted.shape[0] != 1:
            raise ValueError("masked feature loss accepts one item at a time")
        predicted = predicted[0]
        target = target[0]
        if natural is not None:
            natural = natural[0]
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must have equal shape [F,D]")
    if mask.ndim != 1 or mask.shape[0] != predicted.shape[0]:
        raise ValueError("mask must have shape [F]")
    target = target.detach()
    if natural is not None:
        natural = natural.detach()
        if natural.shape != predicted.shape:
            raise ValueError("natural feature shape differs from predicted")
    pred_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    cosine = 1.0 - F.cosine_similarity(pred_norm, target_norm, dim=-1)
    normalized_l1 = F.smooth_l1_loss(pred_norm, target_norm, reduction="none").mean(-1)
    losses = {
        "hubert_tts_cosine": _masked_mean(cosine.unsqueeze(-1), mask, min_frames=min_frames),
        "hubert_tts_l1": _masked_mean(normalized_l1.unsqueeze(-1), mask, min_frames=min_frames),
    }
    losses["hubert_tts"] = losses["hubert_tts_cosine"] + losses["hubert_tts_l1"]
    if natural is not None:
        delta = predicted - natural
        target_delta = target - natural
        losses["hubert_delta"] = _masked_mean(
            F.smooth_l1_loss(delta, target_delta, reduction="none").mean(-1).unsqueeze(-1),
            mask,
            min_frames=min_frames,
        )
    return losses


def _feature_matrix(value: Tensor, name: str) -> Tensor:
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must have batch size 1 when batched")
        value = value[0]
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have shape [F,D] or [1,F,D]")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or infinite values")
    return value.detach().float()


def _feature_mask(mask: Tensor, frame_count: int, *, device: torch.device) -> Tensor:
    if mask.ndim == 2:
        if mask.shape[0] != 1:
            raise ValueError("mask must have batch size 1 when batched")
        mask = mask[0]
    if mask.ndim != 1 or mask.shape[0] != frame_count or mask.dtype != torch.bool:
        raise ValueError("mask must have boolean shape [F] or [1,F]")
    return mask.to(device=device)


def _distance_triplet(left: Tensor, right: Tensor, mask: Tensor) -> dict[str, float]:
    left_norm = F.normalize(left, dim=-1)
    right_norm = F.normalize(right, dim=-1)
    cosine = 1.0 - F.cosine_similarity(left_norm, right_norm, dim=-1)
    smooth_l1 = F.smooth_l1_loss(left_norm, right_norm, reduction="none").mean(dim=-1)
    selected_cosine = cosine[mask].mean()
    selected_smooth_l1 = smooth_l1[mask].mean()
    combined = selected_cosine + selected_smooth_l1
    values = (selected_cosine, selected_smooth_l1, combined)
    if not all(torch.isfinite(value) for value in values):
        raise FloatingPointError("HuBERT proximity distance is non-finite")
    return {
        "cosine": float(selected_cosine.cpu()),
        "smooth_l1": float(selected_smooth_l1.cpu()),
        "combined": float(combined.cpu()),
    }


def masked_hubert_proximity(
    natural_features: Tensor,
    enhanced_features: Tensor,
    aligned_tts_features: Tensor,
    mask: Tensor,
    *,
    layer: int | None = None,
    encoder_model: str | None = None,
    min_frames: int = 1,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Measure masked natural/enhanced distance to aligned raw-TTS features.

    This helper is deliberately feature-only: it never invokes an encoder and
    detaches its inputs before reducing them.  It is therefore safe to call
    alongside the explicit TTS HuBERT loss as descriptive telemetry.
    """
    if min_frames <= 0 or not torch.isfinite(torch.tensor(float(epsilon))) or epsilon <= 0:
        raise ValueError("min_frames must be positive and epsilon must be finite and positive")
    natural = _feature_matrix(natural_features, "natural_features")
    enhanced = _feature_matrix(enhanced_features, "enhanced_features")
    target = _feature_matrix(aligned_tts_features, "aligned_tts_features")
    if natural.shape != enhanced.shape or natural.shape != target.shape:
        raise ValueError("natural, enhanced, and aligned TTS features must have equal shape")
    frame_mask = _feature_mask(mask, natural.shape[0], device=natural.device)
    matched = int(frame_mask.sum().item())
    if matched < min_frames:
        raise ValueError(f"only {matched} valid HuBERT proximity frames; minimum is {min_frames}")
    natural_to_tts = _distance_triplet(natural, target, frame_mask)
    enhanced_to_tts = _distance_triplet(enhanced, target, frame_mask)
    absolute_delta = natural_to_tts["combined"] - enhanced_to_tts["combined"]
    relative_delta = absolute_delta / max(natural_to_tts["combined"], float(epsilon))
    result: dict[str, Any] = {
        "schema_version": 1,
        "encoder_model": encoder_model,
        "layer": int(layer) if layer is not None else None,
        "dim": int(natural.shape[-1]),
        "distance_space": "unit_normalized_feature_space",
        "reduction": "masked_mean_per_frame",
        "natural_frame_count": int(natural.shape[0]),
        "matched_frame_count": matched,
        "coverage": float(matched / natural.shape[0]),
        "natural_to_tts": natural_to_tts,
        "enhanced_to_tts": enhanced_to_tts,
        "toward_tts": {
            "absolute_delta": float(absolute_delta),
            "relative_delta": float(relative_delta),
            "enhanced_closer_than_natural": bool(
                enhanced_to_tts["combined"] < natural_to_tts["combined"]
            ),
        },
        "finite": True,
    }
    return result


__all__ = [
    "align_tts_features_to_natural",
    "frame_centers",
    "masked_hubert_proximity",
    "masked_tts_feature_losses",
]
