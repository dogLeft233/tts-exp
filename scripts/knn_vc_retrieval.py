#!/usr/bin/env python3
"""Deterministic frame retrieval for the valid-only kNN-VC PoC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

SILENCE_LABEL = "<sil>"
VALID_POLICIES = {"unconstrained", "same_phone", "same_phone_position", "wrong_phone"}


@dataclass(frozen=True)
class FrameOwner:
    label: str
    span_index: int
    relative_position: float
    is_silence: bool


@dataclass(frozen=True)
class RetrievalConfig:
    policy: str
    topk: int = 4
    position_weight: float = 0.25
    continuity_weight: float = 0.05
    query_chunk_size: int = 256
    allow_unconstrained_tts_fallback: bool = False


@dataclass
class RetrievalResult:
    features: Tensor
    indices: Tensor
    distances: Tensor
    metadata: dict[str, Any]


def _token_label(token: Mapping[str, Any]) -> tuple[str, bool]:
    raw = str(token.get("token", token.get("label", ""))).strip()
    silence = bool(token.get("is_silence") or token.get("is_non_speech")) or raw.casefold() in {
        "", "sil", "sp", "spn", "<sil>"
    }
    return (SILENCE_LABEL if silence else raw.casefold()), silence


def frame_owners(
    frame_count: int,
    tokens: Sequence[Mapping[str, Any]],
    *,
    frame_stride_samples: int = 320,
    sample_rate: int = 16_000,
) -> list[FrameOwner]:
    """Assign every fixed-rate frame centre to exactly one MFA token span."""
    if frame_count <= 0 or frame_stride_samples <= 0 or sample_rate <= 0:
        raise ValueError("frame count, stride, and sample rate must be positive")
    normalized: list[tuple[float, float, str, bool]] = []
    previous_end = 0.0
    for index, token in enumerate(tokens):
        try:
            start = float(token["start_s"])
            end = float(token["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed token span {index}") from exc
        if start < previous_end - 1e-7 or end <= start:
            raise ValueError(f"invalid or overlapping token span {index}")
        if index == 0 and start > 1e-6:
            raise ValueError("token alignment must start at zero")
        label, silence = _token_label(token)
        normalized.append((start, end, label, silence))
        previous_end = end
    if not normalized:
        raise ValueError("token alignment is empty")

    owners: list[FrameOwner] = []
    span_index = 0
    for frame_index in range(frame_count):
        centre = (frame_index + 0.5) * frame_stride_samples / sample_rate
        while span_index + 1 < len(normalized) and centre >= normalized[span_index][1] - 1e-9:
            span_index += 1
        start, end, label, silence = normalized[span_index]
        if not start - 1e-7 <= centre < end + 1e-7:
            raise ValueError(f"frame {frame_index} at {centre:.6f}s has no MFA owner")
        relative = min(1.0, max(0.0, (centre - start) / (end - start)))
        owners.append(FrameOwner(label, span_index, relative, silence))
    return owners


def _validate_features(query: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
    query = torch.as_tensor(query, dtype=torch.float32)
    candidates = torch.as_tensor(candidates, dtype=torch.float32)
    if query.ndim != 2 or candidates.ndim != 2 or query.shape[1] != candidates.shape[1]:
        raise ValueError("query and candidates must be finite [T,D] tensors with equal D")
    if query.shape[0] == 0 or candidates.shape[0] == 0:
        raise ValueError("query and candidate sequences must be non-empty")
    if not torch.isfinite(query).all() or not torch.isfinite(candidates).all():
        raise FloatingPointError("query and candidate features must be finite")
    return query, candidates


def retrieve(
    query: Tensor,
    candidates: Tensor,
    query_owners: Sequence[FrameOwner],
    candidate_owners: Sequence[FrameOwner],
    config: RetrievalConfig,
) -> RetrievalResult:
    """Retrieve TTS candidate frames for a natural-clock query sequence."""
    query, candidates = _validate_features(query, candidates)
    if config.policy not in VALID_POLICIES:
        raise ValueError(f"unsupported retrieval policy: {config.policy}")
    if config.topk <= 0 or config.query_chunk_size <= 0:
        raise ValueError("topk and query_chunk_size must be positive")
    if len(query_owners) != query.shape[0] or len(candidate_owners) != candidates.shape[0]:
        raise ValueError("frame-owner counts must match feature frame counts")

    device = query.device
    candidates = candidates.to(device)
    query_norm = torch.nn.functional.normalize(query, dim=-1)
    candidate_norm = torch.nn.functional.normalize(candidates, dim=-1)
    candidate_labels = [owner.label for owner in candidate_owners]
    candidate_positions = torch.tensor(
        [owner.relative_position for owner in candidate_owners], device=device, dtype=query.dtype
    )

    output: list[Tensor] = []
    selected_indices: list[Tensor] = []
    selected_distances: list[Tensor] = []
    effective_k: list[int] = []
    fallback_frames = 0
    previous_index: int | None = None
    previous_span: int | None = None

    for start in range(0, query.shape[0], config.query_chunk_size):
        end = min(query.shape[0], start + config.query_chunk_size)
        base = 1.0 - query_norm[start:end] @ candidate_norm.T
        for local_index, query_index in enumerate(range(start, end)):
            owner = query_owners[query_index]
            if config.policy == "unconstrained":
                allowed = torch.ones(candidates.shape[0], dtype=torch.bool, device=device)
            elif config.policy in {"same_phone", "same_phone_position"}:
                allowed = torch.tensor(
                    [label == owner.label for label in candidate_labels], device=device, dtype=torch.bool
                )
            else:
                allowed = torch.tensor(
                    [label != owner.label for label in candidate_labels], device=device, dtype=torch.bool
                )
            allowed_count = int(allowed.sum().item())
            if allowed_count == 0:
                if not config.allow_unconstrained_tts_fallback or config.policy not in {
                    "same_phone", "same_phone_position"
                }:
                    raise ValueError(
                        f"policy {config.policy} has no candidate for frame {query_index} label={owner.label!r}"
                    )
                allowed = torch.ones(candidates.shape[0], dtype=torch.bool, device=device)
                allowed_count = candidates.shape[0]
                fallback_frames += 1
            cost = base[local_index].clone()
            if config.policy == "same_phone_position":
                cost += config.position_weight * torch.abs(candidate_positions - owner.relative_position)
                if previous_index is not None and previous_span == owner.span_index:
                    jump = torch.abs(
                        torch.arange(candidates.shape[0], device=device, dtype=query.dtype) - previous_index - 1
                    )
                    cost += config.continuity_weight * torch.clamp(jump, max=10.0) / 10.0
            cost[~allowed] = torch.inf
            k = min(config.topk, allowed_count)
            # Stable argsort gives deterministic candidate-index tie breaking.
            indices = torch.argsort(cost, stable=True)[:k]
            distances = cost[indices]
            if not torch.isfinite(distances).all():
                raise FloatingPointError("retrieval selected a non-finite distance")
            output.append(candidates[indices].mean(dim=0))
            selected_indices.append(indices.cpu())
            selected_distances.append(distances.cpu())
            effective_k.append(k)
            previous_index = int(indices[0].item())
            previous_span = owner.span_index

    max_k = max(effective_k)
    padded_indices = torch.full((len(selected_indices), max_k), -1, dtype=torch.long)
    padded_distances = torch.full((len(selected_indices), max_k), float("nan"), dtype=torch.float32)
    for row, (indices, distances) in enumerate(zip(selected_indices, selected_distances, strict=True)):
        padded_indices[row, : len(indices)] = indices
        padded_distances[row, : len(distances)] = distances
    speech_count = sum(not owner.is_silence for owner in query_owners)
    silence_count = len(query_owners) - speech_count
    return RetrievalResult(
        features=torch.stack(output),
        indices=padded_indices,
        distances=padded_distances,
        metadata={
            "config": asdict(config),
            "query_frames": len(query_owners),
            "candidate_frames": len(candidate_owners),
            "speech_frames": speech_count,
            "silence_frames": silence_count,
            "coverage": float((len(query_owners) - fallback_frames) / len(query_owners)),
            "fallback_frames": fallback_frames,
            "effective_k_min": min(effective_k),
            "effective_k_max": max(effective_k),
            "mean_selected_cosine_cost": float(
                torch.cat(selected_distances).mean().item()
            ),
        },
    )


def matched_span_map(
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, int], dict[str, int]]:
    """Match equal phone labels in order while tolerating MFA insertions/replacements."""
    natural_labels = [_token_label(token)[0] for token in natural_tokens]
    tts_labels = [_token_label(token)[0] for token in tts_tokens]
    matcher = SequenceMatcher(a=natural_labels, b=tts_labels, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping, {
        "natural_token_count": len(natural_labels),
        "tts_token_count": len(tts_labels),
        "matched_token_count": len(mapping),
        "unmatched_natural_token_count": len(natural_labels) - len(mapping),
        "unmatched_tts_token_count": len(tts_labels) - len(set(mapping.values())),
    }


def mfa_linear_target(
    natural_frame_count: int,
    tts_features: Tensor,
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
    *,
    frame_stride_samples: int = 320,
    sample_rate: int = 16_000,
) -> tuple[Tensor, dict[str, Any]]:
    """Interpolate corresponding TTS token trajectories onto the natural frame grid."""
    tts_features = torch.as_tensor(tts_features, dtype=torch.float32)
    natural_owners = frame_owners(
        natural_frame_count, natural_tokens,
        frame_stride_samples=frame_stride_samples, sample_rate=sample_rate,
    )
    # Validate complete TTS alignment coverage independently of interpolation.
    frame_owners(
        tts_features.shape[0], tts_tokens,
        frame_stride_samples=frame_stride_samples, sample_rate=sample_rate,
    )
    span_mapping, match_stats = matched_span_map(natural_tokens, tts_tokens)

    rows: list[Tensor] = []
    matched_frames = 0
    unmatched_frames = 0
    for owner in natural_owners:
        tts_span_index = span_mapping.get(owner.span_index)
        if tts_span_index is None:
            # No phone-equal target exists. Use a global-relative TTS frame as an
            # explicit TTS-only fallback; never inject a natural feature silently.
            global_position = owner.relative_position if natural_frame_count == 1 else (
                len(rows) / (natural_frame_count - 1)
            )
            tts_position = global_position * (tts_features.shape[0] - 1)
            unmatched_frames += 1
        else:
            token = tts_tokens[tts_span_index]
            mapped_time = float(token["start_s"]) + owner.relative_position * (
                float(token["end_s"]) - float(token["start_s"])
            )
            # WavLM frame centres occur at (index + 0.5) * stride / rate.
            # Interpolate against the complete sequence so even a phone shorter
            # than one 20 ms hop has a well-defined target.
            tts_position = mapped_time * sample_rate / frame_stride_samples - 0.5
            tts_position = min(tts_features.shape[0] - 1, max(0.0, tts_position))
            matched_frames += 1
        left_index = int(tts_position)
        right_index = min(tts_features.shape[0] - 1, left_index + 1)
        alpha = tts_position - left_index
        rows.append(
            tts_features[left_index]
            + alpha * (tts_features[right_index] - tts_features[left_index])
        )
    coverage = matched_frames / natural_frame_count
    return torch.stack(rows), {
        "policy": "mfa_linear",
        "query_frames": natural_frame_count,
        "candidate_frames": int(tts_features.shape[0]),
        "matched_frames": matched_frames,
        "coverage": coverage,
        "fallback_frames": unmatched_frames,
        "match_stats": match_stats,
    }
