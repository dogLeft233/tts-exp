#!/usr/bin/env python3
"""Tests for natural/TTS HuBERT feature alignment."""

from __future__ import annotations

import sys
from pathlib import Path
import json

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from hubert_feature_alignment import (  # noqa: E402
    align_tts_features_to_natural,
    frame_centers,
    masked_hubert_proximity,
    masked_tts_feature_losses,
)


def test_frame_centers_use_encoder_stride_and_half_frame_offset() -> None:
    result = frame_centers(3)
    assert torch.allclose(result, torch.tensor([0.01, 0.03, 0.05]))


def test_alignment_maps_linear_phone_span_and_masks_unmatched_frames() -> None:
    natural = torch.arange(6 * 2, dtype=torch.float32).reshape(1, 6, 2)
    tts = torch.arange(4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    aligned, mask, stats = align_tts_features_to_natural(
        natural,
        tts,
        [{"natural_start_s": 0.01, "natural_end_s": 0.08, "tts_start_s": 0.01, "tts_end_s": 0.08}],
        min_frames=1,
    )
    assert aligned.shape == natural.shape
    assert mask.tolist() == [True, True, True, True, False, False]
    assert stats["matched_frames"] == 4


def test_alignment_rejects_no_valid_style_frames() -> None:
    features = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match="valid HuBERT style frames"):
        align_tts_features_to_natural(features, features, [], min_frames=1)




def test_masked_hubert_proximity_reports_progress_and_masked_values() -> None:
    natural = torch.tensor([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    enhanced = torch.tensor([[1.0, 0.0], [0.8, 0.2], [-9.0, -9.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = masked_hubert_proximity(
        natural,
        enhanced,
        target,
        torch.tensor([True, True, False]),
        layer=6,
        encoder_model="facebook/hubert-base-ls960",
    )
    assert result["matched_frame_count"] == 2
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["natural_to_tts"]["combined"] > 0
    assert result["enhanced_to_tts"]["combined"] < result["natural_to_tts"]["combined"]
    assert result["toward_tts"]["absolute_delta"] > 0
    assert result["toward_tts"]["enhanced_closer_than_natural"] is True
    json.dumps(result, allow_nan=False)


def test_masked_hubert_proximity_rejects_invalid_inputs() -> None:
    values = torch.ones(4, 3)
    with pytest.raises(ValueError, match="equal shape"):
        masked_hubert_proximity(values, values[:3], values, torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean shape"):
        masked_hubert_proximity(values, values, values, torch.ones(4))
    with pytest.raises(ValueError, match="valid HuBERT proximity frames"):
        masked_hubert_proximity(values, values, values, torch.zeros(4, dtype=torch.bool))
    nonfinite = values.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="contains NaN"):
        masked_hubert_proximity(nonfinite, values, values, torch.ones(4, dtype=torch.bool))
    predicted = torch.randn(1, 4, 3, requires_grad=True)
    target = torch.randn(1, 4, 3, requires_grad=True)
    natural = torch.randn(1, 4, 3, requires_grad=True)
    losses = masked_tts_feature_losses(predicted, target, torch.tensor([True, True, False, False]), natural=natural)
    losses["hubert_tts"].backward()
    assert predicted.grad is not None
    assert target.grad is None
    assert natural.grad is None
    assert torch.isfinite(predicted.grad).all()
