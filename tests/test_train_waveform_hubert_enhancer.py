#!/usr/bin/env python3
"""CPU tests for HuBERT feature loss, splits, and checkpoint validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_waveform_hubert_enhancer import (  # noqa: E402
    LossWeights,
    _read_wave,
    _split_items,
    hubert_features,
    hu_feature_loss,
    summarize_hubert_proximity,
    waveform_loss,
)
from waveform_hubert_enhancer import EncoderInterface, HuBERTWaveformEnhancer  # noqa: E402


class FakeHuBERT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool = False):
        frames = waveform[..., :640].reshape(waveform.shape[0], 2, 320).mean(-1, keepdim=True)
        hidden = frames.repeat(1, 1, 768) * self.scale
        return type("Output", (), {"hidden_states": tuple(hidden for _ in range(7))})()


def make_model() -> HuBERTWaveformEnhancer:
    return HuBERTWaveformEnhancer(
        FakeHuBERT(),
        interface=EncoderInterface(),
        channels=8,
        dilations=(1,),
    )




def test_read_wave_explicitly_resamples_24k_tts(tmp_path: Path) -> None:
    path = tmp_path / "tts.wav"
    sf.write(path, np.zeros(24000, dtype=np.float32), 24000, subtype="FLOAT")
    with pytest.raises(ValueError, match="must be 16000"):
        _read_wave(path)
    result = _read_wave(path, allow_24k_resample=True)
    assert result.shape == (1, 1, 16000)


def test_summarize_hubert_proximity_reports_split_counts() -> None:
    rows = [
        {
            "coverage": 0.5,
            "matched_frame_count": 2,
            "natural_to_tts": {"cosine": 0.4, "smooth_l1": 0.2, "combined": 0.6},
            "enhanced_to_tts": {"cosine": 0.2, "smooth_l1": 0.1, "combined": 0.3},
            "toward_tts": {
                "absolute_delta": 0.3,
                "relative_delta": 0.5,
                "enhanced_closer_than_natural": True,
            },
        },
        {
            "coverage": 1.0,
            "matched_frame_count": 4,
            "natural_to_tts": {"cosine": 0.2, "smooth_l1": 0.1, "combined": 0.3},
            "enhanced_to_tts": {"cosine": 0.3, "smooth_l1": 0.2, "combined": 0.5},
            "toward_tts": {
                "absolute_delta": -0.2,
                "relative_delta": -2 / 3,
                "enhanced_closer_than_natural": False,
            },
        },
    ]
    result = summarize_hubert_proximity(rows)
    assert result["item_count"] == 2
    assert result["total_matched_frame_count"] == 6
    assert result["toward_tts"]["enhanced_closer_fraction"] == pytest.approx(0.5)
    assert result["toward_tts"]["mean_absolute_delta"] == pytest.approx(0.05)
    json.dumps(result, allow_nan=False)


def test_summarize_hubert_proximity_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="empty split"):
        summarize_hubert_proximity([])


def test_hubert_feature_loss_is_differentiable_and_target_detached() -> None:
    model = make_model()
    waveform = torch.randn(1, 1, 640)
    predicted = hubert_features(model.encoder, waveform.requires_grad_(), retain_gradient=True)
    target = predicted.detach().clone() + 1
    loss = hu_feature_loss(predicted, target)
    loss.backward()
    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert not target.requires_grad


def test_waveform_loss_reports_all_terms() -> None:
    model = make_model()
    natural = torch.randn(1, 1, 1024)
    target = natural + 0.1 * torch.randn_like(natural)
    output = model(natural)
    target_features = hubert_features(model.encoder, target)
    total, terms = waveform_loss(
        output["waveform"], target, natural, output["enhanced_features"], target_features,
        weights=LossWeights(),
    )
    assert torch.isfinite(total)
    assert set(terms) == {"stft", "mel", "waveform", "hubert", "energy", "residual"}


def test_split_rejects_speaker_leakage() -> None:
    items = [
        {"paired_key": "a", "split": "train", "speaker_group": "spk"},
        {"paired_key": "b", "split": "valid", "speaker_group": "spk"},
    ]
    with pytest.raises(ValueError, match="cross splits"):
        _split_items(items)


def test_split_accepts_disjoint_groups() -> None:
    items = [
        {"paired_key": "a", "split": "train", "speaker_group": "spk1"},
        {"paired_key": "b", "split": "valid", "speaker_group": "spk2"},
        {"paired_key": "c", "split": "heldout", "speaker_group": "spk3"},
    ]
    result = _split_items(items)
    assert [item["paired_key"] for item in result["train"]] == ["a"]
    assert [item["paired_key"] for item in result["heldout"]] == ["c"]
