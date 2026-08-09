from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_rhythm_style_dataset import (
    RHYTHM_DIM,
    STYLE_INPUT_DIM,
    align_content,
    build_rhythm,
    make_splits,
    select_embedding_layer,
)
from rhythm_style_generator import RhythmStyleGenerator, StyleProsodyEncoder


def _inputs(batch: int = 2, frames: int = 24, samples: int = 768):
    return (
        torch.randn(batch, 1, samples),
        torch.randn(batch, frames, 768),
        torch.randn(batch, frames, RHYTHM_DIM),
        torch.randn(batch, 31, STYLE_INPUT_DIM),
        torch.ones(batch, frames, dtype=torch.bool),
    )


def test_fresh_model_is_identity_and_shapes() -> None:
    model = RhythmStyleGenerator()
    natural, content, rhythm, style, mask = _inputs()
    with torch.no_grad():
        output = model(natural, content, rhythm, style, frame_mask=mask)
    assert torch.equal(output["waveform"], natural)
    assert output["waveform"].shape == natural.shape
    assert output["content_prediction"].shape == content.shape
    assert output["rhythm_prediction"].shape == rhythm.shape
    assert output["style"].shape == (2, 64)


def test_forward_backward_has_finite_gradients() -> None:
    model = RhythmStyleGenerator()
    inputs = _inputs(batch=1)
    output = model(
        inputs[0], inputs[1], inputs[2], inputs[3], frame_mask=inputs[4]
    )
    loss = output["waveform"].square().mean() + output["content_prediction"].square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_style_encoder_handles_masked_reference() -> None:
    encoder = StyleProsodyEncoder()
    features = torch.randn(2, 17, STYLE_INPUT_DIM)
    full = encoder(features)
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[:, 8:] = False
    masked = encoder(features, mask)
    assert full.shape == masked.shape == (2, 64)
    assert torch.isfinite(masked).all()


def test_frame_mask_keeps_padded_context_zero() -> None:
    model = RhythmStyleGenerator()
    natural, content, rhythm, style, mask = _inputs(batch=1, frames=24)
    mask[:, 12:] = False
    output = model(natural, content, rhythm, style, frame_mask=mask)
    assert torch.equal(output["context"][:, 12:], torch.zeros_like(output["context"][:, 12:]))


def test_embedding_layer_selection_uses_sidecar_metadata() -> None:
    values = np.random.default_rng(42).normal(size=(12, 7, 768)).astype(np.float32)
    selected = select_embedding_layer(values, {"layers": list(range(12))}, layer=11)
    assert selected.shape == (7, 768)
    assert np.array_equal(selected, values[11])


def test_align_content_and_rhythm_shapes() -> None:
    values = np.zeros((7, 768), dtype=np.float32)
    assert align_content(values, 11).shape == (11, 768)
    from pnp_alignment_warp import Span
    natural = [Span("a", 0.0, 0.2), Span("b", 0.2, 0.4)]
    tts = [Span("a", 0.0, 0.1), Span("b", 0.1, 0.35)]
    rhythm = build_rhythm(natural, tts, 640, alpha=0.5)
    assert rhythm.shape == (2, RHYTHM_DIM)
    assert np.isfinite(rhythm).all()
    assert rhythm[:, 0].any()


def test_speaker_split_is_disjoint_and_deterministic() -> None:
    mapping = {sample: f"speaker-{(sample - 1) // 2}" for sample in range(1, 17)}
    first = make_splits(mapping, seed=42)
    second = make_splits(mapping, seed=42)
    assert first == second
    speakers = {name: {mapping[sample] for sample, split in first.items() if split == name} for name in ("train", "valid", "test")}
    assert not speakers["train"] & speakers["valid"]
    assert not speakers["train"] & speakers["test"]
    assert not speakers["valid"] & speakers["test"]


def test_no_secret_in_model_module() -> None:
    source = Path(__file__).resolve().parent.parent / "scripts" / "rhythm_style_generator.py"
    assert "sk-" not in source.read_text(encoding="utf-8")
