#!/usr/bin/env python3
"""Tests for the frozen Stage-1-conditioned waveform renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from feature_targeted_waveform_renderer import (  # noqa: E402
    FeatureTargetedWaveformInference,
    FeatureTargetedWaveformRenderer,
    FrozenStage1Artifact,
)
from tts_aligned_feature_puller import (  # noqa: E402
    AlignedTTSFeaturePuller,
    FeatureNormalizer,
    ResidualBound,
)
from waveform_hubert_enhancer import EncoderInterface  # noqa: E402

DIM = 768


class FakeHuBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor, *, output_hidden_states: bool):
        assert output_hidden_states is True
        self.calls += 1
        frames = waveform.shape[-1] // 320
        if frames < 1:
            raise ValueError("fake encoder requires at least one frame")
        signal = waveform[:, : frames * 320].reshape(waveform.shape[0], frames, 320).mean(dim=-1)
        base = signal.unsqueeze(-1).expand(-1, -1, DIM) * self.scale
        return type("Output", (), {"hidden_states": tuple(base + index for index in range(7))})()


def make_stage1() -> tuple[AlignedTTSFeaturePuller, FrozenStage1Artifact]:
    normalizer = FeatureNormalizer(
        layer2_mean=torch.zeros(DIM),
        layer2_std=torch.ones(DIM),
        layer6_mean=torch.zeros(DIM),
        layer6_std=torch.ones(DIM),
        source_frame_count=4,
    )
    stage1 = AlignedTTSFeaturePuller(
        normalizer,
        ResidualBound(value=0.5, quantile=0.95, source_frame_count=4),
        hidden_channels=8,
        dilations=(1,),
    )
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    interface = EncoderInterface().as_dict()
    artifact = FrozenStage1Artifact(
        checkpoint_path="/verified/stage1/best.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_epoch=1,
        encoder_interface=interface,
        model_config=stage1.model_config(),
    )
    return stage1, artifact


def make_renderer() -> tuple[FeatureTargetedWaveformRenderer, FakeHuBERT]:
    stage1, artifact = make_stage1()
    encoder = FakeHuBERT()
    return (
        FeatureTargetedWaveformRenderer(
            encoder,
            stage1,
            artifact,
            interface=EncoderInterface(),
            conditioning_dim=8,
            channels=8,
            dilations=(1,),
            residual_scale=0.05,
        ),
        encoder,
    )


def test_fresh_renderer_is_identity_with_nonzero_frozen_condition() -> None:
    renderer, encoder = make_renderer()
    waveform = torch.randn(1, 1, 640)
    with torch.no_grad():
        renderer.stage1.output_projection.bias.fill_(1.0)
        output = renderer(waveform)

    assert encoder.calls == 2
    assert torch.equal(output["waveform"], waveform)
    assert torch.count_nonzero(output["stage1_delta_layer6"]) > 0
    assert output["stage1_delta_layer6"].shape == output["natural_layer6"].shape
    assert output["conditioning"].shape == (1, 24, 640)
    assert output["enhanced_layer2"].shape[-1] == DIM
    assert output["enhanced_layer6"].shape[-1] == DIM


def test_frozen_encoder_and_stage1_receive_no_grads_but_renderer_does() -> None:
    renderer, _ = make_renderer()
    waveform = torch.randn(1, 1, 640)
    output = renderer(waveform)
    loss = torch.nn.functional.smooth_l1_loss(output["enhanced_layer6"], torch.zeros_like(output["enhanced_layer6"]))
    loss.backward()

    assert all(not parameter.requires_grad for parameter in renderer.encoder.parameters())
    assert all(parameter.grad is None for parameter in renderer.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in renderer.stage1.parameters())
    assert all(parameter.grad is None for parameter in renderer.stage1.parameters())
    assert renderer.waveform_model.output_projection.weight.grad is not None
    assert torch.isfinite(renderer.waveform_model.output_projection.weight.grad).all()


def test_renderer_rejects_partial_or_invalid_stage1_conditioning() -> None:
    renderer, _ = make_renderer()
    waveform = torch.randn(1, 1, 640)
    with pytest.raises(ValueError, match="supplied together"):
        renderer(waveform, natural_layer2=torch.zeros(1, 2, DIM))

    natural_l2 = torch.zeros(1, 2, DIM)
    natural_l6 = torch.zeros_like(natural_l2)
    delta = torch.zeros_like(natural_l2)
    delta[0, 0] = 1.0
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="hard bound"):
        renderer(
            waveform,
            natural_layer2=natural_l2,
            natural_layer6=natural_l6,
            delta_layer6=delta,
            padding_mask=mask,
            compute_enhanced_features=False,
        )


def test_renderer_rejects_incompatible_stage1_hubert_interface() -> None:
    stage1, artifact = make_stage1()
    incompatible = FrozenStage1Artifact(
        checkpoint_path=artifact.checkpoint_path,
        checkpoint_sha256=artifact.checkpoint_sha256,
        checkpoint_epoch=artifact.checkpoint_epoch,
        encoder_interface=dict(artifact.encoder_interface, selected_layer=5),
        model_config=artifact.model_config,
    )
    with pytest.raises(ValueError, match="interfaces must match"):
        FeatureTargetedWaveformRenderer(
            FakeHuBERT(),
            stage1,
            incompatible,
            conditioning_dim=8,
            channels=8,
            dilations=(1,),
        )


def test_inference_wrapper_preserves_exact_length() -> None:
    renderer, _ = make_renderer()
    wrapper = FeatureTargetedWaveformInference(
        renderer,
        chunk_seconds=0.08,
        stride_seconds=0.064,
    )
    audio = np.sin(np.linspace(0.0, 4.0, 777, dtype=np.float32))
    enhanced = wrapper.enhance(audio)

    assert enhanced.shape == audio.shape
    assert enhanced.dtype == np.float32
    assert np.array_equal(enhanced, audio)
