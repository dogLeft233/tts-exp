"""Unit tests for the direct raw-TTS-aligned Stage-1 feature puller."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tts_aligned_feature_puller import (
    AlignedTTSFeaturePuller,
    FeatureNormalizer,
    ResidualBound,
    Stage1LossWeights,
    compute_stage1_loss,
    delta_statistics,
    fit_feature_normalizer,
    fit_residual_bound,
    validate_hubert_interface,
)
from waveform_hubert_enhancer import EncoderInterface

DIM = 768


def _features(batch: int = 2, frames: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    natural_l2 = torch.randn(batch, frames, DIM, generator=generator)
    natural_l6 = torch.randn(batch, frames, DIM, generator=generator)
    mask = torch.tensor(
        [[True] * frames, [True] * (frames - 2) + [False, False]],
        dtype=torch.bool,
    )
    return natural_l2, natural_l6, mask


def _normalizer() -> FeatureNormalizer:
    natural_l2, natural_l6, mask = _features()
    return fit_feature_normalizer(natural_l2, natural_l6, mask)


def _model() -> AlignedTTSFeaturePuller:
    return AlignedTTSFeaturePuller(
        _normalizer(),
        ResidualBound(value=0.25, quantile=0.95, source_frame_count=8),
        hidden_channels=8,
        dilations=(1,),
    )


def test_fresh_puller_is_zero_and_goal_is_identity() -> None:
    model = _model()
    natural_l2, natural_l6, mask = _features()

    with torch.no_grad():
        delta = model(natural_l2, natural_l6, mask)
        goal = model.goal_layer6(natural_l6, delta)

    assert delta.shape == natural_l6.shape
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(goal, natural_l6)
    assert model.model_config()["identity_initialized"] is True


def test_fresh_puller_receives_a_finite_direct_target_gradient() -> None:
    model = _model()
    natural_l2, natural_l6, mask = _features()
    target = natural_l6 + 0.1
    target_mask = mask.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    delta = model(natural_l2, natural_l6, mask)
    loss, _, _ = compute_stage1_loss(
        natural_l6,
        delta,
        target,
        mask,
        target_mask,
        model.normalizer,
    )
    loss.backward()

    assert model.output_projection.weight.grad is not None
    assert torch.isfinite(model.output_projection.weight.grad).all()
    assert torch.count_nonzero(model.output_projection.weight.grad) > 0
    optimizer.step()
    optimizer.zero_grad()
    updated_delta = model(natural_l2, natural_l6, mask)
    assert torch.count_nonzero(updated_delta[mask]) > 0
    assert torch.all(torch.linalg.vector_norm(updated_delta[mask], dim=-1) <= 0.25 + 1e-6)


def test_puller_hard_l2_cap_and_padding_zero_after_nonzero_head() -> None:
    model = _model()
    natural_l2, natural_l6, mask = _features()
    with torch.no_grad():
        assert model.output_projection.bias is not None
        model.output_projection.bias.fill_(100.0)
        delta = model(natural_l2, natural_l6, mask)

    norms = torch.linalg.vector_norm(delta, dim=-1)
    assert torch.all(norms[mask] <= 0.25 + 1e-6)
    assert torch.allclose(norms[mask], torch.full_like(norms[mask], 0.25), atol=1e-6)
    assert torch.equal(delta[~mask], torch.zeros_like(delta[~mask]))
    stats = delta_statistics(delta, mask)
    assert stats["max"] <= 0.25 + 1e-6


def test_puller_rejects_incompatible_shape_mask_and_nonfinite_values() -> None:
    model = _model()
    natural_l2, natural_l6, mask = _features()
    with pytest.raises(ValueError, match="natural_layer2"):
        model(natural_l2[..., :2], natural_l6, mask)
    with pytest.raises(TypeError, match="dtype"):
        model(natural_l2, natural_l6, mask.to(torch.int64))
    natural_l6[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="natural_layer6"):
        model(natural_l2, natural_l6, mask)


def test_normalizer_and_bound_only_consume_selected_train_frames() -> None:
    natural_l2 = torch.zeros(1, 3, DIM)
    natural_l6 = torch.zeros(1, 3, DIM)
    natural_l2[0, 1] = 2.0
    natural_l6[0, 1] = 2.0
    natural_l2[0, 2] = 10_000.0
    natural_l6[0, 2] = 10_000.0
    real_mask = torch.tensor([[True, True, False]])
    normalizer = fit_feature_normalizer(natural_l2, natural_l6, real_mask)

    assert normalizer.source_frame_count == 2
    assert torch.allclose(normalizer.layer2_mean, torch.ones(DIM))
    assert torch.allclose(normalizer.layer6_mean, torch.ones(DIM))
    assert torch.all(normalizer.layer6_std > 0)

    aligned_tts = natural_l6.clone()
    aligned_tts[0, 0] += 1.0
    target_mask = torch.tensor([[True, False, False]])
    bound = fit_residual_bound(natural_l6, aligned_tts, target_mask, normalizer, quantile=1.0)
    assert bound.source_frame_count == 1
    assert bound.value > 0

    restored = FeatureNormalizer.from_dict(normalizer.as_dict())
    assert torch.equal(restored.layer6_mean, normalizer.layer6_mean)
    assert ResidualBound.from_dict(bound.as_dict()) == bound


def test_goal_converts_normalized_delta_without_readding_mean() -> None:
    normalizer = FeatureNormalizer(
        layer2_mean=torch.full((DIM,), 11.0),
        layer2_std=torch.full((DIM,), 3.0),
        layer6_mean=torch.full((DIM,), 7.0),
        layer6_std=torch.full((DIM,), 2.0),
        source_frame_count=3,
    )
    model = AlignedTTSFeaturePuller(
        normalizer,
        ResidualBound(value=2.0, quantile=1.0, source_frame_count=1),
        hidden_channels=8,
        dilations=(1,),
    )
    natural = torch.full((1, 1, DIM), 5.0)
    delta = torch.ones_like(natural)

    goal = model.goal_layer6(natural, delta)

    assert torch.equal(goal, torch.full_like(goal, 7.0))


def test_direct_raw_tts_loss_detaches_target_and_penalizes_unmatched_delta() -> None:
    normalizer = FeatureNormalizer(
        layer2_mean=torch.zeros(DIM),
        layer2_std=torch.ones(DIM),
        layer6_mean=torch.zeros(DIM),
        layer6_std=torch.ones(DIM),
        source_frame_count=2,
    )
    natural = torch.zeros(1, 3, DIM)
    aligned_tts = torch.zeros(1, 3, DIM, requires_grad=True)
    aligned_tts.data[0, 0] = 1.0
    delta = torch.zeros(1, 3, DIM, requires_grad=True)
    delta.data[0, 0] = 1.0
    delta.data[0, 1] = 0.5
    real_mask = torch.tensor([[True, True, False]])
    target_mask = torch.tensor([[True, False, False]])

    loss, terms, diagnostics = compute_stage1_loss(
        natural,
        delta,
        aligned_tts,
        real_mask,
        target_mask,
        normalizer,
        weights=Stage1LossWeights(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(terms) == {
        "pull_cosine",
        "pull_smooth_l1",
        "unmatched_zero_delta",
        "delta_magnitude",
        "delta_smoothness",
        "total",
    }
    assert terms["pull_cosine"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["pull_smooth_l1"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["unmatched_zero_delta"].item() > 0
    assert diagnostics == {
        "target_frames": 1,
        "real_frames": 2,
        "unmatched_real_frames": 1,
        "consecutive_real_pairs": 1,
        "target_coverage": 0.5,
    }
    assert delta.grad is not None and torch.isfinite(delta.grad).all()
    assert aligned_tts.grad is None


def test_delta_regularizers_use_per_feature_mean_not_dimension_sum() -> None:
    normalizer = FeatureNormalizer(
        layer2_mean=torch.zeros(DIM),
        layer2_std=torch.ones(DIM),
        layer6_mean=torch.zeros(DIM),
        layer6_std=torch.ones(DIM),
        source_frame_count=2,
    )
    natural = torch.zeros(1, 2, DIM)
    target = torch.zeros_like(natural)
    delta = torch.zeros_like(natural)
    delta[0, 1] = 0.5
    real_mask = torch.tensor([[True, True]])
    target_mask = torch.tensor([[True, False]])

    loss, _, _ = compute_stage1_loss(
        natural,
        delta,
        target,
        real_mask,
        target_mask,
        normalizer,
        weights=Stage1LossWeights(
            pull_cosine=0.0,
            pull_smooth_l1=0.0,
            unmatched_zero_delta=1.0,
            delta_magnitude=0.0,
            delta_smoothness=0.0,
        ),
    )


def test_loss_rejects_empty_or_padded_target_masks() -> None:
    normalizer = _normalizer()
    natural_l2, natural_l6, real_mask = _features()
    delta = torch.zeros_like(natural_l6, requires_grad=True)
    target = natural_l6 + 0.2
    with pytest.raises(ValueError, match="at least one aligned target"):
        compute_stage1_loss(
            natural_l6,
            delta,
            target,
            real_mask,
            torch.zeros_like(real_mask),
            normalizer,
        )
    target_mask = torch.zeros_like(real_mask)
    target_mask[1, -1] = True
    with pytest.raises(ValueError, match="subset"):
        compute_stage1_loss(natural_l6, delta, target, real_mask, target_mask, normalizer)


def test_interface_rejects_ditto_and_nonlocal_hubert() -> None:
    interface = EncoderInterface().as_dict()
    assert validate_hubert_interface(interface)["hidden_size"] == DIM
    incompatible = dict(interface, ditto_native_interface=True)
    with pytest.raises(ValueError, match="Ditto"):
        validate_hubert_interface(incompatible)
    incompatible = dict(interface, selected_layer=11)
    with pytest.raises(ValueError, match="layer 6"):
        validate_hubert_interface(incompatible)
