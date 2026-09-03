"""Mathematical contract tests for the full-curve target-margin objective."""
from __future__ import annotations

import pytest
import torch

from scripts.experiments.mfa_linear_real_video_sync.syncnet_loss import (
    log_mel_trust_violation,
    target_margin_loss_components,
    waveform_qc,
)


def pristine(target_offset: int = 0) -> torch.Tensor:
    curve = torch.full((31,), 2.0)
    curve[target_offset + 15] = 1.0
    return curve


def satisfied(target_offset: int = 0) -> torch.Tensor:
    curve = torch.full((31,), 2.2)
    curve[target_offset + 15] = 0.98
    return curve


def _assert_zero_implications(candidate: torch.Tensor, target_offset: int) -> None:
    parts = target_margin_loss_components(
        candidate,
        pristine_curve=pristine(target_offset),
        target_offset=target_offset,
    )
    assert parts["total"].item() == 0
    target_index = target_offset + 15
    target = candidate[target_index]
    wrong = torch.cat((candidate[:target_index], candidate[target_index + 1 :]))
    assert int(torch.argmin(candidate)) == target_index
    assert torch.count_nonzero(candidate == candidate.min()) == 1
    assert target <= parts["d0"] - parts["delta_d"]
    confidence = candidate.median() - candidate.min()
    assert confidence >= parts["c0"] + parts["delta_c"]
    assert torch.all(wrong - target >= parts["c_goal"])


@pytest.mark.parametrize("target_offset", [-15, 0, 15])
def test_zero_loss_implies_d_c_and_unique_target(target_offset: int) -> None:
    _assert_zero_implications(satisfied(target_offset), target_offset)


def test_d_only_violation_is_positive_with_live_target_gradient() -> None:
    candidate = torch.full((31,), 2.2)
    candidate[15] = 1.0
    candidate.requires_grad_()
    parts = target_margin_loss_components(candidate, pristine_curve=pristine(), target_offset=0)
    assert parts["target_violation"] > 0
    assert parts["ranking_violation"] == 0
    parts["total"].backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad[15] > 0


def test_one_negative_violation_has_both_live_gradient_signs() -> None:
    candidate = satisfied().detach()
    candidate[0] = 1.5
    candidate.requires_grad_()
    parts = target_margin_loss_components(candidate, pristine_curve=pristine(), target_offset=0)
    assert parts["target_violation"] == 0
    assert parts["ranking_violation"] > 0
    parts["total"].backward()
    assert candidate.grad is not None
    assert candidate.grad[0] < 0
    assert candidate.grad[15] > 0


def test_combined_violation_and_tie_are_rejected_by_loss() -> None:
    candidate = torch.full((31,), 2.2)
    candidate[15] = 1.0
    candidate[0] = 1.0
    candidate.requires_grad_()
    parts = target_margin_loss_components(candidate, pristine_curve=pristine(), target_offset=0)
    assert parts["target_violation"] > 0
    assert parts["ranking_violation"] > 0
    parts["total"].backward()
    assert candidate.grad is not None and candidate.grad[0] < 0 and candidate.grad[15] > 0


def test_wrong_offsets_are_not_detached() -> None:
    candidate = torch.full((31,), 1.5, requires_grad=True)
    parts = target_margin_loss_components(candidate, pristine_curve=pristine(), target_offset=0)
    parts["total"].backward()
    assert candidate.grad is not None
    wrong = torch.cat((candidate.grad[:15], candidate.grad[16:]))
    assert torch.all(wrong < 0)


def test_trust_hinge_boundary_and_waveform_qc() -> None:
    below = torch.tensor(0.099, requires_grad=True)
    boundary = torch.tensor(0.10, requires_grad=True)
    above = torch.tensor(0.11, requires_grad=True)
    assert log_mel_trust_violation(below).item() == 0
    assert log_mel_trust_violation(boundary).item() == 0
    torch.testing.assert_close(log_mel_trust_violation(above), torch.tensor(0.1))

    source = torch.zeros(1, 1, 100)
    candidate = source + 0.05
    qc = waveform_qc(candidate, source, residual_bound=0.05)
    assert qc["residual_bound_pass"] and qc["pcm_saturation_pass"]
    with pytest.raises(ValueError, match="residual"):
        waveform_qc(source + 0.051, source, residual_bound=0.05)
    clipped = source.clone()
    clipped[..., 0] = 1.1
    qc = waveform_qc(clipped, source, residual_bound=2.0)
    assert qc["pcm_saturation_fraction"] == 0.01
    assert not qc["pcm_saturation_pass"]


def test_nonfinite_and_invalid_pristine_fail_closed() -> None:
    candidate = satisfied()
    bad = pristine()
    bad[0] = float("nan")
    with pytest.raises(FloatingPointError):
        target_margin_loss_components(candidate, pristine_curve=bad, target_offset=0)
    flat = torch.ones(31)
    with pytest.raises(ValueError, match="positive D/C"):
        target_margin_loss_components(candidate, pristine_curve=flat, target_offset=0)
