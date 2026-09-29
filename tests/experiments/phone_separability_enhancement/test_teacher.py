from __future__ import annotations

import torch

from scripts.experiments.phone_separability_enhancement.teacher import normalize_waveform, phone_margins_from_hidden


def test_normalization_is_zero_mean_unit_variance() -> None:
    value = normalize_waveform(torch.tensor([[1.0, 2.0, 4.0]]))
    assert torch.allclose(value.mean(dim=-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(value.std(dim=-1, unbiased=False), torch.ones(1), atol=1e-5)


def test_phone_margin_loss_has_input_gradient() -> None:
    hidden = torch.randn(1, 20, 4, requires_grad=True)
    times = torch.arange(20, dtype=torch.float32) * 0.01
    tokens = [{"label": "a", "start_s": 0.0, "end_s": 0.10, "speech": True}, {"label": "b", "start_s": 0.10, "end_s": 0.20, "speech": True}]
    loss, rows = phone_margins_from_hidden(hidden, times, tokens, {"a": [1.0, 0.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0, 0.0]}, view="full")
    assert rows
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
