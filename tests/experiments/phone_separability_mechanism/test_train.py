from __future__ import annotations

import torch

from scripts.experiments.phone_separability_mechanism.train import BoundedGainEnhancer, differentiable_ssl_forward, select_checkpoint


class _Teacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, waveform, output_hidden_states=True):
        hidden = waveform[:, ::320].unsqueeze(-1).repeat(1, 1, 4) * self.scale
        return type("Output", (), {"hidden_states": (hidden,)})()


def test_bounded_gain_has_zero_initial_output_and_input_gradient() -> None:
    model = BoundedGainEnhancer(mel_bands=24, channels=8)
    mel = torch.randn(1, 24, 10, requires_grad=True)
    output = model(mel)
    assert output.shape == mel.shape
    assert torch.allclose(output, torch.zeros_like(output))
    output.sum().backward()
    assert mel.grad is not None


def test_differentiable_teacher_freezes_parameters_but_keeps_waveform_grad() -> None:
    teacher = _Teacher()
    waveform = torch.randn(1, 1, 640, requires_grad=True)
    features = differentiable_ssl_forward(teacher, waveform, layer=0)
    features.sum().backward()
    assert waveform.grad is not None
    assert teacher.scale.grad is None


def test_checkpoint_selection_rejects_e_seen_and_xlsr() -> None:
    checkpoint = {"split": "dev", "accuracy": 0.5, "hubert_margin": 0.1, "waveform_qc": True, "timing_qc": True, "step": 1}
    assert select_checkpoint([checkpoint], min_accuracy=0.4)["step"] == 1
    try:
        select_checkpoint([checkpoint], min_accuracy=0.4, use_xlsr=True)
    except ValueError:
        pass
    else:
        raise AssertionError("XLSR must not influence checkpoint selection")
