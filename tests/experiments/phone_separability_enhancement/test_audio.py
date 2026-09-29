from __future__ import annotations

import numpy as np
import torch

from scripts.experiments.phone_separability_enhancement.audio import (
    BandGainRenderer,
    export_pcm,
    gain_field_from_parameters,
    make_protected_mask,
    occurrence_parameter_layout,
    parameter_count,
)


def _tokens() -> list[dict]:
    return [
        {"label": "a", "start_s": 0.10, "end_s": 0.45, "speech": True},
        {"label": "sil", "start_s": 0.45, "end_s": 0.60, "speech": False},
        {"label": "b", "start_s": 0.60, "end_s": 0.95, "speech": True},
    ]


def test_protected_mask_reapplied_after_taper() -> None:
    result = make_protected_mask(_tokens(), 16_000)
    assert np.all(result["mask"][result["protected"]] == 0.0)
    assert np.any(result["mask"] > 0.0)


def test_renderer_zero_gain_is_pcm_identity_and_differentiable() -> None:
    n = 16_000
    source = np.rint(9000 * np.sin(2 * np.pi * 220 * np.arange(n) / 16_000)).astype(np.int16)
    mask_info = make_protected_mask(_tokens(), n)
    renderer = BandGainRenderer()
    waveform = torch.as_tensor(source.astype(np.float32) / 32768.0).requires_grad_(True)
    _, spectrum = renderer.band_features(waveform)
    gain = torch.zeros((1, 24, spectrum.shape[-1]), requires_grad=True)
    output, _ = renderer.render(waveform, gain, torch.as_tensor(mask_info["mask"]))
    pcm, metadata = export_pcm(output, source, mask_info["protected"])
    assert np.array_equal(pcm, source)
    assert metadata["changed_sample_count"] == 0
    output.sum().backward()
    assert gain.grad is not None


def test_dynamic_layout_has_more_than_static_layout() -> None:
    static = occurrence_parameter_layout(_tokens(), dynamic=False)
    dynamic = occurrence_parameter_layout(_tokens(), dynamic=True)
    assert parameter_count(dynamic) >= parameter_count(static)
    assert parameter_count(static) == 48


def test_gain_field_is_zero_outside_speech_occurrences() -> None:
    tokens = _tokens()
    layout = occurrence_parameter_layout(tokens, dynamic=True)
    params = torch.ones((1, parameter_count(layout)))
    times = torch.arange(0, 1.0, 0.01)
    field = gain_field_from_parameters(params, tokens, times, dynamic=True)
    assert torch.all(field[:, :, (times < 0.1) | ((times >= 0.45) & (times < 0.60)) | (times >= 0.95)] == 0)
