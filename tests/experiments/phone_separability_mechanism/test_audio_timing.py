from __future__ import annotations

import numpy as np

from scripts.experiments.phone_separability_mechanism.audio import build_speech_mask, render_roundtrip
from scripts.experiments.phone_separability_mechanism.timing import compare_boundaries, waveform_contract


def _tokens() -> list[dict]:
    return [
        {"label": "a", "start_s": 0.10, "end_s": 0.45, "speech": True},
        {"label": "sil", "start_s": 0.45, "end_s": 0.60, "silence": True},
        {"label": "b", "start_s": 0.60, "end_s": 0.95, "speech": True},
    ]


def test_speech_mask_preserves_pause_pcm_and_roundtrip_length() -> None:
    time = np.arange(16_000, dtype=np.float64) / 16_000.0
    source = np.rint(9000 * np.sin(2 * np.pi * 220 * time)).astype(np.int16)
    mask = build_speech_mask(source.size, _tokens())
    output, _ = render_roundtrip(source, mask=mask)
    assert output.size == source.size
    assert np.array_equal(source[mask <= 0], output[mask <= 0])
    assert waveform_contract(source, output, mask)["pass"] is True


def test_timing_matching_catches_missing_phone() -> None:
    natural = [{"label": "a", "start_s": 0.0, "end_s": 0.1, "speech": True}, {"label": "b", "start_s": 0.1, "end_s": 0.2, "speech": True}]
    candidate = [{"label": "a", "start_s": 0.0, "end_s": 0.1, "speech": True}]
    result = compare_boundaries(natural, candidate)
    assert result["edit_rate"] > 0.0
    assert result["timing_pass"] is False
