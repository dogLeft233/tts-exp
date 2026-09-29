from __future__ import annotations

import numpy as np
import torch

from scripts.experiments.phone_separability_enhancement.optimize import label_permutation, optimize_utterance


class _TinyTeacher:
    device = "cpu"

    def encode(self, waveform, *, layer: int):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        values = waveform[:, ::64]
        hidden = torch.stack((values, -values, values.square(), torch.sin(values)), dim=-1)
        times = torch.arange(hidden.shape[1], dtype=hidden.dtype, device=hidden.device) * 64.0 / 16_000.0
        return hidden, times, {"layer": layer}


def test_label_permutation_has_no_fixed_points() -> None:
    permutation = label_permutation(["a", "b", "c"])
    assert permutation
    assert all(permutation[key] != key for key in permutation)


def test_direct_optimizer_preserves_length_and_hard_pause() -> None:
    sample_rate = 16_000
    n = 8_000
    source = np.rint(5000 * np.sin(2 * np.pi * 220 * np.arange(n) / sample_rate)).astype(np.int16)
    tokens = [{"label": "a", "start_s": 0.05, "end_s": 0.20, "speech": True}, {"label": "sil", "start_s": 0.20, "end_s": 0.30, "speech": False}, {"label": "b", "start_s": 0.30, "end_s": 0.45, "speech": True}]
    result = optimize_utterance(source, tokens, _TinyTeacher(), {"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]}, arm="OPT_DYNAMIC_3", max_gain_db=3.0, steps=1, checkpoints=(0, 1), timeout_s=30)
    output = np.asarray(result["selected_pcm"], dtype=np.int16)
    assert output.shape == source.shape
    protected = np.asarray(result["protected_mask"], dtype=bool)
    assert np.array_equal(output[protected], source[protected])
