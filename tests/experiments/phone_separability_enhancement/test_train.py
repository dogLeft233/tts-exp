from __future__ import annotations

import numpy as np
import torch

from scripts.experiments.phone_separability_enhancement.train import train_enhancer


class _TinyTeacher:
    device = "cpu"

    def encode(self, waveform, *, layer: int):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        values = waveform[:, ::64]
        hidden = torch.stack((values, -values, values.square(), torch.sin(values)), dim=-1)
        times = torch.arange(hidden.shape[1], dtype=hidden.dtype, device=hidden.device) * 64.0 / 16_000.0
        return hidden, times, {"layer": layer}


def test_toy_training_runs_and_writes_checkpoint(tmp_path) -> None:
    sample_rate = 16_000
    n = 2_048
    source = np.rint(3000 * np.sin(2 * np.pi * 220 * np.arange(n) / sample_rate)).astype(np.int16)
    tokens = [{"label": "a", "start_s": 0.02, "end_s": 0.06, "speech": True}, {"label": "b", "start_s": 0.06, "end_s": 0.10, "speech": True}]
    result = train_enhancer([{"pair_id": "toy", "source_group": "g", "audio_pcm": source, "tokens": tokens, "mask_info": None}], _TinyTeacher(), {"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]}, seed=1, output_dir=tmp_path, max_steps=2, timeout_s=30, device="cpu")
    assert result["updates"] == 2
    assert (tmp_path / "last.pt").is_file()
    assert all("candidate" not in row and "source" not in row and "mask" not in row for row in result["history"])
