from __future__ import annotations

import numpy as np
import torch

from scripts.experiments.lrs3_dac16k_codec_identity.audio import _model_forward


class FakeDAC:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def __call__(self, waveform: torch.Tensor, *, sample_rate: int, n_quantizers: int | None) -> dict[str, torch.Tensor]:
        self.calls.append((sample_rate, n_quantizers))
        count = waveform.shape[-1]
        return {
            "audio": waveform,
            "codes": torch.zeros((1, 2, count // 320), dtype=torch.long),
            "latents": torch.zeros((1, 4, count // 320)),
            "z": torch.zeros((1, 4, count // 320)),
        }


def test_model_forward_passes_all_quantizers_and_crops_only_tail() -> None:
    model = FakeDAC()
    waveform = (0.25 * np.sin(np.linspace(0.0, 50.0, 1025))).astype(np.float32)
    result, qc = _model_forward(model, (waveform * 32768).astype(np.int16), "cpu")
    assert model.calls == [(16_000, None)]
    assert result["waveform"].shape == (1025,)
    assert result["provenance"]["model_internal_padded_sample_count"] == 1600
    assert qc["raw_finite"] is True
