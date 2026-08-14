from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from wavlm_knn_vc_adapter import WavLMKNNVCAdapter  # noqa: E402


class FakeModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.eval()


class FakeKNNVC:
    sr = 16_000
    hop_length = 320

    def __init__(self, dim: int = 1024) -> None:
        self.wavlm = FakeModule()
        self.hifigan = FakeModule()
        self.dim = dim

    def get_features(self, waveform: torch.Tensor, vad_trigger_level: int = 0) -> torch.Tensor:
        frames = waveform.shape[-1] // 320
        return torch.ones(frames, self.dim, device=waveform.device)

    def vocode(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, features.shape[1] * 320, device=features.device)


def test_adapter_freezes_modules_and_enforces_shapes() -> None:
    model = FakeKNNVC()
    adapter = WavLMKNNVCAdapter(model, device="cpu")
    assert not any(parameter.requires_grad for parameter in model.wavlm.parameters())
    features = adapter.extract(torch.zeros(1, 1280))
    assert features.shape == (4, 1024)
    waveform = adapter.vocode(features)
    assert waveform.shape == (1280,)
    assert adapter.metadata()["loudness_normalization"] is False


def test_adapter_rejects_hubert_768_features() -> None:
    adapter = WavLMKNNVCAdapter(FakeKNNVC(), device="cpu")
    with pytest.raises(ValueError, match="1024"):
        adapter.vocode(torch.zeros(2, 768))


def test_adapter_rejects_wrong_encoder_dimension() -> None:
    adapter = WavLMKNNVCAdapter(FakeKNNVC(dim=768), device="cpu")
    with pytest.raises(ValueError, match="1024"):
        adapter.extract(torch.zeros(1, 1280))
