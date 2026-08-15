from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_vocoder_input_adapter.py"
spec = importlib.util.spec_from_file_location("train_vocoder_input_adapter", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_adapter_is_identity_initialized_and_bounded() -> None:
    adapter = module.FeatureAffineAdapter(dimension=4, max_scale_delta=0.1, max_bias=0.2)
    features = torch.randn(2, 3, 4)
    output = adapter(features)
    assert torch.equal(output, features)
    with torch.no_grad():
        adapter.scale_logits.fill_(100.0)
        adapter.bias_logits.fill_(-100.0)
    delta = output * 0.0 + adapter(features) - features
    assert torch.all(torch.abs(delta) <= 0.1 * torch.abs(features) + 0.2 + 1e-6)


def test_adapter_rejects_wrong_shape() -> None:
    adapter = module.FeatureAffineAdapter(dimension=4)
    with pytest.raises(ValueError, match="features"):
        adapter(torch.zeros(2, 4))


def test_crop_feature_audio_pads_short_waveform() -> None:
    features = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    waveform = torch.ones(3)
    class ZeroRng:
        def randrange(self, stop: int) -> int:
            return 0

    cropped, target = module._crop_feature_audio(features, waveform, crop_frames=4, rng=ZeroRng())
    assert cropped.shape == (4, 2)
    assert target.shape == (4 * module.FRAME_STRIDE,)
    assert torch.equal(target[:3], waveform)
    assert torch.count_nonzero(target[3:]) == 0


def test_spectral_loss_is_zero_for_identical_waveforms() -> None:
    waveform = torch.randn(1, 1024)
    assert module._spectral_loss(waveform, waveform).item() == pytest.approx(0.0, abs=1e-7)
