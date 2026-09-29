from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.experiments.tts_evidence_temporal_patch.worker import PatchedForward, capture_bottleneck


class _Encoder(torch.nn.Module):
    def forward(self, value):
        mean = value.mean(dim=(1, 2, 3), keepdim=True)
        return mean.repeat(1, 512, 1, 1)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = _Encoder()

    def forward(self, mel, image):
        _ = image
        self.audio_encoder(mel)
        return torch.zeros((mel.shape[0], 3, 96, 96), dtype=mel.dtype, device=mel.device)


def test_capture_keeps_absolute_indices_and_removes_hook() -> None:
    model = _Model()
    chunks = [np.ones((80, 16), dtype=np.float32) * index for index in range(5)]
    images = [np.zeros((96, 96, 6), dtype=np.float32) for _ in chunks]
    result = capture_bottleneck(model, chunks, images, device="cpu", batch_size=2, frame_indices=[10, 11, 12, 13, 14])
    assert result["frame_indices"] == [10, 11, 12, 13, 14]
    assert result["values"].shape == (5, 512)
    assert not model.audio_encoder._forward_hooks


def test_patch_hook_is_cleaned_after_forward() -> None:
    model = _Model()
    patches = np.arange(6 * 512, dtype=np.float32).reshape(6, 512)
    with PatchedForward(model, patches, device="cpu") as hook:
        hook.set_frame_indices([2, 4])
        model(torch.zeros((2, 1, 80, 16)), torch.zeros((2, 6, 96, 96)))
        assert hook.calls == 1
    assert not model.audio_encoder._forward_hooks
