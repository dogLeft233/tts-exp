"""Tests for the local Qwen3-TTS provider options."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from tts import faster_qwen3 as provider_module


class _FakeModel:
    def __init__(self):
        self.kwargs = None

    def generate_voice_clone(self, **kwargs):
        self.kwargs = kwargs
        return [np.zeros(2400, dtype=np.float32)], 24000


def test_provider_forwards_configured_max_new_tokens(tmp_path):
    provider = provider_module.FasterQwen3TTSProvider(
        {"faster_qwen3": {"max_new_tokens": 256}}
    )
    model = _FakeModel()
    provider._model = model

    provider.generate_voice_clone(
        text="sample text",
        ref_audio_path=tmp_path / "ref.wav",
        ref_text="sample text",
        language="French",
    )

    assert model.kwargs["max_new_tokens"] == 256
