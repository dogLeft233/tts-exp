from __future__ import annotations

import hashlib

import numpy as np
import soundfile as sf

from scripts.experiments.fresh_source_inputs.validate import _pcm16_identity


def test_pcm16_identity_hashes_serialized_pcm_bytes(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    samples = np.array([0, 1, -2, 32767, -32768], dtype=np.int16)
    sf.write(path, samples, 16_000, subtype="PCM_16")

    decoded, sample_rate, digest = _pcm16_identity(path)

    assert sample_rate == 16_000
    assert decoded.dtype == np.int16
    assert decoded.tolist() == samples.tolist()
    assert digest == hashlib.sha256(samples.tobytes()).hexdigest()
