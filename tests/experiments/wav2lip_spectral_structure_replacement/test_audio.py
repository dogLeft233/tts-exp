from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_spectral_structure_replacement import audio, config
from scripts.experiments.wav2lip_spectral_structure_replacement.common import (
    ExperimentError,
)


def _pcm(seed: int, length: int = 16_000) -> bytes:
    rng = np.random.default_rng(seed)
    values = np.rint(rng.normal(0.0, 0.08, length) * 32768.0).clip(-32768, 32767).astype("<i2")
    return values.tobytes()


def test_candidates_preserve_length_and_n_identity() -> None:
    natural = _pcm(1)
    mfa = _pcm(2)
    candidates = audio.construct_candidates(natural, mfa)

    assert candidates[config.ARM_N][0] == natural
    assert candidates[config.ARM_N_REPEAT][0] == natural
    for arm in (config.ARM_RT, config.ARM_MAG, config.ARM_ENV):
        pcm, metadata = candidates[arm]
        assert len(pcm) == len(natural)
        assert metadata["length"] == len(natural) // 2
        assert metadata["alpha"] == (0.0 if arm == config.ARM_RT else config.ALPHA)
        assert metadata["finite"] is True
        assert metadata["saturated"] is False


def test_empty_or_mismatched_audio_is_rejected() -> None:
    with pytest.raises(ExperimentError):
        audio.construct_candidates(b"", b"")
    with pytest.raises(ExperimentError):
        audio.construct_candidates(_pcm(1), _pcm(2, 15_999))
