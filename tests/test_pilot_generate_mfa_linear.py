from __future__ import annotations

import numpy as np

from scripts.pilot_generate_mfa_linear import mute_missing_pauses


def test_mute_missing_pauses_keeps_clock_and_other_speech() -> None:
    audio = np.ones(3200, dtype=np.float32)
    output = mute_missing_pauses(audio, [(0.04, 0.16)])
    assert output.shape == audio.shape
    assert np.array_equal(output[:640], audio[:640])
    assert np.array_equal(output[2560:], audio[2560:])
    assert output[960:2240].max() == 0.0
    assert output[640] == 1.0
    assert output[2559] == 1.0
    assert np.array_equal(audio, np.ones_like(audio))


def test_mute_missing_pauses_clips_tail_to_audio_length() -> None:
    audio = np.ones(1600, dtype=np.float32)
    output = mute_missing_pauses(audio, [(0.06, 0.12)])
    assert output.shape == audio.shape
    assert output[0] == 1.0
    assert output[-1] == 1.0
    assert output[1200] == 0.0
