from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_waveform_gain.construct import construct_gains


def test_negative_int16_peak_does_not_overflow():
    result = construct_gains(np.asarray([-32768, 0, 32767], dtype=np.int16))
    assert result["status"] == "INPUT_DEGENERATE"
    assert result["reason"] == "no_amplification_headroom"


def test_silence_is_cohort_degenerate():
    assert construct_gains(np.zeros(8, dtype=np.int16))["status"] == "INPUT_DEGENERATE"


def test_rint_is_ties_to_even_and_no_clip():
    result = construct_gains(np.asarray([-1000, 1000], dtype=np.int16))
    assert result["plus"].dtype == np.int16
    assert np.max(np.abs(result["plus"].astype(np.int64))) <= 32767
    assert np.rint(0.5) == 0.0


def test_headroom_rule_does_not_replace_with_attenuation():
    result = construct_gains(np.asarray([-32768, 32767], dtype=np.int16))
    assert result["gain_plus"] <= 1.0 or result["status"] == "INPUT_DEGENERATE"
