from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lrs3_dac16k_codec_identity.fidelity import (
    pair_metrics,
    si_sdr,
    wav2lip_mels,
)
from scripts.experiments.lrs3_dac16k_codec_identity.protocol import ProtocolError


def test_wav2lip_mel_is_80_bins() -> None:
    waveform = np.sin(np.linspace(0.0, 100.0, 16_000, dtype=np.float32))
    mel = wav2lip_mels(waveform)
    assert mel.shape[0] == 80
    assert np.isfinite(mel).all()


def test_fidelity_uses_zero_lag_and_exact_length() -> None:
    waveform = np.sin(np.linspace(0.0, 100.0, 16_000, dtype=np.float32))
    metrics = pair_metrics(waveform, waveform.copy())
    assert metrics["mel_L1"] == pytest.approx(0.0)
    assert metrics["zero_lag_waveform_correlation"] == pytest.approx(1.0)
    assert metrics["lag_search"] is False
    assert metrics["gain_alignment"] is False
    assert metrics["phase_alignment"] is False


def test_fidelity_rejects_length_correction() -> None:
    with pytest.raises(ProtocolError, match="equal sample counts"):
        pair_metrics(np.zeros(2048), np.zeros(2047))


def test_si_sdr_has_no_lag_search() -> None:
    waveform = np.sin(np.linspace(0.0, 30.0, 4096, dtype=np.float64))
    assert si_sdr(waveform, waveform) > 100.0
