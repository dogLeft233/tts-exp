from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.lrs3_dac16k_codec_identity import config
from scripts.experiments.lrs3_dac16k_codec_identity.audio import (
    float_to_pcm16,
    pcm16_to_float,
    read_pcm16,
    write_pcm16,
)


def test_pcm16_round_trip_preserves_sample_grid(tmp_path: Path) -> None:
    values = np.array([0, 1, -1, 32767, -32768] * 512, dtype=np.int16)
    path = tmp_path / "x.wav"
    write_pcm16(path, values)
    decoded, metadata = read_pcm16(path)
    assert np.array_equal(decoded, values)
    assert metadata["sample_rate"] == config.SAMPLE_RATE
    assert metadata["sample_width"] == 2


def test_float_to_pcm16_rejects_clipping_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="clipped"):
        float_to_pcm16(np.array([1.0] * 1024, dtype=np.float32))
    with pytest.raises(FloatingPointError, match="non-finite"):
        float_to_pcm16(np.array([np.nan] * 1024, dtype=np.float32))


def test_pcm16_float_conversion_is_deterministic() -> None:
    values = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    assert np.array_equal(pcm16_to_float(values), pcm16_to_float(values))
