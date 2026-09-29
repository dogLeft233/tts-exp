import numpy as np
import pytest

from scripts.experiments.lrs3_phase_preserving_replacement_envelope.audio import (
    construct_arm,
    float_to_pcm16,
    invert_pcm16,
    phase_preserving_blend,
    shift_pcm16,
)
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.protocol import (
    ProtocolError,
)


def test_inv_is_exact_and_non_identical() -> None:
    natural = np.asarray([0, 1, -2, 32767, -32767], dtype=np.int16)
    inverted = invert_pcm16(natural)
    np.testing.assert_array_equal(inverted, np.asarray([0, -1, 2, -32767, 32767], dtype=np.int16))
    assert not np.array_equal(natural, inverted)


def test_inv_rejects_unrepresentable_minimum() -> None:
    with pytest.raises(ProtocolError, match="-32768"):
        invert_pcm16(np.asarray([-32768, 0], dtype=np.int16))


def test_shift_200_has_exact_length_and_delay() -> None:
    natural = np.arange(4000, dtype=np.int16)
    shifted = shift_pcm16(natural)
    assert shifted.size == natural.size
    np.testing.assert_array_equal(shifted[:3200], np.zeros(3200, dtype=np.int16))
    np.testing.assert_array_equal(shifted[3200:], natural[:-3200])


def test_float_to_pcm16_rejects_clipping() -> None:
    with pytest.raises(ValueError, match="clipped"):
        float_to_pcm16(np.asarray([0.0, 1.0]))


def test_phase_preserving_blend_is_deterministic_and_exact_length() -> None:
    grid = np.arange(8192, dtype=np.float64) / 8192.0
    natural = np.rint(12_000 * np.sin(2 * np.pi * 220 * grid)).astype(np.int16)
    mfa_linear = np.rint(10_000 * np.sin(2 * np.pi * 330 * grid)).astype(np.int16)
    first, first_meta = phase_preserving_blend(natural, mfa_linear, 0.50)
    second, second_meta = phase_preserving_blend(natural, mfa_linear, 0.50)
    np.testing.assert_array_equal(first, second)
    assert first.size == natural.size
    assert first_meta == second_meta
    assert first_meta["phase_policy"] == "natural_phase"
    assert first_meta["rms_scale"] > 0.0


def test_only_registered_magnitude_strengths_are_allowed() -> None:
    natural = np.ones(4096, dtype=np.int16)
    with pytest.raises(ProtocolError, match="unregistered"):
        phase_preserving_blend(natural, natural, 0.30)


def test_construct_arm_dispatches_registered_controls() -> None:
    natural = np.arange(4096, dtype=np.int16)
    mfa_linear = np.flip(natural).copy()
    for arm in ("N", "INV", "SHIFT_200", "MAG_025", "MAG_050", "MAG_075", "MAG_100"):
        values, metadata = construct_arm(natural, mfa_linear, arm)
        assert values.dtype == np.int16
        assert values.size == natural.size
        assert metadata
