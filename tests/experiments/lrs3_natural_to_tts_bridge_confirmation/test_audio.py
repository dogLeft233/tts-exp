import numpy as np
import pytest

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation import config
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.audio import (
    construct_arm,
    float_to_pcm16,
    local_swap_pcm16,
    phase_preserving_blend,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.protocol import (
    ProtocolError,
)


def test_local_swap_preserves_bytes_as_a_permutation_and_exact_boundaries() -> None:
    natural = np.arange(1024, dtype=np.int16)
    swapped, boundaries = local_swap_pcm16(natural)
    assert boundaries == {"length": 1024, "b1": 256, "b2": 512, "b3": 768}
    np.testing.assert_array_equal(swapped[:256], natural[:256])
    np.testing.assert_array_equal(swapped[256:512], natural[512:768])
    np.testing.assert_array_equal(swapped[512:768], natural[256:512])
    np.testing.assert_array_equal(swapped[768:], natural[768:])
    assert swapped.size == natural.size
    assert sorted(swapped.tolist()) == sorted(natural.tolist())


def test_local_swap_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional int16"):
        local_swap_pcm16(np.zeros((2, 2), dtype=np.int16))


def test_float_to_pcm16_rejects_clipping() -> None:
    with pytest.raises(ValueError, match="clipped"):
        float_to_pcm16(np.asarray([0.0, 1.0]))


def test_bridge_is_deterministic_and_exact_length() -> None:
    grid = np.arange(8192, dtype=np.float64) / 8192.0
    natural = np.rint(12_000 * np.sin(2 * np.pi * 220 * grid)).astype(np.int16)
    mfa_linear = np.rint(10_000 * np.sin(2 * np.pi * 330 * grid)).astype(np.int16)
    first, first_meta = phase_preserving_blend(natural, mfa_linear)
    second, second_meta = phase_preserving_blend(natural, mfa_linear)
    np.testing.assert_array_equal(first, second)
    assert first.size == natural.size
    assert first_meta == second_meta
    assert first_meta["alpha"] == config.BRIDGE_ALPHA
    assert first_meta["phase_policy"] == "natural_phase"
    assert first_meta["rms_scale"] > 0.0


def test_bridge_rejects_mismatched_lengths() -> None:
    natural = np.ones(4096, dtype=np.int16)
    with pytest.raises(ProtocolError, match="lengths differ"):
        phase_preserving_blend(natural, natural[:-1])


def test_registered_arm_dispatch_preserves_natural_bytes() -> None:
    natural = np.arange(4096, dtype=np.int16)
    mfa_linear = np.flip(natural).copy()
    for arm in ("N", "N_REPEAT"):
        values, metadata = construct_arm(natural, mfa_linear, arm)
        assert values.tobytes() == natural.tobytes()
        assert metadata


def test_registered_arm_dispatch_has_fixed_four_arms() -> None:
    assert config.ARMS == ("N", "N_REPEAT", "LOCAL_SWAP", "BRIDGE_075")
    for arm in config.ARMS:
        values, metadata = construct_arm(np.arange(4096, dtype=np.int16), np.flip(np.arange(4096, dtype=np.int16)).copy(), arm)
        assert values.dtype == np.int16
        assert values.size == 4096
        assert metadata
