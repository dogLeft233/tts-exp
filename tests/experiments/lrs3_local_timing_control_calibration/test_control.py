import numpy as np
import pytest

from scripts.experiments.lrs3_local_timing_control_calibration import config
from scripts.experiments.lrs3_local_timing_control_calibration.common import (
    CalibrationError,
)
from scripts.experiments.lrs3_local_timing_control_calibration.control import (
    construct_arm,
    local_swap_pcm16,
    smooth_warp_pcm16,
)


def test_local_swap_preserves_exact_quarter_permutation() -> None:
    values = np.arange(1024, dtype=np.int16)
    result, boundaries = local_swap_pcm16(values)
    assert boundaries == {"length": 1024, "b1": 256, "b2": 512, "b3": 768}
    np.testing.assert_array_equal(result[:256], values[:256])
    np.testing.assert_array_equal(result[256:512], values[512:768])
    np.testing.assert_array_equal(result[512:768], values[256:512])
    np.testing.assert_array_equal(result[768:], values[768:])
    assert result.tobytes() != values.tobytes()
    assert sorted(result.tolist()) == sorted(values.tolist())


def test_smooth_warp_is_the_registered_map_on_a_ramp() -> None:
    length = config.SMOOTH_WARP_MIN_SAMPLES + 17
    values = np.arange(length, dtype=np.int16)
    result, metadata = smooth_warp_pcm16(values)
    indices = np.arange(length, dtype=np.float64)
    source = indices + config.SMOOTH_WARP_AMPLITUDE_SAMPLES * np.sin(2 * np.pi * indices / (length - 1))
    source[0] = 0.0
    source[-1] = float(length - 1)
    np.testing.assert_array_equal(result, np.rint(source).astype(np.int16))
    assert result[0] == values[0]
    assert result[-1] == values[-1]
    assert metadata["mapping_monotone"] is True
    assert metadata["max_positive_displacement_samples"] == pytest.approx(1920.0)
    assert metadata["max_negative_displacement_samples"] == pytest.approx(-1920.0)


def test_smooth_warp_rejects_short_input_and_preserves_int16_extremes() -> None:
    with pytest.raises(CalibrationError, match="too short"):
        smooth_warp_pcm16(np.zeros(config.SMOOTH_WARP_MIN_SAMPLES - 1, dtype=np.int16))
    values = np.resize(np.asarray([np.iinfo(np.int16).min, np.iinfo(np.int16).max], dtype=np.int16), config.SMOOTH_WARP_MIN_SAMPLES + 1)
    result, _ = smooth_warp_pcm16(values)
    assert result.dtype == np.int16
    assert int(result.min()) >= np.iinfo(np.int16).min
    assert int(result.max()) <= np.iinfo(np.int16).max


def test_arm_dispatch_keeps_both_natural_arms_byte_identical() -> None:
    values = np.arange(config.SMOOTH_WARP_MIN_SAMPLES + 1, dtype=np.int16)
    for arm in config.BASE_ARMS:
        result, metadata = construct_arm(values, arm, config.SMOOTH_BRANCH)
        assert result.tobytes() == values.tobytes()
        assert metadata["pcm_identity"] == "byte_identical_to_natural"
