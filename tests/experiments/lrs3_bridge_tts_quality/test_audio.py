from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lrs3_bridge_tts_quality.audio import (
    construct_arm,
    phase_preserving_bridge,
    quantize_pcm16,
)
from scripts.experiments.lrs3_bridge_tts_quality.common import (
    ProtocolError,
    decoded_pcm_sha256,
)


def _signals(length: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(length, dtype=np.float64) / 16000.0
    natural = 0.2 * np.sin(2 * np.pi * 220 * time) + 0.04 * np.sin(2 * np.pi * 410 * time)
    target = 0.2 * np.sin(2 * np.pi * 330 * time) + 0.06 * np.sin(2 * np.pi * 510 * time)
    return quantize_pcm16(natural), quantize_pcm16(target)


def test_bridge_is_deterministic_and_b0_uses_same_algorithm() -> None:
    natural, target = _signals()
    first, first_meta = phase_preserving_bridge(natural, target, alpha=0.75)
    second, second_meta = phase_preserving_bridge(natural, target, alpha=0.75)
    assert np.array_equal(first, second)
    assert first_meta == second_meta
    b0, meta = phase_preserving_bridge(natural, natural, alpha=0.0)
    assert b0.shape == natural.shape
    assert meta["alpha"] == 0.0
    assert meta["phase_policy"] == "natural_phase"
    assert decoded_pcm_sha256(b0) == decoded_pcm_sha256(b0)


def test_same_pcm_target_makes_provider_arms_identical() -> None:
    natural, target = _signals()
    local, _ = construct_arm(natural, target, target, "B_LOCAL")
    cloud, _ = construct_arm(natural, target, target, "B_CLOUD")
    assert np.array_equal(local, cloud)


def test_swapping_targets_swaps_corresponding_outputs() -> None:
    natural, local_target = _signals()
    _, cloud_target = _signals(4096)
    cloud_target = np.roll(cloud_target, 31)
    local, _ = construct_arm(natural, local_target, cloud_target, "B_LOCAL")
    cloud, _ = construct_arm(natural, local_target, cloud_target, "B_CLOUD")
    swapped_local, _ = construct_arm(natural, cloud_target, local_target, "B_LOCAL")
    swapped_cloud, _ = construct_arm(natural, cloud_target, local_target, "B_CLOUD")
    assert np.array_equal(local, swapped_cloud)
    assert np.array_equal(cloud, swapped_local)


def test_bridge_rejects_bad_lengths_nonfinite_and_out_of_range() -> None:
    natural, target = _signals()
    with pytest.raises(ProtocolError):
        phase_preserving_bridge(natural, target[:-1])
    with pytest.raises(ProtocolError):
        quantize_pcm16(np.array([np.nan]))
    with pytest.raises(ProtocolError):
        quantize_pcm16(np.array([1.1]))
