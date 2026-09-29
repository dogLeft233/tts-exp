from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lrs3_wav2lip_timing_transfer import config
from scripts.experiments.lrs3_wav2lip_timing_transfer.protocol import (
    audio_forward_map,
    reconstruct_warped_pcm,
    timing_masks,
)


def test_audio_map_has_frozen_signs_and_exact_nearest_even_reconstruction() -> None:
    samples = (np.arange(64_000, dtype=np.int64) % 257 - 128).astype("<i2")
    pcm, mapped = reconstruct_warped_pcm(samples.tobytes())
    expected = np.rint(np.interp(mapped, np.arange(samples.size, dtype=np.float64), samples.astype(np.float64))).astype("<i2").tobytes()
    assert pcm == expected
    assert mapped[0] == 0.0
    assert mapped[-1] == samples.size - 1
    assert np.all(np.diff(mapped) > 0.0)
    assert np.max(mapped - np.arange(samples.size)) > 1_900
    assert np.min(mapped - np.arange(samples.size)) < -1_900


def test_plus_minus_masks_use_forward_and_inverse_map() -> None:
    sample_count = 64_000
    mapped = audio_forward_map(sample_count)
    masks = timing_masks(sample_count, {arm: 100 for arm in config.VIDEO_ARMS})
    assert len(masks["plus_rows"]) >= config.MIN_LOCAL_ROWS
    assert len(masks["minus_rows"]) >= config.MIN_LOCAL_ROWS
    assert all(masks["d_by_row"][str(row)] >= 2.5 and masks["a_by_row"][str(row)] >= 2.5 for row in masks["plus_rows"])
    assert all(masks["d_by_row"][str(row)] <= -2.5 and masks["a_by_row"][str(row)] <= -2.5 for row in masks["minus_rows"])
    assert masks["forward_mapping_sha256"]
    assert mapped.size == sample_count


def test_short_common_support_is_blocked() -> None:
    with pytest.raises(Exception, match="fewer than"):
        timing_masks(64_000, {arm: 30 for arm in config.VIDEO_ARMS})
