from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lrs3_mfa_linear_replacement.candidate_audio import (
    canonical_pcm_s16le,
    candidate_from_features,
    exact_natural_length,
    validate_raw_vocoder_length,
)
from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import FrameMapping


def _mapping(count: int) -> list[FrameMapping]:
    return [FrameMapping(i, 0, "aa", False, "matched_phone", 0, i, min(i + 1, count - 1), 0.0, None) for i in range(count)]


def test_exact_length_only_right_adjusts() -> None:
    cropped, crop_meta = exact_natural_length(np.arange(5), 3)
    padded, pad_meta = exact_natural_length(np.arange(2), 3)
    assert cropped.tolist() == [0, 1, 2]
    assert padded.tolist() == [0, 1, 0]
    assert crop_meta["action"] == "right_crop"
    assert pad_meta["action"] == "right_zero_pad"


def test_candidate_uses_natural_frame_count_and_vocoder_contract() -> None:
    natural = np.ones((4, 1024), dtype=np.float32)
    tts = np.arange(4 * 1024, dtype=np.float32).reshape(4, 1024)
    output, metadata = candidate_from_features(
        natural_features=natural,
        tts_features=tts,
        mapping=_mapping(4),
        natural_audio_samples=1000,
        vocode=lambda conditioning: np.ones(conditioning.shape[0] * 320, dtype=np.float32) * 0.1,
    )
    assert output.shape == (1000,)
    assert metadata["conditioning_frames"] == 4
    assert metadata["length_adjustment"]["action"] == "right_crop"


def test_raw_vocoder_length_must_match_hop_contract() -> None:
    with pytest.raises(ValueError, match="does not match model contract"):
        validate_raw_vocoder_length(np.zeros(3), 1)


def test_pcm_rejects_clipping_and_roundtrips_length() -> None:
    pcm, metadata = canonical_pcm_s16le(np.array([0.0, 0.25, -0.25], dtype=np.float32), 3)
    assert len(pcm) == 6
    assert metadata["subtype"] == "PCM_16"
    with pytest.raises(ValueError, match="clipped"):
        canonical_pcm_s16le(np.array([1.0, 0.0], dtype=np.float32), 2)
