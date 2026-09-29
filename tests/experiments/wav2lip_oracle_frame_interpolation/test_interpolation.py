from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_oracle_frame_interpolation import config
from scripts.experiments.wav2lip_oracle_frame_interpolation.media import (
    build_linear_indices,
    interpolate_frames,
)
from scripts.experiments.wav2lip_oracle_frame_interpolation.validate import (
    _distance_from_embeddings,
    validate_stored_arrays,
)
from scripts.experiments.wav2lip_roi_retiming_oracle.common import OracleError


def test_linear_indices_are_bounded_and_match_float64_contract() -> None:
    indices = build_linear_indices(96_256, 147)
    u = np.asarray(indices["u"], dtype=np.float64)
    j = np.asarray(indices["j"], dtype=np.int64)
    k = np.asarray(indices["k"], dtype=np.int64)
    assert np.all(np.diff(u) >= 0.0)
    assert np.all(j >= 0)
    assert np.all(k < 147)
    assert np.allclose(u - j, indices["w"], atol=0.0, rtol=0.0)


def test_linear_pixels_use_half_up_and_integer_endpoint() -> None:
    source = np.zeros((4, 1, 1, 3), dtype=np.uint8)
    source[0, 0, 0] = [0, 1, 255]
    source[1, 0, 0] = [1, 2, 254]
    indices = {"j": [0, 0, 3, 3], "k": [1, 1, 3, 3], "w": [0.5, 0.0, 0.0, 0.0]}
    output, _evidence = interpolate_frames(source, indices)
    assert output[0, 0, 0].tolist() == [1, 2, 255]
    assert output[1, 0, 0].tolist() == source[0, 0, 0].tolist()
    assert output[2, 0, 0].tolist() == source[3, 0, 0].tolist()


def test_interpolation_rejects_frame_index_out_of_bounds() -> None:
    source = np.zeros((2, 1, 1, 3), dtype=np.uint8)
    with pytest.raises(OracleError, match="invalid"):
        interpolate_frames(source, {"j": [0, 1], "k": [2, 1], "w": [0.5, 0.0]})


def test_validator_distance_is_independent_and_detects_tampering() -> None:
    visual = np.zeros((2, config.EMBEDDING_DIM), dtype=np.float32)
    audio = np.zeros_like(visual)
    visual[1, 0] = 1.0
    expected = _distance_from_embeddings(visual, audio)
    validate_stored_arrays(visual, audio, expected)
    tampered = expected.copy()
    tampered[0, 0] += 0.01
    with pytest.raises(OracleError, match="stored matrix"):
        validate_stored_arrays(visual, audio, tampered)
