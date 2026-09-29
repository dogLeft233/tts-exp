from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_natural_content_residual_continuation import config
from scripts.experiments.wav2lip_natural_content_residual_continuation.common import ProtocolError
from scripts.experiments.wav2lip_natural_content_residual_continuation.runner import _matched_matrix


def test_only_delay_control_has_translated_support() -> None:
    assert config.NATURAL_LAG_START == -15
    assert config.DELAY_LAG_START == -10
    assert list(range(config.NATURAL_LAG_START, config.NATURAL_LAG_START + config.MATRIX_COLUMNS)) == list(range(-15, 16))
    assert list(range(config.DELAY_LAG_START, config.DELAY_LAG_START + config.MATRIX_COLUMNS)) == list(range(-10, 21))
    assert config.MAX_NEW_VIDEO_COUNT == 50
    assert config.MAX_NEW_SCORE_COUNT == 52


def test_matched_domain_uses_real_embedding_support() -> None:
    visual = np.zeros((config.EMBEDDING_ROWS, config.EMBEDDING_DIM), dtype=np.float32)
    audio = np.zeros_like(visual)
    matrix = _matched_matrix(visual, audio, config.DELAY_LAG_START)
    assert matrix.shape == (len(config.U_ROWS), config.MATRIX_COLUMNS)
    assert np.isfinite(matrix).all()


def test_matched_domain_rejects_missing_support() -> None:
    visual = np.zeros((config.EMBEDDING_ROWS, config.EMBEDDING_DIM), dtype=np.float32)
    audio = np.zeros_like(visual)
    with pytest.raises(ProtocolError, match="lacks real support"):
        _matched_matrix(visual[:40], audio[:40], config.DELAY_LAG_START)
