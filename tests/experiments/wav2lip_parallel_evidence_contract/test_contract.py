from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_parallel_evidence_contract import common


def test_official_tail_chunk_is_292_and_unique_five_frame_support():
    chunks = common.official_chunks()
    assert len(chunks) == 93 and chunks[-1][0] == 292
    assert len(common.supports()[30]) == 28


def test_embedding_matrix_uses_fixed_q_and_rejects_short_support():
    visual = np.zeros((88, 4), dtype=np.float64)
    audio = np.zeros((88, 4), dtype=np.float64)
    matrix = common.matrix_from_embeddings(visual, audio)
    assert matrix.shape == (88, 31)
    with pytest.raises(common.ProtocolError):
        common.matrix_from_embeddings(visual, np.zeros((30, 4)))


def test_metric_anchor_is_natural_argmin_not_candidate_argmin():
    natural = common.metrics(np.tile(np.arange(31, dtype=np.float64), (88, 1)))
    candidate = common.metrics(np.tile(np.arange(31, 0, -1, dtype=np.float64), (88, 1)))
    result = common.gain(candidate, natural)
    assert result["A"] == pytest.approx(natural["curve"][natural["min_index"]] - candidate["curve"][natural["min_index"]])
