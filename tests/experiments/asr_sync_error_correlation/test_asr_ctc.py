import numpy as np
import pytest

from scripts.experiments.asr_sync_error_correlation.asr_ctc import collapse_ctc, greedy_decode, tokens_to_words


def test_ctc_collapse_keeps_repeated_tokens_separated_by_blank():
    ids = np.array([0, 1, 1, 0, 1, 2, 2, 0])
    tokens = collapse_ctc(ids, blank_id=0, delimiter_id=2, frame_stride_s=0.02, id_to_token=lambda i: {1: "A", 2: "|"}[i])
    assert [row["token_id"] for row in tokens] == [1, 1, 2]
    assert tokens[0]["frames"] == [1, 2]
    assert tokens[1]["frames"] == [4]


def test_greedy_decode_words_and_confidence():
    logits = np.full((7, 4), -5.0, dtype=np.float32)
    ids = [0, 1, 1, 0, 2, 3, 0]
    for frame, token_id in enumerate(ids):
        logits[frame, token_id] = 0.0
    result = greedy_decode(logits, blank_id=0, delimiter_id=2, frame_stride_s=0.1, id_to_token=lambda i: {0: "", 1: "A", 2: "|", 3: "B"}[i])
    assert result["transcript"] == "A B"
    assert result["words"][0]["start_s"] == pytest.approx(0.1)
    assert result["words"][0]["end_s"] == pytest.approx(0.3)
    assert 0 < result["words"][0]["confidence"] <= 1
