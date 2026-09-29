from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lrs3_mfa_dtw_replacement.dtw import (
    NOMINAL_SUPPORT_SHARED_FRAME_OWNERSHIP,
    DTWError,
    build_hard_dtw_mapping,
    cosine_cost_matrix,
    hard_dtw_path,
    phone_frame_indices,
)


def test_hard_dtw_reproduces_identity_path() -> None:
    result = hard_dtw_path(np.eye(3, dtype=np.float64), band_ratio=0.5)
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (2, 2)
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(result.path, result.path[1:]))


def test_hard_dtw_tie_order_is_vertical_then_horizontal_then_diagonal() -> None:
    result = hard_dtw_path(np.zeros((2, 2), dtype=np.float64), band_ratio=1.0)
    assert result.path == ((0, 0), (0, 1), (1, 1))


def test_hard_dtw_rejects_band_without_complete_path() -> None:
    with pytest.raises(DTWError, match="no complete"):
        hard_dtw_path(np.zeros((2, 3), dtype=np.float64), band_ratio=0.0)


def test_mapping_keeps_silence_linear_and_uses_tts_only_for_speech() -> None:
    rng = np.random.default_rng(3)
    natural = rng.normal(size=(8, 2)).astype(np.float32)
    tts = rng.normal(size=(10, 2)).astype(np.float32)
    natural_tokens = [
        {"label": "aa", "token": "aa", "start_s": 0.0, "end_s": 0.08, "silence": False},
        {"label": "sil", "token": "sil", "start_s": 0.08, "end_s": 0.16, "silence": True},
    ]
    tts_tokens = [
        {"label": "aa", "token": "aa", "start_s": 0.0, "end_s": 0.16, "silence": False},
        {"label": "sil", "token": "sil", "start_s": 0.16, "end_s": 0.24, "silence": True},
    ]
    mapping, diagnostics, trace = build_hard_dtw_mapping(natural, tts, natural_tokens, tts_tokens)
    assert [row.natural_frame_index for row in mapping] == list(range(8))
    assert [row.mapping_type for row in mapping] == ["hard_dtw"] * 4 + ["matched_phone"] * 4
    assert diagnostics["speech_phone_count"] == 1
    assert diagnostics["silence_frames"] == 4
    assert len(trace["phones"][0]["path"]) >= 1
    assert np.all(np.diff([row.left_frame_index + row.interpolation_alpha for row in mapping]) >= 0)


def test_shared_nominal_support_assigns_boundary_frame_to_both_phones() -> None:
    tokens = [
        {"label": "aa", "start_s": 0.0, "end_s": 0.01, "silence": False},
        {"label": "bb", "start_s": 0.01, "end_s": 0.04, "silence": False},
    ]
    result = phone_frame_indices(
        tokens,
        2,
        policy=NOMINAL_SUPPORT_SHARED_FRAME_OWNERSHIP,
    )
    assert result[0] == [0]
    assert result[1] == [0, 1]


def test_short_tts_phone_can_form_same_phone_dtw_path() -> None:
    natural = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    tts = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    natural_tokens = [
        {"label": "aa", "start_s": 0.0, "end_s": 0.04, "silence": False},
    ]
    tts_tokens = [
        {"label": "aa", "start_s": 0.0, "end_s": 0.01, "silence": False},
        {"label": "sil", "start_s": 0.01, "end_s": 0.04, "silence": True},
    ]
    mapping, diagnostics, trace = build_hard_dtw_mapping(
        natural,
        tts,
        natural_tokens,
        tts_tokens,
        frame_ownership_policy=NOMINAL_SUPPORT_SHARED_FRAME_OWNERSHIP,
    )
    assert len(mapping) == 2
    assert diagnostics["frame_ownership_policy"] == NOMINAL_SUPPORT_SHARED_FRAME_OWNERSHIP
    assert trace["phones"][0]["tts_frame_indices"] == [0]


def test_phone_frame_indices_rejects_unknown_policy() -> None:
    tokens = [{"label": "aa", "start_s": 0.0, "end_s": 0.02, "silence": False}]
    with pytest.raises(DTWError, match="unknown frame ownership policy"):
        phone_frame_indices(tokens, 1, policy="unknown")
    with pytest.raises(DTWError, match="zero vector"):
        cosine_cost_matrix(np.zeros((1, 2)), np.ones((1, 2)))
