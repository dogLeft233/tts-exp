from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from knn_vc_retrieval import (  # noqa: E402
    RetrievalConfig,
    frame_owners,
    mfa_linear_target,
    retrieve,
    unmatched_silence_intervals,
)


def _tokens():
    return [
        {"token": "sil", "start_s": 0.0, "end_s": 0.02, "is_silence": True},
        {"token": "a", "start_s": 0.02, "end_s": 0.06},
        {"token": "b", "start_s": 0.06, "end_s": 0.10},
    ]


def test_frame_owners_include_silence_and_relative_positions() -> None:
    owners = frame_owners(5, _tokens())
    assert owners[0].is_silence
    assert [owner.label for owner in owners] == ["<sil>", "a", "a", "b", "b"]
    assert owners[1].relative_position < owners[2].relative_position


def test_same_phone_and_wrong_phone_respect_labels() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    qowners = frame_owners(2, [{"token": "a", "start_s": 0.0, "end_s": 0.04}])
    cowners = frame_owners(
        3,
        [
            {"token": "a", "start_s": 0.0, "end_s": 0.04},
            {"token": "b", "start_s": 0.04, "end_s": 0.06},
        ],
    )
    same = retrieve(query, candidates, qowners, cowners, RetrievalConfig("same_phone", topk=1))
    wrong = retrieve(query, candidates, qowners, cowners, RetrievalConfig("wrong_phone", topk=1))
    assert same.indices[:, 0].tolist() == [0, 1]
    assert wrong.indices[:, 0].tolist() == [2, 2]


def test_position_penalty_changes_tied_choice_deterministically() -> None:
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    tokens = [{"token": "a", "start_s": 0.0, "end_s": 0.04}]
    owners = frame_owners(2, tokens)
    basic = retrieve(query, candidates, owners, owners, RetrievalConfig("same_phone", topk=1))
    positioned = retrieve(
        query, candidates, owners, owners,
        RetrievalConfig("same_phone_position", topk=1, position_weight=1.0, continuity_weight=0.0),
    )
    assert basic.indices[:, 0].tolist() == [0, 0]
    assert positioned.indices[:, 0].tolist() == [0, 1]


def test_missing_same_phone_candidate_fails_closed() -> None:
    query = torch.ones(1, 2)
    candidates = torch.ones(1, 2)
    qowners = frame_owners(1, [{"token": "a", "start_s": 0.0, "end_s": 0.02}])
    cowners = frame_owners(1, [{"token": "b", "start_s": 0.0, "end_s": 0.02}])
    with pytest.raises(ValueError, match="no candidate"):
        retrieve(query, candidates, qowners, cowners, RetrievalConfig("same_phone"))


def test_mfa_linear_tolerates_mfa_phone_replacement_with_explicit_fallback() -> None:
    natural_tokens = [
        {"token": "sil", "start_s": 0.0, "end_s": 0.02, "is_silence": True},
        {"token": "w", "start_s": 0.02, "end_s": 0.04},
        {"token": "o", "start_s": 0.04, "end_s": 0.06},
    ]
    tts_tokens = [
        {"token": "sil", "start_s": 0.0, "end_s": 0.02, "is_silence": True},
        {"token": "ʔ", "start_s": 0.02, "end_s": 0.04},
        {"token": "o", "start_s": 0.04, "end_s": 0.06},
    ]
    target, metadata = mfa_linear_target(3, torch.arange(3.0).unsqueeze(1), natural_tokens, tts_tokens)
    assert target.shape == (3, 1)
    assert metadata["fallback_frames"] == 1
    assert metadata["coverage"] == pytest.approx(2 / 3)


def test_mfa_linear_maps_each_token_to_natural_grid() -> None:
    natural_tokens = [
        {"token": "a", "start_s": 0.0, "end_s": 0.04},
        {"token": "b", "start_s": 0.04, "end_s": 0.08},
    ]
    tts_tokens = [
        {"token": "a", "start_s": 0.0, "end_s": 0.02},
        {"token": "b", "start_s": 0.02, "end_s": 0.06},
    ]
    tts = torch.tensor([[1.0], [2.0], [4.0]])
    target, metadata = mfa_linear_target(4, tts, natural_tokens, tts_tokens)
    assert target.shape == (4, 1)
    assert target[0, 0] == pytest.approx(1.0)
    assert target[0, 0] <= target[1, 0] < target[2, 0]
    assert metadata["coverage"] == 1.0


def test_unmatched_natural_pause_uses_tts_silence_without_changing_speech() -> None:
    natural = [
        {"token": "a", "start_s": 0.0, "end_s": 0.04},
        {"token": "sil", "start_s": 0.04, "end_s": 0.16, "is_silence": True},
        {"token": "b", "start_s": 0.16, "end_s": 0.20},
    ]
    tts = [
        {"token": "sil", "start_s": 0.0, "end_s": 0.04, "is_silence": True},
        {"token": "a", "start_s": 0.04, "end_s": 0.08},
        {"token": "b", "start_s": 0.08, "end_s": 0.12},
    ]
    features = torch.tensor([[0.0], [0.0], [10.0], [11.0], [20.0], [21.0]])
    old, _ = mfa_linear_target(10, features, natural, tts)
    new, meta = mfa_linear_target(10, features, natural, tts, silence_fallback="tts_silence")
    assert unmatched_silence_intervals(natural, tts) == [(0.04, 0.16)]
    assert torch.equal(new[:2], old[:2])
    assert torch.equal(new[8:], old[8:])
    assert torch.equal(new[2:8], torch.zeros(6, 1))
    assert meta["silence_fallback_frames"] == 6
    assert meta["fallback_frames"] == 6


def test_silence_repair_preserves_speech_fallback_and_handles_no_tts_silence() -> None:
    natural = [
        {"token": "a", "start_s": 0.0, "end_s": 0.02},
        {"token": "sp", "start_s": 0.02, "end_s": 0.14},
        {"token": "b", "start_s": 0.14, "end_s": 0.16},
    ]
    tts = [
        {"token": "a", "start_s": 0.0, "end_s": 0.02},
        {"token": "x", "start_s": 0.02, "end_s": 0.04},
        {"token": "b", "start_s": 0.04, "end_s": 0.06},
    ]
    features = torch.arange(3.0).unsqueeze(1)
    old, _ = mfa_linear_target(8, features, natural, tts)
    new, meta = mfa_linear_target(8, features, natural, tts, silence_fallback="tts_silence")
    assert torch.equal(new, old)
    assert meta["silence_without_tts_frames"] == 6
    assert unmatched_silence_intervals(natural, tts) == [(0.02, 0.14)]
    with pytest.raises(ValueError, match="unsupported silence fallback"):
        mfa_linear_target(8, features, natural, tts, silence_fallback="invalid")


def test_unknown_speech_is_not_muted_or_used_as_silence_source() -> None:
    natural = [
        {"token": "a", "start_s": 0.0, "end_s": 0.02},
        {"token": "spn", "start_s": 0.02, "end_s": 0.14, "is_unknown_speech": True},
        {"token": "b", "start_s": 0.14, "end_s": 0.16},
    ]
    tts = [
        {"token": "a", "start_s": 0.0, "end_s": 0.02},
        {"token": "x", "start_s": 0.02, "end_s": 0.04},
        {"token": "b", "start_s": 0.04, "end_s": 0.06},
        {"token": "spn", "start_s": 0.06, "end_s": 0.16, "is_unknown_speech": True},
    ]
    old, _ = mfa_linear_target(8, torch.arange(8.0).unsqueeze(1), natural, tts)
    new, meta = mfa_linear_target(8, torch.arange(8.0).unsqueeze(1), natural, tts, silence_fallback="tts_silence")
    assert unmatched_silence_intervals(natural, tts) == []
    assert torch.equal(new, old)
    assert meta["silence_fallback_frames"] == 0
