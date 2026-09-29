from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.tts_evidence_temporal_patch.intervention import (
    build_frame_map,
    build_patch,
    continuous_core_blocks,
    decompose_dynamic,
    mel_frame_clock,
    shift_frame_map,
)


def _mapping(length: int = 60):
    centers = [0.1 + 0.0125 * index for index in range(length)]
    segments = [{"n_start_s": 0.0, "n_end_s": 2.0, "t_start_s": 0.0, "t_end_s": 2.0, "segment_index": 0}]
    frame_map = build_frame_map(centers, centers, segments, direction="n_to_t", support_blocks=[{"segment_indices": [0]}])
    return centers, frame_map


def test_map_does_not_interpolate_across_a_gap() -> None:
    centers = [0.10 + index * 0.05 for index in range(30)]
    segments = [
        {"n_start_s": 0.0, "n_end_s": 0.5, "t_start_s": 0.0, "t_end_s": 0.5},
        {"n_start_s": 1.0, "n_end_s": 2.0, "t_start_s": 1.0, "t_end_s": 2.0},
    ]
    frame_map = build_frame_map(centers, centers, segments, direction="n_to_t", support_blocks=[{"segment_indices": [0]}, {"segment_indices": [1]}])
    assert any(row["status"] == "UNSUPPORTED" and row.get("reason") == "MAPPING_GAP_OR_BOUNDARY" for row in frame_map)
    blocks = continuous_core_blocks(frame_map, minimum_frames=1)
    assert all(len(block) < len(frame_map) for block in blocks)


def test_actual_mel_centers_mark_repeated_tail() -> None:
    clock = mel_frame_clock(20, frame_count=8)
    assert clock[0].center_s == pytest.approx((8.0) / 80.0)
    assert clock[-1].tail_reused
    assert clock[-1].start_column == 4


def test_dynamic_patch_preserves_noncore_and_equalizes_norm() -> None:
    centers, frame_map = _mapping()
    time = np.arange(len(centers), dtype=np.float32)[:, None]
    recipient = np.sin(time / 3.0 + np.arange(512, dtype=np.float32)[None, :] / 19.0).astype(np.float32)
    donor = np.sin(time / 2.7 + np.arange(512, dtype=np.float32)[None, :] / 17.0 + 0.4).astype(np.float32)
    blocks = decompose_dynamic(recipient, donor, frame_map, smoothing_frames=5, minimum_core_frames=25)
    assert blocks
    coherent = build_patch(recipient, blocks, "COHERENT")
    scrambled = build_patch(recipient, blocks, "SCRAMBLED")
    core = np.concatenate([block.recipient_indices for block in blocks])
    outside = np.setdiff1d(np.arange(len(recipient)), core)
    assert np.array_equal(recipient[outside], coherent[outside])
    assert np.array_equal(recipient[outside], scrambled[outside])
    delta = blocks[0].delta
    assert np.linalg.norm(delta) == pytest.approx(np.linalg.norm(np.roll(delta, len(delta) // 2, axis=0)), rel=1e-6)


def test_shift_does_not_wrap() -> None:
    _centers, frame_map = _mapping(30)
    shifted = shift_frame_map(frame_map, -100)
    assert all(row["status"] == "UNSUPPORTED" for row in shifted)
