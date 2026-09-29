from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError
from scripts.experiments.mfa_linear_video_retiming.scorer import (
    crop_zero_padded,
    distance_matrix,
    make_audio_tensor,
    make_video_tensor,
    score_embeddings,
    support_indices,
    validate_checkpoint_state,
)


def test_video_tensor_keeps_bgr_0_to_255_and_builds_five_frame_windows() -> None:
    cropped = np.zeros((6, 224, 224, 3), dtype=np.uint8)
    cropped[2, 4, 9] = [12, 128, 255]

    tensor = make_video_tensor(cropped)

    assert tensor.shape == (2, 3, 5, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor.min() == 0.0 and tensor.max() == 255.0
    assert tensor[0, :, 2, 4, 9].tolist() == [12.0, 128.0, 255.0]


def test_audio_tensor_uses_13_by_20_windows_at_four_mfcc_frames_per_video_frame() -> None:
    mfcc = np.arange(13 * 24, dtype=np.float32).reshape(13, 24)

    tensor = make_audio_tensor(mfcc)

    assert tensor.shape == (2, 1, 13, 20)
    assert tensor.dtype == np.float32
    assert np.array_equal(tensor[1, 0], mfcc[:, 4:24])


def test_score_crop_uses_bgr_xyxy_and_zero_padding() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 1] = [3, 21, 240]
    frame[1, 2] = [90, 80, 70]

    crop = crop_zero_padded(frame, [1, -1, 4, 2], output_size=3)

    assert crop.shape == (3, 3, 3)
    assert np.array_equal(crop[0], np.zeros((3, 3), dtype=np.uint8))
    assert crop[1, 0].tolist() == [3, 21, 240]
    assert crop[2, 1].tolist() == [90, 80, 70]


def test_checkpoint_validation_allows_only_explicit_inert_counters() -> None:
    checkpoint = {"weight": torch.ones(2, 3), "bias": torch.zeros(3)}
    model = {**checkpoint, "bn.num_batches_tracked": torch.zeros((), dtype=torch.long)}

    strict_state, audit = validate_checkpoint_state(
        checkpoint, model, allowed_missing=("bn.num_batches_tracked",)
    )

    assert set(strict_state) == set(model)
    assert audit["strict_load_keyset_match_after_buffer_completion"] is True
    assert audit["missing_batchnorm_counter_keys"] == ["bn.num_batches_tracked"]
    with pytest.raises(ProtocolError, match="keyset mismatch"):
        validate_checkpoint_state(checkpoint, model, allowed_missing=())
    with pytest.raises(ProtocolError, match="shapes differ"):
        validate_checkpoint_state({"weight": torch.ones(3, 2), "bias": torch.zeros(3)}, model, allowed_missing=("bn.num_batches_tracked",))


def test_distance_matrix_matches_pytorch_pairwise_distance_epsilon() -> None:
    visual = np.zeros((40, 1024), dtype=np.float32)
    audio = np.zeros((40, 1024), dtype=np.float32)
    matrix = distance_matrix(visual, audio, n_support=40, torch=torch)
    expected = float(torch.nn.functional.pairwise_distance(
        torch.zeros((1, 1024), dtype=torch.float32),
        torch.zeros((1, 1024), dtype=torch.float32),
        p=2,
        eps=1e-6,
    ).item())

    assert matrix.shape == (40, 31)
    assert matrix.dtype == np.float32
    assert matrix[15, 15] == pytest.approx(expected, rel=0, abs=1e-12)
    assert np.isnan(matrix[0, 0])
    assert np.isnan(matrix[-1, -1])


def test_known_audio_lag_has_declared_offset_sign_and_local_support() -> None:
    rng = np.random.default_rng(82)
    n = 81
    audio = rng.normal(size=(n + 15, 1024)).astype(np.float32)
    visual = audio[2:2 + n].copy()

    metrics = score_embeddings(
        visual,
        audio,
        frame_count=90,
        valid_frame_count=86,
        audio_sample_count=90 * 640,
        torch=torch,
    )

    assert metrics["best_lag"] == 2
    assert metrics["offset"] == -2
    assert metrics["W"][0] == 15
    assert metrics["W"][-1] == metrics["conservative_window_count"] - 16
    assert len(metrics["local_windows"]) == 2
    assert metrics["distance_matrix_shape"] == [81, 31]
    assert np.isfinite(np.asarray(metrics["distance_matrix"])[metrics["W"]]).all()


def test_frozen_support_is_not_reduced_below_twenty_five_rows() -> None:
    with pytest.raises(ProtocolError, match="INSUFFICIENT_SUPPORT"):
        support_indices(54)


def test_metric_scoring_rejects_wrong_embedding_dimension() -> None:
    with pytest.raises(ProtocolError, match="1024"):
        distance_matrix(np.zeros((60, 8)), np.zeros((60, 8)), n_support=50, torch=torch)
