from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.phone_gain_static_tfg_mfa.conditioning import PhoneVocabulary, build_conditioning
from scripts.experiments.phone_gain_static_tfg_mfa.calibration import _fresh_local_selection
from scripts.experiments.phone_gain_static_tfg_mfa.model import build_model
from scripts.experiments.phone_gain_static_tfg_mfa.quality import (
    construct_local_boundary_control,
    monotonic_pause_match,
    monotonic_phone_match,
    pcm_contract,
)
from scripts.experiments.phone_gain_static_tfg_mfa.render_worker import validate_request
from scripts.experiments.phone_gain_static_tfg_mfa.sync_support import frozen_support_windows, score_distance_matrix


def _tokens():
    return [
        {"label": "a", "start_s": 0.0, "end_s": 0.2, "speech": True},
        {"label": "b", "start_s": 0.2, "end_s": 0.4, "speech": True},
        {"label": "", "start_s": 0.4, "end_s": 0.6, "speech": False, "silence": True},
        {"label": "c", "start_s": 0.6, "end_s": 0.9, "speech": True},
    ]


def test_conditioning_modes_obey_identity_permissions():
    vocab = PhoneVocabulary.from_labels(["a", "b", "c"])
    mask = np.zeros(16000, dtype=bool)
    a = build_conditioning(_tokens(), sample_count=16000, frame_count=100, vocabulary=vocab, duration_stats={"mean": 0.0, "std": 1.0}, protected_mask=mask, mode="AUDIO_FEATURES")
    b = build_conditioning(_tokens(), sample_count=16000, frame_count=100, vocabulary=vocab, duration_stats={"mean": 0.0, "std": 1.0}, protected_mask=mask, mode="BOUNDARY_TIME")
    c = build_conditioning(_tokens(), sample_count=16000, frame_count=100, vocabulary=vocab, duration_stats={"mean": 0.0, "std": 1.0}, protected_mask=mask, mode="MFA_PHONE_TIME")
    assert np.all(a.phone_ids == vocab.sil_id)
    assert np.all(a.timing == 0)
    assert np.all(b.phone_ids == vocab.sil_id)
    assert np.any(b.timing[2] == 1)
    assert np.any(c.phone_ids != vocab.sil_id)
    with pytest.raises(ValueError):
        a.validate(frames=99)


def test_model_has_zero_initial_output_and_expected_channels():
    import torch

    model = build_model(vocab_size=8)
    features = torch.randn(1, 24, 32)
    ids = torch.zeros(1, 32, dtype=torch.long)
    timing = torch.zeros(1, 3, 32)
    output = model(features, ids, timing, mode="MFA_PHONE_TIME")
    assert output.shape == (1, 24, 32)
    assert torch.equal(output, torch.zeros_like(output))


def test_ordered_phone_and_pause_matching_does_not_zip_crossings():
    source = _tokens()
    target = [dict(source[0]), dict(source[3]), dict(source[1])]
    result = monotonic_phone_match(source, target)
    assert result["matched_count"] == 2
    pauses = monotonic_pause_match([(0.4, 0.6), (0.9, 1.1)], [(0.4, 0.6), (0.9, 1.1)])
    assert sum(bool(row.get("pass")) for row in pauses) == 2


def test_pcm_extreme_values_and_local_control():
    source = np.zeros(16000, dtype=np.int16)
    source[1000:15000] = 1000
    candidate = source.copy()
    protected = np.zeros_like(source, dtype=bool)
    protected[:100] = True
    candidate[:100] = 32767
    result = pcm_contract(source, candidate, protected)
    assert result["new_saturated_samples"] == 100
    assert not result["pass"]
    long_tokens = [
        {"label": "a", "start_s": 0.0, "end_s": 0.3, "speech": True},
        {"label": "b", "start_s": 0.3, "end_s": 0.7, "speech": True},
    ]
    control = construct_local_boundary_control(source, long_tokens)
    assert control["outside_identity"]
    assert control["boundary"]["expected_boundary"] == 5440


def test_timing_calibration_selects_from_fresh_mfa_structure():
    pool = [
        {"pair_id": "a", "transcript": "A", "pcm": np.zeros(16000, dtype=np.int16), "parent_tokens": []},
        {"pair_id": "b", "transcript": "B", "pcm": np.zeros(16000, dtype=np.int16), "parent_tokens": []},
    ]
    reference = [
        {
            "pair_id": "a",
            "status": "COMPLETE",
            "tokens": [
                {"label": "a", "start_s": 0.0, "end_s": 0.10, "speech": True},
                {"label": "b", "start_s": 0.10, "end_s": 0.20, "speech": True},
            ],
        },
        {
            "pair_id": "b",
            "status": "COMPLETE",
            "tokens": [
                {"label": "a", "start_s": 0.0, "end_s": 0.20, "speech": True},
                {"label": "b", "start_s": 0.20, "end_s": 0.40, "speech": True},
            ],
        },
    ]
    selected, meta = _fresh_local_selection(pool, reference, count=1, sample_rate=16000, local_shift_ms=40.0)
    assert [row["pair_id"] for row in selected] == ["b"]
    assert meta["fresh_reference_complete_count"] == 2
    assert meta["fresh_local_eligible_count"] == 1


def test_static_worker_rejects_video_leakage():
    with pytest.raises(ValueError):
        validate_request({"portrait_path": "x", "portrait_rgb_sha256": "x", "audio_path": "x", "box_xyxy": [0, 0, 1, 1], "checkpoint": "x", "ffmpeg": "x", "outfile": "x", "result": "x", "video_path": "leak"})


def test_syncnet_metric_uses_frozen_support_and_tie_first_lag():
    matrix = np.asarray([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    result = score_distance_matrix(matrix, [0, 1], lags=[-1, 0, 1])
    assert result["status"] == "COMPLETE"
    assert result["lag"] == 0
    assert result["C"] == 0.0


def test_frozen_support_requires_every_lag_to_be_in_range():
    result = frozen_support_windows(160_000, 200, vshift=15)
    assert result["support_count"] > 0
    assert min(result["windows"]) >= 16
