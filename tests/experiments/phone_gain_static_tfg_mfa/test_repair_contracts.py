from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.phone_gain_static_tfg_mfa.audio_contract import export_safe_pcm
from scripts.experiments.phone_gain_static_tfg_mfa.conditioning import PhoneVocabulary, build_conditioning
from scripts.experiments.phone_gain_static_tfg_mfa.official_score import parse_score, safe_key
from scripts.experiments.phone_gain_static_tfg_mfa.phone_eval import score_candidate_set
from scripts.experiments.phone_gain_static_tfg_mfa.quality import monotonic_phone_match
from scripts.experiments.phone_gain_static_tfg_mfa.run import _select_dev_candidate
from scripts.experiments.phone_gain_static_tfg_mfa.sync_support import score_distance_matrix


def test_standard_phone_edit_rate_counts_insertions():
    source = [{"label": "a", "start_s": 0.0, "end_s": 0.2, "speech": True}, {"label": "b", "start_s": 0.2, "end_s": 0.4, "speech": True}]
    target = [source[0], {"label": "x", "start_s": 0.2, "end_s": 0.3, "speech": True}, source[1]]
    result = monotonic_phone_match(source, target)
    assert result["insertions"] == 1
    assert result["edit_rate"] == pytest.approx(0.5)


def test_selection_keeps_zero_residual_and_does_not_rank_coverage_or_seed():
    candidates = [
        {"seed": 20260923, "step": 25, "quality_pass": True, "dev": {"margin": 0.2, "accuracy": 0.8, "coverage": 1.0}, "quality": {"max_residual_energy_ratio": 0.01}},
        {"seed": 20260922, "step": 0, "quality_pass": True, "dev": {"margin": 0.2, "accuracy": 0.8, "coverage": 0.9}, "quality": {"max_residual_energy_ratio": 0.0}},
    ]
    selected, _ = _select_dev_candidate(candidates)
    assert selected is not None
    assert selected["step"] == 0


def test_natural_primary_rejects_tts_arm():
    with pytest.raises(ValueError, match="natural_primary"):
        score_candidate_set(object(), {"pair_id": "p", "source_group": "g", "natural": {"tokens": []}, "tts": {"tokens": []}}, {"natural_primary": {"hubert": []}, "mixed_centroids": {"hubert": {}}}, {"T": np.zeros(16000, dtype=np.int16)}, encoder="hubert", view="natural_primary")


def test_conditioning_retains_float_taper_mask_and_special_ids_are_tail_ids():
    vocab = PhoneVocabulary.from_labels(["a", "SIL", "UNK"])
    mask = np.ones(100, dtype=bool)
    edit = np.zeros(100, dtype=np.float32)
    edit[20:80] = np.linspace(0.0, 1.0, 60)
    condition = build_conditioning([{"label": "a", "start_s": 0.0, "end_s": 0.01, "speech": True}], sample_count=100, frame_count=4, vocabulary=vocab, duration_stats={"mean": 0.0, "std": 1.0}, protected_mask=mask, edit_mask=edit, mode="MFA_PHONE_TIME")
    assert vocab.labels[-2:] == ("SIL", "UNK")
    assert np.array_equal(condition.edit_mask, edit)


def test_pcm_export_checks_quantized_contract_and_explicit_identity():
    source = np.zeros(2048, dtype=np.int16)
    source[100:1900] = 1000
    protected = np.ones_like(source, dtype=bool)
    protected[100:1900] = False
    candidate, metadata = export_safe_pcm(source.astype(np.float64) / 32768.0 + 0.00001, source, protected)
    assert metadata["protected_pcm_equal"]
    assert candidate.dtype == np.int16
    identity, identity_meta = export_safe_pcm(source.astype(np.float64) / 32768.0, source, protected)
    assert np.array_equal(identity, source)
    assert identity_meta["identity"]


def test_official_parser_rejects_multiple_tracks_and_key_contains_scope(tmp_path: Path):
    log = tmp_path / "score.log"
    log.write_text("Confidence: 7.0\nMin dist: 6e-1\nConfidence: 8.0\n", encoding="utf-8")
    assert parse_score(log) is None
    key = safe_key("p", "N", "N", portrait_id="6", seed=20260922, scope="SMOKE")
    assert "P_6" in key and "S_20260922" in key and "SMOKE" in key


def test_sync_metric_exposes_n_anchored_distance():
    result = score_distance_matrix(np.asarray([[2.0, 1.0, 2.0], [2.0, 1.0, 2.0]]), [0, 1], lags=[-1, 0, 1], anchor_lag=0)
    assert result["D_anchor"] == pytest.approx(1.0)
    assert result["anchor_lag"] == 0
