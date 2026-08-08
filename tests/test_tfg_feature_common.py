"""Unit tests for scripts/tfg_feature_common.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import tfg_feature_common

from tfg_feature_common import (
    AudioItem,
    EmbeddingRecord,
    TokenSpan,
    STUDY_SAMPLES,
    PILOT_SAMPLES,
    PRIMARY_TTS,
    VALIDATION_TTS,
    TARGET_SR,
    TARGET_LUFS,
    apply_linear_lufs_gain,
    load_audio_mono,
    paired_permutation_test,
    bootstrap_paired_ci,
    fdr_bh_correction,
    ensure_output_dirs,
    embedding_file_stem,
)


# ============================================================================
# Dataclass instantiation tests
# ============================================================================


def test_audio_item_instantiation():
    item = AudioItem(
        sample_id=1,
        condition="natural",
        variant="raw",
        filepath=Path("data/audio/1.wav"),
        duration_s=3.5,
        sample_rate=16000,
    )
    assert item.sample_id == 1
    assert item.condition == "natural"
    assert item.variant == "raw"
    assert item.filepath == Path("data/audio/1.wav")
    assert item.duration_s == 3.5
    assert item.sample_rate == 16000


def test_token_span_instantiation():
    span = TokenSpan(
        token="ba",
        initial="b",
        final="a",
        tone=1,
        start_s=0.5,
        end_s=0.8,
        confidence=0.95,
        viseme="aa",
    )
    assert span.token == "ba"
    assert span.initial == "b"
    assert span.final == "a"
    assert span.tone == 1
    assert span.start_s == 0.5
    assert span.end_s == 0.8
    assert span.confidence == 0.95
    assert span.viseme == "aa"


def test_embedding_record_instantiation():
    emb = np.random.randn(10, 768).astype(np.float32)
    times = np.linspace(0, 1, 10, dtype=np.float32)
    rec = EmbeddingRecord(
        sample_id=3,
        condition="tts",
        layer=6,
        model="hubert",
        frame_times=times,
        embeddings=emb,
    )
    assert rec.sample_id == 3
    assert rec.condition == "tts"
    assert rec.layer == 6
    assert rec.model == "hubert"
    assert rec.embeddings.shape == (10, 768)
    assert rec.frame_times.shape == (10,)


# ============================================================================
# Constant correctness
# ============================================================================


def test_study_samples_excludes_nine():
    assert 9 not in STUDY_SAMPLES
    assert len(STUDY_SAMPLES) == 12


def test_pilot_samples_are_subset():
    assert set(PILOT_SAMPLES).issubset(set(STUDY_SAMPLES))
    assert len(PILOT_SAMPLES) == 5


def test_tts_constants_are_known():
    assert PRIMARY_TTS == "faster_qwen3"
    assert VALIDATION_TTS == "f5_tts"


def test_embedding_stem_uses_stable_sample_id_for_mdc_utterance():
    stem = embedding_file_stem(
        {
            "sample_id": "en_001",
            "utterance_id": "mdc_tts/en_001/natural",
            "condition": "natural",
            "variant": "raw",
        },
        "hubert",
    )
    assert stem == "en_001_natural_raw_hubert"
    assert "/" not in stem


def test_embedding_stem_sanitizes_explicit_stem():
    stem = embedding_file_stem(
        {
            "sample_id": "en_001",
            "condition": "tts",
            "embedding_stem": "mdc_tts/en_001/tts",
        },
        "xlsr",
    )
    assert stem == "mdc_tts_en_001_tts_xlsr"
    assert "/" not in stem


# ============================================================================
# apply_linear_lufs_gain
# ============================================================================


def test_linear_gain_preserves_length_and_changes_lufs_only():
    rng = np.random.default_rng(42)
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float64)
    y = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    original_length = len(y)
    y_scaled, gain_db = apply_linear_lufs_gain(y, sr)

    assert len(y_scaled) == original_length
    assert gain_db != 0.0

    mask = np.abs(y) > 1e-8
    ratio = y_scaled[mask] / y[mask]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)


# ============================================================================
# paired_permutation_test
# ============================================================================


def test_paired_permutation_test_known_result():
    rng = np.random.default_rng(123)
    a = rng.normal(10.0, 2.0, 30)
    b = rng.normal(7.0, 2.0, 30)

    p_val, observed, null_dist = paired_permutation_test(a, b, n_permutations=5000)

    assert observed > 0
    assert p_val < 0.05
    assert len(null_dist) == 5000


def test_paired_permutation_test_null():
    rng = np.random.default_rng(456)
    a = rng.normal(5.0, 1.0, 50)
    b = a.copy()

    p_val, observed, null_dist = paired_permutation_test(a, b, n_permutations=2000)

    assert np.isclose(observed, 0.0, atol=1e-10)
    assert p_val > 0.3
    assert len(null_dist) == 2000


# ============================================================================
# bootstrap_paired_ci
# ============================================================================


def test_bootstrap_ci_contains_known_diff():
    rng = np.random.default_rng(789)
    n = 100
    a = rng.normal(10.0, 1.0, n)
    delta = 2.0
    b = a - delta + rng.normal(0.0, 0.5, n)

    ci_low, ci_high, mean_diff = bootstrap_paired_ci(a, b, n_bootstrap=5000)

    assert ci_low < ci_high
    assert 1.5 < mean_diff < 2.5
    assert ci_low <= mean_diff <= ci_high


# ============================================================================
# fdr_bh_correction
# ============================================================================


def test_fdr_bh_identical_pvalues():
    p = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)
    corrected = fdr_bh_correction(p)
    assert corrected.shape == (5,)
    np.testing.assert_allclose(corrected, corrected[0])


def test_fdr_bh_extreme():
    p = np.array([0.0001, 0.1, 0.2, 0.3, 0.5], dtype=np.float64)
    corrected = fdr_bh_correction(p)

    # When alone, the first p-value would be 0.0001 * 5 = 0.0005
    # With BH, the correction should be larger due to the other tests
    assert corrected[0] > 0.0001
    assert corrected[0] <= 0.01


def test_fdr_bh_empty():
    p = np.array([], dtype=np.float64)
    corrected = fdr_bh_correction(p)
    assert len(corrected) == 0


def test_fdr_bh_monotonic_on_sorted_input():
    # BH guarantees monotonicity of sorted output, so the input p-values
    # must be pre-sorted for this property to hold element-wise.
    p = np.array([0.001, 0.01, 0.05, 0.1], dtype=np.float64)
    corrected = fdr_bh_correction(p)
    for i in range(len(corrected) - 1):
        assert corrected[i] <= corrected[i + 1]


# ============================================================================
# ensure_output_dirs
# ============================================================================


def test_ensure_output_dirs_creates_all(tmp_path, monkeypatch):
    test_output_base = tmp_path / "wav2sem_analysis"
    monkeypatch.setattr(tfg_feature_common, "OUTPUT_BASE", test_output_base)

    dirs = ensure_output_dirs()

    expected_subs = ["manifest", "alignments", "embeddings", "metrics", "figures"]
    assert set(dirs.keys()) == {"base"} | set(expected_subs)
    for sub in expected_subs:
        p = dirs[sub]
        assert isinstance(p, Path)
        assert p.exists()
        assert p.is_dir()
    assert isinstance(dirs["base"], Path)
    assert dirs["base"].exists()
    assert dirs["base"] == test_output_base.resolve()
