"""Unit tests for scripts/16_feature_separability.py.

All tests use synthetic data — no real embeddings or models needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tfg_feature_common import (
    embedding_file_stem,
    paired_permutation_test,
    bootstrap_paired_ci,
    fdr_bh_correction,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "16_feature_separability.py"
)
_spec = importlib.util.spec_from_file_location(
    "_feature_separability", str(_SCRIPT_PATH)
)
_feat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feat)

intra_class_variance = _feat.intra_class_variance
inter_class_separation = _feat.inter_class_separation
fisher_ratio = _feat.fisher_ratio
_silhouette_cosine = _feat._silhouette_cosine
confusable_pairs = _feat.confusable_pairs
linear_probe_cv = _feat.linear_probe_cv
boundary_sharpness = _feat.boundary_sharpness
cohens_d_paired = _feat.cohens_d_paired
_compare_natural_vs_tts = _feat._compare_natural_vs_tts
_process_all = _feat.process_all
_cosine_dist = _feat._cosine_dist
_pool_frames_for_layer = _feat._pool_frames_for_layer


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
DIM = 64


def _make_clustered_embeddings(
    n_classes: int = 3,
    n_per_class: int = 10,
    dim: int = DIM,
    separation: float = 1.0,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic embeddings with known class structure.

    Centroids are placed in the first 2 dimensions with angular spacing
    controlled by ``separation`` (in radians).  Each centroid has unit
    norm.  Gaussian ``noise`` is added in all *dim* dimensions before
    L2 normalisation.
    """
    centroids = np.zeros((n_classes, dim), dtype=np.float64)
    for c in range(n_classes):
        if n_classes > 1:
            angle = c * separation / max(n_classes - 1, 1)
        else:
            angle = 0.0
        centroids[c, 0] = np.cos(angle)
        centroids[c, 1] = np.sin(angle)

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    local_rng = np.random.default_rng(42)
    for c in range(n_classes):
        for _ in range(n_per_class):
            vec = centroids[c] + noise * local_rng.standard_normal(dim)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            embeddings.append(vec)
            labels.append(f"class_{c}")
    return np.stack(embeddings, axis=0).astype(np.float32), np.array(labels, dtype=str)


def _make_frame_embeddings_with_boundary(
    n_frames: int = 50,
    dim: int = DIM,
    boundary_at: float = 0.5,
    step: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Create frame embeddings with a sharp transition at boundary_at."""
    frame_times = np.arange(n_frames, dtype=np.float32) * step
    boundary_idx = int(boundary_at / step)
    embeddings = np.zeros((n_frames, dim), dtype=np.float32)
    # Before boundary: vectors pointing in one direction
    vec_a = RNG.standard_normal(dim).astype(np.float32)
    vec_a /= np.linalg.norm(vec_a)
    # After boundary: vectors pointing in another direction
    vec_b = RNG.standard_normal(dim).astype(np.float32)
    vec_b /= np.linalg.norm(vec_b)
    for i in range(n_frames):
        weight = 1.0 / (1.0 + np.exp(-10.0 * (frame_times[i] - boundary_at)))
        vec = (1 - weight) * vec_a + weight * vec_b
        embeddings[i] = vec / (np.linalg.norm(vec) + 1e-12)
    return frame_times, embeddings


# ---------------------------------------------------------------------------
# Manifest-compatible identifiers
# ---------------------------------------------------------------------------


def test_embedding_stem_supports_string_utterance_id():
    entry = {
        "utterance_id": "utt-001",
        "condition": "tts",
        "variant": "raw",
    }
    assert embedding_file_stem(entry, "hubert") == "utt-001_tts_raw_hubert"


def test_embedding_stem_keeps_tts_provider_distinct():
    entry = {
        "utterance_id": "utt-001",
        "condition": "tts",
        "tts_provider": "faster_qwen3",
        "variant": "raw",
    }
    assert embedding_file_stem(entry, "hubert") == "utt-001_faster_qwen3_raw_hubert"


def test_speaker_grouping_prevents_utterance_leakage():
    embeddings, labels = _make_clustered_embeddings(n_classes=2, n_per_class=6)
    groups = np.array(["speaker-a"] * 6 + ["speaker-b"] * 6)
    result = linear_probe_cv(embeddings, labels, groups, cv=2)
    assert set(result) == {"accuracy", "f1_macro", "f1_weighted"}


# ---------------------------------------------------------------------------
# intra_class_variance
# ---------------------------------------------------------------------------


class TestIntraClassVariance:
    def test_computes_mean_cosine(self):
        emb, labels = _make_clustered_embeddings(3, 10, dim=64, separation=2.0, noise=0.01)
        result = intra_class_variance(emb, labels)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result > 0.0

    def test_smaller_for_tighter_clusters(self):
        emb_tight, labels = _make_clustered_embeddings(3, 10, dim=64, noise=0.01)
        emb_loose, _ = _make_clustered_embeddings(3, 10, dim=64, noise=0.5)
        result_tight = intra_class_variance(emb_tight, labels)
        result_loose = intra_class_variance(emb_loose, labels)
        assert result_tight < result_loose

    def test_nan_for_empty_classes(self):
        emb = np.array([], dtype=np.float32).reshape(0, DIM)
        labels = np.array([], dtype=str)
        result = intra_class_variance(emb, labels)
        assert np.isnan(result)

    def test_excludes_small_classes(self):
        emb, labels = _make_clustered_embeddings(3, 10)
        emb_extra = RNG.standard_normal((2, DIM)).astype(np.float32)
        emb_extra = emb_extra / np.linalg.norm(emb_extra, axis=1, keepdims=True)
        emb_all = np.concatenate([emb, emb_extra], axis=0)
        labels_all = np.concatenate([labels, np.array(["small_c"] * 2, dtype=str)])
        result = intra_class_variance(emb_all, labels_all)
        assert isinstance(result, float)
        assert not np.isnan(result)


# ---------------------------------------------------------------------------
# inter_class_separation
# ---------------------------------------------------------------------------


class TestInterClassSeparation:
    def test_computes_centroid_distance(self):
        emb, labels = _make_clustered_embeddings(3, 10, separation=1.0, noise=0.05)
        result = inter_class_separation(emb, labels)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result > 0.0

    def test_larger_for_separated_classes(self):
        emb_near, labels = _make_clustered_embeddings(
            3, 10, separation=0.1, noise=0.01
        )
        emb_far, _ = _make_clustered_embeddings(3, 10, separation=2.0, noise=0.01)
        near_v = inter_class_separation(emb_near, labels)
        far_v = inter_class_separation(emb_far, labels)
        assert far_v > near_v


# ---------------------------------------------------------------------------
# fisher_ratio
# ---------------------------------------------------------------------------


class TestFisherRatio:
    def test_increases_when_classes_are_separated(self):
        emb_overlap, labels = _make_clustered_embeddings(
            3, 20, separation=0.1, noise=0.5
        )
        emb_sep, _ = _make_clustered_embeddings(3, 20, separation=2.0, noise=0.05)
        f_overlap = fisher_ratio(emb_overlap, labels)
        f_sep = fisher_ratio(emb_sep, labels)
        assert f_sep > f_overlap

    def test_monotonic_with_separation(self):
        ratios: list[float] = []
        for sep in [0.1, 0.5, 1.0, 2.0]:
            emb, labels = _make_clustered_embeddings(3, 20, separation=sep, noise=0.05)
            ratios.append(fisher_ratio(emb, labels))
        for i in range(len(ratios) - 1):
            assert ratios[i] <= ratios[i + 1]

    def test_nan_for_single_class(self):
        emb, labels = _make_clustered_embeddings(1, 10)
        result = fisher_ratio(emb, labels)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# silhouette_cosine
# ---------------------------------------------------------------------------


class TestSilhouetteCosine:
    def test_range(self):
        emb, labels = _make_clustered_embeddings(3, 10, separation=2.0, noise=0.05)
        result = _silhouette_cosine(emb, labels)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_positive_for_well_separated(self):
        emb, labels = _make_clustered_embeddings(3, 10, separation=2.0, noise=0.02)
        result = _silhouette_cosine(emb, labels)
        assert result > 0.3

    def test_near_zero_for_overlapping(self):
        emb, labels = _make_clustered_embeddings(3, 10, separation=0.05, noise=0.5)
        result = _silhouette_cosine(emb, labels)
        assert abs(result) < 0.5


# ---------------------------------------------------------------------------
# confusable_pairs
# ---------------------------------------------------------------------------


class TestConfusablePairs:
    def test_returns_top_k_pairs(self):
        emb, labels = _make_clustered_embeddings(5, 10, separation=1.0, noise=0.05)
        pairs = confusable_pairs(emb, labels, top_k=3)
        assert len(pairs) == 3
        for p in pairs:
            assert "class_a" in p
            assert "class_b" in p
            assert "centroid_distance" in p
            assert 0.0 <= p["centroid_distance"] <= 2.0

    def test_sorted_by_distance(self):
        emb, labels = _make_clustered_embeddings(5, 10)
        pairs = confusable_pairs(emb, labels, top_k=5)
        for i in range(len(pairs) - 1):
            assert pairs[i]["centroid_distance"] <= pairs[i + 1]["centroid_distance"]

    def test_empty_for_single_class(self):
        emb, labels = _make_clustered_embeddings(1, 10)
        pairs = confusable_pairs(emb, labels)
        assert pairs == []


# ---------------------------------------------------------------------------
# linear_probe_cv
# ---------------------------------------------------------------------------


class TestLinearProbeCv:
    def test_groups_not_leaked(self):
        emb, labels = _make_clustered_embeddings(3, 15)
        group_ids = np.repeat(np.arange(5), 9)
        groups = group_ids[: len(emb)]
        emb = emb[: len(groups)]
        labels = labels[: len(groups)]
        result = linear_probe_cv(emb, labels, groups, cv=5)
        assert "accuracy" in result
        assert not np.isnan(result["accuracy"])
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_f1_macro_in_range(self):
        emb, labels = _make_clustered_embeddings(3, 15, separation=2.0, noise=0.02)
        group_ids = np.repeat(np.arange(5), 9)
        groups = group_ids[: len(emb)]
        emb = emb[: len(groups)]
        labels = labels[: len(groups)]
        result = linear_probe_cv(emb, labels, groups, cv=5)
        assert 0.0 <= result["f1_macro"] <= 1.0
        assert 0.0 <= result["f1_weighted"] <= 1.0

    def test_nan_for_insufficient_groups(self):
        emb, labels = _make_clustered_embeddings(3, 10)
        groups = np.array([0] * len(emb), dtype=int)
        result = linear_probe_cv(emb, labels, groups, cv=5)
        assert np.isnan(result["accuracy"])


# ---------------------------------------------------------------------------
# boundary_sharpness
# ---------------------------------------------------------------------------


class TestBoundarySharpness:
    def test_known_change_at_boundary(self):
        frame_times, emb = _make_frame_embeddings_with_boundary(
            n_frames=50, boundary_at=0.5, step=0.02
        )
        boundary, stability = boundary_sharpness(
            frame_times, emb, [0.5]
        )
        assert isinstance(boundary, float)
        assert not np.isnan(boundary)
        assert isinstance(stability, float)
        assert not np.isnan(stability)

    def test_boundary_larger_than_stability(self):
        frame_times, emb = _make_frame_embeddings_with_boundary(
            n_frames=100, boundary_at=0.5, step=0.02
        )
        boundary, stability = boundary_sharpness(
            frame_times, emb, [0.25, 0.5, 0.75]
        )
        assert boundary > stability

    def test_nan_for_empty_boundaries(self):
        frame_times, emb = _make_frame_embeddings_with_boundary(n_frames=30)
        boundary, stability = boundary_sharpness(frame_times, emb, [])
        assert np.isnan(boundary)
        assert np.isnan(stability)

    def test_nan_for_single_frame(self):
        emb = RNG.standard_normal((1, DIM)).astype(np.float32)
        frame_times = np.array([0.0], dtype=np.float32)
        boundary, stability = boundary_sharpness(frame_times, emb, [0.0])
        assert np.isnan(boundary)
        assert np.isnan(stability)


# ---------------------------------------------------------------------------
# cohens_d_paired
# ---------------------------------------------------------------------------


class TestCohensDPaired:
    def test_positive_when_a_larger(self):
        a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d_paired(a, b)
        assert d > 1.0

    def test_near_zero_for_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d_paired(a, b)
        assert abs(d) < 1e-9

    def test_nan_for_single_element(self):
        d = cohens_d_paired(np.array([1.0]), np.array([2.0]))
        assert np.isnan(d)


# ---------------------------------------------------------------------------
# Empty / degenerate input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_embeddings_returns_nan(self):
        emb = np.array([], dtype=np.float32).reshape(0, DIM)
        labels = np.array([], dtype=str)
        assert np.isnan(intra_class_variance(emb, labels))
        assert np.isnan(inter_class_separation(emb, labels))
        assert np.isnan(fisher_ratio(emb, labels))
        assert np.isnan(_silhouette_cosine(emb, labels))
        pairs = confusable_pairs(emb, labels)
        assert pairs == []

    def test_empty_with_probe_returns_nan(self):
        emb = np.array([], dtype=np.float32).reshape(0, DIM)
        labels = np.array([], dtype=str)
        groups = np.array([], dtype=int)
        result = linear_probe_cv(emb, labels, groups)
        assert np.isnan(result["accuracy"])


# ---------------------------------------------------------------------------
# _cosine_dist helper
# ---------------------------------------------------------------------------


class TestCosineDist:
    def test_identical_vectors_zero(self):
        emb = np.ones((5, DIM), dtype=np.float32)
        dist = _cosine_dist(emb, emb)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)

    def test_orthogonal_vectors_one(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        dist = _cosine_dist(a, b)
        np.testing.assert_allclose(dist, 1.0, atol=1e-6)

    def test_opposite_vectors_two(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        dist = _cosine_dist(a, b)
        np.testing.assert_allclose(dist, 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _pool_frames_for_layer helper
# ---------------------------------------------------------------------------


class TestPoolFramesForLayer:
    def test_basic_pooling(self):
        emb = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [
            {"initial": "b", "final": "a", "viseme": "bila", "tone": 1,
             "start_s": 0.0, "end_s": 0.039},
            {"initial": "p", "final": "i", "viseme": "bila", "tone": 2,
             "start_s": 0.04, "end_s": 0.06},
        ]
        pooled, labels = _pool_frames_for_layer(emb, 320, 16000, tokens)
        assert pooled.shape[0] == 2
        assert "initial" in labels
        assert "final" in labels
        assert "viseme" in labels
        assert labels["initial"].tolist() == ["b", "p"]
        assert labels["final"].tolist() == ["a", "i"]

    def test_boundary_frame_is_owned_by_next_span_only(self):
        emb = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        tokens = [
            {"initial": "a", "final": "a", "viseme": "a", "tone": 1, "start_s": 0.0, "end_s": 0.04},
            {"initial": "b", "final": "b", "viseme": "b", "tone": 1, "start_s": 0.04, "end_s": 0.08},
        ]
        pooled, labels = _pool_frames_for_layer(emb, 320, 16000, tokens)
        assert pooled.shape == (2, 2)
        np.testing.assert_allclose(pooled[0], emb[0:2].mean(axis=0))
        np.testing.assert_allclose(pooled[1], emb[2:4].mean(axis=0))
        assert labels["initial"].tolist() == ["a", "b"]

        emb = np.ones((10, DIM), dtype=np.float32)
        tokens = [
            {"initial": "b", "final": "a", "viseme": "bila", "tone": 1,
             "start_s": 0.0, "end_s": 0.05},
            {"initial": "p", "final": "i", "viseme": "bila", "tone": 2,
             "start_s": 0.06, "end_s": 0.1},
            {"initial": "b", "final": "u", "viseme": "round", "tone": 3,
             "start_s": 0.11, "end_s": 0.19},
        ]
        _, labels = _pool_frames_for_layer(emb, 320, 16000, tokens)
        assert labels["(initial,final)"].tolist() == ["b_a", "p_i", "b_u"]
        assert labels["(initial,final,tone)"].tolist() == ["b_a_1", "p_i_2", "b_u_3"]

    def test_pooling_excludes_non_speech_tokens(self):
        emb = np.ones((10, DIM), dtype=np.float32)
        tokens = [
            {"phoneme": "", "viseme": "sil", "is_silence": True,
             "start_s": 0.0, "end_s": 0.05},
            {"phoneme": "aa", "viseme": "vowel", "start_s": 0.06, "end_s": 0.1},
        ]
        pooled, labels = _pool_frames_for_layer(emb, 320, 16000, tokens)
        assert pooled.shape[0] == 1
        assert labels["phoneme"].tolist() == ["aa"]

    def test_skips_token_without_times(self):
        emb = np.ones((10, DIM), dtype=np.float32)
        tokens = [
            {"initial": "x", "final": "y", "viseme": "z", "tone": 0},
            {"initial": "a", "final": "b", "viseme": "c", "tone": 1,
             "start_s": 0.0, "end_s": 0.05},
        ]
        pooled, labels = _pool_frames_for_layer(emb, 320, 16000, tokens)
        assert pooled.shape[0] == 1
        assert labels["initial"].tolist() == ["a"]


# ---------------------------------------------------------------------------
# Integration: process_all smoke test
# ---------------------------------------------------------------------------


class TestProcessAllSmoke:
    """Smoke test using synthetic embedding files."""

    def test_smoke_run(self):
        with tempfile.TemporaryDirectory() as td:
            emb_dir = Path(td) / "embeddings"
            emb_dir.mkdir()
            out_dir = Path(td) / "metrics"
            out_dir.mkdir()

            for sid in [1, 2, 3]:
                for cond in ["natural", "faster_qwen3"]:
                    for variant in ["raw"]:
                        n_frames = 50 + sid * 5
                        emb = RNG.standard_normal((4, n_frames, 768)).astype(np.float32)
                        np.save(emb_dir / f"{sid}_{cond}_{variant}_hubert.npy", emb)
                        tokens = [
                            {
                                "token": f"t{i}",
                                "initial": chr(97 + i % 5),
                                "final": chr(97 + (i + 3) % 5),
                                "tone": (i % 5) + 1,
                                "start_s": round(i * 0.05, 4),
                                "end_s": round((i + 1) * 0.05, 4),
                                "confidence": 0.95,
                                "viseme": ["aa", "pp", "oo", "ee", "uu"][i % 5],
                            }
                            for i in range(10)
                        ]
                        meta = {
                            "sample_id": sid,
                            "condition": cond,
                            "variant": variant,
                            "model": "hubert",
                            "sample_rate": 16000,
                            "duration_s": round(n_frames * 0.02, 4),
                            "layers": [0, 6, 11, 12],
                            "num_frames": n_frames,
                            "embedding_dim": 768,
                            "tokens": tokens,
                        }
                        with open(emb_dir / f"{sid}_{cond}_{variant}_hubert.json", "w") as f:
                            json.dump(meta, f)

            output_path = _process_all(
                embeddings_dir=emb_dir,
                models=["hubert"],
                layers=[12],
                levels=["initial"],
                output_dir=out_dir,
                smoke=True,
            )

            assert output_path.exists()
            with open(output_path, "r") as f:
                data = json.load(f)

            assert "meta" in data
            assert "results" in data
            assert "comparisons" in data
            assert len(data["results"]) > 0


# ---------------------------------------------------------------------------
# Statistical integration tests
# ---------------------------------------------------------------------------


class TestStatisticalIntegration:
    def test_paired_permutation_on_metrics(self):
        """Verify paired permutation works when natural > TTS for silhouette."""
        rng = np.random.default_rng(42)
        nat = rng.normal(0.3, 0.05, 12)
        tts = rng.normal(0.2, 0.05, 12)
        p_val, obs, _ = paired_permutation_test(nat, tts, n_permutations=2000)
        assert obs > 0
        assert p_val < 0.01

    def test_bootstrap_ci_on_metrics(self):
        rng = np.random.default_rng(42)
        nat = rng.normal(0.35, 0.05, 12)
        tts = rng.normal(0.25, 0.05, 12)
        ci_low, ci_high, mean_diff = bootstrap_paired_ci(nat, tts, n_bootstrap=2000)
        assert ci_low < ci_high
        assert mean_diff > 0
        assert ci_low <= mean_diff <= ci_high

    def test_fdr_correction_on_p_values(self):
        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5], dtype=np.float64)
        corrected = fdr_bh_correction(p)
        assert corrected[0] > p[0]
        assert corrected[-1] == 0.5
