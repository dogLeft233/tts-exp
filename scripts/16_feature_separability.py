#!/usr/bin/env python3
"""Wav2Sem-style feature separability metrics for natural vs TTS audio.

Computes discriminability metrics in HuBERT/XLS-R embedding spaces,
modelled after Wav2Sem (Ditto et al., 2024) and following the analysis
framework described in AISHELL-1 study design.

Metrics computed per (model, layer, condition, variant, level):
  - Intra-class compactness  (mean pairwise cosine distance within classes)
  - Inter-class separability  (mean cosine distance between centroids)
  - Fisher ratio              (between-class / within-class variance)
  - Silhouette score          (cosine distance)
  - Near-confusable pairs     (top-k closest centroid pairs)
  - Linear probe accuracy     (grouped 5-fold CV logistic regression)
  - Boundary sharpness        (cosine change rate at token boundaries)
  - Segment stability         (within-segment frame-frame cosine distance)

Natural vs TTS comparison uses paired permutation tests with BH-FDR
correction and Cohen's d effect sizes.

Usage
-----
    python scripts/16_feature_separability.py                          \\
        [--embeddings-dir DIR] [--models hubert,xlsr]                  \\
        [--layers 0,6,11,12] [--level phoneme,viseme]                  \\
        [--output-dir DIR] [--smoke]

Dependencies
------------
    pip install numpy scipy scikit-learn tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import (
    OUTPUT_BASE,
    paired_permutation_test,
    bootstrap_paired_ci,
    fdr_bh_correction,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDINGS_DIR: Path = OUTPUT_BASE / "embeddings"
DEFAULT_METRICS_DIR: Path = OUTPUT_BASE / "metrics"
DEFAULT_MODELS: list[str] = ["hubert", "xlsr"]
DEFAULT_LAYERS: list[int] = [0, 6, 11, 12]
DEFAULT_LEVELS: list[str] = ["phoneme", "viseme"]
MIN_CLASS_SAMPLES: int = 3
"""Minimum samples per class for per-class metrics."""

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two sets of vectors.

    Parameters
    ----------
    a : np.ndarray, shape (n_a, dim)
    b : np.ndarray, shape (n_b, dim)

    Returns
    -------
    sim : np.ndarray, shape (n_a, n_b)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - _cosine_sim(a, b)


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


def intra_class_variance(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean pairwise cosine distance within each class, averaged across classes.

    Classes with fewer than ``MIN_CLASS_SAMPLES`` samples are excluded.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, dim)
    labels : np.ndarray, shape (n_samples,)

    Returns
    -------
    mean_dist : float
        Average intra-class cosine distance.  Lower = more compact.
        NaN if no valid classes.
    """
    unique_labels = np.unique(labels)
    per_class_avg: list[float] = []
    for c in unique_labels:
        idx = np.where(labels == c)[0]
        if len(idx) < MIN_CLASS_SAMPLES:
            continue
        c_emb = embeddings[idx]
        dists = _cosine_dist(c_emb, c_emb)
        n = dists.shape[0]
        if n <= 1:
            continue
        triu_idx = np.triu_indices(n, k=1)
        avg = float(np.mean(dists[triu_idx]))
        per_class_avg.append(avg)
    if not per_class_avg:
        return float("nan")
    return float(np.mean(per_class_avg))


def inter_class_separation(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine distance between class centroids.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, dim)
    labels : np.ndarray, shape (n_samples,)

    Returns
    -------
    mean_dist : float
        Average inter-class centroid cosine distance.  Higher = more separated.
        NaN if < 2 valid classes.
    """
    unique_labels = np.unique(labels)
    centroids: dict[Any, np.ndarray] = {}
    for c in unique_labels:
        idx = np.where(labels == c)[0]
        if len(idx) < MIN_CLASS_SAMPLES:
            continue
        centroids[c] = embeddings[idx].mean(axis=0)
    class_list = list(centroids.keys())
    if len(class_list) < 2:
        return float("nan")
    centroid_matrix = np.stack([centroids[c] for c in class_list], axis=0)
    # Normalise centroids for cosine distance
    c_norm = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10)
    dists = 1.0 - np.dot(c_norm, c_norm.T)
    triu_idx = np.triu_indices(len(class_list), k=1)
    return float(np.mean(dists[triu_idx]))


def fisher_ratio(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Between-class variance / within-class variance, averaged across dimensions.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, dim)
    labels : np.ndarray, shape (n_samples,)

    Returns
    -------
    ratio : float
        Fisher ratio.  Higher = better separation.
        NaN if < 2 valid classes.
    """
    unique_labels = np.unique(labels)
    valid_labels = [c for c in unique_labels if np.sum(labels == c) >= MIN_CLASS_SAMPLES]
    if len(valid_labels) < 2:
        return float("nan")
    overall_mean = embeddings.mean(axis=0)
    ss_between = np.zeros(embeddings.shape[1], dtype=np.float64)
    ss_within = np.zeros(embeddings.shape[1], dtype=np.float64)
    for c in valid_labels:
        idx = np.where(labels == c)[0]
        c_emb = embeddings[idx]
        centroid = c_emb.mean(axis=0)
        n_c = len(idx)
        ss_between += n_c * ((centroid - overall_mean) ** 2)
        ss_within += np.sum((c_emb - centroid) ** 2, axis=0)
    eps = 1e-12
    n_classes = len(valid_labels)
    df_between = n_classes - 1
    df_within = len(embeddings) - n_classes
    ratio_per_dim = (ss_between / df_between) / (ss_within / df_within + eps)
    return float(np.mean(ratio_per_dim))


def _silhouette_cosine(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score using cosine distance.

    Falls back to sklearn if available, otherwise computes directly.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, dim)
    labels : np.ndarray, shape (n_samples,)

    Returns
    -------
    score : float in [-1, 1], or NaN.
    """
    try:
        from sklearn.metrics import silhouette_score
        if len(embeddings) < 2 or len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(embeddings, labels, metric="cosine"))
    except ImportError:
        pass
    except (ValueError, TypeError):
        return float("nan")
    unique_labels = np.unique(labels)
    valid_labels = [c for c in unique_labels if np.sum(labels == c) >= 2]
    if len(valid_labels) < 2:
        return float("nan")
    valid_mask = np.isin(labels, valid_labels)
    emb = embeddings[valid_mask]
    lab = labels[valid_mask]
    n = len(emb)
    dists = _cosine_dist(emb, emb)
    np.fill_diagonal(dists, 0.0)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ci = lab[i]
        in_mask = lab == ci
        in_mask[i] = False
        a_i = float(np.mean(dists[i, in_mask])) if np.any(in_mask) else 0.0
        b_i = float("inf")
        for c_other in valid_labels:
            if c_other == ci:
                continue
            out_mask = lab == c_other
            mean_other = float(np.mean(dists[i, out_mask]))
            if mean_other < b_i:
                b_i = mean_other
        max_ab = max(a_i, b_i)
        scores[i] = (b_i - a_i) / max_ab if max_ab > 1e-12 else 0.0
    return float(np.mean(scores))


def confusable_pairs(
    embeddings: np.ndarray, labels: np.ndarray, top_k: int = 5
) -> list[dict]:
    """Top-k most confusable class pairs by centroid cosine distance.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, dim)
    labels : np.ndarray, shape (n_samples,)
    top_k : int

    Returns
    -------
    pairs : list[dict]
        Each dict has ``"class_a"``, ``"class_b"``, ``"centroid_distance"``.
    """
    unique_labels = np.unique(labels)
    valid_labels = [c for c in unique_labels if np.sum(labels == c) >= MIN_CLASS_SAMPLES]
    if len(valid_labels) < 2:
        return []
    centroids: dict[Any, np.ndarray] = {}
    for c in valid_labels:
        idx = np.where(labels == c)[0]
        centroids[c] = embeddings[idx].mean(axis=0)
    class_list = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in class_list], axis=0)
    c_norm = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-10)
    dists = 1.0 - np.dot(c_norm, c_norm.T)
    pair_distances: list[tuple[float, Any, Any]] = []
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            pair_distances.append((float(dists[i, j]), class_list[i], class_list[j]))
    pair_distances.sort(key=lambda x: x[0])
    k = min(top_k, len(pair_distances))
    return [
        {"class_a": str(a), "class_b": str(b), "centroid_distance": d}
        for d, a, b in pair_distances[:k]
    ]


def linear_probe_cv(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    cv: int = 5,
) -> dict[str, float]:
    """Grouped cross-validation logistic regression probe.

    Splits are performed on *groups* (sample_ids / utterance IDs) so
    no utterance is split across train and test.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, dim)
    labels : np.ndarray, shape (n_samples,)
    groups : np.ndarray, shape (n_samples,) int
    cv : int
        Number of cross-validation folds (default: 5).

    Returns
    -------
    result : dict
        ``"accuracy"``, ``"f1_macro"``, ``"f1_weighted"``.
        NaN values if insufficient data.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        logger.error("scikit-learn is required for linear probe; install: pip install scikit-learn")
        return {"accuracy": float("nan"), "f1_macro": float("nan"), "f1_weighted": float("nan")}

    unique_groups = np.unique(groups)
    if len(unique_groups) < cv:
        logger.debug(
            "Only %d groups, fewer than cv=%d; reducing cv", len(unique_groups), cv
        )
        cv = len(unique_groups)
    if cv < 2:
        return {"accuracy": float("nan"), "f1_macro": float("nan"), "f1_weighted": float("nan")}

    gkf = GroupKFold(n_splits=cv)
    accuracies: list[float] = []
    f1_macros: list[float] = []
    f1_weighteds: list[float] = []
    unique_labels = np.unique(labels)

    for train_idx, test_idx in gkf.split(embeddings, labels, groups):
        X_train, y_train = embeddings[train_idx], labels[train_idx]
        X_test, y_test = embeddings[test_idx], labels[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=None)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(float(accuracy_score(y_test, y_pred)))
        f1_macros.append(float(f1_score(y_test, y_pred, average="macro", labels=unique_labels)))
        f1_weighteds.append(float(f1_score(y_test, y_pred, average="weighted", labels=unique_labels)))

    if not accuracies:
        return {"accuracy": float("nan"), "f1_macro": float("nan"), "f1_weighted": float("nan")}
    return {
        "accuracy": float(np.mean(accuracies)),
        "f1_macro": float(np.mean(f1_macros)),
        "f1_weighted": float(np.mean(f1_weighteds)),
    }


def boundary_sharpness(
    frame_times: np.ndarray,
    frame_embeddings: np.ndarray,
    token_boundaries: list[float],
) -> tuple[float, float]:
    """Cosine change rate at token boundaries and within-segment stability.

    Parameters
    ----------
    frame_times : np.ndarray, shape (n_frames,)
    frame_embeddings : np.ndarray, shape (n_frames, dim)
    token_boundaries : list[float]
        Boundary times in seconds (end times of each token, excluding the
        last one).

    Returns
    -------
    boundary_change : float
        Mean 1-cos_sim between frames straddling each boundary.
    segment_stability : float
        Mean 1-cos_sim between consecutive frames within segments.
    """
    if len(frame_embeddings) < 2 or not token_boundaries:
        return float("nan"), float("nan")

    boundary_cos_diffs: list[float] = []
    segment_cos_diffs: list[float] = []
    n_frames = len(frame_times)
    emb_norm = frame_embeddings / (np.linalg.norm(frame_embeddings, axis=1, keepdims=True) + 1e-10)

    boundary_set: set[int] = set()
    for b in token_boundaries:
        after_idx = int(np.searchsorted(frame_times, b))
        before_idx = after_idx - 1
        if 0 <= before_idx < n_frames and 0 <= after_idx < n_frames:
            boundary_set.add(before_idx)
            cos_sim = float(np.dot(emb_norm[before_idx], emb_norm[after_idx]))
            boundary_cos_diffs.append(1.0 - cos_sim)

    for i in range(n_frames - 1):
        if i not in boundary_set:
            cos_sim = float(np.dot(emb_norm[i], emb_norm[i + 1]))
            segment_cos_diffs.append(1.0 - cos_sim)

    boundary_change = float(np.mean(boundary_cos_diffs)) if boundary_cos_diffs else float("nan")
    segment_stability = float(np.mean(segment_cos_diffs)) if segment_cos_diffs else float("nan")
    return boundary_change, segment_stability


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = a - b
    if len(diff) < 2:
        return float("nan")
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff < 1e-12:
        return 0.0 if abs(mean_diff) < 1e-12 else float("inf") * np.sign(mean_diff)
    return mean_diff / std_diff


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _discover_embedding_files(embeddings_dir: Path) -> list[dict]:
    """Discover all embedding JSON+NPY pairs in the embeddings directory.

    Returns
    -------
    entries : list[dict]
        Each with keys ``"json_path"``, ``"npy_path"``, ``"metadata"``.
    """
    entries: list[dict] = []
    for json_path in sorted(embeddings_dir.glob("*.json")):
        npy_path = json_path.with_suffix(".npy")
        if not npy_path.exists():
            logger.warning("Missing NPY for %s, skipping", json_path.name)
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", json_path.name, exc)
            continue
        metadata["_json_path"] = str(json_path)
        metadata["_npy_path"] = str(npy_path)
        entries.append(metadata)
    return entries


def _pool_frames_for_layer(
    layer_embeddings: np.ndarray,
    frame_stride: int,
    sample_rate: int,
    tokens: list[dict],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pool frame embeddings into per-token vectors for one layer.

    Returns
    -------
    pooled : np.ndarray, shape (n_valid_tokens, dim)
    info : dict
        ``"labels"`` — dict mapping each label key to a numpy array of
        labels, shape (n_valid_tokens,).
    """
    n_frames = layer_embeddings.shape[0]
    frame_times = np.arange(n_frames, dtype=np.float32) * (frame_stride / sample_rate)

    pooled_list: list[np.ndarray] = []
    initial_list: list[str] = []
    final_list: list[str] = []
    viseme_list: list[str] = []
    tone_list: list[int] = []

    for token in tokens:
        if "start_s" not in token or "end_s" not in token:
            continue
        start = float(token["start_s"])
        end = float(token["end_s"])
        mask = (frame_times >= start) & (frame_times <= end)
        if not np.any(mask):
            continue
        vec = layer_embeddings[mask].mean(axis=0)
        pooled_list.append(vec)
        initial_list.append(str(token.get("initial", "")))
        final_list.append(str(token.get("final", "")))
        viseme_list.append(str(token.get("viseme", "")))
        tone_list.append(int(token.get("tone", 0)))

    if not pooled_list:
        return np.array([], dtype=np.float32), {}

    pooled = np.stack(pooled_list, axis=0)
    labels: dict[str, np.ndarray] = {
        "initial": np.array(initial_list, dtype=str),
        "final": np.array(final_list, dtype=str),
        "viseme": np.array(viseme_list, dtype=str),
        "tone": np.array(tone_list, dtype=str),
        "(initial,final)": np.array(
            [f"{i}_{f}" for i, f in zip(initial_list, final_list)], dtype=str
        ),
        "(initial,final,tone)": np.array(
            [f"{i}_{f}_{t}" for i, f, t in zip(initial_list, final_list, tone_list)], dtype=str
        ),
    }
    return pooled, labels


def _gather_layer_data(
    entries: list[dict],
    model: str,
    layer: int,
    condition: str,
    variant: str,
    level: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], float | None, float | None]:
    """Pool tokens across all samples for one (model, layer, condition, variant, level).

    Returns
    -------
    embeddings : np.ndarray, shape (n_tokens, dim)
    labels : np.ndarray, shape (n_tokens,)
    groups : np.ndarray, shape (n_tokens,) int (sample_id)
    layer_indices : list[int]
        Indices in the NPY file for this layer.
    boundary_sharp : float or None
    segment_stability : float or None
    """
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []

    boundary_changes: list[float] = []
    segment_stabilities: list[float] = []

    frame_stride = 320
    target_sr = 16000

    for meta in entries:
        if meta.get("model") != model:
            continue
        if meta.get("condition") != condition:
            continue
        if meta.get("variant") != variant:
            continue

        sample_id = int(meta.get("sample_id", -1))
        layers = meta.get("layers", [])
        tokens = meta.get("tokens", [])
        if not tokens:
            continue

        try:
            layer_idx = layers.index(layer)
        except ValueError:
            continue

        npy_path = Path(meta.get("_npy_path", ""))
        if not npy_path.exists():
            continue

        try:
            emb_all = np.load(npy_path)  # (n_layers, n_frames, dim)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load %s: %s", npy_path, exc)
            continue

        layer_emb = emb_all[layer_idx]
        pooled, label_map = _pool_frames_for_layer(layer_emb, frame_stride, target_sr, tokens)

        if pooled.shape[0] == 0:
            continue
        if level not in label_map:
            continue

        all_embeddings.append(pooled)
        all_labels.append(label_map[level])
        all_groups.append(np.full(pooled.shape[0], sample_id, dtype=int))

        # Compute boundary sharpness per sample
        n_frames = layer_emb.shape[0]
        frame_times = np.arange(n_frames, dtype=np.float32) * (frame_stride / target_sr)
        boundaries: list[float] = []
        for token in tokens:
            end = float(token.get("end_s", 0))
            boundaries.append(end)
        boundaries = sorted(set(boundaries))
        bs, ss = boundary_sharpness(frame_times, layer_emb, boundaries)
        if not np.isnan(bs):
            boundary_changes.append(bs)
        if not np.isnan(ss):
            segment_stabilities.append(ss)

    if not all_embeddings:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=str),
            np.array([], dtype=int),
            [],
            None,
            None,
        )

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    bs_pooled = float(np.mean(boundary_changes)) if boundary_changes else None
    ss_pooled = float(np.mean(segment_stabilities)) if segment_stabilities else None

    return embeddings, labels, groups, [layer], bs_pooled, ss_pooled


def _compute_metrics_for_pool(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    boundary_sharp: float | None,
    segment_stability: float | None,
) -> dict[str, float | None]:
    """Compute all metrics for a pooled set of tokens."""
    result: dict[str, float | None] = {}
    result["intra_class_dist"] = intra_class_variance(embeddings, labels)
    result["inter_class_dist"] = inter_class_separation(embeddings, labels)
    result["fisher_ratio"] = fisher_ratio(embeddings, labels)
    result["silhouette"] = _silhouette_cosine(embeddings, labels)
    probe = linear_probe_cv(embeddings, labels, groups, cv=5)
    result["probe_accuracy"] = probe["accuracy"]
    result["probe_f1_macro"] = probe["f1_macro"]
    result["probe_f1_weighted"] = probe["f1_weighted"]
    result["confusable_pairs"] = confusable_pairs(embeddings, labels, top_k=5)
    result["boundary_sharpness"] = boundary_sharp
    result["segment_stability"] = segment_stability
    return result


def _compute_metrics_per_sample(
    entries: list[dict],
    model: str,
    layer: int,
    level: str,
) -> dict[tuple[str, str], dict[int, dict[str, float | None]]]:
    """Compute metrics per-sample for paired comparison.

    Returns
    -------
    per_sample : dict
        ``{(condition, variant): {sample_id: {metric: value}}}``
    """
    result: dict[tuple[str, str], dict[int, dict[str, float | None]]] = defaultdict(dict)
    frame_stride = 320
    target_sr = 16000

    for meta in entries:
        if meta.get("model") != model:
            continue
        sample_id = int(meta.get("sample_id", -1))
        condition = str(meta.get("condition", ""))
        variant = str(meta.get("variant", ""))
        layers = meta.get("layers", [])
        tokens = meta.get("tokens", [])
        if not tokens:
            continue
        try:
            layer_idx = layers.index(layer)
        except ValueError:
            continue
        npy_path = Path(meta.get("_npy_path", ""))
        if not npy_path.exists():
            continue
        try:
            emb_all = np.load(npy_path)
        except (OSError, ValueError):
            continue
        layer_emb = emb_all[layer_idx]
        pooled, label_map = _pool_frames_for_layer(layer_emb, frame_stride, target_sr, tokens)
        if pooled.shape[0] == 0 or level not in label_map:
            continue
        labels_arr = label_map[level]
        groups_arr = np.full(pooled.shape[0], sample_id, dtype=int)

        n_frames = layer_emb.shape[0]
        frame_times = np.arange(n_frames, dtype=np.float32) * (frame_stride / target_sr)
        boundaries: list[float] = sorted({
            float(token.get("end_s", 0)) for token in tokens
        })
        bs, ss = boundary_sharpness(frame_times, layer_emb, boundaries)

        key = (condition, variant)
        metrics = _compute_metrics_for_pool(pooled, labels_arr, groups_arr, bs, ss)
        result[key][sample_id] = metrics

    return result


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def _compare_natural_vs_tts(
    per_sample: dict[tuple[str, str], dict[int, dict[str, float | None]]],
    model: str,
    layer: int,
    level: str,
) -> list[dict]:
    """Run paired permutation tests, bootstrap CI, and Cohen's d.

    Compares ``("natural", variant)`` vs ``("tts", variant)`` for each variant.
    """
    comparisons: list[dict] = []
    metric_names = [
        "intra_class_dist",
        "inter_class_dist",
        "fisher_ratio",
        "silhouette",
        "probe_accuracy",
        "boundary_sharpness",
        "segment_stability",
    ]

    for variant in ["raw", "gain_matched"]:
        nat_key = ("natural", variant)
        tts_key = (f"faster_qwen3", variant)  # primary TTS
        if tts_key not in per_sample:
            tts_key = ("tts", variant)
        if nat_key not in per_sample or tts_key not in per_sample:
            continue

        nat_samples = per_sample[nat_key]
        tts_samples = per_sample[tts_key]
        common_ids = sorted(set(nat_samples) & set(tts_samples))
        if len(common_ids) < 3:
            continue

        for metric_name in metric_names:
            a_vals: list[float] = []
            b_vals: list[float] = []
            for sid in common_ids:
                nat_val = nat_samples[sid].get(metric_name)
                tts_val = tts_samples[sid].get(metric_name)
                if nat_val is not None and tts_val is not None and not (np.isnan(nat_val) or np.isnan(tts_val)):
                    a_vals.append(float(nat_val))
                    b_vals.append(float(tts_val))
            if len(a_vals) < 3:
                continue

            a_arr = np.array(a_vals, dtype=np.float64)
            b_arr = np.array(b_vals, dtype=np.float64)

            try:
                p_perm, observed, _ = paired_permutation_test(a_arr, b_arr, n_permutations=2000)
            except ValueError:
                continue
            try:
                ci_low, ci_high, mean_diff = bootstrap_paired_ci(a_arr, b_arr, n_bootstrap=5000)
            except ValueError:
                ci_low, ci_high, mean_diff = float("nan"), float("nan"), float("nan")
            d_val = cohens_d_paired(a_arr, b_arr)

            comparisons.append({
                "metric": metric_name,
                "model": model,
                "layer": layer,
                "level": level,
                "variant": variant,
                "natural_mean": float(np.mean(a_arr)),
                "tts_mean": float(np.mean(b_arr)),
                "delta": float(mean_diff),
                "p_perm": float(p_perm),
                "ci_95": [float(ci_low), float(ci_high)],
                "cohens_d": float(d_val) if not np.isnan(d_val) else None,
                "n_pairs": len(a_arr),
            })
    return comparisons


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_all(
    embeddings_dir: Path,
    models: list[str],
    layers: list[int],
    levels: list[str],
    output_dir: Path,
    smoke: bool = False,
) -> Path:
    """Run the full separability analysis.

    Parameters
    ----------
    embeddings_dir : Path
        Directory containing embedding NPY + JSON files.
    models : list[str]
        Model keys (e.g. ``["hubert", "xlsr"]``).
    layers : list[int]
        Layer indices to analyse.
    levels : list[str]
        Classification levels (e.g. ``["initial", "final", "viseme"]``).
    output_dir : Path
        Directory for output JSON.
    smoke : bool
        If True, process only the first model and layer.

    Returns
    -------
    output_path : Path
        Path to the saved metrics JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering embedding files in %s", embeddings_dir)
    entries = _discover_embedding_files(embeddings_dir)
    if not entries:
        logger.warning("No embedding files found")
        return output_dir / "separability_metrics.json"

    logger.info("Found %d embedding files", len(entries))

    if smoke:
        models = models[:1]
        layers = layers[:1]

    results: list[dict] = []
    all_comparisons: list[dict] = []

    conditions = ["natural", "faster_qwen3", "tts"]
    variants = ["raw", "gain_matched"]

    for model in models:
        for layer in layers:
            for level in levels:
                # --- Per-sample metrics for comparison ---
                per_sample = _compute_metrics_per_sample(entries, model, layer, level)
                comparisons = _compare_natural_vs_tts(per_sample, model, layer, level)
                all_comparisons.extend(comparisons)

                # --- Pooled metrics for reporting ---
                for condition in conditions:
                    for variant in variants:
                        emb, labels, groups, _, bs, ss = _gather_layer_data(
                            entries, model, layer, condition, variant, level,
                        )
                        if emb.shape[0] == 0:
                            continue
                        metrics = _compute_metrics_for_pool(emb, labels, groups, bs, ss)

                        result_entry: dict = {
                            "model": model,
                            "layer": layer,
                            "level": level,
                            "condition": condition,
                            "variant": variant,
                        }
                        for k, v in metrics.items():
                            if v is None or (isinstance(v, float) and np.isnan(v)):
                                result_entry[k] = None
                            elif k == "confusable_pairs":
                                result_entry[k] = v
                            else:
                                result_entry[k] = v
                        results.append(result_entry)
                        logger.info(
                            "  %s l=%d %s %s/%s: intra=%.4f inter=%.4f fisher=%.2f "
                            "silh=%.4f probe=%.4f bsharp=%.4f sstab=%.4f",
                            model, layer, level, condition, variant,
                            result_entry.get("intra_class_dist", None) or float("nan"),
                            result_entry.get("inter_class_dist", None) or float("nan"),
                            result_entry.get("fisher_ratio", None) or float("nan"),
                            result_entry.get("silhouette", None) or float("nan"),
                            result_entry.get("probe_accuracy", None) or float("nan"),
                            result_entry.get("boundary_sharpness", None) or float("nan"),
                            result_entry.get("segment_stability", None) or float("nan"),
                        )

    # --- FDR correction across all comparisons ---
    if all_comparisons:
        p_values = np.array([c["p_perm"] for c in all_comparisons], dtype=np.float64)
        corrected = fdr_bh_correction(p_values)
        for i, comp in enumerate(all_comparisons):
            comp["fdr_corrected_p"] = float(corrected[i])
            comp["significant_fdr"] = bool(corrected[i] < 0.05)

    output: dict = {
        "meta": {
            "models": models,
            "layers": layers,
            "levels": levels,
            "num_embedding_files": len(entries),
        },
        "results": results,
        "comparisons": all_comparisons,
    }

    output_path = output_dir / "separability_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved %d results and %d comparisons to %s",
        len(results), len(all_comparisons), output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Wav2Sem-style feature separability metrics.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help="Directory with embedding NPY+JSON files (default: data/wav2sem_analysis/embeddings/).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="hubert,xlsr",
        help="Comma-separated model keys (default: hubert,xlsr).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,6,11,12",
        help="Comma-separated layer indices (default: 0,6,11,12).",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="initial,final,(initial,final),viseme",
        help="Comma-separated classification levels (default: initial,final,(initial,final),viseme).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_METRICS_DIR),
        help="Output directory for metrics JSON (default: data/wav2sem_analysis/metrics/).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process only the first model and first layer.",
    )
    return parser.parse_args(argv)


def _parse_comma_list(raw: str) -> list[str]:
    items: list[str] = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            items.append(x)
    return items


def _parse_comma_ints(raw: str) -> list[int]:
    result: list[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        result.append(int(x))
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    models = _parse_comma_list(args.models)
    if not models:
        print("Error: no valid model keys provided", file=sys.stderr)
        return 1

    layers = _parse_comma_ints(args.layers)
    if not layers:
        print("Error: no layers specified", file=sys.stderr)
        return 1

    levels = _parse_comma_list(args.level)
    if not levels:
        print("Error: no levels specified", file=sys.stderr)
        return 1

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        print(f"Error: embeddings directory not found: {embeddings_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)

    output_path = process_all(
        embeddings_dir=embeddings_dir,
        models=models,
        layers=layers,
        levels=levels,
        output_dir=output_dir,
        smoke=args.smoke,
    )

    logger.info("Output written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
