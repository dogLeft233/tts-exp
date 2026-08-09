#!/usr/bin/env python3
"""Near-homophone feature-space decoupling analysis — Wav2Sem-style (Section 4.4).

Adapts the Wav2Sem paper's feature-space decoupling methodology to Chinese:
    1. Word-level L2 distance between near-homophone syllables
    2. t‑SNE visualisation of phoneme-level feature clusters

Linguistic contrast categories for Chinese IPA (MFA output):
    - Aspiration (unaspirated vs aspirated, same place of articulation)
    - Place of articulation (same manner, different place)
    - Tone contrast (same segment, different tone)

Feature source: HuBERT/XLS‑R frame embeddings, mean-pooled to token-level.

Usage
-----
    python scripts/34_near_homophone_analysis.py \\
        [--model hubert] [--layers 11] [--output-dir DIR] [--smoke]

Dependencies
------------
    pip install numpy scipy scikit-learn matplotlib tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np

from tfg_feature_common import half_open_span_mask
OUTPUT_BASE_ZH = Path(__file__).resolve().parent.parent / "data" / "wav2sem_analysis_zh"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import load_manifest


def _paired_permutation_test(
    a: np.ndarray, b: np.ndarray, n_permutations: int = 10000, random_seed: int = 42
) -> tuple[float, float, np.ndarray]:
    """Paired permutation test for H0: mean(a) == mean(b)."""
    if len(a) != len(b):
        raise ValueError(f"Expected equal-length arrays, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("Expected non-empty arrays")
    rng = np.random.default_rng(random_seed)
    diff = a - b
    observed = float(np.mean(diff))
    null_dist = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        null_dist[i] = float(np.mean(signs * diff))
    p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))
    return p_value, observed, null_dist


def _bootstrap_paired_ci(
    a: np.ndarray, b: np.ndarray, n_bootstrap: int = 10000,
    ci_level: float = 0.95, random_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean of paired differences."""
    if len(a) != len(b):
        raise ValueError(f"Expected equal-length arrays, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("Expected non-empty arrays")
    rng = np.random.default_rng(random_seed)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means[i] = float(np.mean(diff[idx]))
    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_means, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))
    return ci_lower, ci_upper, mean_diff


def _fdr_bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    corrected_sorted = np.minimum(1.0, sorted_p * n / ranks)
    corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
    corrected = np.empty(n, dtype=np.float64)
    corrected[order] = corrected_sorted
    return corrected

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
EMBEDDINGS_DIR: Path = OUTPUT_BASE_ZH / "embeddings"
MANIFEST_PATH: Path = OUTPUT_BASE_ZH / "manifest" / "alignment.json"
DEFAULT_OUTPUT_DIR: Path = OUTPUT_BASE_ZH / "near_homophone"

FRAME_STRIDE: int = 320
TARGET_SR: int = 16000

# ---------------------------------------------------------------------------
# Tone stripping
# ---------------------------------------------------------------------------

_TONE_RE = re.compile(r"[˥˧˦˨˩]+")


def strip_tone(token: str) -> str:
    """Strip tone diacritics from an IPA token, leaving the segment base.

    ``"a˥˩"`` → ``"a"``, ``"tʰ"`` → ``"tʰ"``, ``"ŋ"`` → ``"ŋ"``.
    """
    return _TONE_RE.sub("", token)


def _tone_number(token: str) -> int | None:
    """Map tone diacritics to 1‑4; return None for toneless consonants."""
    if "˥˥" in token or (token.endswith("˥") and "˥˩" not in token):
        return 1
    if "˧˥" in token:
        return 2
    if "˨˩˦" in token or "˨˩" in token:
        return 3
    if "˥˩" in token:
        return 4
    return None


# ---------------------------------------------------------------------------
# Linguistic contrast categories
# ---------------------------------------------------------------------------


ASPIRATION_PAIRS: dict[str, tuple[set[str], set[str]]] = {
    "t_th": ({"t"}, {"tʰ"}),
    "k_kh": ({"k"}, {"kʰ"}),
    "ts_tsh": ({"ts"}, {"tsʰ"}),
    "tɕ_tɕh": ({"tɕ"}, {"tɕʰ"}),
    "ʈʂ_ʈʂh": ({"ʈʂ"}, {"ʈʂʰ"}),
}
"""Aspiration pairs: unaspirated vs aspirated, same place of articulation."""


PLACE_GROUPS: dict[str, dict[str, str]] = {
    "fricative": {"s": "舌尖", "ʂ": "卷舌", "ɕ": "腭面"},
    "affricate_unasp": {"ts": "舌尖", "tɕ": "腭面", "ʈʂ": "卷舌"},
    "affricate_asp": {"tsʰ": "舌尖", "tɕʰ": "腭面", "ʈʂʰ": "卷舌"},
}
"""Place of articulation groups.  Each value maps an IPA token to a
human-readable place label (informational only)."""


def _build_contrast_groups(
    tokens: list[dict],
) -> dict[str, dict[str, list[dict]]]:
    """Partition tokens into linguistic contrast groups.

    Parameters
    ----------
    tokens : list[dict]
        Each must have ``"token"`` (IPA string), ``"start_s"``, ``"end_s"``.

    Returns
    -------
    groups : dict
        Keys are contrast names like ``"asp: t_th"`` or ``"tone: a"`` or
        ``"place: fricative"``.  Values are ``{label: [token_dicts]}``.
    """
    groups: dict[str, dict[str, list[dict]]] = {}

    # --- Aspiration ---
    for pair_name, (unasp_set, asp_set) in ASPIRATION_PAIRS.items():
        key = f"asp: {pair_name}"
        groups[key] = {"unaspirated": [], "aspirated": []}
        for tok in tokens:
            base = strip_tone(tok["token"])
            if base in unasp_set:
                groups[key]["unaspirated"].append(tok)
            elif base in asp_set:
                groups[key]["aspirated"].append(tok)
        if not groups[key]["unaspirated"] or not groups[key]["aspirated"]:
            del groups[key]

    # --- Place ---
    for group_name, place_map in PLACE_GROUPS.items():
        key = f"place: {group_name}"
        groups[key] = {label: [] for label in place_map.values()}
        for tok in tokens:
            base = strip_tone(tok["token"])
            if base in place_map:
                groups[key][place_map[base]].append(tok)
        present = [label for label, toks in groups[key].items() if toks]
        if len(present) < 2:
            del groups[key]

    # --- Tone ---
    tone_groups: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for tok in tokens:
        base = strip_tone(tok["token"])
        tone = _tone_number(tok["token"])
        if tone is None or len(base) < 1:
            continue
        tone_groups[base][f"T{tone}"].append(tok)

    for base, tone_dict in tone_groups.items():
        n_tones = len(tone_dict)
        if n_tones < 2:
            continue
        key = f"tone: {base}"
        groups[key] = {}
        for tone_label, toks in tone_dict.items():
            groups[key][tone_label] = toks

    return groups


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> list[dict]:
    return load_manifest(manifest_path)


def _pool_tokens(
    layer_emb: np.ndarray,
    tokens: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """Mean-pool frame embeddings into per-token vectors.

    Parameters
    ----------
    layer_emb : np.ndarray, shape (n_frames, dim)
    tokens : list[dict]

    Returns
    -------
    pooled : np.ndarray, shape (n_valid, dim)
    valid_tokens : list[dict]
        Tokens that had at least one frame match (with ``"_pooled_idx"`` set).
    """
    n_frames = layer_emb.shape[0]
    frame_times = np.arange(n_frames, dtype=np.float32) * (FRAME_STRIDE / TARGET_SR)

    pooled_list: list[np.ndarray] = []
    valid_tokens: list[dict] = []

    for i, token in enumerate(tokens):
        start = float(token.get("start_s", 0))
        end = float(token.get("end_s", 0))
        mask = half_open_span_mask(frame_times, start, end)
        if not np.any(mask):
            continue
        vec = layer_emb[mask].mean(axis=0)
        pooled_list.append(vec)
        tok_copy = dict(token)
        tok_copy["_pooled_idx"] = len(pooled_list) - 1
        valid_tokens.append(tok_copy)

    if not pooled_list:
        return np.array([], dtype=np.float32), []
    return np.stack(pooled_list, axis=0), valid_tokens


def _load_and_pool(
    embeddings_dir: Path,
    manifest: list[dict],
    model: str,
    layer: int,
) -> dict[tuple[str, str, int], tuple[np.ndarray, list[dict]]]:
    """Load all .npy files for a given model+layer, pool frames→tokens.

    Returns
    -------
    data : dict
        Key ``(condition, variant, sample_id)`` →
        ``(pooled_embeddings, valid_tokens_with_idx)``.
    """
    result: dict[tuple[str, str, int], tuple[np.ndarray, list[dict]]] = {}

    for entry in manifest:
        sample_id = entry["sample_id"]
        condition = entry["condition"]
        variant = entry.get("variant", "raw")
        tokens = entry.get("tokens", [])
        if not tokens:
            continue

        stem = f"{sample_id}_{condition}_{variant}_{model}"
        npy_path = embeddings_dir / f"{stem}.npy"
        json_path = embeddings_dir / f"{stem}.json"

        if not npy_path.exists() or not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta_layers = meta.get("layers", [])
        try:
            layer_idx = meta_layers.index(layer)
        except ValueError:
            continue

        emb_all = np.load(npy_path)
        if layer_idx >= emb_all.shape[0]:
            continue

        layer_emb = emb_all[layer_idx]
        pooled, valid_tokens = _pool_tokens(layer_emb, tokens)

        if pooled.shape[0] == 0:
            continue

        result[(condition, variant, sample_id)] = (pooled, valid_tokens)

    return result


# ---------------------------------------------------------------------------
# Pairwise L2 distance computation
# ---------------------------------------------------------------------------


def _compute_pairwise_l2(
    pooled: np.ndarray,
    group_a_indices: list[int],
    group_b_indices: list[int] | None = None,
) -> float:
    """Mean pairwise L2 distance between two sets of embeddings.

    If *group_b_indices* is None, computes within-set pairwise distances.

    Parameters
    ----------
    pooled : np.ndarray, shape (n_tokens, dim)
    group_a_indices : list[int]
    group_b_indices : list[int] or None

    Returns
    -------
    mean_l2 : float (NaN if either group is empty).
    """
    if not group_a_indices:
        return float("nan")
    if group_b_indices is None:
        group_b_indices = group_a_indices
    if not group_b_indices:
        return float("nan")

    a_vecs = pooled[group_a_indices]
    b_vecs = pooled[group_b_indices]

    # All pairwise L2 distances via broadcasting
    # (n_a, dim) vs (n_b, dim) → (n_a, n_b)
    diffs = a_vecs[:, None, :] - b_vecs[None, :, :]
    l2_matrix = np.sqrt(np.sum(diffs**2, axis=-1))

    if group_a_indices is group_b_indices or group_a_indices == group_b_indices:
        n = l2_matrix.shape[0]
        if n <= 1:
            return float("nan")
        triu = np.triu_indices(n, k=1)
        return float(np.mean(l2_matrix[triu]))

    return float(np.mean(l2_matrix))


def _per_sample_l2_metrics(
    pooled: np.ndarray,
    valid_tokens: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute all contrast-group L2 distances for one sample.

    Returns
    -------
    metrics : dict
        ``{contrast_key: {metric: value}}``.
    """
    groups = _build_contrast_groups(valid_tokens)

    metrics: dict[str, dict[str, float]] = {}

    # Aspiration: between-set L2
    for ck in [k for k in groups if k.startswith("asp:")]:
        unasp_indices = [
            t["_pooled_idx"] for t in groups[ck].get("unaspirated", [])
            if "_pooled_idx" in t
        ]
        asp_indices = [
            t["_pooled_idx"] for t in groups[ck].get("aspirated", [])
            if "_pooled_idx" in t
        ]
        metrics.setdefault(ck, {})["l2_between"] = _compute_pairwise_l2(
            pooled, unasp_indices, asp_indices
        )
        metrics[ck]["n_unaspirated"] = len(unasp_indices)
        metrics[ck]["n_aspirated"] = len(asp_indices)

    # Place: within-set L2 (how far are phones of same class spread?)
    for ck in [k for k in groups if k.startswith("place:")]:
        for label, toks in groups[ck].items():
            indices = [t["_pooled_idx"] for t in toks if "_pooled_idx" in t]
            metrics.setdefault(ck, {})[f"l2_within_{label}"] = _compute_pairwise_l2(
                pooled, indices
            )
            metrics[ck][f"n_{label}"] = len(indices)

        all_indices = []
        for toks in groups[ck].values():
            all_indices.extend(t["_pooled_idx"] for t in toks if "_pooled_idx" in t)
        metrics[ck]["l2_total"] = _compute_pairwise_l2(pooled, all_indices)

    # Tone: between-tone L2
    for ck in [k for k in groups if k.startswith("tone:")]:
        tone_labels = list(groups[ck].keys())
        tone_indices = {
            tl: [t["_pooled_idx"] for t in groups[ck][tl] if "_pooled_idx" in t]
            for tl in tone_labels
        }
        pairwise_l2s: list[float] = []
        for tl1, tl2 in combinations(tone_labels, 2):
            val = _compute_pairwise_l2(
                pooled, tone_indices[tl1], tone_indices[tl2]
            )
            if not np.isnan(val):
                pairwise_l2s.append(val)
        if pairwise_l2s:
            metrics.setdefault(ck, {})["l2_between_tones"] = float(np.mean(pairwise_l2s))
        else:
            logger.debug(
                "  tone group '%s' has no valid between-tone L2 pairs (insufficient tokens)", ck
            )
        for tl in tone_labels:
            metrics.setdefault(ck, {})[f"n_{tl}"] = len(tone_indices[tl])

    return metrics


# ---------------------------------------------------------------------------
# Aggregation & comparison
# ---------------------------------------------------------------------------


def _aggregate_and_compare(
    per_sample: dict[tuple[str, str, int], dict[str, dict[str, float]]],
) -> tuple[list[dict], list[dict]]:
    """Aggregate metrics across samples and run Natural vs TTS comparisons.

    Returns
    -------
    summaries : list[dict]
        Per-condition aggregate metrics (mean over samples).
    comparisons : list[dict]
        Paired comparison results with permutation test, bootstrap CI, Cohen's d.
    """
    # Discover all unique (contrast, metric) keys
    unique_keys: set[tuple[str, str]] = set()
    for (_c, _v, _sid), metrics in per_sample.items():
        for ck, ck_metrics in metrics.items():
            for mname in ck_metrics:
                unique_keys.add((ck, mname))

    # Discover available variants
    variants = sorted({var for (_c, var, _sid) in per_sample})

    comparisons: list[dict] = []
    summaries: list[dict] = []

    # --- Paired Natural vs TTS comparisons ---
    for ck, mname in sorted(unique_keys):
        for variant in variants:
            pair_list: list[tuple[float, float, int]] = []

            # Rebuild pairs by grouping by sample_id
            nat_dict: dict[int, float] = {}
            tts_dict: dict[int, float] = {}
            for (cond, var, sid), metrics in per_sample.items():
                if var != variant:
                    continue
                val = metrics.get(ck, {}).get(mname)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                if cond == "natural":
                    nat_dict[sid] = float(val)
                elif cond == "tts":
                    tts_dict[sid] = float(val)

            common_ids = sorted(set(nat_dict) & set(tts_dict))
            pair_list = [
                (nat_dict[sid], tts_dict[sid], sid) for sid in common_ids
            ]

            if len(pair_list) < 3:
                continue

            a_arr = np.array([p[0] for p in pair_list], dtype=np.float64)
            b_arr = np.array([p[1] for p in pair_list], dtype=np.float64)

            try:
                p_perm, observed, _ = _paired_permutation_test(a_arr, b_arr, n_permutations=2000)
            except ValueError:
                continue

            try:
                ci_low, ci_high, mean_diff = _bootstrap_paired_ci(a_arr, b_arr, n_bootstrap=5000)
            except ValueError:
                ci_low, ci_high, mean_diff = float("nan"), float("nan"), float("nan")

            diff = a_arr - b_arr
            d_val = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)) if len(diff) > 1 else float("nan")

            comparisons.append({
                "contrast": ck,
                "metric": mname,
                "variant": variant,
                "natural_mean": float(np.mean(a_arr)),
                "tts_mean": float(np.mean(b_arr)),
                "delta": float(mean_diff),
                "p_perm": float(p_perm),
                "ci_95": [float(ci_low), float(ci_high)],
                "cohens_d": float(d_val) if not np.isnan(d_val) else None,
                "n_pairs": len(a_arr),
            })

    # --- Per-condition summaries ---
    cond_aggs: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (cond, var, sid), metrics in per_sample.items():
        for ck, ck_metrics in metrics.items():
            for mname, val in ck_metrics.items():
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                cond_aggs[(cond, var, ck, mname)].append(val)

    for (cond, var, ck, mname), vals in cond_aggs.items():
        arr = np.array(vals, dtype=np.float64)
        summaries.append({
            "contrast": ck,
            "metric": mname,
            "condition": cond,
            "variant": var,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": len(arr),
        })

    return summaries, comparisons


# ---------------------------------------------------------------------------
# t‑SNE visualisation
# ---------------------------------------------------------------------------


def _tsne_visualize(
    pooled: np.ndarray,
    valid_tokens: list[dict],
    contrast_key: str,
    output_path: Path,
    condition: str = "",
    sample_id: int = 0,
):
    """Generate a t‑SNE plot for one contrast group."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("sklearn/matplotlib required for t‑SNE; skipping")
        return

    groups = _build_contrast_groups(valid_tokens)
    if contrast_key not in groups:
        logger.warning("Contrast %s not found for sample %d", contrast_key, sample_id)
        return

    label_dict = groups[contrast_key]
    all_embs: list[np.ndarray] = []
    labels: list[str] = []
    for label, toks in label_dict.items():
        for tok in toks:
            pidx = tok.get("_pooled_idx")
            if pidx is not None:
                all_embs.append(pooled[pidx])
                labels.append(label)

    if len(all_embs) < 5:
        logger.warning("Too few tokens (%d) for t‑SNE on %s", len(all_embs), contrast_key)
        return

    X = np.stack(all_embs, axis=0)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) // 2), random_state=42, metric="cosine")
    X_2d = tsne.fit_transform(X)

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            color=colors[i],
            label=label,
            alpha=0.7,
            s=30,
        )
    ax.set_title(f"{contrast_key}  |  {condition}  sample={sample_id}")
    ax.legend(fontsize=8)
    ax.set_xlabel("t‑SNE 1")
    ax.set_ylabel("t‑SNE 2")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("  t‑SNE saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_all(
    embeddings_dir: Path,
    manifest: list[dict],
    model: str,
    layers: list[int],
    output_dir: Path,
    smoke: bool = False,
) -> Path:
    """Run full near-homophone analysis.

    Returns
    -------
    output_path : Path
        Path to the saved metrics JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        layers = layers[:1]

    all_comparisons: list[dict] = []
    all_summaries: list[dict] = []

    for layer in layers:
        logger.info("Model=%s Layer=%d: loading & pooling ...", model, layer)

        pooled_data = _load_and_pool(embeddings_dir, manifest, model, layer)
        if smoke:
            keys = list(pooled_data.keys())[:2]
            pooled_data = {k: pooled_data[k] for k in keys}

        logger.info("  %d samples loaded", len(pooled_data))

        # Per-sample metrics
        per_sample: dict[tuple[str, str, int], dict[str, dict[str, float]]] = {}
        for (cond, var, sid), (pooled, valid_tokens) in pooled_data.items():
            per_sample[(cond, var, sid)] = _per_sample_l2_metrics(pooled, valid_tokens)
            logger.debug(
                "  %s/%s #%d: %d tokens, %d contrast groups",
                cond, var, sid, len(valid_tokens),
                len(per_sample[(cond, var, sid)]),
            )

        # Summaries + comparisons
        summaries, comparisons = _aggregate_and_compare(per_sample)
        for s in summaries:
            s["model"] = model
            s["layer"] = layer
        for c in comparisons:
            c["model"] = model
            c["layer"] = layer

        all_summaries.extend(summaries)
        all_comparisons.extend(comparisons)

        # --- t‑SNE for representative pairs ---
        tsne_dir = output_dir / "tsne"
        tsne_dir.mkdir(parents=True, exist_ok=True)

        # Pick 1 sample per condition for each linguistics category
        for plot_condition in ["natural", "tts"]:
            candidates = [
                (cond, var, sid)
                for (cond, var, sid) in pooled_data
                if cond == plot_condition
            ]
            if not candidates:
                continue
            cond_key = candidates[0] if smoke else candidates[len(candidates) // 2]
            pooled, valid_tokens = pooled_data[cond_key]
            sample_id = cond_key[2]

            groups = _build_contrast_groups(valid_tokens)

            # Pick up to 1 plot per linguistics category in smoke mode
            if smoke:
                to_plot: list[str] = []
                for prefix in ("asp:", "place:", "tone:"):
                    for k in sorted(groups):
                        if k.startswith(prefix):
                            to_plot.append(k)
                            break
                plot_keys = to_plot[:3]
            else:
                plot_keys = sorted(groups.keys())

            for ck in plot_keys:
                out_path = tsne_dir / f"tsne_{model}_L{layer}_{ck}_{plot_condition}_s{sample_id:02d}.png"
                _tsne_visualize(
                    pooled, valid_tokens, ck, out_path,
                    condition=plot_condition, sample_id=sample_id,
                )

    # --- FDR correction ---
    if all_comparisons:
        p_vals = np.array([c["p_perm"] for c in all_comparisons], dtype=np.float64)
        corrected = _fdr_bh_correction(p_vals)
        for i, comp in enumerate(all_comparisons):
            comp["fdr_corrected_p"] = float(corrected[i])
            comp["significant_fdr"] = bool(corrected[i] < 0.05)

    output: dict = {
        "meta": {
            "model": model,
            "layers": layers,
            "num_samples": len(manifest),
        },
        "summaries": all_summaries,
        "comparisons": all_comparisons,
    }

    output_path = output_dir / "near_homophone_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved %d summaries + %d comparisons → %s",
        len(all_summaries), len(all_comparisons), output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Near-homophone feature-space decoupling (Wav2Sem §4.4 style).",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=str(EMBEDDINGS_DIR),
        help="Directory with embedding NPY+JSON files.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_PATH),
        help="Path to alignment.json (MFA tokens).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hubert",
        help="Model key: hubert or xlsr.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="11",
        help="Comma-separated layer indices (default: 11).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for metrics JSON and t‑SNE plots.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process only the first layer and minimal t‑SNE output.",
    )
    return parser.parse_args(argv)


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

    embeddings_dir = Path(args.embeddings_dir)
    manifest_path = Path(args.manifest)
    if not embeddings_dir.exists():
        print(f"Error: embeddings dir not found: {embeddings_dir}", file=sys.stderr)
        return 1
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load_manifest(manifest_path)
    if not manifest:
        print("Error: no entries in manifest", file=sys.stderr)
        return 1

    layers = _parse_comma_ints(args.layers)
    if not layers:
        print("Error: no layers specified", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)

    output_path = process_all(
        embeddings_dir=embeddings_dir,
        manifest=manifest,
        model=args.model,
        layers=layers,
        output_dir=output_dir,
        smoke=args.smoke,
    )

    logger.info("Done → %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
