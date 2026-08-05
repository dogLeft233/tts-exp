#!/usr/bin/env python3
"""t-SNE visualization of HuBERT features: Natural vs TTS, by phoneme/viseme.

Produces:
  1. Global t-SNE — all frames, colored by condition (natural=red, TTS=blue)
  2. Per-class t-SNE — top-N phonemes/visemes, showing condition overlap per class

Usage
-----
    python scripts/37_tsne_visualization.py
        [--layer 6] [--target phoneme] [--n-classes 8] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import embedding_file_stem
from manifest import load_manifest

TARGET_SR = 16000
OUTPUT_BASE_ZH = Path(__file__).resolve().parent.parent / "data" / "wav2sem_analysis_zh"

EMBEDDINGS_DIR = OUTPUT_BASE_ZH / "embeddings"
MANIFEST_PATH = OUTPUT_BASE_ZH / "manifest" / "alignment.json"
VIZ_DIR = OUTPUT_BASE_ZH / "figures"

SAVED_LAYERS: list[int] = [0, 6, 11]
HUERT_FRAME_STRIDE = 320

_TONE_RE = re.compile(r"[\u02E5-\u02E9\u02CA\u02CB]+$")


def strip_tone(token: str) -> str:
    return _TONE_RE.sub("", token)


def load_alignment(manifest_path: Path) -> list[dict]:
    return load_manifest(manifest_path)


def load_hubert_frames(npy_path: Path, layer: int) -> np.ndarray:
    arr = np.load(npy_path)
    idx = SAVED_LAYERS.index(layer)
    return arr[idx].astype(np.float32)


def assign_frame_labels(T, frame_stride, sr, tokens, target="phoneme"):
    labels = np.full(T, "_oob_", dtype=object)
    for i in range(T):
        centre_s = (i + 0.5) * frame_stride / sr
        for tok in tokens:
            if tok["start_s"] <= centre_s < tok["end_s"]:
                if target == "viseme":
                    labels[i] = tok.get("viseme", "other")
                else:
                    labels[i] = strip_tone(tok["token"])
                break
    return labels


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=str, default=str(EMBEDDINGS_DIR))
    p.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    p.add_argument("--model", type=str, default="hubert")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--target", type=str, default="both", choices=["phoneme", "viseme", "both"])
    p.add_argument("--n-classes", type=int, default=8, help="top N classes for per-class plots")
    p.add_argument("--max-frames", type=int, default=5000, help="max frames for t-SNE (subsample)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, default=str(VIZ_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    embeddings_dir = Path(args.embeddings_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_entries = load_alignment(manifest_path)

    # Collect frame data
    X_all: list[np.ndarray] = []
    cond_all: list[str] = []
    label_all: dict[str, list[str]] = {"phoneme": [], "viseme": []}

    for entry in alignment_entries:
        sid = str(entry.get("utterance_id", entry.get("sample_id", "unknown")))
        cond = entry["condition"]
        tokens = entry.get("tokens", [])

        npy_path = embeddings_dir / f"{embedding_file_stem(entry, args.model)}.npy"
        if not npy_path.exists():
            continue

        X = load_hubert_frames(npy_path, args.layer)
        T = X.shape[0]

        for target in ["phoneme", "viseme"]:
            y = assign_frame_labels(T, HUERT_FRAME_STRIDE, TARGET_SR, tokens, target)
            m = y != "_oob_"
            if len(label_all[target]) == 0:  # first entry
                for i in range(len(X)):
                    if m[i]:
                        X_all.append(X[i])
                        cond_all.append(cond)
                        label_all[target].append(y[i])
            else:
                # subsequent entries only add frames
                for i in range(len(X)):
                    if m[i] and target not in [k for k in label_all if len(label_all[k]) < len(X_all)]:
                        pass
                # Actually, simplify: collect once
                pass

    # Simpler: collect once per frame
    X_frames = []
    conds = []
    labels_phn = []
    labels_vis = []

    for entry in alignment_entries:
        sid = str(entry.get("utterance_id", entry.get("sample_id", "unknown")))
        cond = entry["condition"]
        tokens = entry.get("tokens", [])

        npy_path = embeddings_dir / f"{embedding_file_stem(entry, args.model)}.npy"
        if not npy_path.exists():
            continue

        X = load_hubert_frames(npy_path, args.layer)
        T = X.shape[0]
        y_phn = assign_frame_labels(T, HUERT_FRAME_STRIDE, TARGET_SR, tokens, "phoneme")
        y_vis = assign_frame_labels(T, HUERT_FRAME_STRIDE, TARGET_SR, tokens, "viseme")

        for i in range(T):
            if y_phn[i] == "_oob_":
                continue
            X_frames.append(X[i])
            conds.append(cond)
            labels_phn.append(y_phn[i])
            labels_vis.append(y_vis[i])

    X_arr = np.stack(X_frames)  # (N, 768)
    cond_arr = np.array(conds)
    phn_arr = np.array(labels_phn)
    vis_arr = np.array(labels_vis)

    n_total = len(X_arr)
    logger.info("Total frames: %d (nat=%d, tts=%d)", n_total,
                 int((cond_arr == "natural").sum()), int((cond_arr == "tts").sum()))

    # Subsample if needed
    if n_total > args.max_frames:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, args.max_frames, replace=False)
        X_arr = X_arr[idx]
        cond_arr = cond_arr[idx]
        phn_arr = phn_arr[idx]
        vis_arr = vis_arr[idx]
        logger.info("Subsampled to %d frames for t-SNE", len(X_arr))

    # PCA pre-reduction for speed
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, X_arr.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_arr)
    logger.info("PCA: %d -> %d dims, explained var=%.2f",
                 X_arr.shape[1], 50, pca.explained_variance_ratio_.sum())
    X_tsne = X_pca

    # --- t-SNE ---
    from sklearn.manifold import TSNE
    logger.info("Running t-SNE on %s frames ...", len(X_tsne))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1, verbose=1)
    X_2d = tsne.fit_transform(X_tsne)
    logger.info("t-SNE done: %s", X_2d.shape)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for target in (["phoneme", "viseme"] if args.target == "both" else [args.target]):
        labs = phn_arr if target == "phoneme" else vis_arr
        n_classes = 44 if target == "phoneme" else 11

        # --- Plot 1: Joint t-SNE, colored by condition ---
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        colors_map = {"natural": "#E43F3F", "tts": "#3F7FE4"}

        for cond in ["tts", "natural"]:
            mask = cond_arr == cond
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors_map[cond], s=3, alpha=0.5,
                       label=f"{cond} ({mask.sum()})", rasterized=True)

        ax.set_title(f"HuBERT L{args.layer} t-SNE — {target} ({n_classes} classes, "
                     f"nat={int((cond_arr=='natural').sum())}, tts={int((cond_arr=='tts').sum())} frames)",
                     fontsize=14)
        ax.legend(fontsize=10, markerscale=3)
        ax.set_xticks([]); ax.set_yticks([])

        path = output_dir / f"tsne_{target}_L{args.layer}_joint.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)

        # --- Plot 2: Per-class overlap, top N ---
        top_classes = [c for c, _ in Counter((labs).tolist()).most_common(args.n_classes)]
        n_cols = min(4, args.n_classes)
        n_rows = (args.n_classes + n_cols - 1) // n_cols

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes2 = axes2.flatten() if n_rows * n_cols > 1 else [axes2]

        for idx_cls, cls in enumerate(top_classes):
            ax = axes2[idx_cls]
            mask_cls = labs == cls
            nat_mask = mask_cls & (cond_arr == "natural")
            tts_mask = mask_cls & (cond_arr == "tts")
            other_mask = ~mask_cls

            ax.scatter(X_2d[other_mask, 0], X_2d[other_mask, 1], c="lightgray", s=1, alpha=0.15, rasterized=True)
            ax.scatter(X_2d[nat_mask, 0], X_2d[nat_mask, 1], c="#E43F3F", s=10, alpha=0.7,
                       label=f"nat ({nat_mask.sum()})", edgecolors="white", linewidth=0.3)
            ax.scatter(X_2d[tts_mask, 0], X_2d[tts_mask, 1], c="#3F7FE4", s=10, alpha=0.7,
                       label=f"tts ({tts_mask.sum()})", edgecolors="white", linewidth=0.3)
            ax.set_title(cls, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=8, loc="upper right")

        for idx_extra in range(idx_cls + 1, len(axes2)):
            axes2[idx_extra].set_visible(False)

        fig2.suptitle(f"HuBERT L{args.layer} t-SNE per class — {target}", fontsize=16)
        fig2.tight_layout()
        path2 = output_dir / f"tsne_{target}_L{args.layer}_per_class.png"
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("Saved %s", path2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
