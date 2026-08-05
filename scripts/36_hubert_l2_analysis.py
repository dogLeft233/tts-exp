#!/usr/bin/env python3
"""L2 distance analysis on HuBERT features: Natural vs TTS.

Three metrics:
  1. Within-phoneme scatter: mean L2 distance from each frame to its phoneme centroid
  2. Between-phoneme distance: pairwise L2 between phoneme centroids
  3. Temporal delta: mean L2 distance between consecutive frames

Usage
-----
    python scripts/36_hubert_l2_analysis.py
        [--embeddings-dir ...] [--manifest ...] [--layers 0,6,11]
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
OUTPUT_PATH = OUTPUT_BASE_ZH / "metrics" / "hubert_l2_analysis.json"

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


def analyse_l2(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute within-class scatter and between-class distances."""
    mask = labels != "_oob_"
    X = X[mask]
    labels = labels[mask]
    if len(X) < 3:
        return {}

    classes = sorted(set(labels.tolist()))
    centroids: dict[str, np.ndarray] = {}
    for cls in classes:
        m = labels == cls
        if m.sum() >= 2:
            centroids[cls] = X[m].mean(axis=0)

    centroid_list = list(centroids.keys())

    # Within-phoneme scatter
    within_dists = []
    for cls, c in centroids.items():
        m = labels == cls
        if m.sum() >= 2:
            dists = np.linalg.norm(X[m] - c, axis=1)
            within_dists.extend(dists.tolist())

    within_mean = float(np.mean(within_dists)) if within_dists else float("nan")
    within_std = float(np.std(within_dists)) if within_dists else float("nan")

    # Between-phoneme distances
    between_dists = []
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            d = float(np.linalg.norm(centroids[centroid_list[i]] - centroids[centroid_list[j]]))
            between_dists.append(d)

    between_mean = float(np.mean(between_dists)) if between_dists else float("nan")
    between_std = float(np.std(between_dists)) if between_dists else float("nan")

    # Separation index: between / within
    si = between_mean / within_mean if within_mean > 0 else float("nan")

    return {
        "within_mean": within_mean,
        "within_std": within_std,
        "between_mean": between_mean,
        "between_std": between_std,
        "separation_index": si,
        "n_classes": len(centroid_list),
        "n_frames": int(mask.sum()),
    }


def temporal_l2_delta(X: np.ndarray) -> float:
    """Mean L2 distance between consecutive frames."""
    if X.shape[0] < 2:
        return float("nan")
    deltas = np.linalg.norm(X[1:] - X[:-1], axis=1)
    return float(np.mean(deltas))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=str, default=str(EMBEDDINGS_DIR))
    p.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    p.add_argument("--model", type=str, default="hubert")
    p.add_argument("--layers", type=str, default="0,6,11")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    embeddings_dir = Path(args.embeddings_dir)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if output_path.exists() and not args.no_cache:
        newest_in = max(
            (p.stat().st_mtime for p in [embeddings_dir, manifest_path] if p.exists()),
            default=0.0,
        )
        if output_path.stat().st_mtime >= newest_in:
            logger.info("cached: %s", output_path)
            return 0

    alignment_entries = load_alignment(manifest_path)
    if args.smoke:
        alignment_entries = alignment_entries[:4]
    logger.info("Manifest: %d entries", len(alignment_entries))

    results: dict = {"model": args.model, "layers": {}, "per_utterance": {}}

    for layer in layers:
        logger.info("=== Layer %d ===", layer)
        layer_data: dict = {"phoneme": {}, "viseme": {}, "temporal": {}}

        # Collect all frames per condition
        nat_phoneme_X, nat_phoneme_y = [], []
        tts_phoneme_X, tts_phoneme_y = [], []
        nat_viseme_X, nat_viseme_y = [], []
        tts_viseme_X, tts_viseme_y = [], []

        per_utt = []

        for entry in alignment_entries:
            sid = str(entry.get("utterance_id", entry.get("sample_id", "unknown")))
            cond = entry["condition"]
            tokens = entry.get("tokens", [])

            npy_path = embeddings_dir / f"{embedding_file_stem(entry, args.model)}.npy"
            if not npy_path.exists():
                continue
            X = load_hubert_frames(npy_path, layer)
            T = X.shape[0]
            y_phn = assign_frame_labels(T, HUERT_FRAME_STRIDE, TARGET_SR, tokens, "phoneme")
            y_vis = assign_frame_labels(T, HUERT_FRAME_STRIDE, TARGET_SR, tokens, "viseme")

            # Temporal
            if T >= 2:
                td = temporal_l2_delta(X)
                layer_data["temporal"].setdefault(cond, []).append(td)

            # Per-utterance phoneme within
            m = y_phn != "_oob_"
            if m.sum() >= 3:
                phn_analysis = analyse_l2(X, y_phn)
                per_utt.append({"sample_id": sid, "condition": cond, "layer": layer,
                                "target": "phoneme", "metrics": phn_analysis})
                if cond == "natural":
                    nat_phoneme_X.append(X[m])
                    nat_phoneme_y.append(y_phn[m])
                else:
                    tts_phoneme_X.append(X[m])
                    tts_phoneme_y.append(y_phn[m])

            # Viseme
            m = y_vis != "_oob_"
            if m.sum() >= 3:
                vis_analysis = analyse_l2(X, y_vis)
                per_utt.append({"sample_id": sid, "condition": cond, "layer": layer,
                                "target": "viseme", "metrics": vis_analysis})
                if cond == "natural":
                    nat_viseme_X.append(X[m])
                    nat_viseme_y.append(y_vis[m])
                else:
                    tts_viseme_X.append(X[m])
                    tts_viseme_y.append(y_vis[m])

        # Pooled analysis per condition
        for cond_name, X_list, y_list in [
            ("natural", nat_phoneme_X, nat_phoneme_y),
            ("tts", tts_phoneme_X, tts_phoneme_y),
        ]:
            if X_list:
                layer_data["phoneme"][cond_name] = analyse_l2(np.concatenate(X_list), np.concatenate(y_list))
        for cond_name, X_list, y_list in [
            ("natural", nat_viseme_X, nat_viseme_y),
            ("tts", tts_viseme_X, tts_viseme_y),
        ]:
            if X_list:
                layer_data["viseme"][cond_name] = analyse_l2(np.concatenate(X_list), np.concatenate(y_list))

        # Compute temporal summary
        temporal_summary = {}
        for cond in layer_data["temporal"]:
            arr = np.array(layer_data["temporal"][cond])
            if len(arr):
                temporal_summary[f"{cond}_mean"] = float(arr.mean())
                temporal_summary[f"{cond}_std"] = float(arr.std())
        # Natural-to-TTS delta (backward compat)
        nat_t = np.array(layer_data["temporal"].get("natural", []))
        tts_t = np.array(layer_data["temporal"].get("tts", []))
        temporal_summary["delta_nat_minus_tts"] = float(nat_t.mean() - tts_t.mean()) if len(nat_t) and len(tts_t) else None
        layer_data["temporal"] = temporal_summary

        results["layers"][f"L{layer}"] = layer_data
        results["per_utterance"] = {f"{x['sample_id']}_{x['condition']}_{x['target']}": x for x in per_utt}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info("Wrote → %s", output_path)

    # Summary table
    _print_table(results)
    return 0


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    None"
    return f"{v:8.2f}"


def _print_table(results):
    for target in ["phoneme", "viseme"]:
        print(f"\n{'='*80}")
        print(f"  {target.upper()} L2 ANALYSIS")
        print(f"{'='*80}")
        print(f"{'Layer':>6s} {'Cond':>7s} {'within':>8s} {'between':>8s} {'sep_idx':>8s} {'n_cls':>5s} {'frames':>7s}")
        print("-" * 52)
        for layer_key in sorted(results["layers"].keys()):
            ld = results["layers"][layer_key]
            for cond in ["natural", "tts"]:
                m = ld[target].get(cond) or {}
                print(f"{layer_key:>6s} {cond:>7s} {_fmt(m.get('within_mean'))} "
                      f"{_fmt(m.get('between_mean'))} {_fmt(m.get('separation_index'))} "
                      f"{str(m.get('n_classes', '')):>5s} {str(m.get('n_frames', '')):>7s}")

    print(f"\n{'='*80}")
    print("  TEMPORAL L2 DELTA (frame-to-frame)")
    print(f"{'='*80}")
    print(f"{'Layer':>6s} {'nat_mean':>8s} {'tts_mean':>8s} {'Δ':>8s}")
    print("-" * 32)
    for layer_key in sorted(results["layers"].keys()):
        ld = results["layers"][layer_key]
        t = ld["temporal"]
        print(f"{layer_key:>6s} {_fmt(t.get('natural_mean'))} {_fmt(t.get('tts_mean'))} "
              f"{_fmt(t.get('delta_nat_minus_tts'))}")


if __name__ == "__main__":
    sys.exit(main())
