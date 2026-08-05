#!/usr/bin/env python3
"""Phoneme / viseme recognition via nearest-centroid on HuBERT features.

For each condition (natural / TTS), compute per-class centroids from HuBERT
frame embeddings. Then classify every frame by nearest-centroid L2 distance,
measuring error rate (1 − accuracy).

Compares:
  1. Self-classification: train & test on same condition
  2. Cross-classification: natural centroids → classify TTS frames (and vice versa)
  3. The Δ accuracy between self and cross tells us prototype drift

Usage
-----
    # Phoneme probe (tone-stripped)
    python scripts/35_phoneme_recognition_probe.py --target phoneme

    # Viseme probe (Preston Blair 11-class)
    python scripts/35_phoneme_recognition_probe.py --target viseme
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
OUTPUT_PATH = OUTPUT_BASE_ZH / "metrics" / "phoneme_probe.json"
VISEME_OUTPUT_PATH = OUTPUT_BASE_ZH / "metrics" / "viseme_probe.json"

DEFAULT_LAYERS = [0, 6, 11]
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


def assign_frame_labels(
    T: int, frame_stride: int, sr: int, tokens: list[dict],
    target: str = "phoneme",  # "phoneme" | "viseme"
) -> np.ndarray:
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


def nearest_centroid_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, int, dict[str, float]]:
    """Classify test frames by nearest-centroid (L2) from train centroids.

    Returns (accuracy, n_test, per_class_acc).
    """
    classes = sorted(set(y_train.tolist()))
    centroids: dict[str, np.ndarray] = {}
    for cls in classes:
        mask = y_train == cls
        if mask.sum() == 0:
            continue
        centroids[cls] = X_train[mask].mean(axis=0)

    if not centroids:
        return float("nan"), 0, {}

    centroid_ids = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in centroid_ids])  # (K, D)

    # Brute-force L2 nearest-neighbor
    correct = 0
    per_class_correct: dict[str, int] = Counter()
    per_class_total: dict[str, int] = Counter()

    for i in range(len(X_test)):
        ref = y_test[i]
        # L2 distance to all centroids
        diffs = centroid_matrix - X_test[i]
        dists = np.linalg.norm(diffs, axis=1)
        pred = centroid_ids[int(np.argmin(dists))]
        per_class_total[ref] += 1
        if pred == ref:
            correct += 1
            per_class_correct[ref] += 1

    per_class_acc = {
        cls: per_class_correct.get(cls, 0) / max(per_class_total[cls], 1)
        for cls in per_class_total
    }

    return (
        float(correct) / max(len(X_test), 1),
        len(X_test),
        per_class_acc,
    )


def _evaluate_loo(
    X: np.ndarray, y: np.ndarray, utt_all: np.ndarray,
) -> dict:
    """Leave-one-utterance-out nearest-centroid classification."""
    unique_utts = np.unique(utt_all)
    per_utt_acc: list[float] = []
    per_utt_n: list[int] = []
    total_correct = 0
    total_frames = 0

    for test_utt in unique_utts:
        train_mask = utt_all != test_utt
        test_mask = utt_all == test_utt
        acc, n, _ = nearest_centroid_accuracy(
            X[train_mask], y[train_mask],
            X[test_mask], y[test_mask],
        )
        per_utt_acc.append(acc)
        per_utt_n.append(n)
        total_correct += int(round(acc * n))
        total_frames += n

    return {
        "per_utterance_acc": per_utt_acc,
        "per_utterance_n": per_utt_n,
        "total_accuracy": float(total_correct) / max(total_frames, 1),
        "total_frames": total_frames,
        "n_utterances": int(len(unique_utts)),
        "unique_classes": len(set(y.tolist())),
    }


def _cross_condition_evaluation(
    X_nat: np.ndarray, y_nat: np.ndarray,
    X_tts: np.ndarray, y_tts: np.ndarray,
) -> dict:
    """Classify TTS frames using natural centroids, and vice versa."""
    result = {}
    acc_n2t, n_n2t, _ = nearest_centroid_accuracy(X_nat, y_nat, X_tts, y_tts)
    result["natural_proto_to_tts"] = {"accuracy": acc_n2t, "total_frames": n_n2t}
    acc_t2n, n_t2n, _ = nearest_centroid_accuracy(X_tts, y_tts, X_nat, y_nat)
    result["tts_proto_to_natural"] = {"accuracy": acc_t2n, "total_frames": n_t2n}
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=str, default=str(EMBEDDINGS_DIR))
    p.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    p.add_argument("--model", type=str, default="hubert")
    p.add_argument("--layers", type=str, default="0,6,11")
    p.add_argument("--target", type=str, default="phoneme",
                   choices=["phoneme", "viseme"],
                   help="classification target: phoneme (44 classes) or viseme (11 classes)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", type=str, default="cpu")  # kept for future GPU use
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    layers = [int(x) for x in args.layers.split(",")]
    embeddings_dir = Path(args.embeddings_dir)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output) if args.output else (
        VISEME_OUTPUT_PATH if args.target == "viseme" else OUTPUT_PATH
    )

    if output_path.exists() and not args.no_cache:
        newest_in = 0.0
        for p in [embeddings_dir, manifest_path]:
            if p.exists():
                newest_in = max(newest_in, p.stat().st_mtime)
        if output_path.stat().st_mtime >= newest_in:
            logger.info("cached: %s", output_path)
            return 0

    alignment_entries = load_alignment(manifest_path)
    if args.smoke:
        alignment_entries = alignment_entries[:4]
    logger.info("Manifest: %d entries", len(alignment_entries))

    all_results: dict = {
        "model": args.model,
        "target": args.target,
        "layers": layers,
        "n_utterances": len(alignment_entries),
        "layer_results": {},
    }

    for layer in layers:
        logger.info("=== Layer %d ===", layer)

        # Collect frame-level data per condition
        nat_X = []  # list of (T, 768)
        nat_y = []  # list of (T,) labels
        nat_utt = []
        tts_X = []
        tts_y = []
        tts_utt = []

        for entry_idx, entry in enumerate(alignment_entries):
            sid = str(entry.get("utterance_id", entry.get("sample_id", "unknown")))
            cond = entry["condition"]
            tokens = entry.get("tokens", [])

            npy_path = embeddings_dir / f"{embedding_file_stem(entry, args.model)}.npy"
            if not npy_path.exists():
                logger.warning("Missing: %s", npy_path)
                continue

            fp = load_hubert_frames(npy_path, layer)
            T = fp.shape[0]
            labels = assign_frame_labels(T, HUERT_FRAME_STRIDE, TARGET_SR, tokens,
                                          target=args.target)

            mask = labels != "_oob_"
            fp_masked = fp[mask]
            labels_masked = labels[mask]
            if len(fp_masked) < 3:
                continue

            if cond == "natural":
                nat_X.append(fp_masked)
                nat_y.append(labels_masked)
                nat_utt.append(np.full(len(fp_masked), entry_idx, dtype=int))
            else:
                tts_X.append(fp_masked)
                tts_y.append(labels_masked)
                tts_utt.append(np.full(len(fp_masked), entry_idx, dtype=int))

        # Stack per condition
        if nat_X:
            X_nat = np.concatenate(nat_X)
            y_nat = np.concatenate(nat_y)
            utt_nat = np.concatenate(nat_utt)
        else:
            X_nat = y_nat = utt_nat = np.array([])
        if tts_X:
            X_tts = np.concatenate(tts_X)
            y_tts = np.concatenate(tts_y)
            utt_tts = np.concatenate(tts_utt)
        else:
            X_tts = y_tts = utt_tts = np.array([])

        logger.info("Natural: %d frames, %d utterances", len(y_nat), len(nat_X))
        logger.info("TTS:     %d frames, %d utterances", len(y_tts), len(tts_X))

        # --- Self LOO evaluation (within condition) ---
        eval_nat = _evaluate_loo(X_nat, y_nat, utt_nat) if len(y_nat) > 0 else {}
        eval_tts = _evaluate_loo(X_tts, y_tts, utt_tts) if len(y_tts) > 0 else {}

        # --- Cross-condition evaluation ---
        cross = {}
        if len(y_nat) > 0 and len(y_tts) > 0:
            cross = _cross_condition_evaluation(X_nat, y_nat, X_tts, y_tts)

        nat_acc = eval_nat.get("total_accuracy")
        tts_acc = eval_tts.get("total_accuracy")

        delta_per = None
        if nat_acc is not None and tts_acc is not None:
            delta_per = float(tts_acc) - float(nat_acc)

        cross_n2t = cross.get("natural_proto_to_tts", {}).get("accuracy")
        cross_t2n = cross.get("tts_proto_to_natural", {}).get("accuracy")

        layer_result = {
            "natural_self": {
                "accuracy": nat_acc,
                "total_frames": eval_nat.get("total_frames"),
                "n_utterances": eval_nat.get("n_utterances"),
                "n_classes": eval_nat.get("unique_classes"),
            },
            "tts_self": {
                "accuracy": tts_acc,
                "total_frames": eval_tts.get("total_frames"),
                "n_utterances": eval_tts.get("n_utterances"),
                "n_classes": eval_tts.get("unique_classes"),
            },
            "delta_tts_minus_nat_per": delta_per,
            "natural_proto_to_tts": cross.get("natural_proto_to_tts"),
            "tts_proto_to_natural": cross.get("tts_proto_to_natural"),
        }

        all_results["layer_results"][f"L{layer}"] = layer_result

        # Print summary
        print()
        print(f"  L{layer}: PER nat_self={_per(nat_acc)} tts_self={_per(tts_acc)} "
              f"Δ={_per_delta(delta_per)} "
              f"nat→tts={_per(cross_n2t)} tts→nat={_per(cross_t2n)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    logger.info("Wrote → %s", output_path)

    # Final table
    print()
    hdr = f"{'':>6s} {'nat_PER':>9s} {'tts_PER':>9s} {'Δ_PER':>9s} {'nat→tts':>9s} {'tts→nat':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for layer in layers:
        lr = all_results["layer_results"][f"L{layer}"]
        nat_acc = lr["natural_self"]["accuracy"]
        tts_acc = lr["tts_self"]["accuracy"]
        cross_n2t = (lr.get("natural_proto_to_tts") or {}).get("accuracy")
        cross_t2n = (lr.get("tts_proto_to_natural") or {}).get("accuracy")
        print(f"  L{layer}: {_per(nat_acc)} {_per(tts_acc)} "
              f"{_per_delta(lr['delta_tts_minus_nat_per'])} "
              f"{_per(cross_n2t)} {_per(cross_t2n)}")
    return 0


def _per(v):
    """Phoneme Error Rate = 100% - accuracy."""
    if v is None:
        return "    None"
    return f"{(1 - v) * 100:7.2f}%"


def _per_delta(v):
    """Delta in PER points."""
    if v is None:
        return "    None"
    return f"{v * 100:+7.2f}%"


def _fmt(v):
    if v is None:
        return "    None"
    return f"{v:8.4f}"


if __name__ == "__main__":
    sys.exit(main())
