#!/usr/bin/env python3
"""B2 — Per-layer separability curves for HuBERT / XLS-R.

Computes per-layer separability metrics (intra/inter-class distance, fisher
ratio, silhouette, phoneme/viseme probe) for natural vs TTS across a dense
layer grid, then reports the layer profile of each metric.  Purpose: find each
model's optimal layer and test the literature pattern (Pasad et al. ICASSP
2023) that HuBERT-family phonetic information peaks at high layers while
wav2vec2/XLS-R-family peaks at middle layers.

Efficiency: each NPY (n_layers × n_frames × dim) is loaded exactly once and
pooled for every saved layer, instead of reloading per layer.

Usage
-----
    python scripts/37_layer_curves.py                          \\
        [--embeddings-dir results/aishell100_phoneme/embeddings]  \\
        [--models hubert,xlsr] [--level viseme] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "results/aishell100_phoneme/embeddings"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/aishell100_phoneme/layer_curves"

FRAME_STRIDE = 320
TARGET_SR = 16000


def _load_f16() -> Any:
    """Import scripts/16_feature_separability.py as a module (digit-prefixed name)."""
    spec = importlib.util.spec_from_file_location(
        "feature_separability_16", str(PROJECT_ROOT / "scripts/16_feature_separability.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gather_by_layer(
    entries: list[dict],
    model: str,
    level: str,
    condition: str,
    variant: str,
    layers: list[int],
    f16: Any,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pool per-token vectors for every saved layer, loading each NPY once.

    Returns ``{layer: (embeddings, labels, groups)}`` where embeddings are the
    concatenated per-token vectors across all samples for one condition.
    """
    acc: dict[int, list] = defaultdict(lambda: [[], [], []])  # layer -> 3 lists
    n_used = 0
    for meta in entries:
        if meta.get("model") != model:
            continue
        if f16._analysis_condition(meta) != condition:
            continue
        if meta.get("variant") != variant:
            continue
        tokens = meta.get("tokens", [])
        if not tokens:
            continue
        saved_layers = meta.get("layers", [])
        npy_path = Path(meta.get("_npy_path", ""))
        if not npy_path.exists():
            continue
        try:
            emb_all = np.load(npy_path)  # (n_saved_layers, n_frames, dim)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load %s: %s", npy_path.name, exc)
            continue
        group = str(meta.get("speaker_id", meta.get("paired_key", "?")))
        # ``saved_layers`` in the metadata may over-report when extraction
        # dropped layers exceeding model depth (e.g. HuBERT records 0..12 but
        # stores 0..11).  Guard with the actual array depth.
        n_stored = emb_all.shape[0]
        for li, layer in enumerate(saved_layers[:n_stored]):
            if layer not in layers:
                continue
            layer_emb = emb_all[li]
            pooled, label_map = f16._pool_frames_for_layer(
                layer_emb, FRAME_STRIDE, TARGET_SR, tokens
            )
            if pooled.shape[0] == 0 or level not in label_map:
                continue
            acc[layer][0].append(pooled)
            acc[layer][1].append(label_map[level])
            acc[layer][2].append(np.full(pooled.shape[0], group, dtype=object))
        n_used += 1
    logger.info("  %s/%s: %d samples", model, condition, n_used)

    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for layer, (emb_list, lab_list, grp_list) in acc.items():
        if not emb_list:
            continue
        out[layer] = (
            np.concatenate(emb_list, axis=0),
            np.concatenate(lab_list, axis=0),
            np.concatenate(grp_list, axis=0),
        )
    return out


def run(
    embeddings_dir: Path,
    models: list[str],
    level: str,
    out_dir: Path,
) -> int:
    f16 = _load_f16()
    entries = f16._discover_embedding_files(embeddings_dir)
    if not entries:
        logger.error("No embedding files found in %s", embeddings_dir)
        return 1

    # Saved layers per model (union across entries).
    layers_by_model: dict[str, set] = defaultdict(set)
    for e in entries:
        model = e.get("model")
        if model:
            layers_by_model[model].update(e.get("layers", []))
    logger.info("Models: %s", {m: sorted(v) for m, v in layers_by_model.items()})

    conditions = ["natural", "faster_qwen3"]
    curves: dict[str, dict[int, dict[str, dict]]] = {}
    for model in models:
        layers = sorted(layers_by_model.get(model, []))
        if not layers:
            logger.warning("No layers recorded for %s", model)
            continue
        logger.info("Building curve for %s (%d layers)", model, len(layers))
        do_probe = model == "hubert"  # XLS-R probe too slow (1024-dim)
        per_cond: dict[str, dict[int, tuple]] = {}
        for cond in conditions:
            per_cond[cond] = gather_by_layer(entries, model, level, cond, "raw", layers, f16)
        curves[model] = {}
        for layer in layers:
            curves[model][layer] = {}
            for cond in conditions:
                if layer not in per_cond[cond]:
                    continue
                emb, lab, grp = per_cond[cond][layer]
                if emb.shape[0] == 0:
                    continue
                metrics = f16._compute_metrics_for_pool(
                    emb, lab, grp, None, None, do_probe=do_probe
                )
                # Majority-class baseline for probe interpretation.
                if lab.size:
                    counts = np.unique(lab, return_counts=True)[1]
                    baseline = float(counts.max() / counts.sum())
                else:
                    baseline = None
                curves[model][layer][cond] = {
                    k: metrics.get(k) for k in (
                        "intra_class_dist", "inter_class_dist", "fisher_ratio",
                        "silhouette", "probe_accuracy", "probe_f1_macro",
                    )
                }
                curves[model][layer][cond]["probe_baseline"] = baseline
                logger.info(
                    "  %s l%-2d %-9s intra=%.4f inter=%.4f fisher=%.3f sil=%.3f probe=%s",
                    model, layer, cond,
                    metrics.get("intra_class_dist") or float("nan"),
                    metrics.get("inter_class_dist") or float("nan"),
                    metrics.get("fisher_ratio") or float("nan"),
                    metrics.get("silhouette") or float("nan"),
                    "%.3f" % metrics["probe_accuracy"] if metrics.get("probe_accuracy") is not None else "n/a",
                )
            # Degenerate-layer gate: a representation that collapses to near
            # zero distances (all tokens identical) is NOT a compactness peak.
            # Flag the layer if either condition shows collapse.
            if curves[model][layer]:
                inter_vals = [float(c["inter_class_dist"]) for c in curves[model][layer].values()
                              if isinstance(c.get("inter_class_dist"), (int, float))]
                intra_vals = [float(c["intra_class_dist"]) for c in curves[model][layer].values()
                              if isinstance(c.get("intra_class_dist"), (int, float))]
                degenerate = bool(inter_vals and intra_vals and
                                  max(inter_vals) < 1e-3 and max(intra_vals) < 1e-3)
            else:
                degenerate = False
            for cond in curves[model][layer]:
                curves[model][layer][cond]["degenerate"] = degenerate

    # ---- Peak-layer summary per model & metric ---------------------------
    # intra_class_dist: LOWER is more compact → peak = min.  All other
    # metrics are higher-is-better → peak = max.
    metrics_list = ["intra_class_dist", "inter_class_dist", "fisher_ratio", "silhouette", "probe_accuracy"]
    lower_is_better = {"intra_class_dist"}
    peaks: dict[str, dict[str, dict]] = {}

    def _clean_metric(v: Any) -> float | None:
        return float(v) if (v is not None and not np.isnan(float(v))) else None

    for model, curve in curves.items():
        peaks[model] = {}
        for metric in metrics_list:
            per_cond_peaks: dict[str, dict] = {}
            for cond in conditions:
                vals: dict[int, float] = {}
                for l, c in curve.items():
                    if any(cc.get("degenerate") for cc in c.values()):
                        continue  # collapsed representation is not a peak
                    mv = _clean_metric(c.get(cond, {}).get(metric))
                    if mv is not None:
                        vals[int(l)] = mv
                if not vals:
                    continue
                pick = min if metric in lower_is_better else max
                peak_layer = pick(vals, key=vals.__getitem__)
                per_cond_peaks[cond] = {
                    "peak_layer": int(peak_layer),
                    "peak_value": vals[peak_layer],
                    "values": vals,
                }
            # natural-vs-TTS delta is a single (model, metric) quantity; the
            # same curve for both conditions, so compute it once.
            deltas: dict[int, float] = {}
            for l, c in curve.items():
                if any(cc.get("degenerate") for cc in c.values()):
                    continue
                nv = _clean_metric(c.get("natural", {}).get(metric))
                tv = _clean_metric(c.get("faster_qwen3", {}).get(metric))
                if nv is not None and tv is not None:
                    deltas[int(l)] = tv - nv
            max_abs_delta_layer = int(max(deltas, key=lambda l: abs(deltas[l]))) if deltas else None
            if per_cond_peaks:
                peaks[model][metric] = {
                    "conditions": per_cond_peaks,
                    "tts_minus_natural": deltas,
                    "max_abs_delta_layer": max_abs_delta_layer,
                }

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "embeddings_dir": str(embeddings_dir),
            "models": models,
            "level": level,
            "conditions": conditions,
            "do_probe_models": ["hubert"],
            "note": "probe computed for hubert only; XLS-R probe skipped (1024-dim cost)",
        },
        "curves": {m: {int(l): c for l, c in curve.items()} for m, curve in curves.items()},
        "peaks": peaks,
    }
    out_json = out_dir / "layer_curves.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    # ---- Console summary --------------------------------------------------
    print("\n=== B2 Layer curves ===")
    for model, pks in peaks.items():
        print("\n[%s]" % model)
        for metric, entry in pks.items():
            line = "  %-18s" % metric
            for cond in conditions:
                if cond in entry["conditions"]:
                    c = entry["conditions"][cond]
                    line += "  %s: l%d (%.3f)" % (cond[:4], c["peak_layer"], c["peak_value"])
            mdl = entry.get("max_abs_delta_layer")
            if mdl is not None and metric != "silhouette":
                line += "   |Δ|max@l%d" % mdl
            print(line)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--models", type=str, default="hubert,xlsr")
    parser.add_argument("--level", type=str, default="viseme")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    return run(args.embeddings_dir, models, args.level, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
