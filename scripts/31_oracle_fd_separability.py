#!/usr/bin/env python3
"""Construct Fd = FC(Fs ⊕ Fp) and measure Fs-mediated separability gain.

Per spec Section 6.4, this script computes three tiers of separability on the
HuBERT L11 frame embeddings of each sample:

  - Fp_only     : the HuBERT L11 embedding directly
  - Fd_zero     : Fp + Fs_broadcast  (no projection; sanity baseline)
  - Fd_random   : FC(orthogonal-init)(Fp + Fs_broadcast)

For each tier, computes the same viseme-level separability metrics as
script 16. The key signal is ``gain_random_vs_zero``, which is the
Fs-specific contribution after accounting for the FC's orthogonal rotation.

Per the M3 mitigation, H2 is supported only when ≥ 3 of 5 core metrics
(silhouette↑, boundary_sharpness↑, segment_stability↑, intra_class_dist↓,
fisher↑) move in favorable direction across the 13 paired samples.

Usage
-----
    python scripts/31_oracle_fd_separability.py [--samples IDS] [--smoke] \\
        [--fp-layer 11] [--fd-seed 42]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import ENGLISH_STUDY_SAMPLES, OUTPUT_BASE_EN, TARGET_SR

EMBEDDINGS_DIR = OUTPUT_BASE_EN / "embeddings"
ORACLE_FS_PATH = OUTPUT_BASE_EN / "metrics" / "oracle_fs.json"
ALIGNMENT_PATH = OUTPUT_BASE_EN / "manifest" / "alignment.json"
OUTPUT_PATH = OUTPUT_BASE_EN / "metrics" / "oracle_fd_separability.json"

DEFAULT_FP_LAYER = 11   # best from Phase 1 Chinese analysis
DEFAULT_FD_SEED = 42

DEFAULT_REQUESTED_LAYERS: list[int] = [0, 6, 11, 12]


CORE_METRICS_FAVORABLE: dict[str, int] = {
    "silhouette": +1,
    "boundary_sharpness": +1,
    "segment_stability": +1,
    "intra_class_dist": -1,   # lower is better
    "fisher": +1,
}


# ---------------------------------------------------------------------------
# Lazily import script 16 (script 16's filename is normal — but keep symmetry
# with the lazy pattern used for script 27 → 28)
# ---------------------------------------------------------------------------


_sep_module = None


def _get_separability_module():
    global _sep_module
    if _sep_module is not None:
        return _sep_module
    script16 = Path(__file__).resolve().parent / "16_feature_separability.py"
    spec = importlib.util.spec_from_file_location("separability", script16)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _sep_module = m
    return _sep_module


# ---------------------------------------------------------------------------
# Fp loader
# ---------------------------------------------------------------------------


def load_fp_at_layer(
    npy_path: Path,
    fp_layer: int,
    requested_layers: list[int] | None = None,
) -> np.ndarray:
    """Load HuBERT .npy output and slice the row for ``fp_layer``.

    script 15 saves embeddings with shape (n_requested_layers, n_frames, dim).
    The default requested layers are [0, 6, 11, 12]; L11 is at index 2.
    """
    if requested_layers is None:
        requested_layers = DEFAULT_REQUESTED_LAYERS
    arr = np.load(npy_path)
    try:
        idx = requested_layers.index(fp_layer)
    except ValueError:
        raise ValueError(f"fp_layer={fp_layer} not in requested_layers={requested_layers}")
    if idx >= arr.shape[0]:
        raise ValueError(f"layer index {idx} exceeds saved layers {arr.shape[0]}")
    return arr[idx].astype(np.float32)   # (T, 768)


# ---------------------------------------------------------------------------
# Fd construction
# ---------------------------------------------------------------------------


def build_fc_orthogonal(hidden: int = 768, seed: int = 42):
    """Build a 2-layer FC with orthogonal weight init and zero bias.

    Returns an nn.Sequential in eval mode. The output is deterministic given
    the seed.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError(f"torch required for build_fc_orthogonal: {e}")

    torch.manual_seed(seed)
    fc = nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden),
    )
    for m in fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)
    fc.eval()
    return fc


def construct_fd(
    Fp: np.ndarray,
    Fs: np.ndarray,
    fc=None,
) -> np.ndarray:
    """Broadcast Fs to (T, dim) and add to Fp; optionally pass through fc.

    If ``fc`` is None, returns ``Fs_broadcast + Fp`` (the Fd_zero tier).
    Otherwise, returns ``fc(Fs_broadcast + Fp)`` (the Fd_random tier).
    """
    T = Fp.shape[0]
    Fs_broadcast = np.tile(Fs[None, :], (T, 1)).astype(np.float32)
    fused = Fs_broadcast + Fp
    if fc is None:
        return fused.astype(np.float32)
    try:
        import torch
    except ImportError as e:
        raise ImportError(f"torch required for Fd_random tier: {e}")
    with torch.no_grad():
        out = fc(torch.from_numpy(fused)).cpu().numpy().astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Separability per variant
# ---------------------------------------------------------------------------


def _attach_viseme_labels(
    frame_times: np.ndarray,
    tokens: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """For each frame, look up the viseme of the token containing its centre time.

    Returns (viseme_labels, arpabet_labels). Frames outside any token get
    label "__out_of_range__" and are excluded.
    """
    vis = np.empty(len(frame_times), dtype=object)
    arp = np.empty(len(frame_times), dtype=object)
    for i, t in enumerate(frame_times):
        matched = None
        for tok in tokens:
            if tok["start_s"] <= t < tok["end_s"]:
                matched = tok
                break
        if matched is None:
            vis[i] = "__out_of_range__"
            arp[i] = "__out_of_range__"
        else:
            vis[i] = matched.get("viseme", "other")
            arp[i] = matched.get("token", "other")
    return vis, arp


def compute_separability_for_variant(
    variant: np.ndarray,           # (T, 768)
    frame_times: np.ndarray,       # (T,)
    tokens: list[dict],
) -> dict:
    """Compute the 5 core + supplementary separability metrics on a variant.

    Reuses metric functions from script 16. Skips classes with fewer than
    ``MIN_CLASS_SAMPLES`` (≥3 by default) samples.
    """
    _sep = _get_separability_module()
    vis_labels, arp_labels = _attach_viseme_labels(frame_times, tokens)

    # Drop frames outside any token (silence-only or trailing)
    mask = vis_labels != "__out_of_range__"
    X = variant[mask]
    vis_labels = vis_labels[mask]
    arp_labels = arp_labels[mask]

    # Filter classes with ≥ 3 items
    counts = Counter(vis_labels.tolist())
    keep = [v for v, c in counts.items() if c >= 3]
    keep_mask = np.isin(vis_labels, keep)
    X_v = X[keep_mask]
    vis_v = vis_labels[keep_mask]
    frame_times_v = frame_times[mask][keep_mask]

    out: dict = {}

    if len(set(vis_v.tolist())) >= 2 and len(vis_v) >= 5:
        # Three simple metrics: (embedding, labels) signature
        for metric_name, func_name in [
            ("intra_class_dist", "intra_class_variance"),
            ("silhouette", "_silhouette_cosine"),
            ("fisher", "fisher_ratio"),
        ]:
            func = getattr(_sep, func_name, None)
            if func is None:
                out[metric_name] = float("nan")
                continue
            try:
                out[metric_name] = float(func(X_v, vis_v))
            except Exception as e:
                logger.debug("metric %s failed: %s", metric_name, e, exc_info=True)
                out[metric_name] = float("nan")

        # boundary_sharpness returns (boundary_change, segment_stability) tuple.
        # It takes (frame_times, frame_embeddings, token_boundaries).
        # token_boundaries = list of end_s for each non-silence token boundary.
        try:
            boundaries = sorted(set(
                float(tok["end_s"])
                for tok in tokens
                if tok.get("viseme") != "sil" and "end_s" in tok
            ))
            bs_change, ss_within = _sep.boundary_sharpness(frame_times_v, X_v, boundaries)
            out["boundary_sharpness"] = float(bs_change)
            out["segment_stability"] = float(ss_within)
        except Exception as e:
            logger.debug("boundary/segment metrics failed: %s", e, exc_info=True)
            out["boundary_sharpness"] = float("nan")
            out["segment_stability"] = float("nan")
    else:
        for k in CORE_METRICS_FAVORABLE:
            out[k] = float("nan")
    return out


def process_sample(
    sample_id: int,
    condition: str,
    fp_path: Path,
    oracle_fs_entries: list[dict],
    alignment_entries: list[dict],
    fc_random,
    fp_layer: int = DEFAULT_FP_LAYER,
) -> dict:
    """Compute Fp_only / Fd_zero / Fd_random separability for one sample."""
    Fp = load_fp_at_layer(fp_path, fp_layer)
    T = Fp.shape[0]
    frame_stride = 320  # HuBERT default; matches script 15
    frame_times = np.arange(T, dtype=np.float32) * (frame_stride / TARGET_SR)

    # Pull Fs (CLS) for this sample_id + condition
    fs_entry = next(
        (e for e in oracle_fs_entries
         if e["sample_id"] == sample_id and e["condition"] == condition),
        None,
    )
    if fs_entry is None:
        raise ValueError(
            f"oracle Fs missing for sample_id={sample_id} condition={condition}"
        )
    if fs_entry.get("is_degenerate"):
        degenerate_result = {
            "sample_id": sample_id,
            "condition": condition,
            "skipped": "Fs_degenerate",
            "fp_layer": fp_layer,
            "T_frames": 0,
            "fs_norm": 0.0,
            "fp_metrics": {k: float("nan") for k in CORE_METRICS_FAVORABLE},
            "fd_zero_metrics": {k: float("nan") for k in CORE_METRICS_FAVORABLE},
            "fd_random_metrics": {k: float("nan") for k in CORE_METRICS_FAVORABLE},
        }
        # Flat gain keys per spec
        for k in CORE_METRICS_FAVORABLE:
            degenerate_result[f"gain_zero_vs_fp_{k}"] = 0.0
            degenerate_result[f"gain_random_vs_fp_{k}"] = 0.0
            degenerate_result[f"gain_random_vs_zero_{k}"] = 0.0
        return degenerate_result
    Fs = np.asarray(fs_entry["Fs_cls"], dtype=np.float32)

    # Pull alignment tokens
    align_entry = next(
        (e for e in alignment_entries
         if e["sample_id"] == sample_id and e["condition"] == condition),
        None,
    )
    if align_entry is None:
        raise ValueError(
            f"alignment entry missing for sample_id={sample_id} condition={condition}"
        )
    tokens = align_entry.get("tokens", [])

    # Three tiers
    Fd_zero = construct_fd(Fp, Fs, fc=None)
    Fd_random = construct_fd(Fp, Fs, fc=fc_random)

    m_fp = compute_separability_for_variant(Fp, frame_times, tokens)
    m_zero = compute_separability_for_variant(Fd_zero, frame_times, tokens)
    m_random = compute_separability_for_variant(Fd_random, frame_times, tokens)

    gains = {}
    for k in CORE_METRICS_FAVORABLE:
        if k in m_fp and k in m_zero and k in m_random:
            gains["gain_zero_vs_fp_" + k] = float(m_zero[k] - m_fp[k])
            gains["gain_random_vs_fp_" + k] = float(m_random[k] - m_fp[k])
            gains["gain_random_vs_zero_" + k] = float(m_random[k] - m_zero[k])

    return {
        "sample_id": sample_id,
        "condition": condition,
        "fp_layer": fp_layer,
        "T_frames": int(T),
        "fs_norm": float(np.linalg.norm(Fs)),
        "fp_metrics": m_fp,
        "fd_zero_metrics": m_zero,
        "fd_random_metrics": m_random,
        **gains,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str,
        default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fp-layer", type=int, default=DEFAULT_FP_LAYER)
    parser.add_argument("--fd-seed", type=int, default=DEFAULT_FD_SEED)
    parser.add_argument("--embeddings-dir", type=str, default=str(EMBEDDINGS_DIR))
    parser.add_argument("--oracle-fs", type=str, default=str(ORACLE_FS_PATH))
    parser.add_argument("--alignment", type=str, default=str(ALIGNMENT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]

    embeddings_dir = Path(args.embeddings_dir)
    oracle_fs_path = Path(args.oracle_fs)
    alignment_path = Path(args.alignment)
    output_path = Path(args.output)

    if output_path.exists() and not args.no_cache:
        # Re-run if inputs are newer
        newest_input = max(
            (p.stat().st_mtime for p in (embeddings_dir, oracle_fs_path, alignment_path) if p.exists()),
            default=0.0,
        )
        if output_path.stat().st_mtime >= newest_input:
            print(f"[31] cached: {output_path}")
            return 0

    if not oracle_fs_path.exists():
        print(f"[31] oracle Fs not found at {oracle_fs_path}; run script 29 first.", file=sys.stderr)
        return 1
    if not alignment_path.exists():
        print(f"[31] alignment not found at {alignment_path}; run script 28 first.", file=sys.stderr)
        return 1

    oracle_fs_payload = json.loads(oracle_fs_path.read_text())
    oracle_fs_entries = oracle_fs_payload.get("entries", oracle_fs_payload)
    alignment_payload = json.loads(alignment_path.read_text())
    alignment_entries = alignment_payload.get("manifest", alignment_payload)

    fc_random = build_fc_orthogonal(hidden=768, seed=args.fd_seed)

    results: list[dict] = []
    for sid in sample_ids:
        for cond in ("natural", "tts"):
            npy_path = embeddings_dir / f"{sid}_{cond}_raw_hubert.npy"
            if not npy_path.exists():
                print(f"[31] missing embedding: {npy_path}", file=sys.stderr)
                continue
            try:
                r = process_sample(
                    sid, cond, npy_path,
                    oracle_fs_entries, alignment_entries,
                    fc_random, fp_layer=args.fp_layer,
                )
                results.append(r)
                print(f"[31] sample {sid} {cond}: T={r['T_frames']}")
            except Exception as exc:
                print(f"[31] sample {sid} {cond} error: {exc}", file=sys.stderr)
                results.append({
                    "sample_id": sid, "condition": cond,
                    "skipped": str(exc),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "fp_layer": args.fp_layer,
        "fd_seed": args.fd_seed,
        "n_results": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"[31] wrote {len(results)} entries → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
