#!/usr/bin/env python3
"""B4 — Link per-sample separability deltas to per-sample TFG SyncNet deltas.

For each of the 100 paired AISHELL-1 samples, the Ditto talking-head pipeline
generated natural and TTS videos and SyncNet scored them.  This script asks:
does a sample where TTS improves phoneme separability (relative to its natural
counterpart) also show TTS lip-sync improvement (SyncNet-C)?  And which
separability / prosody metrics best predict the lip-sync change?

Feature deltas come from:
  - 09 separability per-sample deltas (tts − natural) — results/.../metrics/separability_metrics.json
  - B1 duration per-sample deltas — results/.../duration_analysis/duration_jsd.json

SyncNet scores come from the server eval (04_eval/{natural_raw,tts_raw}/{sid}/syncnet.json),
aggregated into a single JSON with ``{sid: {"natural": {c,d}, "tts": {c,d}}}``.

Usage
-----
    python scripts/40_link_separability_to_tfg.py --syncnet JSON --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEP = PROJECT_ROOT / "results/aishell100_phoneme/metrics/separability_metrics.json"
DEFAULT_DUR = PROJECT_ROOT / "results/aishell100_phoneme/duration_analysis/duration_jsd.json"
DEFAULT_OUT = PROJECT_ROOT / "results/aishell100_phoneme/tfg_link"


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rho + p-value. Callers must pass NaN-free arrays."""
    from scipy.stats import rankdata

    ra = rankdata(a)
    rb = rankdata(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    n = len(a)
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
        from scipy.stats import t as tdist
        p = float(2 * tdist.sf(abs(t), n - 2))
    return rho, p


def _spearman_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> list[float]:
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(a)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ra = _spearman(a[idx], b[idx])[0]
        if not np.isnan(ra):
            rs.append(ra)
    if not rs:
        return [float("nan"), float("nan")]
    return [float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))]


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Standard Benjamini-Hochberg FDR (right-to-left monotonicisation)."""
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty_like(ranked)
    q[-1] = ranked[-1] * m / m
    for i in range(m - 2, -1, -1):
        q[i] = min(ranked[i] * m / (i + 1), q[i + 1])
    out = np.empty_like(q)
    out[order] = q
    return np.minimum(out, 1.0)


def load_syncnet(path: Path) -> dict[str, dict]:
    """Load aggregated syncnet scores ``{sid: {natural: {c,d}, tts: {c,d}}}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for sid, rec in data.items():
        if "natural" in rec and "tts" in rec:
            out[str(sid)] = rec
    return out


def load_feature_deltas() -> dict[str, dict[str, float]]:
    """Per-sample natural→TTS deltas keyed by metric label."""
    deltas: dict[str, dict[str, float]] = {}
    sep = json.loads(DEFAULT_SEP.read_text(encoding="utf-8"))
    for key, per_sid in sep.get("per_sample", {}).items():
        if not per_sid:
            continue
        deltas[key] = {str(sid): float(v) for sid, v in per_sid.items()}

    dur = json.loads(DEFAULT_DUR.read_text(encoding="utf-8"))
    for key, per_sid in dur.get("per_sample", {}).items():
        if not per_sid:
            continue
        deltas[f"dur_{key}"] = {
            str(sid): float(rec["tts"]) - float(rec["natural"]) for sid, rec in per_sid.items()
        }
    return deltas


def run(syncnet_path: Path, out_dir: Path) -> int:
    sync = load_syncnet(syncnet_path)
    if not sync:
        logger.error("No syncnet scores loaded")
        return 1
    logger.info("Loaded SyncNet scores for %d samples", len(sync))

    feature_deltas = load_feature_deltas()
    logger.info("Loaded %d feature delta metrics", len(feature_deltas))

    # Per-sample sync deltas (tts − natural).
    common_sids = sorted(sync.keys())
    sync_c = {sid: float(sync[sid]["tts"]["c"]) - float(sync[sid]["natural"]["c"]) for sid in common_sids}
    sync_d = {sid: float(sync[sid]["tts"]["d"]) - float(sync[sid]["natural"]["d"]) for sid in common_sids}

    results: list[dict] = []
    seen_vectors: set[tuple] = set()
    for metric, per_sid in feature_deltas.items():
        sids = [sid for sid in common_sids if sid in per_sid]
        # Drop NaN feature values (a single NaN sample would poison rankdata).
        sids = [sid for sid in sids if per_sid[sid] == per_sid[sid]]
        if len(sids) < 8:
            continue
        x = np.array([per_sid[sid] for sid in sids], dtype=np.float64)
        # Dedupe byte-identical metric vectors (e.g. segment_stability is the
        # same across viseme/initial/final levels) so the FDR is not inflated
        # by redundant tests.
        key = tuple(np.round(x, 6))
        if key in seen_vectors:
            continue
        seen_vectors.add(key)
        yc = np.array([sync_c[sid] for sid in sids], dtype=np.float64)
        yd = np.array([sync_d[sid] for sid in sids], dtype=np.float64)
        rho_c, p_c = _spearman(x, yc)
        rho_d, p_d = _spearman(x, yd)
        results.append({
            "metric": metric,
            "n": len(sids),
            "feature_delta_mean": float(np.mean(x)),
            "sync_c_rho": rho_c, "sync_c_p": p_c,
            "sync_c_ci": _spearman_ci(x, yc) if not np.isnan(rho_c) else [float("nan"), float("nan")],
            "sync_d_rho": rho_d, "sync_d_p": p_d,
        })
        logger.info("%-52s n=%d  sync-C rho=%+.3f p=%.3f | sync-D rho=%+.3f p=%.3f",
                    metric, len(sids), rho_c, p_c, rho_d, p_d)

    if not results:
        logger.error("No correlations computed")
        return 1

    # FDR across all sync-C and sync-D tests separately.
    pc = np.array([r["sync_c_p"] for r in results])
    pd_ = np.array([r["sync_d_p"] for r in results])
    qc = _fdr_bh(pc)
    qd = _fdr_bh(pd_)
    for r, q in zip(results, qc):
        r["sync_c_q"] = float(q)
        r["sync_c_significant"] = bool(q < 0.05)
    for r, q in zip(results, qd):
        r["sync_d_q"] = float(q)
        r["sync_d_significant"] = bool(q < 0.05)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "syncnet_samples": len(sync),
            "n_metrics": len(results),
            "note": "feature deltas are natural->TTS; sync deltas are tts-natural; Spearman across samples",
        },
        "results": sorted(results, key=lambda r: (not r["sync_c_significant"], abs(r["sync_c_rho"])), reverse=True),
    }
    out_json = out_dir / "separability_tfg_link.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    print("\n=== B4: separability -> TFG SyncNet correlation ===")
    print("Top metrics by |sync-C rho| (significant after FDR marked *):")
    for r in result["results"]:
        star = "*" if r["sync_c_significant"] else " "
        print("%s %-52s rho=%+.3f p=%.3f q=%.3f" % (star, r["metric"], r["sync_c_rho"], r["sync_c_p"], r["sync_c_q"]))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--syncnet", type=Path, required=True, help="aggregated syncnet JSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args.syncnet, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
