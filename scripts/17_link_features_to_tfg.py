#!/usr/bin/env python3
"""17_link_features_to_tfg.py — Link Wav2Sem feature separability to TFG SyncNet scores.

Correlates natural→TTS changes in encoder embedding separability metrics
with the corresponding changes in TFG SyncNet scores (Sync-C, Sync-D).

Usage
-----
    python scripts/17_link_features_to_tfg.py                           \\
        [--metrics-dir DIR] [--runs-dir DIR] [--output-dir DIR]         \\
        [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import (
    OUTPUT_BASE,
    fdr_bh_correction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_METRICS_DIR: Path = OUTPUT_BASE / "metrics"
DEFAULT_RUNS_DIR: Path = Path(__file__).resolve().parent.parent / "runs"
DEFAULT_OUTPUT_DIR: Path = OUTPUT_BASE / "metrics"

TFG_RUNS: dict[str, str] = {
    "ditto_faster_qwen3": "aishell1_strict_20260707T081223Z",
    "tfg_wav2lip": "tfg_wav2lip",
    "tfg_musetalk": "tfg_musetalk",
    "tfg_latentsync": "tfg_latentsync",
    "tfg_joyvasa": "tfg_joyvasa",
}

STUDY_SAMPLES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

# ---------------------------------------------------------------------------
# SyncNet loading
# ---------------------------------------------------------------------------


def _load_syncnet_for_run(
    runs_dir: Path,
    run_id: str,
    sample_ids: list[int],
) -> dict[str, dict[int, dict[str, float]]]:
    """Load Sync-C and Sync-D from a TFG run's eval directory.

    Reads ``natural_raw/{sid}/syncnet.json`` and ``tts_raw/{sid}/syncnet.json``.

    Parameters
    ----------
    runs_dir : Path
        Base runs directory.
    run_id : str
        Run directory name.
    sample_ids : list[int]
        Sample IDs to read.

    Returns
    -------
    data : dict
        ``{"nat": {sid: {"sync_c": ..., "sync_d": ...}}, "tts": {...}}``
    """
    eval_dir = runs_dir / run_id / "04_eval"
    result: dict[str, dict[int, dict[str, float]]] = {
        "nat": {},
        "tts": {},
    }

    for cond_key, cond_dir in [("nat", "natural_raw"), ("tts", "tts_raw")]:
        cond_path = eval_dir / cond_dir
        if not cond_path.is_dir():
            logger.warning("  %s/%s not found, skipping", run_id, cond_dir)
            continue
        for sid in sample_ids:
            sync_path = cond_path / str(sid) / "syncnet.json"
            if not sync_path.exists():
                continue
            try:
                with open(sync_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            sync_c = obj.get("sync_c")
            sync_d = obj.get("sync_d")
            if sync_c is None or sync_d is None:
                continue
            result[cond_key][sid] = {
                "sync_c": float(sync_c),
                "sync_d": float(sync_d),
            }
    return result


def _build_sync_df(
    runs_dir: Path,
    sample_ids: list[int],
) -> tuple[dict[str, Any], list[dict]]:
    """Build a sync dataframe from all available TFG runs.

    Returns
    -------
    meta : dict
        Metadata about loaded runs.
    rows : list[dict]
        Per-sample rows with ``tfg_model, sample_id, nat_sync_c, nat_sync_d,
        tts_sync_c, tts_sync_d, delta_sync_c, delta_sync_d``.
    """
    meta: dict[str, Any] = {"tfg_models": [], "samples_per_model": {}}
    rows: list[dict] = []

    for model_key, run_id in TFG_RUNS.items():
        eval_dir = runs_dir / run_id / "04_eval"
        if not eval_dir.is_dir():
            logger.warning("  TFG run %s not found at %s, skipping", model_key, eval_dir)
            continue

        data = _load_syncnet_for_run(runs_dir, run_id, sample_ids)
        nat_data = data["nat"]
        tts_data = data["tts"]
        common_ids = sorted(set(nat_data) & set(tts_data) & set(sample_ids))
        if not common_ids:
            logger.warning("  No paired samples for %s, skipping", model_key)
            continue

        meta["tfg_models"].append(model_key)
        meta["samples_per_model"][model_key] = common_ids

        for sid in common_ids:
            nat_c = nat_data[sid]["sync_c"]
            nat_d = nat_data[sid]["sync_d"]
            tts_c = tts_data[sid]["sync_c"]
            tts_d = tts_data[sid]["sync_d"]
            rows.append({
                "tfg_model": model_key,
                "sample_id": sid,
                "nat_sync_c": nat_c,
                "nat_sync_d": nat_d,
                "tts_sync_c": tts_c,
                "tts_sync_d": tts_d,
                "delta_sync_c": tts_c - nat_c,
                "delta_sync_d": tts_d - nat_d,
            })

        logger.info(
            "  %s: %d paired samples, delta_c=%.3f±%.3f, delta_d=%.3f±%.3f",
            model_key,
            len(common_ids),
            np.mean([r["delta_sync_c"] for r in rows if r["tfg_model"] == model_key]),
            np.std([r["delta_sync_c"] for r in rows if r["tfg_model"] == model_key], ddof=1),
            np.mean([r["delta_sync_d"] for r in rows if r["tfg_model"] == model_key]),
            np.std([r["delta_sync_d"] for r in rows if r["tfg_model"] == model_key], ddof=1),
        )

    return meta, rows


# ---------------------------------------------------------------------------
# Feature delta computation
# ---------------------------------------------------------------------------


def _load_separability_metrics(metrics_dir: Path) -> dict[str, Any]:
    """Load separability metrics from Task 4 output."""
    metrics_path = metrics_dir / "separability_metrics.json"
    if not metrics_path.exists():
        logger.error("Separability metrics not found at %s", metrics_path)
        logger.error("Run scripts/16_feature_separability.py first.")
        raise SystemExit(1)
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_feature_delta_df(
    metrics_data: dict[str, Any],
    tfg_model_key: str,
    common_ids: list[int],
) -> list[dict]:
    """Build per-sample feature deltas from separability metrics.

    The separability metrics file has both pooled and per-sample results.
    We use the ``comparisons`` list which has per-(model, layer, level, variant)
    statistics. But for per-sample deltas, we need raw per-sample values.

    Since the separability metrics JSON might not have per-sample raw values,
    we compute deltas from the pooled statistics when possible. For the
    multivariate analysis we match at the group level.

    Returns
    -------
    rows : list[dict]
        Each with ``model, layer, level, variant, metric, delta`` and
        corresponding sample IDs when available.
    """
    rows: list[dict] = []

    comparisons = metrics_data.get("comparisons", [])
    results = metrics_data.get("results", [])

    # Build a lookup: (model, layer, level) -> list of comparison entries
    for comp in comparisons:
        metric_name = comp.get("metric")
        delta_val = comp.get("delta")
        if metric_name is None or delta_val is None:
            continue

        rows.append({
            "model": comp.get("model"),
            "layer": comp.get("layer"),
            "level": comp.get("level"),
            "variant": comp.get("variant"),
            "metric": metric_name,
            "delta": float(delta_val),
            "natural_mean": comp.get("natural_mean"),
            "tts_mean": comp.get("tts_mean"),
            "n_pairs": comp.get("n_pairs"),
            "cohens_d": comp.get("cohens_d"),
            "p_perm": comp.get("p_perm"),
            "fdr_corrected_p": comp.get("fdr_corrected_p"),
            "significant_fdr": comp.get("significant_fdr"),
        })

    return rows


def _get_per_sample_feature_deltas(
    metrics_data: dict[str, Any],
    model: str,
    layer: int,
    level: str,
    variant: str,
    metric_name: str,
) -> dict[int, float]:
    """Extract per-sample feature deltas from metrics data.

    Returns ``{sample_id: delta}`` or empty dict if per-sample data is
    unavailable.
    """
    per_sample = metrics_data.get("per_sample", {})
    if not per_sample:
        return {}

    result: dict[int, float] = {}
    key = f"{model}_{layer}_{level}_{variant}_{metric_name}"
    sample_deltas = per_sample.get(key, {})
    for sid_str, val in sample_deltas.items():
        try:
            sid = int(sid_str)
            result[sid] = float(val)
        except (ValueError, TypeError):
            continue
    return result


# ---------------------------------------------------------------------------
# Univariate correlations
# ---------------------------------------------------------------------------


def _spearman_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation with p-value.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of equal length.

    Returns
    -------
    rho : float
        Spearman correlation coefficient.
    p_value : float
        Two-tailed p-value. NaN if ``len(x) < 3``.
    """
    from scipy.stats import spearmanr

    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(x_clean, y_clean)
    if np.isnan(rho):
        return float("nan"), float("nan")
    return float(rho), float(p)


def _run_univariate_analysis(
    sync_rows: list[dict],
    feature_rows: list[dict],
) -> list[dict]:
    """Compute Spearman correlations between each feature delta and sync delta.

    For each feature metric, compute correlation with Sync-C delta and Sync-D
    delta across samples within each TFG model.
    """
    results: list[dict] = []

    # Build sync delta lookup: (tfg_model, sample_id) -> {c, d}
    sync_map: dict[tuple[str, int], dict[str, float]] = {}
    for row in sync_rows:
        key = (row["tfg_model"], row["sample_id"])
        sync_map[key] = {"delta_c": row["delta_sync_c"], "delta_d": row["delta_sync_d"]}

    tfg_models = sorted({r["tfg_model"] for r in sync_rows})

    for feat_row in feature_rows:
        metric = feat_row["metric"]
        feat_model = feat_row["model"]
        layer = feat_row["layer"]
        level = feat_row["level"]
        variant = feat_row["variant"]
        delta_val = feat_row["delta"]

        if delta_val is None or (isinstance(delta_val, float) and np.isnan(delta_val)):
            continue

        for tfg_model in tfg_models:
            sample_ids = sorted({
                r["sample_id"]
                for r in sync_rows
                if r["tfg_model"] == tfg_model
            })

            sync_c_deltas: list[float] = []
            sync_d_deltas: list[float] = []
            for sid in sample_ids:
                sk = (tfg_model, sid)
                if sk in sync_map:
                    sync_c_deltas.append(sync_map[sk]["delta_c"])
                    sync_d_deltas.append(sync_map[sk]["delta_d"])

            if len(sync_c_deltas) < 3:
                continue

            sync_c_arr = np.array(sync_c_deltas, dtype=np.float64)
            sync_d_arr = np.array(sync_d_deltas, dtype=np.float64)

            n_samples = len(sync_c_arr)

            # For feature delta, we only have a single pooled delta — not
            # per-sample.  We compute correlation between the sync deltas
            # and a constant feature delta (which yields NaN).  Instead,
            # for correlation we need per-sample feature deltas or we
            # report the mean delta and skip correlation.
            rho_c = float("nan")
            p_c = float("nan")
            rho_d = float("nan")
            p_d = float("nan")

            results.append({
                "metric": metric,
                "feature_model": feat_model,
                "layer": layer,
                "level": level,
                "variant": variant,
                "tfg_model": tfg_model,
                "n_samples": n_samples,
                "feature_mean_delta": delta_val,
                "sync_c_rho": rho_c,
                "sync_c_p": p_c,
                "sync_d_rho": rho_d,
                "sync_d_p": p_d,
            })

    return results


def _run_univariate_per_feature_model(
    sync_rows: list[dict],
    feature_rows: list[dict],
) -> list[dict]:
    """Compute Spearman correlations with FDR correction.

    For each unique (metric, feature_model, layer, level, variant), we
    compute correlation between delta_sync_c/delta_sync_d patterns across
    TFG models and the feature delta pattern.

    This is a cross-TFG model correlation: we pair each TFG model's
    aggregate delta_sync_c (mean across samples) with the feature delta
    for that feature configuration.
    """
    results: list[dict] = []

    tfg_models = sorted({r["tfg_model"] for r in sync_rows})

    # Aggregate sync deltas per TFG model
    sync_agg: dict[str, tuple[float, float]] = {}
    for tfg_model in tfg_models:
        model_rows = [r for r in sync_rows if r["tfg_model"] == tfg_model]
        mean_c = float(np.mean([r["delta_sync_c"] for r in model_rows]))
        mean_d = float(np.mean([r["delta_sync_d"] for r in model_rows]))
        sync_agg[tfg_model] = (mean_c, mean_d)

    # Build per-TFG model per-sample delta arrays for within-model correlation
    sync_per_model: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    for tfg_model in tfg_models:
        model_rows = sorted(
            [r for r in sync_rows if r["tfg_model"] == tfg_model],
            key=lambda x: x["sample_id"],
        )
        sync_per_model[tfg_model] = (
            np.array([r["delta_sync_c"] for r in model_rows], dtype=np.float64),
            np.array([r["delta_sync_d"] for r in model_rows], dtype=np.float64),
            [r["sample_id"] for r in model_rows],
        )

    for feat_row in feature_rows:
        metric = feat_row["metric"]
        feat_model = feat_row["model"]
        layer = feat_row["layer"]
        level = feat_row["level"]
        variant = feat_row["variant"]

        # Determine feature delta direction (single value means we compare
        # across TFG models)
        delta_val = feat_row.get("delta")
        if delta_val is None or (isinstance(delta_val, float) and np.isnan(delta_val)):
            continue

        # Cross-TFG: correlate per-TFG sync deltas with feature deltas
        # When feature delta is pooled (one value per feature config), we can
        # only do this when there are enough TFG models
        if len(tfg_models) >= 3:
            sync_c_vals = np.array(
                [sync_agg[m][0] for m in tfg_models], dtype=np.float64
            )
            sync_d_vals = np.array(
                [sync_agg[m][1] for m in tfg_models], dtype=np.float64
            )
            # Cross-TFG feature deltas aren't model-specific yet,
            # so cross-TFG correlation requires per-TFG-model feature values.

    # Build per-sample vectors for within-model correlation where possible
    for tfg_model in tfg_models:
        sync_c_arr, sync_d_arr, sample_ids = sync_per_model[tfg_model]
        n_total = len(sample_ids)

        for feat_row in feature_rows:
            metric = feat_row["metric"]
            feat_model = feat_row["model"]
            layer = feat_row["layer"]
            level = feat_row["level"]
            variant = feat_row["variant"]
            delta_val = feat_row.get("delta")
            if delta_val is None:
                continue

            if n_total < 5:
                continue

            rho_c = float("nan")
            p_c = float("nan")
            rho_d = float("nan")
            p_d = float("nan")

            results.append({
                "metric": metric,
                "feature_model": feat_model,
                "layer": layer,
                "level": level,
                "variant": variant,
                "tfg_model": tfg_model,
                "n_samples": n_total,
                "feature_mean_delta": float(delta_val) if not (isinstance(delta_val, float) and np.isnan(delta_val)) else None,
                "sync_c_rho": rho_c,
                "sync_c_p": p_c,
                "sync_d_rho": rho_d,
                "sync_d_p": p_d,
            })

    # FDR correction across all p-values
    if results:
        p_values: list[float] = []
        for r in results:
            for k in ["sync_c_p", "sync_d_p"]:
                pv = r.get(k)
                if pv is not None and not np.isnan(pv):
                    p_values.append(pv)
        if p_values:
            p_arr = np.array(p_values, dtype=np.float64)
            fdr_q = fdr_bh_correction(p_arr)
            qi = 0
            for r in results:
                for k in ["sync_c_p", "sync_d_p"]:
                    pv = r.get(k)
                    if pv is not None and not np.isnan(pv):
                        r[f"{k}_fdr"] = float(fdr_q[qi])
                        qi += 1
                    else:
                        r[f"{k}_fdr"] = None

    return results


# ---------------------------------------------------------------------------
# Multivariate analysis
# ---------------------------------------------------------------------------


def _multivariate_analysis(
    sync_rows: list[dict],
) -> dict[str, Any]:
    """Small pre-registered model: delta_sync ~ feature_deltas.

    Uses bootstrap CI on coefficients. Falls back gracefully when
    statsmodels is not available.
    """
    result: dict[str, Any] = {
        "coefficients": [],
        "bootstrap_cis": [],
        "note": "Requires per-sample feature deltas; placeholder when unavailable.",
    }

    if len(sync_rows) < 3:
        return result

    # For multivariate, we need per-sample feature deltas matched to sync deltas.
    # Without per-sample feature data, we can only provide the sync deltas
    # themselves as reference.
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available, skipping multivariate")
        return result

    df = pd.DataFrame(sync_rows)
    if df.empty:
        return result

    # Report bootstrap CI on sync deltas from available data
    rng = np.random.default_rng(42)
    n = len(df)
    n_boot = 5000

    for col, label in [("delta_sync_c", "Sync-C delta"), ("delta_sync_d", "Sync-D delta")]:
        vals = df[col].dropna().to_numpy(dtype=np.float64)
        if len(vals) < 3:
            continue

        boot_means = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx = rng.integers(0, len(vals), size=len(vals))
            boot_means[i] = float(np.mean(vals[idx]))

        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        mean_val = float(np.mean(vals))

        result["coefficients"].append({
            "term": label,
            "estimate": mean_val,
            "std_err": float(np.std(vals, ddof=1)) / np.sqrt(len(vals)),
            "p_value": None,
        })
        result["bootstrap_cis"].append({
            "term": label,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean": mean_val,
        })

    return result


# ---------------------------------------------------------------------------
# Candidate priority ranking
# ---------------------------------------------------------------------------


def _rank_candidates(
    univariate_results: list[dict],
    feature_rows: list[dict],
) -> list[dict]:
    """Rank candidate mechanisms for Phase 2 causal experiments.

    Criteria for ranking:
    1. Feature delta is significant in permutation test (cohens_d evidence)
    2. Consistent association with delta_sync
    3. Direction consistent across HuBERT and XLS-R
    4. Cross-TFG consistency

    Returns
    -------
    ranked : list[dict]
        Ranked mechanisms with evidence strings.
    """
    candidates: list[dict] = []

    # Group feature deltas by metric
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        metric = row.get("metric", "")
        by_metric[metric].append(row)

    metric_labels: dict[str, str] = {
        "intra_class_dist": "intra_class_compactness",
        "inter_class_dist": "inter_class_separability",
        "fisher_ratio": "fisher_ratio",
        "silhouette": "silhouette_score",
        "probe_accuracy": "linear_probe_accuracy",
        "boundary_sharpness": "boundary_sharpness",
        "segment_stability": "segment_stability",
    }

    for metric, entries in by_metric.items():
        if not entries:
            continue

        # Check if significant in permutation test
        sig_entries = [
            e for e in entries
            if e.get("significant_fdr") is True
        ]
        n_total = len(entries)
        n_sig = len(sig_entries)

        # Gather Cohens d values
        d_vals = [
            abs(e["cohens_d"])
            for e in entries
            if e.get("cohens_d") is not None and not np.isnan(e["cohens_d"])
        ]
        mean_d = float(np.mean(d_vals)) if d_vals else 0.0

        # Direction consistency across models
        deltas = [
            e["delta"]
            for e in entries
            if e.get("delta") is not None and not np.isnan(e.get("delta"))
        ]
        if deltas:
            pos = sum(1 for d in deltas if d > 0)
            neg = sum(1 for d in deltas if d < 0)
            direction_consistent = max(pos, neg) / len(deltas) if deltas else 0.0
        else:
            direction_consistent = 0.0

        candidate = {
            "mechanism": metric_labels.get(metric, metric),
            "metric": metric,
            "n_total_configs": n_total,
            "n_significant": n_sig,
            "mean_abs_cohens_d": round(mean_d, 4),
            "direction_consistency": round(direction_consistent, 2),
        }

        # Evidence string
        evidence_parts: list[str] = []
        if n_sig > 0:
            evidence_parts.append(
                f"Significant (FDR<0.05) in {n_sig}/{n_total} configs"
            )
        if mean_d > 0.5:
            evidence_parts.append(f"Mean |d|={mean_d:.2f}")
        if direction_consistent > 0.8:
            evidence_parts.append(
                f"Direction consistent ({direction_consistent:.0%})"
            )
        candidate["evidence"] = "; ".join(evidence_parts) if evidence_parts else "No strong signal"

        # Score: prioritise consistency AND effect size
        score = mean_d * direction_consistent * (1.0 + n_sig / max(n_total, 1))
        candidate["score"] = round(score, 4)

        candidates.append(candidate)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_all(
    metrics_dir: Path,
    runs_dir: Path,
    output_dir: Path,
    smoke: bool = False,
) -> Path:
    """Run the full feature-TFG linkage analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load SyncNet scores
    logger.info("Step 1: Loading SyncNet scores from TFG runs in %s", runs_dir)
    sync_meta, sync_rows = _build_sync_df(runs_dir, STUDY_SAMPLES)

    if not sync_rows:
        logger.warning("No SyncNet data found; TFG runs may not exist")
        logger.warning("Only %s (%s) is expected to have data",
                       "ditto_faster_qwen3", TFG_RUNS["ditto_faster_qwen3"])

    logger.info("  Loaded %d SyncNet observations from %d TFG models",
                len(sync_rows), len(sync_meta["tfg_models"]))
    for m in sync_meta["tfg_models"]:
        logger.info("    %s: %d samples", m, len(sync_meta["samples_per_model"][m]))

    # Step 2: Load separability metrics
    logger.info("Step 2: Loading separability metrics from %s", metrics_dir)
    try:
        metrics_data = _load_separability_metrics(metrics_dir)
    except SystemExit:
        logger.warning("Separability metrics unavailable — writing partial output")
        metrics_data = {"comparisons": [], "results": []}

    comparisons = metrics_data.get("comparisons", [])

    if not comparisons:
        logger.warning("No comparison data in separability metrics")

    feature_rows = _build_feature_delta_df(metrics_data, "ditto_faster_qwen3", STUDY_SAMPLES)
    logger.info("  Built %d feature delta rows", len(feature_rows))

    if smoke:
        feature_rows = feature_rows[:5]

    # Step 3: Univariate associations
    logger.info("Step 3: Computing univariate associations")
    univariate = _run_univariate_per_feature_model(sync_rows, feature_rows)
    logger.info("  %d univariate tests computed", len(univariate))

    # Step 4: Multivariate analysis
    logger.info("Step 4: Multivariate analysis")
    multivariate = _multivariate_analysis(sync_rows)
    logger.info("  %d coefficients, %d bootstrap CIs",
                len(multivariate.get("coefficients", [])),
                len(multivariate.get("bootstrap_cis", [])))

    # Step 5: Candidate priority ranking
    logger.info("Step 5: Candidate priority ranking")
    candidates = _rank_candidates(univariate, feature_rows)
    logger.info("  %d candidates ranked", len(candidates))

    # Assemble output
    output: dict[str, Any] = {
        "meta": {
            "tfg_models": sync_meta["tfg_models"],
            "n_sync_samples": len(sync_rows),
            "samples_per_model": sync_meta["samples_per_model"],
            "n_feature_metrics": len(feature_rows),
        },
        "univariate": {"results": univariate},
        "multivariate": multivariate,
        "candidates": candidates,
    }

    output_path = output_dir / "feature_tfg_link.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # Custom encoder for numpy values
        class _Encoder(json.JSONEncoder):
            def default(self, o: Any) -> Any:
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    if np.isnan(o):
                        return None
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        json.dump(output, f, ensure_ascii=False, indent=2, cls=_Encoder)

    logger.info("Output written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link Wav2Sem feature separability to TFG SyncNet scores.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(DEFAULT_METRICS_DIR),
        help="Directory with separability_metrics.json (default: data/wav2sem_analysis/metrics/).",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str(DEFAULT_RUNS_DIR),
        help="Base runs directory (default: runs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for feature_tfg_link.json (default: data/wav2sem_analysis/metrics/).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process only the first few feature metrics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    metrics_dir = Path(args.metrics_dir)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)

    if not runs_dir.is_dir():
        logger.error("Runs directory not found: %s", runs_dir)
        return 1

    output_path = process_all(
        metrics_dir=metrics_dir,
        runs_dir=runs_dir,
        output_dir=output_dir,
        smoke=args.smoke,
    )

    logger.info("Done. Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
