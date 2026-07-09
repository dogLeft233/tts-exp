#!/usr/bin/env python3
"""06_cross_model.py — Compare TTS providers on common-paired sample subset.

Input: 4 runs' 05_report/summary.csv files
Output: per-provider Δ stats + cross-model ANOVA + report
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Providers & runs
# ---------------------------------------------------------------------------

PROVIDER_RUNS = OrderedDict([
    ("faster_qwen3",  "r2_faster_qwen3_20260707T145233Z"),
    ("dashscope_vc",  "r2_dashscope_vc_20260707T143930Z"),
    ("fish_audio",    "r2_fish_audio_20260707T150723Z"),
    ("siliconflow",   "r2_siliconflow_20260707T142139Z"),
])

# Samples where ALL 4 providers have tts_raw data
COMMON_SUBSET = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

# Friendly names
MODEL_NAMES = {
    "faster_qwen3":  "Qwen3-TTS 0.6B (local)",
    "dashscope_vc":  "Qwen3-TTS-VC (cloud)",
    "fish_audio":    "Fish S2 Pro (cloud)",
    "siliconflow":   "CosyVoice2-0.5B (cloud)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_summary(csv_path: Path) -> dict[int, dict]:
    """Return {sample_id: {natural_raw_c, natural_raw_d, tts_raw_c, tts_raw_d}}."""
    data: dict[int, dict] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sid = int(row["sample_id"])
            raw = {}
            for k in ["natural_raw_c", "natural_raw_d", "tts_raw_c", "tts_raw_d"]:
                val = row.get(k, "").strip()
                raw[k] = float(val) if val else np.nan
            data[sid] = raw
    return data


def paired_stats(arr1: np.ndarray, arr2: np.ndarray) -> dict:
    """Compute paired t-test, Wilcoxon, Cohen's d, mean Δ."""
    from scipy import stats

    diffs = arr1 - arr2  # Δ = tts - natural
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    n = len(diffs)

    # Paired t-test
    t_stat, p_t = stats.ttest_rel(arr1, arr2)

    # Wilcoxon signed-rank
    w_stat, p_w = stats.wilcoxon(arr1, arr2, zero_method="zsplit")

    # Cohen's d (paired: mean_diff / sd_diff)
    cohen_d = float(mean_d / std_d) if std_d > 1e-10 else 0.0

    # 95% CI for mean Δ
    se = std_d / np.sqrt(n)
    ci_lo = float(mean_d - 1.96 * se)
    ci_hi = float(mean_d + 1.96 * se)

    return {
        "mean_delta": round(mean_d, 4),
        "sd_delta": round(std_d, 4),
        "n": n,
        "t_stat": round(float(t_stat), 4),
        "p_paired_t": round(float(p_t), 6),
        "wilcoxon_stat": round(float(w_stat), 4),
        "p_wilcoxon": round(float(p_w), 6),
        "cohens_d": round(cohen_d, 4),
        "ci_95": (round(ci_lo, 4), round(ci_hi, 4)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-model comparison")
    ap.add_argument("--runs_dir", default="runs", help="Base runs directory")
    ap.add_argument("--output", default="", help="Output directory (default: runs/r2_summary_<TS>/)")
    args = ap.parse_args()

    repo = Path(args.runs_dir).resolve().parent if Path(args.runs_dir).name == "runs" else Path(args.runs_dir)
    runs_dir = repo / "runs"

    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output) if args.output else runs_dir / f"r2_summary_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 4 summaries
    provider_data: dict[str, dict[int, dict]] = {}
    for name, run_id in PROVIDER_RUNS.items():
        csv_path = runs_dir / run_id / "05_report" / "summary.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping {name}")
            continue
        data = load_summary(csv_path)
        provider_data[name] = data
        print(f"  Loaded {name}: {len(data)} samples")

    # Filter to common subset
    common_sids = COMMON_SUBSET

    # Verify all providers have complete data for common subset
    for name, data in provider_data.items():
        for sid in common_sids:
            if sid not in data:
                print(f"  ERROR: {name} missing sample {sid}")
                raise SystemExit(1)
            row = data[sid]
            if np.isnan(row["tts_raw_c"]) or np.isnan(row["tts_raw_d"]):
                print(f"  ERROR: {name} sample {sid} missing tts_raw")
                raise SystemExit(1)
            if np.isnan(row["natural_raw_c"]) or np.isnan(row["natural_raw_d"]):
                print(f"  ERROR: {name} sample {sid} missing natural_raw")
                raise SystemExit(1)

    print(f"\n  Common-paired subset: n={len(common_sids)} samples")
    print(f"  IDs: {common_sids}")

    # Build arrays for cross-model comparison
    # For each provider × sample, compute Δ_c and Δ_d
    delta_c_dict: dict[str, list[float]] = {name: [] for name in provider_data}
    delta_d_dict: dict[str, list[float]] = {name: [] for name in provider_data}
    natural_c_per_sample: dict[int, list[float]] = {sid: [] for sid in common_sids}
    natural_d_per_sample: dict[int, list[float]] = {sid: [] for sid in common_sids}

    for name, data in provider_data.items():
        for sid in common_sids:
            row = data[sid]
            d_c = row["tts_raw_c"] - row["natural_raw_c"]
            d_d = row["tts_raw_d"] - row["natural_raw_d"]
            delta_c_dict[name].append(d_c)
            delta_d_dict[name].append(d_d)
            natural_c_per_sample[sid].append(row["natural_raw_c"])
            natural_d_per_sample[sid].append(row["natural_raw_d"])

    # Per-provider stats
    print("\n" + "=" * 80)
    print("=== Per-provider paired statistics (n=12) ===")
    per_provider: dict[str, dict] = {}
    for name in provider_data:
        print(f"\n--- {MODEL_NAMES.get(name, name)} ---")

        tts_c = np.array([provider_data[name][sid]["tts_raw_c"] for sid in common_sids])
        nat_c = np.array([provider_data[name][sid]["natural_raw_c"] for sid in common_sids])
        tts_d = np.array([provider_data[name][sid]["tts_raw_d"] for sid in common_sids])
        nat_d = np.array([provider_data[name][sid]["natural_raw_d"] for sid in common_sids])

        print(f"  natural_raw: C={nat_c.mean():.3f}±{nat_c.std():.3f}  D={nat_d.mean():.3f}±{nat_d.std():.3f}")
        print(f"  tts_raw:     C={tts_c.mean():.3f}±{tts_c.std():.3f}  D={tts_d.mean():.3f}±{tts_d.std():.3f}")

        sc = paired_stats(tts_c, nat_c)
        sd = paired_stats(tts_d, nat_d)

        print(f"  Sync-C Δ: {sc['mean_delta']:+.4f} [{sc['ci_95'][0]:+.4f}, {sc['ci_95'][1]:+.4f}]  d={sc['cohens_d']:.3f}  p_t={sc['p_paired_t']:.4f}  p_w={sc['p_wilcoxon']:.4f}")
        print(f"  Sync-D Δ: {sd['mean_delta']:+.4f} [{sd['ci_95'][0]:+.4f}, {sd['ci_95'][1]:+.4f}]  d={sd['cohens_d']:.3f}  p_t={sd['p_paired_t']:.4f}  p_w={sd['p_wilcoxon']:.4f}")

        per_provider[name] = {"sync_c": sc, "sync_d": sd}

    # Cross-model ANOVA on Δ values
    print("\n" + "=" * 80)
    print("=== Cross-model comparison (ANOVA on Δ_c and Δ_d) ===")

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import mixedlm

        for metric_name, delta_dict in [("Sync-C Δ", delta_c_dict), ("Sync-D Δ", delta_d_dict)]:
            print(f"\n--- {metric_name} ---")
            rows = []
            for name, deltas in delta_dict.items():
                for i, val in enumerate(deltas):
                    rows.append({"model": name, "sample": i, "delta": val})
            import pandas as pd
            df = pd.DataFrame(rows)

            # Mixed linear model: delta ~ model (fixed) with sample as random effect
            model = mixedlm("delta ~ C(model)", df, groups=df["sample"])
            result = model.fit()
            print(result.summary())

            # Also simple repeated-measures one-way
            from scipy import stats
            arrays = [delta_dict[name] for name in provider_data if name in delta_dict]
            f_stat, p_anova = stats.f_oneway(*arrays)
            print(f"  One-way ANOVA (ignoring sample pairing): F={f_stat:.4f}, p={p_anova:.6f}")

            # Pairwise comparisons (paired t-test between providers on Δ values)
            print("\n  Pairwise Δ comparisons (paired t-test):")
            names_list = [n for n in provider_data if n in delta_dict]
            for i in range(len(names_list)):
                for j in range(i + 1, len(names_list)):
                    a, b = names_list[i], names_list[j]
                    t_stat, p_val = stats.ttest_rel(np.array(delta_dict[a]), np.array(delta_dict[b]))
                    print(f"    {a} vs {b}: t={t_stat:.3f}, p={p_val:.4f}")

    except ImportError:
        print("  statsmodels not available — skipping mixed model; using scipy only")
        from scipy import stats
        for metric_name, delta_dict in [("Sync-C Δ", delta_c_dict), ("Sync-D Δ", delta_d_dict)]:
            print(f"\n--- {metric_name} ---")
            arrays = [delta_dict[name] for name in provider_data if name in delta_dict]
            if len(arrays) >= 2:
                f_stat, p_anova = stats.f_oneway(*arrays)
                print(f"  One-way ANOVA: F={f_stat:.4f}, p={p_anova:.6f}")
            names_list = [n for n in provider_data if n in delta_dict]
            print("  Pairwise paired t-tests:")
            for i in range(len(names_list)):
                for j in range(i + 1, len(names_list)):
                    a, b = names_list[i], names_list[j]
                    t_stat, p_val = stats.ttest_rel(np.array(delta_dict[a]), np.array(delta_dict[b]))
                    print(f"    {a} vs {b}: t={t_stat:.3f}, p={p_val:.4f}")

    # Summary table (Markdown)
    print("\n" + "=" * 80)
    print("=== Summary table ===")
    summary_lines = []

    # Table header
    header = "| Provider | Model | Sync-C Δ | p_t | p_w | d | Sync-D Δ | p_t | p_w | d |"
    sep    = "|----------|-------|----------|-----|-----|---|----------|-----|-----|---|"
    summary_lines.append(header)
    summary_lines.append(sep)

    for name in provider_data:
        mname = MODEL_NAMES.get(name, name)
        sc = per_provider[name]["sync_c"]
        sd = per_provider[name]["sync_d"]
        line = (
            f"| {name:<15s} | {mname:<40s} | {sc['mean_delta']:+.4f} | {sc['p_paired_t']:.4f} | {sc['p_wilcoxon']:.4f} | {sc['cohens_d']:+.3f} | "
            f"{sd['mean_delta']:+.4f} | {sd['p_paired_t']:.4f} | {sd['p_wilcoxon']:.4f} | {sd['cohens_d']:+.3f} |"
        )
        summary_lines.append(line)
        print(line)

    # Write report
    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("# R2 Cross-Model Comparison Report\n\n")
        f.write(f"**Common-paired subset**: n={len(common_sids)} samples (IDs: {common_sids})\n\n")
        f.write("All 4 providers: faster_qwen3, dashscope_vc, fish_audio, siliconflow\n\n")
        f.write("## Per-provider statistics (n=12 paired)\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for line in summary_lines[2:]:
            f.write(line + "\n")

        f.write("\n## Detailed per-provider results\n\n")
        for name in provider_data:
            mname = MODEL_NAMES.get(name, name)
            sc = per_provider[name]["sync_c"]
            sd = per_provider[name]["sync_d"]
            f.write(f"### {name} ({mname})\n\n")
            f.write(f"- **Sync-C Δ**: {sc['mean_delta']:+.4f} (95% CI [{sc['ci_95'][0]:+.4f}, {sc['ci_95'][1]:+.4f}]), d={sc['cohens_d']:+.3f}\n")
            f.write(f"  - Paired t-test: t={sc['t_stat']:.4f}, p={sc['p_paired_t']:.6f}\n")
            f.write(f"  - Wilcoxon: W={sc['wilcoxon_stat']:.4f}, p={sc['p_wilcoxon']:.6f}\n")
            f.write(f"- **Sync-D Δ**: {sd['mean_delta']:+.4f} (95% CI [{sd['ci_95'][0]:+.4f}, {sd['ci_95'][1]:+.4f}]), d={sd['cohens_d']:+.3f}\n")
            f.write(f"  - Paired t-test: t={sd['t_stat']:.4f}, p={sd['p_paired_t']:.6f}\n")
            f.write(f"  - Wilcoxon: W={sd['wilcoxon_stat']:.4f}, p={sd['p_wilcoxon']:.6f}\n")
            f.write("\n")

        f.write("\n## Data\n\n")
        f.write("All raw values in individual runs' `05_report/summary.csv`. Common subset IDs:\n")
        f.write(f"`{common_sids}`\n")

    print(f"\n  Report written to: {report_path}")


if __name__ == "__main__":
    main()
