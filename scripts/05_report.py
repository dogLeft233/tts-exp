#!/usr/bin/env python3
"""05_report.py - Aggregate SyncNet results into report + figures (issue #6).

Reads 04_eval/*/syncnet.json files for the 4-condition 2×2 experiment.
Generates:
  summary.csv       - one row per successful sample
  report.md         - method, results, replication judgement
  figures/           - paired bar, scatter, Bland-Altman charts
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------


def find_syncnet_results(eval_dir: Path) -> list[dict]:
    """Scan 04_eval for syncnet.json files, return flat list of dicts."""
    rows = []
    for jf in sorted(eval_dir.rglob("syncnet.json")):
        try:
            data = json.loads(jf.read_text())
            if data.get("sync_c") is not None:
                rows.append(data)
        except Exception:
            pass
    return rows


def paired_stats(a: list[float], b: list[float]) -> dict:
    """Compute paired t-test p-value and Cohen's d between two equal-length lists."""
    try:
        from scipy import stats
    except ImportError:
        return {"p_value": None, "cohens_d": None, "note": "scipy not available"}

    import numpy as np

    a_arr, b_arr = np.array(a), np.array(b)
    diff = a_arr - b_arr
    n = len(diff)

    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)

    # Cohen's d (paired)
    d_val = np.mean(diff) / np.std(diff, ddof=1) if n > 1 and np.std(diff, ddof=1) > 0 else 0.0

    return {"p_value": float(p_val), "cohens_d": float(d_val), "n": n, "mean_diff": float(np.mean(diff))}


def build_summary(rows: list[dict]) -> list[dict]:
    """Group rows by condition and sample_id for pairing."""
    grouped: dict[str, dict[int, dict]] = {}
    for r in rows:
        cond = r["condition"]
        sid = r["sample_id"]
        grouped.setdefault(cond, {})[sid] = r
    return [
        {
            "sample_id": sid,
            "natural_raw_c": grouped.get("natural_raw", {}).get(sid, {}).get("sync_c"),
            "natural_raw_d": grouped.get("natural_raw", {}).get(sid, {}).get("sync_d"),
            "natural_resamp_c": grouped.get("natural_resamp", {}).get(sid, {}).get("sync_c"),
            "natural_resamp_d": grouped.get("natural_resamp", {}).get(sid, {}).get("sync_d"),
            "tts_raw_c": grouped.get("tts_raw", {}).get(sid, {}).get("sync_c"),
            "tts_raw_d": grouped.get("tts_raw", {}).get(sid, {}).get("sync_d"),
            "tts_resamp_c": grouped.get("tts_resamp", {}).get(sid, {}).get("sync_c"),
            "tts_resamp_d": grouped.get("tts_resamp", {}).get(sid, {}).get("sync_d"),
        }
        for sid in sorted({r["sample_id"] for r in rows})
    ]


def generate_figures(summary_rows: list[dict], out_dir: Path) -> None:
    """Generate paired bar, scatter, and Bland-Altman charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Extract data for resamp (primary comparison)
    n_resamp_c = [r["natural_resamp_c"] for r in summary_rows if r["natural_resamp_c"] is not None and r["tts_resamp_c"] is not None]
    t_resamp_c = [r["tts_resamp_c"] for r in summary_rows if r["natural_resamp_c"] is not None and r["tts_resamp_c"] is not None]
    n_resamp_d = [r["natural_resamp_d"] for r in summary_rows if r["natural_resamp_d"] is not None and r["tts_resamp_d"] is not None]
    t_resamp_d = [r["tts_resamp_d"] for r in summary_rows if r["natural_resamp_d"] is not None and r["tts_resamp_d"] is not None]

    if not n_resamp_c:
        return

    # --- paired bar chart (Sync-C and Sync-D) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    x = np.arange(len(n_resamp_c))
    w = 0.35

    ax1.bar(x - w / 2, n_resamp_c, w, label="Natural", alpha=0.8)
    ax1.bar(x + w / 2, t_resamp_c, w, label="TTS", alpha=0.8)
    ax1.set_title("Sync-C (↑ better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i + 1) for i in x])
    ax1.legend()

    ax2.bar(x - w / 2, n_resamp_d, w, label="Natural", alpha=0.8)
    ax2.bar(x + w / 2, t_resamp_d, w, label="TTS", alpha=0.8)
    ax2.set_title("Sync-D (↓ better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i + 1) for i in x])
    ax2.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "paired_bar.png", dpi=150)
    plt.close(fig)

    # --- scatter (natural vs tts Sync-C) ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(n_resamp_c, t_resamp_c)
    mn = min(min(n_resamp_c), min(t_resamp_c))
    mx = max(max(n_resamp_c), max(t_resamp_c))
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3)
    ax.set_xlabel("Natural Sync-C")
    ax.set_ylabel("TTS Sync-C")
    ax.set_title("Sync-C: Natural vs TTS")
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter.png", dpi=150)
    plt.close(fig)

    # --- Bland-Altman (Sync-C difference vs mean) ---
    means = [(a + b) / 2 for a, b in zip(n_resamp_c, t_resamp_c)]
    diffs = [a - b for a, b in zip(n_resamp_c, t_resamp_c)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(means, diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs))
    ax.axhline(mean_d, color="r", linestyle="--")
    ax.axhline(mean_d + 1.96 * std_d, color="gray", linestyle=":")
    ax.axhline(mean_d - 1.96 * std_d, color="gray", linestyle=":")
    ax.set_xlabel("Mean Sync-C")
    ax.set_ylabel("Natural - TTS Sync-C")
    ax.set_title("Bland-Altman: Sync-C")
    fig.tight_layout()
    fig.savefig(fig_dir / "bland_altman.png", dpi=150)
    plt.close(fig)


def generate_report(
    summary_rows: list[dict],
    conditions: list[str],
    out_dir: Path,
    mode_used: str = "unknown",
    tts_desc: str = "Qwen3-TTS",
) -> dict:
    """Generate report.md and return stats dict."""
    stats_c = {}
    stats_d = {}
    for cond in conditions:
        c_vals = [r[f"{cond}_c"] for r in summary_rows if r.get(f"{cond}_c") is not None]
        d_vals = [r[f"{cond}_d"] for r in summary_rows if r.get(f"{cond}_d") is not None]
        if c_vals:
            stats_c[cond] = {
                "n": len(c_vals),
                "mean": sum(c_vals) / len(c_vals),
                "min": min(c_vals),
                "max": max(c_vals),
            }
        if d_vals:
            stats_d[cond] = {
                "n": len(d_vals),
                "mean": sum(d_vals) / len(d_vals),
                "min": min(d_vals),
                "max": max(d_vals),
            }

    # Paired stats for resamp (primary comparison)
    n_c = [r["natural_resamp_c"] for r in summary_rows if r["natural_resamp_c"] is not None and r["tts_resamp_c"] is not None]
    t_c = [r["tts_resamp_c"] for r in summary_rows if r["natural_resamp_c"] is not None and r["tts_resamp_c"] is not None]
    n_d = [r["natural_resamp_d"] for r in summary_rows if r["natural_resamp_d"] is not None and r["tts_resamp_d"] is not None]
    t_d = [r["tts_resamp_d"] for r in summary_rows if r["natural_resamp_d"] is not None and r["tts_resamp_d"] is not None]

    paired_c = paired_stats(n_c, t_c) if len(n_c) >= 2 else {"n": len(n_c)}
    paired_d = paired_stats(n_d, t_d) if len(n_d) >= 2 else {"n": len(n_d)}

    # Direction check
    mean_nat_c = stats_c.get("natural_resamp", {}).get("mean", 0)
    mean_tts_c = stats_c.get("tts_resamp", {}).get("mean", 0)
    mean_nat_d = stats_d.get("natural_resamp", {}).get("mean", 0)
    mean_tts_d = stats_d.get("tts_resamp", {}).get("mean", 0)

    # TTS better = Sync-C higher AND Sync-D lower
    sync_c_better = mean_tts_c > mean_nat_c
    sync_d_better = mean_tts_d < mean_nat_d
    direction_ok = sync_c_better and sync_d_better

    report_lines = [
        "# TTS-TFG Enhancement Experiment — Replication Report",
        "",
        f"**Run**: `{out_dir.parent.name}`  ",
        f"**Mode**: {mode_used}  ",
        f"**Conditions**: {', '.join(conditions)}  ",
        "",
        "## Method",
        "",
        f"- **TTS model**: {tts_desc}",
        "- **TFG model**: Ditto (antgroup/ditto-talkinghead)",
        "- **Evaluation**: SyncNet (syncnet_v2.model), Sync-C (confidence) and Sync-D (min distance)",
        "- **Samples**: 10 paired audio-image pairs from AISHELL-1 + HDTF",
        "- **Seed**: torch.manual_seed(42) for both TTS and Ditto",
        "",
        "## Results",
        "",
        "### Sync-C (↑ better)",
        "",
        "| Condition | N | Mean | Min | Max |",
        "|-----------|----|------|-----|-----|",
    ]
    for cond in conditions:
        s = stats_c.get(cond, {})
        report_lines.append(f"| {cond} | {s.get('n', 0)} | {s.get('mean', 0):.3f} | {s.get('min', 0):.3f} | {s.get('max', 0):.3f} |")

    report_lines.extend([
        "",
        "### Sync-D (↓ better)",
        "",
        "| Condition | N | Mean | Min | Max |",
        "|-----------|----|------|-----|-----|",
    ])
    for cond in conditions:
        s = stats_d.get(cond, {})
        report_lines.append(f"| {cond} | {s.get('n', 0)} | {s.get('mean', 0):.3f} | {s.get('min', 0):.3f} | {s.get('max', 0):.3f} |")

    report_lines.extend([
        "",
        "### Paired analysis (natural_resamp vs tts_resamp)",
        "",
        f"- Sync-C paired n={paired_c.get('n', 0)}, p={paired_c.get('p_value', '?'):.4f}, Cohen's d={paired_c.get('cohens_d', '?'):.3f}",
        f"- Sync-D paired n={paired_d.get('n', 0)}, p={paired_d.get('p_value', '?'):.4f}, Cohen's d={paired_d.get('cohens_d', '?'):.3f}",
        "",
        "## Replication Judgement",
        "",
        f"- Sync-C: Natural mean={mean_nat_c:.3f}, TTS mean={mean_tts_c:.3f} → TTS higher = {'YES' if sync_c_better else 'NO'}",
        f"- Sync-D: Natural mean={mean_nat_d:.3f}, TTS mean={mean_tts_d:.3f} → TTS lower = {'YES' if sync_d_better else 'NO'}",
        f"- **Direction**: {'✅ PASS — both metrics agree' if direction_ok else '❌ FAIL — one or both metrics disagree'}",
        "",
        "---",
        "_Generated by scripts/05_report.py_",
    ])

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    return {
        "direction_ok": direction_ok,
        "sync_c_better": sync_c_better,
        "sync_d_better": sync_d_better,
        "paired_c": paired_c,
        "paired_d": paired_d,
        "per_condition_c": stats_c,
        "per_condition_d": stats_d,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate experiment report")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    eval_dir = run_dir / "04_eval"
    out_dir = run_dir / "05_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["natural_raw", "natural_resamp", "tts_raw", "tts_resamp"]
    rows = find_syncnet_results(eval_dir)
    print(f"[report] {len(rows)} syncnet results found")

    if not rows:
        print("[report] no results — writing empty report")
        (out_dir / "report.md").write_text("# No results\n")
        return

    summary_rows = build_summary(rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)

    # Check mode_used from ditto_meta
    ditto_meta = run_dir / "03_ditto" / "ditto_meta.json"
    mode_used = "unknown"
    if ditto_meta.exists():
        mode_used = json.loads(ditto_meta.read_text()).get("mode_used", "unknown")

    # Read TTS backend info for method section
    tts_meta_path = run_dir / "02_tts" / "tts_meta.json"
    tts_desc = "Qwen3-TTS (provider unknown)"
    if tts_meta_path.exists():
        tm = json.loads(tts_meta_path.read_text())
        prov = tm.get("provider", "unknown")
        results = tm.get("results", {})
        if results:
            first = next(iter(results.values()))
            model = first.get("model") or first.get("backend") or prov
            tts_desc = f"{model} via {prov}"
        else:
            tts_desc = prov

    stats = generate_report(summary_rows, conditions, out_dir, mode_used, tts_desc)
    generate_figures(summary_rows, out_dir)

    # Dump full stats json for downstream
    (out_dir / "report_stats.json").write_text(json.dumps(stats, indent=2))
    ok = "PASS" if stats.get("direction_ok") else "FAIL"
    print(f"[report] {ok} — report.md + summary.csv + figures/ done")


if __name__ == "__main__":
    main()
