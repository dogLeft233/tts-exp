#!/usr/bin/env python3
"""07_emo_analysis.py — Statistical analysis of IndexTTS2 emotion × TFG experiment.

Phase 2: One-way repeated-measures ANOVA (9 conditions: baseline + 8 emotions)
Phase 3: Two-way repeated-measures ANOVA (emotion × alpha)
Baseline vs GSV: Paired t-test, 13 samples
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

# ============================================================
# Load data
# ============================================================
def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("nat_sync_c"):  # skip missing
                rows.append(r)
    return rows

p2 = load_csv("/tmp/p2_emotion_sweep.csv")
p3 = load_csv("/tmp/p3_intensity_sweep.csv")
bg = load_csv("/tmp/baseline_gsv.csv")

# ============================================================
# Phase 2: One-way RM-ANOVA (9 conditions)
# ============================================================
print("=" * 70)
print("PHASE 2: Emotion Sweep — Repeated-Measures ANOVA")
print("=" * 70)

# Build subject × condition matrix for Sync-C
# Natural always uses the same 5 samples, so we use TTS scores
emos = ["happy","angry","sad","afraid","disgusted","melancholic","surprised","calm"]
p2_ids = [1,4,7,10,13]

# Also include baseline (only matching samples)
baseline_by_id = {}
for r in bg:
    sid = int(r["sample"])
    if sid in p2_ids and r["condition"] == "r3_indextts2_baseline":
        baseline_by_id[sid] = {
            "nat_c": float(r["nat_sync_c"]),
            "nat_d": float(r["nat_sync_d"]),
            "tts_c": float(r["tts_sync_c"]),
            "tts_d": float(r["tts_sync_d"]),
        }

# Build matrix: rows=samples, cols=conditions
cond_scores_c = defaultdict(dict)
cond_scores_d = defaultdict(dict)

# Baseline
for sid in p2_ids:
    if sid in baseline_by_id:
        cond_scores_c["baseline"][sid] = baseline_by_id[sid]["tts_c"]
        cond_scores_d["baseline"][sid] = baseline_by_id[sid]["tts_d"]

for r in p2:
    emo = r["emotion"]
    sid = int(r["sample"])
    cond_scores_c[emo][sid] = float(r["tts_sync_c"])
    cond_scores_d[emo][sid] = float(r["tts_sync_d"])

# Also get natural scores
nat_c = {}
nat_d = {}
for r in p2:
    sid = int(r["sample"])
    nat_c[sid] = float(r["nat_sync_c"])
    nat_d[sid] = float(r["nat_sync_d"])

conditions_p2 = ["baseline"] + emos
print(f"\nNatural Sync-C per sample: { {k: round(v,3) for k,v in nat_c.items()} }")
print(f"Natural Sync-D per sample: { {k: round(v,3) for k,v in nat_d.items()} }")
print()

# RM-ANOVA for Sync-C
print("### Sync-C (↑ better) ###")
print(f"{'Condition':<14} {'Mean':>7} {'SD':>7} {'vs Baseline Δ':>14} {'TTS>Nat?':>10}")
print("-" * 54)

baseline_c_mean = np.mean(list(cond_scores_c["baseline"].values()))
for cond in conditions_p2:
    vals = [cond_scores_c[cond][sid] for sid in p2_ids]
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1)
    delta = mean - baseline_c_mean
    tts_better = "ALL" if all(cond_scores_c[cond][sid] > nat_c[sid] for sid in p2_ids) else f"{sum(1 for sid in p2_ids if cond_scores_c[cond][sid] > nat_c[sid])}/5"
    print(f"{cond:<14} {mean:7.3f} {sd:7.3f} {delta:+14.3f} {tts_better:>10}")

# RM-ANOVA test (if scipy available)
try:
    # Build matrix for RM-ANOVA: rows=samples, cols=conditions
    matrix_c = []
    for sid in p2_ids:
        row = [cond_scores_c[cond][sid] for cond in conditions_p2]
        matrix_c.append(row)
    matrix_c = np.array(matrix_c)
    
    f_stat, p_val = stats.f_oneway(*[matrix_c[:,i] for i in range(len(conditions_p2))])
    print(f"\nOne-way ANOVA (between conditions): F={f_stat:.3f}, p={p_val:.6f}")
    
    # Pairwise t-tests vs baseline
    print("\nPairwise vs baseline (paired t-test):")
    baseline_vals = [cond_scores_c["baseline"][sid] for sid in p2_ids]
    for cond in emos:
        t_vals = [cond_scores_c[cond][sid] for sid in p2_ids]
        t_stat, p = stats.ttest_rel(t_vals, baseline_vals)
        d = np.mean(np.array(t_vals) - np.array(baseline_vals)) / np.std(np.array(t_vals) - np.array(baseline_vals), ddof=1)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {cond:<14} t={t_stat:+.3f} p={p:.4f} d={d:+.3f} {sig}")
except Exception as e:
    print(f"\n  ANOVA error: {e}")

print()
print("### Sync-D (↓ better) ###")
print(f"{'Condition':<14} {'Mean':>7} {'SD':>7} {'vs Baseline Δ':>14} {'TTS<Nat?':>10}")
print("-" * 54)

baseline_d_mean = np.mean(list(cond_scores_d["baseline"].values()))
for cond in conditions_p2:
    vals = [cond_scores_d[cond][sid] for sid in p2_ids]
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1)
    delta = mean - baseline_d_mean
    tts_better = "ALL" if all(cond_scores_d[cond][sid] < nat_d[sid] for sid in p2_ids) else f"{sum(1 for sid in p2_ids if cond_scores_d[cond][sid] < nat_d[sid])}/5"
    print(f"{cond:<14} {mean:7.3f} {sd:7.3f} {delta:+14.3f} {tts_better:>10}")

# ============================================================
# Phase 3: Two-way RM-ANOVA (emotion × alpha)
# ============================================================
print()
print("=" * 70)
print("PHASE 3: Intensity Dose-Response")
print("=" * 70)

p3_ids = [1, 7, 13]
p3_data_c = defaultdict(dict)
p3_data_d = defaultdict(dict)

for r in p3:
    emo = r["emotion"]
    alpha = r["alpha"]
    sid = int(r["sample"])
    key = f"{emo}_α{alpha}"
    p3_data_c[key][sid] = float(r["tts_sync_c"])
    p3_data_d[key][sid] = float(r["tts_sync_d"])

print("\n### Sync-C ###")
print(f"{'Condition':<18} {'Mean':>7} {'SD':>7}")
print("-" * 30)
for emo in ["calm", "happy"]:
    for alpha in ["0.3", "0.6", "1.0"]:
        key = f"{emo}_α{alpha}"
        vals = [p3_data_c[key][sid] for sid in p3_ids]
        print(f"{key:<18} {np.mean(vals):7.3f} {np.std(vals,ddof=1):7.3f}")

# Trend analysis
print("\n### Dose-Response Trend (Sync-C) ###")
for emo in ["calm", "happy"]:
    a03 = np.mean([p3_data_c[f"{emo}_α0.3"][sid] for sid in p3_ids])
    a06 = np.mean([p3_data_c[f"{emo}_α0.6"][sid] for sid in p3_ids])
    a10 = np.mean([p3_data_c[f"{emo}_α1.0"][sid] for sid in p3_ids])
    trend = "↗ monotonic" if a10 > a06 > a03 else "↘ monotonic" if a10 < a06 < a03 else "∩ peak at α=0.6" if a06 > a10 and a06 > a03 else "∪ valley at α=0.6"
    print(f"  {emo:6s}: α=0.3→{a03:.3f}  α=0.6→{a06:.3f}  α=1.0→{a10:.3f}  {trend}")

# ============================================================
# Baseline vs GSV
# ============================================================
print()
print("=" * 70)
print("BASELINE COMPARISON: IndexTTS2 vs GSV-TTS")
print("=" * 70)

idx_c = []
idx_d = []
gsv_c = []
gsv_d = []
for r in bg:
    sid = int(r["sample"])
    if r["condition"] == "r3_indextts2_baseline":
        idx_c.append(float(r["tts_sync_c"]))
        idx_d.append(float(r["tts_sync_d"]))
    else:
        gsv_c.append(float(r["tts_sync_c"]))
        gsv_d.append(float(r["tts_sync_d"]))

for metric, idx_vals, gsv_vals, direction in [
    ("Sync-C (↑)", idx_c, gsv_c, "higher"),
    ("Sync-D (↓)", idx_d, gsv_d, "lower")
]:
    t_stat, p = stats.ttest_rel(idx_vals, gsv_vals)
    idx_mean = np.mean(idx_vals)
    gsv_mean = np.mean(gsv_vals)
    d = (idx_mean - gsv_mean) / np.sqrt((np.std(idx_vals,ddof=1)**2 + np.std(gsv_vals,ddof=1)**2)/2)
    winner = "IndexTTS2" if (direction == "higher" and idx_mean > gsv_mean) or (direction == "lower" and idx_mean < gsv_mean) else "GSV-TTS"
    print(f"  {metric}: IndexTTS2={idx_mean:.3f}  GSV-TTS={gsv_mean:.3f}")
    print(f"    t({len(idx_vals)-1})={t_stat:+.3f}, p={p:.4f}, d={d:+.3f}  → {winner} wins")

# ============================================================
# Final ranking
# ============================================================
print()
print("=" * 70)
print("FINAL RANKING — TTS Sync-C Improvement over Natural")
print("=" * 70)

all_conds = []
# P2
for cond in conditions_p2:
    vals = [cond_scores_c[cond][sid] - nat_c[sid] for sid in p2_ids]
    all_conds.append((cond, np.mean(vals), len(p2_ids), "P2"))
# P3
for emo in ["calm", "happy"]:
    for alpha in ["0.3", "0.6", "1.0"]:
        key = f"{emo}_α{alpha}"
        p3_nat_c = {1: nat_c[1], 7: nat_c[7], 13: nat_c[13]}
        vals = [p3_data_c[key][sid] - p3_nat_c[sid] for sid in p3_ids]
        all_conds.append((key, np.mean(vals), len(p3_ids), "P3"))
# Baseline
idx_nat_c = {}
for r in bg:
    if r["condition"] == "r3_indextts2_baseline":
        idx_nat_c[int(r["sample"])] = float(r["nat_sync_c"])
idx_baseline_delta = np.mean([float(r["tts_sync_c"]) - idx_nat_c[int(r["sample"])] for r in bg if r["condition"] == "r3_indextts2_baseline"])
all_conds.append(("indextts2_baseline", idx_baseline_delta, 13, "BASELINE"))
# GSV
gsv_baseline_delta = np.mean([float(r["tts_sync_c"]) - idx_nat_c[int(r["sample"])] for r in bg if r["condition"] == "r3_gsv_tts"])
all_conds.append(("gsv_tts", gsv_baseline_delta, 13, "BASELINE"))

all_conds.sort(key=lambda x: x[1], reverse=True)
for rank, (name, delta, n, phase) in enumerate(all_conds, 1):
    print(f"  {rank:2d}. {name:<25s} ΔC={delta:+.3f}  (n={n}, {phase})")
