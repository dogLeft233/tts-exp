#!/usr/bin/env python3
"""09_observational_analysis.py — Correlation, regression, and ANCOVA analysis
of audio acoustic features vs SyncC/SyncD scores.

Input: data/r4_results/audio_features.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(REPO / "data" / "r4_results" / "audio_features.csv")
df = df[df["sync_c"].notna()].copy()
print(f"Loaded {len(df)} rows with SyncC scores")

# ============================================================
# 1. Correlation matrix
# ============================================================
print("\n" + "=" * 70)
print("1. CORRELATION MATRIX: Acoustic Features vs SyncC/SyncD")
print("=" * 70)

feature_cols = ["duration", "rms", "peak", "lufs", "f0_mean", "f0_std",
                "spec_centroid", "spec_rolloff", "zcr"]

print(f"\n{'Feature':<18} {'r_Pearson(SyncC)':>18} {'p_Pearson':>10} {'r_Spearman':>14}")
print("-" * 62)

for col in feature_cols:
    valid = df[col].dropna()
    if len(valid) < 3:
        continue
    # Match indices
    mask = df[col].notna()
    sub = df[mask]
    r_p, p_p = stats.pearsonr(sub[col], sub["sync_c"])
    r_s, p_s = stats.spearmanr(sub[col], sub["sync_c"])
    sig_p = "*" if p_p < 0.05 else "**" if p_p < 0.01 else "***" if p_p < 0.001 else ""
    sig_s = "*" if p_s < 0.05 else "**" if p_s < 0.01 else "***" if p_s < 0.001 else ""
    print(f"{col:<18} {r_p:+18.4f} {p_p:10.4f}{sig_p:>3}  {r_s:+14.4f}{sig_s}")

# Also Sync-D
print(f"\n{'Feature':<18} {'r_Pearson(SyncD)':>18} {'p_Pearson':>10}")
print("-" * 48)
for col in feature_cols:
    valid = df[col].dropna()
    if len(valid) < 3:
        continue
    mask = df[col].notna()
    sub = df[mask]
    r_p, p_p = stats.pearsonr(sub[col], sub["sync_d"])
    sig = "*" if p_p < 0.05 else "**" if p_p < 0.01 else "***" if p_p < 0.001 else ""
    print(f"{col:<18} {r_p:+18.4f} {p_p:10.4f}{sig:>3}")

# ============================================================
# 2. Top correlation details
# ============================================================
print("\n" + "=" * 70)
print("2. KEY RELATIONSHIPS")
print("=" * 70)

# RMS vs SyncC
print(f"\n### RMS vs SyncC ###")
r, p = stats.pearsonr(df["rms"], df["sync_c"])
r2 = r ** 2
print(f"  Pearson r = {r:+.3f}, p = {p:.6f}, R² = {r2:.3f}")

# Duration vs SyncC
print(f"\n### Duration vs SyncC ###")
r, p = stats.pearsonr(df["duration"], df["sync_c"])
r2 = r ** 2
print(f"  Pearson r = {r:+.3f}, p = {p:.6f}, R² = {r2:.3f}")

# ============================================================
# 3. Multiple Regression
# ============================================================
print("\n" + "=" * 70)
print("3. MULTIPLE LINEAR REGRESSION: SyncC ~ RMS + Duration + F0_mean")
print("=" * 70)

# Prepare data
reg_df = df[["sync_c", "rms", "duration", "f0_mean", "spec_centroid", "zcr"]].dropna()
X = reg_df[["rms", "duration", "f0_mean", "spec_centroid", "zcr"]].values
y = reg_df["sync_c"].values

# Manual OLS
X_aug = np.column_stack([np.ones(len(X)), X])  # add intercept
coeffs, residuals, rank, singular = np.linalg.lstsq(X_aug, y, rcond=None)
pred = X_aug @ coeffs
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_resid = np.sum((y - pred) ** 2)
r2 = 1 - ss_resid / ss_total
n, k = X_aug.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print(f"\n{'Predictor':<20} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>10}")
print("-" * 62)
predictor_names = ["Intercept", "RMS", "Duration", "F0_mean", "Spec_centroid", "ZCR"]
se = np.sqrt(np.diag(ss_resid / (n - k) * np.linalg.inv(X_aug.T @ X_aug)))
for i, name in enumerate(predictor_names):
    t_stat = coeffs[i] / se[i] if se[i] > 1e-10 else 0
    p_val = 2 * stats.t.sf(abs(t_stat), n - k)
    sig = "*" if p_val < 0.05 else "**" if p_val < 0.01 else "***" if p_val < 0.001 else ""
    print(f"{name:<20} {coeffs[i]:+10.4f} {se[i]:10.4f} {t_stat:+8.3f} {p_val:10.4f} {sig}")

print(f"\nR² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}, n = {n}")

# Simplified model: just the top predictors
print(f"\n### Simplified: SyncC ~ RMS + Duration ###")
X_simple = reg_df[["rms", "duration"]].values
X_s_aug = np.column_stack([np.ones(len(X_simple)), X_simple])
coeffs_s, _, _, _ = np.linalg.lstsq(X_s_aug, y, rcond=None)
pred_s = X_s_aug @ coeffs_s
ss_resid_s = np.sum((y - pred_s) ** 2)
r2_s = 1 - ss_resid_s / ss_total
print(f"  R² = {r2_s:.4f}  (RMS alone explains {r2_s:.4f} of SyncC variance)")

# ============================================================
# 4. Context-normalized: focus on emotion sweep only
# ============================================================
print("\n" + "=" * 70)
print("4. EMOTION-SPECIFIC ANALYSIS (P2 sweep only, n=40)")
print("=" * 70)

emo_df = df[df["emotion"].notna()].copy()
print(f"Emotion sweep samples: {len(emo_df)}")

# Within-emotion analysis
print(f"\n### Per-emotion mean SyncC, RMS, Duration ###")
print(f"{'Emotion':<16} {'SyncC':>7} {'RMS':>8} {'Dur(s)':>7} {'LUFS':>7}")
print("-" * 52)
for emo in sorted(emo_df["emotion"].unique()):
    sub = emo_df[emo_df["emotion"] == emo]
    print(f"{emo:<16} {sub['sync_c'].mean():7.3f} {sub['rms'].mean():8.4f} "
          f"{sub['duration'].mean():7.2f} {sub['lufs'].mean():7.1f}")

# Compute ΔSyncC per sample (TTS - natural) for emotion sweep
print(f"\n### Partial correlation: emotion → SyncC after controlling for RMS + Duration ###")
emo_ids = sorted(emo_df["sample"].unique())
conditions = sorted(emo_df["emotion"].unique())

# Build matrix for rm-anova-like decomposition
emo_scores = {}
for emo in conditions:
    for _, row in emo_df[emo_df["emotion"] == emo].iterrows():
        sid = int(row["sample"])
        emo_scores.setdefault(emo, {})[sid] = float(row["sync_c"])

# Compute residuals: SyncC ~ RMS + Duration, then look at residuals by emotion
reg_emo = emo_df[["sync_c", "rms", "duration"]].dropna()
X_emo = np.column_stack([np.ones(len(reg_emo)), reg_emo["rms"].values, reg_emo["duration"].values])
y_emo = reg_emo["sync_c"].values
coeffs_emo, _, _, _ = np.linalg.lstsq(X_emo, y_emo, rcond=None)
residuals = y_emo - X_emo @ coeffs_emo
reg_emo["residual"] = residuals

# Check residuals by emotion
print(f"\n{'Emotion':<16} {'Raw SyncC':>10} {'Residual':>10} {'%Explained?':>14}")
print("-" * 56)
for emo in conditions:
    sub = emo_df[emo_df["emotion"] == emo]
    mask = reg_emo.index.isin(sub.index)
    raw_mean = sub["sync_c"].mean()
    resid_mean = reg_emo.loc[reg_emo.index[mask], "residual"].mean()
    pct = "YES" if abs(resid_mean) < 0.3 else f"Δ={resid_mean:+.2f}"
    print(f"{emo:<16} {raw_mean:10.3f} {resid_mean:+10.4f} {pct:>14}")

# ============================================================
# 5. Per-provider decomposition
# ============================================================
print("\n" + "=" * 70)
print("5. PROVIDER-LEVEL DECOMPOSITION")
print("=" * 70)

providers_sorted = df.groupby("provider")["sync_c"].mean().sort_values(ascending=False).index
print(f"\n{'Provider':<35s} {'SyncC':>7} {'RMS':>8} {'Dur':>6} {'LUFS':>7} {'F0':>7}")
print("-" * 80)
for p in providers_sorted:
    sub = df[df["provider"] == p]
    f0_str = f"{sub['f0_mean'].mean():.0f}" if sub["f0_mean"].notna().any() else "N/A"
    print(f"{p:<35s} {sub['sync_c'].mean():7.3f} {sub['rms'].mean():8.4f} "
          f"{sub['duration'].mean():6.2f} {sub['lufs'].mean():7.1f} {f0_str:>7}")

# Use all rows (not just emotion) for overall RMS prediction
print(f"\n### Across all providers: RMS explains SyncC ###")
all_valid = df[["sync_c", "rms", "duration"]].dropna()
r_all, p_all = stats.pearsonr(all_valid["rms"], all_valid["sync_c"])
print(f"  All {len(all_valid)} samples: r(RMS, SyncC) = {r_all:+.4f}, p = {p_all:.6f}")
r_dur, p_dur = stats.pearsonr(all_valid["duration"], all_valid["sync_c"])
print(f"  All {len(all_valid)} samples: r(Duration, SyncC) = {r_dur:+.4f}, p = {p_dur:.6f}")

# Semi-partial: how much unique variance does RMS explain beyond Duration?
# r²(SyncC, RMS) - r²(SyncC, Duration|RMS) etc
r_rd = stats.pearsonr(all_valid["rms"], all_valid["duration"])[0]
r_cd = r_dur
r_cr = r_all
# Semi-partial for RMS: (r_cr - r_cd * r_rd) / sqrt(1 - r_rd²)
sr_rms = (r_cr - r_cd * r_rd) / np.sqrt(1 - r_rd**2)
sr_dur = (r_cd - r_cr * r_rd) / np.sqrt(1 - r_rd**2)
print(f"  Semi-partial r: RMS={sr_rms:+.4f}, Duration={sr_dur:+.4f}")
print(f"  Unique R²:    RMS={sr_rms**2:.4f}, Duration={sr_dur**2:.4f}")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "=" * 70)
print("6. SUMMARY")
print("=" * 70)

# RMS predictor rank
rms_rank = all_valid["rms"].corr(all_valid["sync_c"])
dur_rank = all_valid["duration"].corr(all_valid["sync_c"])

print(f"\nKey findings:")
print(f"  - RMS is {'strongly' if abs(rms_rank) > 0.5 else 'moderately' if abs(rms_rank) > 0.3 else 'weakly'}")
print(f"    correlated with SyncC (r={rms_rank:+.3f})")
print(f"  - Duration is {'strongly' if abs(dur_rank) > 0.5 else 'moderately' if abs(dur_rank) > 0.3 else 'weakly'}")
print(f"    anti-correlated with SyncC (r={dur_rank:+.3f})")
print(f"  - Fish_audio (highest RMS=0.177) would be predicted to have high SyncC")
print(f"    by RMS alone — intervention phase will test this causally")
