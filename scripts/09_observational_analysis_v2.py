#!/usr/bin/env python3
"""09_observational_analysis_v2.py — Comprehensive correlation & regression
analysis of 53 audio features vs SyncD/SyncC scores.

Key changes from v1:
- Uses audio_features_v2.csv (477 rows, 53 features)
- Primary metric: SyncD (lower=better), secondary: SyncC
- LASSO regression for feature selection
- Random Forest feature importance
- Correlation heatmap
- PCA visualization
- Per-provider decomposition
- Partial correlation controlling for provider

Output: data/r4_results/observational_analysis_v2.txt
Figures: figures/obs_v2_*.png
"""
import sys, os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR = REPO / "data" / "r4_results"
RES_DIR.mkdir(exist_ok=True)

# ====================================================================
# Config
# ====================================================================
TARGET = "sync_d"  # primary: SyncD (lower is better)
TARGET2 = "sync_c"  # secondary

# Features to include in ML models (exclude column-identity cols)
META_COLS = ["provider", "sample", "run_id", "src_round", "emotion", "alpha",
             "norm_type", "orig_provider", "sync_c", "sync_d", "nat_sync_c",
             "nat_sync_d", "filepath"]

# Features that are likely problematic (high NaN rate)
SKIP_FEATURES = ["lra"]  # 73% NaN

# ====================================================================
# Load
# ====================================================================
df = pd.read_csv(RES_DIR / "audio_features_v2.csv")
scored = df[df["sync_d"].notna()].copy()
scored = scored[scored["provider"] != "natural"]  # natural has no SyncNet
print(f"Scored rows: {len(scored)}")

feat_cols = [c for c in df.columns if c not in META_COLS and c not in SKIP_FEATURES]
# Drop features with >50% NaN
for c in feat_cols[:]:
    nan_rate = scored[c].isna().mean()
    if nan_rate > 0.5:
        print(f"  Dropping {c} ({nan_rate*100:.0f}% NaN)")
        feat_cols.remove(c)

print(f"Feature count for analysis: {len(feat_cols)}")

# Build clean dataset for ML
ml_cols = [TARGET, TARGET2] + feat_cols
ml_df = scored[ml_cols].dropna()
print(f"Clean rows (no NaN in features): {len(ml_df)}")

X = ml_df[feat_cols].values
y_d = ml_df[TARGET].values  # SyncD (lower=better)
y_c = ml_df[TARGET2].values  # SyncC (higher=better)

# Standardize for regression/PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

out_lines = []
def pline(s=""):
    out_lines.append(s)
    print(s)

# ====================================================================
# 1. CORRELATION ANALYSIS
# ====================================================================
pline("=" * 80)
pline("1. CORRELATION: Features vs SyncD (primary) & SyncC (secondary)")
pline("=" * 80)

pline(f"\n{'Feature':<24} {'r_Spear(SyncD)':>16} {'p':>10}  {'r_Pear(SyncD)':>16} {'r_Spear(SyncC)':>16}")
pline("-" * 88)

corr_results = []
for col in feat_cols:
    sub = ml_df[[TARGET, TARGET2, col]].dropna()
    if len(sub) < 10:
        continue
    r_s_d, p_s_d = stats.spearmanr(sub[col], sub[TARGET])
    r_p_d, _ = stats.pearsonr(sub[col], sub[TARGET])
    r_s_c, _ = stats.spearmanr(sub[col], sub[TARGET2])
    corr_results.append({
        "feature": col,
        "r_spear_syncd": r_s_d,
        "p_spear_syncd": p_s_d,
        "r_pear_syncd": r_p_d,
        "r_spear_syncc": r_s_c,
    })

# Sort by abs(Spearman SyncD)
corr_results.sort(key=lambda x: abs(x["r_spear_syncd"]), reverse=True)

for cr in corr_results:
    sig = " ***" if cr["p_spear_syncd"] < 0.001 else " **" if cr["p_spear_syncd"] < 0.01 else " *" if cr["p_spear_syncd"] < 0.05 else ""
    pline(f"{cr['feature']:<24} {cr['r_spear_syncd']:+16.4f} {cr['p_spear_syncd']:10.4f}{sig}"
          f"  {cr['r_pear_syncd']:+16.4f} {cr['r_spear_syncc']:+16.4f}")

# Top-10 SyncD predictors
pline(f"\n### Top-10 SyncD predictors (Spearman) ###")
for i, cr in enumerate(corr_results[:10]):
    pline(f"  {i+1}. {cr['feature']:<24s} r={cr['r_spear_syncd']:+.4f} p={cr['p_spear_syncd']:.4f}")

# Top-10 SyncC predictors (for comparison)
corr_by_c = sorted(corr_results, key=lambda x: abs(x["r_spear_syncc"]), reverse=True)
pline(f"\n### Top-10 SyncC predictors (Spearman) ###")
for i, cr in enumerate(corr_by_c[:10]):
    pline(f"  {i+1}. {cr['feature']:<24s} r={cr['r_spear_syncc']:+.4f}")

# ====================================================================
# 1b. Correlation Heatmap (top features)
# ====================================================================
pline("\nGenerating correlation heatmap...")

top_n = 20
top_features = [cr["feature"] for cr in corr_results[:top_n]]
heatmap_cols = [TARGET, TARGET2] + top_features
corr_matrix = ml_df[heatmap_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title(f"Spearman Correlation: Top-{top_n} Features vs SyncD/SyncC\n(n={len(ml_df)})",
             fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "obs_v2_corr_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
pline(f"  Saved: figures/obs_v2_corr_heatmap.png")

# ====================================================================
# 2. LASSO REGRESSION
# ====================================================================
pline("\n" + "=" * 80)
pline("2. LASSO REGRESSION: Feature selection for SyncD")
pline("=" * 80)

# LASSO with CV
lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
lasso.fit(X_scaled, y_d)
pline(f"  Best alpha: {lasso.alpha_:.6f}")
pline(f"  R² (training): {lasso.score(X_scaled, y_d):.4f}")

# Feature coefficients
lasso_coefs = list(zip(feat_cols, lasso.coef_))
non_zero = [(f, c) for f, c in lasso_coefs if abs(c) > 0.001]
non_zero.sort(key=lambda x: abs(x[1]), reverse=True)
pline(f"  Non-zero features: {len(non_zero)}")
pline(f"\n  {'Feature':<26s} {'LASSO coef':>12s}")
pline(f"  {'-'*40}")
for f, c in non_zero[:15]:
    direction = "↓ better SyncD" if c < 0 else "↑ worse SyncD"
    pline(f"  {f:<26s} {c:+12.6f}  ({direction})")

# ====================================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# ====================================================================
pline("\n" + "=" * 80)
pline("3. RANDOM FOREST: Feature Importance for SyncD")
pline("=" * 80)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_scaled, y_d)
pline(f"  RF R² (training): {rf.score(X_scaled, y_d):.4f}")

importances = list(zip(feat_cols, rf.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)

pline(f"\n  {'Feature':<26s} {'Importance':>12s}")
pline(f"  {'-'*40}")
for f, imp in importances[:15]:
    pline(f"  {f:<26s} {imp:12.6f}")

# RF importance barplot
fig, ax = plt.subplots(figsize=(10, 8))
top_imp = importances[:15]
ax.barh([f for f, _ in top_imp], [imp for _, imp in top_imp], color="steelblue")
ax.set_xlabel("Feature Importance")
ax.set_title(f"RF Feature Importance for SyncD (n={len(ml_df)})")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIG_DIR / "obs_v2_rf_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ====================================================================
# 4. TOP FEATURE SCATTER PLOTS
# ====================================================================
pline("\n" + "=" * 80)
pline("4. SCATTER PLOTS: Top features vs SyncD & SyncC")
pline("=" * 80)

top4 = [cr["feature"] for cr in corr_results[:4]]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, feat in enumerate(top4):
    ax_d = axes[0, i]
    ax_c = axes[1, i]

    sub = ml_df[[feat, TARGET, TARGET2]].dropna()

    # SyncD plot
    ax_d.scatter(sub[feat], sub[TARGET], alpha=0.4, s=15, c="steelblue", edgecolors="none")
    r_s, p_s = stats.spearmanr(sub[feat], sub[TARGET])
    ax_d.set_title(f"{feat} vs SyncD\nr={r_s:+.3f} p={p_s:.3f}")
    ax_d.set_ylabel("SyncD (↓ better)")
    ax_d.set_xlabel(feat)

    # Try to fit a trend line
    if len(sub) > 3:
        z = np.polyfit(sub[feat], sub[TARGET], 1)
        line_x = np.linspace(sub[feat].min(), sub[feat].max(), 100)
        ax_d.plot(line_x, np.polyval(z, line_x), "r--", alpha=0.5)

    # SyncC plot
    ax_c.scatter(sub[feat], sub[TARGET2], alpha=0.4, s=15, c="darkorange", edgecolors="none")
    r_s2, p_s2 = stats.spearmanr(sub[feat], sub[TARGET2])
    ax_c.set_title(f"{feat} vs SyncC\nr={r_s2:+.3f} p={p_s2:.3f}")
    ax_c.set_ylabel("SyncC (↑ better)")
    ax_c.set_xlabel(feat)

    if len(sub) > 3:
        z = np.polyfit(sub[feat], sub[TARGET2], 1)
        line_x = np.linspace(sub[feat].min(), sub[feat].max(), 100)
        ax_c.plot(line_x, np.polyval(z, line_x), "r--", alpha=0.5)

plt.suptitle(f"Top-4 Features vs SyncD & SyncC (n={len(ml_df)})", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "obs_v2_scatter_top4.png", dpi=150, bbox_inches="tight")
plt.close()
pline(f"  Saved: figures/obs_v2_scatter_top4.png")

# ====================================================================
# 5. PCA VISUALIZATION
# ====================================================================
pline("\n" + "=" * 80)
pline("5. PCA: Feature space colored by SyncD quantile")
pline("=" * 80)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pline(f"  PCA variance: PC1={pca.explained_variance_ratio_[0]:.3f} PC2={pca.explained_variance_ratio_[1]:.3f}")

# Color by SyncD quantile
sync_d_quantile = pd.qcut(ml_df[TARGET].values, q=5, labels=["Q1(best)", "Q2", "Q3", "Q4", "Q5(worst)"])

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=ml_df[TARGET].values,
                      cmap="RdYlGn_r", alpha=0.6, s=20, edgecolors="none")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("SyncD (↓ better)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"PCA of Audio Features (n={len(ml_df)}), color=SyncD")
plt.tight_layout()
fig.savefig(FIG_DIR / "obs_v2_pca.png", dpi=150, bbox_inches="tight")
plt.close()
pline(f"  Saved: figures/obs_v2_pca.png")

# Top PC loadings
loadings = pca.components_.T
for pc_idx, pc_name in [(0, "PC1"), (1, "PC2")]:
    top_load = sorted(zip(feat_cols, loadings[:, pc_idx]), key=lambda x: abs(x[1]), reverse=True)
    pline(f"  {pc_name} top loadings:")
    for f, l in top_load[:5]:
        pline(f"    {f:<26s} {l:+8.4f}")

# ====================================================================
# 6. PARTIAL CORRELATION (CONTROLLING FOR PROVIDER)
# ====================================================================
pline("\n" + "=" * 80)
pline("6. PARTIAL CORRELATION: Within-provider only")
pline("=" * 80)

# Strategy: compute correlation within each provider, then average
providers = ["f5_tts", "f5_tts_loudnorm", "higgs_tts", "higgs_loudnorm",
             "indextts2_baseline", "gsv_tts", "chattts_original", "qwen3_loudnorm"]
providers = [p for p in providers if p in scored["provider"].values]

pline(f"\n  Within-provider Spearman r with SyncD:")
header = f"  {'Feature':<24s}"
for p in providers:
    header += f" {p[:12]:>12s}"
header += f" {'Mean':>8s}"
pline(header)
pline(f"  {'-'*24}" + f"{'':->{(13 * len(providers) + 8)}s}")

for cr in corr_results[:10]:
    feat = cr["feature"]
    line = f"  {feat:<24s}"
    r_vals = []
    for p in providers:
        sub = scored[scored["provider"] == p][[feat, TARGET]].dropna()
        if len(sub) >= 5:
            r_s, _ = stats.spearmanr(sub[feat], sub[TARGET])
            r_vals.append(r_s)
            line += f" {r_s:+12.4f}"
        else:
            line += f" {'--':>12s}"
    mean_r = np.mean(r_vals) if r_vals else 0
    line += f" {mean_r:+8.4f}"
    pline(line)

# ====================================================================
# 7. LUFS / RMS EFFECT DEEP DIVE
# ====================================================================
pline("\n" + "=" * 80)
pline("7. LOUDNESS EFFECT: LUFS-normalized vs Original within same model")
pline("=" * 80)

# Compare models that have both original and LUFS variants
pairs = [
    ("f5_tts", "f5_tts_loudnorm", "F5-TTS"),
    ("higgs_tts", "higgs_loudnorm", "Higgs TTS 3"),
    ("chattts_original", "chattts_loudnorm", "ChatTTS"),
    ("indextts2_baseline", "indextts2_loudnorm", "IndexTTS2"),
    ("gsv_tts", "gsv_loudnorm", "GSV-TTS"),
    ("fish_audio", "fish_loudnorm", "Fish S2 Pro"),
]

pline(f"\n  {'Model':<20s} {'Orig SyncD':>10} {'LUFS SyncD':>10} {'Δ':>8} {'Orig SyncC':>10} {'LUFS SyncC':>10}"
      f"  {'Orig LUFS':>10} {'LUFS LUFS':>10}")
pline(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for orig_p, lufs_p, label in pairs:
    orig = scored[scored["provider"] == orig_p]
    lufs_d = scored[scored["provider"] == lufs_p]
    if len(orig) == 0 or len(lufs_d) == 0:
        continue
    # Match by sample
    common = set(orig["sample"]) & set(lufs_d["sample"])
    orig_sub = orig[orig["sample"].isin(common)]
    lufs_sub = lufs_d[lufs_d["sample"].isin(common)]

    o_d = orig_sub[TARGET].mean()
    l_d = lufs_sub[TARGET].mean()
    o_c = orig_sub[TARGET2].mean()
    l_c = lufs_sub[TARGET2].mean()
    o_lufs = orig_sub["lufs"].mean()
    l_lufs = lufs_sub["lufs"].mean()

    delta_d = l_d - o_d
    direction_d = "BETTER" if delta_d < 0 else "worse" if delta_d > 0 else "same"
    delta_c = l_c - o_c
    direction_c = "BETTER" if delta_c > 0 else "worse" if delta_c < 0 else "same"

    pline(f"  {label:<20s} {o_d:10.3f} {l_d:10.3f} {delta_d:+8.3f}  {o_c:10.3f} {l_c:10.3f}"
          f"  {o_lufs:10.1f} {l_lufs:10.1f}  SyncD {direction_d}, SyncC {direction_c}")

# ====================================================================
# 8. PER-PROVIDER FEATURE PROFILE
# ====================================================================
pline("\n" + "=" * 80)
pline("8. PROVIDER FEATURE PROFILES (top models only)")
pline("=" * 80)

main_providers = ["f5_tts", "f5_tts_loudnorm", "fish_audio", "dashscope_vc",
                  "qwen3_loudnorm", "higgs_tts", "higgs_loudnorm",
                  "indextts2_baseline", "gsv_tts", "chattts_original",
                  "faster_qwen3", "siliconflow"]
main_providers = [p for p in main_providers if p in scored["provider"].values]

key_feats = [cr["feature"] for cr in corr_results[:8]]
header = f"\n  {'Provider':<24s} {'SyncD':>7} {'SyncC':>7}"
for f in key_feats[:4]:
    header += f" {f[:10]:>10s}"
pline(header)
pline(f"  {'-'*24} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for p in main_providers:
    sub = scored[scored["provider"] == p]
    line = f"  {p:<24s} {sub[TARGET].mean():7.3f} {sub[TARGET2].mean():7.3f}"
    for f in key_feats[:4]:
        if f in sub.columns:
            line += f" {sub[f].mean():10.3f}"
        else:
            line += f" {'--':>10s}"
    pline(line)

# ====================================================================
# 9. FEATURE PAIRWISE SPACE
# ====================================================================
pline("\nGenerating pairwise plot for top features...")

# Select top 6 features + sync_d + sync_c
pair_cols = [TARGET, TARGET2] + [cr["feature"] for cr in corr_results[:6]]
pair_df = ml_df[pair_cols].sample(min(200, len(ml_df))) if len(ml_df) > 200 else ml_df[pair_cols]

g = sns.pairplot(pair_df, diag_kind="kde",
                 plot_kws={"alpha": 0.3, "s": 10, "edgecolor": "none"})
g.fig.suptitle("Pairwise relationships: Top features + SyncD + SyncC", y=1.02, fontsize=14)
g.fig.savefig(FIG_DIR / "obs_v2_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
pline(f"  Saved: figures/obs_v2_pairplot.png")

# ====================================================================
# 10. SUMMARY
# ====================================================================
pline("\n" + "=" * 80)
pline("10. SUMMARY")
pline("=" * 80)

pline(f"\nDataset: {len(scored)} scored rows, {len(feat_cols)} features, {len(ml_df)} clean rows")
pline(f"\nTop SyncD predictors (lower SyncD = better lip sync):")
for i, cr in enumerate(corr_results[:5]):
    direction = "negative (good)" if cr["r_spear_syncd"] < 0 else "positive (bad)"
    pline(f"  {i+1}. {cr['feature']:<24s} Spearman r={cr['r_spear_syncd']:+.4f} ({direction})")

pline(f"\nTop SyncC predictors (higher SyncC = better confidence):")
for i, cr in enumerate(corr_by_c[:5]):
    direction = "positive (good)" if cr["r_spear_syncc"] > 0 else "negative (bad)"
    pline(f"  {i+1}. {cr['feature']:<24s} Spearman r={cr['r_spear_syncc']:+.4f} ({direction})")

pline(f"\nLASSO selected {len(non_zero)} non-zero features for SyncD prediction")
pline(f"Random Forest top-3 importance:")
for f, imp in importances[:3]:
    pline(f"  {f:<26s} {imp:.4f}")

# Write to file
out_path = RES_DIR / "observational_analysis_v2.txt"
with open(out_path, "w") as fh:
    fh.write("\n".join(out_lines))
pline(f"\nSaved: {out_path}")
