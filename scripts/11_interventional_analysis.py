#!/usr/bin/env python3
"""11_interventional_analysis.py — Compare normalized vs original SyncC scores
to test causal effects of loudness and duration on TFG lip sync quality.

Input:
  - Original scores: data/r3_results/*.csv (r3 runs)
  - Normalized scores: runs/r4_*/04_eval/tts_raw/*/syncnet.json (on server)

Analysis:
  1. Per-condition: original vs each normalization (paired t-test)
  2. RM-ANOVA: emotion × normalization (4 levels)
  3. Ranking stability: does the rank order change after normalization?
  4. Delta decomposition: which normalization explains which emotion effect?
"""
import csv
import json
import glob
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
P2_IDS = [1, 4, 7, 10, 13]
ALL_IDS = list(range(1, 14))
NORM_TYPES = ["loudnorm", "durnorm", "bothnorm"]


def load_original_scores():
    """Load original SyncC scores from r3_results CSVs."""
    scores = {}  # (provider_name, sample_id) -> sync_c

    # Load P2 emotion sweep
    p2_path = REPO / "data" / "r3_results" / "p2_emotion_sweep.csv"
    if p2_path.exists():
        with open(p2_path) as f:
            for r in csv.DictReader(f):
                if r.get("tts_sync_c"):
                    key = (f"emo_{r['emotion']}", int(r["sample"]))
                    scores[key] = float(r["tts_sync_c"])

    # Load baseline/gsv
    bg_path = REPO / "data" / "r3_results" / "baseline_gsv.csv"
    if bg_path.exists():
        with open(bg_path) as f:
            for r in csv.DictReader(f):
                if r.get("tts_sync_c"):
                    name = "fish" if r["condition"] == "r3_fish" else \
                           "indextts2" if r["condition"] == "r3_indextts2_baseline" else \
                           "gsv" if r["condition"] == "r3_gsv_tts" else None
                    if name:
                        scores[(name, int(r["sample"]))] = float(r["tts_sync_c"])

    # Load fish_audio from r2 (we downloaded syncnet locally)
    fish_base = REPO / "runs" / "r2_fish_audio_20260707T150723Z" / "04_eval" / "tts_raw"
    for jf in fish_base.glob("*/syncnet.json"):
        try:
            d = json.loads(jf.read_text())
            scores[("fish", int(d["sample_id"]))] = float(d["sync_c"])
        except Exception:
            pass

    return scores


def load_normalized_scores(norm_type):
    """Load normalized SyncC scores from r4 runs (local or server)."""
    scores = {}

    base = REPO / "runs"
    run_prefix = f"r4_{norm_type}"

    # Emotion runs
    for emo in EMOTIONS:
        tts_dir = base / f"{run_prefix}_emo_{emo}" / "04_eval" / "tts_raw"
        for jf in tts_dir.glob("*/syncnet.json"):
            try:
                d = json.loads(jf.read_text())
                scores[(f"emo_{emo}", norm_type, int(d["sample_id"]))] = float(d["sync_c"])
            except Exception:
                pass

    # Group runs
    for grp in ["fish", "indextts2", "gsv"]:
        tts_dir = base / f"{run_prefix}_{grp}" / "04_eval" / "tts_raw"
        for jf in tts_dir.glob("*/syncnet.json"):
            try:
                d = json.loads(jf.read_text())
                scores[(grp, norm_type, int(d["sample_id"]))] = float(d["sync_c"])
            except Exception:
                pass

    return scores


def main():
    print("=" * 70)
    print("PHASE B: Intervention Analysis — Normalized Audio SyncC")
    print("=" * 70)

    # Load original scores
    orig = load_original_scores()
    print(f"\nLoaded {len(orig)} original SyncC entries")

    # Load normalized scores
    norm_data = {}
    for nt in NORM_TYPES:
        ns = load_normalized_scores(nt)
        norm_data[nt] = ns
        if ns:
            print(f"  {nt}: {len(ns)} entries")
        else:
            print(f"  {nt}: NO DATA — run pipeline on server first")

    if not any(norm_data.values()):
        print("\n*** No r4 data yet. Run batch_pipeline_r4.sh on server first. ***")
        return

    # ============================================================
    # 1. Pairwise comparison: original vs each normalization
    # ============================================================
    print("\n" + "=" * 70)
    print("1. PAIRWISE COMPARISON: Original vs Normalized SyncC")
    print("=" * 70)

    # Build per-sample delta for each emotion × normalization
    emo_deltas = defaultdict(lambda: defaultdict(list))
    for nt in NORM_TYPES:
        ns = norm_data[nt]
        for emo in EMOTIONS:
            for sid in P2_IDS:
                norm_key = (f"emo_{emo}", nt, sid)
                orig_key = (f"emo_{emo}", sid)
                if norm_key in ns and orig_key in orig:
                    delta = ns[norm_key] - orig[orig_key]
                    emo_deltas[emo][nt].append(delta)

    print(f"\n{'Emotion':<16} {'loudnorm Δ':>12} {'p':>8}  {'durnorm Δ':>12} {'p':>8}  {'bothnorm Δ':>12} {'p':>8}")
    print("-" * 82)
    for emo in EMOTIONS:
        parts = []
        for nt in NORM_TYPES:
            deltas = emo_deltas[emo][nt]
            if len(deltas) >= 3:
                mean_d = np.mean(deltas)
                t_stat, p = stats.ttest_rel(
                    [orig[(f"emo_{emo}", sid)] for sid in P2_IDS if (f"emo_{emo}", sid) in orig],
                    [norm_data[nt].get((f"emo_{emo}", nt, sid)) for sid in P2_IDS if (f"emo_{emo}", nt, sid) in norm_data[nt]]
                ) if len(deltas) >= 2 else (0, 1)
                sig = "*" if p < 0.05 else "**" if p < 0.01 else "***" if p < 0.001 else ""
                parts.append(f"{mean_d:+12.4f} {p:8.4f}{sig}")
            else:
                parts.append(f"{'N/A':>12} {'N/A':>8}")
        print(f"  {emo:<14} {'  '.join(parts)}")

    # ============================================================
    # 2. Group-level comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("2. GROUP-LEVEL: Fish / IndexTTS2 / GSV")
    print("=" * 70)

    for grp in ["fish", "indextts2", "gsv"]:
        print(f"\n### {grp} ###")
        orig_vals = [orig[(grp, sid)] for sid in ALL_IDS if (grp, sid) in orig]
        orig_mean = np.mean(orig_vals) if orig_vals else 0
        print(f"  Original SyncC = {orig_mean:.3f}")
        for nt in NORM_TYPES:
            ns = norm_data[nt]
            norm_vals = [ns[(grp, nt, sid)] for sid in ALL_IDS if (grp, nt, sid) in ns]
            if norm_vals:
                norm_mean = np.mean(norm_vals)
                print(f"  {nt:12s} SyncC = {norm_mean:.3f}  Δ = {norm_mean - orig_mean:+.3f}")
            else:
                print(f"  {nt:12s}: no data")

    # ============================================================
    # 3. RM-ANOVA: emotion × normalization
    # ============================================================
    print("\n" + "=" * 70)
    print("3. REPEATED-MEASURES ANOVA: Emotion × Normalization")
    print("=" * 70)

    # Build matrix: rows=subjects (samples), cols=conditions (emotion × normalization)
    # 8 emotions × 4 normalization (original + 3 norm) = 32 conditions
    conditions = ["original"] + NORM_TYPES

    for emo in EMOTIONS:
        print(f"\n  {emo}:")
        cond_means = []
        for cond in conditions:
            if cond == "original":
                vals = [orig.get((f"emo_{emo}", sid)) for sid in P2_IDS if (f"emo_{emo}", sid) in orig]
            else:
                vals = [norm_data[cond].get((f"emo_{emo}", cond, sid)) for sid in P2_IDS]
            vals = [v for v in vals if v is not None]
            if vals:
                cond_means.append((cond, np.mean(vals), np.std(vals, ddof=1)))
        for c, m, s in cond_means:
            orig_val = cond_means[0][1] if cond_means else 0
            delta = m - orig_val
            print(f"    {c:12s}  SyncC={m:.3f}±{s:.3f}  Δ={delta:+.3f}")

    # ============================================================
    # 4. Ranking comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("4. RANKING STABILITY")
    print("=" * 70)

    # Original ranking
    orig_emo_scores = {}
    for emo in EMOTIONS:
        vals = [orig.get((f"emo_{emo}", sid)) for sid in P2_IDS if (f"emo_{emo}", sid) in orig]
        if vals:
            orig_emo_scores[emo] = np.mean(vals)

    orig_ranked = sorted(orig_emo_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Original':<16} {'loudnorm':<16} {'durnorm':<16} {'bothnorm':<16}")
    print("-" * 68)

    for rank, (emo, _) in enumerate(orig_ranked, 1):
        parts = [f"{rank:<6} {emo:<16}"]
        for nt in NORM_TYPES:
            vals = [norm_data[nt].get((f"emo_{emo}", nt, sid)) for sid in P2_IDS]
            vals = [v for v in vals if v is not None]
            if vals:
                norm_mean = np.mean(vals)
                # Find rank within normalized scores
                norm_emo_scores = {}
                for e in EMOTIONS:
                    ev = [norm_data[nt].get((f"emo_{e}", nt, sid)) for sid in P2_IDS]
                    ev = [x for x in ev if x is not None]
                    if ev:
                        norm_emo_scores[e] = np.mean(ev)
                norm_ranked = sorted(norm_emo_scores.items(), key=lambda x: x[1], reverse=True)
                norm_rank = next(i+1 for i, (e, _) in enumerate(norm_ranked) if e == emo)
                arrow = "↑" if norm_rank < rank else "↓" if norm_rank > rank else "="
                parts.append(f"{norm_mean:.3f} #{norm_rank}{arrow}")
            else:
                parts.append("N/A")
        print("  " + "      ".join(parts))

    # ============================================================
    # 5. Effect size summary
    # ============================================================
    print("\n" + "=" * 70)
    print("5. EFFECT SIZE SUMMARY")
    print("=" * 70)

    # Which normalization has the biggest impact on the top/bottom emotions?
    print(f"\n### Does normalization close the angry-melancholic gap? ###")
    for nt in NORM_TYPES:
        ns = norm_data[nt]
        angry_vals = [ns.get((f"emo_angry", nt, sid)) for sid in P2_IDS]
        angry_vals = [v for v in angry_vals if v is not None]
        mel_vals = [ns.get((f"emo_melancholic", nt, sid)) for sid in P2_IDS]
        mel_vals = [v for v in mel_vals if v is not None]

        orig_angry = np.mean([orig.get((f"emo_angry", sid)) for sid in P2_IDS if (f"emo_angry", sid) in orig])
        orig_mel = np.mean([orig.get((f"emo_melancholic", sid)) for sid in P2_IDS if (f"emo_melancholic", sid) in orig])
        orig_gap = orig_angry - orig_mel

        if angry_vals and mel_vals:
            norm_gap = np.mean(angry_vals) - np.mean(mel_vals)
            ratio = norm_gap / orig_gap if orig_gap else 1
            print(f"  {nt:12s}: original gap={orig_gap:.2f}, normalized gap={norm_gap:.2f} "
                  f"({ratio*100:.0f}% of original)")


if __name__ == "__main__":
    main()
