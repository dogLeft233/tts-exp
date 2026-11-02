#!/usr/bin/env python
"""Aggregate per-video metric rows into group summaries, paired deltas and
distribution distances (FID / KID / FVD).

Usage
-----
    python scripts/43_aggregate_video_metrics.py \
        --run-dir F:/.../runs/a2 --manifest F:/.../manifests/manifest_a2.json \
        --out F:/.../runs/a2/report --sync-json sync_a2.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

LOWER_BETTER = {"niqe", "brisque", "lpips_temporal", "lpips_vs_first", "kid_x1000",
                "fid", "fvd", "frame_diff_l1"}
HIGHER_BETTER = {"clipiqa", "ssim_vs_first", "psnr_vs_first", "sharpness_lapvar"}


# ---------------------------------------------------------------- loading
def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(run_dir.glob("per_video_shard*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if "error" not in r]
    for r in rows:
        # Sampled frames are `stride` native frames apart.  The natural and TTS
        # arms have different durations, hence different strides, so raw
        # frame-to-frame differences are not comparable across arms.  Convert
        # to a per-native-frame rate (motion is ~linear over 2-3 frame gaps).
        total, sampled = r.get("n_frames_total"), r.get("n_frames_sampled")
        if total and sampled and sampled > 1:
            stride = total / sampled
            r["sampled_stride"] = stride
            if isinstance(r.get("frame_diff_l1"), (int, float)):
                r["diff_per_frame"] = r["frame_diff_l1"] / stride
            if isinstance(r.get("lpips_temporal"), (int, float)):
                r["lpips_per_frame"] = r["lpips_temporal"] / stride
    return rows


def scalar_keys(rows: list[dict]) -> list[str]:
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in (
                    "fps", "n_frames_total"):
                keys.add(k)
    return sorted(keys)


# ---------------------------------------------------------------- stats
def boot_ci(v: np.ndarray, n_boot: int = 4000, seed: int = 11) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: per-item values (paired). Returns delta = b - a statistics."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"n": 0}
    d = b - a
    out = {
        "n": int(a.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": float(d.mean()),
        "win_rate_b_gt_a": float((d > 0).mean()),
        "median_delta": float(np.median(d)),
    }
    if a.size >= 5:  # a bootstrap CI over 2-4 items is not informative
        out["delta_ci"] = list(boot_ci(d))
    if a.size >= 5:
        try:
            from scipy import stats

            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
            out["cohens_dz"] = float(d.mean() / (d.std(ddof=1) + 1e-12))
        except Exception:
            pass
    return out


def load_feats(rows: list[dict], key: str) -> np.ndarray:
    mats = []
    for r in rows:
        f = r.get(key)
        if f and Path(f).exists():
            mats.append(np.load(f))
    return np.concatenate(mats, 0) if mats else np.zeros((0, 0), np.float32)


def kid(x: np.ndarray, y: np.ndarray, subsets: int = 50, size: int = 100,
        seed: int = 20261021) -> dict:
    if len(x) < 10 or len(y) < 10:
        return {"kid_x1000": float("nan"), "kid_std_x1000": float("nan")}
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), size)
    d = x.shape[1]
    vals = []
    for _ in range(subsets):
        i = rng.choice(len(x), n, replace=False)
        j = rng.choice(len(y), n, replace=False)
        a, b = x[i], y[j]
        kaa = (a @ a.T / d + 1.0) ** 3
        kbb = (b @ b.T / d + 1.0) ** 3
        kab = (a @ b.T / d + 1.0) ** 3
        saa = (kaa.sum() - np.trace(kaa)) / (n * (n - 1))
        sbb = (kbb.sum() - np.trace(kbb)) / (n * (n - 1))
        vals.append(saa + sbb - 2 * kab.mean())
    v = np.asarray(vals)
    return {"kid_x1000": float(v.mean() * 1000), "kid_std_x1000": float(v.std() * 1000)}


def frechet(x: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> float:
    if len(x) < 10 or len(y) < 10:
        return float("nan")
    mu1, mu2 = x.mean(0).astype(np.float64), y.mean(0).astype(np.float64)
    d = mu1.size
    s1 = np.cov(x, rowvar=False).astype(np.float64).reshape(d, d)
    s2 = np.cov(y, rowvar=False).astype(np.float64).reshape(d, d)
    s1 += np.eye(d) * eps
    s2 += np.eye(d) * eps
    from scipy.linalg import sqrtm

    cm = sqrtm(s1 @ s2)
    cm = np.real(cm)
    if not np.all(np.isfinite(cm)):
        return float("nan")
    val = float((mu1 - mu2) @ (mu1 - mu2) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cm))
    return max(val, 0.0)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sync-json", default="")
    ap.add_argument("--pairs", default="", help="JSON file with [[gA,gB],...]")
    args = ap.parse_args()

    run_dir, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(run_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    family = manifest.get("family", "?")
    groups = sorted({r["group"] for r in rows})
    by_group: dict[str, list[dict]] = {g: [r for r in rows if r["group"] == g] for g in groups}
    keys = scalar_keys(rows)

    # ---- group summaries
    summary: dict[str, dict] = {}
    for g, rs in by_group.items():
        summary[g] = {"n_videos": len(rs)}
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rs], dtype=np.float64)
            if np.isfinite(v).any():
                m, lo, hi = boot_ci(v)
                summary[g][k] = {"mean": m, "ci_lo": lo, "ci_hi": hi,
                                 "n": int(np.isfinite(v).sum())}

    # ---- pairs
    pairs: list[list[str]] = []
    if args.pairs:
        pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8-sig"))
    elif family == "a2":
        models = sorted({g.split("/")[1] for g in groups})
        pairs = [[f"A2/{m}/natural_raw", f"A2/{m}/tts_raw"] for m in models
                 if f"A2/{m}/natural_raw" in by_group and f"A2/{m}/tts_raw" in by_group]
    elif family == "a1":
        pairs = [["A1/natural_raw", "A1/tts_raw"]]
    elif family == "b":
        stems = sorted({g.rsplit("/", 1)[0] for g in groups})
        for s in stems:
            for inorm in ("off", "on"):
                q, z, l = (f"{s}/{inorm}norm_testm6", f"{s}/{inorm}norm_testp0",
                           f"{s}/{inorm}norm_testp6")
                for a, b in ((q, z), (z, l), (q, l)):
                    if a in by_group and b in by_group:
                        pairs.append([a, b])
    elif family == "c":
        stems = sorted({g.rsplit("/", 1)[0] for g in groups})
        for s in stems:
            for a, b in (("s0.50", "s1.00"), ("s1.00", "s2.00"), ("s0.50", "s2.00")):
                ga, gb = f"{s}/{a}", f"{s}/{b}"
                if ga in by_group and gb in by_group:
                    pairs.append([ga, gb])

    # ---- distribution distances per pair
    pair_results = []
    for ga, gb in pairs:
        if ga not in by_group or gb not in by_group:
            continue
        res = {"a": ga, "b": gb, "note": "delta = b - a"}
        fid_a = load_feats(by_group[ga], "fid_feat_file")
        fid_b = load_feats(by_group[gb], "fid_feat_file")
        fvd_a = load_feats(by_group[ga], "fvd_feat_file")
        fvd_b = load_feats(by_group[gb], "fvd_feat_file")
        if len(fid_a) and len(fid_b):
            res["fid"] = frechet(fid_a, fid_b)
            res["kid"] = kid(fid_a, fid_b)
            res["n_frames_fid"] = [int(len(fid_a)), int(len(fid_b))]
        if len(fvd_a) and len(fvd_b):
            res["fvd"] = frechet(fvd_a, fvd_b)
            res["n_clips_fvd"] = [int(len(fvd_a)), int(len(fvd_b))]

        # paired per-item scalar deltas
        items_a = {r["item"]: r for r in by_group[ga] if "item" in r}
        items_b = {r["item"]: r for r in by_group[gb] if "item" in r}
        shared = sorted(set(items_a) & set(items_b), key=lambda s: str(s))
        paired = {}
        for k in keys:
            va = np.array([items_a[i].get(k, np.nan) for i in shared], dtype=np.float64)
            vb = np.array([items_b[i].get(k, np.nan) for i in shared], dtype=np.float64)
            if np.isfinite(va).any() and np.isfinite(vb).any():
                paired[k] = paired_stats(va, vb)
        res["paired_items"] = [str(s) for s in shared]
        res["paired"] = paired
        pair_results.append(res)

    # ---- Sync-C join
    sync_note = None
    if args.sync_json:
        sync = json.loads(Path(args.sync_json).read_text(encoding="utf-8-sig"))
        for res in pair_results:
            sa = sync.get(res["a"], {})
            sb = sync.get(res["b"], {})
            shared = [i for i in res["paired_items"] if i in sa and i in sb]
            if len(shared) < 5:
                continue
            da = np.array([sa[i] for i in shared], dtype=np.float64)
            db = np.array([sb[i] for i in shared], dtype=np.float64)
            res["sync"] = paired_stats(da, db)
            # correlation between metric delta and sync delta
            corr = {}
            from scipy import stats as sstats

            dsync = db - da
            for k, st in res["paired"].items():
                if st.get("n", 0) < max(4, int(0.6 * len(shared))):
                    continue
                items_a = {r["item"]: r for r in by_group[res["a"]]}
                items_b = {r["item"]: r for r in by_group[res["b"]]}
                dm = np.array([items_b[i].get(k, np.nan) - items_a[i].get(k, np.nan)
                               for i in shared], dtype=np.float64)
                ok = np.isfinite(dm)
                if ok.sum() >= 8 and np.std(dm[ok]) > 0:
                    rho, p = sstats.spearmanr(dm[ok], dsync[ok])
                    corr[k] = {"spearman_rho": float(rho), "p": float(p), "n": int(ok.sum())}
            res["sync_metric_corr"] = corr

    report = {"family": family, "run_dir": str(run_dir), "n_videos": len(rows),
              "groups": groups, "group_summary": summary,
              "pairs": pair_results, "lower_is_better": sorted(LOWER_BETTER),
              "higher_is_better": sorted(HIGHER_BETTER), "sync_note": sync_note}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    # ---- markdown
    lines = [f"# Metric report — family {family}", "",
             f"{len(rows)} videos, {len(groups)} groups.", ""]
    for res in pair_results:
        lines.append(f"## {res['a']}  vs  {res['b']}")
        lines.append("")
        lines.append("| metric | mean A | mean B | Δ (B−A) | 95% CI | win-rate B>A | n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for k, st in res["paired"].items():
            if st.get("n", 0) < 1:
                continue
            ci = st.get("delta_ci", [math.nan, math.nan, math.nan])
            lines.append(
                f"| {k} | {st['mean_a']:.5g} | {st['mean_b']:.5g} | {st['delta']:+.5g} | "
                f"[{ci[1]:+.4g}, {ci[2]:+.4g}] | {st['win_rate_b_gt_a']:.2f} | {st['n']} |")
        if "fid" in res:
            lines.append("")
            lines.append(f"FID={res.get('fid', float('nan')):.2f}  "
                         f"KID×1000={res.get('kid', {}).get('kid_x1000', float('nan')):.2f} "
                         f"(±{res.get('kid', {}).get('kid_std_x1000', float('nan')):.2f})  "
                         f"FVD={res.get('fvd', float('nan')):.2f}  "
                         f"[frames {res.get('n_frames_fid')}, clips {res.get('n_clips_fvd')}]")
        if "sync" in res:
            s = res["sync"]
            lines.append(f"Sync-C: A={s['mean_a']:.3f} B={s['mean_b']:.3f} "
                         f"Δ={s['delta']:+.3f} CI[{s['delta_ci'][1]:+.3f},{s['delta_ci'][2]:+.3f}] "
                         f"win={s['win_rate_b_gt_a']:.2f} n={s['n']}")
        if res.get("sync_metric_corr"):
            lines.append("")
            lines.append("Δmetric ↔ ΔSync-C (Spearman): " + ", ".join(
                f"{k} ρ={v['spearman_rho']:+.2f}(p={v['p']:.2g})"
                for k, v in res["sync_metric_corr"].items()))
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'report.json'} ({len(pair_results)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
