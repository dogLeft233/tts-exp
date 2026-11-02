#!/usr/bin/env python
"""Render the markdown report for the video-quality metric suite.

Usage:
  python scripts/45_render_video_metric_report.py \
      --a1 runs/a1/report/report.json --a2 runs/a2/report/report.json \
      --b runs/b/report/report.json --c runs/c/report/report.json \
      --sync-b sync_b.json --sync-c sync_c.json --out report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# metrics shown in the per-model comparison table (delta = TTS - natural)
A_TABLE = [
    ("niqe", "ΔNIQE", "lower"),
    ("brisque", "ΔBRISQUE", "lower"),
    ("clipiqa", "ΔCLIP-IQA", "higher"),
    ("diff_per_frame", "Δmotion/frame", "neutral"),
    ("ssim_vs_first", "ΔSSIM(ref)", "higher"),
]
CELL_METRICS = ["niqe", "brisque", "clipiqa", "diff_per_frame"]


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pair_delta(rep: dict, a: str, b: str) -> dict:
    for pr in rep["pairs"]:
        if pr["a"] == a and pr["b"] == b:
            return pr
    return {}


def fmt(x, nd=3, plus=False):
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if v != v:
        return "—"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def table_a(rep: dict, title: str) -> list[str]:
    lines = [f"### {title}", "",
             "| model | ΔSync-C | " + " | ".join(c[1] for c in A_TABLE) +
             " | FID | FVD |", "|---|" + "---:|" * (len(A_TABLE) + 3)]
    for pr in rep["pairs"]:
        model = pr["a"].split("/")[1]
        sync = pr.get("sync") or {}
        cells = [f"{model}", fmt(sync.get("delta"), 3, True)]
        for key, _, _ in A_TABLE:
            cells.append(fmt((pr["paired"].get(key) or {}).get("delta"), 4, True))
        cells.append(fmt(pr.get("fid"), 2))
        cells.append(fmt(pr.get("fvd"), 2))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def table_cells(rep: dict, sync: dict, title: str, row_key) -> list[str]:
    lines = [f"### {title}", "",
             "| cell | Sync-C | " + " | ".join(CELL_METRICS) + " | FID vs base | KID×1e3 |",
             "|---|" + "---:|" * (len(CELL_METRICS) + 3)]
    summary = rep["group_summary"]
    for g in rep["groups"]:
        sc = sync.get(g, {})
        sc_mean = sum(sc.values()) / len(sc) if sc else None
        row = [g.split("/", 1)[1] if "/" in g else g, fmt(sc_mean, 3)]
        for m in CELL_METRICS:
            row.append(fmt((summary.get(g, {}).get(m) or {}).get("mean"), 4))
        fid = None
        kidv = None
        for pr in rep["pairs"]:
            if g in (pr["a"], pr["b"]):
                fid = pr.get("fid")
                kidv = (pr.get("kid") or {}).get("kid_x1000")
                break
        row.append(fmt(fid, 2))
        row.append(fmt(kidv, 2, True))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def agreement(rep: dict) -> list[str]:
    """How many models move a metric in the same direction as Sync-C."""
    out = []
    pairs = rep["pairs"]
    for key, label, direction in A_TABLE:
        ds = [(pr["paired"].get(key) or {}).get("delta") for pr in pairs]
        ds = [d for d in ds if d is not None and d == d]
        if not ds:
            continue
        pos = sum(1 for d in ds if d > 0)
        out.append(f"- {label}: {pos}/{len(ds)} models positive, "
                   f"median {sorted(ds)[len(ds) // 2]:+.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("a1", "a2", "b", "c"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--sync-b", required=True)
    ap.add_argument("--sync-c", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a1, a2 = load(args.a1), load(args.a2)
    b, c = load(args.b), load(args.c)
    sb = json.loads(Path(args.sync_b).read_text(encoding="utf-8"))
    sc = json.loads(Path(args.sync_c).read_text(encoding="utf-8"))

    L: list[str] = []
    L += ["## 结果表（自动生成）", ""]

    L += ["### 表 1 — 12 个 TFG 模型，TTS 音轨 vs 自然音轨（13 样本配对）", "",
          "Δ = TTS − 自然。NIQE/BRISQUE 越低越好，CLIP-IQA 越高越好，"
          "motion/frame 为去掉时长差异后的每帧运动量（中性方向）。FID/FVD 是两条音轨"
          "输出分布之间的距离（不是质量，只说明音轨换了以后画面变了多少）。", ""]
    L += table_a(a2, "A2 逐模型（13 样本，AISHELL-1）")[2:]
    L += [""]

    L += agreement(a2)
    L += [""]

    L += ["### 表 2 — 250 样本跨数据集（LeapTalk）", ""]
    L += table_a(a1, "A1（250 配对样本，5 数据集）")[2:]
    pr = a1["pairs"][0]
    if pr.get("sync"):
        s = pr["sync"]
        L += ["", f"Sync-C: 自然 {s['mean_a']:.3f} → TTS {s['mean_b']:.3f} "
                  f"(Δ {s['delta']:+.3f}, 95%CI [{s['delta_ci'][1]:+.3f}, "
                  f"{s['delta_ci'][2]:+.3f}], 正向 {s['win_rate_b_gt_a'] * 100:.0f}%, n={s['n']})"]
    if pr.get("sync_metric_corr"):
        L += ["", "Δmetric 与 ΔSync-C 的 Spearman 相关： " + ", ".join(
            f"{k} ρ={v['spearman_rho']:+.2f}(p={v['p']:.2g})"
            for k, v in pr["sync_metric_corr"].items())]
    L += [""]

    L += ["### 表 3 — 测试电平 × 推理归一化（SyncTalk May，固定生成器）", ""]
    L += table_cells(b, sb, "B：训练增益 / 推理归一化 / 测试增益", None)
    L += [""]

    L += ["### 表 4 — 训练尺度 × 推理尺度（May post-norm）", ""]
    L += table_cells(c, sc, "C：训练尺度 vs 推理尺度", None)
    L += [""]

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
