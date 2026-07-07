#!/usr/bin/env python3
"""05_report.py - Stub (issue #1).

Real implementation in issue #6: 汇总 04_eval 的 syncnet.json,产
summary.csv + report.md + figures/{paired_bar,scatter,bland_altman}。
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "runs" / args.run_id / "05_report"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.csv").write_text("sample_id,condition,sync_c,sync_d,av_offset\n")
    (out_dir / "report.md").write_text(
        "# tts-tfg 提升实验复现 report (stub)\n\n"
        f"run_id: `{args.run_id}`  smoke: `{args.smoke}`\n\n"
        "Real report (mean Sync-C/D per condition, paired t, Cohen's d, "
        "成功率对比, figures) comes in issue #6.\n"
    )
    # 占位 figures
    for fig in ("paired_bar.png", "scatter.png", "bland_altman.png"):
        (fig_dir / fig).write_bytes(b"")
    meta = {"status": "stub", "note": "Real report comes in issue #6"}
    (out_dir / "report_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[stub] wrote {out_dir}/summary.csv + report.md + figures/")


if __name__ == "__main__":
    main()