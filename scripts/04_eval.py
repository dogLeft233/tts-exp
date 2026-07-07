#!/usr/bin/env python3
"""04_eval.py - Stub (issue #1).

Real implementation in issue #5: subprocess 调 syncnet_python/run_syncnet.py
解析 stdout 的 Confidence(→Sync-C)、Min dist(→Sync-D)、AV offset。
"""
import argparse
import json
from pathlib import Path

CONDITIONS = ["natural_raw", "natural_resamp", "tts_raw", "tts_resamp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    base = repo / "runs" / args.run_id / "04_eval"
    base.mkdir(parents=True, exist_ok=True)

    samples = [1] if args.smoke else list(range(1, 11))
    for cond in CONDITIONS:
        cond_dir = base / cond
        cond_dir.mkdir(exist_ok=True)
        for i in samples:
            sample_dir = cond_dir / str(i)
            sample_dir.mkdir(exist_ok=True)
            (sample_dir / "syncnet.json").write_text(json.dumps({
                "status": "stub",
                "sample_id": i,
                "condition": cond,
                "sync_c": None,
                "sync_d": None,
                "av_offset": None,
                "note": "Real SyncNet eval comes in issue #5",
            }, indent=2))
    meta = {
        "status": "stub",
        "failed": [],
        "note": "Real SyncNet eval comes in issue #5",
    }
    (base / "eval_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[stub] wrote {base}/{{conditions}}/{{i}}/syncnet.json + eval_meta.json")


if __name__ == "__main__":
    main()