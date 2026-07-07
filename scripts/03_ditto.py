#!/usr/bin/env python3
"""03_ditto.py - Stub (issue #1).

Real implementation in issue #4: subprocess 调 ditto-talkinghead/inference.py
4 个条件 × 10 个样本 = 40 条视频推理。
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
    base = repo / "runs" / args.run_id / "03_ditto"
    base.mkdir(parents=True, exist_ok=True)

    samples = [1] if args.smoke else list(range(1, 11))
    for cond in CONDITIONS:
        cond_dir = base / cond
        cond_dir.mkdir(exist_ok=True)
        for i in samples:
            (cond_dir / f"{i}.mp4").write_bytes(b"")  # empty stub
    meta = {
        "status": "stub",
        "mode_used": "trt_online (placeholder)",
        "seed": 42,
        "conditions": CONDITIONS,
        "samples": samples,
        "failed": [],
        "note": "Real DiTo 4-condition inference comes in issue #4",
    }
    (base / "ditto_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[stub] wrote {base}/{{conditions}}/{{1..10}}.mp4 + ditto_meta.json")


if __name__ == "__main__":
    main()