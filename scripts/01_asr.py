#!/usr/bin/env python3
"""01_asr.py - Stub (issue #1).

Real implementation in issue #2: 调用 Qwen ASR Flash(DashScope API),
key 来自 scripts/.env, 产 runs/<run_id>/01_transcript/{1..10}.txt + transcript.json.
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
    out_dir = repo / "runs" / args.run_id / "01_transcript"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [1] if args.smoke else list(range(1, 11))
    for i in samples:
        (out_dir / f"{i}.txt").write_text(f"[stub] transcript for sample {i}\n")
    summary = {
        "status": "stub",
        "samples": samples,
        "failed": [],
        "note": "Real ASR (DashScope qwen-asr-flash) comes in issue #2",
    }
    (out_dir / "transcript.json").write_text(json.dumps(summary, indent=2))
    print(f"[stub] wrote {out_dir}/{{1..10}}.txt + transcript.json")


if __name__ == "__main__":
    main()