#!/usr/bin/env python3
"""00_datacheck.py - Stub (issue #1).

Real implementation in issue #2: 只读体检 wav 时长/采样率/声道、png 尺寸,
对 <4s 的 wav 报警但不断流(见 plan Q10/Q21)。
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
    out_dir = repo / "runs" / args.run_id / "00_datacheck"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [1] if args.smoke else list(range(1, 11))
    placeholder = {
        "status": "stub",
        "samples": samples,
        "note": "Real check (wav duration/sr, png size) comes in issue #2",
    }
    (out_dir / "00_datacheck.json").write_text(json.dumps(placeholder, indent=2))
    print(f"[stub] wrote {out_dir / '00_datacheck.json'}")


if __name__ == "__main__":
    main()