#!/usr/bin/env python3
"""02_tts.py - Stub (issue #1).

Real implementation in issue #3: faster-qwen3-tts generate_voice_clone
ICL 自克隆,text=ref_text=ASR(wav_i), ref_audio=wav_i, language=Chinese, seed=42.
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
    out_dir = repo / "runs" / args.run_id / "02_tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [1] if args.smoke else list(range(1, 11))
    for i in samples:
        (out_dir / f"{i}.wav").write_bytes(b"RIFstub")  # 0-byte placeholder
    meta = {
        "status": "stub",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "mode": "icl",
        "seed": 42,
        "samples": samples,
        "failed": [],
        "note": "Real TTS (faster-qwen3-tts) comes in issue #3",
    }
    (out_dir / "tts_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[stub] wrote {out_dir}/{{1..10}}.wav + tts_meta.json")


if __name__ == "__main__":
    main()