#!/usr/bin/env python3
"""02_tts.py - Self-clone TTS via faster-qwen3-tts ICL mode (issue #3).

For each sample i (1..10):
  text      = ASR(wav_i)        ← from 01_transcript/transcript.json
  ref_audio = data/data/audio/i.wav   ← the natural audio itself
  ref_text  = ASR(wav_i)        ← same as text (self-clone)
  language  = "Chinese"
  seed      = 42

Uses FasterQwen3TTS.generate_voice_clone with default ICL (x_vector_only=False).
Output: runs/<run_id>/02_tts/{1..10}.wav + tts_meta.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

FASTER_QWEN3_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
LANGUAGE = "Chinese"
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_transcripts(run_dir: Path, sample_ids: list[int]) -> dict[int, str]:
    """Load transcripts from 01_asr step for this run."""
    tjson = run_dir / "01_transcript" / "transcript.json"
    if tjson.exists():
        data = json.loads(tjson.read_text())
        return {r["sample_id"]: r["text"] for r in data.get("results", {}).values()}
    # Fallback: read individual .txt files
    texts: dict[int, str] = {}
    for i in sample_ids:
        txt = run_dir / "01_transcript" / f"{i}.txt"
        if txt.exists():
            texts[i] = txt.read_text().strip()
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-clone TTS generation")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    out_dir = run_dir / "02_tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [1] if args.smoke else list(range(1, 11))

    # Load transcripts from 01_asr
    transcripts = load_transcripts(run_dir, sample_ids)
    if not transcripts:
        print("[tts] ERROR: no transcripts found. Run 01_asr.py first.")
        raise SystemExit(1)
    print(f"[tts] loaded {len(transcripts)} transcripts")

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model (auto-downloads weights on first run via HF_ENDPOINT)
    print(f"[tts] loading model {FASTER_QWEN3_TTS_MODEL} ...")
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(FASTER_QWEN3_TTS_MODEL)
    print("[tts] model loaded")

    audio_dir = repo / "data" / "data" / "audio"
    results: dict[int, dict] = {}
    failed: list[dict] = []
    t0 = time.monotonic()

    for i in sample_ids:
        text = transcripts.get(i, "")
        if not text:
            failed.append({"sample_id": i, "error": "no transcript"})
            continue

        ref_audio_path = str(audio_dir / f"{i}.wav")
        ref_text = text  # self-clone: ref_text == target text

        print(f"[tts] generating sample {i} ...")
        last_err = None
        for attempt in range(2):  # 1 retry per plan (retry=1 → 2 total attempts)
            try:
                if attempt > 0:
                    time.sleep(1)
                audio_list, sr = model.generate_voice_clone(
                    text=text,
                    language=LANGUAGE,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                )
                # audio_list is list[np.ndarray]; sr is sample rate
                audio = audio_list[0] if isinstance(audio_list, list) else audio_list
                out_path = out_dir / f"{i}.wav"
                sf.write(str(out_path), audio, int(sr), subtype="PCM_16")
                dur = len(audio) / sr
                results[i] = {
                    "sample_id": i,
                    "duration_s": round(dur, 2),
                    "sample_rate": int(sr),
                    "text": text,
                }
                print(f"  -> {out_path} ({dur:.1f}s @ {sr}Hz)")
                break
            except Exception as e:
                last_err = str(e)
        else:
            failed.append({"sample_id": i, "error": last_err})
            print(f"  -> FAILED: {last_err}")

    elapsed = time.monotonic() - t0
    meta = {
        "model": FASTER_QWEN3_TTS_MODEL,
        "language": LANGUAGE,
        "clone_mode": "icl",
        "x_vector_only": False,
        "seed": SEED,
        "time_s": round(elapsed, 1),
        "samples_total": len(sample_ids),
        "samples_ok": len(results),
        "samples_failed": len(failed),
        "results": results,
        "failed": failed,
    }
    (out_dir / "tts_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[tts] {len(results)}/{len(sample_ids)} ok, {len(failed)} failed ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
