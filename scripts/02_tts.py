#!/usr/bin/env python3
"""02_tts.py - Self-clone TTS via pluggable provider (issue #3).

For each sample i (1..10):
  text      = ASR(wav_i)        ← from 01_transcript/transcript.json
  ref_audio = data/data/audio/i.wav   ← the natural audio itself
  ref_text  = ASR(wav_i)        ← same as text (self-clone)
  language  = "Chinese"

Backend is selected via cfg["tts"]["provider"]:
  - "faster_qwen3"  → local Qwen3-TTS-0.6B-Base ICL mode (former default)
  - "dashscope_vc"  → DashScope cloud qwen3-tts-vc-2026-01-22 (register+synthesize)

See scripts/tts/ for the provider implementations.

Output: runs/<run_id>/02_tts/{1..10}.wav + tts_meta.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(repo: Path) -> dict:
    cfg_path = repo / "scripts" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}


def load_transcripts(run_dir: Path, sample_ids: list[int]) -> dict[int, str]:
    tjson = run_dir / "01_transcript" / "transcript.json"
    if tjson.exists():
        data = json.loads(tjson.read_text())
        return {r["sample_id"]: r["text"] for r in data.get("results", {}).values()}
    texts: dict[int, str] = {}
    for i in sample_ids:
        txt = run_dir / "01_transcript" / f"{i}.txt"
        if txt.exists():
            texts[i] = txt.read_text().strip()
    return texts


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override dict into base dict recursively in place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-clone TTS generation")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--config",
        default="",
        help="Optional config override (path relative to repo root, e.g. scripts/configs/r1_dashscope_flash.yaml). "
             "Merged on top of default scripts/config.yaml.",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    out_dir = run_dir / "02_tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(repo)
    if args.config:
        override_path = repo / args.config
        if not override_path.exists():
            print(f"[tts] ERROR: override config not found: {override_path}")
            raise SystemExit(1)
        override = yaml.safe_load(override_path.read_text()) or {}
        _deep_merge(cfg, override)
        print(f"[tts] loaded config override: {args.config}")

    tts_cfg = cfg.get("tts", {})
    seed = int(tts_cfg.get("seed", 42))
    language = tts_cfg.get("language", "Chinese")
    retries = int(tts_cfg.get("retry", 1))

    from utils import detect_sample_ids
    sample_ids = detect_sample_ids(repo, args.smoke)
    transcripts = load_transcripts(run_dir, sample_ids)
    if not transcripts:
        print("[tts] ERROR: no transcripts found. Run 01_asr.py first.")
        raise SystemExit(1)
    print(f"[tts] loaded {len(transcripts)} transcripts")

    # Set global seed (torch.cuda / numpy). Providers may read torch RNG state.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build provider via factory
    from tts import get_tts_provider
    provider = get_tts_provider(
        tts_cfg,
        run_id=args.run_id,
        repo_root=repo,
        env=dict(os.environ),
    )
    provider_name = provider.name
    print(f"[tts] provider: {provider_name}")

    audio_dir = repo / "data" / "data" / "audio"
    results: dict[int, dict] = {}
    failed: list[dict] = []
    t0 = time.monotonic()

    # Voice registration hint: DashScope VC takes sample_id for caching; others ignore it.
    def _call_provider(text: str, ref_audio: Path, i: int):
        # FasterQwen3 respects ref_text; DashScope VC ignores it.
        kwargs = {}
        # DashScope provider supports sample_id for cache lookup via __init__-mandated signature
        sig_extra = {"sample_id": i}
        try:
            return provider.generate_voice_clone(
                text=text,
                ref_audio_path=ref_audio,
                ref_text=text,  # self-clone: ref_text == target text
                language=language,
                **sig_extra,
            )
        except TypeError:
            # Provider doesn't accept sample_id — drop it
            return provider.generate_voice_clone(
                text=text,
                ref_audio_path=ref_audio,
                ref_text=text,
                language=language,
            )

    for i in sample_ids:
        text = transcripts.get(i, "")
        if not text:
            failed.append({"sample_id": i, "error": "no transcript"})
            continue

        ref_audio_path = audio_dir / f"{i}.wav"
        print(f"[tts] generating sample {i} ...")
        last_err = None
        success = False
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(1)
                result = _call_provider(text, ref_audio_path, i)
                out_path = out_dir / f"{i}.wav"
                # Save PCM_16 mono — force mono to keep shape (N,)
                audio = result.audio
                if audio.ndim > 1:
                    audio = audio.mean(axis=tuple(range(1, audio.ndim)))
                # Save at the provider's native rate; downstream resamp step normalizes.
                sf.write(str(out_path), audio, int(result.sample_rate), subtype="PCM_16")
                results[i] = {
                    "sample_id": i,
                    "duration_s": result.duration_s,
                    "sample_rate": int(result.sample_rate),
                    "text": text,
                    "backend": result.backend_meta.get("backend", provider_name),
                    "model": result.backend_meta.get("model") or result.backend_meta.get("model_id"),
                    "voice_id": result.backend_meta.get("voice_id"),
                }
                print(f"  -> {out_path} ({result.duration_s:.1f}s @ {result.sample_rate}Hz)")
                success = True
                break
            except Exception as e:
                last_err = str(e)
        if not success:
            failed.append({"sample_id": i, "error": last_err})
            print(f"  -> FAILED: {last_err}")

    elapsed = time.monotonic() - t0
    meta = {
        "provider": provider_name,
        "language": language,
        "seed": seed,
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