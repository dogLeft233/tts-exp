#!/usr/bin/env python3
"""01_asr.py - Transcribe AISHELL-1 audio via Qwen ASR Flash (DashScope) (issue #2).

Uses the multimodal-generation API endpoint with base64-encoded local wav files.
Retries once on transient failure (per plan Q21/B).
API key is read from DASHSCOPE_API_KEY env var (loaded from scripts/.env via python-dotenv).
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests


API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
MODEL = "qwen3-asr-flash"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(repo: Path) -> dict:
    """Load config.yaml (for model name fallback, retry count, temperature)."""
    import yaml

    cfg_path = repo / "scripts" / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text()) or {}
    return {}


def load_api_key(repo: Path) -> str:
    """Try python-dotenv first, then fall back to env var, then config."""
    try:
        from dotenv import load_dotenv

        load_dotenv(repo / "scripts" / ".env")
    except Exception:
        pass
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if key:
        return key
    cfg = load_config(repo)
    return cfg.get("asr", {}).get("api_key_env", "")


def transcribe(audio_path: Path, api_key: str, retries: int = 1) -> dict | None:
    """Transcribe one wav file. Returns dict with text + usage, or None on failure."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    data_uri = f"data:audio/wav;base64,{b64}"

    payload = {
        "model": MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"audio": data_uri}],
                }
            ],
        },
        "parameters": {"asr_options": {"enable_itn": False}},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(2)
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            result = resp.json()
            if "output" in result and result["output"].get("choices"):
                choice = result["output"]["choices"][0]
                text = choice["message"]["content"][0]["text"]
                usage = result.get("usage", {})
                return {"sample_id": 0, "text": text, "usage": usage}
            last_err = result
        except Exception as e:
            last_err = {"error": str(e)}
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe audio via Qwen ASR Flash")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "runs" / args.run_id / "01_transcript"
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key(repo)
    if not api_key:
        print(
            "[asr] ERROR: DASHSCOPE_API_KEY not found. "
            "Set it in scripts/.env or env var."
        )
        raise SystemExit(1)

    sample_ids = [1] if args.smoke else list(range(1, 11))
    audio_dir = repo / "data" / "data" / "audio"

    results: dict[int, dict] = {}
    failed: list[dict] = []

    for i in sample_ids:
        p = audio_dir / f"{i}.wav"
        print(f"[asr] transcribing sample {i} ...")
        r = transcribe(p, api_key, retries=1)
        if r is not None:
            text = r["text"]
            results[i] = {"sample_id": i, "text": text, "usage": r["usage"]}
            (out_dir / f"{i}.txt").write_text(text + "\n", encoding="utf-8")
            print(f"  -> {text}")
        else:
            failed.append({"sample_id": i, "error": "transcribe failed"})
            print(f"  -> FAILED")

    summary = {
        "model": MODEL,
        "samples_total": len(sample_ids),
        "samples_ok": len(results),
        "samples_failed": len(failed),
        "results": results,
        "failed": failed,
    }
    (out_dir / "transcript.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[asr] {len(results)}/{len(sample_ids)} ok, {len(failed)} failed")


if __name__ == "__main__":
    main()
