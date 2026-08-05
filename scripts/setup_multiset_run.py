#!/usr/bin/env python3
"""setup_multiset_run.py - Build runs/<run_id> inputs for the multiset pipeline check.

For each record in video_manifest.json (1..N):
  - extract mono 16k wav from mp4 (or reuse existing wav)  -> data/data/audio/{i}.wav
  - copy reference frame runs/multiset_mp/00_frames/{i}.png -> data/data/image/{i}.png
  - ASR transcript (wav2vec2-large-xlsr-53, local)          -> runs/<run_id>/01_transcript/transcript.json
Run inside /root/autodl-tmp/envs/ditto.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path("/root/autodl-tmp/repos/tts-exp")
FFMPEG = "/root/autodl-tmp/envs/ditto/bin/ffmpeg"
RUN_ID = "multiset_pipeline"
MANIFEST = REPO / "data/dataset_samples/video_manifest_250.json"
ASR_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
ASR_MODEL = "qwen3-asr-flash"
FRAMES_DIR = REPO / "runs/multiset_mp/00_frames"


def extract_audio(video_path: Path, out_path: Path) -> bool:
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", "16000", str(out_path)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def dashscope_asr(wav_path: Path, api_key: str, retries: int = 2) -> str:
    """Transcribe one 16k wav via qwen3-asr-flash (standard choices format)."""
    b64 = base64.b64encode(wav_path.read_bytes()).decode()
    data_uri = f"data:audio/wav;base64,{b64}"
    payload = {
        "model": ASR_MODEL,
        "input": {
            "messages": [{"role": "user", "content": [{"audio": data_uri}]}],
        },
        "parameters": {"asr_options": {"enable_itn": False}},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = ""
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(2)
            resp = requests.post(ASR_URL, headers=headers, json=payload, timeout=120)
            result = resp.json()
            if "output" in result and result["output"].get("choices"):
                text = result["output"]["choices"][0]["message"]["content"][0]["text"]
                if text and text.strip():
                    return text.strip()
            if result.get("output", {}).get("text"):
                return result["output"]["text"].strip()
            last_err = str(result)[:200]
        except Exception as e:
            last_err = str(e)
    print(f"[setup] ASR FAILED {wav_path.name}: {last_err}")
    return ""


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    records = manifest["records"]
    audio_dir = REPO / "data/data/audio"
    image_dir = REPO / "data/data/image"
    audio_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    tdir = REPO / f"runs/{RUN_ID}/01_transcript"
    tdir.mkdir(parents=True, exist_ok=True)

    audio_paths: list[Path] = []
    for i, rec in enumerate(records, 1):
        vid = REPO / rec["video_local_path"]
        audio_out = audio_dir / f"{i}.wav"
        if not audio_out.exists():
            rec_audio = rec.get("audio_local_path")
            if rec_audio:
                src = REPO / rec_audio
                shutil.copy(src, audio_out)
            else:
                if not extract_audio(vid, audio_out):
                    print(f"[setup] FAILED extract audio {i} {rec['stem']}")
                    sys.exit(1)
        audio_paths.append(audio_out)
        frame_src = FRAMES_DIR / f"{i}.png"
        if frame_src.exists() and not (image_dir / f"{i}.png").exists():
            shutil.copy(frame_src, image_dir / f"{i}.png")
        print(f"[setup] audio {i} <- {rec['stem']}")

    # ---- ASR via DashScope qwen3-asr-flash (cloud) ----
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("[setup] ERROR: DASHSCOPE_API_KEY env var required for ASR")
        sys.exit(1)
    print(f"[setup] ASR via {ASR_MODEL} (cloud) for all datasets — LRS3 uses ASR of the clip (official txt is sentence-level, longer than clip)")

    results = {}
    for i, wav_path in enumerate(audio_paths, 1):
        text = dashscope_asr(wav_path, api_key)
        print(f"[setup] asr {i}: {text[:80]}")
        results[str(i)] = {"sample_id": i, "text": text}

    (tdir / "transcript.json").write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=False))
    print(f"[setup] wrote transcript.json for {len(results)} samples")


if __name__ == "__main__":
    main()
