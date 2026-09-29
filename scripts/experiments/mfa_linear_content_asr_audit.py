#!/usr/bin/env python3
"""Independent frozen Whisper-base transcription screening on the n=10 cohort."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel
from opencc import OpenCC

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "runs/aishell1_qwen_mfa_linear_n100_20260816"
PRIOR = REPO / "runs/mfa_linear_vocoder_wav2lip_split_20260925"
OUT = REPO / "runs/mfa_linear_content_pause_prosody_mapping_20260925"
TRAD_TO_SIMP = OpenCC("t2s")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: str) -> str:
    value = TRAD_TO_SIMP.convert(unicodedata.normalize("NFKC", value).lower())
    return "".join(c for c in value if not c.isspace() and not unicodedata.category(c).startswith("P"))


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "small"), default="base")
    model_name = parser.parse_args().model
    rows = read(PRIOR / "manifest.json")["rows"]
    meta = read(COHORT / "01_tts_retry/tts_meta.json")["results"]
    previous_path = OUT / ("content_asr_audit.json" if model_name == "base" else "content_asr_audit_small.json")
    previous = {r["id"]: r for r in read(previous_path)["rows"]} if previous_path.is_file() else {}
    model_dir = OUT / ("asr_model" if model_name == "base" else "asr_model_small")
    model = None if len(previous) == len(rows) else WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    output = []
    for row in rows:
        sid = row["id"]
        truth = normalize(meta[sid]["transcript"])
        arms = {"N": row["natural"], "TTS": meta[sid]["canonical_16k_audio"], "M": row["mfa"]}
        result = {"id": sid, "reference": truth, "arms": {}}
        for arm, path in arms.items():
            if sid in previous and arm in previous[sid]["arms"]:
                raw = previous[sid]["arms"][arm]["raw"]
            else:
                segments, _ = model.transcribe(path, language="zh", beam_size=5,
                                               vad_filter=False, condition_on_previous_text=False)
                raw = "".join(segment.text for segment in segments)
            normalized = normalize(raw)
            result["arms"][arm] = {"raw": raw, "normalized": normalized,
                                   "edit_distance": edit_distance(truth, normalized),
                                   "cer": edit_distance(truth, normalized) / max(1, len(truth))}
        output.append(result)
        payload = {"protocol": f"faster-whisper {model_name}, CPU int8, Mandarin forced, beam=5, no transcript prompt; OpenCC traditional-to-simplified normalization; ASR screening only, model errors possible", "rows": output}
        OUT.mkdir(parents=True, exist_ok=True)
        previous_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("asr", sid, {arm: result["arms"][arm]["cer"] for arm in arms}, flush=True)


if __name__ == "__main__":
    main()
