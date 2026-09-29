"""Fixed faster-whisper content audit for every audio arm."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import proxy_environment


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(value).lower())


def word_edit_distance(reference: str, hypothesis: str) -> dict[str, Any]:
    ref = _words(reference)
    hyp = _words(hypothesis)
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + int(ref[i - 1] != hyp[j - 1]))
    return {"reference_words": len(ref), "hypothesis_words": len(hyp), "edit_distance": int(dp[-1][-1]), "wer": None if not ref else float(dp[-1][-1] / len(ref))}


def transcribe(path: str | Path, config: Mapping[str, Any], *, model: Any | None = None) -> dict[str, Any]:
    audio = Path(path).resolve()
    quality = config.get("quality", {})
    if not audio.is_file():
        raise FileNotFoundError(audio)
    try:
        if model is None:
            model = _load_model(config)
        segments, info = model.transcribe(str(audio), language=str(quality.get("asr_language", "en")), beam_size=int(quality.get("asr_beam_size", 5)), temperature=float(quality.get("asr_temperature", 0.0)), vad_filter=bool(quality.get("asr_vad_filter", False)), condition_on_previous_text=bool(quality.get("asr_condition_on_previous_text", False)))
        text = " ".join(str(segment.text).strip() for segment in segments).strip()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("ASR_BACKEND_MISSING:faster_whisper") from exc
    return {"status": "COMPLETE", "audio": str(audio), "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "model": str(quality.get("asr_model", "base.en")), "language": str(quality.get("asr_language", "en")), "text": text, "language_probability": getattr(info, "language_probability", None)}


def _load_model(config: Mapping[str, Any]) -> Any:
    from faster_whisper import WhisperModel

    quality = config.get("quality", {})
    device = str(quality.get("asr_device", "cpu"))
    compute_type = str(quality.get("asr_compute_type", "int8" if device == "cpu" else "float16"))
    with proxy_environment(str(config.get("runtime", {}).get("proxy", "")) or None):
        return WhisperModel(str(quality.get("asr_model", "base.en")), device=device, compute_type=compute_type, download_root=quality.get("asr_download_root"))


def batch_content_audit(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        model = _load_model(config)
        results: list[dict[str, Any]] = []
        for row in rows:
            result = transcribe(row["path"], config, model=model)
            reference = str(row.get("reference", ""))
            result.update({"pair_id": row.get("pair_id"), "source_group": row.get("source_group"), "arm": row.get("arm"), "reference": reference, **word_edit_distance(reference, result["text"])})
            results.append(result)
        groups: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            groups.setdefault(str(result.get("source_group", "")), []).append(result)
        group_wer = {}
        for group, values in groups.items():
            ref_words = sum(int(value["reference_words"]) for value in values)
            edits = sum(int(value["edit_distance"]) for value in values)
            group_wer[group] = {"edit_distance": edits, "reference_words": ref_words, "wer": None if ref_words == 0 else float(edits / ref_words)}
        return {"status": "COMPLETE", "rows": results, "group_wer": group_wer}
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"status": "DEPENDENCY_BLOCKED", "reason": str(exc), "rows": []}


__all__ = ["batch_content_audit", "transcribe", "word_edit_distance"]
