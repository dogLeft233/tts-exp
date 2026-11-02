#!/usr/bin/env python3
"""Qwen content guard for CPU-staged v4 dev waveforms (audio-only).

One new candidate ASR request is made for each of the 20 frozen dev clips.
The original-natural ASR result is reused only after exact PCM16 derivation is
verified. No video rows are fabricated, and this output is not a SyncNet or
merged candidate-family result. The API key is read only from the process
environment and is never serialized or logged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross-tfg-v4-audio-only-asr-results-v1"
STAGE_SCHEMA = "cross-tfg-v4-dev-audio-stage-v1"
TRANSCRIPT_SCHEMA = "cross-tfg-audio-head-aishell1-transcript-subset-v1"
NATURAL_ITEMS_SCHEMA = "cross-tfg-audio-head-asr-content-items-v1"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
MODEL = "qwen3-asr-flash"
FFMPEG_SHA256 = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
TRANSCRIPT_MANIFEST_SHA256 = "b0a08345a35f61a910a03ea5098b525541157c322b5faf69046c6227ec34803d"
NATURAL_INPUT_MANIFEST_SHA256 = "ac0651e070186d09ceca342ba10b19a8d5a1261a82e3d1865ea0a96b1d25cf50"
NATURAL_ITEMS_SHA256 = "076bcd497359da4a70c5a467a8b55e4b9c84466264bfd974e4499a6f90a52403"
NATURAL_RESULTS_SHA256 = "9c0a8cdb5cb65d4fc59a31b34ba54be93907f9af10208874a933f05a312fc5c9"
NATURAL_RESULT_MANIFEST_SHA256 = "33b042f6a72af447b170cfab21b95de79200f51e88ffe2a16623762cf9eaeb21"


class GuardError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical_json({k: v for k, v in value.items() if k != field}))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GuardError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise GuardError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json(dict(value)) + b"\n")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("xb") as f:
        for row in rows:
            f.write(canonical_json(dict(row)) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in normalized if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "Z")))


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    errors = edit_distance(ref, hyp)
    return {
        "reference_characters": len(ref), "hypothesis_characters": len(hyp),
        "edit_errors": errors, "cer": float(errors / len(ref)) if ref else None,
        "normalization": "NFKC+casefold; remove Unicode whitespace/separators and punctuation; character-level Levenshtein",
    }


def verify_float_wav(ref: Mapping[str, Any], where: str, wavfile: Any) -> tuple[Path, bytes, str]:
    path = Path(ref["path"]).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise GuardError(f"{where} is not a regular non-symlink WAV: {path}")
    raw = path.read_bytes()
    if sha_file(path) != ref.get("file_sha256"):
        raise GuardError(f"{where} file SHA mismatch")
    rate, samples = wavfile.read(str(path), mmap=False)
    import numpy as np
    if rate != 16000 or samples.ndim != 1 or samples.dtype != np.dtype("float32"):
        raise GuardError(f"{where} must be mono 16-kHz IEEE float32 WAV")
    pcm = np.ascontiguousarray(samples, dtype="<f4").tobytes(order="C")
    pcm_sha = sha_bytes(pcm)
    if len(samples) != ref.get("sample_count") or pcm_sha != ref.get("pcm_float32_sha256"):
        raise GuardError(f"{where} decoded float32 PCM/count mismatch")
    if not np.isfinite(samples).all():
        raise GuardError(f"{where} contains non-finite samples")
    return path, raw, pcm_sha


def convert_pcm16(ffmpeg: Path, source: Path, destination: Path, expected_samples: int, wavfile: Any) -> dict[str, Any]:
    import subprocess
    command = [
        str(ffmpeg), "-nostdin", "-n", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn",
        "-c:a", "pcm_s16le", "-f", "wav", "-ar", "16000", "-ac", "1", str(destination),
    ]
    cp = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0 or not destination.is_file():
        raise GuardError(f"FFmpeg PCM16 conversion failed (exit {cp.returncode}) for {source}")
    rate, samples = wavfile.read(str(destination), mmap=False)
    import numpy as np
    if rate != 16000 or samples.ndim != 1 or samples.dtype != np.dtype("int16") or samples.size != expected_samples:
        raise GuardError(f"PCM16 conversion format/count mismatch for {source}")
    if not np.isfinite(samples).all():
        raise GuardError(f"PCM16 conversion contains non-finite samples: {source}")
    pcm_bytes = np.ascontiguousarray(samples, dtype="<i2").tobytes(order="C")
    with wave.open(str(destination), "rb") as wf:
        if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()) != (16000, 1, 2, expected_samples):
            raise GuardError(f"PCM16 RIFF metadata mismatch: {source}")
    return {
        "path": str(destination), "file_sha256": sha_file(destination), "file_bytes": destination.stat().st_size,
        "pcm16_sha256": sha_bytes(pcm_bytes), "sample_count": int(samples.size), "sample_rate_hz": int(rate),
        "channels": 1, "encoding": "pcm_s16le", "ffmpeg_argv": command,
    }


def transcribe_once(audio_bytes: bytes, api_key: str, timeout_seconds: float) -> dict[str, Any]:
    import base64
    import requests
    data_uri = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")
    payload = {
        "model": MODEL,
        "input": {"messages": [{"role": "user", "content": [{"audio": data_uri}]}]},
        "parameters": {"asr_options": {"language": "zh", "enable_itn": False}},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout_seconds)
    except Exception as exc:
        return {"status": "FAILED_TRANSPORT", "failure_reason": type(exc).__name__, "http_status": None, "request_id": None}
    request_id = response.headers.get("X-DashScope-Request-Id")
    try:
        body = response.json()
    except Exception:
        body = None
    if response.status_code < 200 or response.status_code >= 300:
        code = body.get("code") if isinstance(body, dict) and isinstance(body.get("code"), str) else None
        return {"status": "FAILED_HTTP", "failure_reason": "non_2xx_response", "http_status": response.status_code, "api_error_code": code, "request_id": request_id}
    if not isinstance(body, dict):
        return {"status": "FAILED_RESPONSE", "failure_reason": "response_not_json_object", "http_status": response.status_code, "request_id": request_id}
    output = body.get("output")
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        code = body.get("code") if isinstance(body.get("code"), str) else None
        return {"status": "FAILED_RESPONSE", "failure_reason": "missing_output_choices", "http_status": response.status_code, "api_error_code": code, "request_id": request_id}
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    texts = [block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)] if isinstance(content, list) else ([content] if isinstance(content, str) else [])
    if not texts:
        return {"status": "FAILED_RESPONSE", "failure_reason": "no_text_transcription", "http_status": response.status_code, "request_id": request_id}
    return {
        "status": "PASS", "hypothesis": "".join(texts).strip(), "http_status": response.status_code,
        "request_id": request_id or body.get("request_id"), "usage": body.get("usage"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage_path = Path(args.staging_manifest).resolve(strict=True)
    if sha_file(stage_path) != args.expected_staging_manifest_sha256:
        raise GuardError("audio-stage manifest raw SHA mismatch")
    stage = load_json(stage_path)
    if stage.get("schema_version") != STAGE_SCHEMA or stage.get("status") != "PASS":
        raise GuardError("audio-stage manifest is not a complete PASS")
    if stage.get("manifest_sha256") != payload_sha(stage, "manifest_sha256"):
        raise GuardError("audio-stage manifest payload hash mismatch")
    if stage.get("confirmation_audio_read") is not False or stage.get("syncnet_called") is not False:
        raise GuardError("audio-stage scope flags are not closed")
    if stage.get("method_id") != "spectral_head_fit_v4_native_attempt01" or stage.get("checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise GuardError("audio-stage candidate/checkpoint pin differs")
    items = stage.get("rows")
    if not isinstance(items, list) or len(items) != 20:
        raise GuardError("audio-only family must contain exactly 20 staged dev clips")
    sample_ids = [str(item.get("sample_id")) for item in items]
    if len(set(sample_ids)) != 20 or any(item.get("split") != "dev" for item in items):
        raise GuardError("staged audio sample roster is duplicated or non-dev")

    transcript_manifest_path = Path(args.transcript_manifest).resolve(strict=True)
    if sha_file(transcript_manifest_path) != args.expected_transcript_manifest_sha256 or args.expected_transcript_manifest_sha256 != TRANSCRIPT_MANIFEST_SHA256:
        raise GuardError("official AISHELL transcript manifest raw SHA mismatch")
    transcript_manifest = load_json(transcript_manifest_path)
    if transcript_manifest.get("schema_version") != TRANSCRIPT_SCHEMA or transcript_manifest.get("status") != "PASS" or transcript_manifest.get("sample_count") != 20:
        raise GuardError("official AISHELL transcript subset is not qualified for exactly 20 clips")
    if transcript_manifest.get("manifest_sha256") != payload_sha(transcript_manifest, "manifest_sha256"):
        raise GuardError("official transcript manifest payload hash mismatch")
    if set(transcript_manifest.get("files", {})) != set(sample_ids):
        raise GuardError("official transcript IDs differ from the frozen dev roster")
    transcript_root = Path(args.transcripts_root).resolve(strict=True)
    transcript_by_sample: dict[str, dict[str, Any]] = {}
    for item in items:
        sample = str(item["sample_id"])
        ref = transcript_manifest["files"][sample]
        path = transcript_root / f"{sample}.txt"
        raw = path.read_bytes()
        if ref.get("path") != f"{sample}.txt" or sha_bytes(raw) != ref.get("raw_sha256"):
            raise GuardError(f"official transcript source bytes differ for {sample}")
        text = raw.decode("utf-8-sig").strip()
        if not normalize_text(text):
            raise GuardError(f"official transcript is empty after frozen normalization for {sample}")
        src_row = next((r for r in transcript_manifest.get("rows", []) if r.get("sample_id") == sample), None)
        if not isinstance(src_row, dict) or src_row.get("speaker_id") != item.get("speaker_id") or src_row.get("text_raw_sha256") != sha_bytes(raw):
            raise GuardError(f"official transcript speaker/source-line binding differs for {sample}")
        transcript_by_sample[sample] = {"text": text, "raw_sha256": sha_bytes(raw), "speaker_id": item["speaker_id"]}

    nat_manifest_path = Path(args.natural_input_manifest).resolve(strict=True)
    natural_items_path = Path(args.natural_items).resolve(strict=True)
    natural_results_path = Path(args.natural_results).resolve(strict=True)
    natural_result_manifest_path = Path(args.natural_result_manifest).resolve(strict=True)
    pinned_natural_paths = (
        (nat_manifest_path, args.expected_natural_input_manifest_sha256, NATURAL_INPUT_MANIFEST_SHA256),
        (natural_items_path, args.expected_natural_items_sha256, NATURAL_ITEMS_SHA256),
        (natural_results_path, args.expected_natural_results_sha256, NATURAL_RESULTS_SHA256),
        (natural_result_manifest_path, args.expected_natural_result_manifest_sha256, NATURAL_RESULT_MANIFEST_SHA256),
    )
    for path, expected, fixed in pinned_natural_paths:
        if sha_file(path) != expected or expected != fixed:
            raise GuardError(f"pinned natural ASR source SHA mismatch: {path.name}")
    natural_manifest = load_json(nat_manifest_path)
    natural_items = load_jsonl(natural_items_path)
    natural_results = load_jsonl(natural_results_path)
    natural_result_manifest = load_json(natural_result_manifest_path)
    if natural_manifest.get("schema_version") != NATURAL_ITEMS_SCHEMA or natural_manifest.get("manifest_sha256") != payload_sha(natural_manifest, "manifest_sha256"):
        raise GuardError("natural identity ASR item manifest schema/payload mismatch")
    if natural_result_manifest.get("status") != "PASS" or natural_result_manifest.get("manifest_sha256") != payload_sha(natural_result_manifest, "manifest_sha256"):
        raise GuardError("natural identity ASR result manifest is not PASS/sealed")
    natural_items_by_sample = {r.get("sample_id"): r for r in natural_items if r.get("method_id") == "identity"}
    natural_results_by_sample = {r.get("sample_id"): r for r in natural_results if r.get("method_id") == "identity"}
    if set(natural_items_by_sample) != set(sample_ids) or set(natural_results_by_sample) != set(sample_ids):
        raise GuardError("natural identity ASR rows do not exactly cover the 20 frozen clips")

    ffmpeg = Path(args.ffmpeg).resolve(strict=True)
    if sha_file(ffmpeg) != args.expected_ffmpeg_sha256 or args.expected_ffmpeg_sha256 != FFMPEG_SHA256:
        raise GuardError("FFmpeg binary pin differs from the already qualified direct PCM16 conversion")
    out = Path(args.out_dir).resolve()
    tmp = Path(args.tmp_dir).resolve()
    if out.exists() or tmp.exists() or out == tmp or out in tmp.parents or tmp in out.parents:
        raise GuardError("ASR output/temp paths must be distinct and fresh")
    out.mkdir(parents=True)
    pcm16_dir = out / "candidate_pcm16_wav"
    pcm16_dir.mkdir()

    import scipy
    from scipy.io import wavfile
    import requests  # preflight all API dependencies before conversion/calls
    del scipy, requests
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise GuardError("DASHSCOPE_API_KEY missing from private process environment")
    if not args.confirm_api_calls:
        raise GuardError("billable calls require the explicit --confirm-api-calls switch")

    # Verify all staged F32 audio and produce all PCM16 payloads before making a call.
    tmp.mkdir(parents=True)
    prepared: list[dict[str, Any]] = []
    for item in items:
        sample_id = str(item["sample_id"])
        natural_ref, candidate_ref = item.get("natural_pcm"), item.get("candidate_pcm")
        if not isinstance(natural_ref, dict) or not isinstance(candidate_ref, dict):
            raise GuardError(f"natural/candidate PCM bindings missing for {sample_id}")
        natural_path, _, natural_float_sha = verify_float_wav(natural_ref, f"{sample_id}.natural", wavfile)
        candidate_path, candidate_raw, candidate_float_sha = verify_float_wav(candidate_ref, f"{sample_id}.v4", wavfile)
        if natural_float_sha != natural_ref.get("pcm_float32_sha256") or candidate_float_sha != candidate_ref.get("pcm_float32_sha256"):
            raise GuardError(f"staged audio canonical SHA mismatch for {sample_id}")
        if candidate_ref.get("sample_count") != natural_ref.get("sample_count"):
            raise GuardError(f"v4 candidate changed sample count for {sample_id}")
        identity_item = natural_items_by_sample[sample_id]
        identity_asr = natural_results_by_sample[sample_id]
        if identity_item.get("natural_pcm", {}).get("pcm_sha256") != natural_float_sha:
            raise GuardError(f"reused natural identity item is for different PCM: {sample_id}")
        if identity_asr.get("asr_status") != "PASS" or not isinstance(identity_asr.get("hypothesis"), str) or not identity_asr["hypothesis"].strip():
            raise GuardError(f"natural identity ASR hypothesis unavailable for {sample_id}")
        natural_conversion = convert_pcm16(ffmpeg, natural_path, tmp / f"natural_{sample_id}.wav", int(natural_ref["sample_count"]), wavfile)
        natural_asr_input = identity_asr.get("input_pcm16_sha256")
        if natural_conversion["pcm16_sha256"] != natural_asr_input:
            raise GuardError(f"natural identity ASR direct PCM16 input does not match: {sample_id}")
        candidate_conversion = convert_pcm16(ffmpeg, candidate_path, pcm16_dir / f"{sample_id}.v4.pcm16.wav", int(candidate_ref["sample_count"]), wavfile)
        pcm16_bytes = (pcm16_dir / f"{sample_id}.v4.pcm16.wav").read_bytes()
        if len(pcm16_bytes) > args.max_input_wav_bytes or float(candidate_ref["sample_count"]) / 16000.0 > args.max_duration_seconds:
            raise GuardError(f"candidate ASR audio exceeds frozen request limits: {sample_id}")
        prepared.append({
            "sample_id": sample_id, "speaker_id": item["speaker_id"], "split": "dev",
            "natural_pcm": dict(natural_ref), "candidate_pcm": dict(candidate_ref),
            "natural_conversion": natural_conversion, "candidate_conversion": candidate_conversion,
            "candidate_pcm16_path": str(pcm16_dir / f"{sample_id}.v4.pcm16.wav"),
            "candidate_pcm16_bytes": pcm16_bytes,
            "natural_asr": identity_asr,
            "reference": transcript_by_sample[sample_id],
        })
    if len(prepared) != 20 or len({r["candidate_conversion"]["pcm16_sha256"] for r in prepared}) == 0:
        raise GuardError("audio-only ASR preflight did not prepare all 20 candidate inputs")

    # A single call per candidate/clip. Errors remain rows; no silent retry or drop.
    asr_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for index, item in enumerate(prepared):
        api_result = transcribe_once(item["candidate_pcm16_bytes"], api_key, args.timeout_seconds)
        success = api_result.get("status") == "PASS" and bool(api_result.get("hypothesis"))
        common = {
            "item_id": f"{item['sample_id']}__spectral_head_fit_v4_native_attempt01",
            "method_id": "spectral_head_fit_v4_native_attempt01",
            "sample_id": item["sample_id"], "speaker_id": item["speaker_id"], "split": "dev",
            "status": "READY" if success else "FAILED",
            "asr_status": api_result.get("status"), "api_call_made": True,
            "api_endpoint": API_URL, "asr_model": MODEL,
            "api_error_code": api_result.get("api_error_code"),
            "failure_reason": api_result.get("failure_reason"),
            "http_status": api_result.get("http_status"), "request_id": api_result.get("request_id"),
            "usage": api_result.get("usage"),
            "input_pcm16_file_sha256": item["candidate_conversion"]["file_sha256"],
            "input_pcm16_sha256": item["candidate_conversion"]["pcm16_sha256"],
            "input_pcm16_sample_count": item["candidate_conversion"]["sample_count"],
            "candidate_pcm_float32_sha256": item["candidate_pcm"]["pcm_float32_sha256"],
            "natural_pcm_float32_sha256": item["natural_pcm"]["pcm_float32_sha256"],
            "natural_asr_input_pcm16_sha256": item["natural_asr"].get("input_pcm16_sha256"),
            "transcription_reused_from_audio_sha256": None,
            "reference_transcripts_sent_to_provider": False,
        }
        if success:
            common["hypothesis"] = api_result["hypothesis"]
        asr_rows.append(common)
        if success:
            ref_text = item["reference"]["text"]
            score_rows.append({
                "item_id": common["item_id"], "method_id": common["method_id"],
                "sample_id": item["sample_id"], "speaker_id": item["speaker_id"], "split": "dev",
                "status": "PASS", "planned_in_denominator": True,
                "audio_pcm_sha256": item["candidate_pcm"]["pcm_float32_sha256"],
                "reference_status": "AVAILABLE", "transcript_raw_sha256": item["reference"]["raw_sha256"],
                "cer_vs_reference": cer(ref_text, api_result["hypothesis"]),
                "asr_disagreement_vs_natural": cer(item["natural_asr"]["hypothesis"], api_result["hypothesis"]),
                "natural_asr_baseline_item_id": item["natural_asr"].get("item_id"),
                "natural_asr_baseline_source_results_sha256": args.expected_natural_results_sha256,
                "natural_asr_baseline_input_pcm16_sha256": item["natural_asr"].get("input_pcm16_sha256"),
            })
        else:
            score_rows.append({
                "item_id": common["item_id"], "method_id": common["method_id"],
                "sample_id": item["sample_id"], "speaker_id": item["speaker_id"], "split": "dev",
                "status": "FAILED", "planned_in_denominator": True,
                "audio_pcm_sha256": item["candidate_pcm"]["pcm_float32_sha256"],
                "reference_status": "AVAILABLE", "failure_reason": api_result.get("failure_reason") or api_result.get("status"),
                "transcript_raw_sha256": item["reference"]["raw_sha256"],
            })
        if index + 1 < len(prepared):
            time.sleep(args.delay_seconds)

    write_jsonl(out / "asr_results.jsonl", asr_rows)
    write_jsonl(out / "content_scores.jsonl", score_rows)
    successes = [r for r in score_rows if r.get("status") == "PASS"]
    errors_by_speaker: dict[str, list[float]] = defaultdict(list)
    disagreement_by_speaker: dict[str, list[float]] = defaultdict(list)
    for result in successes:
        cer_value = result.get("cer_vs_reference", {}).get("cer")
        disagreement = result.get("asr_disagreement_vs_natural", {}).get("cer")
        if cer_value is not None:
            errors_by_speaker[result["speaker_id"]].append(float(cer_value))
        if disagreement is not None:
            disagreement_by_speaker[result["speaker_id"]].append(float(disagreement))
    speaker_cer = {speaker: sum(values) / len(values) for speaker, values in sorted(errors_by_speaker.items()) if values}
    speaker_disagreement = {speaker: sum(values) / len(values) for speaker, values in sorted(disagreement_by_speaker.items()) if values}
    summary = {
        "schema_version": "cross-tfg-v4-audio-only-asr-summary-v1",
        "status": "PASS" if len(successes) == 20 else "COMPLETE_WITH_FAILURES",
        "split": "dev", "method_id": "spectral_head_fit_v4_native_attempt01",
        "candidate_calls_planned": 20, "candidate_calls_made": sum(r.get("api_call_made") is True for r in asr_rows),
        "candidate_calls_succeeded": len(successes), "candidate_calls_failed": 20 - len(successes),
        "planned_clips": 20, "planned_speakers": 10,
        "estimable_clips": len(successes), "estimable_speakers": len(speaker_cer),
        "equal_speaker_mean_cer_vs_official_reference": sum(speaker_cer.values()) / len(speaker_cer) if speaker_cer else None,
        "equal_speaker_mean_asr_disagreement_vs_natural": sum(speaker_disagreement.values()) / len(speaker_disagreement) if speaker_disagreement else None,
        "speaker_mean_cer": speaker_cer, "speaker_mean_asr_disagreement": speaker_disagreement,
        "interpretation": "development descriptive only; ASR agreement is not a perceptual intelligibility claim or selection criterion",
        "api": {"provider": "Alibaba Cloud DashScope", "model": MODEL, "endpoint": API_URL, "language": "zh", "enable_itn": False},
        "confirmation_audio_read": False, "video_or_syncnet_used": False,
        "reference_transcripts_sent_to_provider": False,
        "natural_asr_reuse": {"source_raw_sha256": args.expected_natural_results_sha256, "candidate_api_calls_repeated": 0},
    }
    write_json(out / "content_summary.json", summary)
    manifest: dict[str, Any] = {
        "schema_version": "cross-tfg-v4-audio-only-asr-manifest-v1",
        "status": summary["status"], "source_staging_manifest_raw_sha256": args.expected_staging_manifest_sha256,
        "source_staging_manifest_payload_sha256": stage["manifest_sha256"],
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "transcript_manifest_raw_sha256": args.expected_transcript_manifest_sha256,
        "natural_asr_source_raw_sha256": args.expected_natural_results_sha256,
        "ffmpeg_sha256": args.expected_ffmpeg_sha256,
        "planned_items": 20, "asr_result_rows": len(asr_rows), "score_rows": len(score_rows),
        "status_counts": dict(Counter(r.get("status", "UNKNOWN") for r in score_rows)),
        "asr_results_raw_sha256": sha_file(out / "asr_results.jsonl"),
        "content_scores_raw_sha256": sha_file(out / "content_scores.jsonl"),
        "content_summary_raw_sha256": sha_file(out / "content_summary.json"),
        "confirmation_audio_read": False, "video_or_syncnet_used": False,
        "reference_transcripts_sent_to_provider": False,
    }
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    write_json(out / "content_result_manifest.json", manifest)
    return {
        "status": summary["status"], "out_dir": str(out), "planned_items": 20,
        "candidate_calls_made": summary["candidate_calls_made"], "candidate_calls_succeeded": len(successes),
        "candidate_calls_failed": 20 - len(successes), "manifest_raw_sha256": sha_file(out / "content_result_manifest.json"),
        "manifest_payload_sha256": manifest["manifest_sha256"], "confirmation_audio_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-manifest", required=True)
    parser.add_argument("--expected-staging-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--transcript-manifest", required=True)
    parser.add_argument("--expected-transcript-manifest-sha256", required=True)
    parser.add_argument("--transcripts-root", required=True)
    parser.add_argument("--natural-input-manifest", required=True)
    parser.add_argument("--expected-natural-input-manifest-sha256", required=True)
    parser.add_argument("--natural-items", required=True)
    parser.add_argument("--expected-natural-items-sha256", required=True)
    parser.add_argument("--natural-results", required=True)
    parser.add_argument("--expected-natural-results-sha256", required=True)
    parser.add_argument("--natural-result-manifest", required=True)
    parser.add_argument("--expected-natural-result-manifest-sha256", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--expected-ffmpeg-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--max-input-wav-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--max-duration-seconds", type=float, default=8.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--confirm-api-calls", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
