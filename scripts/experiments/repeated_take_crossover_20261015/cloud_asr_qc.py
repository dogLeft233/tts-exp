#!/usr/bin/env python3
"""Bounded DashScope ASR content audit for the frozen 32-waveform cohort.

This diagnostic transcribes each canonical mono PCM16/16 kHz WAV once with
qwen-audio-3.0-asr-flash. It preserves the returned text and computes a
punctuation/whitespace-normalized CER against the speaker's manifest_v3 target.
It does not select, alter, or regenerate audio.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import platform
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path, PurePosixPath
from typing import Any

import requests


REPO = Path(__file__).resolve().parents[3]
REMOTE_REPO_PREFIX = "/home/wjj/tts-audio/tts-exp/"
MODEL = "qwen-audio-3.0-asr-flash"
ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
SPEAKERS = {"S0003", "S0006", "S0007", "S0008", "S0014", "S0015", "S0016", "S0020"}
ROLES = ("N1", "N2", "T1", "T2")
ROLE_ORDER = {role: index for index, role in enumerate(ROLES)}
TRANSPORT_TIMEOUT = (15, 50)
MAX_ROWS = 32


class InputError(RuntimeError):
    """Raised when frozen inputs do not satisfy the audio contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise InputError(f"input JSON is missing or unsafe: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"input JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise InputError(f"input JSON must contain an object: {path}")
    return value, bytes_sha256(raw)


def _inside_repo(path: Path, repo_root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repo_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise InputError("canonical audio path does not resolve inside the repository") from exc
    if not resolved.is_file():
        raise InputError("canonical audio path is not a regular file")
    return resolved


def map_canonical_path(value: Any, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise InputError("canonical_path is missing")
    if value.startswith(REMOTE_REPO_PREFIX):
        rel = PurePosixPath(value[len(REMOTE_REPO_PREFIX):])
        if rel.is_absolute() or not rel.parts or ".." in rel.parts:
            raise InputError("remote canonical_path is not a safe repository-relative path")
        candidate = repo_root.joinpath(*rel.parts)
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
    return _inside_repo(candidate, repo_root)


def read_pcm16_mono_16k(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
            pcm = stream.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise InputError("canonical WAV cannot be decoded") from exc
    if (channels != 1 or width != 2 or sample_rate != 16000 or
            frame_count < 1 or compression != "NONE" or len(pcm) != frame_count * 2):
        raise InputError("canonical WAV violates mono PCM16/16 kHz contract")
    return pcm, {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": width,
        "samples": frame_count,
        "duration_s": frame_count / sample_rate,
        "decoded_pcm_sha256": bytes_sha256(pcm),
    }


def _sha_field(record: dict[str, Any], key: str, expected: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise InputError(f"audio manifest has invalid {key}")
    if value != expected:
        raise InputError(f"audio manifest {key} does not match the canonical WAV")


def load_inputs(audio_manifest_path: Path, transcript_manifest_path: Path,
                repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    audio_manifest, audio_manifest_sha = read_json(audio_manifest_path)
    transcript_manifest, transcript_manifest_sha = read_json(transcript_manifest_path)
    records = audio_manifest.get("records")
    speaker_rows = transcript_manifest.get("speakers")
    if not isinstance(records, list) or len(records) != MAX_ROWS:
        raise InputError("audio_manifest must contain exactly 32 records")
    if not isinstance(speaker_rows, list):
        raise InputError("manifest_v3 has no speakers list")

    references: dict[str, str] = {}
    for row in speaker_rows:
        if not isinstance(row, dict):
            raise InputError("manifest_v3 speaker entry is invalid")
        speaker = row.get("speaker")
        transcript = row.get("target_transcript")
        if (not isinstance(speaker, str) or not isinstance(transcript, str) or
                not transcript.strip() or speaker in references):
            raise InputError("manifest_v3 speaker IDs or target transcripts are invalid/duplicated")
        references[speaker] = transcript
    if set(references) != SPEAKERS:
        raise InputError("manifest_v3 speaker set differs from the frozen 8-speaker cohort")

    seen: set[tuple[str, str]] = set()
    prepared: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise InputError("audio_manifest record is invalid")
        speaker, role = record.get("speaker"), record.get("role")
        if (not isinstance(speaker, str) or not isinstance(role, str) or
                speaker not in SPEAKERS or role not in ROLE_ORDER or
                (speaker, role) in seen):
            raise InputError("audio_manifest has an unexpected or duplicate speaker/role")
        seen.add((speaker, role))
        path = map_canonical_path(record.get("canonical_path"), repo_root)
        actual_file_sha = file_sha256(path)
        _sha_field(record, "canonical_sha256", actual_file_sha)
        pcm, metadata = read_pcm16_mono_16k(path)
        _sha_field(record, "canonical_pcm_sha256", metadata["decoded_pcm_sha256"])
        for key in ("sample_rate_hz", "channels", "sample_width_bytes", "samples"):
            if record.get(key) != metadata[key]:
                raise InputError(f"audio_manifest {key} does not match {speaker}/{role}")
        prepared.append({
            "speaker": speaker,
            "role": role,
            "path": path,
            "canonical_sha256": actual_file_sha,
            "canonical_pcm_sha256": metadata["decoded_pcm_sha256"],
            "samples": metadata["samples"],
            "duration_s": metadata["duration_s"],
            "reference_transcript": references[speaker],
            "audio_bytes": path.read_bytes(),
        })

    expected_pairs = {(speaker, role) for speaker in SPEAKERS for role in ROLES}
    if seen != expected_pairs:
        raise InputError("audio_manifest does not contain exactly N1/N2/T1/T2 for every speaker")
    prepared.sort(key=lambda row: (row["speaker"], ROLE_ORDER[row["role"]]))
    return prepared, {"audio_manifest_sha256": audio_manifest_sha,
                      "transcript_manifest_sha256": transcript_manifest_sha}, references


def normalize_transcript(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in normalized
                   if not char.isspace() and not unicodedata.category(char).startswith("P"))


def edit_distance(reference: str, hypothesis: str) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, row[j] + 1,
                               row[j - 1] + (ref_char != hyp_char)))
        row = current
    return row[-1]


def _safe_ascii_code(value: Any, fallback: str) -> str:
    if not isinstance(value, (str, int)):
        return fallback
    code = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value))[:96]
    return code or fallback


def _request_id(response: requests.Response, body: dict[str, Any]) -> str | None:
    value = (response.headers.get("X-Request-Id") or
             response.headers.get("X-DashScope-Request-Id") or
             body.get("request_id"))
    return _safe_ascii_code(value, "") or None


def _coarse_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {}
    allowed = {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"}
    result = {}
    for key in allowed:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
    return result


def _body_error_code(body: dict[str, Any], fallback: str) -> str:
    return _safe_ascii_code(body.get("code", body.get("error_code")), fallback)


def transcribe_one(row: dict[str, Any], api_key: str,
                   retry_state: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    encoded = base64.b64encode(row["audio_bytes"]).decode("ascii")
    payload = {
        "model": MODEL,
        "input": {"messages": [{"role": "user", "content": [{
            "type": "input_audio",
            "input_audio": {"data": "data:audio/wav;base64," + encoded},
        }]}]},
        "parameters": {"format": "wav", "sample_rate": "16000"},
    }
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }
    retries_for_row = 0
    while True:
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload,
                                     timeout=TRANSPORT_TIMEOUT)
        except requests.RequestException:
            if not retry_state["transport_retry_used"]:
                retry_state["transport_retry_used"] = True
                retry_state["transport_retry_for"] = row["speaker"] + "/" + row["role"]
                retries_for_row += 1
                continue
            return ({**_row_identity(row), "status": "transport_error",
                     "model": MODEL, "error_code": "transport_error",
                     "transport_retries_for_row": retries_for_row}, False)

        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code < 200 or response.status_code >= 300:
            result = {**_row_identity(row), "status": "api_error", "model": MODEL,
                      "http_status": int(response.status_code),
                      "error_code": _body_error_code(body, f"http_{response.status_code}") if isinstance(body, dict) else f"http_{response.status_code}",
                      "request_id": _request_id(response, body) if isinstance(body, dict) else None,
                      "transport_retries_for_row": retries_for_row}
            return result, False
        if not isinstance(body, dict):
            return ({**_row_identity(row), "status": "schema_error", "model": MODEL,
                     "http_status": int(response.status_code), "error_code": "invalid_json",
                     "request_id": _request_id(response, {}),
                     "transport_retries_for_row": retries_for_row}, False)
        output = body.get("output")
        text = output.get("text") if isinstance(output, dict) else None
        if not isinstance(text, str):
            return ({**_row_identity(row), "status": "schema_error", "model": MODEL,
                     "http_status": int(response.status_code),
                     "error_code": _body_error_code(body, "missing_output_text"),
                     "request_id": _request_id(response, body),
                     "transport_retries_for_row": retries_for_row}, False)

        reference = row["reference_transcript"]
        normalized_reference = normalize_transcript(reference)
        normalized_hypothesis = normalize_transcript(text)
        distance = edit_distance(normalized_reference, normalized_hypothesis)
        numeric_chars = set("0123456789０１２３４５６７８９零〇一二两三四五六七八九十百千万亿廿卅")
        has_numerals = any(char in numeric_chars for char in reference + text)
        return ({
            **_row_identity(row),
            "status": "ok",
            "http_status": int(response.status_code),
            "request_id": _request_id(response, body),
            "model": MODEL,
            "reference_transcript": reference,
            "hypothesis_raw": text,
            "reference_normalized": normalized_reference,
            "hypothesis_normalized": normalized_hypothesis,
            "char_edit_distance": distance,
            "cer": distance / max(1, len(normalized_reference)),
            "numeral_formatting_review": has_numerals,
            "possible_numeral_format_mismatch": has_numerals and normalized_reference != normalized_hypothesis,
            "usage": _coarse_usage(body),
            "transport_retries_for_row": retries_for_row,
        }, True)


def _row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "speaker": row["speaker"],
        "role": row["role"],
        "canonical_sha256": row["canonical_sha256"],
        "canonical_pcm_sha256": row["canonical_pcm_sha256"],
        "sample_rate_hz": 16000,
        "samples": row["samples"],
        "duration_s": row["duration_s"],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-manifest", required=True, type=Path)
    parser.add_argument("--transcript-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve(strict=True)
    audio_manifest_path = args.audio_manifest.resolve(strict=True)
    transcript_manifest_path = args.transcript_manifest.resolve(strict=True)
    prepared, source_hashes, _ = load_inputs(audio_manifest_path,
                                             transcript_manifest_path,
                                             repo_root)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(repo_root)
    except ValueError as exc:
        raise InputError("output directory must be inside the repository") from exc
    if output_dir.exists():
        raise InputError("output directory already exists; refusing duplicate API calls")
    if not sys.stdin.isatty():
        raise InputError("API key must be entered through an interactive no-echo prompt")
    api_key = getpass.getpass("DashScope API key (input hidden): ")
    if not api_key:
        raise InputError("API key is empty")

    output_dir.mkdir(parents=True, exist_ok=False)
    results_path = output_dir / "asr_results.jsonl"
    protocol = {
        "schema": "repeated_take_cloud_asr_content_qc_v1",
        "status": "started",
        "scope": "diagnostic only; no exclusion, selection, or regeneration",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "input_count": len(prepared),
        "input_order": [row["speaker"] + "/" + row["role"] for row in prepared],
        "sources": {
            "audio_manifest": {"path": str(audio_manifest_path), "sha256": source_hashes["audio_manifest_sha256"]},
            "transcript_manifest": {"path": str(transcript_manifest_path), "sha256": source_hashes["transcript_manifest_sha256"]},
        },
        "normalization": "Unicode NFKC casefold; remove Unicode punctuation and whitespace; no numeral rewriting",
        "cer": "character edit distance divided by normalized target transcript length",
        "transport_retry_limit": 1,
        "runtime": {"python": platform.python_version(), "requests": requests.__version__},
        "api_key_stored": False,
    }
    write_json(output_dir / "protocol.json", protocol)
    retry_state: dict[str, Any] = {"transport_retry_used": False, "transport_retry_for": None}
    started = time.time()
    completed = 0
    failure_status: str | None = None
    try:
        with results_path.open("x", encoding="utf-8") as stream:
            for row in prepared:
                result, succeeded = transcribe_one(row, api_key, retry_state)
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False) + "\n")
                stream.flush()
                completed += int(succeeded)
                print(f"{row['speaker']}/{row['role']}: {result['status']}", flush=True)
                if not succeeded:
                    failure_status = result["status"]
                    break
    finally:
        api_key = ""

    status = "complete" if completed == MAX_ROWS else (failure_status or "incomplete")
    receipt = {
        "schema": "repeated_take_cloud_asr_content_qc_receipt_v1",
        "status": status,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "input_count": MAX_ROWS,
        "successful_rows": completed,
        "result_rows": sum(1 for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "transport_retry_used": retry_state["transport_retry_used"],
        "transport_retry_for": retry_state["transport_retry_for"],
        "elapsed_seconds": round(time.time() - started, 3),
        "audio_manifest_sha256": source_hashes["audio_manifest_sha256"],
        "transcript_manifest_sha256": source_hashes["transcript_manifest_sha256"],
        "results_sha256": file_sha256(results_path),
        "api_key_stored": False,
    }
    write_json(output_dir / "receipt.json", receipt)
    print(json.dumps({"status": status, "successful_rows": completed,
                      "results": str(results_path), "receipt": str(output_dir / "receipt.json")},
                     sort_keys=True))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
