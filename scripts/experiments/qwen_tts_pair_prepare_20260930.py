"""Prepare frozen Chinese natural/TTS pairs and safely record Qwen VC calls.

This command has two offline stages (``prepare-local-pilot`` and
``select-hf``) and one online stage (``synthesize``). The API key is read only
from ``DASHSCOPE_API_KEY`` at synthesis time. Request bodies, authorization
headers, signed audio URLs, and exception messages are never written or printed.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import wave
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

MODEL = "qwen3-tts-vc-2026-01-22"
PROVIDER = "dashscope_vc"
LANGUAGE = "Chinese"
PROVIDER_RATE_HZ = 24_000
CANONICAL_RATE_HZ = 16_000
HF_DATASET = "AISHELL/AISHELL-1"
HF_REVISION = "bbe295d530192a4cd41644b711c9aecd087df653"
CAL_SPEAKERS = tuple(f"S{i:04d}" for i in range(52, 62))
EVAL_SPEAKERS = tuple(f"S{i:04d}" for i in range(62, 102))
MIN_DURATION_S = 3.0
MAX_DURATION_S = 10.0
MAX_ATTEMPTS = 3
RETRY_DELAYS_S = (1.0, 2.0)
HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x3134F),
    (0x2F800, 0x2FA1F),
)
USAGE_KEYS = {
    "count", "characters", "input_tokens", "output_tokens", "total_tokens",
    "audio_tokens", "text_tokens", "cached_tokens",
}
ROOT = Path(__file__).resolve().parents[2]
PILOT_DEFAULT = ROOT / "runs/recovered_20260930/local/qwen_source_axis/pilot_manifest.json"
SELECTION_RULE = {
    "dataset": HF_DATASET,
    "revision": HF_REVISION,
    "calibration_speakers": list(CAL_SPEAKERS),
    "evaluation_speakers": list(EVAL_SPEAKERS),
    "selection": "per speaker, sort WAV members by filename and take the first two whose official transcript is nonempty Han-only after whitespace removal and whose WAV duration is within the closed interval [3.0, 10.0] seconds",
    "transcript_normalization": "remove Unicode whitespace for TTS text; preserve source text verbatim in manifest",
    "excluded_cohorts": ["previous natural holdout S0002-S0041", "AISHELL test speakers S0764-S0916"],
    "use": "10 calibration speakers x 2 utterances; 40 holdout speakers x 2 utterances",
}


class SafeApiError(RuntimeError):
    """An API failure with a fixed, non-sensitive message."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def manifest_fingerprint(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return sha256_bytes(canonical_json_bytes(clean))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = manifest_fingerprint(result)
    return result


def verify_fingerprint(payload: dict[str, Any]) -> None:
    expected = payload.get("manifest_sha256")
    if expected and expected != manifest_fingerprint(payload):
        raise ValueError("input manifest fingerprint mismatch")


def duration_from_wav_bytes(data: bytes) -> tuple[float, int, int]:
    with wave.open(io.BytesIO(data), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        frames = stream.getnframes()
        if channels < 1 or sample_rate < 1 or frames < 1:
            raise ValueError("invalid WAV dimensions")
        return frames / sample_rate, sample_rate, channels


def full_audio_frame_audit(path: Path, block_frames: int = 65536) -> dict[str, Any]:
    """Count decoded frames to EOF separately from the WAV header declaration."""
    import soundfile as sf

    try:
        with wave.open(str(path), "rb") as header:
            declared_frames = int(header.getnframes())
            header_rate = int(header.getframerate())
            header_channels = int(header.getnchannels())
    except (wave.Error, EOFError, OSError):
        declared_frames = None
        header_rate = None
        header_channels = None
    with sf.SoundFile(str(path), mode="r") as stream:
        soundfile_frames = int(stream.frames)
        sample_rate = int(stream.samplerate)
        channels = int(stream.channels)
        subtype = str(stream.subtype)
        decoded_frames = 0
        while True:
            chunk = stream.read(block_frames, dtype="float32", always_2d=True)
            if not chunk.size:
                break
            decoded_frames += int(chunk.shape[0])
    return {
        "declared_frames": declared_frames,
        "soundfile_header_frames": soundfile_frames,
        "decoded_frames": decoded_frames,
        "declared_vs_payload_mismatch": declared_frames is not None and declared_frames != decoded_frames,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "riff_header_sample_rate_hz": header_rate,
        "riff_header_channels": header_channels,
        "subtype": subtype,
        "duration_s": round(decoded_frames / sample_rate, 9) if sample_rate else None,
    }


def project_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_ref_path(raw: str, manifest_dir: Path, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (repo_root / path).resolve()
    if candidate.is_file():
        return candidate
    return (manifest_dir / path).resolve()


def _text_is_han_only(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact) and all(any(lo <= ord(char) <= hi for lo, hi in HAN_RANGES) for char in compact)


def _parse_official_transcripts(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                continue
            utterance_id, text = fields[0], fields[1].strip()
            transcripts[utterance_id] = text
    return transcripts


def prepare_local_pilot(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    alignment_path = args.alignment.resolve()
    audio_root = args.audio_dir.resolve()
    alignment = read_json(alignment_path)
    source_rows = alignment.get("manifest", [])
    by_id = {
        str(row["sample_id"]): row
        for row in source_rows
        if row.get("condition") == "natural" and row.get("variant") == "raw"
    }
    records: list[dict[str, Any]] = []
    for sample_id in args.ids:
        key = str(sample_id)
        row = by_id.get(key)
        if row is None:
            raise ValueError(f"natural raw alignment entry missing for sample {sample_id}")
        audio_path = audio_root / f"{sample_id}.wav"
        if not audio_path.is_file():
            raise ValueError(f"pilot audio missing for sample {sample_id}")
        expected_audio = Path(str(row.get("filepath", "")))
        if expected_audio.name != audio_path.name:
            raise ValueError(f"pilot alignment path mismatch for sample {sample_id}")
        text = str(row.get("text", "")).strip()
        if not _text_is_han_only(text):
            raise ValueError(f"pilot transcript is not Han-only for sample {sample_id}")
        duration, rate, channels = duration_from_wav_bytes(audio_path.read_bytes())
        records.append({
            "id": f"pilot_local_{sample_id}",
            "speaker": f"local_clip_{sample_id}",
            "speaker_identity_status": "clip-scoped label; source metadata does not identify the human speaker",
            "text": "".join(text.split()),
            "transcript_source_text": text,
            "transcript_source_path": project_path(alignment_path, repo_root),
            "transcript_source_sha256": sha256_file(alignment_path),
            "ref_path": project_path(audio_path, repo_root),
            "ref_sha256": sha256_file(audio_path),
            "ref_sample_rate_hz": rate,
            "ref_channels": channels,
            "ref_duration_s": round(duration, 6),
            "split": "pilot",
            "language": LANGUAGE,
            "reference_role": "natural_self_clone_reference",
        })
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("duplicate pilot IDs")
    payload = add_fingerprint({
        "schema_version": 1,
        "manifest_type": "qwen_tts_pair_input_manifest",
        "status": "complete",
        "frozen_before_tts": True,
        "cohort": "historical local Chinese pilot; not a scientific or independent confirmation cohort",
        "dataset": {
            "name": "project-local natural audio",
            "audio_dir": project_path(audio_root, repo_root),
            "alignment_manifest": project_path(alignment_path, repo_root),
            "alignment_manifest_sha256": sha256_file(alignment_path),
        },
        "provider": PROVIDER,
        "model": MODEL,
        "language": LANGUAGE,
        "selection_rule": "Configured pilot IDs 1 and 2; use only existing natural/raw transcript rows and bind audio/transcript hashes before TTS.",
        "records_expected": len(records),
        "records": records,
    })
    write_json(args.output.resolve(), payload)
    print(f"PREPARED {len(records)} local pilot inputs: {args.output.resolve()}")
    return 0


def _archive_path_for(archives_dir: Path, speaker: str) -> Path:
    candidates = [archives_dir / f"{speaker}.tar.gz", archives_dir / f"{speaker}.tgz", archives_dir / speaker]
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate
    raise ValueError(f"speaker archive missing: {speaker}")


def _speaker_members(
    archive_path: Path,
    speaker: str,
    transcripts: dict[str, str],
) -> Iterator[tuple[str, bytes]]:
    """Yield eligible transcript WAVs in filename order, one body at a time."""
    if archive_path.is_dir():
        paths = sorted(
            (p for p in archive_path.rglob("*.wav") if p.is_file() and speaker in p.stem),
            key=lambda p: (p.name.casefold(), str(p).casefold()),
        )
        for path in paths:
            if _text_is_han_only(transcripts.get(path.stem, "")):
                yield str(path), path.read_bytes()
        return
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = sorted(
            (
                m for m in archive.getmembers()
                if m.isfile()
                and PurePosixPath(m.name).suffix.lower() == ".wav"
                and speaker in PurePosixPath(m.name).stem
            ),
            key=lambda m: (PurePosixPath(m.name).name.casefold(), m.name.casefold()),
        )
        for member in members:
            utterance_id = PurePosixPath(member.name).stem
            if not _text_is_han_only(transcripts.get(utterance_id, "")):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            with stream:
                yield member.name, stream.read()


def _speaker_archive_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def select_hf(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    archives_dir = args.archives_dir.resolve()
    transcript_path = args.transcript_file.resolve()
    output_path = args.output.resolve()
    output_dir = args.reference_out.resolve()
    transcript_map = _parse_official_transcripts(transcript_path)
    transcript_sha = sha256_file(transcript_path)
    rule = dict(SELECTION_RULE)
    selected: list[dict[str, Any]] = []
    speaker_audit: list[dict[str, Any]] = []
    archive_meta: dict[str, dict[str, Any]] = {}

    for split, speakers in (("calibration", CAL_SPEAKERS), ("evaluation", EVAL_SPEAKERS)):
        for speaker in speakers:
            archive_path = _archive_path_for(archives_dir, speaker)
            archive_digest = _speaker_archive_hash(archive_path)
            archive_meta[speaker] = {
                "path": project_path(archive_path, repo_root),
                "sha256": archive_digest,
                "dataset_path": f"data_aishell/wav/{speaker}.tar.gz" if archive_path.is_file() else None,
            }
            valid: list[tuple[str, bytes, str, float, int, int]] = []
            for member, data in _speaker_members(archive_path, speaker, transcript_map):
                filename = PurePosixPath(member).name
                utterance_id = Path(filename).stem
                source_text = transcript_map.get(utterance_id)
                try:
                    duration, sample_rate, channels = duration_from_wav_bytes(data)
                except (wave.Error, EOFError, ValueError):
                    continue
                if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
                    continue
                valid.append((member, data, source_text, duration, sample_rate, channels))
                if len(valid) == 2:
                    break
            if len(valid) < 2:
                raise ValueError(f"fewer than two eligible utterances for {speaker}")
            speaker_selected: list[dict[str, Any]] = []
            for member, data, source_text, duration, sample_rate, channels in valid[:2]:
                filename = PurePosixPath(member).name
                destination = output_dir / speaker / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and sha256_file(destination) != sha256_bytes(data):
                    raise ValueError(f"existing selected audio hash differs: {speaker}/{filename}")
                if not destination.exists():
                    destination.write_bytes(data)
                record = {
                    "id": f"{speaker}_{Path(filename).stem}",
                    "speaker": speaker,
                    "text": "".join(source_text.split()),
                    "transcript_source_text": source_text,
                    "transcript_source_path": project_path(transcript_path, repo_root),
                    "transcript_source_sha256": transcript_sha,
                    "ref_path": project_path(destination, repo_root),
                    "ref_sha256": sha256_bytes(data),
                    "ref_sample_rate_hz": sample_rate,
                    "ref_channels": channels,
                    "ref_duration_s": round(duration, 6),
                    "split": split,
                    "language": LANGUAGE,
                    "reference_role": "natural_self_clone_reference",
                    "source_dataset": HF_DATASET,
                    "source_revision": HF_REVISION,
                    "source_archive": archive_meta[speaker],
                    "source_archive_member": member,
                }
                selected.append(record)
                speaker_selected.append(record)
            speaker_audit.append({
                "speaker": speaker,
                "split": split,
                "status": "selected",
                "selected_ids": [record["id"] for record in speaker_selected],
                "selected_members": [record["source_archive_member"] for record in speaker_selected],
                "selected_count": len(speaker_selected),
                "rejection_reason": None,
            })

    expected = len(CAL_SPEAKERS) * 2 + len(EVAL_SPEAKERS) * 2
    if len(selected) != expected:
        raise ValueError("selected row count does not match frozen cohort")
    payload = add_fingerprint({
        "schema_version": 1,
        "manifest_type": "qwen_tts_pair_input_manifest",
        "status": "complete",
        "frozen_before_tts": True,
        "dataset": {"name": HF_DATASET, "revision": HF_REVISION, "transcript_path": project_path(transcript_path, repo_root), "transcript_sha256": transcript_sha},
        "selection_rule": rule,
        "selection_rule_sha256": sha256_bytes(canonical_json_bytes(rule)),
        "archives": archive_meta,
        "speaker_selection_audit": speaker_audit,
        "rejected_speakers": [entry["speaker"] for entry in speaker_audit if entry["status"] != "selected"],
        "provider": PROVIDER,
        "model": MODEL,
        "language": LANGUAGE,
        "records_expected": expected,
        "records": selected,
    })
    write_json(output_path, payload)
    print(f"PREPARED {len(selected)} HF inputs: {output_path}")
    return 0


def load_rows(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        payload: dict[str, Any] = {"manifest_type": "csv_input_manifest", "records": rows}
    else:
        payload = read_json(manifest_path)
        if isinstance(payload, list):
            payload = {"manifest_type": "json_input_manifest", "records": payload}
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError("JSON manifest must contain a records list")
        verify_fingerprint(payload)
        rows = list(payload["records"])
    required = {"ref_path", "speaker", "text", "id", "split"}
    seen: set[str] = set()
    for row in rows:
        if not required.issubset(row):
            raise ValueError("input row is missing a required field")
        identity = str(row["id"]).strip()
        if not identity or identity in seen:
            raise ValueError("input IDs must be nonempty and unique")
        seen.add(identity)
        if not str(row["speaker"]).strip() or not str(row["text"]).strip():
            raise ValueError(f"empty speaker or text for input {identity}")
    if not rows:
        raise ValueError("input manifest has no records")
    return payload, rows


def _request_id(response: Any, body: Any, secret: str) -> str | None:
    candidate = body.get("request_id") if isinstance(body, dict) else None
    if not candidate:
        for key, value in response.headers.items():
            if key.lower() in {"x-dashscope-requestid", "x-dashscope-request-id", "x-request-id"}:
                candidate = value
                break
    if not isinstance(candidate, str) or (secret and secret in candidate):
        return None
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", candidate):
        return None
    return candidate


def _safe_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in USAGE_KEYS and isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = item
        elif key in {"input_tokens_details", "output_tokens_details"} and isinstance(item, dict):
            nested = {k: v for k, v in item.items() if k in USAGE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)}
            if nested:
                result[key] = nested
    return result or None


def _safe_response_code(response: Any) -> str | None:
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code):
        return code
    return None


def _safe_error_detail(body: Any, secret: str) -> str | None:
    """Return a tightly filtered API message, never a body, URL, or token."""
    if not isinstance(body, dict):
        return None
    raw = body.get("message", body.get("Message"))
    if not isinstance(raw, str):
        return None
    detail = raw.strip()
    if not detail or len(detail) > 300 or (secret and secret in detail):
        return None
    if not detail.isascii() or any(ord(char) < 0x20 for char in detail):
        return None
    if re.search(r"(?i)https?://|data:[^\s]+|\bbearer\s+|api[_ -]?key|authorization|base64|\btoken\b", detail):
        return None
    if re.search(r"[A-Za-z0-9+/]{64,}={0,2}", detail):
        return None
    if not re.fullmatch(r"[A-Za-z0-9 _.,:;!?()'\"/\\\-\[\]]+", detail):
        return None
    return detail


def _retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


class AuditedDashScopeProvider:
    """Narrow safe wrapper around the project's DashScope VC provider."""

    def __init__(self, api_key: str, outdir: Path, repo_root: Path, base_url: str | None = None):
        sys.path.insert(0, str(repo_root / "scripts"))
        from tts.dashscope_vc import DashScopeQwen3VCProvider

        class _Provider(DashScopeQwen3VCProvider):
            def _init_registry(self, _repo_root: Path | None, _run_id: str) -> None:
                self.registry_path = outdir / "voice_registry.json"
                self._registry = {}
                if self.registry_path.is_file():
                    try:
                        self._registry = read_json(self.registry_path)
                    except (OSError, json.JSONDecodeError):
                        raise ValueError("voice registry is unreadable; preserve it and use a new output directory")

            def _register_voice(self, ref_audio_path: Path, voice_name: str) -> str:
                return self.audit_register(ref_audio_path, voice_name)

            def _synthesize(self, text: str, voice_id: str, language: str) -> tuple[Any, int]:
                return self.audit_synthesize(text, voice_id, language)

            def _download_audio(self, url: str) -> tuple[Any, int]:
                return self.audit_download(url)

        self.api_key = api_key
        cfg: dict[str, Any] = {"dashscope_vc": {"target_model": MODEL, "voice_prefix": "ttsx"}}
        if base_url:
            cfg["dashscope_vc"]["base_url"] = base_url
        self.provider = _Provider(cfg=cfg, env={"DASHSCOPE_API_KEY": api_key}, run_id="qwen_pair_prepare_20260930", repo_root=repo_root)
        self.provider.audit_register = self._register_voice
        self.provider.audit_synthesize = self._synthesize
        self.provider.audit_download = self._download_audio
        self.provider.last_registration: dict[str, Any] = {}
        self.provider.last_synthesis: dict[str, Any] = {}
        self.provider.last_download: dict[str, Any] = {}
        self.provider.last_audio_bytes = b""

    def _request_meta(self, response: Any, body: Any) -> dict[str, Any]:
        return {
            "status_code": int(response.status_code),
            "request_id": _request_id(response, body, self.api_key),
            "usage": _safe_usage(body.get("usage")) if isinstance(body, dict) else None,
            "error_code": _safe_response_code(response) if not response.ok else None,
            "error_detail": _safe_error_detail(body, self.api_key) if not response.ok else None,
        }

    def _register_voice(self, ref_audio_path: Path, voice_name: str) -> str:
        import requests

        data = base64.b64encode(ref_audio_path.read_bytes()).decode("ascii")
        ext = ref_audio_path.suffix.lower().lstrip(".")
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(ext, "audio/wav")
        payload = {
            "model": "qwen-voice-enrollment",
            "input": {
                "action": "create",
                "target_model": MODEL,
                "preferred_name": voice_name,
                "audio": {"data": f"data:{mime};base64,{data}"},
            },
        }
        url = f"{self.provider.base_url}/services/audio/tts/customization"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=120)
                try:
                    body = response.json()
                except Exception:
                    body = {}
                meta = self._request_meta(response, body)
                meta["attempt"] = attempt
                meta["phase"] = "voice_registration"
                attempts.append(meta)
                if response.status_code == 200 and isinstance(body, dict):
                    output = body.get("output", {})
                    voice_id = output.get("voice") if isinstance(output, dict) else None
                    if isinstance(voice_id, str) and voice_id and self.api_key not in voice_id:
                        self.provider.last_registration = {"attempts": attempts, **meta}
                        return voice_id
                    meta["error_code"] = "invalid_success_response"
                if not _retryable_http_status(int(response.status_code)):
                    break
            except requests.RequestException as exc:
                attempts.append({"attempt": attempt, "phase": "voice_registration", "transport_error_type": type(exc).__name__})
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAYS_S[min(attempt - 1, len(RETRY_DELAYS_S) - 1)])
        self.provider.last_registration = {"attempts": attempts, "phase": "voice_registration", "status": "failed"}
        raise SafeApiError("voice registration failed")

    def _synthesize(self, text: str, voice_id: str, language: str) -> tuple[Any, int]:
        import requests

        payload = {"model": MODEL, "input": {"text": text, "voice": voice_id, "language_type": language}}
        url = f"{self.provider.base_url}/services/aigc/multimodal-generation/generation"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=120)
                try:
                    body = response.json()
                except Exception:
                    body = {}
                meta = self._request_meta(response, body)
                meta["attempt"] = attempt
                meta["phase"] = "synthesis"
                if response.status_code == 200 and isinstance(body, dict):
                    output = body.get("output", {})
                    audio = output.get("audio", {}) if isinstance(output, dict) else {}
                    audio_url = audio.get("url") if isinstance(audio, dict) else None
                    if isinstance(audio_url, str) and audio_url:
                        waveform, sample_rate = self._download_audio(audio_url)
                        meta["download"] = dict(self.provider.last_download)
                        attempts.append(meta)
                        self.provider.last_synthesis = {"attempts": attempts, **meta}
                        return waveform, sample_rate
                    meta["error_code"] = "missing_audio_url"
                attempts.append(meta)
                if not _retryable_http_status(int(response.status_code)):
                    break
            except SafeApiError as exc:
                attempts.append({"attempt": attempt, "phase": "audio_download", "transport_error_type": type(exc).__name__})
            except requests.RequestException as exc:
                attempts.append({"attempt": attempt, "phase": "synthesis", "transport_error_type": type(exc).__name__})
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAYS_S[min(attempt - 1, len(RETRY_DELAYS_S) - 1)])
        self.provider.last_synthesis = {"attempts": attempts, "phase": "synthesis", "status": "failed"}
        raise SafeApiError("synthesis failed")

    def _download_audio(self, url: str) -> tuple[Any, int]:
        import requests
        import numpy as np
        import soundfile as sf

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            body = response.content
            audio, sample_rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise ValueError("empty or non-finite provider audio")
            waveform = audio.mean(axis=1, dtype=np.float32)
            self.provider.last_audio_bytes = body
            self.provider.last_download = {
                "status_code": int(response.status_code),
                "request_id": _request_id(response, None, self.api_key),
                "bytes": len(body),
                "sha256": sha256_bytes(body),
                "content_type": response.headers.get("Content-Type", "").split(";", 1)[0][:80],
            }
            return waveform, int(sample_rate)
        except requests.RequestException as exc:
            self.provider.last_download = {"status": "failed", "transport_error_type": type(exc).__name__}
            raise SafeApiError("audio download failed")

    def voice_for(self, sample_id: str, reference: Path, reference_sha256: str) -> tuple[str, dict[str, Any]]:
        registry = self.provider._registry
        cached = registry.get(sample_id)
        short_sha = self.provider._audio_sha(reference)
        if cached and cached.get("audio_sha") == short_sha and cached.get("audio_sha256", reference_sha256) == reference_sha256:
            self.provider.last_registration = {
                "cached": True,
                "request_id": cached.get("request_id"),
                "usage": cached.get("usage"),
                "voice_name": cached.get("voice_name"),
            }
            return str(cached["voice_id"]), dict(self.provider.last_registration)
        voice_name = preferred_voice_name(sample_id)
        voice_id = self._register_voice(reference, voice_name)
        meta = dict(self.provider.last_registration)
        registry[sample_id] = {
            "audio_sha": short_sha,
            "audio_sha256": reference_sha256,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "model": MODEL,
            "request_id": meta.get("request_id"),
            "usage": meta.get("usage"),
            "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.provider._save_registry()
        return voice_id, meta


def preferred_voice_name(sample_id: str) -> str:
    """Provider-safe stable enrollment name (ASCII alnum/underscore, <=16)."""
    value = "ttsxp_" + hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    if len(value) > 16 or not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError("preferred voice name violates the provider name contract")
    return value


def resolve_ffmpeg(explicit: str | None) -> Path:
    candidate: str | None = explicit
    if candidate:
        found = shutil.which(candidate)
        if found:
            candidate = found
        elif not Path(candidate).is_file():
            raise ValueError("configured ffmpeg executable was not found")
    else:
        candidate = shutil.which("ffmpeg")
        if not candidate:
            try:
                import imageio_ffmpeg
                candidate = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                candidate = None
    if not candidate or not Path(candidate).is_file():
        raise ValueError("ffmpeg executable could not be resolved; install imageio-ffmpeg or pass --ffmpeg")
    return Path(candidate).resolve()


def ffmpeg_provenance(executable: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(executable), "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except Exception:
        raise ValueError("ffmpeg version probe failed") from None
    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    if not first_line or len(first_line) > 300:
        raise ValueError("ffmpeg version probe returned an invalid result")
    return {
        "executable_path": str(executable),
        "executable_sha256": sha256_file(executable),
        "version_line": first_line,
        "canonicalization_command_template": [
            str(executable), "-y", "-v", "error", "-i", "<provider_audio>",
            "-ar", str(CANONICAL_RATE_HZ), "-ac", "1", "-c:a", "pcm_s16le",
            "<canonical_output>",
        ],
    }


def canonicalize(ffmpeg: Path, provider_path: Path, canonical_path: Path) -> dict[str, Any]:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [str(ffmpeg), "-y", "-v", "error", "-i", str(provider_path), "-ar", str(CANONICAL_RATE_HZ), "-ac", "1", "-c:a", "pcm_s16le", str(canonical_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise ValueError("ffmpeg canonicalization failed") from None
    import soundfile as sf
    info = sf.info(str(canonical_path))
    full_audit = full_audio_frame_audit(canonical_path)
    if info.samplerate != CANONICAL_RATE_HZ or info.channels != 1 or info.subtype != "PCM_16" or full_audit["decoded_frames"] <= 0:
        raise ValueError("canonical audio is not mono 16 kHz PCM16")
    return {
        "canonical_16k_audio": str(canonical_path.resolve()),
        "canonical_audio_sha256": sha256_file(canonical_path),
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "samples": int(full_audit["decoded_frames"]),
        "duration_s": round(float(full_audit["decoded_frames"] / info.samplerate), 6),
        "canonical_full_decode_audit": full_audit,
    }


def synthesize(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    output_dir = args.outdir.resolve()
    repo_root = args.repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required")
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    ffmpeg_meta = ffmpeg_provenance(ffmpeg)
    raw_payload, raw_rows = load_rows(manifest_path)
    frozen_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        item = dict(row)
        ref_path = resolve_ref_path(str(item["ref_path"]), manifest_path.parent, repo_root)
        if not ref_path.is_file():
            raise ValueError(f"reference audio missing for {item['id']}")
        digest = sha256_file(ref_path)
        expected = str(item.get("ref_sha256", "")).strip()
        if expected and expected != digest:
            raise ValueError(f"reference hash mismatch for {item['id']}")
        item["ref_path"] = str(ref_path)
        item["ref_sha256"] = digest
        item.setdefault("language", LANGUAGE)
        frozen_rows.append(item)
    frozen = add_fingerprint({
        "schema_version": 1,
        "manifest_type": "qwen_tts_frozen_request_manifest",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_manifest_fingerprint": raw_payload.get("manifest_sha256"),
        "frozen_before_tts": True,
        "provider": PROVIDER,
        "model": MODEL,
        "records_expected": len(frozen_rows),
        "records": frozen_rows,
    })
    frozen_path = output_dir / "input_manifest.frozen.json"
    if frozen_path.exists():
        existing = read_json(frozen_path)
        verify_fingerprint(existing)
        if existing.get("manifest_sha256") != frozen.get("manifest_sha256"):
            raise ValueError("output directory is frozen to a different input manifest")
    else:
        write_json(frozen_path, frozen)
    meta_path = output_dir / "tts_meta.json"
    results: dict[str, dict[str, Any]] = {}
    if meta_path.exists():
        existing_meta = read_json(meta_path)
        if existing_meta.get("input_manifest_sha256") != frozen["manifest_sha256"]:
            raise ValueError("existing TTS metadata belongs to a different frozen manifest")
        results = dict(existing_meta.get("results", {}))
    provider = AuditedDashScopeProvider(api_key, output_dir, repo_root, args.base_url)
    for index, row in enumerate(frozen_rows, 1):
        sample_id = str(row["id"])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or f"item_{index:03d}"
        ref_path = Path(row["ref_path"])
        if sha256_file(ref_path) != row["ref_sha256"]:
            raise ValueError(f"reference changed after freeze for {sample_id}")
        previous = results.get(sample_id)
        if previous and previous.get("status") == "ok":
            provider_file = Path(previous.get("provider_audio", ""))
            canonical_file = Path(previous.get("canonical_16k_audio", ""))
            if (
                provider_file.is_file() and canonical_file.is_file()
                and sha256_file(provider_file) == previous.get("provider_audio_sha256")
                and sha256_file(canonical_file) == previous.get("canonical_audio_sha256")
            ):
                print(f"REUSE {index}/{len(frozen_rows)} {sample_id}", flush=True)
                continue
        item_result: dict[str, Any] = {
            "id": sample_id,
            "speaker": str(row["speaker"]),
            "split": str(row["split"]),
            "text": str(row["text"]),
            "reference_audio": str(ref_path),
            "reference_audio_sha256": row["ref_sha256"],
            "provider": PROVIDER,
            "model": MODEL,
            "language": row.get("language", LANGUAGE),
            "status": "running",
            "retry_status": [],
        }
        phase = "voice_registration"
        try:
            voice_id, registration = provider.voice_for(sample_id, ref_path, row["ref_sha256"])
            item_result["voice_id"] = voice_id
            item_result["voice_registration"] = registration
            phase = "synthesis"
            waveform, sample_rate = provider._synthesize(str(row["text"]), voice_id, str(row.get("language", LANGUAGE)))
            raw_bytes = provider.provider.last_audio_bytes
            if not raw_bytes:
                raise ValueError("provider returned no original audio bytes")
            provider_path = output_dir / "audio_24k_raw" / f"{safe_id}.provider.wav"
            provider_path.parent.mkdir(parents=True, exist_ok=True)
            if provider_path.exists() and sha256_file(provider_path) != sha256_bytes(raw_bytes):
                raise ValueError("existing provider audio hash differs; use a new output directory")
            if not provider_path.exists():
                provider_path.write_bytes(raw_bytes)
            provider_header_audit = full_audio_frame_audit(provider_path)
            import numpy as np
            values = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("provider waveform is invalid")
            canonical_path = output_dir / "audio_16k_canonical" / f"{safe_id}.wav"
            canonical = canonicalize(ffmpeg, provider_path, canonical_path)
            provider_info = provider.provider.last_download
            item_result.update({
                "provider_audio": str(provider_path.resolve()),
                "provider_audio_sha256": sha256_file(provider_path),
                "provider_audio_bytes": len(raw_bytes),
                "provider_sample_rate_hz": int(sample_rate),
                "provider_expected_24k": int(sample_rate) == PROVIDER_RATE_HZ,
                "provider_duration_s": round(float(values.size / int(sample_rate)), 6),
                "provider_full_decode_audit": provider_header_audit,
                "provider_download": provider_info,
                "synthesis": provider.provider.last_synthesis,
                **canonical,
                "status": "ok",
                "retry_status": [
                    *registration.get("attempts", []),
                    *provider.provider.last_synthesis.get("attempts", []),
                ],
            })
            results[sample_id] = item_result
            print(f"OK {index}/{len(frozen_rows)} {sample_id}", flush=True)
        except Exception as exc:
            item_result.update({
                "status": "failed",
                "failed_phase": phase,
                "error_type": type(exc).__name__,
                "voice_registration": dict(provider.provider.last_registration),
                "synthesis": dict(provider.provider.last_synthesis),
                "retry_status": [
                    *provider.provider.last_registration.get("attempts", []),
                    *provider.provider.last_synthesis.get("attempts", []),
                ],
            })
            results[sample_id] = item_result
            latest_attempts = item_result["retry_status"]
            latest = latest_attempts[-1] if latest_attempts else {}
            diagnostic = {
                "sample_id": sample_id,
                "phase": phase,
                "error_type": type(exc).__name__,
                "status_code": latest.get("status_code"),
                "error_code": latest.get("error_code"),
                "error_detail": latest.get("error_detail"),
            }
            print("FAIL " + json.dumps(diagnostic, separators=(",", ":"), ensure_ascii=True), flush=True)
        current = {
            "schema_version": 1,
            "manifest_type": "qwen_tts_pair_generation_result",
            "input_manifest": str(frozen_path.resolve()),
            "input_manifest_sha256": frozen["manifest_sha256"],
            "provider": PROVIDER,
            "model": MODEL,
            "language": LANGUAGE,
            "expected_provider_sample_rate_hz": PROVIDER_RATE_HZ,
            "canonical_sample_rate_hz": CANONICAL_RATE_HZ,
            "canonicalization": ffmpeg_meta,
            "records_expected": len(frozen_rows),
            "samples_ok": sum(value.get("status") == "ok" for value in results.values()),
            "samples_failed": sum(value.get("status") == "failed" for value in results.values()),
            "complete": len(results) == len(frozen_rows) and all(value.get("status") == "ok" for value in results.values()),
            "results": results,
        }
        write_json(meta_path, current)
    complete = len(results) == len(frozen_rows) and all(value.get("status") == "ok" for value in results.values())
    print(json.dumps({"samples_ok": sum(value.get("status") == "ok" for value in results.values()), "samples_failed": sum(value.get("status") == "failed" for value in results.values()), "complete": complete}, separators=(",", ":")))
    return 0 if complete else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("prepare-local-pilot", help="freeze the two configured project-local Chinese natural clips")
    local.add_argument("--repo-root", type=Path, default=ROOT)
    local.add_argument("--audio-dir", type=Path, default=ROOT / "data/data/audio")
    local.add_argument("--alignment", type=Path, default=ROOT / "data/wav2sem_analysis_zh/manifest/alignment.json")
    local.add_argument("--ids", type=int, nargs="+", default=[1, 2])
    local.add_argument("--output", type=Path, default=PILOT_DEFAULT)
    local.set_defaults(func=prepare_local_pilot)

    hf = subparsers.add_parser("select-hf", help="select and hash the frozen AISHELL source-axis cohorts from per-speaker tar files")
    hf.add_argument("--repo-root", type=Path, default=ROOT)
    hf.add_argument("--archives-dir", type=Path, required=True)
    hf.add_argument("--transcript-file", type=Path, required=True)
    hf.add_argument("--reference-out", type=Path, required=True)
    hf.add_argument("--output", type=Path, required=True)
    hf.set_defaults(func=select_hf)

    synth = subparsers.add_parser("synthesize", help="create Qwen VC audio while preserving safe HTTP provenance")
    synth.add_argument("--manifest", type=Path, required=True, help="JSON or CSV with ref_path,speaker,text,id,split")
    synth.add_argument("--outdir", type=Path, required=True)
    synth.add_argument("--repo-root", type=Path, default=ROOT)
    synth.add_argument("--base-url", default=None)
    synth.add_argument("--ffmpeg", default=None, help="optional path or command name for ffmpeg")
    synth.set_defaults(func=synthesize)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        # Deliberately suppress exception text: HTTP clients can include request
        # material or signed URLs in their messages.
        print(f"ERROR {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
