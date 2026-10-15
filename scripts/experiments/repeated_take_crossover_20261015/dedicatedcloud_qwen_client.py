"""Audited, one-shot Qwen voice-clone client for the repeated-take crossover.

This script deliberately requires ``--execute`` before making any network call.
It uses the project's known DashScope enrollment and synthesis endpoints, keeps
provider audio bytes unchanged, and emits a redacted request/audio audit. 16 kHz
canonicalization is a separate downstream step so it can use the pinned remote
FFmpeg rather than changing the provider artifact on this workstation.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import io
import json
import os
import re
import time
import wave
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import requests


MODEL = "qwen3-tts-vc-2026-01-22"
ENROLLMENT_MODEL = "qwen-voice-enrollment"
BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
ENROLLMENT_PATH = "/services/audio/tts/customization"
SYNTHESIS_PATH = "/services/aigc/multimodal-generation/generation"
MAX_SPEAKERS = 8
MAX_TRANSPORT_RETRIES = 1
MAX_DUPLICATE_PROBES = 1
SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,160}\Z")
SAFE_VOICE = re.compile(r"[A-Za-z0-9._-]{1,160}\Z")
USAGE_KEYS = {
    "count", "characters", "input_tokens", "output_tokens", "total_tokens",
    "audio_tokens", "text_tokens", "cached_tokens",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "runs/repeated_take_crossover_20261015_attempt01/assets_discovery/aishell_repeated_realizations/manifest.json"


class SafeCallError(RuntimeError):
    def __init__(self, phase: str, status: int | None = None, code: str | None = None,
                 request_id: str | None = None, transport: bool = False,
                 safe_detail: str | None = None):
        super().__init__(phase)
        self.phase = phase
        self.status = status
        self.code = code
        self.request_id = request_id
        self.transport = transport
        self.safe_detail = safe_detail

    def audit(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status_code": self.status,
            "error_code": self.code,
            "request_id": self.request_id,
            "transport_error": self.transport,
            "safe_error_message": self.safe_detail,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _safe_request_id(headers: Any, body: Any) -> str | None:
    candidate = body.get("request_id") if isinstance(body, dict) else None
    if not candidate:
        for key, value in headers.items():
            if key.lower() in {"x-dashscope-requestid", "x-dashscope-request-id", "x-request-id"}:
                candidate = value
                break
    return candidate if isinstance(candidate, str) and SAFE_ID.fullmatch(candidate) else None


def _safe_usage(body: Any) -> dict[str, Any] | None:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    safe = {k: v for k, v in usage.items()
            if k in USAGE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)}
    for key in ("input_tokens_details", "output_tokens_details"):
        value = usage.get(key)
        if isinstance(value, dict):
            nested = {k: v for k, v in value.items()
                      if k in USAGE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)}
            if nested:
                safe[key] = nested
    return safe or None


def _safe_error_code(body: Any) -> str | None:
    code = body.get("code") if isinstance(body, dict) else None
    return code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) else None


def _safe_error_message(body: Any, secret: str) -> str | None:
    raw = body.get("message") if isinstance(body, dict) else None
    if not isinstance(raw, str):
        return None
    text = " ".join(raw.split())
    if not text or len(text) > 300 or not text.isascii():
        return None
    if secret and secret in text:
        return None
    if re.search(r"(?i)(https?:|www\\.|bearer|authorization|api.?key|access.?key|secret|token|credential|signature|x-amz|aliyuncs\\.com)", text):
        return None
    if re.search(r"[A-Za-z0-9_-]{100,}", text):
        return None
    if not re.fullmatch(r"[A-Za-z0-9 _.,:;()'\"/-]{1,300}", text):
        return None
    return text


def _decode_json(response: requests.Response, phase: str,
                 secret: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        body = response.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    request_id = _safe_request_id(response.headers, body)
    meta = {
        "phase": phase,
        "status_code": int(response.status_code),
        "request_id": request_id,
        "usage": _safe_usage(body),
        "error_code": _safe_error_code(body) if not response.ok else None,
        "safe_error_message": _safe_error_message(body, secret) if not response.ok else None,
    }
    if response.status_code != 200:
        raise SafeCallError(phase, int(response.status_code), meta["error_code"], request_id,
                            safe_detail=meta["safe_error_message"])
    return body, meta


def _resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()
    # Discovery manifests carry the Linux server path; map its repo-relative
    # suffix to this checkout when the staged files have also been copied here.
    marker = "/tts-exp/"
    if marker in value.replace("\\", "/"):
        suffix = value.replace("\\", "/").split(marker, 1)[1]
        candidate = REPO_ROOT.joinpath(*PurePosixPath(suffix).parts)
        if candidate.is_file():
            return candidate.resolve()
    candidate = (manifest_dir / value).resolve()
    if candidate.is_file():
        return candidate
    raise ValueError("manifest audio path is unavailable")


def _load_plan(manifest_path: Path, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    speakers = source.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise ValueError("manifest has no speaker records")
    if not 1 <= limit <= MAX_SPEAKERS:
        raise ValueError("speaker limit must be 1..8")
    plan: list[dict[str, Any]] = []
    for row in speakers[:limit]:
        speaker = str(row.get("speaker", ""))
        if not re.fullmatch(r"S[0-9]{4}", speaker):
            raise ValueError("manifest contains an invalid speaker id")
        target = row.get("target_transcript")
        text = "".join(str(target or "").split())
        if not text:
            raise ValueError("manifest target transcript is empty")
        reference = row.get("reference")
        pair = row.get("target_pair")
        if not isinstance(reference, dict) or not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("manifest speaker record is incomplete")
        ref_path = _resolve_manifest_path(str(reference.get("staged_path", "")), manifest_path.parent)
        ref_sha = sha256_file(ref_path)
        if ref_sha != reference.get("raw_sha256"):
            raise ValueError("reference audio hash does not match discovery manifest")
        takes: dict[str, dict[str, Any]] = {}
        for item in pair:
            role = str(item.get("role", ""))
            if role not in {"N1", "N2"} or role in takes:
                raise ValueError("natural take roles are invalid")
            path = _resolve_manifest_path(str(item.get("staged_path", "")), manifest_path.parent)
            if sha256_file(path) != item.get("raw_sha256"):
                raise ValueError("natural audio hash does not match discovery manifest")
            if str(item.get("transcript", "")).translate(str.maketrans("", "", " \t\r\n")) != text:
                raise ValueError("natural transcript differs from exact TTS text")
            takes[role] = {"utterance_id": item.get("utterance_id"), "path": str(path),
                           "sha256": item.get("raw_sha256")}
        if set(takes) != {"N1", "N2"}:
            raise ValueError("both N1 and N2 are required")
        plan.append({
            "speaker": speaker,
            "text": text,
            "text_sha256": sha256_bytes(text.encode("utf-8")),
            "reference_path": str(ref_path),
            "reference_sha256": ref_sha,
            "reference_utterance_id": reference.get("utterance_id"),
            "natural": takes,
        })
    return source, plan


def _read_wav_or_soundfile(data: bytes) -> dict[str, Any]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            rate, count = wav.getframerate(), wav.getnframes()
            channels, width = wav.getnchannels(), wav.getsampwidth()
            if rate <= 0 or count <= 0 or channels <= 0 or not frames:
                raise ValueError("empty WAV")
            domain = f"wav-pcm:{rate}:{channels}:{width}:".encode("ascii")
            decoded_hash = sha256_bytes(domain + frames)
            duration = count / rate
            kind = "wav"
    except (wave.Error, EOFError):
        # Some provider formats are compressed. Decode only in memory for the
        # duplicate check; preserve the exact downloaded bytes on disk.
        import numpy as np
        import soundfile as sf
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("provider audio cannot be decoded") from None
        count, channels = audio.shape
        domain = f"decoded-f32:{int(rate)}:{channels}:".encode("ascii")
        decoded_hash = sha256_bytes(domain + audio.astype("<f4", copy=False).tobytes(order="C"))
        duration, width, kind = count / int(rate), None, "decoded"
    return {"decoded_pcm_sha256": decoded_hash, "sample_rate_hz": int(rate),
            "samples_per_channel": int(count), "channels": int(channels),
            "sample_width_bytes": width, "duration_s": round(float(duration), 6),
            "container": kind}


class Client:
    def __init__(self, api_key: str, outdir: Path):
        self.api_key = api_key
        self.outdir = outdir
        self.session = requests.Session()
        self.auth_headers = {"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"}
        self.transport_retries_left = MAX_TRANSPORT_RETRIES
        self.duplicate_probes_left = MAX_DUPLICATE_PROBES
        self.audit: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "model": MODEL,
            "enrollment_model": ENROLLMENT_MODEL,
            "endpoint_base": BASE_URL,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "transport_retries_allowed": MAX_TRANSPORT_RETRIES,
            "duplicate_probes_allowed": MAX_DUPLICATE_PROBES,
            "speakers": [],
        }

    def _post_json(self, url: str, payload: dict[str, Any], phase: str,
                   allow_transport_retry: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        while True:
            try:
                response = self.session.post(url, json=payload, headers=self.auth_headers,
                                             timeout=120)
            except requests.RequestException:
                if allow_transport_retry and self.transport_retries_left:
                    self.transport_retries_left -= 1
                    self.audit["transport_retries_used"] = MAX_TRANSPORT_RETRIES - self.transport_retries_left
                    self._save()
                    continue
                raise SafeCallError(phase, transport=True) from None
            return _decode_json(response, phase, self.api_key)

    def _save(self) -> None:
        self.audit["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(self.outdir / "client_audit.json", self.audit)

    def register(self, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        ref_path = Path(row["reference_path"])
        data = ref_path.read_bytes()
        mime = "audio/wav" if data[:12].startswith(b"RIFF") and data[8:12] == b"WAVE" else "audio/wav"
        # Match the provider-safe naming rule used by the project's audited
        # DashScope client: ASCII letters/digits/underscore, at most 16 chars.
        preferred_name = "ttsxp_" + hashlib.sha256(row["speaker"].encode("utf-8")).hexdigest()[:10]
        payload = {
            "model": ENROLLMENT_MODEL,
            "input": {"action": "create", "target_model": MODEL,
                      "preferred_name": preferred_name,
                      "audio": {"data": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}},
        }
        body, meta = self._post_json(BASE_URL + ENROLLMENT_PATH, payload, "voice_registration", False)
        output = body.get("output")
        voice_id = output.get("voice") if isinstance(output, dict) else None
        if not isinstance(voice_id, str) or not SAFE_VOICE.fullmatch(voice_id):
            raise SafeCallError("voice_registration", meta.get("status_code"), "invalid_success_response", meta.get("request_id"))
        meta["voice_name"] = preferred_name
        return voice_id, meta

    def synthesize_one(self, text: str, voice_id: str, speaker: str, take: str) -> dict[str, Any]:
        payload = {"model": MODEL,
                   "input": {"text": text, "voice": voice_id, "language_type": "Chinese"}}
        body, meta = self._post_json(BASE_URL + SYNTHESIS_PATH, payload, "synthesis", True)
        if not meta.get("request_id"):
            raise SafeCallError("synthesis", meta.get("status_code"), "missing_request_id")
        output = body.get("output")
        audio = output.get("audio") if isinstance(output, dict) else None
        audio_url = audio.get("url") if isinstance(audio, dict) else None
        if not isinstance(audio_url, str) or not audio_url.startswith("https://"):
            raise SafeCallError("synthesis", meta.get("status_code"), "missing_secure_audio_url", meta.get("request_id"))
        try:
            response = self.session.get(audio_url, timeout=90)
            if response.status_code != 200:
                raise SafeCallError("audio_download", int(response.status_code), None, None)
            raw = response.content
        except requests.RequestException:
            raise SafeCallError("audio_download", transport=True) from None
        if not raw:
            raise SafeCallError("audio_download", int(response.status_code), "empty_audio")
        suffix = ".wav" if raw[:12].startswith(b"RIFF") and raw[8:12] == b"WAVE" else ".provider-audio"
        path = self.outdir / speaker / f"{take}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)
        result = {
            "take": take,
            "request_id": meta["request_id"],
            "usage": meta.get("usage"),
            "model": MODEL,
            "path": str(path.resolve()),
            "provider_bytes": len(raw),
            "provider_audio_sha256": sha256_bytes(raw),
        }
        try:
            result.update(_read_wav_or_soundfile(raw))
            result["decode_status"] = "complete"
        except Exception:
            # The exact billed response bytes already exist create-only on disk;
            # retain their audit row even when this runtime cannot decode them.
            result["decode_status"] = "failed"
        return result

    def run_speaker(self, row: dict[str, Any]) -> None:
        voice_id, registration = self.register(row)
        speaker_record: dict[str, Any] = {
            "speaker": row["speaker"], "text_sha256": row["text_sha256"],
            "reference_utterance_id": row["reference_utterance_id"],
            "reference_sha256": row["reference_sha256"],
            "voice_id": voice_id, "registration": registration,
            "natural": row["natural"], "tts": [],
        }
        self.audit["speakers"].append(speaker_record)
        self._save()
        first = self.synthesize_one(row["text"], voice_id, row["speaker"], "T1")
        speaker_record["tts"].append(first)
        self._save()
        if first.get("decode_status") != "complete":
            raise SafeCallError("audio_decode", None, "provider_audio_decode_failed", first.get("request_id"))
        second = self.synthesize_one(row["text"], voice_id, row["speaker"], "T2")
        speaker_record["tts"].append(second)
        self._save()
        if second.get("decode_status") != "complete":
            raise SafeCallError("audio_decode", None, "provider_audio_decode_failed", second.get("request_id"))
        ids = [item["request_id"] for item in speaker_record["tts"]]
        if len(set(ids)) != len(ids):
            raise SafeCallError("synthesis", None, "duplicate_request_id")
        pcm_hashes = [item["decoded_pcm_sha256"] for item in speaker_record["tts"]]
        if pcm_hashes[0] == pcm_hashes[1]:
            if not self.duplicate_probes_left:
                raise SafeCallError("synthesis", None, "duplicate_audio_probe_budget_exhausted")
            self.duplicate_probes_left -= 1
            self.audit["duplicate_probes_used"] = MAX_DUPLICATE_PROBES - self.duplicate_probes_left
            self._save()
            third = self.synthesize_one(row["text"], voice_id, row["speaker"], "T3_probe")
            speaker_record["duplicate_probe"] = third
            self._save()
            if third.get("decode_status") != "complete":
                raise SafeCallError("audio_decode", None, "provider_audio_decode_failed", third.get("request_id"))
            if third["decoded_pcm_sha256"] in pcm_hashes:
                raise SafeCallError("synthesis", None, "provider_audio_not_independent")
            # Retain a genuinely distinct pair while preserving the first output
            # and the duplicate audit. The probe replaces the duplicate T2 slot.
            speaker_record["tts"][1] = {**third, "take": "T2", "duplicate_probe_path": third["path"]}
            self._save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="required to make cloud calls")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    outdir = args.outdir.resolve()
    _, plan = _load_plan(manifest_path, args.max_speakers)
    if not args.execute:
        print(json.dumps({"status": "plan_only", "speaker_count": len(plan),
                          "expected_synthesis_calls": 2 * len(plan),
                          "model": MODEL, "input_manifest_sha256": sha256_file(manifest_path)}, indent=2))
        return 0
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit("output directory must be new or empty; no artifacts were changed")
    outdir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("DashScope API key (hidden): ").strip()
    if not api_key:
        raise SystemExit("no API key supplied")
    client = Client(api_key, outdir)
    client.audit["input_manifest_sha256"] = sha256_file(manifest_path)
    client.audit["planned_speakers"] = [row["speaker"] for row in plan]
    client.audit["input_manifest_path"] = str(manifest_path)
    client._save()
    try:
        for row in plan:
            client.run_speaker(row)
        client.audit["status"] = "complete"
    except SafeCallError as exc:
        client.audit["status"] = "failed_safe_api_call"
        client.audit["failure"] = exc.audit()
        client._save()
        print(json.dumps({"status": client.audit["status"], "failure": exc.audit()}, indent=2))
        return 2
    except Exception as exc:
        # Never serialize an exception message: request libraries may include
        # signed audio URLs or request details in it.
        client.audit["status"] = "failed_local_processing"
        client.audit["failure"] = {"exception_type": type(exc).__name__}
        client._save()
        print(json.dumps({"status": client.audit["status"], "failure": client.audit["failure"]}, indent=2))
        return 2
    finally:
        client.session.close()
        client.api_key = ""
        api_key = ""
    client.audit["finished_utc"] = datetime.now(timezone.utc).isoformat()
    client._save()
    print(json.dumps({"status": client.audit["status"], "speaker_count": len(client.audit["speakers"]),
                      "output_dir": str(outdir), "audit_path": str(outdir / "client_audit.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
