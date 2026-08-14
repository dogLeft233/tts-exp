#!/usr/bin/env python3
"""Build strict weak phone-local TTS waveform targets for HuBERT training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pnp_alignment_warp import phone_local_warp  # type: ignore[import-not-found]

TARGET_SAMPLE_RATE = 16_000
TARGET_SCHEMA_VERSION = 1
TARGET_DEFINITION = "weak_phone_local_tts_target"
DEFAULT_MIN_CONFIDENCE = 0.8
DEFAULT_MIN_TOKEN_COVERAGE = 0.9
DEFAULT_MIN_DURATION_COVERAGE = 0.9
NON_SPEECH_LABELS = {
    "", "sil", "sp", "spn", "pau", "<eps>", "<unk>", "unknown", "<unknown>",
    "<sil>", "h#", "noise", "<noise>", "[noise]", "nsn", "br", "laugh",
    "<laugh>", "vocalization", "<vocalization>", "cough", "<cough>",
}


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_label(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ")


def is_speech_token(token: Mapping[str, Any]) -> bool:
    if any(bool(token.get(key, False)) for key in ("is_silence", "is_noise", "is_unknown", "is_non_speech")):
        return False
    return normalize_label(token.get("label", token.get("token", ""))) not in NON_SPEECH_LABELS


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", payload.get("entries", payload)) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("manifest must contain a records/entries list")
    return [dict(record) for record in records]


def _resolve_path(value: str | Path, *, manifest_path: Path, repo_root: Path | None) -> Path:
    candidate = Path(value).expanduser()
    bases = [manifest_path.parent]
    if repo_root is not None:
        bases.append(repo_root)
    matches = []
    for base in bases:
        resolved = (base / candidate if not candidate.is_absolute() else candidate).resolve()
        if resolved.exists():
            matches.append(resolved)
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"audio path is missing or ambiguous: {value}")
    return unique[0]


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono: {path}")
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"audio must be non-empty and finite: {path}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)

def _safe_pair_key(value: Any) -> str:
    key = str(value)
    if not key or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
        raise ValueError(f"unsafe paired_key: {key!r}")
    return key


def _alignment_source(value: Any) -> str:
    source = str(value or "").strip().lower()
    if source not in {"mfa", "provided"}:
        raise ValueError("alignment_source must be mfa or provided")
    return source


def _resolve_alignment_path(value: str | Path, *, manifest_path: Path, repo_root: Path | None) -> Path:
    return _resolve_path(value, manifest_path=manifest_path, repo_root=repo_root)


def _span_records(
    record: Mapping[str, Any],
    *,
    duration_s: float,
    manifest_path: Path,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    _alignment_source(record.get("alignment_source"))
    tokens = record.get("tokens")
    alignment_path = record.get("alignment_path", record.get("alignment_file"))
    alignment_sha256 = record.get("alignment_sha256")
    actual_alignment_sha256: str | None = None
    if tokens is None and alignment_path:
        resolved_alignment = _resolve_alignment_path(
            alignment_path, manifest_path=manifest_path, repo_root=repo_root
        )
        actual_alignment_sha256 = file_sha256(resolved_alignment)
        if alignment_sha256 and str(alignment_sha256) != actual_alignment_sha256:
            raise ValueError("alignment artifact hash mismatch")
        tokens = json.loads(resolved_alignment.read_text(encoding="utf-8"))
    if tokens is not None and not isinstance(tokens, list):
        raise ValueError("token alignment must be a list")
    if tokens is None:
        raise ValueError("record requires inline tokens or alignment_path")
    if actual_alignment_sha256 is None:
        actual_alignment_sha256 = _json_hash(tokens)
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("record requires a non-empty token alignment")
    spans: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, raw in enumerate(tokens):
        if not isinstance(raw, Mapping):
            raise ValueError(f"token {index} must be an object")
        if "start_s" not in raw or "end_s" not in raw:
            raise ValueError(f"token {index} requires start_s/end_s")
        start, end = float(raw["start_s"]), float(raw["end_s"])
        confidence_value = raw.get("alignment_confidence", raw.get("confidence", 1.0))
        confidence = float(confidence_value)
        if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(confidence)):
            raise ValueError("alignment contains non-finite values")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("alignment confidence must be in [0, 1]")
        if start < 0 or end <= start or end > duration_s + 1e-5:
            raise ValueError("alignment span is outside audio duration or non-positive")
        if start < previous_end - 1e-7:
            raise ValueError("alignment spans must be ordered and non-overlapping")
        if index == 0 and start > 1e-5:
            raise ValueError("alignment must start at zero")
        if index > 0 and start > previous_end + 1e-5:
            raise ValueError("alignment contains a gap between token intervals")
        label = normalize_label(raw.get("label", raw.get("token", "")))
        if not label:
            raise ValueError("alignment token label must be non-empty")
        spans.append({
            "label": label,
            "start_s": start,
            "end_s": end,
            "confidence": confidence,
            "is_silence": bool(raw.get("is_silence", False)),
            "is_noise": bool(raw.get("is_noise", False)),
            "is_unknown": bool(raw.get("is_unknown", False)),
            "is_non_speech": bool(raw.get("is_non_speech", False)),
            "is_speech": is_speech_token(raw),
        })
        previous_end = end
    return spans


def _speech_token(span: Mapping[str, Any] | Any) -> bool:
    if isinstance(span, Mapping):
        return is_speech_token(span)
    return normalize_label(getattr(span, "label", "")) not in NON_SPEECH_LABELS


def _speech_duration(spans: Iterable[Mapping[str, Any] | Any], min_confidence: float) -> float:
    total = 0.0
    for span in spans:
        if isinstance(span, Mapping):
            start = span.get("start_s")
            end = span.get("end_s")
            confidence = span.get("confidence", 1.0)
            speech = is_speech_token(span)
        else:
            if hasattr(span, "natural_start_s"):
                start = getattr(span, "natural_start_s")
                end = getattr(span, "natural_end_s")
                confidence = getattr(span, "natural_confidence", 1.0)
            else:
                start = getattr(span, "start_s")
                end = getattr(span, "end_s")
                confidence = getattr(span, "confidence", 1.0)
            speech = _speech_token(span)
        if start is None or end is None:
            raise ValueError("span is missing start/end")
        if float(confidence) >= min_confidence and speech:
            total += float(end) - float(start)
    return float(total)


def _pair_groups(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for index, raw in enumerate(records):
        record = dict(raw)
        pair_key = record.get("paired_key", record.get("pair_key", record.get("sample_id")))
        condition = str(record.get("condition", ""))
        if not pair_key or condition not in {"natural", "tts"}:
            raise ValueError(f"record {index} requires pair key and natural/tts condition")
        key = str(pair_key)
        group = groups.setdefault(key, {})
        if condition in group:
            raise ValueError(f"duplicate {condition} arm for pair {key}")
        group[condition] = record
    return groups


def validate_pair(
    group: Mapping[str, Mapping[str, Any]],
    *,
    manifest_path: Path,
    repo_root: Path | None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_token_coverage: float = DEFAULT_MIN_TOKEN_COVERAGE,
    min_duration_coverage: float = DEFAULT_MIN_DURATION_COVERAGE,
) -> tuple[dict[str, Any], np.ndarray]:
    if set(group) != {"natural", "tts"}:
        raise ValueError("pair must contain exactly natural and tts arms")
    natural, tts = group["natural"], group["tts"]
    text_nat = " ".join(str(natural.get("transcript", "")).split())
    text_tts = " ".join(str(tts.get("transcript", "")).split())
    if not text_nat or text_nat != text_tts:
        raise ValueError("natural and TTS transcripts must match")
    if natural.get("split") != tts.get("split"):
        raise ValueError("natural and TTS splits must match")
    speaker_group = natural.get("speaker_group", natural.get("speaker_id"))
    if speaker_group is None:
        raise ValueError("speaker_group or speaker_id is required")
    natural_path = _resolve_path(natural["audio_path"], manifest_path=manifest_path, repo_root=repo_root)
    tts_path = _resolve_path(tts["audio_path"], manifest_path=manifest_path, repo_root=repo_root)
    natural_audio, natural_sr = _load_audio(natural_path)
    tts_audio, tts_source_sample_rate = _load_audio(tts_path)
    tts_source_sample_count = int(tts_audio.size)
    tts_source_duration_s = tts_source_sample_count / tts_source_sample_rate
    if natural_sr != TARGET_SAMPLE_RATE:
        raise ValueError("natural arm must be 16 kHz before target construction")
    if tts_source_sample_rate not in {TARGET_SAMPLE_RATE, 24_000}:
        raise ValueError("TTS arm must be 16 kHz or explicit 24 kHz Faster-Qwen3 output")
    natural_spans = _span_records(
        natural,
        duration_s=len(natural_audio) / natural_sr,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    tts_spans = _span_records(
        tts,
        duration_s=tts_source_duration_s,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    tts_resampling: dict[str, Any] = {
        "applied": tts_source_sample_rate != TARGET_SAMPLE_RATE,
        "method": "scipy.signal.resample_poly" if tts_source_sample_rate != TARGET_SAMPLE_RATE else None,
        "source_sample_rate": tts_source_sample_rate,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "source_duration_validation": "tts_alignment_spans_validated_against_source_duration_before_resampling",
    }
    if tts_source_sample_rate != TARGET_SAMPLE_RATE:
        gcd = math.gcd(tts_source_sample_rate, TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // gcd
        down = tts_source_sample_rate // gcd
        tts_audio = resample_poly(tts_audio, up, down).astype(np.float32, copy=False)
        tts_resampling.update({"up": up, "down": down})
    tts_sr = TARGET_SAMPLE_RATE
    tts_resampling["converted_sample_count"] = int(tts_audio.size)
    tts_resampling["converted_duration_s"] = tts_audio.size / tts_sr
    if not natural_spans or not tts_spans:
        raise ValueError("alignments must contain tokens")
    audio_duration_nat = len(natural_audio) / natural_sr
    if (
        natural_spans[-1]["end_s"] < audio_duration_nat - 1e-5
        or tts_spans[-1]["end_s"] < tts_source_duration_s - 1e-5
    ):
        raise ValueError("alignment does not cover the complete audio duration")
    natural_speech_spans = [span for span in natural_spans if is_speech_token(span)]
    tts_speech_spans = [span for span in tts_spans if is_speech_token(span)]
    if not natural_speech_spans or not tts_speech_spans:
        raise ValueError("both alignments must contain speech tokens")
    speech_nat = _speech_duration(natural_speech_spans, min_confidence)
    if speech_nat <= 0:
        raise ValueError("natural alignment has no accepted speech duration")
    result = phone_local_warp(
        natural_audio,
        tts_audio,
        TARGET_SAMPLE_RATE,
        natural_speech_spans,
        tts_speech_spans,
        min_confidence=min_confidence,
        crossfade_ms=8.0,
        alignment_source="mfa_or_provided",
    )
    matched_speech = _speech_duration(
        [span for span in result.matched_spans if _speech_token(span)],
        min_confidence,
    )
    used_count = int(result.metadata.get("used_span_count", 0))
    token_coverage = float(used_count / max(1, len(natural_speech_spans)))
    duration_coverage = float(matched_speech / speech_nat)
    if token_coverage < min_token_coverage:
        raise ValueError(f"matched token coverage {token_coverage:.3f} is below threshold")
    if duration_coverage < min_duration_coverage:
        raise ValueError(f"matched speech coverage {duration_coverage:.3f} is below threshold")
    if result.audio.shape != natural_audio.shape or not np.isfinite(result.audio).all():
        raise ValueError("target must be finite and exactly natural length")
    return {
        "paired_key": _safe_pair_key(natural.get("paired_key", natural.get("pair_key", natural.get("sample_id")))),
        "split": str(natural["split"]),
        "speaker_group": str(speaker_group),
        "transcript": text_nat,
        "natural_path": str(natural_path),
        "tts_path": str(tts_path),
        "natural_sha256": file_sha256(natural_path),
        "tts_sha256": file_sha256(tts_path),
        "natural_alignment_sha256": _json_hash(natural_spans),
        "tts_alignment_sha256": _json_hash(tts_spans),
        "natural_sample_rate": natural_sr,
        "tts_source_sample_rate": tts_source_sample_rate,
        "tts_source_sample_count": tts_source_sample_count,
        "tts_source_duration_s": tts_source_duration_s,
        "tts_sample_rate": tts_sr,
        "natural_sample_count": int(natural_audio.size),
        "tts_sample_count": int(tts_audio.size),
        "tts_converted_duration_s": tts_audio.size / tts_sr,
        "tts_resampling": tts_resampling,
        "target_sample_count": int(result.audio.size),
        "target_definition": TARGET_DEFINITION,
        "target_definition_version": TARGET_SCHEMA_VERSION,
        "matched_token_count": int(result.metadata["matched_count"]),
        "natural_token_count": len(natural_spans),
        "tts_token_count": len(tts_spans),
        "matched_token_coverage": token_coverage,
        "matched_speech_duration_coverage": duration_coverage,
        "weak_target_warning": "not disentangled ground truth; local TTS acoustic splice",
        "alignment_source": "independent_natural_and_tts_alignments",
        "metadata": result.metadata,
    }, result.audio


def build_targets(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_token_coverage: float = DEFAULT_MIN_TOKEN_COVERAGE,
    min_duration_coverage: float = DEFAULT_MIN_DURATION_COVERAGE,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    groups = _pair_groups(_load_records(manifest))
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for pair_key, group in sorted(groups.items()):
        try:
            metadata, audio = validate_pair(
                group,
                manifest_path=manifest,
                repo_root=Path(repo_root).resolve() if repo_root else None,
                min_confidence=min_confidence,
                min_token_coverage=min_token_coverage,
                min_duration_coverage=min_duration_coverage,
            )
            output_path = root / f"{pair_key}.wav"
            metadata_path = root / f"{pair_key}.json"
            sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
            metadata["target_sha256"] = file_sha256(output_path)
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            accepted.append({**metadata, "target_path": str(output_path), "metadata_path": str(metadata_path)})
        except (KeyError, OSError, ValueError, TypeError) as exc:
            excluded.append({"paired_key": pair_key, "reason": str(exc)})
    payload = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "target_definition": TARGET_DEFINITION,
        "source_manifest": str(manifest),
        "source_manifest_sha256": file_sha256(manifest),
        "min_confidence": min_confidence,
        "min_token_coverage": min_token_coverage,
        "min_duration_coverage": min_duration_coverage,
        "accepted": accepted,
        "excluded": excluded,
    }
    (root / "target_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min-token-coverage", type=float, default=DEFAULT_MIN_TOKEN_COVERAGE)
    parser.add_argument("--min-duration-coverage", type=float, default=DEFAULT_MIN_DURATION_COVERAGE)
    args = parser.parse_args(argv)
    result = build_targets(
        args.manifest,
        args.output_dir,
        repo_root=args.repo_root,
        min_confidence=args.min_confidence,
        min_token_coverage=args.min_token_coverage,
        min_duration_coverage=args.min_duration_coverage,
    )
    print(json.dumps({k: result[k] for k in ("schema_version", "target_definition", "accepted", "excluded")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
