#!/usr/bin/env python3
"""Prepare a strict train/valid-only manifest for two-stage HuBERT training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_waveform_hubert_targets import is_speech_token

TARGET_MANIFEST_SCHEMA_VERSION = 1
TARGET_DEFINITION = "weak_phone_local_tts_target"
STAGE_MANIFEST_SCHEMA_VERSION = 1
STAGE_MANIFEST_TYPE = "two_stage_hubert_pair_eligibility"
STAGE_CONTRACT = "direct_raw_tts_features_train_valid_only"
TARGET_SAMPLE_RATE = 16_000
ALLOWED_STAGE_SPLITS = frozenset({"train", "valid"})
ALLOWED_SOURCE_SPLITS = ALLOWED_STAGE_SPLITS | {"heldout"}
_PAIR_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

_STAGE_RECORD_KEYS = {
    "schema_version",
    "paired_key",
    "split",
    "speaker_group",
    "transcript",
    "transcript_sha256",
    "natural_path",
    "natural_sha256",
    "natural_sample_rate",
    "natural_sample_count",
    "natural_alignment_sha256",
    "tts_path",
    "tts_sha256",
    "tts_source_sample_rate",
    "tts_source_sample_count",
    "tts_source_duration_s",
    "tts_sample_rate",
    "tts_sample_count",
    "tts_converted_duration_s",
    "tts_resampling",
    "tts_alignment_sha256",
    "matched_span_metadata_path",
    "matched_span_metadata_sha256",
    "matched_spans",
    "matched_spans_sha256",
    "minimum_alignment_confidence",
    "provenance",
}


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_text(value: Any, *, field: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _normalized_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("_", " ")
    if not label:
        raise ValueError("matched span label must be non-empty")
    return label


def _safe_pair_key(value: Any) -> str:
    key = str(value or "")
    if not _PAIR_KEY_RE.fullmatch(key):
        raise ValueError(f"unsafe paired_key: {key!r}")
    return key


def _resolve_existing_path(
    value: Any,
    *,
    manifest_path: Path,
    repo_root: Path | None,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    bases = [manifest_path.parent]
    if repo_root is not None:
        bases.append(repo_root)
    matches: list[Path] = []
    for base in bases:
        resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
        if resolved.exists():
            matches.append(resolved)
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"{field} is missing or ambiguous: {value}")
    return unique[0]


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _require_finite(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    number = int(value)
    if number <= 0 or number != value:
        raise ValueError(f"{field} must be a positive integer")
    return number


def _source_records(payload: Any) -> list[dict[str, Any]]:
    records = payload.get("records", payload.get("entries")) if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("source manifest must contain a records or entries list")
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"source record {index} must be an object")
        validated.append(dict(raw))
    return validated


def _source_pair_index(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for index, raw in enumerate(records):
        key = _safe_pair_key(raw.get("paired_key", raw.get("pair_key", raw.get("sample_id"))))
        condition = str(raw.get("condition") or "").strip().lower()
        if condition not in {"natural", "tts"}:
            raise ValueError(f"source record {index} has unsupported condition")
        pair = pairs.setdefault(key, {})
        if condition in pair:
            raise ValueError(f"source manifest has duplicate {condition} arm for {key}")
        pair[condition] = dict(raw)
    return pairs


def _mfa_tokens(record: Mapping[str, Any], *, arm: str) -> list[dict[str, Any]]:
    if str(record.get("alignment_source") or "").strip().lower() != "mfa":
        raise ValueError(f"source {arm} arm must use MFA alignment")
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"source {arm} arm requires non-empty inline MFA tokens")
    spans: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, raw in enumerate(tokens):
        if not isinstance(raw, Mapping):
            raise ValueError(f"source {arm} MFA token {index} must be an object")
        start = _require_finite(raw.get("start_s"), field=f"source {arm} MFA token start_s")
        end = _require_finite(raw.get("end_s"), field=f"source {arm} MFA token end_s")
        confidence = _require_finite(
            raw.get("alignment_confidence", raw.get("confidence", 1.0)),
            field=f"source {arm} MFA token confidence",
        )
        if start < 0.0 or end <= start:
            raise ValueError(f"source {arm} MFA token has invalid interval")
        if start < previous_end - 1e-7:
            raise ValueError(f"source {arm} MFA tokens overlap or are unordered")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"source {arm} MFA token confidence must be in [0, 1]")
        spans.append(
            {
                "label": _normalized_label(raw.get("label", raw.get("token"))),
                "start_s": start,
                "end_s": end,
                "confidence": confidence,
                "is_silence": bool(raw.get("is_silence", False)),
                "is_noise": bool(raw.get("is_noise", False)),
                "is_unknown": bool(raw.get("is_unknown", False)),
                "is_non_speech": bool(raw.get("is_non_speech", False)),
                "is_speech": is_speech_token(raw),
            }
        )
        previous_end = end
    return spans


def _validate_source_arm(
    record: Mapping[str, Any],
    *,
    arm: str,
    paired_key: str,
    split: str,
    transcript: str,
    row_path: Path,
    row_sha256: str,
    manifest_path: Path,
    repo_root: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if str(record.get("condition") or "").strip().lower() != arm:
        raise ValueError(f"source {arm} arm has incompatible condition")
    if str(record.get("variant") or "").strip().lower() != "raw":
        raise ValueError(f"source {arm} arm must be raw")
    if _safe_pair_key(record.get("paired_key", record.get("pair_key", record.get("sample_id")))) != paired_key:
        raise ValueError(f"source {arm} arm paired_key mismatch")
    if str(record.get("split") or "") != split:
        raise ValueError(f"source {arm} arm split mismatch")
    if _normalized_text(record.get("transcript"), field=f"source {arm} transcript") != transcript:
        raise ValueError(f"source {arm} arm transcript mismatch")
    source_path = _resolve_existing_path(
        record.get("audio_path", record.get("filepath")),
        manifest_path=manifest_path,
        repo_root=repo_root,
        field=f"source {arm} audio_path",
    )
    source_sha256 = _require_sha256(record.get("source_sha256"), field=f"source {arm} source_sha256")
    if file_sha256(source_path) != source_sha256:
        raise ValueError(f"source {arm} audio hash mismatch")
    if source_path != row_path or source_sha256 != row_sha256:
        raise ValueError(f"source {arm} arm path or hash does not match target manifest")
    return dict(record), _mfa_tokens(record, arm=arm)


def _validate_resampling(row: Mapping[str, Any]) -> dict[str, Any]:
    source_rate = _require_positive_int(row.get("tts_source_sample_rate"), field="tts_source_sample_rate")
    source_count = _require_positive_int(row.get("tts_source_sample_count"), field="tts_source_sample_count")
    source_duration = _require_finite(row.get("tts_source_duration_s"), field="tts_source_duration_s")
    converted_rate = _require_positive_int(row.get("tts_sample_rate"), field="tts_sample_rate")
    converted_count = _require_positive_int(row.get("tts_sample_count"), field="tts_sample_count")
    converted_duration = _require_finite(row.get("tts_converted_duration_s"), field="tts_converted_duration_s")
    if source_rate not in {TARGET_SAMPLE_RATE, 24_000}:
        raise ValueError("tts_source_sample_rate must be 16 kHz or explicit 24 kHz")
    if converted_rate != TARGET_SAMPLE_RATE:
        raise ValueError("tts_sample_rate must be 16 kHz")
    if not math.isclose(source_duration, source_count / source_rate, abs_tol=1e-9):
        raise ValueError("tts_source_duration_s is inconsistent with source sample facts")
    if not math.isclose(converted_duration, converted_count / converted_rate, abs_tol=1e-9):
        raise ValueError("tts_converted_duration_s is inconsistent with converted sample facts")
    raw = row.get("tts_resampling")
    if not isinstance(raw, Mapping):
        raise ValueError("tts_resampling must be an object")
    resampling = dict(raw)
    if resampling.get("source_sample_rate") != source_rate:
        raise ValueError("tts_resampling source sample rate mismatch")
    if resampling.get("target_sample_rate") != TARGET_SAMPLE_RATE:
        raise ValueError("tts_resampling target sample rate mismatch")
    if resampling.get("converted_sample_count") != converted_count:
        raise ValueError("tts_resampling converted sample count mismatch")
    if not math.isclose(
        _require_finite(resampling.get("converted_duration_s"), field="tts_resampling converted_duration_s"),
        converted_duration,
        abs_tol=1e-9,
    ):
        raise ValueError("tts_resampling converted duration mismatch")
    if resampling.get("source_duration_validation") != (
        "tts_alignment_spans_validated_against_source_duration_before_resampling"
    ):
        raise ValueError("tts_resampling must declare source-duration MFA validation")
    expected_applied = source_rate != TARGET_SAMPLE_RATE
    if resampling.get("applied") is not expected_applied:
        raise ValueError("tts_resampling applied flag mismatch")
    if expected_applied:
        if resampling.get("method") != "scipy.signal.resample_poly":
            raise ValueError("24 kHz TTS must use declared polyphase resampling")
    elif resampling.get("method") is not None:
        raise ValueError("16 kHz TTS must not declare a resampling method")
    return resampling


def _matching_source_span(
    spans: Sequence[Mapping[str, Any]],
    *,
    start: float,
    end: float,
    label: str,
    start_index: int,
    arm: str,
) -> int:
    for index in range(start_index, len(spans)):
        source = spans[index]
        if (
            source["label"] == label
            and math.isclose(source["start_s"], start, abs_tol=1e-5)
            and math.isclose(source["end_s"], end, abs_tol=1e-5)
        ):
            return index
    raise ValueError(f"matched span cannot be traced to ordered MFA {arm} tokens")


def _matched_span_metadata(metadata_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    if metadata_artifact.get("alignment_source") != "independent_natural_and_tts_alignments":
        raise ValueError("metadata artifact has incompatible alignment provenance")
    metadata = metadata_artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata artifact requires phone-warp metadata")
    return metadata


def _validated_matched_spans(
    metadata: Mapping[str, Any],
    *,
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
    min_confidence: float,
) -> list[dict[str, Any]]:
    if metadata.get("alignment_source") != "mfa_or_provided":
        raise ValueError("matched-span metadata has incompatible alignment provenance")
    if metadata.get("sample_rate") != TARGET_SAMPLE_RATE:
        raise ValueError("matched-span metadata must use 16 kHz sample rate")
    raw_spans = metadata.get("spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("matched-span metadata requires non-empty spans")
    if metadata.get("used_span_count") != len(raw_spans):
        raise ValueError("matched-span used_span_count mismatch")
    canonical: list[dict[str, Any]] = []
    natural_cursor = 0
    tts_cursor = 0
    previous_natural_end = 0.0
    previous_tts_end = 0.0
    for index, raw in enumerate(raw_spans):
        if not isinstance(raw, Mapping):
            raise ValueError(f"matched span {index} must be an object")
        label = _normalized_label(raw.get("label"))
        natural_start = _require_finite(raw.get("natural_start_s"), field="natural_start_s")
        natural_end = _require_finite(raw.get("natural_end_s"), field="natural_end_s")
        tts_start = _require_finite(raw.get("tts_start_s"), field="tts_start_s")
        tts_end = _require_finite(raw.get("tts_end_s"), field="tts_end_s")
        natural_confidence = _require_finite(raw.get("natural_confidence"), field="natural_confidence")
        tts_confidence = _require_finite(raw.get("tts_confidence"), field="tts_confidence")
        if natural_start < 0 or natural_end <= natural_start or tts_start < 0 or tts_end <= tts_start:
            raise ValueError("matched span has invalid interval")
        if natural_start < previous_natural_end - 1e-7 or tts_start < previous_tts_end - 1e-7:
            raise ValueError("matched spans overlap or are unordered")
        if not min_confidence <= natural_confidence <= 1.0 or not min_confidence <= tts_confidence <= 1.0:
            raise ValueError("matched span confidence is below the parent manifest threshold")
        natural_index = _matching_source_span(
            natural_tokens,
            start=natural_start,
            end=natural_end,
            label=label,
            start_index=natural_cursor,
            arm="natural",
        )
        tts_index = _matching_source_span(
            tts_tokens,
            start=tts_start,
            end=tts_end,
            label=label,
            start_index=tts_cursor,
            arm="tts",
        )
        source_natural = natural_tokens[natural_index]
        source_tts = tts_tokens[tts_index]
        if not math.isclose(source_natural["confidence"], natural_confidence, abs_tol=1e-8):
            raise ValueError("matched span natural confidence does not match MFA token")
        if not math.isclose(source_tts["confidence"], tts_confidence, abs_tol=1e-8):
            raise ValueError("matched span TTS confidence does not match MFA token")
        canonical.append(
            {
                "label": label,
                "natural_start_s": natural_start,
                "natural_end_s": natural_end,
                "tts_start_s": tts_start,
                "tts_end_s": tts_end,
                "natural_confidence": natural_confidence,
                "tts_confidence": tts_confidence,
            }
        )
        natural_cursor = natural_index + 1
        tts_cursor = tts_index + 1
        previous_natural_end = natural_end
        previous_tts_end = tts_end
    return canonical


def _validate_parent_row(
    row: Mapping[str, Any],
    *,
    parent_path: Path,
    source_path: Path,
    source_pairs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    repo_root: Path | None,
    min_confidence: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("target_definition") != TARGET_DEFINITION:
        raise ValueError("parent accepted row has incompatible target_definition")
    if row.get("target_definition_version") != TARGET_MANIFEST_SCHEMA_VERSION:
        raise ValueError("parent accepted row has incompatible target definition version")
    paired_key = _safe_pair_key(row.get("paired_key"))
    split = str(row.get("split") or "")
    if split not in ALLOWED_SOURCE_SPLITS:
        raise ValueError(f"pair {paired_key} has unsupported split: {split!r}")
    transcript = _normalized_text(row.get("transcript"), field="transcript")
    speaker_group = str(row.get("speaker_group") or "").strip()
    if not speaker_group:
        raise ValueError("speaker_group must be non-empty")
    natural_path = _resolve_existing_path(
        row.get("natural_path"),
        manifest_path=parent_path,
        repo_root=repo_root,
        field="natural_path",
    )
    tts_path = _resolve_existing_path(
        row.get("tts_path"),
        manifest_path=parent_path,
        repo_root=repo_root,
        field="tts_path",
    )
    natural_sha256 = _require_sha256(row.get("natural_sha256"), field="natural_sha256")
    tts_sha256 = _require_sha256(row.get("tts_sha256"), field="tts_sha256")
    if file_sha256(natural_path) != natural_sha256:
        raise ValueError("natural audio hash mismatch")
    if file_sha256(tts_path) != tts_sha256:
        raise ValueError("TTS audio hash mismatch")
    if _require_positive_int(row.get("natural_sample_rate"), field="natural_sample_rate") != TARGET_SAMPLE_RATE:
        raise ValueError("natural_sample_rate must be 16 kHz")
    natural_sample_count = _require_positive_int(row.get("natural_sample_count"), field="natural_sample_count")
    natural_alignment_sha256 = _require_sha256(
        row.get("natural_alignment_sha256"), field="natural_alignment_sha256"
    )
    tts_alignment_sha256 = _require_sha256(row.get("tts_alignment_sha256"), field="tts_alignment_sha256")
    resampling = _validate_resampling(row)
    if row.get("alignment_source") != "independent_natural_and_tts_alignments":
        raise ValueError("parent accepted row has incompatible alignment_source")
    source_pair = source_pairs.get(paired_key)
    if source_pair is None or set(source_pair) != {"natural", "tts"}:
        raise ValueError(f"source manifest requires exactly natural and TTS arms for {paired_key}")
    natural_source, natural_tokens = _validate_source_arm(
        source_pair["natural"],
        arm="natural",
        paired_key=paired_key,
        split=split,
        transcript=transcript,
        row_path=natural_path,
        row_sha256=natural_sha256,
        manifest_path=source_path,
        repo_root=repo_root,
    )
    tts_source, tts_tokens = _validate_source_arm(
        source_pair["tts"],
        arm="tts",
        paired_key=paired_key,
        split=split,
        transcript=transcript,
        row_path=tts_path,
        row_sha256=tts_sha256,
        manifest_path=source_path,
        repo_root=repo_root,
    )
    if str(tts_source.get("tts_provider") or "").strip().lower() != "faster_qwen3":
        raise ValueError("source TTS arm must declare the faster_qwen3 provider")
    source_speaker = str(
        natural_source.get("speaker_group", natural_source.get("speaker_id", "")) or ""
    ).strip()
    if source_speaker != speaker_group:
        raise ValueError("source natural speaker group mismatch")
    metadata_path = _resolve_existing_path(
        row.get("metadata_path"),
        manifest_path=parent_path,
        repo_root=repo_root,
        field="metadata_path",
    )
    metadata_sha256 = file_sha256(metadata_path)
    metadata = _strict_json_load(metadata_path)
    if not isinstance(metadata, Mapping):
        raise ValueError("matched-span metadata must be an object")
    expected_metadata_artifact = {
        key: value
        for key, value in row.items()
        if key not in {"target_path", "metadata_path"}
    }
    if metadata != expected_metadata_artifact:
        raise ValueError("parent metadata artifact does not match accepted row")
    matched_metadata = _matched_span_metadata(metadata)
    spans = _validated_matched_spans(
        matched_metadata,
        natural_tokens=natural_tokens,
        tts_tokens=tts_tokens,
        min_confidence=min_confidence,
    )
    if row.get("matched_token_count") != len(spans):
        raise ValueError("parent matched_token_count does not match validated spans")
    if row.get("natural_token_count") != len(natural_tokens):
        raise ValueError("parent natural_token_count does not match MFA provenance")
    if row.get("tts_token_count") != len(tts_tokens):
        raise ValueError("parent tts_token_count does not match MFA provenance")
    if _json_hash(natural_tokens) != natural_alignment_sha256:
        raise ValueError("natural alignment hash does not match MFA provenance")
    if _json_hash(tts_tokens) != tts_alignment_sha256:
        raise ValueError("TTS alignment hash does not match MFA provenance")
    stage_record = {
        "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
        "paired_key": paired_key,
        "split": split,
        "speaker_group": speaker_group,
        "transcript": transcript,
        "transcript_sha256": _json_hash(transcript),
        "natural_path": str(natural_path),
        "natural_sha256": natural_sha256,
        "natural_sample_rate": TARGET_SAMPLE_RATE,
        "natural_sample_count": natural_sample_count,
        "natural_alignment_sha256": natural_alignment_sha256,
        "tts_path": str(tts_path),
        "tts_sha256": tts_sha256,
        "tts_source_sample_rate": row["tts_source_sample_rate"],
        "tts_source_sample_count": row["tts_source_sample_count"],
        "tts_source_duration_s": row["tts_source_duration_s"],
        "tts_sample_rate": row["tts_sample_rate"],
        "tts_sample_count": row["tts_sample_count"],
        "tts_converted_duration_s": row["tts_converted_duration_s"],
        "tts_resampling": resampling,
        "tts_alignment_sha256": tts_alignment_sha256,
        "matched_span_metadata_path": str(metadata_path),
        "matched_span_metadata_sha256": metadata_sha256,
        "matched_spans": spans,
        "matched_spans_sha256": _json_hash(spans),
        "minimum_alignment_confidence": min_confidence,
        "provenance": {
            "natural": {"condition": "natural", "variant": "raw", "alignment_source": "mfa"},
            "tts": {
                "condition": "tts",
                "variant": "raw",
                "alignment_source": "mfa",
                "tts_provider": "faster_qwen3",
            },
        },
    }
    identity = {
        "paired_key": paired_key,
        "speaker_group": speaker_group,
        "natural_sha256": natural_sha256,
        "tts_sha256": tts_sha256,
        "transcript_sha256": _json_hash(transcript),
    }
    return stage_record, identity


def _validate_stage_record(
    row: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path | None,
) -> dict[str, Any]:
    if set(row) != _STAGE_RECORD_KEYS:
        unexpected = sorted(set(row) - _STAGE_RECORD_KEYS)
        missing = sorted(_STAGE_RECORD_KEYS - set(row))
        raise ValueError(f"stage record schema mismatch; unexpected={unexpected}, missing={missing}")
    if row.get("schema_version") != STAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("stage record has incompatible schema version")
    paired_key = _safe_pair_key(row.get("paired_key"))
    split = str(row.get("split") or "")
    if split not in ALLOWED_STAGE_SPLITS:
        raise ValueError(f"stage manifest cannot load split: {split!r}")
    transcript = _normalized_text(row.get("transcript"), field="stage transcript")
    if _require_sha256(row.get("transcript_sha256"), field="transcript_sha256") != _json_hash(transcript):
        raise ValueError("stage transcript hash mismatch")
    speaker_group = str(row.get("speaker_group") or "").strip()
    if not speaker_group:
        raise ValueError("stage speaker_group must be non-empty")
    natural_path = _resolve_existing_path(
        row.get("natural_path"), manifest_path=manifest_path, repo_root=repo_root, field="natural_path"
    )
    tts_path = _resolve_existing_path(
        row.get("tts_path"), manifest_path=manifest_path, repo_root=repo_root, field="tts_path"
    )
    natural_sha256 = _require_sha256(row.get("natural_sha256"), field="natural_sha256")
    tts_sha256 = _require_sha256(row.get("tts_sha256"), field="tts_sha256")
    if file_sha256(natural_path) != natural_sha256:
        raise ValueError("stage natural audio hash mismatch")
    if file_sha256(tts_path) != tts_sha256:
        raise ValueError("stage TTS audio hash mismatch")
    if _require_positive_int(row.get("natural_sample_rate"), field="natural_sample_rate") != TARGET_SAMPLE_RATE:
        raise ValueError("stage natural sample rate must be 16 kHz")
    _require_positive_int(row.get("natural_sample_count"), field="natural_sample_count")
    _require_sha256(row.get("natural_alignment_sha256"), field="natural_alignment_sha256")
    _require_sha256(row.get("tts_alignment_sha256"), field="tts_alignment_sha256")
    resampling = _validate_resampling(row)
    metadata_path = _resolve_existing_path(
        row.get("matched_span_metadata_path"),
        manifest_path=manifest_path,
        repo_root=repo_root,
        field="matched_span_metadata_path",
    )
    if file_sha256(metadata_path) != _require_sha256(
        row.get("matched_span_metadata_sha256"), field="matched_span_metadata_sha256"
    ):
        raise ValueError("stage matched-span metadata hash mismatch")
    metadata_artifact = _strict_json_load(metadata_path)
    if not isinstance(metadata_artifact, Mapping):
        raise ValueError("stage matched-span metadata must be an object")
    metadata = _matched_span_metadata(metadata_artifact)
    spans = row.get("matched_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("stage matched_spans must be a non-empty list")
    if _json_hash(spans) != _require_sha256(row.get("matched_spans_sha256"), field="matched_spans_sha256"):
        raise ValueError("stage matched span hash mismatch")
    min_confidence = _require_finite(
        row.get("minimum_alignment_confidence"), field="minimum_alignment_confidence"
    )
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("minimum_alignment_confidence must be in [0, 1]")
    metadata_spans = metadata.get("spans")
    if not isinstance(metadata_spans, list):
        raise ValueError("stage matched-span metadata requires spans")
    canonical = _validated_matched_spans(
        metadata,
        natural_tokens=[
            {
                "label": _normalized_label(span.get("label")),
                "start_s": _require_finite(span.get("natural_start_s"), field="natural_start_s"),
                "end_s": _require_finite(span.get("natural_end_s"), field="natural_end_s"),
                "confidence": _require_finite(span.get("natural_confidence"), field="natural_confidence"),
            }
            for span in spans
        ],
        tts_tokens=[
            {
                "label": _normalized_label(span.get("label")),
                "start_s": _require_finite(span.get("tts_start_s"), field="tts_start_s"),
                "end_s": _require_finite(span.get("tts_end_s"), field="tts_end_s"),
                "confidence": _require_finite(span.get("tts_confidence"), field="tts_confidence"),
            }
            for span in spans
        ],
        min_confidence=min_confidence,
    )
    if canonical != spans:
        raise ValueError("stage matched spans are not canonical metadata spans")
    provenance = row.get("provenance")
    expected_provenance = {
        "natural": {"condition": "natural", "variant": "raw", "alignment_source": "mfa"},
        "tts": {
            "condition": "tts",
            "variant": "raw",
            "alignment_source": "mfa",
            "tts_provider": "faster_qwen3",
        },
    }
    if provenance != expected_provenance:
        raise ValueError("stage record has incompatible raw/MFA provenance")
    normalized = dict(row)
    normalized["paired_key"] = paired_key
    normalized["split"] = split
    normalized["speaker_group"] = speaker_group
    normalized["transcript"] = transcript
    normalized["natural_path"] = str(natural_path)
    normalized["tts_path"] = str(tts_path)
    normalized["matched_span_metadata_path"] = str(metadata_path)
    normalized["tts_resampling"] = resampling
    return normalized


def _validate_split_isolation(records: Iterable[Mapping[str, Any]]) -> None:
    seen_pairs: set[str] = set()
    split_values: dict[str, dict[str, set[str]]] = {
        "speaker_group": defaultdict(set),
        "natural_path": defaultdict(set),
        "tts_path": defaultdict(set),
        "natural_sha256": defaultdict(set),
        "tts_sha256": defaultdict(set),
        "transcript_sha256": defaultdict(set),
    }
    for row in records:
        paired_key = str(row["paired_key"])
        if paired_key in seen_pairs:
            raise ValueError(f"duplicate paired_key across source splits: {paired_key}")
        seen_pairs.add(paired_key)
        split = str(row["split"])
        values = {
            "speaker_group": str(row["speaker_group"]),
            "natural_path": str(row["natural_path"]),
            "tts_path": str(row["tts_path"]),
            "natural_sha256": str(row["natural_sha256"]),
            "tts_sha256": str(row["tts_sha256"]),
            "transcript_sha256": str(row["transcript_sha256"]),
        }
        for kind, value in values.items():
            split_values[kind][value].add(split)
    for kind, values in split_values.items():
        leaked = sorted(value for value, splits in values.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"cross-split {kind} leakage: {leaked[:3]}")


def build_two_stage_manifest(
    target_manifest_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    parent_path = Path(target_manifest_path).resolve()
    output = Path(output_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else None
    parent = _strict_json_load(parent_path)
    if not isinstance(parent, Mapping):
        raise ValueError("target manifest must be an object")
    if parent.get("schema_version") != TARGET_MANIFEST_SCHEMA_VERSION:
        raise ValueError("target manifest has incompatible schema version")
    if parent.get("target_definition") != TARGET_DEFINITION:
        raise ValueError("target manifest has incompatible target definition")
    source_path = _resolve_existing_path(
        parent.get("source_manifest"),
        manifest_path=parent_path,
        repo_root=root,
        field="source_manifest",
    )
    if file_sha256(source_path) != _require_sha256(
        parent.get("source_manifest_sha256"), field="source_manifest_sha256"
    ):
        raise ValueError("target manifest source manifest hash mismatch")
    source_pairs = _source_pair_index(_source_records(_strict_json_load(source_path)))
    accepted = parent.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("target manifest requires a non-empty accepted list")
    min_confidence = _require_finite(parent.get("min_confidence"), field="min_confidence")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    all_rows: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(accepted):
        if not isinstance(raw, Mapping):
            raise ValueError(f"accepted target row {index} must be an object")
        row, identity = _validate_parent_row(
            raw,
            parent_path=parent_path,
            source_path=source_path,
            source_pairs=source_pairs,
            repo_root=root,
            min_confidence=min_confidence,
        )
        all_rows.append(row)
        identities[row["paired_key"]] = identity
    _validate_split_isolation(all_rows)
    stage_rows = [row for row in all_rows if row["split"] in ALLOWED_STAGE_SPLITS]
    heldout = [identities[row["paired_key"]] for row in all_rows if row["split"] == "heldout"]
    if not stage_rows:
        raise ValueError("no train/valid rows remain after heldout exclusion")
    payload = {
        "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
        "manifest_type": STAGE_MANIFEST_TYPE,
        "source_target_manifest": {
            "path": str(parent_path),
            "sha256": file_sha256(parent_path),
            "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
            "target_definition": TARGET_DEFINITION,
        },
        "contract": STAGE_CONTRACT,
        "records": stage_rows,
        "excluded_provenance": {
            "heldout": heldout,
            "parent_rejected_pair_count": len(parent.get("excluded", [])),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return payload


def load_two_stage_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = Path(manifest_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else None
    payload = _strict_json_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("stage manifest must be an object")
    expected_keys = {
        "schema_version",
        "manifest_type",
        "source_target_manifest",
        "contract",
        "records",
        "excluded_provenance",
    }
    if set(payload) != expected_keys:
        raise ValueError("stage manifest root schema mismatch")
    if payload.get("schema_version") != STAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("stage manifest has incompatible schema version")
    if payload.get("manifest_type") != STAGE_MANIFEST_TYPE:
        raise ValueError("stage manifest has incompatible manifest type")
    if payload.get("contract") != STAGE_CONTRACT:
        raise ValueError("stage manifest has incompatible contract")
    parent = payload.get("source_target_manifest")
    if not isinstance(parent, Mapping) or set(parent) != {
        "path", "sha256", "schema_version", "target_definition"
    }:
        raise ValueError("stage source target manifest provenance is malformed")
    if parent.get("schema_version") != TARGET_MANIFEST_SCHEMA_VERSION or parent.get("target_definition") != TARGET_DEFINITION:
        raise ValueError("stage source target manifest provenance is incompatible")
    parent_path = _resolve_existing_path(
        parent.get("path"), manifest_path=path, repo_root=root, field="source_target_manifest path"
    )
    if file_sha256(parent_path) != _require_sha256(parent.get("sha256"), field="source_target_manifest sha256"):
        raise ValueError("stage source target manifest hash mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("stage manifest requires non-empty records")
    normalized = [
        _validate_stage_record(raw, manifest_path=path, repo_root=root)
        if isinstance(raw, Mapping)
        else (_ for _ in ()).throw(ValueError(f"stage record {index} must be an object"))
        for index, raw in enumerate(records)
    ]
    _validate_split_isolation(normalized)
    excluded = payload.get("excluded_provenance")
    if not isinstance(excluded, Mapping) or set(excluded) != {"heldout", "parent_rejected_pair_count"}:
        raise ValueError("stage excluded provenance is malformed")
    heldout = excluded.get("heldout")
    if not isinstance(heldout, list):
        raise ValueError("stage heldout provenance must be a list")
    for index, identity in enumerate(heldout):
        if not isinstance(identity, Mapping) or set(identity) != {
            "paired_key", "speaker_group", "natural_sha256", "tts_sha256", "transcript_sha256"
        }:
            raise ValueError(f"stage heldout provenance item {index} is malformed")
        _safe_pair_key(identity.get("paired_key"))
        if not str(identity.get("speaker_group") or "").strip():
            raise ValueError("stage heldout provenance speaker_group must be non-empty")
        for field in ("natural_sha256", "tts_sha256", "transcript_sha256"):
            _require_sha256(identity.get(field), field=f"stage heldout {field}")
    if not isinstance(excluded.get("parent_rejected_pair_count"), int) or excluded["parent_rejected_pair_count"] < 0:
        raise ValueError("stage parent_rejected_pair_count must be non-negative")
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    payload = build_two_stage_manifest(
        args.target_manifest,
        args.output_manifest,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "manifest_type": payload["manifest_type"],
                "train_valid_pair_count": len(payload["records"]),
                "excluded_heldout_pair_count": len(payload["excluded_provenance"]["heldout"]),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
