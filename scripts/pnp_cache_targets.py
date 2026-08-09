#!/usr/bin/env python3
"""CPU-only cache and provenance audit for paired waveform targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pnp_alignment_warp import Span, match_token_spans, phone_local_warp


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return a deterministic content hash without loading the whole file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError("manifest must contain a records list")
    return [dict(record) for record in records]


def _resolve_audio_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    if repo_root is not None:
        rooted = Path(repo_root) / candidate
        if rooted.exists():
            return rooted
    raise FileNotFoundError(candidate)


def _record_spans(record: Mapping[str, Any]) -> list[Span]:
    return [
        Span(
            label=str(token.get("token", token.get("label", ""))),
            start_s=float(token["start_s"]),
            end_s=float(token["end_s"]),
            confidence=float(token.get("confidence", record.get("alignment_confidence", 1.0))),
        )
        for token in record.get("tokens", [])
    ]


def pair_records(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Group records by paired key and condition/provider."""
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_record in records:
        record = dict(raw_record)
        key = str(record.get("paired_key", record.get("sample_id", record.get("utterance_id"))))
        condition = str(record.get("condition", ""))
        provider = str(record.get("tts_provider") or condition)
        group = pairs.setdefault(key, {})
        group[provider if condition == "tts" else "natural"] = record
    return pairs


def _audio_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float | int | bool]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    return {
        "sample_rate": int(sample_rate),
        "sample_count": int(values.size),
        "duration_s": float(values.size / sample_rate),
        "peak": peak,
        "rms": rms,
        "finite": bool(np.isfinite(values).all()),
        "clipped_sample_count": int(np.count_nonzero(np.abs(values) >= 1.0)),
    }


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio for the common enhancer interface."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_sr == target_sr:
        return values
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    from math import gcd

    divisor = gcd(source_sr, target_sr)
    return np.asarray(
        resample_poly(values, target_sr // divisor, source_sr // divisor),
        dtype=np.float32,
    )


def audit_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    min_confidence: float = 0.8,
    target_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Audit paths, provenance, alignment coverage, and duration ratios."""
    records = _load_manifest(manifest_path)
    pairs = pair_records(records)
    rows: list[dict[str, Any]] = []
    missing = 0
    for pair_key, group in sorted(pairs.items()):
        natural_record = group.get("natural")
        if natural_record is None:
            continue
        natural_path = _resolve_audio_path(natural_record["audio_path"], repo_root)
        natural_audio, natural_sr = sf.read(natural_path, dtype="float32", always_2d=False)
        if natural_audio.ndim != 1:
            raise ValueError(f"natural audio must be mono: {natural_path}")
        natural_stats = _audio_stats(natural_audio, natural_sr)
        if target_sample_rate is not None:
            natural_audio_for_ratio = _resample_audio(natural_audio, natural_sr, target_sample_rate)
        else:
            natural_audio_for_ratio = natural_audio
        for provider, tts_record in sorted(group.items()):
            if provider == "natural":
                continue
            try:
                tts_path = _resolve_audio_path(tts_record["audio_path"], repo_root)
            except FileNotFoundError:
                missing += 1
                rows.append({"paired_key": pair_key, "provider": provider, "missing": True})
                continue
            tts_audio, tts_sr = sf.read(tts_path, dtype="float32", always_2d=False)
            if tts_audio.ndim != 1:
                raise ValueError(f"TTS audio must be mono: {tts_path}")
            if target_sample_rate is not None:
                tts_audio_for_ratio = _resample_audio(tts_audio, tts_sr, target_sample_rate)
            else:
                tts_audio_for_ratio = tts_audio
            natural_spans = _record_spans(natural_record)
            tts_spans = _record_spans(tts_record)
            matches, match_stats = match_token_spans(
                natural_spans,
                tts_spans,
                min_confidence=min_confidence,
            )
            row = {
                "paired_key": pair_key,
                "provider": provider,
                "sample_id": natural_record.get("sample_id"),
                "speaker_id": natural_record.get("speaker_id"),
                "transcript": natural_record.get("transcript"),
                "natural_path": str(natural_path),
                "tts_path": str(tts_path),
                "natural_sha256": file_sha256(natural_path),
                "tts_sha256": file_sha256(tts_path),
                "natural": natural_stats,
                "tts": _audio_stats(tts_audio, tts_sr),
                "natural_token_count": len(natural_spans),
                "tts_token_count": len(tts_spans),
                "matched_token_count": len(matches),
                "unmatched_natural_count": match_stats["unmatched_natural_count"],
                "confidence_rejected_count": match_stats["confidence_rejected_count"],
                "label_mismatch_count": match_stats["label_mismatch_count"],
                "natural_alignment_source": natural_record.get("alignment_source"),
                "tts_alignment_source": tts_record.get("alignment_source"),
                "natural_uniform_fallback": natural_record.get("alignment_source") == "uniform_fallback",
                "tts_uniform_fallback": tts_record.get("alignment_source") == "uniform_fallback",
                "duration_ratio_tts_to_natural": float(
                    len(tts_audio_for_ratio) / max(1, len(natural_audio_for_ratio))
                ),
            }
            row["coverage"] = float(len(matches) / max(1, len(natural_spans)))
            rows.append(row)
    provider_summary: dict[str, dict[str, int | float]] = {}
    for row in rows:
        if row.get("missing"):
            continue
        provider = str(row["provider"])
        summary = provider_summary.setdefault(provider, {"pairs": 0, "low_coverage": 0, "uniform_fallback": 0})
        summary["pairs"] += 1
        summary["low_coverage"] += int(float(row["coverage"]) < 0.9)
        summary["uniform_fallback"] += int(
            bool(row["natural_uniform_fallback"]) or bool(row["tts_uniform_fallback"])
        )
    return {
        "schema_version": 1,
        "manifest_path": str(Path(manifest_path)),
        "min_confidence": min_confidence,
        "pair_count": len(pairs),
        "row_count": len(rows),
        "missing_audio_count": missing,
        "provider_summary": provider_summary,
        "rows": rows,
    }


def cache_phone_local_targets(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    min_confidence: float = 0.8,
    crossfade_ms: float = 8.0,
    provider: str | None = None,
    target_sample_rate: int = 16_000,
) -> dict[str, Any]:
    """Write warped WAVs and metadata for existing paired records."""
    records = _load_manifest(manifest_path)
    pairs = pair_records(records)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cached: list[dict[str, Any]] = []
    for pair_key, group in sorted(pairs.items()):
        natural_record = group.get("natural")
        if natural_record is None:
            continue
        natural_path = _resolve_audio_path(natural_record["audio_path"], repo_root)
        natural_audio, natural_source_sr = sf.read(natural_path, dtype="float32", always_2d=False)
        natural_sr = target_sample_rate
        natural_audio = _resample_audio(natural_audio, natural_source_sr, natural_sr)
        for candidate_provider, tts_record in sorted(group.items()):
            if candidate_provider == "natural" or (provider and candidate_provider != provider):
                continue
            tts_path = _resolve_audio_path(tts_record["audio_path"], repo_root)
            tts_audio, tts_source_sr = sf.read(tts_path, dtype="float32", always_2d=False)
            tts_sr = target_sample_rate
            tts_audio = _resample_audio(tts_audio, tts_source_sr, tts_sr)
            result = phone_local_warp(
                natural_audio,
                tts_audio,
                natural_sr,
                _record_spans(natural_record),
                _record_spans(tts_record),
                min_confidence=min_confidence,
                crossfade_ms=crossfade_ms,
                alignment_source=str(tts_record.get("alignment_source", "unknown")),
            )
            key = f"{pair_key}_{candidate_provider}"
            audio_path = root / f"{key}.wav"
            metadata_path = root / f"{key}.json"
            sf.write(audio_path, result.audio, natural_sr, subtype="PCM_16")
            metadata = {
                "schema_version": 1,
                "target_type": "tts_phone_local_warp",
                "paired_key": pair_key,
                "provider": candidate_provider,
                "source_manifest": str(Path(manifest_path)),
                "natural_path": str(natural_path),
                "tts_path": str(tts_path),
                "natural_sha256": file_sha256(natural_path),
                "tts_sha256": file_sha256(tts_path),
                "output_sha256": file_sha256(audio_path),
                "natural_record": {k: v for k, v in natural_record.items() if k != "tokens"},
                "tts_record": {k: v for k, v in tts_record.items() if k != "tokens"},
                "natural_source_sample_rate": int(natural_source_sr),
                "tts_source_sample_rate": int(tts_source_sr),
                "target_sample_rate": int(target_sample_rate),
                **result.metadata,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            cached.append(
                {
                    "paired_key": pair_key,
                    "provider": candidate_provider,
                    "natural_path": str(natural_path),
                    "tts_path": str(tts_path),
                    "audio_path": str(audio_path),
                    "metadata_path": str(metadata_path),
                    **result.metadata,
                }
            )
    summary = {
        "schema_version": 1,
        "target_type": "tts_phone_local_warp",
        "manifest_path": str(Path(manifest_path)),
        "output_dir": str(root),
        "cached_count": len(cached),
        "items": cached,
    }
    (root / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("manifest", type=Path)
    audit_parser.add_argument("--repo-root", type=Path)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--min-confidence", type=float, default=0.8)
    audit_parser.add_argument("--target-sample-rate", type=int)
    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("manifest", type=Path)
    cache_parser.add_argument("output_dir", type=Path)
    cache_parser.add_argument("--repo-root", type=Path)
    cache_parser.add_argument("--provider")
    cache_parser.add_argument("--min-confidence", type=float, default=0.8)
    cache_parser.add_argument("--crossfade-ms", type=float, default=8.0)
    cache_parser.add_argument("--target-sample-rate", type=int, default=16000)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit_manifest(
            args.manifest,
            repo_root=args.repo_root,
            min_confidence=args.min_confidence,
            target_sample_rate=args.target_sample_rate,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        result = cache_phone_local_targets(
            args.manifest,
            args.output_dir,
            repo_root=args.repo_root,
            min_confidence=args.min_confidence,
            crossfade_ms=args.crossfade_ms,
            provider=args.provider,
            target_sample_rate=args.target_sample_rate,
        )
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "items"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
