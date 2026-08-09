#!/usr/bin/env python3
"""Construct exact-length TTS-like targets on a natural utterance timeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import resample


@dataclass(frozen=True)
class Span:
    """A labelled time span in seconds."""

    label: str
    start_s: float
    end_s: float
    confidence: float = 1.0


@dataclass(frozen=True)
class MatchedSpan:
    """A natural span paired with a same-label TTS span."""

    label: str
    natural_start_s: float
    natural_end_s: float
    tts_start_s: float
    tts_end_s: float
    natural_confidence: float
    tts_confidence: float


@dataclass(frozen=True)
class WarpResult:
    """Warped waveform and auditable construction metadata."""

    audio: np.ndarray
    matched_spans: tuple[MatchedSpan, ...]
    metadata: dict[str, Any]


def _span_value(span: Span | Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(span, Mapping):
        return span.get(name, default)
    return getattr(span, name, default)


def as_spans(spans: Iterable[Span | Mapping[str, Any] | Any]) -> list[Span]:
    """Normalize alignment records from dicts, dataclasses, or objects."""
    normalized: list[Span] = []
    for item in spans:
        label = _span_value(item, "token", _span_value(item, "label", ""))
        start = _span_value(item, "start_s")
        end = _span_value(item, "end_s")
        confidence = _span_value(item, "confidence", 1.0)
        if start is None or end is None:
            raise ValueError("each span requires start_s and end_s")
        normalized.append(
            Span(
                label=str(label),
                start_s=float(start),
                end_s=float(end),
                confidence=float(confidence),
            )
        )
    return normalized


def _validate_spans(spans: Sequence[Span], name: str) -> None:
    previous_end = -np.inf
    for span in spans:
        if not np.isfinite(span.start_s) or not np.isfinite(span.end_s):
            raise ValueError(f"{name} contains a non-finite span")
        if span.start_s < 0 or span.end_s <= span.start_s:
            raise ValueError(f"{name} contains an invalid span: {span}")
        if span.start_s < previous_end - 1e-7:
            raise ValueError(f"{name} must be sorted and non-overlapping")
        if not 0.0 <= span.confidence <= 1.0:
            raise ValueError(f"{name} confidence must be in [0, 1]")
        previous_end = span.end_s


def match_token_spans(
    natural_spans: Iterable[Span | Mapping[str, Any] | Any],
    tts_spans: Iterable[Span | Mapping[str, Any] | Any],
    *,
    min_confidence: float = 0.8,
) -> tuple[list[MatchedSpan], dict[str, int]]:
    """Match repeated labels in order without crossing token boundaries."""
    natural = as_spans(natural_spans)
    tts = as_spans(tts_spans)
    _validate_spans(natural, "natural_spans")
    _validate_spans(tts, "tts_spans")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    matches: list[MatchedSpan] = []
    tts_index = 0
    confidence_rejected = 0
    label_mismatch = 0
    for nat in natural:
        if nat.confidence < min_confidence:
            confidence_rejected += 1
            continue
        candidate_index = next(
            (index for index in range(tts_index, len(tts)) if tts[index].label == nat.label),
            None,
        )
        if candidate_index is None:
            continue
        label_mismatch += candidate_index - tts_index
        tts_index = candidate_index + 1
        candidate = tts[candidate_index]
        if candidate.confidence < min_confidence:
            confidence_rejected += 1
            continue
        matches.append(
            MatchedSpan(
                label=nat.label,
                natural_start_s=nat.start_s,
                natural_end_s=nat.end_s,
                tts_start_s=candidate.start_s,
                tts_end_s=candidate.end_s,
                natural_confidence=nat.confidence,
                tts_confidence=candidate.confidence,
            )
        )

    stats = {
        "natural_count": len(natural),
        "tts_count": len(tts),
        "matched_count": len(matches),
        "unmatched_natural_count": len(natural) - len(matches),
        "confidence_rejected_count": confidence_rejected,
        "label_mismatch_count": label_mismatch,
    }
    return matches, stats


def local_resample(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a mono segment to exactly ``target_length`` samples."""
    source = np.asarray(audio, dtype=np.float32).reshape(-1)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if target_length == 0:
        return np.empty(0, dtype=np.float32)
    if source.size == 0:
        raise ValueError("cannot resample an empty source segment")
    if source.size == target_length:
        return source.copy()
    if source.size == 1:
        return np.full(target_length, source[0], dtype=np.float32)
    return np.asarray(resample(source, target_length), dtype=np.float32)


def _sample_bounds(start_s: float, end_s: float, sample_rate: int, length: int) -> tuple[int, int]:
    start = int(np.rint(start_s * sample_rate))
    end = int(np.rint(end_s * sample_rate))
    start = max(0, min(length, start))
    end = max(start, min(length, end))
    return start, end


def _smooth_boundaries(audio: np.ndarray, boundaries: Sequence[int], fade_samples: int) -> None:
    """Blend the left side of each splice toward the right side in place."""
    if fade_samples <= 0:
        return
    n = len(audio)
    for boundary in sorted(set(boundaries)):
        if boundary <= 0 or boundary >= n:
            continue
        fade = min(fade_samples, boundary, n - boundary)
        if fade < 2:
            continue
        left = audio[boundary - fade : boundary].copy()
        right = audio[boundary : boundary + fade].copy()
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[boundary - fade : boundary] = left * (1.0 - ramp) + right * ramp


def identity_target(natural_audio: np.ndarray) -> np.ndarray:
    """Return an exact copy of the natural waveform."""
    audio = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    return audio.copy()


def global_duration_match(tts_audio: np.ndarray, target_length: int) -> np.ndarray:
    """Create the duration-only control by globally resampling TTS."""
    return local_resample(tts_audio, target_length)


def phone_local_warp(
    natural_audio: np.ndarray,
    tts_audio: np.ndarray,
    sample_rate: int,
    natural_spans: Iterable[Span | Mapping[str, Any] | Any],
    tts_spans: Iterable[Span | Mapping[str, Any] | Any],
    *,
    min_confidence: float = 0.8,
    crossfade_ms: float = 8.0,
    alignment_source: str = "mfa",
) -> WarpResult:
    """Copy TTS phone-local acoustics onto the natural sample timeline.

    Natural samples outside matched phone spans remain unchanged. Boundary
    smoothing only touches the immediate splice neighbourhood, so pause
    interiors and the output sample count are preserved.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    natural = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    tts = np.asarray(tts_audio, dtype=np.float32).reshape(-1)
    if natural.size == 0:
        raise ValueError("natural_audio must not be empty")
    if tts.size == 0:
        raise ValueError("tts_audio must not be empty")
    if crossfade_ms < 0:
        raise ValueError("crossfade_ms must be non-negative")

    matches, stats = match_token_spans(
        natural_spans, tts_spans, min_confidence=min_confidence
    )
    output = natural.copy()
    splice_boundaries: list[int] = []
    used_spans: list[dict[str, Any]] = []
    for match in matches:
        nat_start, nat_end = _sample_bounds(
            match.natural_start_s, match.natural_end_s, sample_rate, len(natural)
        )
        tts_start, tts_end = _sample_bounds(
            match.tts_start_s, match.tts_end_s, sample_rate, len(tts)
        )
        target_length = nat_end - nat_start
        source = tts[tts_start:tts_end]
        if target_length <= 0 or source.size <= 0:
            continue
        output[nat_start:nat_end] = local_resample(source, target_length)
        splice_boundaries.extend((nat_start, nat_end))
        used_spans.append(
            {
                **asdict(match),
                "natural_start_sample": nat_start,
                "natural_end_sample": nat_end,
                "tts_start_sample": tts_start,
                "tts_end_sample": tts_end,
            }
        )

    fade_samples = int(round(sample_rate * crossfade_ms / 1000.0))
    _smooth_boundaries(output, splice_boundaries, fade_samples)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    metadata: dict[str, Any] = {
        **stats,
        "alignment_source": alignment_source,
        "min_confidence": min_confidence,
        "crossfade_ms": crossfade_ms,
        "crossfade_samples": fade_samples,
        "sample_rate": sample_rate,
        "natural_sample_count": int(natural.size),
        "output_sample_count": int(output.size),
        "exact_sample_count": bool(output.size == natural.size),
        "natural_duration_s": float(natural.size / sample_rate),
        "tts_duration_s": float(tts.size / sample_rate),
        "duration_ratio_tts_to_natural": float(tts.size / natural.size),
        "used_span_count": len(used_spans),
        "spans": used_spans,
    }
    return WarpResult(audio=output, matched_spans=tuple(matches), metadata=metadata)


def load_alignment_records(path: str | Path, *, condition: str | None = None) -> list[dict[str, Any]]:
    """Load records from a JSON alignment manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError("alignment manifest must contain a records list")
    if condition is None:
        return [dict(record) for record in records]
    return [dict(record) for record in records if record.get("condition") == condition]


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--tts", type=Path, required=True)
    parser.add_argument("--natural-spans", type=Path, required=True)
    parser.add_argument("--tts-spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--crossfade-ms", type=float, default=8.0)
    parser.add_argument("--metadata", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import soundfile as sf

    args = _cli().parse_args(argv)
    natural, natural_sr = sf.read(args.natural, dtype="float32", always_2d=False)
    tts, tts_sr = sf.read(args.tts, dtype="float32", always_2d=False)
    if natural.ndim != 1 or tts.ndim != 1:
        raise ValueError("input WAV files must be mono")
    if natural_sr != args.sample_rate or tts_sr != args.sample_rate:
        raise ValueError("input WAV sample rates must equal --sample-rate")
    natural_spans = json.loads(args.natural_spans.read_text(encoding="utf-8"))
    tts_spans = json.loads(args.tts_spans.read_text(encoding="utf-8"))
    result = phone_local_warp(
        natural,
        tts,
        args.sample_rate,
        natural_spans,
        tts_spans,
        min_confidence=args.min_confidence,
        crossfade_ms=args.crossfade_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, result.audio, args.sample_rate, subtype="PCM_16")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    print(json.dumps(result.metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
