#!/usr/bin/env python3
"""Build valid-only pause-allocation counterfactuals from TTS waveforms.

Speech samples, speech-span durations, total gap samples, and output length are
preserved exactly in memory. Only the allocation of existing gap samples among
phone boundaries changes. These are diagnostic controls, not training targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import librosa
import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pnp_alignment_warp import Span
from pnp_feature_controls import reallocate_pause_samples

REPO = SCRIPT_DIR.parent
DEFAULT_MANIFEST = REPO / "runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json"
DEFAULT_OUTPUT = REPO / "runs/rhythm_timing/20260812_pause_counterfactual_valid50"
LEGACY_ROOT = "/mnt/e/Documents/tts-audio/tts-exp"
TARGET_SR = 16_000
VALID_SPEAKER = "S0765"
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>", "h#", "pau", "pause"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw_path: str | Path) -> Path:
    raw = str(raw_path)
    candidates = [Path(raw)]
    if raw.startswith(LEGACY_ROOT + "/"):
        candidates.append(REPO / raw[len(LEGACY_ROOT) + 1 :])
    elif not Path(raw).is_absolute():
        candidates.append(REPO / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    values, source_rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if source_rate != TARGET_SR:
        values = librosa.resample(values, orig_sr=source_rate, target_sr=TARGET_SR)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"empty or non-finite audio: {path}")
    return values, int(source_rate)


def is_silence(token: Mapping[str, Any]) -> bool:
    if bool(token.get("is_silence")) or bool(token.get("is_non_speech")):
        return True
    label = str(token.get("token", token.get("raw_token", ""))).strip().lower()
    return label in SILENCE_LABELS


def speech_spans(record: Mapping[str, Any]) -> list[Span]:
    result: list[Span] = []
    for token in record.get("tokens", []):
        if is_silence(token):
            continue
        start = float(token["start_s"])
        end = float(token["end_s"])
        if end <= start:
            continue
        result.append(
            Span(
                label=str(token.get("token", token.get("raw_token", ""))),
                start_s=start,
                end_s=end,
                confidence=float(token.get("confidence", 1.0)),
            )
        )
    return result


def sample_bounds(span: Span, length: int) -> tuple[int, int]:
    start = max(0, min(length, int(np.rint(span.start_s * TARGET_SR))))
    end = max(start, min(length, int(np.rint(span.end_s * TARGET_SR))))
    return start, end


def gap_counts(spans: Sequence[Span], length: int) -> list[int]:
    counts: list[int] = []
    cursor = 0
    for span in spans:
        start, end = sample_bounds(span, length)
        if start < cursor:
            raise ValueError("speech spans overlap after conversion to samples")
        counts.append(start - cursor)
        cursor = end
    counts.append(length - cursor)
    return counts


def allocate_proportions(weights: Sequence[float], total: int) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("weights must be a finite non-negative vector")
    if float(values.sum()) <= 0.0:
        values = np.ones_like(values)
    raw = values / values.sum() * total
    result = np.floor(raw).astype(np.int64)
    remainder = int(total - int(result.sum()))
    order = np.argsort(-(raw - result), kind="stable")
    for index in order[:remainder]:
        result[index] += 1
    return [int(value) for value in result]


def target_from_natural(natural_counts: Sequence[int], tts_total_gap: int) -> list[int]:
    return allocate_proportions([float(value) for value in natural_counts], tts_total_gap)


def interpolate_counts(source: Sequence[int], target: Sequence[int], alpha: float) -> list[int]:
    if len(source) != len(target):
        raise ValueError("source and target gap vectors must have equal length")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    total = int(sum(source))
    if int(sum(target)) != total:
        raise ValueError("source and target gap totals must match")
    if alpha == 0.0:
        return [int(value) for value in source]
    if alpha == 1.0:
        return [int(value) for value in target]
    weights = [(1.0 - alpha) * source_value + alpha * target_value for source_value, target_value in zip(source, target)]
    return allocate_proportions(weights, total)


def internal_to_edges(source: Sequence[int]) -> list[int]:
    if len(source) < 2:
        return [int(value) for value in source]
    internal = int(sum(source[1:-1]))
    leading = int(source[0]) + internal // 2
    trailing = int(source[-1]) + internal - internal // 2
    return [leading, *([0] * max(0, len(source) - 2)), trailing]


def reverse_internal(source: Sequence[int]) -> list[int]:
    if len(source) <= 2:
        return [int(value) for value in source]
    return [int(source[0]), *[int(value) for value in reversed(source[1:-1])], int(source[-1])]


def build_pairs(manifest_path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in payload.get("records", []):
        if raw.get("split") != "valid" or raw.get("speaker_id") != VALID_SPEAKER:
            continue
        condition = str(raw.get("condition", ""))
        if condition not in {"natural", "tts"}:
            continue
        groups.setdefault(str(raw["paired_key"]), {})[condition] = dict(raw)
    pairs = [(groups[key]["natural"], groups[key]["tts"]) for key in sorted(groups) if {"natural", "tts"} <= groups[key].keys()]
    if len(pairs) != 50:
        raise ValueError(f"expected 50 valid pairs, found {len(pairs)}")
    return pairs


def output_metrics(values: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    return {
        "sample_count": int(len(values)),
        "exact_sample_count": bool(len(values) == len(source)),
        "finite": bool(np.isfinite(values).all()),
        "peak": float(np.max(np.abs(values))) if len(values) else 0.0,
        "clipped_samples": int(np.sum(np.abs(values) >= 1.0)),
        "source_sample_multiset_conserved": bool(np.array_equal(np.sort(values), np.sort(source))),
    }


def run(manifest_path: Path, output_dir: Path, count: int) -> dict[str, Any]:
    pairs = build_pairs(manifest_path)[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for pair_index, (natural_record, tts_record) in enumerate(pairs, 1):
        sample_id = int(natural_record["sample_id"])
        key = str(natural_record["paired_key"])
        natural_path = resolve_path(natural_record["audio_path"])
        tts_path = resolve_path(tts_record["audio_path"])
        natural, natural_source_rate = load_audio(natural_path)
        tts, tts_source_rate = load_audio(tts_path)
        natural_spans = speech_spans(natural_record)
        tts_spans = speech_spans(tts_record)
        natural_labels = [span.label for span in natural_spans]
        tts_labels = [span.label for span in tts_spans]
        label_match_count = sum(natural == tts for natural, tts in zip(natural_labels, tts_labels))
        label_match_coverage = label_match_count / max(len(natural_labels), len(tts_labels), 1)
        if len(natural_spans) != len(tts_spans) or label_match_coverage < 0.9:
            rejected.append(
                {
                    "sample_id": sample_id,
                    "paired_key": key,
                    "reason": "incompatible_speech_boundary_slots",
                    "natural_phone_count": len(natural_spans),
                    "tts_phone_count": len(tts_spans),
                    "label_match_coverage": label_match_coverage,
                }
            )
            continue
        source_counts = gap_counts(tts_spans, len(tts))
        natural_counts = gap_counts(natural_spans, len(natural))
        natural_target = target_from_natural(natural_counts, sum(source_counts))
        conditions = {
            "pause_alpha_0": source_counts,
            "pause_alpha_0.5": interpolate_counts(source_counts, natural_target, 0.5),
            "pause_alpha_1": natural_target,
            "pause_internal_to_edges": internal_to_edges(source_counts),
            "pause_reverse": reverse_internal(source_counts),
        }
        for condition, target_counts in conditions.items():
            result = reallocate_pause_samples(tts, TARGET_SR, tts_spans, target_counts)
            if condition == "pause_alpha_0" and not np.array_equal(result.audio, tts):
                raise AssertionError(f"pause alpha 0 is not identity for sample {sample_id}")
            output_path = output_dir / "audio" / f"{sample_id:04d}_{condition}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, result.audio, TARGET_SR, subtype="PCM_16")
            metadata = {
                "schema_version": 1,
                "evaluation_role": "pause_allocation_counterfactual_not_training_target",
                "sample_id": sample_id,
                "paired_key": key,
                "speaker_id": VALID_SPEAKER,
                "split": "valid",
                "condition": condition,
                "natural_source": str(natural_path),
                "tts_source": str(tts_path),
                "natural_source_rate": natural_source_rate,
                "tts_source_rate": tts_source_rate,
                "processing_rate": TARGET_SR,
                "natural_source_sha256": sha256_file(natural_path),
                "tts_source_sha256": sha256_file(tts_path),
                "speech_phone_count": len(tts_spans),
                "phone_label_match_count": label_match_count,
                "phone_label_match_coverage": label_match_coverage,
                "source_gap_sample_counts": source_counts,
                "natural_normalized_target_gap_sample_counts": natural_target,
                "target_gap_sample_counts": target_counts,
                "source_total_gap_samples": int(sum(source_counts)),
                "target_total_gap_samples": int(sum(target_counts)),
                "output_path": str(output_path),
                "output_sha256": sha256_file(output_path),
                "transform_metadata": result.metadata,
                "audio": output_metrics(result.audio, tts),
                "heldout_excluded": True,
            }
            metadata_path = output_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(metadata)
            print(
                f"OK {pair_index}/{len(pairs)} {condition} {sample_id} gaps={sum(value > 0 for value in target_counts)}",
                flush=True,
            )
    payload = {
        "schema_version": 1,
        "evaluation": "valid_only_pause_allocation_counterfactuals",
        "valid_speaker": VALID_SPEAKER,
        "heldout_excluded": True,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "count_requested": count,
        "pair_count": len(pairs) - len(rejected),
        "conditions": sorted({row["condition"] for row in rows}),
        "records": rows,
        "rejected": rejected,
        "invariants": {
            "speech_waveform_samples": "preserved exactly in memory",
            "speech_span_durations": "preserved exactly",
            "total_pause_samples": "preserved exactly",
            "output_sample_count": "preserved exactly",
            "gap_waveform_samples": "preserved as a multiset and used once",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE rows={len(rows)} rejected={len(rejected)} output={output_dir / 'summary.json'}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.count <= 50:
        raise SystemExit("--count must be in [1,50]")
    manifest = args.manifest if args.manifest.is_absolute() else (REPO / args.manifest).resolve()
    output = args.output_dir if args.output_dir.is_absolute() else (REPO / args.output_dir).resolve()
    result = run(manifest, output, args.count)
    return 0 if not result["rejected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
