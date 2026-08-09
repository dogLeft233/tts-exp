#!/usr/bin/env python3
"""Prepare a local paired rhythm/style dataset for the acoustic MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pnp_alignment_warp import Span, match_token_spans, phone_local_warp

TARGET_SR = 16_000
FRAME_HOP = 320
N_FFT = 512
N_MELS = 80
STYLE_INPUT_DIM = 84
RHYTHM_DIM = 6
CONTENT_DIM = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    candidates = [candidate]
    if repo_root is not None:
        root = Path(repo_root).resolve()
        candidates.extend([root / candidate, root / candidate.as_posix().lstrip("/")])
    for item in candidates:
        if item.exists():
            return item.resolve()
    raise FileNotFoundError(f"audio path does not exist: {path}")


def records_from_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError("manifest must contain a records list")
    return [dict(record) for record in records]


def pair_records(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in records:
        record = dict(raw)
        key = str(record.get("paired_key", record.get("sample_id", "")))
        if not key:
            raise ValueError("each alignment record needs paired_key or sample_id")
        condition = str(record.get("condition", ""))
        if condition == "natural":
            slot = "natural"
        elif condition == "tts":
            provider = str(record.get("tts_provider") or "tts")
            slot = provider
        else:
            continue
        if slot in pairs.setdefault(key, {}):
            raise ValueError(f"duplicate {slot} record for pair {key}")
        pairs[key][slot] = record
    return pairs


def spans(record: Mapping[str, Any]) -> list[Span]:
    return [
        Span(
            label=str(token.get("token", token.get("label", ""))),
            start_s=float(token["start_s"]),
            end_s=float(token["end_s"]),
            confidence=float(token.get("confidence", record.get("alignment_confidence", 1.0))),
        )
        for token in record.get("tokens", [])
    ]


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if np.asarray(audio).ndim != 1:
        raise ValueError(f"audio must be mono: {path}")
    values = np.asarray(audio, dtype=np.float32)
    if sample_rate != TARGET_SR:
        values = librosa.resample(values, orig_sr=sample_rate, target_sr=TARGET_SR)
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return values


def _frame_count(sample_count: int) -> int:
    return max(1, int(math.ceil(sample_count / FRAME_HOP)))


def _pad_or_trim(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) >= count:
        return values[:count]
    if len(values) == 0:
        return np.zeros(count, dtype=np.float32)
    return np.pad(values, (0, count - len(values)), mode="edge")


def _frame_times(count: int) -> np.ndarray:
    return (np.arange(count, dtype=np.float32) * FRAME_HOP + FRAME_HOP / 2) / TARGET_SR


def extract_style_features(audio: np.ndarray) -> np.ndarray:
    """Return [frames, 84] log-mel plus four normalized prosody channels."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_fft=N_FFT, hop_length=FRAME_HOP,
        win_length=N_FFT, n_mels=N_MELS, power=2.0, center=True,
    )
    log_mel = np.log(np.maximum(mel, 1e-7)).T.astype(np.float32)
    rms = librosa.feature.rms(
        y=audio, frame_length=N_FFT, hop_length=FRAME_HOP, center=True
    )[0]
    log_rms = np.log(np.maximum(rms, 1e-7)).astype(np.float32)
    try:
        f0 = librosa.yin(
            audio, fmin=50.0, fmax=600.0, sr=TARGET_SR,
            frame_length=N_FFT, hop_length=FRAME_HOP,
        ).astype(np.float32)
    except (ValueError, RuntimeError):
        f0 = np.zeros(max(1, len(log_rms)), dtype=np.float32)
    voiced = np.isfinite(f0) & (f0 > 50.0) & (f0 < 600.0) & (rms > 1e-4)
    log_f0 = np.zeros_like(f0, dtype=np.float32)
    log_f0[voiced] = np.log(f0[voiced])
    if voiced.any():
        voiced_values = log_f0[voiced]
        log_f0[voiced] = (voiced_values - voiced_values.mean()) / max(float(voiced_values.std()), 1e-4)
    energy_derivative = np.diff(log_rms, prepend=log_rms[:1])
    count = max(len(log_mel), len(log_rms), len(log_f0))
    channels = np.stack(
        [
            _pad_or_trim(log_rms, count),
            _pad_or_trim(log_f0, count),
            _pad_or_trim(voiced.astype(np.float32), count),
            _pad_or_trim(energy_derivative, count),
        ], axis=1,
    )
    features = np.concatenate([_pad_or_trim_2d(log_mel, count), channels], axis=1)
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _pad_or_trim_2d(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[0] >= count:
        return values[:count]
    if values.shape[0] == 0:
        return np.zeros((count, values.shape[1]), dtype=np.float32)
    return np.pad(values, ((0, count - values.shape[0]), (0, 0)), mode="edge")


def select_embedding_layer(array: np.ndarray, metadata: Mapping[str, Any], layer: int = 11) -> np.ndarray:
    layers = [int(value) for value in metadata.get("layers", [])]
    if layer not in layers:
        raise ValueError(f"embedding sidecar does not contain layer {layer}")
    values = np.asarray(array)
    if values.ndim == 2:
        if len(layers) != 1:
            raise ValueError("2-D embedding requires exactly one sidecar layer")
        selected = values
    elif values.ndim == 3:
        index = layers.index(layer)
        if values.shape[0] == len(layers):
            selected = values[index]
        elif values.shape[1] == len(layers):
            selected = values[:, index]
        else:
            raise ValueError(f"cannot identify layer axis in embedding shape {values.shape}")
    else:
        raise ValueError(f"embedding must be 2-D or 3-D, got {values.shape}")
    if selected.ndim != 2 or selected.shape[1] != CONTENT_DIM:
        raise ValueError(f"selected content must have shape [frames,{CONTENT_DIM}]")
    return np.asarray(selected, dtype=np.float32)


def align_content(content: np.ndarray, frame_count: int) -> np.ndarray:
    if content.shape[0] == frame_count:
        return content
    source = torch.from_numpy(content.T).unsqueeze(0)
    aligned = torch.nn.functional.interpolate(source, size=frame_count, mode="linear", align_corners=False)
    return aligned.squeeze(0).T.numpy().astype(np.float32)


def embedding_for_sample(
    embedding_dir: Path,
    sample_id: int,
    condition: str,
    record: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if condition == "natural":
        legacy_stem = f"{sample_id}:natural:raw_natural_raw_hubert"
        safe_stems = [f"{sample_id}_natural_raw_hubert"]
    else:
        legacy_stem = f"{sample_id}:tts:raw_faster_qwen3_raw_hubert"
        provider = str((record or {}).get("tts_provider") or "faster_qwen3")
        safe_stems = [f"{sample_id}_{provider}_raw_hubert", f"{sample_id}_tts_raw_hubert"]
    candidates = [embedding_dir / f"{legacy_stem}.npy"]
    candidates.extend(embedding_dir / f"{stem}.npy" for stem in safe_stems)
    array_path = next((path for path in candidates if path.exists()), None)
    if array_path is None:
        raise FileNotFoundError(f"missing HuBERT cache for {sample_id} {condition}")
    sidecar_path = array_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"missing HuBERT sidecar for {array_path.name}")
    metadata = load_json(sidecar_path)
    if metadata.get("condition") not in {condition, "faster_qwen3" if condition == "tts" else condition}:
        raise ValueError(f"embedding condition mismatch for {array_path.name}")
    return select_embedding_layer(np.load(array_path), metadata, layer=11)


def _find_token_index(token_spans: Sequence[Span], time_s: float) -> int | None:
    for index, span in enumerate(token_spans):
        if span.start_s <= time_s < span.end_s:
            return index
    return None


def _tts_progress_from_matches(
    natural_time_s: float,
    matches: Sequence[Any],
) -> float:
    """Map natural utterance time to the actual TTS token-local progress."""
    if not matches:
        return 0.0
    natural_total = sum(max(match.natural_end_s - match.natural_start_s, 1e-6) for match in matches)
    tts_total = sum(max(match.tts_end_s - match.tts_start_s, 1e-6) for match in matches)
    natural_elapsed = np.clip(
        natural_time_s - matches[0].natural_start_s,
        0.0,
        natural_total,
    )
    tts_time = matches[0].tts_start_s + float(natural_elapsed) / natural_total * tts_total
    current = next(
        (match for match in matches if match.tts_start_s <= tts_time < match.tts_end_s),
        matches[-1],
    )
    return float(
        np.clip(
            (tts_time - current.tts_start_s)
            / max(current.tts_end_s - current.tts_start_s, 1e-6),
            0.0,
            1.0,
        )
    )


def build_rhythm(
    natural_spans: Sequence[Span], tts_spans: Sequence[Span], sample_count: int, alpha: float = 1.0
) -> np.ndarray:
    frame_count = _frame_count(sample_count)
    times = _frame_times(frame_count)
    matches, _ = match_token_spans(natural_spans, tts_spans, min_confidence=0.8)
    values = np.zeros((frame_count, RHYTHM_DIM), dtype=np.float32)
    global_ratio = math.log(
        max(sum(max(1e-6, m.tts_end_s - m.tts_start_s) for m in matches), 1e-6)
        / max(sum(max(1e-6, m.natural_end_s - m.natural_start_s) for m in matches), 1e-6)
    ) if matches else 0.0
    for frame, time_s in enumerate(times):
        natural_index = _find_token_index(natural_spans, float(time_s))
        if natural_index is None:
            continue
        natural_span = natural_spans[natural_index]
        match = next(
            (candidate for candidate in matches
             if abs(candidate.natural_start_s - natural_span.start_s) < 1e-7
             and abs(candidate.natural_end_s - natural_span.end_s) < 1e-7),
            None,
        )
        if match is None:
            continue
        natural_duration = max(match.natural_end_s - match.natural_start_s, 1e-6)
        tts_duration = max(match.tts_end_s - match.tts_start_s, 1e-6)
        natural_progress = np.clip((time_s - match.natural_start_s) / natural_duration, 0.0, 1.0)
        tts_progress = _tts_progress_from_matches(float(time_s), matches)
        boundary = float(
            abs(time_s - match.natural_start_s) < FRAME_HOP / TARGET_SR
            or abs(time_s - match.natural_end_s) < FRAME_HOP / TARGET_SR
        )
        values[frame] = [
            float(alpha * math.log(tts_duration / natural_duration)),
            float(natural_progress),
            tts_progress,
            boundary,
            1.0,
            float(alpha * global_ratio),
        ]
    return values


def _source_speaker_map(path: str | Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for record in records_from_manifest(path):
        if "sample_id" in record and "speaker_id" in record:
            result[int(record["sample_id"])] = str(record["speaker_id"])
    return result


def make_splits(sample_to_speaker: Mapping[int, str], seed: int = 42) -> dict[int, str]:
    speakers = sorted(set(sample_to_speaker.values()))
    rng = random.Random(seed)
    rng.shuffle(speakers)
    if len(speakers) < 3:
        raise ValueError("at least three speakers are required for train/valid/test")
    n_valid = max(1, round(len(speakers) * 0.125))
    n_test = max(1, round(len(speakers) * 0.125))
    assignments: dict[str, str] = {}
    for speaker in speakers[: len(speakers) - n_valid - n_test]:
        assignments[speaker] = "train"
    for speaker in speakers[len(speakers) - n_valid - n_test : len(speakers) - n_test]:
        assignments[speaker] = "valid"
    for speaker in speakers[len(speakers) - n_test :]:
        assignments[speaker] = "test"
    return {sample_id: assignments[speaker] for sample_id, speaker in sample_to_speaker.items()}


def _resolve_embedding_dir(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def prepare_dataset(
    alignment_manifest: str | Path,
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    embedding_dir: str | Path = "results/aishell100_phoneme/embeddings",
    cache_dir: str | Path | None = None,
    seed: int = 42,
    min_coverage: float = 0.9,
    max_items: int | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd()).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else output.parent / "rhythm_style_cache"
    if not cache.is_absolute():
        cache = root / cache
    cache.mkdir(parents=True, exist_ok=True)
    embedding_root = _resolve_embedding_dir(embedding_dir, root)
    source_speakers = _source_speaker_map(source_manifest)
    pairs = pair_records(records_from_manifest(alignment_manifest))
    split_map = make_splits(source_speakers, seed)
    items: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for key, group in sorted(pairs.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else pair[0]):
        natural_record = group.get("natural")
        tts_record = group.get("faster_qwen3") or group.get("tts")
        if natural_record is None or tts_record is None:
            rejected.append({"paired_key": key, "reason": "missing_natural_or_tts"})
            continue
        sample_id = int(natural_record["sample_id"])
        if sample_id not in source_speakers:
            rejected.append({"paired_key": key, "reason": "missing_source_speaker"})
            continue
        natural_path = resolve_path(natural_record["audio_path"], root)
        tts_path = resolve_path(tts_record["audio_path"], root)
        natural = load_audio(natural_path)
        tts = load_audio(tts_path)
        natural_spans = spans(natural_record)
        tts_spans = spans(tts_record)
        warped = phone_local_warp(natural, tts, TARGET_SR, natural_spans, tts_spans, min_confidence=0.8)
        coverage = len(warped.matched_spans) / max(1, len(natural_spans))
        if coverage < min_coverage:
            rejected.append({"paired_key": key, "reason": "low_coverage", "coverage": coverage})
            continue
        if natural_record.get("alignment_source") == "uniform_fallback" or tts_record.get("alignment_source") == "uniform_fallback":
            rejected.append({"paired_key": key, "reason": "uniform_fallback_alignment"})
            continue
        frame_count = _frame_count(len(natural))
        content = align_content(embedding_for_sample(embedding_root, sample_id, "natural"), frame_count)
        style = extract_style_features(tts)
        rhythm = build_rhythm(natural_spans, tts_spans, len(natural), alpha=1.0)
        target_path = cache / f"{sample_id}_faster_qwen3_phone_local.wav"
        sf.write(target_path, warped.audio, TARGET_SR, subtype="PCM_16")
        metadata_path = cache / f"{sample_id}_faster_qwen3_phone_local.json"
        metadata = {
            "schema_version": 1,
            "target_type": "tts_phone_local_warp_weak_supervision",
            "paired_key": key,
            "sample_id": sample_id,
            "natural_path": str(natural_path),
            "tts_path": str(tts_path),
            "natural_sha256": sha256_file(natural_path),
            "tts_sha256": sha256_file(tts_path),
            "output_sha256": sha256_file(target_path),
            "coverage": coverage,
            "matched_spans": [match.__dict__ for match in warped.matched_spans],
            "warp_metadata": warped.metadata,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        items.append({
            "paired_key": key,
            "sample_id": sample_id,
            "speaker_id": source_speakers[sample_id],
            "split": split_map[sample_id],
            "natural_audio_path": str(natural_path),
            "tts_audio_path": str(tts_path),
            "weak_target_path": str(target_path),
            "natural": torch.from_numpy(natural),
            "weak_target": torch.from_numpy(np.asarray(warped.audio, dtype=np.float32)),
            "natural_content": torch.from_numpy(content),
            "rhythm": torch.from_numpy(rhythm),
            "style_features": torch.from_numpy(style),
            "frame_mask": torch.ones(frame_count, dtype=torch.bool),
            "sample_count": len(natural),
            "frame_count": frame_count,
            "coverage": float(coverage),
            "metadata_path": str(metadata_path),
        })
        if max_items is not None and len(items) >= max_items:
            break
    if not items:
        raise ValueError("no usable paired items")
    splits = {name: [item for item in items if item["split"] == name] for name in ("train", "valid", "test")}
    for name, records in splits.items():
        torch.save(records, output / f"{name}.pt")
    split_speakers = {name: {item["speaker_id"] for item in records} for name, records in splits.items()}
    if split_speakers["train"] & split_speakers["valid"] or split_speakers["train"] & split_speakers["test"] or split_speakers["valid"] & split_speakers["test"]:
        raise AssertionError("speaker appears in multiple splits")
    manifest = {
        "schema_version": 1,
        "target_type": "tts_phone_local_warp_weak_supervision",
        "alignment_manifest": str(Path(alignment_manifest).resolve()),
        "source_manifest": str(Path(source_manifest).resolve()),
        "embedding_dir": str(embedding_root),
        "sample_rate": TARGET_SR,
        "frame_hop": FRAME_HOP,
        "style_input_dim": STYLE_INPUT_DIM,
        "rhythm_dim": RHYTHM_DIM,
        "content_dim": CONTENT_DIM,
        "seed": seed,
        "min_coverage": min_coverage,
        "counts": {name: len(records) for name, records in splits.items()},
        "split_speakers": {name: sorted(values) for name, values in split_speakers.items()},
        "items": [
            {key: value for key, value in item.items() if not isinstance(value, torch.Tensor)}
            for item in items
        ],
        "rejected": rejected,
        "weak_target_warning": "phone-local warp is supervision only, not a validated disentangled target",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "normalization.json").write_text(json.dumps({"sample_rate": TARGET_SR, "frame_hop": FRAME_HOP}, indent=2), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--embedding-dir", type=Path, default=Path("results/aishell100_phoneme/embeddings"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args(argv)
    result = prepare_dataset(
        args.alignment_manifest, args.source_manifest, args.output_dir,
        repo_root=args.repo_root, embedding_dir=args.embedding_dir,
        cache_dir=args.cache_dir, seed=args.seed,
        min_coverage=args.min_coverage, max_items=args.max_items,
    )
    print(json.dumps({"counts": result["counts"], "rejected": len(result["rejected"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
