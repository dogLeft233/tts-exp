#!/usr/bin/env python3
"""Build a small valid-only rhythm counterfactual audio smoke set.

The outputs are diagnostic controls, not training targets. Each output keeps
natural waveform length, records transformation metadata, and excludes S0770.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError as exc:
    raise ImportError("librosa is required for 24 kHz to 16 kHz resampling") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pnp_alignment_warp import Span
from pnp_feature_controls import duration_alpha_target, pitch_preserving_resample

REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = REPO_ROOT / "runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs/rhythm_timing/20260812_counterfactual_smoke"
TARGET_SR = 16_000
VALID_SPEAKER = "S0765"
SMOKE_COUNT = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw_path: str | Path) -> Path:
    raw = str(raw_path)
    candidates = [Path(raw)]
    prefix = "/mnt/e/Documents/tts-audio/tts-exp/"
    if raw.startswith(prefix):
        candidates.append(REPO_ROOT / raw[len(prefix):])
    elif not Path(raw).is_absolute():
        candidates.append(REPO_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(raw)


def load_audio(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if sample_rate != TARGET_SR:
        values = librosa.resample(values, orig_sr=sample_rate, target_sr=TARGET_SR)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"empty or non-finite audio: {path}")
    return values


def spans(record: Mapping[str, Any]) -> list[Span]:
    result: list[Span] = []
    for token in record.get("tokens", []):
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


def exact_length(values: np.ndarray, target_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == target_count:
        return values.copy()
    if len(values) > target_count:
        return values[:target_count].copy()
    return np.pad(values, (0, target_count - len(values))).astype(np.float32)


def audio_meta(values: np.ndarray, natural_count: int) -> dict[str, Any]:
    return {
        "sample_count": int(len(values)),
        "natural_sample_count": int(natural_count),
        "exact_sample_count": bool(len(values) == natural_count),
        "finite": bool(np.isfinite(values).all()),
        "peak": float(np.max(np.abs(values))) if len(values) else 0.0,
        "clipped_samples": int(np.sum(np.abs(values) >= 1.0)),
        "rms": float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0,
    }


def write_wav(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, values.astype(np.float32), TARGET_SR, subtype="PCM_16")


def make_tts_global_natural_length(tts: np.ndarray, natural_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    transformed = pitch_preserving_resample(tts, natural_count)
    output = exact_length(transformed.audio, natural_count)
    return output, {**transformed.metadata, "condition": "tts_global_natural_length"}


def build_record_pairs(manifest: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in payload.get("records", []):
        if raw.get("split") != "valid" or raw.get("speaker_id") != VALID_SPEAKER:
            continue
        condition = str(raw.get("condition", ""))
        if condition not in {"natural", "tts"}:
            continue
        groups.setdefault(str(raw["paired_key"]), {})[condition] = dict(raw)
    pairs = []
    for key in sorted(groups):
        group = groups[key]
        if "natural" in group and "tts" in group:
            pairs.append((group["natural"], group["tts"]))
    if len(pairs) != 50:
        raise ValueError(f"expected 50 valid pairs, found {len(pairs)}")
    return pairs


def run(manifest: Path, output_dir: Path, count: int) -> dict[str, Any]:
    pairs = build_record_pairs(manifest)[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (natural_record, tts_record) in enumerate(pairs, 1):
        key = str(natural_record["paired_key"])
        natural_path = resolve_path(natural_record["audio_path"])
        tts_path = resolve_path(tts_record["audio_path"])
        natural = load_audio(natural_path)
        tts = load_audio(tts_path)
        natural_spans = spans(natural_record)
        tts_spans = spans(tts_record)
        conditions: list[tuple[str, np.ndarray, dict[str, Any]]] = [
            (
                "natural_noop",
                natural.copy(),
                {"transform": "identity", "condition": "natural_noop"},
            ),
            (
                "tts_noop",
                exact_length(tts, len(tts)),
                {"transform": "identity", "condition": "tts_noop"},
            ),
        ]
        global_audio, global_meta = make_tts_global_natural_length(tts, len(natural))
        conditions.append(("tts_global_natural_length", global_audio, global_meta))
        for alpha in (0.0, 0.5, 1.0):
            control = duration_alpha_target(
                natural,
                tts,
                TARGET_SR,
                natural_spans,
                tts_spans,
                alpha=alpha,
                min_confidence=0.8,
            )
            conditions.append(
                (
                    f"duration_alpha_{alpha:g}",
                    control.audio,
                    {**control.metadata, "condition": f"duration_alpha_{alpha:g}"},
                )
            )
        for condition, values, transform_meta in conditions:
            if condition == "tts_noop":
                target_count = len(tts)
            else:
                target_count = len(natural)
            metadata = {
                "schema_version": 1,
                "paired_key": key,
                "sample_id": int(natural_record["sample_id"]),
                "speaker_id": VALID_SPEAKER,
                "split": "valid",
                "condition": condition,
                "natural_source": str(natural_path),
                "tts_source": str(tts_path),
                "natural_source_sha256": sha256_file(natural_path),
                "tts_source_sha256": sha256_file(tts_path),
        "output_sample_rate": TARGET_SR,
        "input_resampling": {"natural": "none_or_16khz", "tts": "24khz_to_16khz_if_needed"},
        "output_sha256": None,
                "transform_metadata": transform_meta,
                "audio": audio_meta(values, target_count),
            }
            output_path = output_dir / "audio" / f"{int(natural_record['sample_id']):04d}_{condition}.wav"
            write_wav(output_path, values)
            metadata["output_sha256"] = sha256_file(output_path)
            metadata["output_path"] = str(output_path)
            metadata_path = output_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            records.append(metadata)
            print(f"OK {index}/{len(pairs)} {condition} samples={len(values)} peak={metadata['audio']['peak']:.4f}", flush=True)
    result = {
        "schema_version": 1,
        "evaluation": "valid_only_rhythm_counterfactual_smoke",
        "valid_speaker": VALID_SPEAKER,
        "count": len(pairs),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "heldout_excluded": True,
        "sample_rate": TARGET_SR,
        "records": records,
        "conditions": sorted({record["condition"] for record in records}),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=SMOKE_COUNT)
    args = parser.parse_args()
    if not 1 <= args.count <= 50:
        raise SystemExit("--count must be in [1, 50]")
    result = run(args.manifest, args.output_dir, args.count)
    print(json.dumps({"count": result["count"], "conditions": result["conditions"]}, ensure_ascii=False, indent=2))
    print(f"output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
