#!/usr/bin/env python3
"""Generate fixed, auditable acoustic-control conditions from an alignment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pnp_alignment_warp import global_duration_match
from pnp_feature_controls import (
    duration_alpha_target,
    energy_envelope_transfer,
    f0_transfer,
    loudness_transfer,
    ordinary_phone_local_warp,
    pitch_preserving_phone_local_warp,
    spectral_timbre_transfer,
)


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError("manifest must contain a records list")
    return [dict(record) for record in records]


def _resolve(path: str | Path, repo_root: str | Path | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    if repo_root is not None:
        rooted = Path(repo_root) / candidate
        if rooted.exists():
            return rooted
    raise FileNotFoundError(candidate)


def _spans(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "token": str(token.get("token", token.get("label", ""))),
            "start_s": float(token["start_s"]),
            "end_s": float(token["end_s"]),
            "confidence": float(token.get("confidence", record.get("alignment_confidence", 1.0))),
        }
        for token in record.get("tokens", [])
    ]


def _stats(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    return {
        "sample_count": int(len(values)),
        "duration_s": float(len(values) / sample_rate),
        "peak": float(np.max(np.abs(values))) if len(values) else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0,
        "finite": bool(np.isfinite(values).all()),
        "clipped_sample_count": int(np.count_nonzero(np.abs(values) >= 1.0)),
    }


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    from math import gcd

    divisor = gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(audio, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def _paired_records(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("paired_key", record.get("sample_id", record.get("utterance_id"))))
        condition = str(record.get("condition", ""))
        provider = str(record.get("tts_provider") or record.get("provider") or "")
        group = grouped.setdefault(key, {})
        if condition == "natural":
            group["natural"] = record
        elif condition == "tts":
            group[provider or "tts"] = record
    pairs = []
    for key, group in sorted(grouped.items()):
        natural = group.get("natural")
        if natural is None:
            continue
        for provider, tts in sorted(group.items()):
            if provider != "natural":
                pairs.append((key, natural, tts))
    return pairs


def _conditions(alpha_values: Sequence[float]) -> list[tuple[str, float | None]]:
    result: list[tuple[str, float | None]] = [
        ("natural_identity", None),
        ("tts_raw_oracle", None),
        ("global_duration", None),
        ("ordinary_phone_local", None),
        ("pitch_preserving_phone_local", None),
        ("loudness_transfer", None),
        ("energy_transfer", None),
        ("spectral_timbre_transfer", None),
        ("f0_transfer", None),
    ]
    result.extend((f"duration_alpha_{alpha:g}", float(alpha)) for alpha in alpha_values)
    return result


def _build_condition(
    name: str,
    alpha: float | None,
    natural: np.ndarray,
    tts: np.ndarray,
    sample_rate: int,
    natural_record: Mapping[str, Any],
    tts_record: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    natural_spans = _spans(natural_record)
    tts_spans = _spans(tts_record)
    if name == "natural_identity":
        return natural.copy(), {"transform": "identity"}
    if name == "tts_raw_oracle":
        return tts.copy(), {
            "transform": "identity_raw_tts_oracle",
            "exact_n_required": False,
        }
    if name == "global_duration":
        return global_duration_match(tts, len(natural)), {"transform": "global_duration_match"}
    if name == "ordinary_phone_local":
        result = ordinary_phone_local_warp(
            natural, tts, sample_rate, natural_spans, tts_spans, crossfade_ms=8.0
        )
        return result.audio, result.metadata
    if name == "pitch_preserving_phone_local":
        result = pitch_preserving_phone_local_warp(
            natural, tts, sample_rate, natural_spans, tts_spans
        )
        return result.audio, result.metadata
    if name == "loudness_transfer":
        result = loudness_transfer(natural, tts, sample_rate)
        return result.audio, result.metadata
    if name == "energy_transfer":
        result = energy_envelope_transfer(natural, tts, sample_rate)
        return result.audio, result.metadata
    if name == "spectral_timbre_transfer":
        result = spectral_timbre_transfer(natural, tts, sample_rate)
        return result.audio, result.metadata
    if name == "f0_transfer":
        result = f0_transfer(natural, tts, sample_rate)
        return result.audio, result.metadata
    if name.startswith("duration_alpha_") and alpha is not None:
        result = duration_alpha_target(
            natural, tts, sample_rate, natural_spans, tts_spans, alpha=alpha
        )
        return result.audio, result.metadata
    raise ValueError(f"unknown condition: {name}")


def run_experiment(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    natural_dir: str | Path | None = None,
    tts_dir: str | Path | None = None,
    provider: str | None = None,
    sample_ids: Sequence[int] | None = None,
    alpha_values: Sequence[float] = DEFAULT_ALPHAS,
    sample_rate: int = 16_000,
) -> dict[str, Any]:
    records = _load_records(manifest_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected_ids = {int(value) for value in sample_ids} if sample_ids else None
    rows: list[dict[str, Any]] = []
    for paired_key, natural_record, tts_record in _paired_records(records):
        if provider and str(tts_record.get("tts_provider") or tts_record.get("provider")) != provider:
            continue
        sample_id = natural_record.get("sample_id")
        if sample_id is None:
            sample_id = paired_key
        if selected_ids is not None and int(sample_id) not in selected_ids:
            continue
        natural_path = (
            Path(natural_dir) / f"{int(sample_id)}.wav"
            if natural_dir is not None
            else _resolve(natural_record["audio_path"], repo_root)
        )
        tts_path = (
            Path(tts_dir) / f"{int(sample_id)}.wav"
            if tts_dir is not None
            else _resolve(tts_record["audio_path"], repo_root)
        )
        if not natural_path.exists() or not tts_path.exists():
            raise FileNotFoundError(f"missing overridden audio: {natural_path}, {tts_path}")
        natural, natural_sr = sf.read(natural_path, dtype="float32", always_2d=False)
        tts, tts_sr = sf.read(tts_path, dtype="float32", always_2d=False)
        if natural.ndim != 1 or tts.ndim != 1:
            raise ValueError("control experiment inputs must be mono")
        natural_source_sr = int(natural_sr)
        tts_source_sr = int(tts_sr)
        natural = _resample(natural, natural_source_sr, sample_rate)
        tts = _resample(tts, tts_source_sr, sample_rate)
        for condition, alpha in _conditions(alpha_values):
            audio, metadata = _build_condition(
                condition, alpha, natural, tts, sample_rate, natural_record, tts_record
            )
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if condition != "tts_raw_oracle" and len(audio) != len(natural):
                raise ValueError(f"condition {condition} did not preserve exact-N")
            if not np.isfinite(audio).all():
                raise ValueError(f"condition {condition} produced non-finite audio")
            stem = f"{paired_key}_{tts_record.get('tts_provider', 'tts')}_{condition}"
            audio_path = root / f"{stem}.wav"
            metadata_path = root / f"{stem}.json"
            sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
            row = {
                "paired_key": paired_key,
                "sample_id": sample_id,
                "provider": tts_record.get("tts_provider", tts_record.get("provider", "")),
                "condition": condition,
                "alpha": alpha,
                "natural_path": str(natural_path),
                "tts_path": str(tts_path),
                "natural_sha256": _sha256(natural_path),
                "tts_sha256": _sha256(tts_path),
                "output_sha256": _sha256(audio_path),
                "sample_rate": sample_rate,
                "natural_source_sample_rate": natural_source_sr,
                "tts_source_sample_rate": tts_source_sr,
                "input": _stats(natural, sample_rate),
                "output": _stats(audio, sample_rate),
                "exact_sample_count": bool(len(audio) == len(natural)),
                "metadata": metadata,
                "audio_path": str(audio_path),
                "metadata_path": str(metadata_path),
            }
            metadata_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(row)
    summary = {
        "schema_version": 1,
        "manifest_path": str(Path(manifest_path)),
        "output_dir": str(root),
        "sample_rate": sample_rate,
        "provider": provider,
        "conditions": [name for name, _ in _conditions(alpha_values)],
        "row_count": len(rows),
        "items": rows,
    }
    (root / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--natural-dir", type=Path)
    parser.add_argument("--tts-dir", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--sample-id", type=int, action="append", dest="sample_ids")
    parser.add_argument("--alpha", type=float, action="append", dest="alphas")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args(argv)
    result = run_experiment(
        args.manifest,
        args.output_dir,
        repo_root=args.repo_root,
        natural_dir=args.natural_dir,
        tts_dir=args.tts_dir,
        provider=args.provider,
        sample_ids=args.sample_ids,
        alpha_values=tuple(args.alphas) if args.alphas else DEFAULT_ALPHAS,
        sample_rate=args.sample_rate,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "items"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
