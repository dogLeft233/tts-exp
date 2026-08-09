#!/usr/bin/env python3
"""Summarize duration and acoustic metrics from a control-experiment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def _spectral_tilt(audio: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    mask = (frequencies > 80.0) & (frequencies < sample_rate / 2 - 1) & (spectrum > 1e-12)
    if int(np.count_nonzero(mask)) < 4:
        return 0.0
    return float(np.polyfit(np.log10(frequencies[mask]), np.log10(spectrum[mask]), 1)[0])


def _median_f0(audio: np.ndarray, sample_rate: int) -> float | None:
    if len(audio) < 1024:
        return None
    try:
        f0, voiced, _ = librosa.pyin(
            audio, fmin=50.0, fmax=600.0, sr=sample_rate,
            frame_length=1024, hop_length=256,
        )
    except (ValueError, RuntimeError, librosa.util.exceptions.ParameterError):
        return None
    values = f0[voiced & np.isfinite(f0)]
    return float(np.median(values)) if len(values) else None


def _features(path: str | Path, sample_rate: int) -> dict[str, Any]:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if source_rate != sample_rate or audio.ndim != 1:
        raise ValueError(f"expected mono {sample_rate} Hz WAV: {path}")
    meter = pyln.Meter(sample_rate)
    try:
        lufs = float(meter.integrated_loudness(audio))
    except (ValueError, RuntimeError):
        lufs = float("nan")
    envelope = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    return {
        "sample_count": int(len(audio)),
        "duration_s": float(len(audio) / sample_rate),
        "lufs": lufs,
        "rms": float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0,
        "spectral_tilt": _spectral_tilt(audio, sample_rate),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))),
        "energy_env_std": float(np.std(envelope)),
        "f0_median": _median_f0(audio, sample_rate),
        "finite": bool(np.isfinite(audio).all()),
        "clipped_sample_count": int(np.count_nonzero(np.abs(audio) >= 1.0)),
    }


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def summarize(manifest_path: str | Path, *, sample_rate: int = 16_000) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("manifest must contain an items list")
    rows: list[dict[str, Any]] = []
    natural: dict[int, dict[str, Any]] = {}
    for item in items:
        sample_id = int(item["sample_id"])
        features = _features(item["audio_path"], sample_rate)
        row = {**item, "features": features}
        rows.append(row)
        if item["condition"] == "natural_identity":
            natural[sample_id] = features
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    conditions: dict[str, Any] = {}
    metric_names = (
        "duration_s", "lufs", "rms", "spectral_tilt", "spectral_centroid",
        "energy_env_std", "f0_median",
    )
    for condition, condition_rows in by_condition.items():
        values = [row["features"] for row in condition_rows]
        summary: dict[str, Any] = {
            "sample_count": len(values),
            "exact_sample_count": sum(bool(row["exact_sample_count"]) for row in condition_rows),
            "finite_count": sum(bool(value["finite"]) for value in values),
            "clipped_sample_count": sum(int(value["clipped_sample_count"]) for value in values),
            "means": {name: _mean([float(value[name]) for value in values if value[name] is not None]) for name in metric_names},
        }
        deltas: dict[str, list[float]] = {name: [] for name in metric_names}
        for row in condition_rows:
            baseline = natural.get(int(row["sample_id"]))
            if baseline is None:
                continue
            for name in metric_names:
                current = row["features"][name]
                reference = baseline[name]
                if current is not None and reference is not None and np.isfinite(current) and np.isfinite(reference):
                    deltas[name].append(float(current - reference))
        summary["mean_delta_vs_natural"] = {name: _mean(values) for name, values in deltas.items()}
        conditions[condition] = summary
    return {
        "schema_version": 1,
        "manifest_path": str(Path(manifest_path)),
        "sample_rate": sample_rate,
        "item_count": len(rows),
        "conditions": conditions,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args(argv)
    result = summarize(args.manifest, sample_rate=args.sample_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"conditions", "rows"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
