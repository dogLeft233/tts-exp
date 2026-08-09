#!/usr/bin/env python3
"""Evaluate cached waveform enhancer outputs without a TFG frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

from pnp_audio_enhancer import ResidualTCN, WaveformEnhancer
from train_pnp_audio_enhancer import log_mel_loss, multi_resolution_stft_loss


def _stats(audio: np.ndarray, sample_rate: int) -> dict[str, float | int | bool]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    return {
        "sample_count": int(values.size),
        "duration_s": float(values.size / sample_rate),
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "finite": bool(np.isfinite(values).all()),
        "clipped_sample_count": int(np.count_nonzero(np.abs(values) >= 1.0)),
    }


def _load_items(manifest_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("manifest must contain an items list")
    return [dict(item) for item in items]


def _load_model(checkpoint: str | Path | None, channels: int = 64) -> ResidualTCN:
    model = ResidualTCN(channels=channels)
    if checkpoint is None:
        return model
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model


def evaluate_items(
    items: Iterable[Mapping[str, Any]],
    *,
    checkpoint: str | Path | None = None,
    sample_rate: int = 16_000,
    chunk_seconds: float = 4.0,
    stride_seconds: float = 2.0,
) -> dict[str, Any]:
    """Report paired audio metrics and shortcut guards for cached samples."""
    wrapper = WaveformEnhancer(
        _load_model(checkpoint),
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        stride_seconds=stride_seconds,
    )
    rows: list[dict[str, Any]] = []
    for item in items:
        natural, natural_sr = sf.read(item["natural_path"], dtype="float32", always_2d=False)
        target, target_sr = sf.read(item["audio_path"], dtype="float32", always_2d=False)
        if natural_sr != sample_rate or target_sr != sample_rate:
            raise ValueError("input and target sample rates must equal sample_rate")
        if natural.ndim != 1 or target.ndim != 1 or len(natural) != len(target):
            raise ValueError("input and target must be mono and exact-length")
        enhanced = wrapper.enhance(natural)
        input_tensor = torch.from_numpy(natural[None, None, :])
        target_tensor = torch.from_numpy(target[None, None, :])
        enhanced_tensor = torch.from_numpy(enhanced[None, None, :])
        with torch.no_grad():
            target_loss = float(multi_resolution_stft_loss(target_tensor, target_tensor))
            identity_stft = float(multi_resolution_stft_loss(input_tensor, target_tensor))
            enhanced_stft = float(multi_resolution_stft_loss(enhanced_tensor, target_tensor))
            identity_mel = float(log_mel_loss(input_tensor, target_tensor, sample_rate=sample_rate))
            enhanced_mel = float(log_mel_loss(enhanced_tensor, target_tensor, sample_rate=sample_rate))
        rows.append(
            {
                "paired_key": item.get("paired_key"),
                "provider": item.get("provider"),
                "input": _stats(natural, sample_rate),
                "target": _stats(target, sample_rate),
                "enhanced": _stats(enhanced, sample_rate),
                "exact_length": bool(len(enhanced) == len(natural)),
                "residual_rms": float(np.sqrt(np.mean(np.square(enhanced - natural)))),
                "identity_stft_loss": identity_stft,
                "enhanced_stft_loss": enhanced_stft,
                "identity_log_mel_loss": identity_mel,
                "enhanced_log_mel_loss": enhanced_mel,
                "target_self_stft_loss": target_loss,
            }
        )
    valid_rows = [row for row in rows if row["exact_length"] and row["enhanced"]["finite"]]
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "sample_rate": sample_rate,
        "item_count": len(rows),
        "valid_count": len(valid_rows),
        "rows": rows,
        "mean_identity_stft_loss": float(np.mean([row["identity_stft_loss"] for row in rows])) if rows else None,
        "mean_enhanced_stft_loss": float(np.mean([row["enhanced_stft_loss"] for row in rows])) if rows else None,
        "mean_identity_log_mel_loss": float(np.mean([row["identity_log_mel_loss"] for row in rows])) if rows else None,
        "mean_enhanced_log_mel_loss": float(np.mean([row["enhanced_log_mel_loss"] for row in rows])) if rows else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_items(_load_items(args.manifest), checkpoint=args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
