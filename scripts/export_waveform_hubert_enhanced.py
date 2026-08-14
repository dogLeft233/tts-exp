#!/usr/bin/env python3
"""Export exact-length enhanced WAVs from a trained HuBERT waveform enhancer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch

from train_waveform_hubert_enhancer import (
    _load_hubert,
    _load_target_manifest,
    _read_wave,
    file_sha256,
    load_checkpoint,
)
from waveform_hubert_enhancer import HuBERTWaveformEnhancer


def _audio_stats(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("audio must be non-empty and finite")
    return {
        "sample_count": int(flat.size),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "peak": float(np.max(np.abs(flat))),
        "mean": float(np.mean(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "finite": True,
        "out_of_normalized_range_count": int(np.sum(np.abs(flat) > 1.0)),
    }


def export_enhanced(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("manifest or checkpoint is missing")

    items = _load_target_manifest(manifest)
    torch_device = torch.device(device)
    encoder, interface = _load_hubert(device=torch_device)
    model = HuBERTWaveformEnhancer(encoder, interface=interface).to(torch_device)
    checkpoint_payload = load_checkpoint(checkpoint, model, map_location=torch_device)
    model.eval()

    exported: list[dict[str, Any]] = []
    seen: set[str] = set()
    with torch.no_grad():
        for item in items:
            key = str(item["paired_key"])
            if key in seen:
                raise ValueError(f"duplicate paired_key: {key}")
            seen.add(key)
            natural_path = Path(str(item["natural_path"])).resolve()
            if file_sha256(natural_path) != str(item["natural_sha256"]):
                raise ValueError(f"natural audio hash mismatch: {key}")
            natural = _read_wave(natural_path).to(torch_device)
            result = model(natural, compute_enhanced_features=False)
            enhanced = result["waveform"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            natural_values = natural[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            if enhanced.shape != natural_values.shape or not np.isfinite(enhanced).all():
                raise FloatingPointError(f"invalid enhanced waveform for {key}")
            residual = enhanced - natural_values
            if float(np.max(np.abs(residual))) > model.waveform_model.residual_scale + 1e-5:
                raise FloatingPointError(f"residual bound exceeded for {key}")
            wav_path = output / f"{key}.wav"
            sf.write(wav_path, enhanced, 16_000, subtype="FLOAT")
            exported.append({
                "paired_key": key,
                "split": str(item["split"]),
                "speaker_group": str(item["speaker_group"]),
                "natural_path": str(natural_path),
                "natural_sha256": str(item["natural_sha256"]),
                "enhanced_path": str(wav_path),
                "enhanced_sha256": file_sha256(wav_path),
                "sample_rate": 16_000,
                "natural_sample_count": int(natural_values.size),
                "enhanced_sample_count": int(enhanced.size),
                "natural_stats": _audio_stats(natural_values),
                "enhanced_stats": _audio_stats(enhanced),
                "residual_stats": _audio_stats(residual),
                "residual_scale": float(model.waveform_model.residual_scale),
            })

    payload = {
        "schema_version": 1,
        "purpose": "trained_waveform_enhancer_export",
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "encoder_interface": model.interface.as_dict(),
        "checkpoint_manifest_sha256": checkpoint_payload["manifest_sha256"],
        "sample_rate": 16_000,
        "wav_subtype": "FLOAT",
        "count": len(exported),
        "items": exported,
    }
    (output / "enhanced_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = export_enhanced(
        args.manifest,
        args.checkpoint,
        args.output_dir,
        device=args.device,
    )
    print(json.dumps({
        "output_dir": str(Path(args.output_dir).resolve()),
        "count": result["count"],
        "splits": {
            split: sum(1 for item in result["items"] if item["split"] == split)
            for split in ("train", "valid", "heldout")
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
