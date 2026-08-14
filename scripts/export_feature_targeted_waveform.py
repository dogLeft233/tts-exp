#!/usr/bin/env python3
"""Export natural audio through a verified frozen Stage-1/Stage-2 renderer pair."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

from feature_targeted_waveform_renderer import (
    FeatureTargetedWaveformRenderer,
    load_frozen_stage1,
)
from prepare_two_stage_hubert_manifest import file_sha256
from train_feature_targeted_waveform_renderer import load_checkpoint as load_stage2_checkpoint
from train_tts_aligned_feature_puller import _load_stage1_hubert, _read_stage1_wave


@dataclass(frozen=True)
class ExportItem:
    paired_key: str
    natural_path: str
    natural_sha256: str

    def __post_init__(self) -> None:
        if not self.paired_key.strip():
            raise ValueError("paired_key must be non-empty")
        if not self.natural_path.strip() or len(self.natural_sha256) != 64:
            raise ValueError("export item requires natural path and SHA-256")


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, Mapping):
        raise ValueError("export list must be an object")
    return payload


def load_export_items(path: str | Path) -> list[ExportItem]:
    """Read an explicit natural-only export list; no paired TTS fields are accepted."""
    source = Path(path).resolve()
    payload = _strict_json(source)
    if set(payload) != {"schema_version", "purpose", "items"}:
        raise ValueError("export list schema is incompatible")
    if payload.get("schema_version") != 1 or payload.get("purpose") != "two_stage_natural_audio_export":
        raise ValueError("export list purpose is incompatible")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("export list requires a non-empty items list")
    result: list[ExportItem] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping) or set(raw) != {"paired_key", "natural_path", "natural_sha256"}:
            raise ValueError("export item must contain only paired_key and natural audio provenance")
        item = ExportItem(
            paired_key=str(raw["paired_key"]),
            natural_path=str(raw["natural_path"]),
            natural_sha256=str(raw["natural_sha256"]),
        )
        if item.paired_key in seen:
            raise ValueError(f"duplicate paired_key: {item.paired_key}")
        seen.add(item.paired_key)
        result.append(item)
    return result


def _audio_stats(values: np.ndarray) -> dict[str, float | int | bool]:
    audio = np.asarray(values, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("audio must be non-empty and finite")
    return {
        "sample_count": int(audio.size),
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "mean_abs": float(np.mean(np.abs(audio))),
        "peak": float(np.max(np.abs(audio))),
        "clipped_sample_count": int(np.sum(np.abs(audio) >= 1.0)),
        "finite": True,
    }


def _load_renderer(
    stage1_checkpoint: Path,
    stage2_checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[FeatureTargetedWaveformRenderer, Mapping[str, Any]]:
    stage1, artifact = load_frozen_stage1(stage1_checkpoint, map_location="cpu")
    raw_stage2 = torch.load(stage2_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw_stage2, Mapping):
        raise ValueError("Stage 2 checkpoint must be an object")
    config = raw_stage2.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("Stage 2 checkpoint model config is missing")
    streams = config.get("condition_streams")
    waveform = config.get("waveform_model")
    if not isinstance(streams, Mapping) or not isinstance(waveform, Mapping):
        raise ValueError("Stage 2 checkpoint renderer config is malformed")
    encoder, interface = _load_stage1_hubert(device=device)
    renderer = FeatureTargetedWaveformRenderer(
        encoder,
        stage1.to(device),
        artifact,
        interface=interface,
        conditioning_dim=int(streams["natural_layer2"]["output_dim"]),
        channels=int(waveform["channels"]),
        dilations=tuple(int(value) for value in waveform["dilations"]),
        residual_scale=float(waveform["residual_scale"]),
    ).to(device)
    checkpoint = load_stage2_checkpoint(stage2_checkpoint, renderer, map_location=device)
    if checkpoint.get("stage1_checkpoint") != artifact.as_dict():
        raise ValueError("Stage 2 checkpoint does not bind the supplied Stage 1 checkpoint")
    renderer.eval()
    return renderer, checkpoint


def export_two_stage_natural_audio(
    export_list_path: str | Path,
    stage1_checkpoint_path: str | Path,
    stage2_checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Export only the natural audio explicitly named in a strict export list."""
    export_list = Path(export_list_path).resolve()
    stage1_checkpoint = Path(stage1_checkpoint_path).resolve()
    stage2_checkpoint = Path(stage2_checkpoint_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"two-stage export output directory must be fresh: {output}")
    if not export_list.is_file() or not stage1_checkpoint.is_file() or not stage2_checkpoint.is_file():
        raise FileNotFoundError("export list or checkpoint is missing")
    items = load_export_items(export_list)
    torch_device = torch.device(device)
    renderer, checkpoint = _load_renderer(stage1_checkpoint, stage2_checkpoint, device=torch_device)
    output.mkdir(parents=True, exist_ok=False)

    exported: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in items:
            natural_path = Path(item.natural_path).resolve()
            if not natural_path.is_file() or file_sha256(natural_path) != item.natural_sha256:
                raise ValueError(f"natural source hash mismatch: {item.paired_key}")
            natural = _read_stage1_wave(natural_path).to(torch_device)
            result = renderer(natural, compute_enhanced_features=False)
            natural_values = natural[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            enhanced = result["waveform"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            residual = enhanced - natural_values
            if enhanced.shape != natural_values.shape or not np.isfinite(enhanced).all():
                raise FloatingPointError(f"invalid enhanced waveform: {item.paired_key}")
            if float(np.max(np.abs(residual))) > renderer.waveform_model.residual_scale + 1e-5:
                raise FloatingPointError(f"residual bound exceeded: {item.paired_key}")
            if int(np.sum(np.abs(enhanced) >= 1.0)):
                raise FloatingPointError(f"enhanced waveform clipped: {item.paired_key}")
            wav_path = output / f"{item.paired_key}.wav"
            sf.write(wav_path, enhanced, 16_000, subtype="FLOAT")
            exported.append({
                "paired_key": item.paired_key,
                "natural_path": str(natural_path),
                "natural_sha256": item.natural_sha256,
                "enhanced_path": str(wav_path),
                "enhanced_sha256": file_sha256(wav_path),
                "sample_rate": 16_000,
                "natural": _audio_stats(natural_values),
                "enhanced": _audio_stats(enhanced),
                "residual": _audio_stats(residual),
            })

    payload = {
        "schema_version": 1,
        "purpose": "verified_two_stage_natural_audio_export",
        "scope": "natural_audio_only_no_tts_or_heldout_input",
        "export_list": {"path": str(export_list), "sha256": file_sha256(export_list)},
        "stage1_checkpoint": {
            "path": str(stage1_checkpoint),
            "sha256": file_sha256(stage1_checkpoint),
        },
        "stage2_checkpoint": {
            "path": str(stage2_checkpoint),
            "sha256": file_sha256(stage2_checkpoint),
            "selection": checkpoint["selection"],
        },
        "encoder_interface": dict(renderer.interface),
        "sample_rate": 16_000,
        "wav_subtype": "FLOAT",
        "count": len(exported),
        "items": exported,
    }
    manifest_path = output / "enhanced_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-list", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = export_two_stage_natural_audio(
        args.export_list,
        args.stage1_checkpoint,
        args.stage2_checkpoint,
        args.output_dir,
        device=args.device,
    )
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "count": result["count"]}, allow_nan=False))
    return 0


__all__ = [
    "ExportItem",
    "export_two_stage_natural_audio",
    "load_export_items",
]


if __name__ == "__main__":
    raise SystemExit(main())
