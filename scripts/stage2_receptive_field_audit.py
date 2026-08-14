#!/usr/bin/env python3
"""Compare Stage-2 renderer receptive fields on the K=1 target contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from feature_targeted_waveform_renderer import FeatureTargetedWaveformRenderer, load_frozen_stage1
from prepare_two_stage_hubert_manifest import load_two_stage_manifest
from train_tts_aligned_feature_puller import _load_stage1_hubert, _read_stage1_wave
from waveform_hubert_enhancer import EncoderInterface


def _selected(manifest: str | Path, paired_key: str | None, device: torch.device) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.nn.Module, Any, Any]:
    records = load_two_stage_manifest(manifest)
    train = sorted((row for row in records if row["split"] == "train"), key=lambda row: str(row["paired_key"]))
    record = next((row for row in train if paired_key is None or row["paired_key"] == paired_key), None)
    if record is None:
        raise ValueError(f"paired_key not found: {paired_key}")
    waveform = _read_stage1_wave(record["natural_path"]).to(device)
    encoder, interface = _load_stage1_hubert(device=device)
    return record, waveform, encoder, interface, load_frozen_stage1(
        "runs/two_stage_hubert_aishell1_20260810/stage1_direct_raw_tts_20260810_v6_scale_corrected_seed29/best.pt",
        map_location=device,
    )


def run(manifest: str | Path, *, paired_key: str | None = None, device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    record, waveform, encoder, interface, (stage1, artifact) = _selected(manifest, paired_key, device_obj)
    stage1 = stage1.to(device_obj)
    stage1.eval()
    layer2, layer6, delta = FeatureTargetedWaveformRenderer(
        encoder,
        stage1,
        artifact,
        interface=interface.as_dict(),
        conditioning_dim=32,
        channels=64,
        dilations=(1, 2, 4, 8, 16, 32),
        residual_scale=0.05,
    ).natural_features(waveform)
    outputs: list[dict[str, Any]] = []
    for dilations in ((1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32, 64, 128), (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        renderer = FeatureTargetedWaveformRenderer(
            encoder,
            stage1,
            artifact,
            interface=interface.as_dict(),
            conditioning_dim=32,
            channels=64,
            dilations=dilations,
            residual_scale=0.05,
        ).to(device_obj)
        renderer.eval()
        with torch.no_grad():
            output = renderer(
                waveform,
                natural_layer2=layer2,
                natural_layer6=layer6,
                delta_layer6=delta,
                padding_mask=torch.ones(layer6.shape[:2], dtype=torch.bool, device=device_obj),
                compute_enhanced_features=False,
            )
        outputs.append({
            "dilations": list(dilations),
            "nominal_receptive_field_samples": int(1 + 4 * sum(dilations)),
            "nominal_receptive_field_ms": float((1 + 4 * sum(dilations)) / 16.0),
            "parameter_count": renderer.parameter_count,
            "input_samples": int(waveform.shape[-1]),
            "output_samples": int(output["waveform"].shape[-1]),
            "exact_sample_count": bool(output["waveform"].shape == waveform.shape),
            "finite": bool(torch.isfinite(output["waveform"]).all()),
            "residual_peak": float(output["residual"].abs().max().cpu()),
        })
    return {
        "schema_version": 1,
        "diagnostic_type": "stage2_receptive_field_audit",
        "paired_key": str(record["paired_key"]),
        "device": str(device_obj),
        "outputs": outputs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paired-key")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.manifest, paired_key=args.paired_key, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in result["outputs"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
