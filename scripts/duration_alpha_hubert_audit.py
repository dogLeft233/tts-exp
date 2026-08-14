#!/usr/bin/env python3
"""Audit duration-alpha waveform controls against aligned HuBERT targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from hubert_feature_alignment import align_tts_features_to_natural
from pnp_feature_controls import duration_alpha_target
from prepare_two_stage_hubert_manifest import load_two_stage_manifest
from train_tts_aligned_feature_puller import _load_stage1_hubert, _read_stage1_wave
from direct_waveform_hubert_upper_bound import _encode_l6, _finite_json, _masked_feature_metrics

SAMPLE_RATE = 16_000


def _side_spans(spans: Sequence[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    return [
        {
            "label": str(span["label"]),
            "start_s": float(span[f"{side}_start_s"]),
            "end_s": float(span[f"{side}_end_s"]),
            "confidence": min(
                float(span.get(f"{side}_confidence", span.get("confidence", 1.0))),
                1.0,
            ),
        }
        for span in spans
    ]


def run(
    manifest: str | Path,
    *,
    paired_key: str | None = None,
    device: str = "cuda",
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    device_obj = torch.device(device)
    records = load_two_stage_manifest(manifest)
    train = sorted((row for row in records if row["split"] == "train"), key=lambda row: str(row["paired_key"]))
    record = next((row for row in train if paired_key is None or row["paired_key"] == paired_key), None)
    if record is None:
        raise ValueError(f"paired_key not found: {paired_key}")
    natural = _read_stage1_wave(record["natural_path"]).to(device_obj)
    tts = _read_stage1_wave(record["tts_path"], allow_24k_resample=True).to(device_obj)
    encoder, interface = _load_stage1_hubert(device=device_obj)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        natural_l6 = _encode_l6(encoder, natural)
        tts_l6 = _encode_l6(encoder, tts)
        aligned, mask, alignment = align_tts_features_to_natural(
            natural_l6,
            tts_l6,
            record["matched_spans"],
            frame_stride_samples=int(interface.frame_stride_samples),
            sample_rate=int(interface.sample_rate),
            min_frames=1,
        )
    natural_spans = _side_spans(record["matched_spans"], "natural")
    tts_spans = _side_spans(record["matched_spans"], "tts")
    result: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_type": "duration_alpha_hubert_audit",
        "paired_key": str(record["paired_key"]),
        "device": str(device_obj),
        "target_mask": {
            "matched_frame_count": int(mask.sum().item()),
            "natural_frame_count": int(mask.numel()),
            "coverage": float(mask.float().mean().cpu()),
        },
        "alignment": alignment,
        "alphas": [],
    }
    for alpha in alphas:
        control = duration_alpha_target(
            natural[0, 0].detach().cpu().numpy(),
            tts[0, 0].detach().cpu().numpy(),
            SAMPLE_RATE,
            natural_spans,
            tts_spans,
            alpha=float(alpha),
            min_confidence=0.8,
        )
        waveform = torch.from_numpy(control.audio).to(device_obj)[None, None, :]
        with torch.no_grad():
            features = _encode_l6(encoder, waveform)
            metrics = _masked_feature_metrics(features, aligned, mask.unsqueeze(0))
        result["alphas"].append(
            {
                "alpha": float(alpha),
                "feature_metrics": metrics,
                "control_metadata": control.metadata,
                "exact_sample_count": bool(control.audio.size == natural.shape[-1]),
                "finite": bool(np.isfinite(control.audio).all()),
                "peak": float(np.abs(control.audio).max()),
                "clipped_sample_count": int((np.abs(control.audio) >= 1.0).sum()),
            }
        )
    return _finite_json(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paired-key")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.manifest, paired_key=args.paired_key, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for row in result["alphas"]:
        metrics = row["feature_metrics"]
        print(row["alpha"], metrics["combined"], row["exact_sample_count"], row["peak"], row["clipped_sample_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
