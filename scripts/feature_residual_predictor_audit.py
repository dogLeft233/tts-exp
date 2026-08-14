#!/usr/bin/env python3
"""Probe source-grid HuBERT feature residual learnability without waveform rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.optim import Adam
import torch.nn.functional as F

from direct_waveform_hubert_upper_bound import _encode_l6, _finite_json, _masked_feature_metrics
from hubert_feature_alignment import align_tts_features_to_natural
from prepare_two_stage_hubert_manifest import load_two_stage_manifest
from train_tts_aligned_feature_puller import _load_stage1_hubert, _read_stage1_wave


class FeatureResidualPredictor(torch.nn.Module):
    def __init__(self, dimension: int, *, hidden_dim: int = 128, residual_scale: float = 1.0) -> None:
        super().__init__()
        if dimension <= 0 or hidden_dim <= 0 or residual_scale <= 0:
            raise ValueError("dimension, hidden_dim, and residual_scale must be positive")
        self.residual_scale = float(residual_scale)
        self.residual = torch.nn.Sequential(
            torch.nn.LayerNorm(dimension),
            torch.nn.Linear(dimension, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dimension),
        )
        torch.nn.init.zeros_(self.residual[-1].weight)
        torch.nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,F,D]")
        return features + self.residual_scale * self.residual(features)


def _loss(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    cosine = 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1, eps=1e-8)
    smooth_l1 = F.smooth_l1_loss(predicted, target.detach(), reduction="none").mean(dim=-1)
    return cosine[mask].mean() + smooth_l1[mask].mean()


def _load_item(record: dict[str, Any], encoder: torch.nn.Module, interface: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    natural = _read_stage1_wave(record["natural_path"]).to(device)
    tts = _read_stage1_wave(record["tts_path"], allow_24k_resample=True).to(device)
    with torch.no_grad():
        natural_l6 = _encode_l6(encoder, natural)
        tts_l6 = _encode_l6(encoder, tts)
        target, mask, alignment = align_tts_features_to_natural(
            natural_l6,
            tts_l6,
            record["matched_spans"],
            frame_stride_samples=int(interface.frame_stride_samples),
            sample_rate=int(interface.sample_rate),
            min_frames=1,
        )
    return natural_l6, target, mask.unsqueeze(0)


def run(
    manifest: str | Path,
    *,
    device: str = "cuda",
    max_items: int = 4,
    steps: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
) -> dict[str, Any]:
    if max_items <= 0 or steps <= 0:
        raise ValueError("max_items and steps must be positive")
    device_obj = torch.device(device)
    records = load_two_stage_manifest(manifest)
    train = sorted((row for row in records if row["split"] == "train"), key=lambda row: str(row["paired_key"]))[:max_items]
    if not train:
        raise ValueError("manifest has no train records")
    encoder, interface = _load_stage1_hubert(device=device_obj)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    items = [_load_item(row, encoder, interface, device_obj) for row in train]
    dimension = int(items[0][0].shape[-1])
    model = FeatureResidualPredictor(dimension, hidden_dim=hidden_dim).to(device_obj)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial = [
            _masked_feature_metrics(natural, target, mask)
            for natural, target, mask in items
        ]
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = [_loss(model(natural), target, mask) for natural, target, mask in items]
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach().cpu())})
    with torch.no_grad():
        final = [
            _masked_feature_metrics(model(natural), target, mask)
            for natural, target, mask in items
        ]
    rows = []
    for record, before, after in zip(train, initial, final):
        rows.append({
            "paired_key": str(record["paired_key"]),
            "initial": before,
            "final": after,
            "gap_reduction_fraction": float((before["combined"] - after["combined"]) / max(before["combined"], 1e-12)),
        })
    return _finite_json({
        "schema_version": 1,
        "diagnostic_type": "feature_residual_predictor",
        "device": str(device_obj),
        "item_count": len(rows),
        "steps": steps,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "target": "MFA-linear aligned raw-TTS HuBERT L6 on matched phone spans",
        "rows": rows,
        "history": history,
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-items", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.manifest, device=args.device, max_items=args.max_items, steps=args.steps, learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for row in result["rows"]:
        print(row["paired_key"], row["initial"]["combined"], row["final"]["combined"], row["gap_reduction_fraction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
