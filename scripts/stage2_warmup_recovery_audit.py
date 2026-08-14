#!/usr/bin/env python3
"""Audit realization warmup followed by preservation-loss recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from feature_targeted_waveform_renderer import FeatureTargetedWaveformRenderer, load_frozen_stage1
from train_feature_targeted_waveform_renderer import (
    Stage2LossWeights,
    _load_stage1_hubert,
    compute_stage2_loss,
    prepare_records,
)
from train_tts_aligned_feature_puller import _read_stage1_wave, load_stage1_records


TERM_NAMES = (
    "realize_cosine",
    "realize_smooth_l1",
    "content_cosine",
    "content_smooth_l1",
    "energy",
    "residual_l1",
    "residual_smoothness",
    "anti_clipping",
)


def _weights(realization_only: bool, base: Stage2LossWeights) -> dict[str, float]:
    values = {name: 0.0 for name in TERM_NAMES}
    if realization_only:
        values["realize_cosine"] = float(base.realize_cosine)
        values["realize_smooth_l1"] = float(base.realize_smooth_l1)
    else:
        values.update({name: float(getattr(base, name)) for name in TERM_NAMES})
    return values


def _forward(
    renderer: FeatureTargetedWaveformRenderer,
    record: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    natural = _read_stage1_wave(record["natural_path"]).to(device)
    output = renderer(
        natural,
        natural_layer2=record["natural_layer2"],
        natural_layer6=record["natural_layer6"],
        delta_layer6=record["stage1_delta_layer6"],
        padding_mask=record["padding_mask"],
    )
    loss, terms, diagnostics = compute_stage2_loss(
        output,
        natural,
        active_weights=_weights(False, Stage2LossWeights()),
        smooth_l1_beta=Stage2LossWeights().smooth_l1_beta,
        realize_target=record["realize_target_layer6"],
        realize_mask=record["realize_target_mask"],
        target_mode="aligned_raw_tts_oracle",
    )
    diagnostics = dict(diagnostics)
    diagnostics["residual_peak"] = float(output["residual"].detach().abs().max().cpu())
    diagnostics["clipped_sample_count"] = int((output["waveform"].detach().abs() >= 1.0).sum().cpu())
    return loss, terms, diagnostics


def _evaluate(
    renderer: FeatureTargetedWaveformRenderer,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    renderer.eval()
    totals = {name: 0.0 for name in (*TERM_NAMES, "total")}
    peaks: list[float] = []
    clipped = 0
    with torch.no_grad():
        for record in records:
            _, terms, diagnostics = _forward(renderer, record, device)
            for name in totals:
                if name in terms:
                    totals[name] += float(terms[name].cpu())
            peaks.append(float(diagnostics["residual_peak"]))
            clipped += int(diagnostics["clipped_sample_count"])
    count = max(1, len(records))
    result = {name: value / count for name, value in totals.items()}
    result["realize_combined"] = result["realize_cosine"] + result["realize_smooth_l1"]
    result["max_residual_peak"] = max(peaks, default=0.0)
    result["clipped_sample_count"] = float(clipped)
    result["eligible"] = float(clipped == 0)
    return result


def _train_phase(
    renderer: FeatureTargetedWaveformRenderer,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    steps: int,
    active_weights: Mapping[str, float],
    smooth_l1_beta: float,
) -> None:
    renderer.train()
    for _ in range(steps):
        for record in records:
            natural = _read_stage1_wave(record["natural_path"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = renderer(
                natural,
                natural_layer2=record["natural_layer2"],
                natural_layer6=record["natural_layer6"],
                delta_layer6=record["stage1_delta_layer6"],
                padding_mask=record["padding_mask"],
            )
            loss, _, _ = compute_stage2_loss(
                output,
                natural,
                active_weights=active_weights,
                smooth_l1_beta=smooth_l1_beta,
                realize_target=record["realize_target_layer6"],
                realize_mask=record["realize_target_mask"],
                target_mode="aligned_raw_tts_oracle",
            )
            loss.backward()
            optimizer.step()


def _run_one(
    manifest: Path,
    stage1_checkpoint: Path,
    *,
    device: torch.device,
    warmup_steps: int,
    recovery_steps: int,
    learning_rate: float,
    residual_scale: float,
) -> dict[str, Any]:
    groups, _ = load_stage1_records(manifest)
    encoder, interface = _load_stage1_hubert(device=device)
    stage1, artifact = load_frozen_stage1(stage1_checkpoint, map_location="cpu")
    stage1 = stage1.to(device)
    renderer = FeatureTargetedWaveformRenderer(
        encoder,
        stage1,
        artifact,
        interface=interface.as_dict(),
        residual_scale=residual_scale,
    ).to(device)
    prepared = {
        split: prepare_records(
            rows,
            renderer=renderer,
            device=device,
            read_wave=_read_stage1_wave,
            target_mode="aligned_raw_tts_oracle",
        )
        for split, rows in groups.items()
    }
    base = Stage2LossWeights()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in renderer.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    _train_phase(
        renderer,
        prepared["train"],
        device=device,
        optimizer=optimizer,
        steps=warmup_steps,
        active_weights=_weights(True, base),
        smooth_l1_beta=base.smooth_l1_beta,
    )
    warmup_train = _evaluate(renderer, prepared["train"], device)
    warmup_valid = _evaluate(renderer, prepared["valid"], device)
    _train_phase(
        renderer,
        prepared["train"],
        device=device,
        optimizer=optimizer,
        steps=recovery_steps,
        active_weights=_weights(False, base),
        smooth_l1_beta=base.smooth_l1_beta,
    )
    recovery_train = _evaluate(renderer, prepared["train"], device)
    recovery_valid = _evaluate(renderer, prepared["valid"], device)
    return {
        "manifest": str(manifest),
        "stage1_checkpoint": str(stage1_checkpoint),
        "train_count": len(prepared["train"]),
        "valid_count": len(prepared["valid"]),
        "warmup_steps": warmup_steps,
        "recovery_steps": recovery_steps,
        "learning_rate": learning_rate,
        "residual_scale": residual_scale,
        "warmup": {"train": warmup_train, "valid": warmup_valid},
        "recovery": {"train": recovery_train, "valid": recovery_valid},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-k1", type=Path, required=True)
    parser.add_argument("--stage1-k1", type=Path, required=True)
    parser.add_argument("--manifest-k4", type=Path, required=True)
    parser.add_argument("--stage1-k4", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--recovery-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup_steps <= 0 or args.recovery_steps <= 0:
        raise ValueError("warmup and recovery steps must be positive")
    torch.manual_seed(29)
    device = torch.device(args.device)
    result = {
        "schema_version": 1,
        "diagnostic_type": "stage2_realization_warmup_recovery",
        "device": str(device),
        "k1": _run_one(
            args.manifest_k1,
            args.stage1_k1,
            device=device,
            warmup_steps=args.warmup_steps,
            recovery_steps=args.recovery_steps,
            learning_rate=args.learning_rate,
            residual_scale=args.residual_scale,
        ),
        "k4": _run_one(
            args.manifest_k4,
            args.stage1_k4,
            device=device,
            warmup_steps=args.warmup_steps,
            recovery_steps=args.recovery_steps,
            learning_rate=args.learning_rate,
            residual_scale=args.residual_scale,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for name in ("k1", "k4"):
        value = result[name]
        print(name, "warmup", value["warmup"]["train"]["realize_combined"], value["warmup"]["valid"]["realize_combined"])
        print(name, "recovery", value["recovery"]["train"]["realize_combined"], value["recovery"]["valid"]["realize_combined"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
