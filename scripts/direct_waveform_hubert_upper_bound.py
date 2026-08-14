#!/usr/bin/env python3
"""Probe whether an exact-length waveform can realize an aligned HuBERT target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam

from hubert_feature_alignment import align_tts_features_to_natural
from pnp_alignment_warp import phone_local_warp
from prepare_two_stage_hubert_manifest import load_two_stage_manifest
from train_tts_aligned_feature_puller import _load_stage1_hubert, _read_stage1_wave
from waveform_hubert_enhancer import _extract_hidden_states

TARGET_LAYER = 6
TARGET_SAMPLE_RATE = 16_000


def _finite_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise FloatingPointError("diagnostic contains a non-finite float")
    return value


def _masked_feature_metrics(predicted: Tensor, target: Tensor, mask: Tensor) -> dict[str, float | int]:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target features must share [B,F,D] shape")
    if mask.shape != predicted.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("feature mask must be Boolean [B,F]")
    count = int(mask.sum().item())
    if count <= 0:
        raise ValueError("feature mask must select at least one frame")
    cosine = 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1, eps=1e-8)
    smooth_l1 = F.smooth_l1_loss(
        predicted,
        target.detach(),
        reduction="none",
    ).mean(dim=-1)
    cosine_value = float(cosine[mask].mean().detach().cpu())
    smooth_value = float(smooth_l1[mask].mean().detach().cpu())
    return {
        "cosine": cosine_value,
        "smooth_l1": smooth_value,
        "combined": cosine_value + smooth_value,
        "matched_frame_count": count,
        "coverage": float(count / mask.numel()),
    }


def _encode_l6(encoder: torch.nn.Module, waveform: Tensor) -> Tensor:
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape [B,1,N]")
    output = encoder(waveform[:, 0, :], output_hidden_states=True)
    hidden = _extract_hidden_states(output)
    features = hidden[TARGET_LAYER]
    if features.ndim != 3 or features.shape[0] != waveform.shape[0]:
        raise ValueError("HuBERT layer-6 output has an invalid shape")
    if not torch.isfinite(features).all():
        raise FloatingPointError("HuBERT layer-6 output is non-finite")
    return features


def _read_item(item: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    natural = _read_stage1_wave(item["natural_path"]).to(device)
    tts = _read_stage1_wave(item["tts_path"], allow_24k_resample=True).to(device)
    encoder, interface = _load_stage1_hubert(device=device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    natural_features = _encode_l6(encoder, natural)
    with torch.no_grad():
        tts_features = _encode_l6(encoder, tts)
        aligned, mask, alignment = align_tts_features_to_natural(
            natural_features.detach(),
            tts_features.detach(),
            item["matched_spans"],
            frame_stride_samples=int(interface.frame_stride_samples),
            sample_rate=int(interface.sample_rate),
            min_frames=1,
        )
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    item["_encoder"] = encoder
    item["_interface"] = interface.as_dict()
    item["_alignment"] = alignment
    item["_natural_features"] = natural_features.detach()
    item["_tts_features"] = tts_features.detach()
    item["_aligned_target"] = aligned.detach()
    item["_target_mask"] = mask.detach()
    return natural, tts, item["matched_spans"]


def _spans_for_warp(spans: Sequence[Mapping[str, Any]], side: str) -> list[dict[str, Any]]:
    if side not in {"natural", "tts"}:
        raise ValueError("side must be natural or tts")
    return [
        {
            "label": str(span["label"]),
            "start_s": float(span[f"{side}_start_s"]),
            "end_s": float(span[f"{side}_end_s"]),
            "confidence": min(float(span[f"{side}_confidence"]), 1.0),
        }
        for span in spans
    ]


def _phone_warp_metrics(
    natural: Tensor,
    tts: Tensor,
    spans: Sequence[Mapping[str, Any]],
    encoder: torch.nn.Module,
    target: Tensor,
    mask: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    warped = phone_local_warp(
        natural[0, 0].detach().cpu().numpy(),
        tts[0, 0].detach().cpu().numpy(),
        TARGET_SAMPLE_RATE,
        _spans_for_warp(spans, "natural"),
        _spans_for_warp(spans, "tts"),
        min_confidence=0.8,
    )
    waveform = torch.from_numpy(warped.audio).to(device=device)[None, None, :]
    with torch.no_grad():
        features = _encode_l6(encoder, waveform)
    metrics = _masked_feature_metrics(features, target, mask)
    metrics.update(
        {
            "exact_sample_count": bool(warped.audio.size == natural.shape[-1]),
            "sample_count": int(warped.audio.size),
            "finite": bool(np.isfinite(warped.audio).all()),
            "peak": float(np.abs(warped.audio).max()),
            "warp_metadata": warped.metadata,
        }
    )
    return metrics


def _optimize_direct_waveform(
    natural: Tensor,
    target: Tensor,
    mask: Tensor,
    encoder: torch.nn.Module,
    *,
    mode: str,
    residual_scale: float,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    if mode not in {"bounded", "unbounded"}:
        raise ValueError("mode must be bounded or unbounded")
    if residual_scale <= 0 or steps <= 0 or learning_rate <= 0:
        raise ValueError("residual_scale, steps, and learning_rate must be positive")
    raw_residual = torch.nn.Parameter(torch.zeros_like(natural))
    optimizer = Adam([raw_residual], lr=learning_rate)
    history: list[dict[str, float | int]] = []
    best: dict[str, Any] | None = None
    for step in range(steps):
        residual = residual_scale * torch.tanh(raw_residual) if mode == "bounded" else raw_residual
        enhanced = natural + residual
        predicted = _encode_l6(encoder, enhanced)
        cosine = 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1, eps=1e-8)
        smooth_l1 = F.smooth_l1_loss(predicted, target.detach(), reduction="none").mean(dim=-1)
        loss = cosine[mask].mean() + smooth_l1[mask].mean()
        metrics = _masked_feature_metrics(predicted, target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        residual_now = residual.detach()
        row = {
            **metrics,
            "step": step,
            "loss": float(loss.detach().cpu()),
            "residual_peak": float(residual_now.abs().max().cpu()),
            "residual_rms": float(residual_now.square().mean().sqrt().cpu()),
            "clipped_sample_count": int((enhanced.detach().abs() >= 1.0).sum().cpu()),
            "bound_hit_count": int((residual_now.abs() >= residual_scale - 1e-5).sum().cpu()) if mode == "bounded" else 0,
        }
        history.append(row)
        if best is None or row["combined"] < best["combined"]:
            best = row
    with torch.no_grad():
        final_residual = (
            residual_scale * torch.tanh(raw_residual) if mode == "bounded" else raw_residual
        )
        final_waveform = natural + final_residual
        final_predicted = _encode_l6(encoder, final_waveform)
        final_metrics = _masked_feature_metrics(final_predicted, target, mask)
    final = {
        **final_metrics,
        "step": steps,
        "loss": final_metrics["combined"],
        "residual_peak": float(final_residual.abs().max().cpu()),
        "residual_rms": float(final_residual.square().mean().sqrt().cpu()),
        "clipped_sample_count": int((final_waveform.abs() >= 1.0).sum().cpu()),
        "bound_hit_count": int((final_residual.abs() >= residual_scale - 1e-5).sum().cpu()) if mode == "bounded" else 0,
    }
    if best is None or final["combined"] < best["combined"]:
        best = final
    assert best is not None
    initial = history[0]
    return {
        "mode": mode,
        "residual_scale": residual_scale,
        "steps": steps,
        "learning_rate": learning_rate,
        "initial": initial,
        "best": best,
        "final": final,
        "gap_reduction_fraction": float(
            (initial["combined"] - final["combined"]) / max(initial["combined"], 1e-12)
        ),
        "history": history,
    }


def run(
    manifest_path: str | Path,
    *,
    paired_key: str | None = None,
    device: str = "cuda",
    steps: int = 50,
    learning_rate: float = 0.01,
    residual_scales: Sequence[float] = (0.05, 0.2),
) -> dict[str, Any]:
    records = load_two_stage_manifest(manifest_path)
    train_records = sorted(
        (record for record in records if record["split"] == "train"),
        key=lambda record: str(record["paired_key"]),
    )
    if not train_records:
        raise ValueError("manifest has no train records")
    selected = next(
        (record for record in train_records if paired_key is None or record["paired_key"] == paired_key),
        None,
    )
    if selected is None:
        raise ValueError(f"paired_key not found: {paired_key}")
    device_obj = torch.device(device)
    natural, tts, spans = _read_item(selected, device_obj)
    encoder = selected.pop("_encoder")
    interface = selected.pop("_interface")
    alignment = selected.pop("_alignment")
    natural_features = selected.pop("_natural_features")
    tts_features = selected.pop("_tts_features")
    target = selected.pop("_aligned_target")
    mask = selected.pop("_target_mask")
    with torch.no_grad():
        natural_metrics = _masked_feature_metrics(natural_features, target, mask)
    result: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_type": "direct_waveform_hubert_upper_bound",
        "paired_key": str(selected["paired_key"]),
        "device": str(device_obj),
        "objective": {
            "feature_layer": TARGET_LAYER,
            "target": "aligned_raw_tts_layer6_matched_phone_spans",
            "loss": "masked_cosine_plus_smooth_l1",
            "preservation_terms": [],
            "waveform_length": "natural_exact_sample_count",
        },
        "source_provenance": {
            "natural_sha256": str(selected["natural_sha256"]),
            "tts_sha256": str(selected["tts_sha256"]),
            "matched_spans_sha256": str(selected["matched_spans_sha256"]),
        },
        "encoder_interface": interface,
        "alignment": alignment,
        "target_mask": {
            "matched_frame_count": int(mask.sum().item()),
            "natural_frame_count": int(mask.numel()),
            "coverage": float(mask.float().mean().cpu()),
        },
        "natural_to_aligned_target": natural_metrics,
        "phone_warp": _phone_warp_metrics(
            natural,
            tts,
            spans,
            encoder,
            target,
            mask,
            device_obj,
        ),
        "direct_optimization": [],
    }
    for scale in residual_scales:
        result["direct_optimization"].append(
            _optimize_direct_waveform(
                natural,
                target,
                mask,
                encoder,
                mode="bounded",
                residual_scale=float(scale),
                steps=steps,
                learning_rate=learning_rate,
            )
        )
    result["direct_optimization"].append(
        _optimize_direct_waveform(
            natural,
            target,
            mask,
            encoder,
            mode="unbounded",
            residual_scale=1.0,
            steps=steps,
            learning_rate=learning_rate,
        )
    )
    return _finite_json(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paired-key")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.manifest,
        paired_key=args.paired_key,
        device=args.device,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "diagnostic_type": result["diagnostic_type"],
        "paired_key": result["paired_key"],
        "natural_to_aligned_target": result["natural_to_aligned_target"],
        "phone_warp": result["phone_warp"],
        "direct_best": [row["best"] for row in result["direct_optimization"]],
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
