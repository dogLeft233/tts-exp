#!/usr/bin/env python3
"""Compare phone-level pooled and constrained hard-DTW HuBERT targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from hubert_feature_alignment import align_tts_features_to_natural, frame_centers
from prepare_two_stage_hubert_manifest import load_two_stage_manifest
from train_tts_aligned_feature_puller import _load_stage1_hubert, _read_stage1_wave
from direct_waveform_hubert_upper_bound import (
    _encode_l6,
    _finite_json,
    _masked_feature_metrics,
    _optimize_direct_waveform,
)

TARGET_SAMPLE_RATE = 16_000
FRAME_STRIDE_SAMPLES = 320


def _span_frames(
    spans: Sequence[Mapping[str, Any]],
    natural_count: int,
    tts_count: int,
    *,
    device: torch.device,
) -> list[tuple[Mapping[str, Any], Tensor, Tensor]]:
    natural_centers = frame_centers(
        natural_count,
        frame_stride_samples=FRAME_STRIDE_SAMPLES,
        sample_rate=TARGET_SAMPLE_RATE,
        device=device,
    )
    tts_centers = frame_centers(
        tts_count,
        frame_stride_samples=FRAME_STRIDE_SAMPLES,
        sample_rate=TARGET_SAMPLE_RATE,
        device=device,
    )
    result: list[tuple[Mapping[str, Any], Tensor, Tensor]] = []
    for span in spans:
        n_start = float(span["natural_start_s"])
        n_end = float(span["natural_end_s"])
        t_start = float(span["tts_start_s"])
        t_end = float(span["tts_end_s"])
        n_indices = torch.where((natural_centers >= n_start) & (natural_centers < n_end))[0]
        t_indices = torch.where((tts_centers >= t_start) & (tts_centers < t_end))[0]
        result.append((span, n_indices, t_indices))
    return result


def _cosine_cost(left: Tensor, right: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(
        F.normalize(left, dim=-1),
        F.normalize(right, dim=-1),
        dim=-1,
        eps=1e-8,
    )


def _hard_dtw_path(cost: Tensor, *, band_ratio: float = 0.5) -> list[tuple[int, int]]:
    if cost.ndim != 2 or cost.shape[0] <= 0 or cost.shape[1] <= 0:
        raise ValueError("cost must have non-empty [N,M] shape")
    if not 0.0 <= band_ratio <= 1.0:
        raise ValueError("band_ratio must be in [0,1]")
    n, m = cost.shape
    inf = float("inf")
    dp = torch.full((n, m), inf, dtype=cost.dtype, device=cost.device)
    back = torch.full((n, m), -1, dtype=torch.int8, device=cost.device)
    for i in range(n):
        for j in range(m):
            if n > 1 and m > 1:
                ratio_gap = abs(i / (n - 1) - j / (m - 1))
                if ratio_gap > band_ratio:
                    continue
            if i == 0 and j == 0:
                dp[i, j] = cost[i, j]
                continue
            candidates: list[tuple[Tensor, int]] = []
            if i > 0 and torch.isfinite(dp[i - 1, j]):
                candidates.append((dp[i - 1, j], 0))
            if j > 0 and torch.isfinite(dp[i, j - 1]):
                candidates.append((dp[i, j - 1], 1))
            if i > 0 and j > 0 and torch.isfinite(dp[i - 1, j - 1]):
                candidates.append((dp[i - 1, j - 1], 2))
            if candidates:
                best_value, best_step = min(candidates, key=lambda item: float(item[0]))
                dp[i, j] = cost[i, j] + best_value
                back[i, j] = best_step
    if not torch.isfinite(dp[-1, -1]):
        raise ValueError("band constraint admits no complete DTW path")
    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        step = int(back[i, j].item())
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        elif step == 2:
            i -= 1
            j -= 1
        else:
            raise RuntimeError("invalid DTW backpointer")
    path.reverse()
    return path


def _pool_target(
    natural_features: Tensor,
    tts_features: Tensor,
    span_frames: Sequence[tuple[Mapping[str, Any], Tensor, Tensor]],
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    target = torch.zeros_like(natural_features)
    mask = torch.zeros(natural_features.shape[0], dtype=torch.bool, device=natural_features.device)
    used = 0
    skipped_natural = 0
    skipped_tts = 0
    for _, n_indices, t_indices in span_frames:
        if n_indices.numel() == 0:
            skipped_natural += 1
            continue
        if t_indices.numel() == 0:
            skipped_tts += 1
            continue
        target[n_indices] = tts_features[t_indices].mean(dim=0)
        mask[n_indices] = True
        used += 1
    return target, mask, {
        "strategy": "phone_level_mean_pool_source_grid",
        "used_spans": used,
        "skipped_natural_spans": skipped_natural,
        "skipped_tts_spans": skipped_tts,
    }


def _dtw_target(
    natural_features: Tensor,
    tts_features: Tensor,
    span_frames: Sequence[tuple[Mapping[str, Any], Tensor, Tensor]],
    *,
    band_ratio: float,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    target = torch.zeros_like(natural_features)
    mask = torch.zeros(natural_features.shape[0], dtype=torch.bool, device=natural_features.device)
    paths: list[dict[str, Any]] = []
    skipped_natural = 0
    skipped_tts = 0
    for span, n_indices, t_indices in span_frames:
        if n_indices.numel() == 0:
            skipped_natural += 1
            continue
        if t_indices.numel() == 0:
            skipped_tts += 1
            continue
        natural_part = natural_features[n_indices]
        tts_part = tts_features[t_indices]
        cost = 1.0 - F.cosine_similarity(
            F.normalize(natural_part, dim=-1)[:, None, :],
            F.normalize(tts_part, dim=-1)[None, :, :],
            dim=-1,
            eps=1e-8,
        )
        path = _hard_dtw_path(cost, band_ratio=band_ratio)
        grouped: list[list[int]] = [[] for _ in range(n_indices.numel())]
        for i, j in path:
            grouped[i].append(j)
        for local_i, t_js in enumerate(grouped):
            if not t_js:
                raise RuntimeError("DTW path did not cover a natural frame")
            target[n_indices[local_i]] = tts_part[t_js].mean(dim=0)
            mask[n_indices[local_i]] = True
        paths.append(
            {
                "label": str(span["label"]),
                "natural_frame_count": int(n_indices.numel()),
                "tts_frame_count": int(t_indices.numel()),
                "path_length": len(path),
                "path_cost": float(cost[[i for i, _ in path], [j for _, j in path]].mean().cpu()),
                "path": [[int(i), int(j)] for i, j in path],
            }
        )
    return target, mask, {
        "strategy": "phone_anchored_hard_dtw_source_grid",
        "band_ratio": band_ratio,
        "used_spans": len(paths),
        "skipped_natural_spans": skipped_natural,
        "skipped_tts_spans": skipped_tts,
        "paths": paths,
    }


def _load_selected(manifest: str | Path, paired_key: str | None, device: torch.device) -> tuple[dict[str, Any], torch.nn.Module, Tensor, Tensor, Tensor, Tensor, Tensor, list[dict[str, Any]]]:
    records = load_two_stage_manifest(manifest)
    train = sorted((r for r in records if r["split"] == "train"), key=lambda r: str(r["paired_key"]))
    record = next((r for r in train if paired_key is None or r["paired_key"] == paired_key), None)
    if record is None:
        raise ValueError(f"paired_key not found: {paired_key}")
    natural = _read_stage1_wave(record["natural_path"]).to(device)
    tts = _read_stage1_wave(record["tts_path"], allow_24k_resample=True).to(device)
    encoder, interface = _load_stage1_hubert(device=device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        natural_l6 = _encode_l6(encoder, natural)
        tts_l6 = _encode_l6(encoder, tts)
    return record, encoder, natural, tts, natural_l6[0], tts_l6[0], interface, record["matched_spans"]


def run(
    manifest: str | Path,
    *,
    paired_key: str | None = None,
    device: str = "cuda",
    steps: int = 30,
    learning_rate: float = 0.01,
    band_ratio: float = 0.5,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    record, encoder, natural, tts, natural_l6, tts_l6, interface, spans = _load_selected(manifest, paired_key, device_obj)
    span_frames = _span_frames(spans, natural_l6.shape[0], tts_l6.shape[0], device=device_obj)
    current_target, current_mask, current_alignment = align_tts_features_to_natural(
        natural_l6,
        tts_l6,
        spans,
        frame_stride_samples=int(interface.frame_stride_samples),
        sample_rate=int(interface.sample_rate),
        min_frames=1,
    )
    pooled_target, pooled_mask, pooled_alignment = _pool_target(natural_l6, tts_l6, span_frames)
    dtw_target, dtw_mask, dtw_alignment = _dtw_target(
        natural_l6,
        tts_l6,
        span_frames,
        band_ratio=band_ratio,
    )
    target_specs = {
        "mfa_linear": (current_target, current_mask, current_alignment),
        "phone_pool": (pooled_target, pooled_mask, pooled_alignment),
        "hard_dtw": (dtw_target, dtw_mask, dtw_alignment),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_type": "phone_alignment_target_audit",
        "paired_key": str(record["paired_key"]),
        "device": str(device_obj),
        "target_specs": {},
    }
    for name, (target, mask, alignment) in target_specs.items():
        natural_metrics = _masked_feature_metrics(natural_l6.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0))
        optimized = _optimize_direct_waveform(
            natural,
            target.unsqueeze(0),
            mask.unsqueeze(0),
            encoder,
            mode="bounded",
            residual_scale=0.2,
            steps=steps,
            learning_rate=learning_rate,
        )
        result["target_specs"][name] = {
            "alignment": alignment,
            "mask": {
                "matched_frame_count": int(mask.sum().item()),
                "natural_frame_count": int(mask.numel()),
                "coverage": float(mask.float().mean().cpu()),
            },
            "natural_to_target": natural_metrics,
            "direct_waveform": {
                "initial": optimized["initial"],
                "final": optimized["final"],
                "gap_reduction_fraction": optimized["gap_reduction_fraction"],
            },
        }
    return _finite_json(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paired-key")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--band-ratio", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.manifest,
        paired_key=args.paired_key,
        device=args.device,
        steps=args.steps,
        learning_rate=args.learning_rate,
        band_ratio=args.band_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for name, value in result["target_specs"].items():
        print(name, value["mask"], value["natural_to_target"]["combined"], value["direct_waveform"]["final"]["combined"], value["direct_waveform"]["gap_reduction_fraction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
