#!/usr/bin/env python3
"""Train a bounded contextual adapter for MFA-linear WavLM trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from knn_vc_retrieval import mfa_linear_target  # noqa: E402
from pilot_generate_mfa_linear import extend_tokens_for_feature_tail  # noqa: E402
from run_aishell1_vocoder_domain_split import (  # noqa: E402
    ALL_N25_SPEAKERS,
    KNN_VC_REVISION,
    SAMPLE_RATE,
    feature_distance,
    load_adapter,
    load_inputs,
)

FEATURE_DIM = 1024
FRAME_STRIDE = 320
TRAIN_SPEAKERS = tuple(speaker for speaker in ALL_N25_SPEAKERS if speaker != "S0765")
VALID_SPEAKER = "S0765"
EVAL_SPEAKERS = ("S0765", "S0901", "S0912")


class PhoneTrajectoryAdapter(nn.Module):
    """Identity-initialized normalized feature denoiser with bounded residuals."""

    def __init__(
        self,
        mean: Tensor,
        std: Tensor,
        *,
        hidden_channels: int = 128,
        dilations: Sequence[int] = (1, 2, 4, 8),
        max_normalized_delta: float = 0.25,
    ) -> None:
        super().__init__()
        if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
            raise ValueError("normalizer tensors must have shape [feature_dim]")
        if torch.any(std <= 0) or hidden_channels <= 0 or not dilations:
            raise ValueError("normalizer and model dimensions must be positive")
        if max_normalized_delta <= 0:
            raise ValueError("max_normalized_delta must be positive")
        self.dimension = FEATURE_DIM
        self.hidden_channels = int(hidden_channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.max_normalized_delta = float(max_normalized_delta)
        self.register_buffer("mean", mean.detach().clone().float())
        self.register_buffer("std", std.detach().clone().float())
        self.input_projection = nn.Conv1d(FEATURE_DIM, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(self._block(hidden_channels, dilation) for dilation in self.dilations)
        )
        self.output_projection = nn.Conv1d(hidden_channels, FEATURE_DIM, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _block(channels: int, dilation: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"features must have shape [B,T,{FEATURE_DIM}]")
        normalized = (features - self.mean) / self.std
        hidden = self.input_projection(normalized.transpose(1, 2))
        hidden = self.blocks(hidden)
        delta = self.max_normalized_delta * torch.tanh(self.output_projection(hidden)).transpose(1, 2)
        output = features + delta * self.std
        if not torch.isfinite(output).all():
            raise FloatingPointError("trajectory adapter produced non-finite features")
        return output

    def config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "dimension": FEATURE_DIM,
            "hidden_channels": self.hidden_channels,
            "dilations": list(self.dilations),
            "max_normalized_delta": self.max_normalized_delta,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "identity_initialized": True,
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_wave(path: str | Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim != 1 or int(sample_rate) != SAMPLE_RATE or not np.isfinite(values).all():
        raise ValueError(f"invalid 16 kHz mono audio: {path}")
    return np.asarray(values, dtype=np.float32)


def _boundary_indices(tokens: Sequence[Mapping[str, Any]], frame_count: int) -> list[int]:
    times = (torch.arange(frame_count, dtype=torch.float32) + 0.5) * FRAME_STRIDE / SAMPLE_RATE
    return [int(torch.argmin(torch.abs(times - float(token["end_s"]))).item()) for token in tokens[:-1]]


def _localize_tokens(
    tokens: Sequence[Mapping[str, Any]],
    start_frame: int,
    frame_count: int,
) -> list[dict[str, float]]:
    offset_s = start_frame * FRAME_STRIDE / SAMPLE_RATE
    end_s = offset_s + frame_count * FRAME_STRIDE / SAMPLE_RATE
    localized: list[dict[str, float]] = []
    for token in tokens:
        start_s = max(0.0, float(token["start_s"]) - offset_s)
        token_end_s = min(end_s, float(token["end_s"])) - offset_s
        if token_end_s > start_s:
            localized.append({"start_s": start_s, "end_s": token_end_s})
    return localized


def corrupt_trajectory(
    features: Tensor,
    tokens: Sequence[Mapping[str, Any]],
    rng: random.Random,
    *,
    max_span_frames: int = 3,
) -> Tensor:
    """Create a boundary-aware interpolation corruption for denoising training."""
    if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"features must have shape [T,{FEATURE_DIM}]")
    if max_span_frames <= 0:
        raise ValueError("max_span_frames must be positive")
    output = features.clone()
    centers = _boundary_indices(tokens, features.shape[0])
    if not centers:
        centers = [features.shape[0] // 2]
    selected = [centers[rng.randrange(len(centers))]]
    if features.shape[0] > max_span_frames * 2 + 2:
        selected.append(rng.randrange(1, features.shape[0] - 1))
    for center in selected:
        span = min(rng.randint(1, max_span_frames), max(1, (features.shape[0] - 1) // 2))
        left = max(0, center - span)
        right = min(features.shape[0] - 1, center + span)
        if right <= left:
            continue
        for index in range(left, right + 1):
            weight = float(index - left) / float(right - left)
            output[index] = (1.0 - weight) * features[left] + weight * features[right]
    return output


def _normalizer(items: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    if not items:
        raise ValueError("no feature items")
    values = torch.cat(items, dim=0).float()
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-3)
    return mean, std


def _load_tts_features(inputs: Sequence[Mapping[str, Any]], extractor: Any, device: torch.device) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in inputs:
        waveform = torch.from_numpy(_read_wave(str(row["tts_path"]))).to(device)
        with torch.no_grad():
            features = extractor.extract(waveform.unsqueeze(0)).detach().clone()
        items.append({
            "sample_id": str(row["sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "features": features,
            "tokens": list(row["tts_tokens"]),
        })
    return items


def train_adapter(
    items: Sequence[Mapping[str, Any]],
    adapter: PhoneTrajectoryAdapter,
    *,
    steps: int,
    crop_frames: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float | str]]:
    if not items or steps <= 0:
        raise ValueError("items and steps must be positive")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    history: list[dict[str, float | str]] = []
    for step in range(steps):
        item = items[rng.randrange(len(items))]
        features = item["features"]
        length = min(crop_frames, int(features.shape[0]))
        start = rng.randrange(int(features.shape[0] - length + 1))
        clean = features[start : start + length].unsqueeze(0)
        local_tokens = _localize_tokens(item["tokens"], start, length)
        corrupted = corrupt_trajectory(clean.squeeze(0), local_tokens, rng).unsqueeze(0)
        prediction = adapter(corrupted)
        target = clean.detach()
        loss_feature = F.smooth_l1_loss(prediction, target)
        loss_velocity = F.smooth_l1_loss(
            prediction[:, 1:] - prediction[:, :-1],
            target[:, 1:] - target[:, :-1],
        )
        loss_residual = torch.mean(torch.abs(prediction - corrupted))
        loss = loss_feature + 0.2 * loss_velocity + 0.001 * loss_residual
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        history.append({
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "feature_loss": float(loss_feature.detach().cpu()),
            "velocity_loss": float(loss_velocity.detach().cpu()),
            "residual_loss": float(loss_residual.detach().cpu()),
            "speaker_id": str(item["speaker_id"]),
        })
    return history


def _audio_qc(values: np.ndarray, natural_samples: int) -> dict[str, Any]:
    return {
        "sample_count": int(values.size),
        "duration_s": float(values.size / SAMPLE_RATE),
        "natural_duration_s": float(natural_samples / SAMPLE_RATE),
        "exact_natural_length": bool(values.size == natural_samples),
        "finite": bool(np.isfinite(values).all()),
        "peak": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
        "clipped_sample_count": int(np.sum(np.abs(values) >= 0.999)),
    }


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    extractor: Any,
    vocoder: nn.Module,
    adapter: PhoneTrajectoryAdapter,
    device: torch.device,
    outdir: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        natural = torch.from_numpy(_read_wave(str(row["natural_path"]))).to(device)
        tts = torch.from_numpy(_read_wave(str(row["tts_path"]))).to(device)
        with torch.no_grad():
            natural_features = extractor.extract(natural.unsqueeze(0))
            tts_features = extractor.extract(tts.unsqueeze(0))
        natural_tokens, _ = extend_tokens_for_feature_tail(list(row["natural_tokens"]), natural_features.shape[0])
        tts_tokens, _ = extend_tokens_for_feature_tail(list(row["tts_tokens"]), tts_features.shape[0])
        with torch.no_grad():
            mfa_features, mfa_meta = mfa_linear_target(natural_features.shape[0], tts_features, natural_tokens, tts_tokens)
            adapted_features = adapter(mfa_features.unsqueeze(0)).squeeze(0)
            base_output = vocoder(mfa_features.unsqueeze(0)).squeeze(0).squeeze(0)
            adapted_output = vocoder(adapted_features.unsqueeze(0)).squeeze(0).squeeze(0)
        for variant, features, output_tensor in (
            ("mfa_linear_regular", mfa_features, base_output),
            ("trajectory_regular", adapted_features, adapted_output),
        ):
            output = output_tensor.detach().cpu().numpy().astype(np.float32)
            if output.size > natural.numel():
                output = output[: natural.numel()]
                length_action = "right_crop"
            elif output.size < natural.numel():
                output = np.pad(output, (0, natural.numel() - output.size)).astype(np.float32)
                length_action = "right_zero_pad"
            else:
                length_action = "none"
            path = outdir / "audio" / variant / f"{row['sample_id']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, output, SAMPLE_RATE, subtype="FLOAT")
            with torch.no_grad():
                encoded = extractor.extract(torch.from_numpy(output).to(device).unsqueeze(0))
            results.append({
                "sample_id": str(row["sample_id"]),
                "speaker_id": str(row["speaker_id"]),
                "paired_key": row["paired_key"],
                "variant": variant,
                "audio_path": str(path),
                "audio_sha256": _sha256(path),
                "length_action": length_action,
                "audio_qc": _audio_qc(output, int(natural.numel())),
                "output_to_condition_distance": feature_distance(encoded, features),
                "output_to_raw_tts_distance": feature_distance(encoded, tts_features),
                "output_to_natural_distance": feature_distance(encoded, natural_features),
                "mfa_linear_meta": mfa_meta,
            })
    return results


def run(
    cohort_path: Path,
    tts_meta_path: Path,
    tokens_path: Path,
    reference_mfa_path: Path,
    outdir: Path,
    *,
    device: str,
    steps: int,
    crop_frames: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    device_obj = torch.device(device)
    train_inputs = load_inputs(cohort_path, tts_meta_path, tokens_path, reference_mfa_path, TRAIN_SPEAKERS)
    eval_inputs = load_inputs(cohort_path, tts_meta_path, tokens_path, reference_mfa_path, EVAL_SPEAKERS)
    vocoder_adapter = load_adapter(device, prematched=False)
    extractor = vocoder_adapter
    train_items = _load_tts_features(train_inputs, extractor, device_obj)
    mean, std = _normalizer([item["features"] for item in train_items])
    adapter = PhoneTrajectoryAdapter(mean.to(device_obj), std.to(device_obj)).to(device_obj)
    history = train_adapter(train_items, adapter, steps=steps, crop_frames=crop_frames, learning_rate=learning_rate, seed=seed)
    checkpoint = outdir / "adapter.pt"
    torch.save({
        "schema_version": 1,
        "model_state_dict": adapter.state_dict(),
        "model_config": adapter.config(),
        "train_speakers": list(TRAIN_SPEAKERS),
        "eval_speakers": list(EVAL_SPEAKERS),
        "knn_vc_revision": KNN_VC_REVISION,
        "vocoder": vocoder_adapter.metadata(),
        "seed": seed,
        "steps": steps,
        "crop_frames": crop_frames,
        "learning_rate": learning_rate,
    }, checkpoint)
    results = _evaluate(eval_inputs, extractor, vocoder_adapter.model.hifigan, adapter, device_obj, outdir)
    summary = {
        "schema_version": 1,
        "experiment": "aishell1_phone_trajectory_denoising_adapter",
        "status": "complete",
        "train_speakers": list(TRAIN_SPEAKERS),
        "eval_speakers": list(EVAL_SPEAKERS),
        "train_pair_count": len(train_inputs),
        "eval_pair_count": len(eval_inputs),
        "train_item_count": len(train_items),
        "steps": steps,
        "crop_frames": crop_frames,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": device,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "adapter": adapter.config(),
        "history": history,
        "results": results,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def write_report(summary: Mapping[str, Any], outdir: Path) -> None:
    rows = list(summary["results"])
    lines = [
        "# Learned phone trajectory denoising adapter",
        "",
        "## Protocol",
        "",
        "A bounded identity-initialized temporal adapter is trained on native raw-TTS WavLM-L6 phone trajectories after artificial boundary/frame interpolation corruption. It is applied to MFA-linear natural-clock trajectories and vocoded by frozen regular HiFi-GAN. Four speakers train; S0765 is speaker-disjoint validation. No SyncNet or S0770 is used.",
        "",
        "## Summary",
        "",
        "| Speaker | Variant | Mean peak | Mean RMS | Mean output→condition distance | Mean output→raw-TTS distance | Clipped |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for speaker in summary["eval_speakers"]:
        for variant in ("mfa_linear_regular", "trajectory_regular"):
            subset = [row for row in rows if row["speaker_id"] == speaker and row["variant"] == variant]
            if not subset:
                continue
            qc = [row["audio_qc"] for row in subset]
            condition_distance = [row["output_to_condition_distance"]["cosine_distance"] for row in subset]
            raw_distance = [row["output_to_raw_tts_distance"]["cosine_distance"] for row in subset]
            lines.append(
                f"| {speaker} | {variant} | {np.mean([x['peak'] for x in qc]):.4f} | {np.mean([x['rms'] for x in qc]):.4f} | {np.mean(condition_distance):.4f} | {np.mean(raw_distance):.4f} | {sum(x['clipped_sample_count'] for x in qc)} |"
            )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "The native raw-TTS denoising target is a self-supervised proxy, not a hybrid waveform ground truth. Lower feature distance or peak alone does not establish pronunciation, speaker preservation, naturalness or lip-sync.",
    ])
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--tts-meta", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--reference-mfa", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--crop-frames", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    summary = run(
        args.cohort.resolve(), args.tts_meta.resolve(), args.tokens.resolve(), args.reference_mfa.resolve(), args.outdir.resolve(),
        device=args.device, steps=args.steps, crop_frames=args.crop_frames, learning_rate=args.learning_rate, seed=args.seed,
    )
    write_report(summary, args.outdir.resolve())
    print(json.dumps({"status": summary["status"], "results": len(summary["results"]), "checkpoint": summary["checkpoint"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
