#!/usr/bin/env python3
"""Train the compact local rhythm/style waveform generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rhythm_style_generator import GeneratorConfig, RhythmStyleGenerator

from train_pnp_audio_enhancer import (
    energy_envelope_loss,
    log_mel_loss,
    mel_filterbank,
    multi_resolution_stft_loss,
)


@dataclass(frozen=True)
class LossWeights:
    stft: float = 1.0
    mel: float = 0.3
    waveform: float = 0.1
    content: float = 0.2
    style: float = 0.05
    rhythm: float = 0.05
    energy: float = 0.05
    residual: float = 0.01


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(dataset_dir: str | Path, split: str) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / f"{split}.pt"
    try:
        records = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions without weights_only
        records = torch.load(path, map_location="cpu")
    if not isinstance(records, list) or not records:
        raise ValueError(f"dataset split is empty or invalid: {path}")
    return [dict(record) for record in records]


def _crop(record: Mapping[str, Any], crop_frames: int, rng: random.Random) -> dict[str, Tensor]:
    natural = record["natural"].float()
    target = record["weak_target"].float()
    content = record["natural_content"].float()
    rhythm = record["rhythm"].float()
    style = record["style_features"].float()
    frame_count = min(content.shape[0], rhythm.shape[0])
    crop_frames = min(max(1, crop_frames), frame_count)
    start_frame = rng.randrange(frame_count - crop_frames + 1) if frame_count > crop_frames else 0
    start_sample = start_frame * 320
    end_sample = min(natural.shape[0], (start_frame + crop_frames) * 320)
    sample_count = end_sample - start_sample
    return {
        "natural_waveform": natural[start_sample:end_sample].reshape(1, -1),
        "target_waveform": target[start_sample:end_sample].reshape(1, -1),
        "content": content[start_frame : start_frame + crop_frames],
        "rhythm": rhythm[start_frame : start_frame + crop_frames],
        "style_features": style,
        "frame_mask": torch.ones(crop_frames, dtype=torch.bool),
        "sample_count": torch.tensor(sample_count, dtype=torch.long),
    }


def _batch(records: list[dict[str, Any]], crop_frames: int, rng: random.Random) -> dict[str, Tensor]:
    samples = [_crop(record, crop_frames, rng) for record in records]
    n_samples = max(sample["natural_waveform"].shape[-1] for sample in samples)
    n_frames = max(sample["content"].shape[0] for sample in samples)
    n_style = max(sample["style_features"].shape[0] for sample in samples)
    result: dict[str, Tensor] = {
        "natural_waveform": torch.zeros(len(samples), 1, n_samples),
        "target_waveform": torch.zeros(len(samples), 1, n_samples),
        "content": torch.zeros(len(samples), n_frames, samples[0]["content"].shape[-1]),
        "rhythm": torch.zeros(len(samples), n_frames, samples[0]["rhythm"].shape[-1]),
        "style_features": torch.zeros(len(samples), n_style, samples[0]["style_features"].shape[-1]),
        "style_mask": torch.zeros(len(samples), n_style, dtype=torch.bool),
        "frame_mask": torch.zeros(len(samples), n_frames, dtype=torch.bool),
        "sample_mask": torch.zeros(len(samples), 1, n_samples),
        "sample_count": torch.zeros(len(samples), dtype=torch.long),
    }
    for index, sample in enumerate(samples):
        n = sample["natural_waveform"].shape[-1]
        f = sample["content"].shape[0]
        s = sample["style_features"].shape[0]
        result["natural_waveform"][index, :, :n] = sample["natural_waveform"]
        result["target_waveform"][index, :, :n] = sample["target_waveform"]
        result["content"][index, :f] = sample["content"]
        result["rhythm"][index, :f] = sample["rhythm"]
        result["style_features"][index, :s] = sample["style_features"]
        result["style_mask"][index, :s] = True
        result["frame_mask"][index, :f] = True
        result["sample_mask"][index, :, :n] = 1.0
        result["sample_count"][index] = n
    return result


def _waveform_style_summary(waveform: Tensor, sample_rate: int = 16_000) -> Tensor:
    """Return differentiable coarse acoustic statistics from a waveform."""
    values = waveform[:, 0]
    rms = torch.sqrt(values.square().mean(dim=-1, keepdim=True).clamp_min(1e-7))
    stft = torch.stft(
        values,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window=torch.hann_window(512, device=values.device, dtype=values.dtype),
        return_complex=True,
    ).abs().clamp_min(1e-7)
    mel = torch.einsum(
        "mf,bft->bmt",
        mel_filterbank(sample_rate, 512, 80, device=values.device, dtype=values.dtype),
        stft.square(),
    ).clamp_min(1e-7)
    log_mel = torch.log(mel)
    return torch.stack(
        [
            torch.log(rms.squeeze(-1)),
            log_mel.mean(dim=(1, 2)),
            log_mel.std(dim=(1, 2)),
        ],
        dim=-1,
    )


def _reference_style_summary(features: Tensor, mask: Tensor | None = None) -> Tensor:
    """Extract waveform-comparable log-mel summary from [B,T,84] features."""
    mel = features[..., :80]
    log_rms = features[..., 80]
    if mask is None:
        weights = torch.ones(features.shape[:2], device=features.device, dtype=features.dtype)
    else:
        weights = mask.to(features.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mel_weights = weights.unsqueeze(-1).expand_as(mel)
    mel_weight_sum = mel_weights.sum(dim=(1, 2)).clamp_min(1.0)
    mel_mean = (mel * mel_weights).sum(dim=(1, 2)) / mel_weight_sum
    mel_centered = mel - mel_mean[:, None, None]
    mel_std = torch.sqrt(
        ((mel_centered.square() * mel_weights).sum(dim=(1, 2)) / mel_weight_sum).clamp_min(1e-8)
    )
    return torch.stack([(log_rms * weights).sum(dim=1), mel_mean, mel_std], dim=-1)


def _pad_for_spectral_loss(values: Tensor, minimum: int = 512) -> Tensor:
    if values.shape[-1] >= minimum:
        return values
    return F.pad(values, (0, minimum - values.shape[-1]))


def _masked_waveform_terms(
    prediction: Tensor,
    target: Tensor,
    natural: Tensor,
    sample_mask: Tensor,
    style_reference: Tensor,
    sample_rate: int,
) -> dict[str, Tensor]:
    terms: dict[str, list[Tensor]] = {
        name: [] for name in ("stft", "mel", "waveform", "energy", "style", "residual")
    }
    for index in range(prediction.shape[0]):
        valid = int(sample_mask[index, 0].sum().item())
        if valid <= 0:
            continue
        pred = prediction[index : index + 1, :, :valid]
        truth = target[index : index + 1, :, :valid]
        identity = natural[index : index + 1, :, :valid]
        spectral_pred = _pad_for_spectral_loss(pred)
        spectral_truth = _pad_for_spectral_loss(truth)
        terms["stft"].append(
            multi_resolution_stft_loss(spectral_pred, spectral_truth, fft_sizes=(256, 512))
        )
        terms["mel"].append(
            log_mel_loss(spectral_pred, spectral_truth, sample_rate=sample_rate, n_fft=512)
        )
        terms["waveform"].append(torch.mean(torch.abs(pred - truth)))
        terms["energy"].append(energy_envelope_loss(pred, truth))
        terms["style"].append(
            torch.mean(
                torch.abs(
                    _waveform_style_summary(spectral_pred, sample_rate)
                    - style_reference[index : index + 1]
                )
            )
        )
        terms["residual"].append(torch.mean(torch.abs(pred - identity)))
    if not terms["stft"]:
        raise ValueError("batch contains no valid waveform samples")
    return {name: torch.stack(values).mean() for name, values in terms.items()}


def acoustic_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    *,
    weights: LossWeights = LossWeights(),
    sample_rate: int = 16_000,
) -> tuple[Tensor, dict[str, Tensor]]:
    if multi_resolution_stft_loss is None or log_mel_loss is None or energy_envelope_loss is None:
        raise RuntimeError("waveform loss helpers are unavailable")
    prediction = output["waveform"]
    target = batch["target_waveform"]
    natural = batch["natural_waveform"]
    content_target = batch["content"]
    frame_mask = batch["frame_mask"]
    sample_mask = batch.get("sample_mask")
    if sample_mask is None:
        sample_mask = torch.ones_like(prediction)
    style_reference = _reference_style_summary(batch["style_features"], batch.get("style_mask"))
    waveform_terms = _masked_waveform_terms(
        prediction, target, natural, sample_mask, style_reference, sample_rate
    )
    content_values = output["content_prediction"][frame_mask] - content_target[frame_mask]
    rhythm_values = output["rhythm_prediction"][frame_mask] - batch["rhythm"][frame_mask]
    terms: dict[str, Tensor] = {
        **waveform_terms,
        "content": torch.mean(torch.abs(content_values)),
        "rhythm": torch.mean(torch.abs(rhythm_values)),
    }
    total = (
        weights.stft * terms["stft"]
        + weights.mel * terms["mel"]
        + weights.waveform * terms["waveform"]
        + weights.content * terms["content"]
        + weights.style * terms["style"]
        + weights.rhythm * terms["rhythm"]
        + weights.energy * terms["energy"]
        + weights.residual * terms["residual"]
        + 0.01 * terms["residual"]
    )
    return total, terms


def _manifest_hash(dataset_dir: Path) -> str:
    path = dataset_dir / "dataset_manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _try_load_split(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    path = dataset_dir / f"{split}.pt"
    if not path.exists():
        return []
    try:
        records = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        records = torch.load(path, map_location="cpu")
    return [dict(record) for record in records] if isinstance(records, list) else []


def _epoch(
    model: RhythmStyleGenerator,
    records: list[dict[str, Any]],
    *,
    crop_frames: int,
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: Any = None,
    use_amp: bool = False,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {"loss": 0.0}
    steps = 0
    for start in range(0, len(records), batch_size):
        batch = _batch(records[start : start + batch_size], crop_frames, rng)
        batch = {key: value.to(device) for key, value in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(
                    batch["natural_waveform"], batch["content"], batch["rhythm"],
                    batch["style_features"], frame_mask=batch["frame_mask"],
                    style_mask=batch["style_mask"],
                )
                loss, terms = acoustic_loss(output, batch)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        sums["loss"] += float(loss.detach().cpu())
        for name, value in terms.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().cpu())
        steps += 1
    if not steps:
        raise ValueError("no batches in epoch")
    return {name: value / steps for name, value in sums.items()}


def _checkpoint_payload(
    model: RhythmStyleGenerator,
    optimizer: AdamW,
    scaler: Any,
    epoch: int,
    history: list[dict[str, float]],
    dataset_root: Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "loss_weights": asdict(LossWeights()),
        "sample_rate": 16_000,
        "seed": seed,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataset_manifest_hash": _manifest_hash(dataset_root),
        "target_type": "tts_phone_local_warp_weak_supervision",
        "weak_target_warning": "phone-local warp is supervision only, not a validated disentangled target",
        "history": history,
        "device": str(device),
    }


def train(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 1,
    crop_frames: int = 128,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    amp: bool = False,
    grad_clip: float = 1.0,
    seed: int = 42,
    max_items: int | None = None,
    resume: bool = False,
    best_checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(seed)
    dataset_root = Path(dataset_dir)
    records = load_split(dataset_root, "train")
    if max_items is not None:
        records = records[:max_items]
    if not records:
        raise ValueError("no training records")
    valid_records = _try_load_split(dataset_root, "valid")
    device_obj = torch.device(device)
    checkpoint = Path(checkpoint_path)
    model = RhythmStyleGenerator().to(device_obj)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    use_amp = bool(amp and device_obj.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
    history: list[dict[str, float]] = []
    start_epoch = 0
    if resume and checkpoint.exists():
        try:
            payload = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location=device_obj)
        model = RhythmStyleGenerator(GeneratorConfig(**payload.get("model_config", {}))).to(device_obj)
        model.load_state_dict(payload["model_state_dict"])
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        if payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload.get("scaler_state_dict"):
            scaler.load_state_dict(payload["scaler_state_dict"])
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("epoch", 0))
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    best_path = Path(best_checkpoint_path) if best_checkpoint_path else checkpoint.with_name(f"{checkpoint.stem}_best{checkpoint.suffix}")
    best_value = min((row.get("val_loss", float("inf")) for row in history), default=float("inf"))
    for epoch in range(start_epoch, start_epoch + epochs):
        train_metrics = _epoch(
            model, records, crop_frames=crop_frames, batch_size=batch_size,
            rng=random.Random(seed + epoch), device=device_obj, optimizer=optimizer,
            scaler=scaler, use_amp=use_amp, grad_clip=grad_clip,
        )
        valid_metrics = _epoch(
            model, valid_records, crop_frames=crop_frames, batch_size=batch_size,
            rng=random.Random(seed + 100_000 + epoch), device=device_obj,
        ) if valid_records else {"loss": float("nan")}
        row = {"epoch": float(epoch + 1), "loss": train_metrics["loss"], "val_loss": valid_metrics["loss"]}
        row.update({f"train_{name}": value for name, value in train_metrics.items() if name != "loss"})
        row.update({f"val_{name}": value for name, value in valid_metrics.items() if name != "loss"})
        history.append(row)
        payload = _checkpoint_payload(model, optimizer, scaler, epoch + 1, history, dataset_root, seed, device_obj)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint)
        score = valid_metrics["loss"] if valid_records else train_metrics["loss"]
        if score < best_value:
            best_value = score
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, best_path)
    return history


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-frames", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--best-checkpoint", type=Path)
    args = parser.parse_args(argv)
    history = train(
        args.dataset, args.checkpoint, epochs=args.epochs, batch_size=args.batch_size,
        crop_frames=args.crop_frames, learning_rate=args.learning_rate,
        device=args.device, amp=args.amp, grad_clip=args.grad_clip,
        seed=args.seed, max_items=args.max_items, resume=args.resume,
        best_checkpoint_path=args.best_checkpoint,
    )
    print(json.dumps(history, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
