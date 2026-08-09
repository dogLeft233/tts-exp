#!/usr/bin/env python3
"""Differentiable CPU-side losses and a minimal paired waveform trainer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import soundfile as sf
import torch
from torch import Tensor
from torch.optim import AdamW

from pnp_audio_enhancer import ResidualTCN


@dataclass(frozen=True)
class LossWeights:
    """Weights for the minimum viable acoustic objective."""

    stft: float = 1.0
    mel: float = 0.3
    waveform: float = 0.1
    ssl: float = 0.1
    energy: float = 0.05
    residual: float = 0.01


def _check_pair(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("prediction and target must have shape [batch, 1, samples]")
    if prediction.shape != target.shape or prediction.shape[1] != 1:
        raise ValueError("prediction and target must have equal [batch, 1, samples] shapes")
    if prediction.shape[-1] < 2:
        raise ValueError("waveforms must contain at least two samples")
    return prediction[:, 0], target[:, 0]


def _stft_magnitude(waveform: Tensor, n_fft: int, hop_length: int) -> Tensor:
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    ).abs().clamp_min(1e-7)


def multi_resolution_stft_loss(
    prediction: Tensor,
    target: Tensor,
    fft_sizes: Sequence[int] = (256, 512, 1024),
) -> Tensor:
    """Compare log magnitude and spectral convergence at multiple resolutions."""
    pred, truth = _check_pair(prediction, target)
    losses: list[Tensor] = []
    for n_fft in fft_sizes:
        if n_fft > pred.shape[-1]:
            continue
        hop = max(1, n_fft // 4)
        pred_mag = _stft_magnitude(pred, n_fft, hop)
        truth_mag = _stft_magnitude(truth, n_fft, hop)
        log_loss = torch.mean(torch.abs(torch.log(pred_mag) - torch.log(truth_mag)))
        convergence = torch.linalg.vector_norm(pred_mag - truth_mag) / (
            torch.linalg.vector_norm(truth_mag) + 1e-7
        )
        losses.append(log_loss + convergence)
    if not losses:
        raise ValueError("no fft size is smaller than the waveform")
    return torch.stack(losses).mean()


def _hz_to_mel(hz: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: Tensor) -> Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a triangular mel filterbank as a fixed differentiable constant."""
    if sample_rate <= 0 or n_fft <= 0 or n_mels <= 0:
        raise ValueError("sample_rate, n_fft, and n_mels must be positive")
    nyquist = sample_rate / 2.0
    f_max = nyquist if f_max is None else f_max
    if not 0 <= f_min < f_max <= nyquist:
        raise ValueError("mel frequency bounds must be within [0, Nyquist]")
    mel_points = torch.linspace(
        float(_hz_to_mel(torch.tensor(f_min))),
        float(_hz_to_mel(torch.tensor(f_max))),
        n_mels + 2,
        device=device,
        dtype=dtype,
    )
    hz_points = _mel_to_hz(mel_points)
    frequencies = torch.linspace(0.0, nyquist, n_fft // 2 + 1, device=device, dtype=dtype)
    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    rising = (frequencies[None, :] - lower) / (center - lower).clamp_min(1e-7)
    falling = (upper - frequencies[None, :]) / (upper - center).clamp_min(1e-7)
    return torch.clamp(torch.minimum(rising, falling), min=0.0)


def log_mel_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 512,
    n_mels: int = 80,
) -> Tensor:
    """Compare log mel power spectra while keeping gradients to prediction."""
    pred, truth = _check_pair(prediction, target)
    if n_fft > pred.shape[-1]:
        raise ValueError("n_fft must not exceed waveform length")
    hop = max(1, n_fft // 4)
    pred_mag = _stft_magnitude(pred, n_fft, hop)
    truth_mag = _stft_magnitude(truth, n_fft, hop)
    filters = mel_filterbank(
        sample_rate,
        n_fft,
        n_mels,
        device=pred.device,
        dtype=pred.dtype,
    )
    pred_mel = torch.einsum("mf,bft->bmt", filters, pred_mag.square()).clamp_min(1e-7)
    truth_mel = torch.einsum("mf,bft->bmt", filters, truth_mag.square()).clamp_min(1e-7)
    return torch.mean(torch.abs(torch.log(pred_mel) - torch.log(truth_mel)))


def energy_envelope_loss(prediction: Tensor, target: Tensor, frame_size: int = 320) -> Tensor:
    """Match short-time absolute-amplitude envelopes."""
    pred, truth = _check_pair(prediction, target)
    kernel = min(max(2, frame_size), pred.shape[-1])
    stride = max(1, kernel // 2)
    pred_env = torch.nn.functional.avg_pool1d(pred.abs()[:, None, :], kernel, stride, ceil_mode=True)
    truth_env = torch.nn.functional.avg_pool1d(truth.abs()[:, None, :], kernel, stride, ceil_mode=True)
    length = min(pred_env.shape[-1], truth_env.shape[-1])
    return torch.mean(torch.abs(pred_env[..., :length] - truth_env[..., :length]))


def acoustic_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    input_waveform: Tensor | None = None,
    ssl_encoder: torch.nn.Module | None = None,
    target_ssl_features: Tensor | None = None,
    weights: LossWeights = LossWeights(),
    sample_rate: int = 16_000,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return weighted losses with an optional frozen SSL teacher branch."""
    _check_pair(prediction, target)
    terms: dict[str, Tensor] = {
        "stft": multi_resolution_stft_loss(prediction, target),
        "mel": log_mel_loss(prediction, target, sample_rate=sample_rate),
        "waveform": torch.mean(torch.abs(prediction - target)),
        "energy": energy_envelope_loss(prediction, target),
    }
    if ssl_encoder is None:
        if target_ssl_features is not None:
            raise ValueError("target_ssl_features requires ssl_encoder")
        terms["ssl"] = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    else:
        ssl_encoder.eval()
        for parameter in ssl_encoder.parameters():
            parameter.requires_grad_(False)
        student_features = ssl_encoder(prediction)
        if target_ssl_features is None:
            with torch.no_grad():
                target_ssl_features = ssl_encoder(target)
        assert target_ssl_features is not None
        if student_features.shape != target_ssl_features.shape:
            raise ValueError("student and target SSL features must have equal shapes")
        terms["ssl"] = torch.mean(torch.abs(student_features - target_ssl_features.detach()))
    if input_waveform is None:
        terms["residual"] = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    else:
        _check_pair(prediction, input_waveform)
        terms["residual"] = torch.mean(torch.abs(prediction - input_waveform))
    weighted = (
        weights.stft * terms["stft"]
        + weights.mel * terms["mel"]
        + weights.waveform * terms["waveform"]
        + weights.ssl * terms["ssl"]
        + weights.energy * terms["energy"]
        + weights.residual * terms["residual"]
    )
    return weighted, terms


def load_cache_items(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load cache item records written by ``pnp_cache_targets.py``."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("cache manifest must contain an items list")
    return [dict(item) for item in items]


def _read_item(item: Mapping[str, Any], sample_rate: int) -> tuple[Tensor, Tensor]:
    target, target_sr = sf.read(item["audio_path"], dtype="float32", always_2d=False)
    natural, natural_sr = sf.read(item["natural_path"], dtype="float32", always_2d=False)
    if target_sr != sample_rate or natural_sr != sample_rate:
        raise ValueError("cached input and target must use the trainer sample rate")
    if target.ndim != 1 or natural.ndim != 1 or len(target) != len(natural):
        raise ValueError("cached input and target must be mono and exact-length")
    return torch.from_numpy(natural[None, None, :]), torch.from_numpy(target[None, None, :])


def train_paired(
    items: Iterable[Mapping[str, Any]],
    *,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    model: ResidualTCN | None = None,
    ssl_encoder: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
    weights: LossWeights = LossWeights(),
    checkpoint_path: str | Path | None = None,
    sample_rate: int = 16_000,
) -> list[dict[str, float]]:
    """Train on cached paired utterances, one full utterance at a time."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    records = [dict(item) for item in items]
    if not records:
        raise ValueError("items must not be empty")
    device_obj = torch.device(device)
    network = model or ResidualTCN()
    network.to(device_obj)
    network.train()
    optimizer = AdamW(network.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        total = 0.0
        for item in records:
            input_waveform, target = _read_item(item, sample_rate)
            input_waveform = input_waveform.to(device_obj)
            target = target.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            prediction = network(input_waveform)
            loss, _ = acoustic_loss(
                prediction,
                target,
                input_waveform=input_waveform,
                ssl_encoder=ssl_encoder,
                weights=weights,
                sample_rate=sample_rate,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append({"epoch": float(epoch + 1), "loss": total / len(records)})
    if checkpoint_path is not None:
        output = Path(checkpoint_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "model_config": {"class": "ResidualTCN"},
                "loss_weights": asdict(weights),
                "sample_rate": sample_rate,
                "history": history,
            },
            output,
        )
    return history


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="pnp_cache_targets manifest.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    history = train_paired(
        load_cache_items(args.manifest),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(history, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
