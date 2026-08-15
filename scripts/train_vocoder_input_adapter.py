#!/usr/bin/env python3
"""Train a tiny feature affine adapter to distill regular HiFi-GAN behavior."""

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
    KNN_VC_LOCAL,
    SAMPLE_RATE,
    feature_distance,
    load_adapter,
    load_inputs,
)

FEATURE_DIM = 1024
FRAME_STRIDE = 320
TRAIN_SPEAKERS = tuple(speaker for speaker in ALL_N25_SPEAKERS if speaker != "S0765")
VALID_SPEAKER = "S0765"
CONDITIONS = ("natural_direct", "raw_tts_direct", "mfa_linear")


class FeatureAffineAdapter(nn.Module):
    """Identity-initialized bounded per-feature affine calibration."""

    def __init__(self, dimension: int = FEATURE_DIM, max_scale_delta: float = 0.1, max_bias: float = 0.1) -> None:
        super().__init__()
        if dimension <= 0 or max_scale_delta <= 0 or max_bias <= 0:
            raise ValueError("adapter dimensions and bounds must be positive")
        self.dimension = int(dimension)
        self.max_scale_delta = float(max_scale_delta)
        self.max_bias = float(max_bias)
        self.scale_logits = nn.Parameter(torch.zeros(dimension))
        self.bias_logits = nn.Parameter(torch.zeros(dimension))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.dimension:
            raise ValueError(f"features must have shape [B,T,{self.dimension}]")
        scale = 1.0 + self.max_scale_delta * torch.tanh(self.scale_logits)
        bias = self.max_bias * torch.tanh(self.bias_logits)
        return features * scale.view(1, 1, -1) + bias.view(1, 1, -1)

    def regularization(self) -> Tensor:
        return self.scale_logits.square().mean() + self.bias_logits.square().mean()

    def config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "dimension": self.dimension,
            "max_scale_delta": self.max_scale_delta,
            "max_bias": self.max_bias,
            "identity_initialized": True,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
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


def _spectral_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("spectral loss requires equal waveform shapes")
    window = torch.hann_window(256, device=prediction.device, dtype=prediction.dtype)
    pred = torch.stft(prediction, n_fft=256, hop_length=64, win_length=256, window=window, return_complex=True).abs()
    truth = torch.stft(target, n_fft=256, hop_length=64, win_length=256, window=window, return_complex=True).abs()
    return F.l1_loss(torch.log1p(pred), torch.log1p(truth))


def _waveform_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(prediction, target)


def _crop_feature_audio(features: Tensor, waveform: Tensor, crop_frames: int, rng: random.Random) -> tuple[Tensor, Tensor]:
    if features.ndim != 2 or waveform.ndim != 1:
        raise ValueError("features and waveform must be [T,D] and [N]")
    if features.shape[0] < 2:
        raise ValueError("feature sequence is too short")
    length = min(int(crop_frames), int(features.shape[0]))
    start = rng.randrange(int(features.shape[0] - length + 1))
    end = start + length
    target = waveform[start * FRAME_STRIDE : end * FRAME_STRIDE]
    target_length = length * FRAME_STRIDE
    if target.numel() < target_length:
        target = F.pad(target, (0, target_length - target.numel()))
    return features[start:end], target[:target_length]


def _prepare_native_items(inputs: Sequence[Mapping[str, Any]], extractor: Any, device: torch.device) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in inputs:
        for arm, path_key in (("natural", "natural_path"), ("raw_tts", "tts_path")):
            waveform = torch.from_numpy(_read_wave(str(row[path_key]))).to(device)
            with torch.no_grad():
                features = extractor.extract(waveform.unsqueeze(0)).detach().clone()
            items.append({
                "sample_id": str(row["sample_id"]),
                "speaker_id": str(row["speaker_id"]),
                "arm": arm,
                "features": features,
                "waveform": waveform,
            })
    return items


def _vocode(model: nn.Module, features: Tensor) -> Tensor:
    output = model(features.unsqueeze(0)).squeeze(0).squeeze(0)
    if output.ndim != 1 or not torch.isfinite(output).all():
        raise FloatingPointError("HiFi-GAN returned invalid waveform")
    return output


def train_adapter(
    train_items: Sequence[Mapping[str, Any]],
    regular_model: nn.Module,
    prematched_model: nn.Module,
    device: torch.device,
    *,
    steps: int,
    crop_frames: int,
    learning_rate: float,
    seed: int,
) -> tuple[FeatureAffineAdapter, list[dict[str, float]]]:
    if not train_items or steps <= 0:
        raise ValueError("training items and steps must be positive")
    adapter = FeatureAffineAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    regular_model.eval()
    prematched_model.eval()
    for step in range(steps):
        item = train_items[rng.randrange(len(train_items))]
        features, target_shape = _crop_feature_audio(
            item["features"], item["waveform"], crop_frames, rng
        )
        features = features.unsqueeze(0)
        with torch.no_grad():
            teacher = regular_model(features).squeeze(0).squeeze(0)
        calibrated = adapter(features)
        prediction = prematched_model(calibrated).squeeze(0).squeeze(0)
        loss_spectral = _spectral_loss(prediction.unsqueeze(0), teacher.unsqueeze(0))
        loss_waveform = _waveform_loss(prediction, teacher)
        loss = loss_spectral + 0.1 * loss_waveform + 1e-4 * adapter.regularization()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        history.append({
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "spectral_loss": float(loss_spectral.detach().cpu()),
            "waveform_loss": float(loss_waveform.detach().cpu()),
            "speaker_id": item["speaker_id"],
            "arm": item["arm"],
        })
    return adapter, history


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


def _evaluate_record(
    row: Mapping[str, Any],
    extractor: Any,
    regular_model: nn.Module,
    prematched_model: nn.Module,
    adapter: FeatureAffineAdapter,
    device: torch.device,
    outdir: Path,
) -> list[dict[str, Any]]:
    natural = torch.from_numpy(_read_wave(str(row["natural_path"]))).to(device)
    tts = torch.from_numpy(_read_wave(str(row["tts_path"]))).to(device)
    with torch.no_grad():
        natural_features = extractor.extract(natural.unsqueeze(0))
        tts_features = extractor.extract(tts.unsqueeze(0))
    natural_tokens, _ = extend_tokens_for_feature_tail(list(row["natural_tokens"]), natural_features.shape[0])
    tts_tokens, _ = extend_tokens_for_feature_tail(list(row["tts_tokens"]), tts_features.shape[0])
    with torch.no_grad():
        mfa_features, mfa_meta = mfa_linear_target(natural_features.shape[0], tts_features, natural_tokens, tts_tokens)
    feature_map = {
        "natural_direct": natural_features,
        "raw_tts_direct": tts_features,
        "mfa_linear": mfa_features,
    }
    results: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        features = feature_map[condition]
        if condition == "mfa_linear":
            target_samples = natural.numel()
        else:
            target_samples = None
        with torch.no_grad():
            base_regular = _vocode(regular_model, features)
            base_prematched = _vocode(prematched_model, features)
            adapted = _vocode(prematched_model, adapter(features.unsqueeze(0)).squeeze(0))
        variants = {
            "regular": base_regular,
            "prematched": base_prematched,
            "adapted_prematched": adapted,
        }
        for variant, output_tensor in variants.items():
            output = output_tensor.detach().cpu().numpy().astype(np.float32)
            length_action = "native_feature_length"
            if target_samples is not None:
                if output.size > target_samples:
                    output = output[:target_samples]
                    length_action = "right_crop"
                elif output.size < target_samples:
                    output = np.pad(output, (0, target_samples - output.size)).astype(np.float32)
                    length_action = "right_zero_pad"
            path = outdir / "audio" / f"{condition}_{variant}" / f"{row['sample_id']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, output, SAMPLE_RATE, subtype="FLOAT")
            with torch.no_grad():
                encoded = extractor.extract(torch.from_numpy(output).to(device).unsqueeze(0))
            results.append({
                "sample_id": str(row["sample_id"]),
                "speaker_id": str(row["speaker_id"]),
                "paired_key": row["paired_key"],
                "condition": condition,
                "variant": variant,
                "audio_path": str(path),
                "audio_sha256": _sha256(path),
                "length_action": length_action,
                "audio_qc": _audio_qc(output, int(natural.numel())),
                "output_to_condition_distance": feature_distance(encoded, features),
                "output_to_raw_tts_distance": feature_distance(encoded, tts_features),
                "output_to_natural_distance": feature_distance(encoded, natural_features),
                "mfa_linear_meta": mfa_meta if condition == "mfa_linear" else None,
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
    device_obj = torch.device(device)
    train_inputs = load_inputs(cohort_path, tts_meta_path, tokens_path, reference_mfa_path, TRAIN_SPEAKERS)
    valid_inputs = load_inputs(cohort_path, tts_meta_path, tokens_path, reference_mfa_path, (VALID_SPEAKER,))
    regular_adapter = load_adapter(device, prematched=False)
    prematched_adapter = load_adapter(device, prematched=True)
    regular_model = regular_adapter.model.hifigan
    prematched_model = prematched_adapter.model.hifigan
    for model in (regular_model, prematched_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    train_items = _prepare_native_items(train_inputs, prematched_adapter, device_obj)
    adapter, history = train_adapter(
        train_items,
        regular_model,
        prematched_model,
        device_obj,
        steps=steps,
        crop_frames=crop_frames,
        learning_rate=learning_rate,
        seed=seed,
    )
    checkpoint = outdir / "adapter.pt"
    torch.save({
        "schema_version": 1,
        "model_state_dict": adapter.state_dict(),
        "model_config": adapter.config(),
        "train_speakers": list(TRAIN_SPEAKERS),
        "valid_speaker": VALID_SPEAKER,
        "knn_vc_revision": KNN_VC_REVISION,
        "regular_vocoder": regular_adapter.metadata(),
        "prematched_vocoder": prematched_adapter.metadata(),
        "seed": seed,
        "steps": steps,
        "crop_frames": crop_frames,
        "learning_rate": learning_rate,
    }, checkpoint)
    results: list[dict[str, Any]] = []
    for row in valid_inputs:
        results.extend(_evaluate_record(row, prematched_adapter, regular_model, prematched_model, adapter, device_obj, outdir))
    summary = {
        "schema_version": 1,
        "experiment": "aishell1_regular_teacher_prematched_feature_affine_adapter",
        "status": "complete",
        "train_speakers": list(TRAIN_SPEAKERS),
        "valid_speaker": VALID_SPEAKER,
        "train_pair_count": len(train_inputs),
        "valid_pair_count": len(valid_inputs),
        "train_native_item_count": len(train_items),
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
        "# Regular-teacher prematched-vocoder feature affine adapter",
        "",
        "## Protocol",
        "",
        "A 2048-parameter identity-initialized feature affine adapter is trained on six-speaker-excluded native natural/raw-TTS features to make frozen prematched HiFi-GAN match frozen regular HiFi-GAN log-spectrum and waveform behavior. S0765 is speaker-disjoint validation. No SyncNet or S0770 is used.",
        "",
        "## Validation summary",
        "",
        "| Condition | Variant | Mean peak | Mean RMS | Mean output→condition distance | Clipped |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for variant in ("regular", "prematched", "adapted_prematched"):
            subset = [row for row in rows if row["condition"] == condition and row["variant"] == variant]
            if not subset:
                continue
            qc = [row["audio_qc"] for row in subset]
            distances = [row["output_to_condition_distance"]["cosine_distance"] for row in subset]
            lines.append(
                f"| {condition} | {variant} | {np.mean([x['peak'] for x in qc]):.4f} | {np.mean([x['rms'] for x in qc]):.4f} | {np.mean(distances):.4f} | {sum(x['clipped_sample_count'] for x in qc)} |"
            )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "The regular-vocoder output is a frozen teacher, not a ground-truth hybrid waveform. Improvement means successful domain calibration only; it does not establish pronunciation, target-speaker preservation, naturalness or lip-sync.",
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
    parser.add_argument("--steps", type=int, default=250)
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
