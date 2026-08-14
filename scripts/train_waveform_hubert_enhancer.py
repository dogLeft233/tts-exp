#!/usr/bin/env python3
"""Train the single-encoder HuBERT waveform enhancer with strict diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor
from torch.optim import AdamW

from train_pnp_audio_enhancer import (  # type: ignore[import-not-found]
    energy_envelope_loss,
    log_mel_loss,
    multi_resolution_stft_loss,
)
from waveform_hubert_enhancer import (  # type: ignore[import-not-found]
    DEFAULT_FRAME_STRIDE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    TARGET_SAMPLE_RATE,
    EncoderInterface,
    HuBERTWaveformEnhancer,
    _extract_hidden_states,
    _finite_tensor,
)
from hubert_feature_alignment import (  # type: ignore[import-not-found]
    align_tts_features_to_natural,
    masked_hubert_proximity,
    masked_tts_feature_losses,
)

SCHEMA_VERSION = 1
EXPERIMENT_TYPE = "single_encoder_waveform_enhancer"


@dataclass(frozen=True)
class LossWeights:
    stft: float = 1.0
    mel: float = 0.3
    waveform: float = 0.1
    hubert: float = 0.1
    hubert_tts: float = 1.0
    hubert_delta: float = 0.25
    hubert_content: float = 0.15
    energy: float = 0.05
    residual: float = 0.01

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _finite_scalar(value: Tensor, name: str) -> Tensor:
    if value.ndim != 0 or not torch.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite")
    return value


def _audio_stats(audio: Tensor) -> dict[str, float | int | bool]:
    values = audio.detach().float().reshape(-1)
    _finite_tensor(values, "audio")
    return {
        "sample_count": int(values.numel()),
        "rms": float(torch.sqrt(torch.mean(values.square())).cpu()),
        "peak": float(values.abs().max().cpu()),
        "mean": float(values.mean().cpu()),
        "finite": True,
        "clipped_sample_count": int((values.abs() >= 1.0).sum().cpu()),
    }


def _feature_stats(features: Tensor) -> dict[str, float | int | bool]:
    values = features.detach().float()
    _finite_tensor(values, "features")
    return {
        "frames": int(values.shape[-2]),
        "dim": int(values.shape[-1]),
        "mean": float(values.mean().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
        "min": float(values.min().cpu()),
        "max": float(values.max().cpu()),
        "finite": True,
    }


def _gradient_norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    if not values:
        return 0.0
    result = torch.stack(values).norm()
    _finite_scalar(result, "gradient norm")
    return float(result.cpu())


def _parameter_norm(module: torch.nn.Module) -> float:
    values = [p.detach().float().norm() for p in module.parameters()]
    result = torch.stack(values).norm()
    _finite_scalar(result, "parameter norm")
    return float(result.cpu())


def _read_wave(
    path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    allow_24k_resample: bool = False,
) -> Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1 or audio.size < 1024:
        raise ValueError(f"audio must be mono with >=1024 samples: {path}")
    if int(sr) != sample_rate:
        if not allow_24k_resample or int(sr) != 24_000 or sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"audio must be {sample_rate} Hz: {path} (found {sr})")
        gcd = math.gcd(int(sr), sample_rate)
        audio = resample_poly(
            np.asarray(audio, dtype=np.float32),
            sample_rate // gcd,
            int(sr) // gcd,
        )
    values = torch.from_numpy(np.asarray(audio, dtype=np.float32)).view(1, 1, -1)
    _finite_tensor(values, f"audio {path}")
    return values


def _load_hubert(model_id: str = DEFAULT_MODEL_ID, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, EncoderInterface]:
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        model_id,
        output_hidden_states=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    config = encoder.config
    hidden_size = int(getattr(config, "hidden_size", -1))
    strides = getattr(config, "conv_stride", [5, 2, 2, 2, 2, 2, 2])
    stride = int(np.prod([int(value) for value in strides]))
    layers = int(getattr(config, "num_hidden_layers", -1))
    if hidden_size != DEFAULT_HIDDEN_SIZE or stride != DEFAULT_FRAME_STRIDE or DEFAULT_LAYER > layers:
        raise ValueError(
            f"unexpected HuBERT interface: hidden_size={hidden_size}, stride={stride}, layers={layers}"
        )
    interface = EncoderInterface(
        model_id=model_id,
        selected_layer=DEFAULT_LAYER,
        hidden_size=hidden_size,
        frame_stride_samples=stride,
        sample_rate=TARGET_SAMPLE_RATE,
        conditioning_dim=64,
        frozen=True,
    )
    return encoder, interface


def hubert_features(
    encoder: torch.nn.Module,
    waveform: Tensor,
    *,
    layer: int = DEFAULT_LAYER,
    expected_dim: int = DEFAULT_HIDDEN_SIZE,
    retain_gradient: bool = False,
) -> Tensor:
    """Extract one hidden state, retaining input gradients when requested."""
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape [B,1,N]")
    signal = waveform[:, 0, :]
    if retain_gradient:
        output = encoder(signal, output_hidden_states=True)
    else:
        with torch.no_grad():
            output = encoder(signal, output_hidden_states=True)
    hidden = _extract_hidden_states(output)
    if layer < 0 or layer >= len(hidden):
        raise ValueError(f"HuBERT layer {layer} is unavailable")
    features = hidden[layer]
    if features.ndim != 3 or features.shape[0] != waveform.shape[0] or features.shape[-1] != expected_dim:
        raise ValueError("HuBERT feature shape does not match the declared interface")
    return _finite_tensor(features, "HuBERT feature")


def hu_feature_loss(predicted: Tensor, target: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"HuBERT feature shapes differ: {predicted.shape} != {target.shape}")
    return torch.nn.functional.smooth_l1_loss(predicted, target.detach())


def waveform_loss(
    enhanced: Tensor,
    target: Tensor,
    natural: Tensor,
    predicted_features: Tensor,
    target_features: Tensor,
    *,
    aligned_tts_features: Tensor | None = None,
    style_mask: Tensor | None = None,
    natural_content_features: Tensor | None = None,
    enhanced_content_features: Tensor | None = None,
    natural_style_features: Tensor | None = None,
    min_style_frames: int = 1,
    weights: LossWeights = LossWeights(),
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[Tensor, dict[str, Tensor]]:
    if enhanced.shape != target.shape or enhanced.shape != natural.shape:
        raise ValueError("natural, target, and enhanced waveform shapes must match")
    terms: dict[str, Tensor] = {
        "stft": multi_resolution_stft_loss(enhanced, target),
        "mel": log_mel_loss(enhanced, target, sample_rate=sample_rate),
        "waveform": torch.mean(torch.abs(enhanced - target)),
        "hubert": hu_feature_loss(predicted_features, target_features),
        "energy": energy_envelope_loss(enhanced, target),
        "residual": torch.mean(torch.abs(enhanced - natural)),
    }
    if (aligned_tts_features is None) != (style_mask is None):
        raise ValueError("aligned TTS features and style mask must be supplied together")
    if aligned_tts_features is not None and style_mask is not None:
        style_terms = masked_tts_feature_losses(
            predicted_features,
            aligned_tts_features,
            style_mask,
            natural=natural_style_features,
            min_frames=min_style_frames,
        )
        terms["hubert_tts"] = style_terms["hubert_tts"]
        terms["hubert_delta"] = style_terms["hubert_delta"] if "hubert_delta" in style_terms else torch.zeros_like(terms["hubert_tts"])
    if (natural_content_features is None) != (enhanced_content_features is None):
        raise ValueError("natural and enhanced content features must be supplied together")
    if natural_content_features is not None and enhanced_content_features is not None:
        if natural_content_features.shape != enhanced_content_features.shape:
            raise ValueError("natural and enhanced content feature shapes differ")
        content_loss = torch.nn.functional.smooth_l1_loss(
            torch.nn.functional.normalize(enhanced_content_features, dim=-1),
            torch.nn.functional.normalize(natural_content_features.detach(), dim=-1),
        )
        terms["hubert_content"] = content_loss
    for name, value in terms.items():
        _finite_scalar(value, f"loss term {name}")
    weighted_names = set(weights.as_dict())
    total = sum(
        (getattr(weights, name) * value for name, value in terms.items() if name in weighted_names),
        torch.zeros_like(terms["stft"]),
    )
    return _finite_scalar(total, "total loss"), terms


def _record_hashes(item: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("natural_path", "tts_path", "target_path", "metadata_path"):
        if item.get(key):
            path = Path(str(item[key]))
            if not path.is_file():
                raise FileNotFoundError(path)
            result[f"{key}_sha256"] = file_sha256(path)
    return result


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _split_items(items: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    split_items: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "heldout": []}
    seen_keys: set[str] = set()
    speakers: dict[str, set[str]] = {}
    for raw in items:
        item = dict(raw)
        key = str(item.get("paired_key", ""))
        split = str(item.get("split", ""))
        speaker = str(item.get("speaker_group", item.get("speaker_id", "")))
        if not key or split not in split_items or not speaker:
            raise ValueError("accepted item requires paired_key, speaker_group, and train/valid/heldout split")
        if key in seen_keys:
            raise ValueError(f"duplicate paired_key: {key}")
        seen_keys.add(key)
        split_items[split].append(item)
        speakers.setdefault(speaker, set()).add(split)
    leaked = sorted(speaker for speaker, values in speakers.items() if len(values) > 1)
    if leaked:
        raise ValueError(f"speaker groups cross splits: {leaked}")
    if not split_items["train"] or not split_items["valid"]:
        raise ValueError("train and valid splits must not be empty")
    return split_items


def _load_target_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("target_definition") != "weak_phone_local_tts_target":
        raise ValueError("unsupported or untrusted target manifest")
    source_manifest = payload.get("source_manifest")
    source_hash = payload.get("source_manifest_sha256")
    if not source_manifest or not source_hash or file_sha256(source_manifest) != source_hash:
        raise ValueError("source manifest is missing or has changed")
    items = payload.get("accepted")
    if not isinstance(items, list) or not items:
        raise ValueError("target manifest must contain a non-empty accepted list")
    checked: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        if item.get("target_definition") != "weak_phone_local_tts_target":
            raise ValueError("accepted item has an incompatible target definition")
        for path_key, hash_key in (("natural_path", "natural_sha256"), ("tts_path", "tts_sha256"), ("target_path", "target_sha256"), ("metadata_path", None)):
            candidate = Path(str(item.get(path_key, "")))
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            if hash_key is not None and not item.get(hash_key):
                raise ValueError(f"{path_key} hash is missing")
            if hash_key and file_sha256(candidate) != item[hash_key]:
                raise ValueError(f"{path_key} hash mismatch")
        target_meta = json.loads(Path(item["metadata_path"]).read_text(encoding="utf-8"))
        if target_meta.get("target_definition") != "weak_phone_local_tts_target" or target_meta.get("target_definition_version") != 1:
            raise ValueError("target metadata has an incompatible definition")
        for field in ("natural_sha256", "tts_sha256", "natural_alignment_sha256", "tts_alignment_sha256"):
            if not target_meta.get(field) or not item.get(field) or target_meta[field] != item[field]:
                raise ValueError(f"target metadata {field} is missing or mismatched")
        if target_meta.get("target_sha256") != item.get("target_sha256"):
            raise ValueError("target metadata hash does not match manifest")
        checked.append(item)
    return checked


def train_items(
    model: HuBERTWaveformEnhancer,
    items: Sequence[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer,
    weights: LossWeights,
    device: torch.device,
    train: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if train:
        model.train()
    else:
        model.eval()
    totals: dict[str, float] = {name: 0.0 for name in (*weights.as_dict(), "total")}
    diagnostics: list[dict[str, Any]] = []
    for item in items:
        if not item.get("tts_path"):
            raise ValueError(f"raw TTS path is required for explicit style loss: {item.get('paired_key')}")
        natural = _read_wave(item["natural_path"]).to(device)
        target = _read_wave(item["target_path"]).to(device)
        if natural.shape != target.shape:
            raise ValueError(f"natural and target lengths differ for {item.get('paired_key')}")
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            output = model(natural, compute_enhanced_features=True)
            predicted = output["enhanced_features"]
            target_features = hubert_features(
                model.encoder,
                target,
                layer=model.interface.selected_layer,
                expected_dim=model.interface.hidden_size,
                retain_gradient=False,
            )
            tts = _read_wave(item["tts_path"], allow_24k_resample=True).to(device)
            tts_features = hubert_features(
                model.encoder,
                tts,
                layer=model.interface.selected_layer,
                expected_dim=model.interface.hidden_size,
                retain_gradient=False,
            )
            metadata = json.loads(Path(item["metadata_path"]).read_text(encoding="utf-8"))
            aligned_tts, style_mask, alignment_stats = align_tts_features_to_natural(
                output["natural_features"],
                tts_features,
                metadata.get("metadata", {}).get("spans", metadata.get("spans", [])),
                frame_stride_samples=model.interface.frame_stride_samples,
                sample_rate=model.interface.sample_rate,
                min_frames=1,
            )
            content_enhanced = output["enhanced_content_features"]
            loss, terms = waveform_loss(
                output["waveform"], target, natural, predicted, target_features,
                aligned_tts_features=aligned_tts,
                style_mask=style_mask,
                natural_content_features=output["natural_content_features"],
                enhanced_content_features=content_enhanced,
                natural_style_features=output["natural_features"],
                weights=weights,
            )
            proximity = masked_hubert_proximity(
                output["natural_features"],
                predicted,
                aligned_tts,
                style_mask,
                layer=model.interface.selected_layer,
                encoder_model=model.interface.model_id,
                min_frames=1,
            )
            if train:
                loss.backward()
                gradient_norm = _gradient_norm(model)
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("gradient norm is non-finite")
                optimizer.step()
            else:
                gradient_norm = 0.0
        _finite_tensor(output["waveform"], "enhanced waveform")
        row = {
            "paired_key": str(item.get("paired_key")),
            "split": str(item.get("split")),
            "speaker_group": str(item.get("speaker_group")),
            "natural": _audio_stats(natural),
            "target": _audio_stats(target),
            "enhanced": _audio_stats(output["waveform"]),
            "residual": _audio_stats(output["residual"]),
            "natural_features": _feature_stats(output["natural_features"]),
            "target_features": _feature_stats(target_features),
            "enhanced_features": _feature_stats(predicted),
            "style_target_features": _feature_stats(aligned_tts),
            "style_alignment": alignment_stats,
            "style_matched_frame_count": int(style_mask.sum().item()),
            "hubert_proximity": proximity,
            "feature_frame_count_equal": bool(predicted.shape == target_features.shape),
            "losses": {name: float(value.detach().cpu()) for name, value in terms.items()},
            "total_loss": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
            "hashes": _record_hashes(item),
        }
        diagnostics.append(row)
        for name, value in terms.items():
            totals[name] += float(value.detach().cpu())
        totals["total"] += float(loss.detach().cpu())
    divisor = float(len(items))
    means = {key: value / divisor for key, value in totals.items()}
    return means, diagnostics


def _trainable_state_dict(model: HuBERTWaveformEnhancer) -> dict[str, Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }

def make_checkpoint(
    model: HuBERTWaveformEnhancer,
    optimizer: torch.optim.Optimizer,
    *,
    history: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    manifest_path: str | Path,
    seed: int,
    weights: LossWeights,
) -> dict[str, Any]:
    split_items = _split_items(items)
    split_keys = {split: [str(item["paired_key"]) for item in rows] for split, rows in split_items.items()}
    path_identity: dict[str, set[str]] = {}
    for item in items:
        for key in ("natural_path", "tts_path", "target_path", "metadata_path"):
            if item.get(key):
                identity = f"path:{Path(str(item[key])).resolve()}"
                path_identity.setdefault(identity, set()).add(str(item["split"]))
        for key in ("natural_sha256", "tts_sha256", "target_sha256", "natural_alignment_sha256", "tts_alignment_sha256"):
            if item.get(key):
                identity = f"hash:{item[key]}"
                path_identity.setdefault(identity, set()).add(str(item["split"]))
        transcript = " ".join(str(item.get("transcript", "")).lower().split())
        if transcript:
            identity = f"transcript:{transcript}"
            path_identity.setdefault(identity, set()).add(str(item["split"]))
    leaked_audio = sorted(identity for identity, splits in path_identity.items() if len(splits) > 1)
    if leaked_audio:
        raise ValueError(f"natural audio crosses splits: {leaked_audio}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": EXPERIMENT_TYPE,
        "model_config": model.model_config(),
        "state_dict": _trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_weights": weights.as_dict(),
        "encoder_interface": model.interface.as_dict(),
        "target_definition": "weak_phone_local_tts_target",
        "weak_target_warning": "not disentangled ground truth",
        "split_keys": split_keys,
        "speaker_groups_by_split": {split: sorted({str(item["speaker_group"]) for item in rows}) for split, rows in split_items.items()},
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "source_hashes": {str(item["paired_key"]): _record_hashes(item) for item in items},
        "seed": int(seed),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "training_history": list(history),
    }
    return payload


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)


def load_checkpoint(path: str | Path, model: HuBERTWaveformEnhancer, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported waveform enhancer checkpoint schema")
    if payload.get("experiment_type") != EXPERIMENT_TYPE:
        raise ValueError("checkpoint is not a single-encoder waveform enhancer")
    interface = payload.get("encoder_interface")
    expected_interface = model.interface.as_dict()
    if interface != expected_interface:
        raise ValueError("checkpoint encoder interface does not match model")
    if payload.get("model_config") != model.model_config():
        raise ValueError("checkpoint model_config does not match model")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state_dict is missing")
    expected = {
        name: value for name, value in model.state_dict().items()
        if name in {key for key, parameter in model.named_parameters() if parameter.requires_grad}
    }
    if set(state) != set(expected):
        raise ValueError("checkpoint state keys do not match model")
    for key, value in state.items():
        if key not in expected or not isinstance(value, Tensor) or value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid checkpoint tensor: {key}")
    model.load_state_dict({**model.state_dict(), **state}, strict=True)
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def summarize_hubert_proximity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item proximity telemetry without recomputing features."""
    if not rows:
        raise ValueError("cannot aggregate HuBERT proximity for an empty split")
    required = (
        "coverage",
        "matched_frame_count",
        "natural_to_tts",
        "enhanced_to_tts",
        "toward_tts",
    )
    for row in rows:
        if not all(key in row for key in required):
            raise ValueError("proximity row is missing required fields")
    def mean(path: tuple[str, ...]) -> float:
        values = [float(_nested(row, path)) for row in rows]
        result = float(np.mean(values))
        if not math.isfinite(result):
            raise FloatingPointError("HuBERT proximity aggregate is non-finite")
        return result
    result = {
        "schema_version": 1,
        "item_count": len(rows),
        "total_matched_frame_count": int(sum(int(row["matched_frame_count"]) for row in rows)),
        "mean_matched_frame_count": mean(("matched_frame_count",)),
        "mean_coverage": mean(("coverage",)),
        "natural_to_tts": {
            "cosine": mean(("natural_to_tts", "cosine")),
            "smooth_l1": mean(("natural_to_tts", "smooth_l1")),
            "combined": mean(("natural_to_tts", "combined")),
        },
        "enhanced_to_tts": {
            "cosine": mean(("enhanced_to_tts", "cosine")),
            "smooth_l1": mean(("enhanced_to_tts", "smooth_l1")),
            "combined": mean(("enhanced_to_tts", "combined")),
        },
        "toward_tts": {
            "mean_absolute_delta": mean(("toward_tts", "absolute_delta")),
            "mean_relative_delta": mean(("toward_tts", "relative_delta")),
            "enhanced_closer_fraction": float(
                np.mean([
                    bool(row["toward_tts"]["enhanced_closer_than_natural"])
                    for row in rows
                ])
            ),
        },
        "finite": True,
    }
    return _json_safe(result)


def _nested(row: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"proximity row is missing {'/'.join(path)}")
        value = value[key]
    return value


def train_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    items = _load_target_manifest(manifest_path)
    split_items = _split_items(items)
    encoder, interface = _load_hubert(device=device)
    model = HuBERTWaveformEnhancer(encoder, interface=interface).to(device)
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)
    weights = LossWeights()
    history: list[dict[str, Any]] = []
    debug: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": EXPERIMENT_TYPE,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "accepted_pair_keys": [str(item["paired_key"]) for item in items],
        "excluded_pair_keys": [],
        "split_keys": {split: [str(item["paired_key"]) for item in rows] for split, rows in split_items.items()},
        "style_objective": {
            "style_layer": interface.selected_layer,
            "content_layer": interface.content_layer,
            "alignment": "phone_span_linear_interpolation",
            "min_style_frames": 1,
        },
        "proximity_telemetry": {
            "enabled": True,
            "uses_existing_encoder_outputs": True,
            "extra_encoder_calls": 0,
            "layer": interface.selected_layer,
            "mask_source": "independent_MFA_phone_spans",
            "descriptive_only": True,
        },
        "epochs": [],
    }
    for epoch in range(epochs):
        train_mean, train_rows = train_items(model, split_items["train"], optimizer=optimizer, weights=weights, device=torch.device(device), train=True)
        with torch.no_grad():
            with torch.no_grad():
                valid_mean, valid_rows = train_items(model, split_items["valid"], optimizer=optimizer, weights=weights, device=torch.device(device), train=False)
        row = {"epoch": epoch + 1, "train": train_mean, "valid": valid_mean, "train_hubert_proximity": summarize_hubert_proximity([item["hubert_proximity"] for item in train_rows]), "valid_hubert_proximity": summarize_hubert_proximity([item["hubert_proximity"] for item in valid_rows]), "parameter_norm": _parameter_norm(model), "train_items": train_rows, "valid_items": valid_rows}
        history.append(row)
        debug["epochs"].append(row)
        with (output / "epoch_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")
    checkpoint = make_checkpoint(model, optimizer, history=history, items=items, manifest_path=manifest_path, seed=seed, weights=weights)
    save_checkpoint(output / "checkpoint.pt", checkpoint)
    (output / "training_debug.json").write_text(json.dumps(_json_safe(debug), indent=2, allow_nan=False), encoding="utf-8")
    return {"output_dir": str(output), "history": history, "checkpoint": str(output / "checkpoint.pt")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="target_manifest.json from build_waveform_hubert_targets.py")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = train_from_manifest(args.manifest, args.output_dir, epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed, device=args.device)
    print(json.dumps({"output_dir": result["output_dir"], "checkpoint": result["checkpoint"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
