#!/usr/bin/env python3
"""Train the bounded Stage-1 raw-TTS HuBERT feature puller.

The trainer consumes only the strict train/valid pair-eligibility manifest.
Raw TTS layer-6 features are phone-locally aligned to natural frames for direct
feature supervision.  Weak waveform targets are intentionally absent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor, nn
from torch.optim import AdamW

from hubert_feature_alignment import align_tts_features_to_natural, masked_hubert_proximity
from prepare_two_stage_hubert_manifest import (
    ALLOWED_STAGE_SPLITS,
    STAGE_CONTRACT,
    file_sha256,
    load_two_stage_manifest,
)
from tts_aligned_feature_puller import (
    AlignedTTSFeaturePuller,
    DEFAULT_BOUND_QUANTILE,
    DEFAULT_CONTENT_LAYER,
    FeatureNormalizer,
    ResidualBound,
    Stage1LossWeights,
    compute_stage1_loss,
    delta_statistics,
    fit_feature_normalizer,
    fit_residual_bound,
    validate_hubert_interface,
)
from waveform_hubert_enhancer import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_LAYER,
    DEFAULT_MODEL_ID,
    TARGET_SAMPLE_RATE,
    EncoderInterface,
    _extract_hidden_states,
    _finite_tensor,
)

SCHEMA_VERSION = 3
EXPERIMENT_TYPE = "aligned_raw_tts_feature_puller"
CHECKPOINT_SELECTION_METRIC = "valid_pull_combined"
CHECKPOINT_SELECTION_POLICY = (
    "lowest finite valid pull_cosine plus pull_smooth_l1; "
    "ties select the earlier epoch"
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    bound_quantile: float = DEFAULT_BOUND_QUANTILE
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.bound_quantile) or not 0 < self.bound_quantile <= 1:
            raise ValueError("bound_quantile must be in (0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_stage1_wave(
    path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    allow_24k_resample: bool = False,
) -> Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1 or audio.size < 1024:
        raise ValueError(f"audio must be mono with >=1024 samples: {path}")
    if int(source_rate) != sample_rate:
        if not allow_24k_resample or int(source_rate) != 24_000:
            raise ValueError(f"audio must be {sample_rate} Hz: {path} (found {source_rate})")
        gcd = math.gcd(int(source_rate), sample_rate)
        audio = resample_poly(
            np.asarray(audio, dtype=np.float32),
            sample_rate // gcd,
            int(source_rate) // gcd,
        )
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).view(1, 1, -1)
    return _finite_tensor(waveform, f"audio {path}")


def _load_stage1_hubert(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, EncoderInterface]:
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
    stride = int(np.prod([int(value) for value in getattr(config, "conv_stride", [])]))
    layer_count = int(getattr(config, "num_hidden_layers", -1))
    if (
        hidden_size != 768
        or stride != DEFAULT_FRAME_STRIDE
        or DEFAULT_LAYER > layer_count
    ):
        raise ValueError(
            f"unexpected HuBERT interface: hidden_size={hidden_size}, "
            f"stride={stride}, layers={layer_count}"
        )
    interface = EncoderInterface(
        model_id=model_id,
        selected_layer=DEFAULT_LAYER,
        content_layer=DEFAULT_CONTENT_LAYER,
        hidden_size=hidden_size,
        frame_stride_samples=stride,
        sample_rate=TARGET_SAMPLE_RATE,
        frozen=True,
    )
    validate_hubert_interface(interface.as_dict())
    return encoder, interface


def _strict_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    return payload


def _finite_scalar(value: Tensor, name: str) -> float:
    if value.ndim != 0 or not torch.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite")
    return float(value.detach().cpu())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_hubert_layers(
    encoder: nn.Module,
    waveform: Tensor,
    *,
    layers: Sequence[int],
    interface: Mapping[str, Any],
) -> dict[int, Tensor]:
    """Extract requested frozen HuBERT layers in exactly one forward call."""
    contract = validate_hubert_interface(interface)
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape [B,1,N]")
    if not layers:
        raise ValueError("layers must be non-empty")
    requested = tuple(int(layer) for layer in layers)
    if len(set(requested)) != len(requested) or any(layer < 0 for layer in requested):
        raise ValueError("layers must contain distinct non-negative indices")
    _finite_tensor(waveform, "HuBERT input waveform")
    encoder.eval()
    with torch.no_grad():
        output = encoder(waveform[:, 0, :], output_hidden_states=True)
    hidden = _extract_hidden_states(output)
    result: dict[int, Tensor] = {}
    for layer in requested:
        if layer >= len(hidden):
            raise ValueError(f"HuBERT layer {layer} is unavailable")
        value = hidden[layer]
        if (
            value.ndim != 3
            or value.shape[0] != waveform.shape[0]
            or value.shape[-1] != int(contract["hidden_size"])
        ):
            raise ValueError("HuBERT feature shape does not match the declared interface")
        result[layer] = _finite_tensor(value.detach(), f"HuBERT layer {layer}")
    return result


def _split_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    seen_keys: set[str] = set()
    seen_identity: dict[str, str] = {}
    for raw in records:
        item = dict(raw)
        split = str(item.get("split") or "")
        key = str(item.get("paired_key") or "")
        if split not in ALLOWED_STAGE_SPLITS:
            raise ValueError(f"Stage 1 cannot consume split {split!r}")
        if not key or key in seen_keys:
            raise ValueError(f"duplicate or empty paired_key: {key!r}")
        seen_keys.add(key)
        for field in ("natural_path", "tts_path", "natural_sha256", "tts_sha256", "transcript_sha256"):
            value = str(item.get(field) or "")
            if not value:
                raise ValueError(f"{field} is required for {key}")
            identity = f"{field}:{value}"
            previous = seen_identity.setdefault(identity, split)
            if previous != split:
                raise ValueError(f"cross-split identity leakage: {field}")
        groups[split].append(item)
    if not groups["train"] or not groups["valid"]:
        raise ValueError("Stage 1 requires non-empty train and valid records")
    return groups


def load_stage1_records(manifest_path: str | Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load the strict manifest and reject any heldout entry before extraction."""
    manifest = Path(manifest_path).resolve()
    payload = _strict_json(manifest)
    if payload.get("contract") != STAGE_CONTRACT:
        raise ValueError("Stage 1 requires the direct raw-TTS train/valid contract")
    records = load_two_stage_manifest(manifest)
    return _split_records(records), payload


def _record_hashes(item: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path_key, hash_key in (
        ("natural_path", "natural_sha256"),
        ("tts_path", "tts_sha256"),
        ("matched_span_metadata_path", "matched_span_metadata_sha256"),
    ):
        path = Path(str(item.get(path_key) or ""))
        expected = str(item.get(hash_key) or "")
        if not path.is_file() or not expected:
            raise ValueError(f"missing Stage 1 provenance for {path_key}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{path_key} hash mismatch")
        result[hash_key] = actual
    return result


def prepare_record(
    item: Mapping[str, Any],
    *,
    encoder: nn.Module,
    interface: Mapping[str, Any],
    device: torch.device,
    read_wave: Callable[..., Tensor] = _read_stage1_wave,
) -> dict[str, Any]:
    """Create one direct raw-TTS-aligned feature item without weak targets."""
    split = str(item.get("split") or "")
    if split not in ALLOWED_STAGE_SPLITS:
        raise ValueError(f"Stage 1 cannot prepare split {split!r}")
    hashes = _record_hashes(item)
    natural = read_wave(item["natural_path"]).to(device)
    tts = read_wave(item["tts_path"], allow_24k_resample=True).to(device)
    natural_features = extract_hubert_layers(
        encoder,
        natural,
        layers=(DEFAULT_CONTENT_LAYER, DEFAULT_LAYER),
        interface=interface,
    )
    tts_features = extract_hubert_layers(
        encoder,
        tts,
        layers=(DEFAULT_LAYER,),
        interface=interface,
    )
    aligned_tts, target_mask, alignment = align_tts_features_to_natural(
        natural_features[DEFAULT_LAYER],
        tts_features[DEFAULT_LAYER],
        item["matched_spans"],
        frame_stride_samples=int(interface["frame_stride_samples"]),
        sample_rate=int(interface["sample_rate"]),
        min_frames=1,
    )
    if target_mask.ndim == 1:
        target_mask = target_mask.unsqueeze(0)
    if target_mask.dtype != torch.bool or not bool(target_mask.any()):
        raise ValueError("raw-TTS feature alignment produced no target frames")
    padding_mask = torch.ones(
        natural_features[DEFAULT_LAYER].shape[:2],
        dtype=torch.bool,
        device=device,
    )
    if tuple(aligned_tts.shape) != tuple(natural_features[DEFAULT_LAYER].shape):
        raise ValueError("aligned raw-TTS features do not match natural L6 shape")
    return {
        "paired_key": str(item["paired_key"]),
        "split": split,
        "speaker_group": str(item["speaker_group"]),
        "natural_layer2": natural_features[DEFAULT_CONTENT_LAYER],
        "natural_layer6": natural_features[DEFAULT_LAYER],
        "aligned_tts_layer6": aligned_tts.detach(),
        "padding_mask": padding_mask,
        "target_mask": target_mask.detach(),
        "alignment": alignment,
        "hashes": hashes,
        "encoder_calls": {"natural_l2_l6": 1, "raw_tts_l6": 1},
    }


def prepare_records(
    records: Sequence[Mapping[str, Any]],
    *,
    encoder: nn.Module,
    interface: Mapping[str, Any],
    device: torch.device,
    read_wave: Callable[..., Tensor] = _read_stage1_wave,
) -> list[dict[str, Any]]:
    return [
        prepare_record(
            item,
            encoder=encoder,
            interface=interface,
            device=device,
            read_wave=read_wave,
        )
        for item in records
    ]


def fit_stage1_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    bound_quantile: float = DEFAULT_BOUND_QUANTILE,
) -> tuple[FeatureNormalizer, ResidualBound]:
    if not records or any(str(item.get("split")) != "train" for item in records):
        raise ValueError("Stage 1 statistics require train records only")
    selected_l2 = torch.cat(
        [
            item["natural_layer2"].detach().cpu()[item["padding_mask"].detach().cpu()]
            for item in records
        ],
        dim=0,
    )
    selected_l6 = torch.cat(
        [
            item["natural_layer6"].detach().cpu()[item["padding_mask"].detach().cpu()]
            for item in records
        ],
        dim=0,
    )
    selected_target_natural = torch.cat(
        [
            item["natural_layer6"].detach().cpu()[item["target_mask"].detach().cpu()]
            for item in records
        ],
        dim=0,
    )
    selected_target_tts = torch.cat(
        [
            item["aligned_tts_layer6"].detach().cpu()[item["target_mask"].detach().cpu()]
            for item in records
        ],
        dim=0,
    )
    normalizer_mask = torch.ones(
        (1, selected_l2.shape[0]), dtype=torch.bool
    )
    target_mask = torch.ones(
        (1, selected_target_natural.shape[0]), dtype=torch.bool
    )
    normalizer = fit_feature_normalizer(
        selected_l2.unsqueeze(0),
        selected_l6.unsqueeze(0),
        normalizer_mask,
    )
    bound = fit_residual_bound(
        selected_target_natural.unsqueeze(0),
        selected_target_tts.unsqueeze(0),
        target_mask,
        normalizer,
        quantile=bound_quantile,
    )
    return normalizer, bound


def _proximity(
    natural: Tensor,
    goal: Tensor,
    target: Tensor,
    mask: Tensor,
    interface: Mapping[str, Any],
) -> dict[str, Any]:
    return masked_hubert_proximity(
        natural.detach(),
        goal.detach(),
        target.detach(),
        mask.detach(),
        layer=int(interface["selected_layer"]),
        encoder_model=str(interface["model_id"]),
        min_frames=1,
    )


def run_epoch(
    model: AlignedTTSFeaturePuller,
    records: Sequence[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer | None,
    weights: Stage1LossWeights,
    interface: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in (*weights.as_dict(), "total") if name != "smooth_l1_beta"}
    rows: list[dict[str, Any]] = []
    for item in records:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            delta = model(item["natural_layer2"], item["natural_layer6"], item["padding_mask"])
            loss, terms, diagnostics = compute_stage1_loss(
                item["natural_layer6"],
                delta,
                item["aligned_tts_layer6"],
                item["padding_mask"],
                item["target_mask"],
                model.normalizer,
                weights=weights,
            )
            if training:
                loss.backward()
                gradients = [
                    parameter.grad.detach().float().norm()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if not gradients:
                    raise FloatingPointError("Stage 1 received no gradients")
                gradient_norm = float(torch.stack(gradients).norm().cpu())
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("Stage 1 gradient norm is non-finite")
                optimizer.step()
            else:
                gradient_norm = 0.0
        goal = model.goal_layer6(item["natural_layer6"], delta)
        delta_info = delta_statistics(delta, item["padding_mask"])
        if delta_info["max"] > model.residual_bound.value + 1e-5:
            raise FloatingPointError("Stage 1 delta hard bound invariant failed")
        if torch.count_nonzero(delta[~item["padding_mask"]]):
            raise FloatingPointError("Stage 1 padding invariant failed")
        proximity = _proximity(
            item["natural_layer6"],
            goal,
            item["aligned_tts_layer6"],
            item["target_mask"],
            interface,
        )
        term_values = {name: _finite_scalar(value, f"loss term {name}") for name, value in terms.items()}
        for name, value in term_values.items():
            totals[name] += value
        rows.append(
            {
                "paired_key": item["paired_key"],
                "split": item["split"],
                "losses": term_values,
                "alignment": item["alignment"],
                "loss_diagnostics": diagnostics,
                "delta": delta_info,
                "proximity": proximity,
                "gradient_norm": gradient_norm,
                "encoder_calls": dict(item["encoder_calls"]),
                "hashes": dict(item["hashes"]),
            }
        )
    if not rows:
        raise ValueError("epoch requires at least one record")
    means = {name: value / len(rows) for name, value in totals.items()}
    means[CHECKPOINT_SELECTION_METRIC] = means["pull_cosine"] + means["pull_smooth_l1"]
    return means, rows


def _mean_validation_proximity(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("validation proximity requires at least one row")
    natural = [
        float(row["proximity"]["natural_to_tts"]["combined"])
        for row in rows
    ]
    goal = [
        float(row["proximity"]["enhanced_to_tts"]["combined"])
        for row in rows
    ]
    if not all(math.isfinite(value) for value in (*natural, *goal)):
        raise FloatingPointError("validation proximity is non-finite")
    mean_natural = float(sum(natural) / len(natural))
    mean_goal = float(sum(goal) / len(goal))
    return {
        "item_count": len(rows),
        "natural_to_raw_tts_combined": mean_natural,
        "goal_to_raw_tts_combined": mean_goal,
        "absolute_improvement": mean_natural - mean_goal,
        "relative_improvement": (mean_natural - mean_goal) / max(mean_natural, 1e-12),
        "goal_closer_item_count": sum(
            bool(row["proximity"]["toward_tts"]["enhanced_closer_than_natural"])
            for row in rows
        ),
    }


def _trainable_state_dict(model: AlignedTTSFeaturePuller) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    }


def _split_identity(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            "paired_keys": [str(item["paired_key"]) for item in rows],
            "source_hashes": {str(item["paired_key"]): dict(item["hashes"]) for item in rows},
        }
        for split, rows in records.items()
    }


def make_checkpoint(
    model: AlignedTTSFeaturePuller,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_path: str | Path,
    manifest_payload: Mapping[str, Any],
    interface: Mapping[str, Any],
    weights: Stage1LossWeights,
    config: TrainConfig,
    valid_proximity: Mapping[str, Any],
) -> dict[str, Any]:
    checked_interface = validate_hubert_interface(interface)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    state = _trainable_state_dict(model)
    if not state or any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Stage 1 checkpoint has missing or non-finite trainable state")
    manifest = Path(manifest_path).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": EXPERIMENT_TYPE,
        "epoch": int(epoch),
        "state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.model_config(),
        "encoder_interface": checked_interface,
        "loss_weights": weights.as_dict(),
        "train_config": config.as_dict(),
        "source_target_manifest": dict(manifest_payload.get("source_target_manifest", {})),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "split_identity": _split_identity(prepared),
        "selection": {
            "metric": CHECKPOINT_SELECTION_METRIC,
            "policy": CHECKPOINT_SELECTION_POLICY,
            "eligible_splits": ["valid"],
            "heldout_excluded": True,
        },
        "valid_raw_tts_proximity": dict(valid_proximity),
        "history": [dict(row) for row in history],
    }


def validate_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Stage 1 checkpoint schema")
    if payload.get("experiment_type") != EXPERIMENT_TYPE:
        raise ValueError("checkpoint is not an aligned raw-TTS feature puller")
    interface = validate_hubert_interface(payload.get("encoder_interface", {}))
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint model_config is required")
    if (
        model_config.get("class") != "AlignedTTSFeaturePuller"
        or model_config.get("input_dim") != 768
        or model_config.get("input_layers") != [DEFAULT_CONTENT_LAYER, DEFAULT_LAYER]
        or model_config.get("target_layer") != DEFAULT_LAYER
        or model_config.get("residual_space") != "normalized_hubert_layer6"
        or model_config.get("loss_reduction") != "per_frame_feature_mean"
    ):
        raise ValueError("checkpoint model contract is incompatible")
    normalizer = FeatureNormalizer.from_dict(model_config.get("normalizer", {}))
    bound = ResidualBound.from_dict(model_config.get("hard_bound", {}))
    hidden_channels = int(model_config.get("hidden_channels", 0))
    dilations = model_config.get("dilations")
    if hidden_channels <= 0 or not isinstance(dilations, list):
        raise ValueError("checkpoint model structure is invalid")
    reconstructed = AlignedTTSFeaturePuller(
        normalizer,
        bound,
        hidden_channels=hidden_channels,
        dilations=tuple(int(value) for value in dilations),
    )
    expected_state = {
        name: value
        for name, value in reconstructed.state_dict().items()
        if name in {name for name, parameter in reconstructed.named_parameters() if parameter.requires_grad}
    }
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or set(state) != set(expected_state):
        raise ValueError("checkpoint trainable state_dict keys are incompatible")
    for name, value in state.items():
        if (
            not isinstance(value, Tensor)
            or tuple(value.shape) != tuple(expected_state[name].shape)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("checkpoint state_dict contains invalid values")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping) or selection.get("metric") != CHECKPOINT_SELECTION_METRIC:
        raise ValueError("checkpoint selection policy is incompatible")
    proximity = payload.get("valid_raw_tts_proximity")
    required_proximity = {
        "item_count",
        "natural_to_raw_tts_combined",
        "goal_to_raw_tts_combined",
        "absolute_improvement",
        "relative_improvement",
        "goal_closer_item_count",
    }
    if not isinstance(proximity, Mapping) or required_proximity.difference(proximity):
        raise ValueError("checkpoint requires valid raw-TTS proximity diagnostics")
    if any(
        not math.isfinite(float(proximity[key]))
        for key in required_proximity.difference({"item_count", "goal_closer_item_count"})
    ):
        raise ValueError("checkpoint valid raw-TTS proximity is non-finite")
    split_identity = payload.get("split_identity")
    if not isinstance(split_identity, Mapping) or set(split_identity) != {"train", "valid"}:
        raise ValueError("checkpoint must contain only train and valid identities")
    return {
        "encoder_interface": interface,
        "normalizer": normalizer,
        "residual_bound": bound,
        "model_config": dict(model_config),
    }


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be an object")
    validate_checkpoint(payload)
    return dict(payload)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    validate_checkpoint(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)


def train_stage1(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: TrainConfig = TrainConfig(),
    weights: Stage1LossWeights = Stage1LossWeights(),
    encoder: nn.Module | None = None,
    interface: EncoderInterface | Mapping[str, Any] | None = None,
    read_wave: Callable[..., Tensor] = _read_stage1_wave,
) -> dict[str, Any]:
    """Run deterministic Stage-1 train/valid training without heldout access."""
    _set_seed(config.seed)
    device = torch.device(config.device)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Stage 1 output directory must be fresh: {output}")
    grouped, manifest_payload = load_stage1_records(manifest_path)
    if encoder is None:
        encoder, loaded_interface = _load_stage1_hubert(device=device)
        interface_payload: Mapping[str, Any] = loaded_interface.as_dict()
    else:
        if interface is None:
            raise ValueError("an explicit HuBERT interface is required with an injected encoder")
        interface_payload = interface.as_dict() if isinstance(interface, EncoderInterface) else interface
        encoder.to(device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    interface_payload = validate_hubert_interface(interface_payload)
    prepared = {
        split: prepare_records(
            rows,
            encoder=encoder,
            interface=interface_payload,
            device=device,
            read_wave=read_wave,
        )
        for split, rows in grouped.items()
    }
    normalizer, bound = fit_stage1_statistics(prepared["train"], bound_quantile=config.bound_quantile)
    model = AlignedTTSFeaturePuller(normalizer, bound).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    for epoch in range(config.epochs):
        train_metrics, train_rows = run_epoch(
            model,
            prepared["train"],
            optimizer=optimizer,
            weights=weights,
            interface=interface_payload,
        )
        with torch.no_grad():
            valid_metrics, valid_rows = run_epoch(
                model,
                prepared["valid"],
                optimizer=None,
                weights=weights,
                interface=interface_payload,
            )
        if not math.isfinite(valid_metrics[CHECKPOINT_SELECTION_METRIC]):
            raise FloatingPointError("valid pull metric is non-finite")
        valid_proximity = _mean_validation_proximity(valid_rows)
        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
            "valid_raw_tts_proximity": valid_proximity,
            "train_diagnostics": train_rows,
            "valid_diagnostics": valid_rows,
        }
        history.append(entry)
        checkpoint = make_checkpoint(
            model,
            optimizer,
            epoch=epoch,
            history=history,
            prepared=prepared,
            manifest_path=manifest_path,
            manifest_payload=manifest_payload,
            interface=interface_payload,
            weights=weights,
            config=config,
            valid_proximity=valid_proximity,
        )
        save_checkpoint(output / "last.pt", checkpoint)
        metric = valid_metrics[CHECKPOINT_SELECTION_METRIC]
        if best_metric is None or metric < best_metric:
            save_checkpoint(output / "best.pt", checkpoint)
            best_metric = metric
    return {
        "output_dir": str(output.resolve()),
        "best_metric": best_metric,
        "epochs": config.epochs,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--bound-quantile", type=float, default=DEFAULT_BOUND_QUANTILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train_stage1(
        args.manifest,
        args.output_dir,
        config=TrainConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bound_quantile=args.bound_quantile,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


__all__ = [
    "CHECKPOINT_SELECTION_METRIC",
    "CHECKPOINT_SELECTION_POLICY",
    "EXPERIMENT_TYPE",
    "SCHEMA_VERSION",
    "TrainConfig",
    "extract_hubert_layers",
    "fit_stage1_statistics",
    "load_checkpoint",
    "load_stage1_records",
    "make_checkpoint",
    "prepare_record",
    "prepare_records",
    "run_epoch",
    "save_checkpoint",
    "train_stage1",
    "validate_checkpoint",
]


if __name__ == "__main__":
    main()
