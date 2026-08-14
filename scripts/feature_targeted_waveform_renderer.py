#!/usr/bin/env python3
"""Frozen Stage-1-conditioned residual waveform renderer.

The renderer accepts only natural audio at inference.  It obtains frozen local
HuBERT layers 2 and 6 once, uses a verified frozen Stage-1 puller to create a
layer-6 target, and renders a bounded waveform residual.  Its output remains a
waveform; Ditto must re-encode that waveform through its own native frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train_tts_aligned_feature_puller import load_checkpoint as load_stage1_checkpoint
from tts_aligned_feature_puller import (
    AlignedTTSFeaturePuller,
    FeatureNormalizer,
    ResidualBound,
    validate_hubert_interface,
)
from waveform_hubert_enhancer import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LAYER,
    TARGET_SAMPLE_RATE,
    ConditionedResidualTCN,
    EncoderInterface,
    _extract_hidden_states,
    _finite_tensor,
)

DEFAULT_STAGE2_CONDITIONING_DIM = 32
DEFAULT_STAGE2_CHANNELS = 64
DEFAULT_STAGE2_DILATIONS = (1, 2, 4, 8, 16, 32)
DEFAULT_STAGE2_RESIDUAL_SCALE = 0.05
_EXPECTED_NORMALIZER_BUFFERS = frozenset({
    "layer2_mean",
    "layer2_std",
    "layer6_mean",
    "layer6_std",
})


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_waveform(waveform: Tensor, name: str) -> None:
    if waveform.ndim != 3 or waveform.shape[1] != 1 or waveform.shape[-1] < DEFAULT_FRAME_STRIDE:
        raise ValueError(f"{name} must have shape [B,1,N] with N >= {DEFAULT_FRAME_STRIDE}")
    _finite_tensor(waveform, name)


def _require_features(value: Tensor, reference: Tensor, name: str) -> None:
    if value.ndim != 3 or value.shape != reference.shape or value.shape[-1] != DEFAULT_HIDDEN_SIZE:
        raise ValueError(f"{name} must match natural layer-6 shape [B,F,{DEFAULT_HIDDEN_SIZE}]")
    _finite_tensor(value, name)


def _require_padding_mask(mask: Tensor, features: Tensor) -> None:
    if mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError("padding_mask must have shape [B,F] matching HuBERT features")
    if mask.dtype != torch.bool:
        raise TypeError("padding_mask must have dtype torch.bool")
    if not bool(mask.any()):
        raise ValueError("padding_mask must contain at least one real frame")


@dataclass(frozen=True)
class FrozenStage1Artifact:
    """Verified provenance for a Stage-1 checkpoint frozen by Stage 2."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    encoder_interface: dict[str, Any]
    model_config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_epoch": self.checkpoint_epoch,
            "encoder_interface": dict(self.encoder_interface),
            "model_config": dict(self.model_config),
            "frozen": True,
            "required_filename": "best.pt",
        }


def load_frozen_stage1(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[AlignedTTSFeaturePuller, FrozenStage1Artifact]:
    """Load only a validated Stage-1 best checkpoint and freeze it."""
    path = Path(checkpoint_path).resolve()
    if path.name != "best.pt":
        raise ValueError("Stage 2 requires the selected Stage-1 best.pt checkpoint")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = load_stage1_checkpoint(path, map_location=map_location)
    model_config = payload["model_config"]
    if (
        model_config.get("residual_space") != "normalized_hubert_layer6"
        or model_config.get("loss_reduction") != "per_frame_feature_mean"
    ):
        raise ValueError("Stage 1 checkpoint lacks the corrected normalized residual contract")
    normalizer = FeatureNormalizer.from_dict(model_config["normalizer"])
    residual_bound = ResidualBound.from_dict(model_config["hard_bound"])
    model = AlignedTTSFeaturePuller(
        normalizer,
        residual_bound,
        hidden_channels=int(model_config["hidden_channels"]),
        dilations=tuple(int(value) for value in model_config["dilations"]),
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    if (
        set(incompatible.missing_keys) != _EXPECTED_NORMALIZER_BUFFERS
        or incompatible.unexpected_keys
    ):
        raise ValueError("Stage 1 state does not match its serialized normalizer")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    artifact = FrozenStage1Artifact(
        checkpoint_path=str(path),
        checkpoint_sha256=_file_sha256(path),
        checkpoint_epoch=int(payload["epoch"]),
        encoder_interface=validate_hubert_interface(payload["encoder_interface"]),
        model_config=dict(model_config),
    )
    return model, artifact


class FeatureTargetedWaveformRenderer(nn.Module):
    """Residual waveform renderer conditioned on frozen Stage-1 feature goals."""

    def __init__(
        self,
        encoder: nn.Module,
        stage1: AlignedTTSFeaturePuller,
        stage1_artifact: FrozenStage1Artifact,
        *,
        interface: EncoderInterface | Mapping[str, Any] = EncoderInterface(),
        conditioning_dim: int = DEFAULT_STAGE2_CONDITIONING_DIM,
        channels: int = DEFAULT_STAGE2_CHANNELS,
        dilations: Sequence[int] = DEFAULT_STAGE2_DILATIONS,
        residual_scale: float = DEFAULT_STAGE2_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        contract = interface.as_dict() if isinstance(interface, EncoderInterface) else dict(interface)
        self.interface = validate_hubert_interface(contract)
        if self.interface != stage1_artifact.encoder_interface:
            raise ValueError("Stage 1 and Stage 2 HuBERT interfaces must match exactly")
        if conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be positive")
        self.encoder = encoder
        self.stage1 = stage1
        self.stage1_artifact = stage1_artifact
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.stage1.eval()
        self.conditioning_dim = int(conditioning_dim)
        self.layer2_projection = nn.Linear(DEFAULT_HIDDEN_SIZE, self.conditioning_dim)
        self.layer6_projection = nn.Linear(DEFAULT_HIDDEN_SIZE, self.conditioning_dim)
        self.delta_projection = nn.Linear(DEFAULT_HIDDEN_SIZE, self.conditioning_dim)
        self.waveform_model = ConditionedResidualTCN(
            conditioning_dim=3 * self.conditioning_dim,
            channels=channels,
            dilations=dilations,
            residual_scale=residual_scale,
        )

    def train(self, mode: bool = True) -> "FeatureTargetedWaveformRenderer":
        super().train(mode)
        self.encoder.eval()
        self.stage1.eval()
        return self

    def encode_layers(
        self,
        waveform: Tensor,
        layers: Sequence[int],
        *,
        retain_gradient: bool,
    ) -> dict[int, Tensor]:
        """Extract multiple HuBERT layers in one call with explicit input gradients."""
        _require_waveform(waveform, "HuBERT input waveform")
        requested = tuple(int(layer) for layer in layers)
        if not requested or len(set(requested)) != len(requested) or any(layer < 0 for layer in requested):
            raise ValueError("layers must contain distinct non-negative values")
        self.encoder.eval()
        signal = waveform[:, 0, :]
        if retain_gradient:
            output = self.encoder(signal, output_hidden_states=True)
        else:
            with torch.no_grad():
                output = self.encoder(signal, output_hidden_states=True)
        hidden = _extract_hidden_states(output)
        result: dict[int, Tensor] = {}
        for layer in requested:
            if layer >= len(hidden):
                raise ValueError(f"HuBERT layer {layer} is unavailable")
            features = hidden[layer]
            if (
                features.ndim != 3
                or features.shape[0] != waveform.shape[0]
                or features.shape[-1] != DEFAULT_HIDDEN_SIZE
            ):
                raise ValueError("HuBERT hidden state must have shape [B,F,768]")
            result[layer] = _finite_tensor(features, f"HuBERT layer {layer} features")
        return result

    def natural_features(self, waveform: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Obtain natural L2/L6 once and form a detached frozen Stage-1 goal."""
        layers = self.encode_layers(
            waveform,
            (int(self.interface["content_layer"]), int(self.interface["selected_layer"])),
            retain_gradient=False,
        )
        natural_layer2 = layers[int(self.interface["content_layer"])]
        natural_layer6 = layers[int(self.interface["selected_layer"])]
        padding_mask = torch.ones(
            natural_layer6.shape[:2], dtype=torch.bool, device=natural_layer6.device
        )
        with torch.no_grad():
            delta_layer6 = self.stage1(natural_layer2, natural_layer6, padding_mask)
        return natural_layer2.detach(), natural_layer6.detach(), delta_layer6.detach()

    def _frame_to_sample_gate(self, padding_mask: Tensor, samples: int, dtype: torch.dtype) -> Tensor:
        gate = F.interpolate(
            padding_mask.to(dtype=dtype).unsqueeze(1),
            size=samples,
            mode="nearest",
        )
        return _finite_tensor(gate, "frame-to-sample validity gate")

    def _condition_stream(self, projection: nn.Linear, features: Tensor, samples: int) -> Tensor:
        projected = projection(features).transpose(1, 2)
        return F.interpolate(projected, size=samples, mode="linear", align_corners=False)

    def conditioning(
        self,
        natural_layer2: Tensor,
        natural_layer6: Tensor,
        delta_layer6: Tensor,
        padding_mask: Tensor,
        samples: int,
    ) -> Tensor:
        """Fuse separately projected L2/L6/delta streams without padding leakage."""
        _require_features(natural_layer2, natural_layer6, "natural_layer2")
        _require_features(delta_layer6, natural_layer6, "delta_layer6")
        _require_padding_mask(padding_mask, natural_layer6)
        if samples < DEFAULT_FRAME_STRIDE:
            raise ValueError("samples must provide at least one HuBERT frame")
        gate = self._frame_to_sample_gate(padding_mask, samples, natural_layer6.dtype)
        streams = (
            self._condition_stream(self.layer2_projection, natural_layer2.detach(), samples),
            self._condition_stream(self.layer6_projection, natural_layer6.detach(), samples),
            self._condition_stream(self.delta_projection, delta_layer6.detach(), samples),
        )
        conditioning = torch.cat(streams, dim=1) * gate
        return _finite_tensor(conditioning, "frozen Stage-1 waveform conditioning")

    def forward(
        self,
        natural_waveform: Tensor,
        *,
        natural_layer2: Tensor | None = None,
        natural_layer6: Tensor | None = None,
        delta_layer6: Tensor | None = None,
        padding_mask: Tensor | None = None,
        compute_enhanced_features: bool = True,
    ) -> dict[str, Tensor]:
        _require_waveform(natural_waveform, "natural waveform")
        supplied = (natural_layer2, natural_layer6, delta_layer6, padding_mask)
        if any(value is not None for value in supplied) and any(value is None for value in supplied):
            raise ValueError("natural L2/L6, Stage-1 delta, and padding_mask must be supplied together")
        if natural_layer2 is None:
            natural_layer2, natural_layer6, delta_layer6 = self.natural_features(natural_waveform)
            padding_mask = torch.ones(
                natural_layer6.shape[:2], dtype=torch.bool, device=natural_layer6.device
            )
        assert natural_layer6 is not None and delta_layer6 is not None and padding_mask is not None
        _require_features(natural_layer2, natural_layer6, "natural_layer2")
        _require_features(delta_layer6, natural_layer6, "delta_layer6")
        _require_padding_mask(padding_mask, natural_layer6)
        if torch.any(torch.linalg.vector_norm(delta_layer6, dim=-1) > self.stage1.residual_bound.value + 1e-5):
            raise ValueError("supplied Stage-1 delta exceeds its serialized hard bound")
        conditioning = self.conditioning(
            natural_layer2,
            natural_layer6,
            delta_layer6,
            padding_mask,
            natural_waveform.shape[-1],
        )
        enhanced = self.waveform_model(natural_waveform, conditioning)
        result: dict[str, Tensor] = {
            "waveform": enhanced,
            "residual": enhanced - natural_waveform,
            "natural_layer2": natural_layer2.detach(),
            "natural_layer6": natural_layer6.detach(),
            "stage1_delta_layer6": delta_layer6.detach(),
            "stage1_goal_layer6": self.stage1.goal_layer6(natural_layer6, delta_layer6).detach(),
            "padding_mask": padding_mask.detach(),
            "conditioning": conditioning,
        }
        if compute_enhanced_features:
            enhanced_layers = self.encode_layers(
                enhanced,
                (int(self.interface["content_layer"]), int(self.interface["selected_layer"])),
                retain_gradient=True,
            )
            result["enhanced_layer2"] = enhanced_layers[int(self.interface["content_layer"])]
            result["enhanced_layer6"] = enhanced_layers[int(self.interface["selected_layer"])]
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def model_config(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "stage1_conditioning": self.stage1_artifact.as_dict(),
            "condition_streams": {
                "natural_layer2": {"input_dim": DEFAULT_HIDDEN_SIZE, "output_dim": self.conditioning_dim},
                "natural_layer6": {"input_dim": DEFAULT_HIDDEN_SIZE, "output_dim": self.conditioning_dim},
                "stage1_delta_layer6": {"input_dim": DEFAULT_HIDDEN_SIZE, "output_dim": self.conditioning_dim},
                "frame_to_sample_gate": "nearest_masked_no_padding_smear",
            },
            "waveform_model": {
                "class": self.waveform_model.__class__.__name__,
                "conditioning_dim": self.waveform_model.conditioning_dim,
                "channels": self.waveform_model.channels,
                "dilations": list(self.waveform_model.dilations),
                "residual_scale": self.waveform_model.residual_scale,
            },
            "encoder_calls": {
                "natural_l2_l6_and_stage1": "one_frozen_no_grad_call",
                "enhanced_l2_l6": "one_frozen_input_gradient_call",
            },
            "parameter_count": self.parameter_count,
        }


@dataclass
class FeatureTargetedWaveformInference:
    """Exact-length NumPy inference wrapper for a verified Stage-1/Stage-2 pair."""

    model: FeatureTargetedWaveformRenderer
    chunk_seconds: float = 4.0
    stride_seconds: float = 2.0
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        self.chunk_seconds = float(self.chunk_seconds)
        self.stride_seconds = float(self.stride_seconds)
        if self.chunk_seconds < 0.064 or self.stride_seconds < 0.064:
            raise ValueError("chunk_seconds and stride_seconds must provide HuBERT context")
        if self.stride_seconds > self.chunk_seconds:
            raise ValueError("stride_seconds cannot exceed chunk_seconds")
        self.device = torch.device(self.device)
        self.model.to(self.device).eval()

    @property
    def chunk_samples(self) -> int:
        return max(DEFAULT_FRAME_STRIDE, int(round(self.chunk_seconds * TARGET_SAMPLE_RATE)))

    @property
    def stride_samples(self) -> int:
        return max(DEFAULT_FRAME_STRIDE, int(round(self.stride_seconds * TARGET_SAMPLE_RATE)))

    def enhance(self, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        values = np.asarray(audio)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {TARGET_SAMPLE_RATE}")
        if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.number):
            raise ValueError("audio must be a non-empty numeric mono vector")
        if not np.isfinite(values).all():
            raise ValueError("audio must be finite")
        waveform = torch.from_numpy(values.astype(np.float32, copy=False)).to(self.device)
        with torch.no_grad():
            enhanced = self._enhance_tensor(waveform)
        result = enhanced.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != values.shape or not np.isfinite(result).all():
            raise FloatingPointError("renderer returned invalid waveform output")
        return result

    def _enhance_tensor(self, waveform: Tensor) -> Tensor:
        from waveform_hubert_enhancer import _reflect_pad

        samples = waveform.numel()
        chunk = self.chunk_samples
        if samples <= chunk:
            padded = _reflect_pad(waveform, chunk)
            return self.model(padded[None, None, :], compute_enhanced_features=False)["waveform"][0, 0, :samples]
        stride = self.stride_samples
        residual = torch.zeros_like(waveform)
        weights = torch.zeros_like(waveform)
        window = torch.hann_window(chunk, periodic=False, device=waveform.device).clamp_min(1e-3)
        starts = list(range(0, samples - chunk + 1, stride))
        final_start = samples - chunk
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            segment = waveform[start : start + chunk]
            enhanced = self.model(segment[None, None, :], compute_enhanced_features=False)["waveform"][0, 0]
            residual[start : start + chunk] += (enhanced - segment) * window
            weights[start : start + chunk] += window
        return waveform + residual / weights.clamp_min(1e-6)


__all__ = [
    "DEFAULT_STAGE2_CHANNELS",
    "DEFAULT_STAGE2_CONDITIONING_DIM",
    "DEFAULT_STAGE2_DILATIONS",
    "DEFAULT_STAGE2_RESIDUAL_SCALE",
    "FeatureTargetedWaveformInference",
    "FeatureTargetedWaveformRenderer",
    "FrozenStage1Artifact",
    "load_frozen_stage1",
]
