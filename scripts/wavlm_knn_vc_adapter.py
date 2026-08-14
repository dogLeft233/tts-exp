#!/usr/bin/env python3
"""Strict adapter around the frozen upstream kNN-VC encoder and vocoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

KNN_VC_REPOSITORY = "bshall/knn-vc"
KNN_VC_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"
SAMPLE_RATE = 16_000
FRAME_STRIDE_SAMPLES = 320
FEATURE_DIM = 1024
FEATURE_LAYER = 6


@dataclass(frozen=True)
class WavLMInterface:
    repository: str = KNN_VC_REPOSITORY
    revision: str = KNN_VC_REVISION
    sample_rate: int = SAMPLE_RATE
    frame_stride_samples: int = FRAME_STRIDE_SAMPLES
    selected_layer: int = FEATURE_LAYER
    feature_dim: int = FEATURE_DIM
    prematched_vocoder: bool = True
    frozen: bool = True
    loudness_normalization: bool = False
    wavlm_checkpoint_sha256: str | None = None
    vocoder_checkpoint_sha256: str | None = None


class WavLMKNNVCAdapter:
    def __init__(self, model: Any, *, device: str | torch.device = "cuda") -> None:
        self.model = model
        self.device = torch.device(device)
        self.interface = WavLMInterface(
            wavlm_checkpoint_sha256=self._checkpoint_sha256("WavLM-Large.pt"),
            vocoder_checkpoint_sha256=self._checkpoint_sha256("prematch_g_02500000.pt"),
        )
        self._validate_model()

    @staticmethod
    def _checkpoint_sha256(filename: str) -> str | None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def load_pretrained(
        cls,
        *,
        device: str = "cuda",
        source: str | Path = KNN_VC_REPOSITORY,
        revision: str = KNN_VC_REVISION,
    ) -> "WavLMKNNVCAdapter":
        source_value = str(source)
        if Path(source_value).exists():
            model = torch.hub.load(
                source_value, "knn_vc", source="local", prematched=True,
                pretrained=True, device=device,
            )
        else:
            if source_value != KNN_VC_REPOSITORY or revision != KNN_VC_REVISION:
                raise ValueError("remote loading is restricted to the pinned kNN-VC repository and revision")
            model = torch.hub.load(
                f"{source_value}:{revision}", "knn_vc", prematched=True,
                trust_repo=True, pretrained=True, device=device,
            )
        return cls(model, device=device)

    def _validate_model(self) -> None:
        sample_rate = int(getattr(self.model, "sr", -1))
        hop_length = int(getattr(self.model, "hop_length", -1))
        if sample_rate != SAMPLE_RATE or hop_length != FRAME_STRIDE_SAMPLES:
            raise ValueError(
                f"unexpected kNN-VC interface: sample_rate={sample_rate}, hop_length={hop_length}"
            )
        wavlm = getattr(self.model, "wavlm", None)
        hifigan = getattr(self.model, "hifigan", None)
        if wavlm is None or hifigan is None:
            raise ValueError("kNN-VC model must expose wavlm and hifigan modules")
        if getattr(wavlm, "training", True) or getattr(hifigan, "training", True):
            raise ValueError("WavLM and HiFi-GAN must be in eval mode")
        for module_name, module in (("wavlm", wavlm), ("hifigan", hifigan)):
            if any(parameter.requires_grad for parameter in module.parameters()):
                # Upstream sets eval but not requires_grad=False. Freeze explicitly.
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError(f"{module_name} parameters must be frozen")

    def metadata(self) -> dict[str, Any]:
        return asdict(self.interface)

    @torch.inference_mode()
    def extract(self, waveform: Tensor) -> Tensor:
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] < 1024:
            raise ValueError("waveform must have shape [1,N] with at least 1024 samples")
        if not torch.isfinite(waveform).all():
            raise FloatingPointError("waveform must be finite")
        features = self.model.get_features(waveform.to(self.device), vad_trigger_level=0)
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM or features.shape[0] == 0:
            raise ValueError(f"WavLM layer-{FEATURE_LAYER} must produce [T,{FEATURE_DIM}], got {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise FloatingPointError("WavLM features are non-finite")
        return features.detach()

    @torch.inference_mode()
    def vocode(self, features: Tensor) -> Tensor:
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM or features.shape[0] == 0:
            raise ValueError(f"vocoder features must have shape [T,{FEATURE_DIM}]")
        if not torch.isfinite(features).all():
            raise FloatingPointError("vocoder features must be finite")
        waveform = self.model.vocode(features.unsqueeze(0).to(self.device)).squeeze(0)
        waveform = torch.as_tensor(waveform, dtype=torch.float32).flatten().cpu()
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise FloatingPointError("vocoder returned empty or non-finite waveform")
        return waveform
