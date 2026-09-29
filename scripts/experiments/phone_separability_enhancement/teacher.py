"""Frozen HuBERT/XLSR teachers with a differentiable processor-equivalent path."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.lrs3_phone_rules_metrics import derive_frame_times, normalize_phone
from scripts.experiments.lrs3_phone_rules_worker import frontend_config_from_model, processor_fingerprint

from .metrics import normalize_vector


def _torch():
    import torch

    return torch


def normalize_waveform(waveform: Any, *, epsilon: float = 1e-7) -> Any:
    torch = _torch()
    value = waveform
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError("waveform must have shape [B, N] or [N]")
    mean = value.mean(dim=-1, keepdim=True)
    variance = (value - mean).square().mean(dim=-1, keepdim=True)
    return (value - mean) / torch.sqrt(variance + float(epsilon))


@contextlib.contextmanager
def _proxy_environment(proxy: str | None):
    if not proxy:
        yield
        return
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = str(proxy)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass
class FrozenPhoneTeacher:
    key: str
    model: Any
    processor: Any
    device: str
    revision: str | None
    frontend: dict[str, Any]
    processor_info: dict[str, Any]

    def __post_init__(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size"))

    def encode(self, waveform: Any, *, layer: int) -> tuple[Any, Any, dict[str, Any]]:
        torch = _torch()
        value = waveform
        if value.ndim == 3:
            if value.shape[1] != 1:
                raise ValueError("waveform channel dimension must be one")
            value = value[:, 0, :]
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError("waveform must have shape [B, N]")
        if value.device.type != torch.device(self.device).type or (value.device.type == "cuda" and value.device.index != torch.device(self.device).index):
            value = value.to(self.device)
        inputs = normalize_waveform(value)
        outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or int(layer) >= len(hidden_states):
            raise ValueError(f"teacher has no hidden layer {layer}")
        hidden = hidden_states[int(layer)]
        frontend = self.frontend
        frame_times = derive_frame_times(
            int(hidden.shape[1]),
            16_000,
            frontend["conv_kernel"],
            frontend["conv_stride"],
            frontend.get("conv_dilation"),
            frontend.get("conv_padding"),
        )
        return hidden, torch.as_tensor(frame_times, device=hidden.device, dtype=hidden.dtype), {"layer": int(layer), "processor": self.processor_info, "frontend": frontend, "normalized": True}


def load_frozen_teacher(model_cfg: Mapping[str, Any], *, device: str, proxy: str | None = None, allow_download: bool = False) -> FrozenPhoneTeacher:
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch and transformers are required") from exc
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    name = str(model_cfg["model_name"])
    processor_name = str(model_cfg.get("processor_name", name))
    revision = model_cfg.get("revision")
    kwargs: dict[str, Any] = {"trust_remote_code": False, "local_files_only": not bool(allow_download)}
    if revision:
        kwargs["revision"] = str(revision)
    with _proxy_environment(proxy):
        try:
            processor = AutoProcessor.from_pretrained(processor_name, **kwargs)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            processor = AutoFeatureExtractor.from_pretrained(processor_name, **kwargs)
        model = AutoModel.from_pretrained(name, output_hidden_states=True, **kwargs)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return FrozenPhoneTeacher(
        key=str(model_cfg.get("key", name)),
        model=model,
        processor=processor,
        device=str(device),
        revision=str(revision) if revision is not None else None,
        frontend=frontend_config_from_model(model),
        processor_info=processor_fingerprint(processor),
    )


def processor_parity(teacher: FrozenPhoneTeacher, waveform: np.ndarray, *, sample_rate: int = 16_000) -> dict[str, Any]:
    torch = _torch()
    value = np.asarray(waveform, dtype=np.float32).reshape(-1)
    processor_output = teacher.processor(value, sampling_rate=sample_rate, return_tensors="pt", padding=False)
    expected = processor_output["input_values"]
    manual = normalize_waveform(torch.as_tensor(value, dtype=torch.float32)).detach().cpu().numpy()
    observed = expected.detach().cpu().numpy()
    if observed.ndim == 2:
        observed = observed[0]
    diff = np.asarray(observed, dtype=np.float64) - np.asarray(manual[0], dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(diff))), "l2": float(np.linalg.norm(diff)), "cosine": float(np.dot(observed, manual[0]) / max(np.linalg.norm(observed) * np.linalg.norm(manual[0]), 1e-12)), "pass": bool(np.max(np.abs(diff)) <= 1e-4 and 1.0 - float(np.dot(observed, manual[0]) / max(np.linalg.norm(observed) * np.linalg.norm(manual[0]), 1e-12)) <= 1e-5), "processor": teacher.processor_info}


def pool_hidden(hidden: Any, frame_times: Any, tokens: Sequence[Mapping[str, Any]], *, view: str = "core") -> dict[str, Any]:
    torch = _torch()
    if hidden.ndim == 3:
        hidden = hidden[0]
    if frame_times.ndim != 1 or hidden.shape[0] != frame_times.numel():
        raise ValueError("hidden/frame_times shape mismatch")
    result: dict[str, Any] = {}
    for index, token in enumerate(tokens):
        if not bool(token.get("speech", not token.get("silence", False))):
            continue
        label = normalize_phone(token.get("label", ""))
        start = float(token["start_s"])
        end = float(token["end_s"])
        indices = torch.nonzero((frame_times >= start) & (frame_times < end), as_tuple=False).reshape(-1)
        if view == "core":
            duration = end - start
            indices = indices[(frame_times[indices] >= start + 0.20 * duration) & (frame_times[indices] < end - 0.20 * duration)]
        elif view == "matched_1frame" and indices.numel():
            middle = torch.argmin((frame_times[indices] - (start + end) / 2.0).abs())
            indices = indices[middle:middle + 1]
        if indices.numel() == 0:
            continue
        pooled = hidden[indices].mean(dim=0)
        pooled = pooled / torch.linalg.vector_norm(pooled).clamp_min(1e-8)
        key = str(token.get("token_id", f":{index}"))
        result[key] = {"token_id": key, "token_index": int(index), "label": label, "start_s": start, "end_s": end, "embedding": pooled, "valid": True}
    return result


def phone_margins_from_hidden(hidden: Any, frame_times: Any, tokens: Sequence[Mapping[str, Any]], centroids: Mapping[str, Sequence[float]], *, view: str = "core", label_permutation: Mapping[str, str] | None = None, target_margin: float = 0.05, temperature: float = 0.1) -> tuple[Any, list[dict[str, Any]]]:
    torch = _torch()
    pooled = pool_hidden(hidden, frame_times, tokens, view=view)
    labels = sorted(str(label) for label in centroids)
    if len(labels) < 2:
        return hidden.new_zeros(()), []
    center = torch.as_tensor(np.stack([normalize_vector(centroids[label]) for label in labels]), device=hidden.device, dtype=hidden.dtype)
    margins: list[Any] = []
    rows: list[dict[str, Any]] = []
    for key, row in pooled.items():
        true_label = str(row["label"])
        target_label = str(label_permutation.get(true_label, true_label)) if label_permutation else true_label
        if target_label not in centroids:
            continue
        scores = torch.matmul(center, row["embedding"])
        target_index = labels.index(target_label)
        wrong = torch.cat([scores[:target_index], scores[target_index + 1:]])
        margin = scores[target_index] - wrong.max()
        margins.append(margin)
        rows.append({"token_id": key, "token_index": int(row["token_index"]), "label": true_label, "target_label": target_label, "margin": margin, "embedding": row["embedding"]})
    if not margins:
        return hidden.new_zeros(()), rows
    values = torch.stack(margins)
    loss = torch.nn.functional.softplus((float(target_margin) - values) / float(temperature)).mean()
    return loss, rows


__all__ = [
    "FrozenPhoneTeacher",
    "load_frozen_teacher",
    "normalize_waveform",
    "phone_margins_from_hidden",
    "pool_hidden",
    "processor_parity",
]
