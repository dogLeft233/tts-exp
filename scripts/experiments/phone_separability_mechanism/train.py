"""Small conditional enhancer and training diagnostics for the gated branch."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _torch():
    import torch

    return torch


class BoundedGainEnhancer:  # replaced with a torch module at construction time
    """Factory-compatible name; the real class is created lazily below."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _torch()
        nn = torch.nn
        F = torch.nn.functional
        max_gain_db = float(kwargs.pop("max_gain_db", 6.0))
        channels = int(kwargs.pop("channels", 64))
        if args:
            mel_bands = int(args[0])
        else:
            mel_bands = int(kwargs.pop("mel_bands", 24))
        if kwargs:
            raise TypeError(f"unexpected enhancer arguments: {sorted(kwargs)}")

        class _Enhancer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mel_bands = mel_bands
                self.channels = channels
                self.max_gain_db = max_gain_db
                self.input = nn.Conv1d(mel_bands, channels, kernel_size=5, padding=2)
                self.blocks = nn.ModuleList()
                for dilation in (1, 2, 4):
                    self.blocks.append(nn.Sequential(
                        nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                        nn.GELU(),
                        nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                    ))
                self.output = nn.Conv1d(channels, mel_bands, kernel_size=1)
                nn.init.zeros_(self.output.weight)
                nn.init.zeros_(self.output.bias)

            def forward(self, mel: Any) -> Any:
                if mel.ndim != 3 or mel.shape[1] != self.mel_bands:
                    raise ValueError(f"mel must have shape [B,{self.mel_bands},T]")
                hidden = F.gelu(self.input(mel))
                for block in self.blocks:
                    hidden = F.gelu(hidden + block(hidden))
                return torch.tanh(self.output(hidden)) * self.max_gain_db

            def config(self) -> dict[str, Any]:
                return {"class": "BoundedGainEnhancer", "mel_bands": self.mel_bands, "channels": self.channels, "dilations": [1, 2, 4], "max_gain_db": self.max_gain_db}

        return _Enhancer()


def differentiable_ssl_forward(teacher: Any, waveform: Any, *, layer: int = 6) -> Any:
    """Call a frozen teacher while retaining gradients through its input."""

    torch = _torch()
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape [B,1,N]")
    for parameter in teacher.parameters() if hasattr(teacher, "parameters") else []:
        if parameter.requires_grad:
            raise ValueError("teacher parameters must be frozen")
    if hasattr(teacher, "encode_layers"):
        result = teacher.encode_layers(waveform, [int(layer)], retain_gradient=True)
        return result[int(layer)]
    signal = waveform[:, 0, :]
    output = teacher(signal, output_hidden_states=True)
    hidden = getattr(output, "hidden_states", None)
    if hidden is None:
        raise ValueError("teacher output lacks hidden_states")
    return hidden[int(layer)]


def phone_margin_loss(margins: Any, *, target: float = 0.05, temperature: float = 0.1) -> Any:
    torch = _torch()
    if margins.numel() == 0:
        return margins.new_zeros(())
    return torch.nn.functional.softplus((float(target) - margins) / float(temperature)).mean()


def keep_loss(source: Any, candidate: Any, mask: Any | None = None) -> Any:
    torch = _torch()
    if mask is None:
        mask = torch.ones_like(source)
    while mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)
    numerator = ((candidate - source) * mask).square().mean()
    denominator = source.square().mean().clamp_min(1e-8)
    return numerator / denominator


def gradient_diagnostics(
    phone_objective: Any,
    keep_objective: Any,
    parameters: Sequence[Any],
) -> dict[str, float | None]:
    torch = _torch()
    params = [parameter for parameter in parameters if parameter.requires_grad]
    phone_grads = torch.autograd.grad(phone_objective, params, retain_graph=True, allow_unused=True)
    keep_grads = torch.autograd.grad(keep_objective, params, retain_graph=True, allow_unused=True)
    phone = torch.cat([grad.detach().reshape(-1) for grad in phone_grads if grad is not None]) if any(grad is not None for grad in phone_grads) else torch.zeros(1, device=phone_objective.device)
    keep = torch.cat([grad.detach().reshape(-1) for grad in keep_grads if grad is not None]) if any(grad is not None for grad in keep_grads) else torch.zeros_like(phone)
    phone_norm = float(torch.linalg.vector_norm(phone).cpu())
    keep_norm = float(torch.linalg.vector_norm(keep).cpu())
    cosine = float(torch.dot(phone, keep).cpu() / max(phone_norm * keep_norm, 1e-12))
    return {"phone_grad_norm": phone_norm, "keep_grad_norm": keep_norm, "keep_to_phone_norm": keep_norm / max(phone_norm, 1e-12), "cosine": cosine}


def select_checkpoint(checkpoints: Sequence[Mapping[str, Any]], *, min_accuracy: float, use_xlsr: bool = False, include_e_seen: bool = False) -> dict[str, Any] | None:
    """Select only DEV-valid checkpoints; forbidden fields are rejected."""

    if use_xlsr or include_e_seen:
        raise ValueError("checkpoint selection cannot use XLSR or E_SEEN")
    eligible = []
    for row in checkpoints:
        if str(row.get("split", "dev")) != "dev":
            continue
        if float(row.get("accuracy", -math.inf)) < float(min_accuracy):
            continue
        if not bool(row.get("waveform_qc", True)) or not bool(row.get("timing_qc", True)):
            continue
        if row.get("hubert_margin") is None:
            continue
        eligible.append(dict(row))
    if not eligible:
        return None
    eligible.sort(key=lambda row: (-float(row["hubert_margin"]), float(row.get("keep_loss", math.inf)), int(row.get("step", 10**9))))
    return eligible[0]


def train_one_seed(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    phone_objective: Callable[[Mapping[str, Any], Any], Any],
    max_steps: int = 1500,
    seed: int = 20260921,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    timeout_s: float = 7200.0,
    diagnostics_every: int = 100,
) -> dict[str, Any]:
    """Generic gated loop; the runner supplies frozen feature/support logic."""

    torch = _torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    model.train()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    conflict_streak = 0
    for step in range(1, int(max_steps) + 1):
        batch = batches[(step - 1) % len(batches)]
        output = model(batch["mel"])
        phone = phone_objective(batch, output)
        keep = keep_loss(batch["source"], batch.get("candidate", batch["source"]), batch.get("mask"))
        keep_scale = 0.1 if step <= 200 else (0.1 + 0.9 * min(1.0, (step - 200) / 200.0)) if step <= 400 else 1.0
        total = phone + keep_scale * 10.0 * keep
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row: dict[str, Any] = {"step": step, "phone_loss": float(phone.detach().cpu()), "keep_loss": float(keep.detach().cpu()), "total_loss": float(total.detach().cpu()), "keep_scale": keep_scale}
        if step % int(diagnostics_every) == 0:
            diag = gradient_diagnostics(phone, 10.0 * keep, list(model.parameters()))
            row["gradient_diagnostics"] = diag
            if float(diag["cosine"] or 0.0) < -0.5 and float(diag["keep_to_phone_norm"] or 0.0) > 10.0:
                conflict_streak += 1
            else:
                conflict_streak = 0
            if conflict_streak >= 3:
                return {"status": "OBJECTIVE_CONFLICT", "seed": int(seed), "steps": step, "history": history + [row]}
        history.append(row)
        if time.monotonic() - started >= timeout_s:
            return {"status": "BUDGET_LIMITED", "seed": int(seed), "steps": step, "history": history}
    return {"status": "COMPLETE", "seed": int(seed), "steps": int(max_steps), "history": history}


__all__ = ["BoundedGainEnhancer", "differentiable_ssl_forward", "gradient_diagnostics", "keep_loss", "phone_margin_loss", "select_checkpoint", "train_one_seed"]
