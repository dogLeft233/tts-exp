"""Identity waveform model and frozen SyncNet state contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from scripts.pnp_audio_enhancer import ResidualTCN

from .config import MODEL_CHANNELS, MODEL_DILATIONS, MODEL_RESIDUAL_SCALE, SYNCNET_MODEL_SHA256
from .protocol import sha256_file


def build_waveform_model(*, seed: int, device: str | torch.device = "cpu") -> ResidualTCN:
    torch.manual_seed(seed)
    model = ResidualTCN(
        channels=MODEL_CHANNELS,
        dilations=MODEL_DILATIONS,
        residual_scale=MODEL_RESIDUAL_SCALE,
    )
    return model.to(device)


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _syncnet_class(syncnet_root: Path) -> type[nn.Module]:
    source = syncnet_root / "SyncNetModel.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("_mfa_linear_syncnet_model", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SyncNet model source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.S


def load_frozen_syncnet(
    checkpoint: str | Path,
    *,
    device: str | torch.device,
    expected_sha256: str = SYNCNET_MODEL_SHA256,
) -> nn.Module:
    path = Path(checkpoint).resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("SyncNet checkpoint hash mismatch")
    model_class = _syncnet_class(path.parent.parent)
    model = model_class(num_layers_in_fc_layers=1024)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    freeze_syncnet(model)
    return model


def freeze_syncnet(syncnet: nn.Module) -> None:
    syncnet.eval()
    for parameter in syncnet.parameters():
        parameter.requires_grad_(False)


def assert_syncnet_frozen(syncnet: nn.Module) -> None:
    training_modules = [name or "<root>" for name, module in syncnet.named_modules() if module.training]
    if training_modules:
        raise RuntimeError(f"FROZEN_STATE_MUTATION: SyncNet modules entered training mode: {training_modules}")
    if any(parameter.requires_grad for parameter in syncnet.parameters()):
        raise RuntimeError("FROZEN_STATE_MUTATION: SyncNet parameter requires gradients")


def trainable_parameters(model: nn.Module, syncnet: nn.Module) -> list[nn.Parameter]:
    assert_syncnet_frozen(syncnet)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    syncnet_ids = {id(parameter) for parameter in syncnet.parameters()}
    if any(id(parameter) in syncnet_ids for parameter in parameters):
        raise RuntimeError("SyncNet parameters leaked into the optimizer")
    return parameters


def assert_identity(model: nn.Module, waveform: Tensor) -> None:
    with torch.no_grad():
        candidate = model(waveform)
    if candidate.shape != waveform.shape or not torch.equal(candidate, waveform):
        raise RuntimeError("IDENTITY_FAILURE: fresh model is not sample-identical")


def _finite_positive_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("NONFINITE_TRAINING: non-finite gradient")
        squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def disposable_two_backward_smoke(
    *,
    seed: int,
    waveform: Tensor,
    objective: Callable[[nn.Module, Tensor], Tensor],
    learning_rate: float = 1e-4,
) -> dict[str, object]:
    """Prove output and upstream gradients, discard, then verify fresh hash."""
    model = build_waveform_model(seed=seed, device=waveform.device)
    assert_identity(model, waveform)
    initial_hash = module_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    optimizer.zero_grad(set_to_none=True)
    first_loss = objective(model, waveform)
    if first_loss.ndim != 0 or not torch.isfinite(first_loss.detach()):
        raise RuntimeError("NONFINITE_TRAINING: invalid first smoke loss")
    first_loss.backward()
    output_parameters = list(model.output_projection.parameters())
    first_output_norm = _finite_positive_norm(output_parameters)
    if not math.isfinite(first_output_norm) or first_output_norm <= 0:
        raise RuntimeError("NO_CANDIDATE_GRADIENT: zero output-projection gradient")
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second_loss = objective(model, waveform)
    second_loss.backward()
    upstream = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("output_projection.")
    ]
    second_upstream_norm = _finite_positive_norm(upstream)
    if not math.isfinite(second_upstream_norm) or second_upstream_norm <= 0:
        raise RuntimeError("NO_CANDIDATE_GRADIENT: zero upstream gradient after disposable update")

    del optimizer, model
    fresh = build_waveform_model(seed=seed, device=waveform.device)
    fresh_hash = module_state_sha256(fresh)
    if fresh_hash != initial_hash:
        raise RuntimeError("fresh P1 initialization hash differs after disposable smoke")
    return {
        "status": "GO",
        "initial_model_sha256": initial_hash,
        "fresh_model_sha256": fresh_hash,
        "first_loss": float(first_loss.detach().cpu()),
        "second_loss": float(second_loss.detach().cpu()),
        "first_output_projection_gradient_norm": first_output_norm,
        "second_upstream_gradient_norm": second_upstream_norm,
        "disposable_model_discarded": True,
    }


__all__ = [
    "assert_identity",
    "assert_syncnet_frozen",
    "build_waveform_model",
    "disposable_two_backward_smoke",
    "freeze_syncnet",
    "load_frozen_syncnet",
    "module_state_sha256",
    "trainable_parameters",
]
