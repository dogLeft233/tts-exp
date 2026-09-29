"""Real learned-enhancer loop for the natural-clock v2 branch."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.phone_separability_mechanism.train import BoundedGainEnhancer as _ReferenceBoundedGainEnhancer

from .audio import BandGainRenderer, export_pcm, make_protected_mask, validate_distortion
from .teacher import FrozenPhoneTeacher, phone_margins_from_hidden


def _torch():
    import torch

    return torch


class BoundedGainEnhancer(_ReferenceBoundedGainEnhancer):
    """Explicit v2 alias; the architecture is frozen to the v1 24-band TCN."""


def _model_input(renderer: BandGainRenderer, waveform: Any, *, mean: Any | None = None, std: Any | None = None) -> Any:
    features, _ = renderer.band_features(waveform)
    if mean is not None and std is not None:
        torch = _torch()
        mean_value = mean if isinstance(mean, torch.Tensor) else torch.as_tensor(mean, device=features.device, dtype=features.dtype)
        std_value = std if isinstance(std, torch.Tensor) else torch.as_tensor(std, device=features.device, dtype=features.dtype)
        features = (features - mean_value) / std_value.clamp_min(1e-6)
    return features


def _gradient_norms(model: Any) -> dict[str, float]:
    torch = _torch()
    values = [parameter.grad.detach().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
    vector = torch.cat(values) if values else torch.zeros(1)
    return {"norm": float(torch.linalg.vector_norm(vector).cpu()), "finite": bool(torch.isfinite(vector).all().cpu())}


def _loss_for_sample(model: Any, sample: Mapping[str, Any], teacher: FrozenPhoneTeacher, centroids: Mapping[str, Sequence[float]], renderer: BandGainRenderer, *, device: str, phone_weight: float = 1.0, keep_weight: float = 1.0, tv_weight: float = 0.01, dose_scale: float = 1.0) -> tuple[Any, dict[str, Any], Any]:
    torch = _torch()
    source_pcm = np.asarray(sample["audio_pcm"], dtype=np.int16)
    x = torch.as_tensor(source_pcm.astype(np.float32) / 32768.0, device=device)
    mask_info = sample.get("mask_info") or make_protected_mask(sample["tokens"], source_pcm.size, sample_rate=renderer.sample_rate)
    mask = torch.as_tensor(mask_info["mask"], device=device, dtype=torch.float32)
    model_input = _model_input(renderer, x, mean=sample.get("feature_mean"), std=sample.get("feature_std"))
    raw_gain = model(model_input)
    gain_db = raw_gain * float(dose_scale)
    y, render_meta = renderer.render(x, gain_db, mask)
    hidden, frame_times, _ = teacher.encode(y, layer=int(sample.get("layer", 6)))
    phone, phone_rows = phone_margins_from_hidden(hidden, frame_times, sample["tokens"], centroids, view=str(sample.get("view", "core")))
    keep = ((y - x) * mask).square().sum() / x.square().sum().clamp_min(1e-8)
    normalized = gain_db / renderer.max_gain_db
    tv_time = normalized[:, :, 1:].sub(normalized[:, :, :-1]).square().mean() if normalized.shape[-1] > 1 else normalized.new_zeros(())
    tv_freq = normalized[:, 1:, :].sub(normalized[:, :-1, :]).square().mean() if normalized.shape[1] > 1 else normalized.new_zeros(())
    tv = tv_time + tv_freq
    total = float(phone_weight) * phone + float(keep_weight) * keep + float(tv_weight) * tv
    return total, {"phone_loss": float(phone.detach().cpu()), "keep_loss": float(keep.detach().cpu()), "tv_loss": float(tv.detach().cpu()), "render": render_meta, "phone_tokens": len(phone_rows), "phone_rows": [{"token_id": row["token_id"], "label": row["label"], "margin": float(row["margin"].detach().cpu())} for row in phone_rows], "candidate": y, "source": x, "mask": mask, "raw_gain": raw_gain}, y


def _balanced_order(samples: Sequence[Mapping[str, Any]], seed: int) -> list[int]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(str(sample.get("source_group", "")), []).append(index)
    rng = random.Random(int(seed))
    for indices in groups.values():
        rng.shuffle(indices)
    result: list[int] = []
    while any(groups.values()):
        for group in sorted(groups):
            if groups[group]:
                result.append(groups[group].pop(0))
    return result


def train_enhancer(
    samples: Sequence[Mapping[str, Any]],
    teacher: FrozenPhoneTeacher,
    centroids: Mapping[str, Sequence[float]],
    *,
    seed: int,
    output_dir: str | Path,
    max_steps: int = 1500,
    timeout_s: float = 7200.0,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    gradient_accumulation: int = 4,
    dose_scale: float = 1.0,
    no_phone: bool = False,
    renderer: BandGainRenderer | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Train on FIT only; every keep loss uses the current rendered waveform."""

    torch = _torch()
    if not samples:
        return {"status": "NO_TRAINING_SAMPLES", "seed": int(seed), "updates": 0, "history": []}
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    random.seed(int(seed))
    renderer = renderer or BandGainRenderer(max_gain_db=6.0)
    model = BoundedGainEnhancer(mel_bands=24, channels=64, max_gain_db=6.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    order = _balanced_order(samples, seed)
    history: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    status = "COMPLETE"
    def save_checkpoint(step: int) -> str:
        path = output / f"step_{int(step):04d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": int(seed), "updates": int(step), "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "rng_python": random.getstate(), "config": {"dose_scale": dose_scale, "no_phone": no_phone}}, path)
        checkpoint_paths.append(str(path))
        return str(path)

    save_checkpoint(0)
    for update in range(1, int(max_steps) + 1):
        sample = samples[order[(update - 1) % len(order)]]
        total, diagnostics, _ = _loss_for_sample(model, sample, teacher, centroids, renderer, device=device, phone_weight=0.0 if no_phone else 1.0, dose_scale=dose_scale)
        (total / max(1, int(gradient_accumulation))).backward()
        if update % int(gradient_accumulation) == 0 or update == int(max_steps):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad = _gradient_norms(model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad = {"norm": None, "finite": True}
        if update == 1 or update % 100 == 0 or update == int(max_steps):
            history.append({"update": update, "loss": float(total.detach().cpu()), **{key: value for key, value in diagnostics.items() if key not in {"candidate", "source", "mask", "raw_gain"}}, "gradient": grad, "no_phone": bool(no_phone)})
        if update % 100 == 0 or update == int(max_steps):
            save_checkpoint(update)
        if not bool(grad.get("finite", True)):
            status = "ENGINEERING_FAILURE"
            break
        if time.monotonic() - started >= float(timeout_s):
            status = "BUDGET_LIMITED"
            break
    checkpoint = output / ("no_phone.pt" if no_phone else "last.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": int(seed), "updates": int(update), "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "rng_python": random.getstate(), "config": {"dose_scale": dose_scale, "no_phone": no_phone}}, checkpoint)
    return {"status": status, "seed": int(seed), "updates": int(update), "history": history, "checkpoint": str(checkpoint), "checkpoint_paths": checkpoint_paths, "model": model}


def load_checkpoint(model: Any, checkpoint: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return payload


def evaluate_model(model: Any, samples: Sequence[Mapping[str, Any]], teacher: FrozenPhoneTeacher, centroids: Mapping[str, Sequence[float]], *, device: str = "cuda:0", dose_scale: float = 1.0, renderer: BandGainRenderer | None = None) -> list[dict[str, Any]]:
    """Evaluate a frozen checkpoint; this function never updates model state."""

    torch = _torch()
    renderer = renderer or BandGainRenderer(max_gain_db=6.0)
    was_training = bool(model.training)
    model.eval()
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            total, diagnostics, candidate = _loss_for_sample(model, sample, teacher, centroids, renderer, device=device, dose_scale=dose_scale)
            source_pcm = np.asarray(sample["audio_pcm"], dtype=np.int16)
            protected = (sample.get("mask_info") or make_protected_mask(sample["tokens"], source_pcm.size, sample_rate=renderer.sample_rate))["protected"]
            candidate_pcm, pcm_meta = export_pcm(candidate, source_pcm, protected=protected)
            distortion = validate_distortion(source_pcm, candidate_pcm, ~protected)
            margins = [float(row["margin"]) for row in diagnostics["phone_rows"]]
            results.append({"pair_id": sample.get("pair_id"), "source_group": sample.get("source_group"), "accuracy": float(np.mean([float(value >= 0.0) for value in margins])) if margins else None, "signed_margin": float(np.mean(margins)) if margins else None, "t0_pass": bool(pcm_meta.get("new_saturated_samples", 1) == 0), "distortion_pass": bool(distortion.get("pass")), "residual_energy_ratio": distortion.get("residual_energy_ratio"), "pcm": candidate_pcm, "pcm_meta": pcm_meta, "diagnostics": {key: value for key, value in diagnostics.items() if key not in {"candidate", "source", "mask", "raw_gain"}}})
    if was_training:
        model.train()
    return results


def select_dev_candidate(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(row) for row in checkpoints if str(row.get("split")) == "dev" and row.get("signed_margin") is not None and bool(row.get("t0_pass")) and bool(row.get("distortion_pass"))]
    eligible.sort(key=lambda row: (-float(row["signed_margin"]), -float(row.get("accuracy", -1.0)), float(row.get("residual_energy_ratio", float("inf"))), int(row.get("step", 10**9))))
    return eligible[0] if eligible else None


__all__ = ["BoundedGainEnhancer", "evaluate_model", "load_checkpoint", "select_dev_candidate", "train_enhancer"]
