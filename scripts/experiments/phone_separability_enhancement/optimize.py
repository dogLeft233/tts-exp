"""Per-utterance low-dimensional optimization under the natural-clock renderer."""

from __future__ import annotations

import time
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .audio import BandGainRenderer, export_pcm, gain_field_from_parameters, make_protected_mask, occurrence_parameter_layout, parameter_count, validate_distortion
from .teacher import FrozenPhoneTeacher, phone_margins_from_hidden


def _torch():
    import torch

    return torch


def _gradient_vector(objective: Any, parameter: Any, *, retain_graph: bool = True) -> Any:
    torch = _torch()
    gradient = torch.autograd.grad(objective, parameter, retain_graph=retain_graph, allow_unused=True)
    value = gradient[0] if gradient[0] is not None else torch.zeros_like(parameter)
    return value.reshape(-1)


def gradient_diagnostics(phone: Any, keep: Any, tv: Any, parameter: Any) -> dict[str, float | None]:
    torch = _torch()
    phone_vector = _gradient_vector(phone, parameter, retain_graph=True)
    keep_vector = _gradient_vector(keep, parameter, retain_graph=True)
    tv_vector = _gradient_vector(tv, parameter, retain_graph=True)
    phone_norm = float(torch.linalg.vector_norm(phone_vector).detach().cpu())
    keep_norm = float(torch.linalg.vector_norm(keep_vector).detach().cpu())
    tv_norm = float(torch.linalg.vector_norm(tv_vector).detach().cpu())
    cosine = float(torch.dot(phone_vector, keep_vector).detach().cpu() / max(phone_norm * keep_norm, 1e-12))
    return {"phone_grad_norm": phone_norm, "keep_grad_norm": keep_norm, "tv_grad_norm": tv_norm, "keep_to_phone_norm": keep_norm / max(phone_norm, 1e-12), "cosine": cosine, "parameter_count": int(parameter.numel())}


def _sentence_score(teacher: FrozenPhoneTeacher, waveform: Any, tokens: Sequence[Mapping[str, Any]], centroids: Mapping[str, Sequence[float]], *, layer: int, view: str, device: str) -> dict[str, Any]:
    torch = _torch()
    with torch.no_grad():
        hidden, frame_times, _ = teacher.encode(waveform.detach(), layer=layer)
        _, rows = phone_margins_from_hidden(hidden, frame_times, tokens, centroids, view=view)
    if not rows:
        return {"accuracy": None, "signed_margin": None, "valid_tokens": 0, "rows": []}
    margins = [float(row["margin"].detach().cpu()) for row in rows]
    return {"accuracy": float(np.mean([float(value >= 0.0) for value in margins])), "signed_margin": float(np.mean(margins)), "valid_tokens": len(rows), "rows": [{"token_id": row["token_id"], "label": row["label"], "margin": float(row["margin"].detach().cpu())} for row in rows]}


def label_permutation(labels: Sequence[str]) -> dict[str, str]:
    values = sorted({str(label) for label in labels})
    if len(values) < 2:
        return {}
    return {value: values[(index + 1) % len(values)] for index, value in enumerate(values)}


def optimize_utterance(
    audio_pcm: np.ndarray,
    tokens: Sequence[Mapping[str, Any]],
    teacher: FrozenPhoneTeacher,
    centroids: Mapping[str, Sequence[float]],
    *,
    arm: str,
    max_gain_db: float,
    layer: int = 6,
    view: str = "core",
    sample_rate: int = 16_000,
    node_ms: float = 80.0,
    lr: float = 0.05,
    target_margin: float = 0.05,
    temperature: float = 0.1,
    keep_weight: float = 1.0,
    tv_weight: float = 0.01,
    steps: int = 120,
    checkpoints: Sequence[int] = (0, 25, 50, 75, 100, 120),
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Optimize one utterance and select only by its frozen HuBERT score."""

    torch = _torch()
    source_pcm = np.asarray(audio_pcm, dtype=np.int16).reshape(-1)
    if source_pcm.size <= 256:
        return {"status": "SHORT_AUDIO", "arm": arm, "steps": 0, "checkpoints": []}
    dynamic = "DYNAMIC" in arm or "LABEL_PERM" in arm
    permutation = label_permutation(list(centroids)) if "LABEL_PERM" in arm else None
    if arm in {"N_ID", "STFT_RT"}:
        return {"status": "IDENTITY_RETURNED" if arm == "N_ID" else "IDENTITY_RETURNED", "arm": arm, "steps": 0, "checkpoints": [{"step": 0, "score": {"accuracy": None, "signed_margin": None}, "identity": True}], "parameters": np.zeros(0, dtype=np.float32).tolist(), "dynamic": False}
    layout = occurrence_parameter_layout(tokens, dynamic=dynamic, sample_rate=sample_rate, node_ms=node_ms)
    count = parameter_count(layout)
    if count <= 0:
        return {"status": "BASELINE_INELIGIBLE", "arm": arm, "steps": 0, "checkpoints": [], "parameters": []}
    device = teacher.device
    x = torch.as_tensor(source_pcm.astype(np.float32) / 32768.0, device=device)
    protected_info = make_protected_mask(tokens, source_pcm.size, sample_rate=sample_rate)
    mask = torch.as_tensor(protected_info["mask"], device=device, dtype=torch.float32)
    protected = protected_info["protected"]
    renderer = BandGainRenderer(sample_rate=sample_rate, n_bands=24, max_gain_db=max_gain_db)
    parameter = torch.nn.Parameter(torch.zeros((1, count), device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=float(lr))
    with torch.no_grad():
        source_hidden, source_times, _ = teacher.encode(x.detach(), layer=layer)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    started = time.monotonic()
    checkpoint_set = {int(value) for value in checkpoints}
    if 0 in checkpoint_set:
        baseline = _sentence_score(teacher, x, tokens, centroids, layer=layer, view=view, device=device)
        history.append({"step": 0, "loss": None, "score": baseline, "render": {"identity": True}, "gradient": None})
        best = {"step": 0, "score": baseline, "parameters": parameter.detach().cpu().numpy().copy(), "pcm": source_pcm.copy(), "render": {"identity": True}, "distortion": {"pass": True, "residual_energy_ratio": 0.0, "snr_db": float("inf"), "rms_change_db": 0.0}}
    final_status = "COMPLETE"
    for step in range(1, int(steps) + 1):
        frame_times = source_times
        gain_db = gain_field_from_parameters(parameter, tokens, frame_times, dynamic=dynamic, sample_rate=sample_rate, node_ms=node_ms, max_gain_db=max_gain_db)
        rendered, render_meta = renderer.render(x, gain_db, mask)
        hidden, candidate_times, _ = teacher.encode(rendered, layer=layer)
        phone, phone_rows = phone_margins_from_hidden(hidden, candidate_times, tokens, centroids, view=view, label_permutation=permutation, target_margin=target_margin, temperature=temperature)
        keep = ((rendered - x) * mask).square().sum() / x.square().sum().clamp_min(1e-8)
        normalized = gain_db / float(max_gain_db)
        tv_time = normalized[:, :, 1:].sub(normalized[:, :, :-1]).square().mean() if normalized.shape[-1] > 1 else normalized.new_zeros(())
        tv_freq = normalized[:, 1:, :].sub(normalized[:, :-1, :]).square().mean() if normalized.shape[1] > 1 else normalized.new_zeros(())
        tv = tv_time + tv_freq
        total = phone + float(keep_weight) * keep + float(tv_weight) * tv
        diagnostics = None
        if step % 25 == 0 or step == 1:
            diagnostics = gradient_diagnostics(phone, keep, tv, parameter)
            if not all(value is None or np.isfinite(float(value)) for value in diagnostics.values()):
                final_status = "ENGINEERING_FAILURE"
                break
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        optimizer.step()
        if step in checkpoint_set or step == int(steps):
            with torch.no_grad():
                candidate_pcm, pcm_meta = export_pcm(rendered, source_pcm, protected=protected)
            distortion = validate_distortion(source_pcm, candidate_pcm, protected_info["mask"], min_snr_db=20.0, max_rms_change_db=1.0)
            score = _sentence_score(teacher, torch.as_tensor(candidate_pcm.astype(np.float32) / 32768.0, device=device), tokens, centroids, layer=layer, view=view, device=device)
            row = {"step": step, "loss": float(total.detach().cpu()), "phone_loss": float(phone.detach().cpu()), "keep_loss": float(keep.detach().cpu()), "tv_loss": float(tv.detach().cpu()), "score": score, "render": render_meta, "pcm": pcm_meta, "distortion": distortion, "gradient": diagnostics}
            history.append(row)
            candidate = {"step": step, "score": score, "parameters": parameter.detach().cpu().numpy().copy(), "pcm": candidate_pcm, "render": render_meta, "distortion": distortion}
            if distortion.get("pass") and (best is None or float(score.get("signed_margin", -math.inf)) > float(best["score"].get("signed_margin", -math.inf)) or (float(score.get("signed_margin", -math.inf)) == float(best["score"].get("signed_margin", -math.inf)) and float(distortion.get("residual_energy_ratio", 1.0)) < float(best["distortion"].get("residual_energy_ratio", 1.0)))):
                best = candidate
        if time.monotonic() - started >= float(timeout_s):
            final_status = "BUDGET_LIMITED"
            break
    if best is None:
        best = {"step": 0, "score": {"accuracy": None, "signed_margin": None}, "parameters": parameter.detach().cpu().numpy().copy(), "pcm": source_pcm.copy(), "render": {"identity": True}, "distortion": {"pass": True}}
        final_status = "ENGINEERING_FAILURE" if final_status == "COMPLETE" else final_status
    if int(best["step"]) == 0 and final_status == "COMPLETE":
        final_status = "IDENTITY_RETURNED"
    return {"status": final_status, "arm": arm, "dynamic": dynamic, "steps": max([int(row.get("step", 0)) for row in history], default=0), "selected_step": int(best["step"]), "selected_score": best["score"], "selected_render": best["render"], "selected_distortion": best["distortion"], "parameters": np.asarray(best["parameters"], dtype=np.float32).reshape(-1).tolist(), "selected_pcm": best["pcm"], "history": history, "protected_mask": protected_info["protected"].astype(np.uint8).tolist(), "edit_mask": protected_info["mask"].astype(np.float32).tolist(), "label_permutation": permutation}


__all__ = ["gradient_diagnostics", "label_permutation", "optimize_utterance"]
