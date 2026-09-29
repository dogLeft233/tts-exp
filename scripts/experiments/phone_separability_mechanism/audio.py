"""Time-preserving spectral construction for the mechanism experiment."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_phone_rules_audio import (
    build_edit_mask,
    make_gain_control,
    render_rule_arm,
    validate_waveform,
)


def _as_pcm(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.int16 or array.size == 0:
        raise ValueError("audio must be a non-empty one-dimensional int16 vector")
    return array


def _float_pcm(values: np.ndarray) -> np.ndarray:
    return _as_pcm(values).astype(np.float64) / 32768.0


def _quantize_masked(source_pcm: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    source = _as_pcm(source_pcm)
    x = _float_pcm(source)
    y = np.asarray(candidate, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=np.float64).reshape(-1)
    if y.shape != x.shape or m.shape != x.shape or not np.all(np.isfinite(y)):
        raise ValueError("candidate and mask must match source and be finite")
    residual = m * (y - x)
    alpha = 1.0
    positive = residual > 0.0
    negative = residual < 0.0
    if np.any(positive):
        alpha = min(alpha, float(np.min((32767.0 / 32768.0 - x[positive]) / residual[positive])))
    if np.any(negative):
        alpha = min(alpha, float(np.min((-1.0 - x[negative]) / residual[negative])))
    alpha = float(max(0.0, min(1.0, alpha)))
    output = source.copy()
    editable = m > 0.0
    rendered = x + alpha * residual
    if np.any(editable):
        values = np.rint(np.clip(rendered[editable], -1.0, 32767.0 / 32768.0) * 32768.0).astype(np.int64)
        output[editable] = values.astype(np.int16)
    return output, alpha


def build_speech_mask(
    sample_count: int,
    tokens: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int = 16_000,
    edge_guard_s: float = 0.010,
    taper_s: float = 0.005,
) -> np.ndarray:
    """Allow edits in continuous speech while preserving pause boundaries."""

    if sample_count <= 0 or sample_rate <= 0:
        raise ValueError("sample_count and sample_rate must be positive")
    mask = np.zeros(int(sample_count), dtype=np.float64)
    speech_spans: list[tuple[int, int]] = []
    silence_spans: list[tuple[int, int]] = []
    for token in tokens:
        begin = max(0, min(sample_count, int(math.ceil(float(token["start_s"]) * sample_rate))))
        end = max(0, min(sample_count, int(math.floor(float(token["end_s"]) * sample_rate))))
        if end <= begin:
            continue
        if bool(token.get("speech", not token.get("silence", False))):
            speech_spans.append((begin, end))
        else:
            silence_spans.append((begin, end))
    for begin, end in speech_spans:
        mask[begin:end] = 1.0
    guard = int(round(edge_guard_s * sample_rate))
    taper = int(round(taper_s * sample_rate))
    if speech_spans:
        first = min(begin for begin, _ in speech_spans)
        last = max(end for _, end in speech_spans)
        mask[max(0, first - guard) : min(sample_count, first + guard)] = 0.0
        mask[max(0, last - guard) : min(sample_count, last + guard)] = 0.0
    for begin, end in silence_spans:
        mask[max(0, begin - guard) : min(sample_count, begin + guard)] = 0.0
        mask[max(0, end - guard) : min(sample_count, end + guard)] = 0.0
    if taper > 0:
        # Smooth only the transition between editable speech and protected
        # regions.  Internal phone boundaries remain editable.
        padded = np.pad(mask, (1, 1), mode="edge")
        transitions = np.flatnonzero(np.diff(padded) != 0)
        for index in transitions:
            left = max(0, int(index) - taper)
            right = min(sample_count, int(index) + taper)
            if right <= left:
                continue
            if mask[min(sample_count - 1, int(index))] > 0.0:
                mask[left:right] = np.maximum(mask[left:right], np.linspace(0.0, 1.0, right - left))
            else:
                mask[left:right] = np.minimum(mask[left:right], np.linspace(1.0, 0.0, right - left))
    return np.clip(mask, 0.0, 1.0)


def _stft(waveform: np.ndarray, *, sample_rate: int, n_fft: int, win_length: int, hop_length: int) -> tuple[Any, Any]:
    import torch

    x = np.asarray(waveform, dtype=np.float64)
    if x.size <= n_fft // 2:
        raise ValueError("waveform is too short for reflect-centered STFT")
    tensor = torch.from_numpy(x)
    window = torch.hann_window(win_length, periodic=True, dtype=torch.float64)
    spectrum = torch.stft(tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, pad_mode="reflect", return_complex=True)
    return spectrum, window


def _istft(spectrum: Any, window: Any, length: int, *, n_fft: int, win_length: int, hop_length: int) -> np.ndarray:
    import torch

    result = torch.istft(spectrum, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, length=int(length))
    output = result.detach().cpu().numpy().astype(np.float64)
    if output.size != length or not np.all(np.isfinite(output)):
        raise ValueError("ISTFT output is invalid")
    return output


def _frame_labels(tokens: Sequence[Mapping[str, Any]], frame_count: int, hop_length: int, sample_rate: int) -> list[str | None]:
    labels: list[str | None] = []
    for index in range(frame_count):
        time = index * hop_length / sample_rate
        label = None
        for token in tokens:
            if bool(token.get("speech", not token.get("silence", False))) and float(token["start_s"]) <= time < float(token["end_s"]):
                label = str(token.get("label", token.get("token", "")))
                break
        labels.append(label)
    return labels


def _shape_from_audio(pcm: np.ndarray, tokens: Sequence[Mapping[str, Any]], *, sample_rate: int, n_fft: int, win_length: int, hop_length: int) -> dict[str, list[np.ndarray]]:
    spectrum, _ = _stft(_float_pcm(pcm), sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    magnitude = np.maximum(np.abs(spectrum.detach().cpu().numpy()), 1e-5)
    log_mag = np.log(magnitude)
    shape = log_mag - log_mag.mean(axis=0, keepdims=True)
    labels = _frame_labels(tokens, shape.shape[1], hop_length, sample_rate)
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, label in enumerate(labels):
        if label:
            by_label[label].append(shape[:, index])
    return {label: values for label, values in by_label.items()}


def fit_phone_shape_templates(
    pairs: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int = 16_000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
    min_tokens: int = 20,
    min_groups: int = 3,
) -> dict[str, Any]:
    """Fit T-minus-natural spectral-shape templates on FIT pairs only."""

    buckets: dict[str, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for pair in pairs:
        if str(pair.get("analysis_split")) != "fit":
            continue
        group = str(pair["source_group"])
        sides = pair.get("sides", {})
        for condition in ("natural", "tts"):
            side = sides.get(condition)
            if not isinstance(side, Mapping) or side.get("audit_status") != "OK":
                continue
            audio = np.asarray(side.get("_pcm")) if side.get("_pcm") is not None else None
            if audio is None:
                from scripts.experiments.lrs3_phone_rules_worker import read_pcm16

                audio, _ = read_pcm16(Path(str(side["audio_path"])), sample_rate=sample_rate)
            shapes = _shape_from_audio(audio, side.get("tokens", []), sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            for label, values in shapes.items():
                buckets[label][condition][group].extend(values)
    templates: dict[str, np.ndarray] = {}
    support: dict[str, Any] = {}
    for label in sorted(buckets):
        info: dict[str, Any] = {}
        condition_means: dict[str, np.ndarray] = {}
        eligible = True
        for condition in ("natural", "tts"):
            groups = buckets[label].get(condition, {})
            token_count = sum(len(values) for values in groups.values())
            group_names = sorted(groups)
            info[condition] = {"token_count": token_count, "group_count": len(group_names), "groups": group_names}
            if token_count < min_tokens or len(group_names) < min_groups:
                eligible = False
                continue
            condition_means[condition] = np.mean(np.stack([np.mean(np.stack(groups[group]), axis=0) for group in group_names]), axis=0)
        support[label] = info
        if eligible and set(condition_means) == {"natural", "tts"}:
            delta = condition_means["tts"] - condition_means["natural"]
            delta = delta - delta.mean()
            templates[label] = delta.astype(np.float64)
    return {"schema_version": 1, "templates": {label: value.tolist() for label, value in templates.items()}, "support": support, "n_fft": n_fft, "win_length": win_length, "hop_length": hop_length, "sample_rate": sample_rate, "fit_scope": "fit_groups_only", "template_type": "T_minus_N_log_spectral_shape"}


def _render_with_delta(
    source_pcm: np.ndarray,
    tokens: Sequence[Mapping[str, Any]],
    delta_by_label: Mapping[str, Sequence[float]],
    *,
    beta: float,
    mask: np.ndarray,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    max_gain_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = _as_pcm(source_pcm)
    spectrum, window = _stft(_float_pcm(source), sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    magnitude = np.maximum(np.abs(spectrum.detach().cpu().numpy()), 1e-5)
    phase = spectrum.detach().cpu().numpy() / magnitude
    logs = np.log(magnitude)
    labels = _frame_labels(tokens, logs.shape[1], hop_length, sample_rate)
    changed_frames = 0
    for index, label in enumerate(labels):
        if label is None:
            continue
        template = delta_by_label.get(label, delta_by_label.get("__global__"))
        if template is None:
            continue
        delta = np.asarray(template, dtype=np.float64).reshape(-1)
        if delta.size != logs.shape[0] or not np.all(np.isfinite(delta)):
            continue
        delta = np.clip(float(beta) * (delta - delta.mean()), -float(max_gain_db) * math.log(10.0) / 20.0, float(max_gain_db) * math.log(10.0) / 20.0)
        original_energy = float(np.sum(np.square(magnitude[:, index])))
        new_mag = np.exp(logs[:, index] + delta)
        new_energy = float(np.sum(np.square(new_mag)))
        if new_energy > 1e-12 and original_energy > 1e-12:
            new_mag *= math.sqrt(original_energy / new_energy)
        logs[:, index] = np.log(np.maximum(new_mag, 1e-5))
        changed_frames += 1
    modified = np.exp(logs) * phase
    output = _istft(
        __import__("torch").from_numpy(modified).to(dtype=__import__("torch").complex128),
        window,
        source.size,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    rendered, alpha = _quantize_masked(source, output, mask)
    return rendered, {"algorithm": "phone_conditioned_log_spectral_shape", "beta": float(beta), "max_gain_db": float(max_gain_db), "changed_frame_count": changed_frames, "alpha": alpha, "mask_sha256": hashlib.sha256(np.asarray(mask, dtype="<f8").tobytes()).hexdigest(), "mask_edit_fraction": float(np.mean(mask > 0.0)), "changed_sample_count": int(np.count_nonzero(rendered != source)), "effective_edit": bool(np.any(rendered != source) and alpha > 0.0)}


def render_roundtrip(source_pcm: np.ndarray, *, mask: np.ndarray | None = None, sample_rate: int = 16_000, n_fft: int = 512, win_length: int = 512, hop_length: int = 128) -> tuple[np.ndarray, dict[str, Any]]:
    source = _as_pcm(source_pcm)
    spectrum, window = _stft(_float_pcm(source), sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    output = _istft(spectrum, window, source.size, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    chosen = np.ones(source.size, dtype=np.float64) if mask is None else np.asarray(mask, dtype=np.float64)
    result, alpha = _quantize_masked(source, output, chosen)
    return result, {"algorithm": "stft_istft_identity", "alpha": alpha, "max_abs_pcm_error": int(np.max(np.abs(result.astype(np.int64) - source.astype(np.int64)))), "sample_count": int(source.size)}


def render_phone_shape(
    source_pcm: np.ndarray,
    tokens: Sequence[Mapping[str, Any]],
    templates: Mapping[str, Sequence[float]] | Mapping[str, Any],
    *,
    beta: float = 1.0,
    mask_mode: str = "speech",
    sample_rate: int = 16_000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
    edge_guard_s: float = 0.010,
    taper_s: float = 0.005,
    max_gain_db: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = _as_pcm(source_pcm)
    mask = build_edit_mask(source.size, tokens, sample_rate=sample_rate, edge_guard_s=edge_guard_s, taper_s=taper_s) if mask_mode == "core" else build_speech_mask(source.size, tokens, sample_rate=sample_rate, edge_guard_s=edge_guard_s, taper_s=taper_s)
    values = templates.get("templates", templates) if isinstance(templates, Mapping) else templates
    if not isinstance(values, Mapping):
        raise ValueError("templates must be a label-to-vector mapping")
    output, metadata = _render_with_delta(source, tokens, values, beta=beta, mask=mask, sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, max_gain_db=max_gain_db)
    metadata["mask_mode"] = mask_mode
    metadata["template_count"] = len(values)
    metadata["template_labels"] = sorted(str(label) for label in values)
    return output, metadata


def render_paired_shape(
    source_pcm: np.ndarray,
    donor_pcm: np.ndarray,
    source_tokens: Sequence[Mapping[str, Any]],
    donor_tokens: Sequence[Mapping[str, Any]],
    *,
    beta: float = 1.0,
    mask_mode: str = "speech",
    sample_rate: int = 16_000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
    edge_guard_s: float = 0.010,
    taper_s: float = 0.005,
    max_gain_db: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Oracle same-sentence donor, retained only as a mechanism control."""

    from .features import match_occurrences

    matching = match_occurrences(source_tokens, donor_tokens)
    # The paired donor is represented by phone-conditioned donor-minus-source
    # templates.  Only identity-unique occurrences contribute.
    source_shape = _shape_from_audio(_as_pcm(source_pcm), source_tokens, sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    donor_shape = _shape_from_audio(_as_pcm(donor_pcm), donor_tokens, sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    delta: dict[str, np.ndarray] = {}
    for match in matching["matches"]:
        label = str(match["label"])
        if label not in source_shape or label not in donor_shape:
            continue
        delta[label] = np.mean(np.stack(donor_shape[label]), axis=0) - np.mean(np.stack(source_shape[label]), axis=0)
    output, metadata = render_phone_shape(source_pcm, source_tokens, delta, beta=beta, mask_mode=mask_mode, sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, edge_guard_s=edge_guard_s, taper_s=taper_s, max_gain_db=max_gain_db)
    metadata.update({"oracle": True, "matching": matching, "donor_pcm_sha256": hashlib.sha256(_as_pcm(donor_pcm).astype("<i2").tobytes()).hexdigest()})
    return output, metadata


__all__ = [
    "build_edit_mask",
    "build_speech_mask",
    "fit_phone_shape_templates",
    "make_gain_control",
    "render_phone_shape",
    "render_paired_shape",
    "render_roundtrip",
    "render_rule_arm",
    "validate_waveform",
]
