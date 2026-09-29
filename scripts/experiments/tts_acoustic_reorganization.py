#!/usr/bin/env python3
"""Controlled P/R trajectory intervention for the TTS acoustic-reorganization hypothesis.

The experiment is deliberately staged.  ``prepare`` binds the historical parent
run, ``features`` builds donor-only templates and writes the two interventions,
``audio`` decodes them with the frozen prematched HiFi-GAN, ``render`` runs the
same Wav2Lip command for every evaluation arm, ``score`` computes the frozen
SyncNet curves, and ``analyze``/``validate`` produce auditable outputs.

The module also exposes the small numerical kernels used by the CPU tests.  No
training, source replacement, or historical SyncNet score is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from knn_vc_retrieval import frame_owners, matched_span_map  # noqa: I001


PROTOCOL = "tts_acoustic_reorganization_v1"
RUN_ID = "tts_acoustic_reorganization_20260916"
DEFAULT_OUTPUT = REPO / "runs" / RUN_ID
DEFAULT_PARENT = REPO / "runs" / "mfa_linear_trajectory_ablation_20260916"
DEFAULT_KNN_SOURCE = REPO / "third_party" / "knn-vc"
DEFAULT_WAV2LIP = REPO / "third_party" / "Wav2Lip"
DEFAULT_SYNCNET = REPO / "third_party" / "syncnet_python"
DEFAULT_WAV2LIP_PYTHON = Path.home() / ".venvs" / "wav2lip" / "bin" / "python"
DEFAULT_SYNCNET_PYTHON = Path.home() / ".venvs" / "syncnet" / "bin" / "python"
DEFAULT_WAV2LIP_CHECKPOINT = DEFAULT_WAV2LIP / "checkpoints" / "wav2lip_gan.pth"
DEFAULT_SYNCNET_MODEL = DEFAULT_SYNCNET / "data" / "syncnet_v2.model"
SPEC_PATH = REPO / "basic-memory" / "Research" / "TTS 声学变化重组的轨迹与剩余项干预 Spec.md"

SAMPLE_RATE = 16_000
FRAME_STRIDE = 320
FEATURE_DIM = 1024
FPS = 25.0
TEMPLATE_POINTS = 8
MIN_OCCURRENCE_FRAMES = 5
EDGE_FRAMES = 1
MIN_INTERNAL_FRAMES = 3
MIN_ACTIVE_OCCURRENCES = 3
MIN_ACTIVE_FRACTION = 0.10
EPS = 1e-8
FLOAT_TOL = 1e-8
FLOAT32_TOL = 1e-5
VSHIFT = 15
MIN_COMMON_WINDOWS = 30
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_DRAWS = 20_000
BONFERRONI_ALPHA = 0.05 / 6.0

DONOR_IDS = tuple(range(1, 6))
EVALUATION_IDS = tuple(range(6, 16))
ARMS = ("N_RAW", "N_ID", "N_R_DOWN", "N_P_DOWN", "T_ID", "T_R_DOWN", "T_P_DOWN")
RESYNTH_ARMS = ARMS[1:]
N_ARMS = ("N_ID", "N_R_DOWN", "N_P_DOWN")
T_ARMS = ("T_ID", "T_R_DOWN", "T_P_DOWN")


class ProtocolError(RuntimeError):
    """Raised when a frozen experiment contract is violated."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc


def _resolve_asset(value: str | Path, *, expected_sha256: str | None = None) -> Path:
    raw = Path(str(value)).expanduser()
    candidates: list[Path] = [raw]
    if raw.is_absolute() and "tts-exp" in raw.parts:
        index = len(raw.parts) - 1 - raw.parts[::-1].index("tts-exp")
        candidates.append(REPO.joinpath(*raw.parts[index + 1 :]))
    elif not raw.is_absolute():
        candidates.append(REPO / raw)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if expected_sha256 is None or file_sha256(candidate) == expected_sha256:
            return candidate
    suffix = f" sha256={expected_sha256}" if expected_sha256 else ""
    raise ProtocolError(f"missing asset{suffix}: {value}")


def _audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        values, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # pragma: no cover - backend-specific error
        raise ProtocolError(f"cannot decode audio {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or int(rate) != SAMPLE_RATE:
        raise ProtocolError(f"audio must be mono {SAMPLE_RATE} Hz: {path} shape={values.shape} rate={rate}")
    if not np.isfinite(values).all() or np.max(np.abs(values), initial=0.0) > 1.0:
        raise ProtocolError(f"audio is non-finite or outside [-1,1]: {path}")
    return values, int(rate)


def _load_features(path: Path, *, key: str = "features") -> np.ndarray:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or key not in payload:
        raise ProtocolError(f"feature file {path} has no {key!r}")
    values = np.asarray(torch.as_tensor(payload[key], dtype=torch.float32).cpu(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or values.shape[0] == 0:
        raise ProtocolError(f"feature shape is not [T,{FEATURE_DIM}]: {path} {values.shape}")
    if not np.isfinite(values).all():
        raise ProtocolError(f"feature file is non-finite: {path}")
    return values


def _save_feature_file(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
    write_json(path.with_suffix(".json"), dict(metadata))


def _token_occurrences(
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
    natural_frame_count: int,
    tts_frame_count: int,
) -> list[dict[str, Any]]:
    """Pair natural and native-TTS MFA spans without changing their labels."""
    nowners = frame_owners(natural_frame_count, natural_tokens, frame_stride_samples=FRAME_STRIDE, sample_rate=SAMPLE_RATE)
    towners = frame_owners(tts_frame_count, tts_tokens, frame_stride_samples=FRAME_STRIDE, sample_rate=SAMPLE_RATE)
    mapping, _match_stats = matched_span_map(natural_tokens, tts_tokens)
    n_by_span: dict[int, list[int]] = defaultdict(list)
    t_by_span: dict[int, list[int]] = defaultdict(list)
    for index, owner in enumerate(nowners):
        n_by_span[owner.span_index].append(index)
    for index, owner in enumerate(towners):
        t_by_span[owner.span_index].append(index)
    result: list[dict[str, Any]] = []
    for span, token in enumerate(natural_tokens):
        nidx = n_by_span.get(span, [])
        tspan = mapping.get(span)
        tidx = t_by_span.get(tspan, []) if tspan is not None else []
        label = nowners[nidx[0]].label if nidx else str(token.get("token", token.get("label", ""))).casefold()
        silence = bool(nidx and nowners[nidx[0]].is_silence) or label in {"", "sil", "sp", "spn", "<sil>"}
        if silence:
            reason = "silence"
        elif tspan is None:
            reason = "unmatched"
        elif len(nidx) < MIN_OCCURRENCE_FRAMES or len(tidx) < MIN_OCCURRENCE_FRAMES:
            reason = "fewer_than_five_frames"
        elif len(nidx) - 2 < MIN_INTERNAL_FRAMES or len(tidx) - 2 < MIN_INTERNAL_FRAMES:
            reason = "internal_short"
        else:
            reason = "eligible"
        result.append({
            "occurrence_index": int(span),
            "label": label,
            "tts_span_index": None if tspan is None else int(tspan),
            "natural_frame_indices": [int(i) for i in nidx],
            "tts_frame_indices": [int(i) for i in tidx],
            "frame_count_n": len(nidx),
            "frame_count_t": len(tidx),
            "eligible": reason == "eligible",
            "reason": reason,
            "is_silence": silence,
        })
    return result


def _parent_occurrences(
    parent_occurrences: Sequence[Mapping[str, Any]],
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
    frame_count: int,
) -> list[dict[str, Any]]:
    """Build the evaluation mask required by the spec from the parent mask."""
    # The parent TTS grid has already been interpolated onto the natural clock.
    # Do not pass that frame count through the native TTS token timeline: its
    # final token ends earlier/later by design.  Only the natural owners are
    # used for the intervention mask; matched_span_map supplies the identity
    # check against the TTS alignment.
    owners = frame_owners(frame_count, natural_tokens, frame_stride_samples=FRAME_STRIDE, sample_rate=SAMPLE_RATE)
    mapping, _ = matched_span_map(natural_tokens, tts_tokens)
    by_span: dict[int, list[int]] = defaultdict(list)
    for index, owner in enumerate(owners):
        by_span[owner.span_index].append(index)
    parent_by_index = {int(row.get("occurrence_index", -1)): row for row in parent_occurrences}
    result: list[dict[str, Any]] = []
    for span, token in enumerate(natural_tokens):
        indices = by_span.get(span, [])
        label = owners[indices[0]].label if indices else str(token.get("token", token.get("label", ""))).casefold()
        silence = bool(indices and owners[indices[0]].is_silence) or label in {"", "sil", "sp", "spn", "<sil>"}
        parent = parent_by_index.get(span)
        if parent is not None and parent.get("frame_indices") is not None:
            parent_indices = [int(i) for i in parent.get("frame_indices", [])]
            if parent_indices != indices:
                raise ProtocolError(f"parent natural frame ownership changed at occurrence {span}")
        if silence:
            reason = "silence"
        elif span not in mapping:
            reason = "unmatched"
        elif parent is None or parent.get("eligible") is not True:
            reason = "parent_ineligible"
        elif len(indices) < MIN_OCCURRENCE_FRAMES:
            reason = "fewer_than_five_frames"
        elif len(indices) - 2 < MIN_INTERNAL_FRAMES:
            reason = "internal_short"
        else:
            reason = "eligible"
        result.append({
            "occurrence_index": int(span), "label": label, "tts_span_index": int(mapping[span]) if span in mapping else None,
            "natural_frame_indices": [int(i) for i in indices], "tts_frame_indices": [int(i) for i in indices],
            "frame_count_n": len(indices), "frame_count_t": len(indices), "eligible": reason == "eligible", "reason": reason, "is_silence": silence,
        })
    return result


def _center_and_interpolate(values: np.ndarray, points: int = TEMPLATE_POINTS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("trajectory must be [T,D] with at least two frames")
    centered = values - values.mean(axis=0, keepdims=True)
    old_x = np.linspace(0.0, 1.0, centered.shape[0])
    new_x = np.linspace(0.0, 1.0, points)
    result = np.stack([np.interp(new_x, old_x, centered[:, dim]) for dim in range(centered.shape[1])], axis=1)
    return result - result.mean(axis=0, keepdims=True)


def _template_to_length(template: np.ndarray, length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("template target length must be at least two")
    result = _center_and_interpolate(np.asarray(template, dtype=np.float64), length)
    return result - result.mean(axis=0, keepdims=True)


def _explained(values: np.ndarray, template: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    nv = float(np.linalg.norm(values))
    nq = float(np.linalg.norm(template))
    if nv <= EPS or nq <= EPS:
        return None
    return float(np.dot(values.ravel(), template.ravel()) ** 2 / (nv * nv * nq * nq))


def _project_components(values: np.ndarray, template: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return one full-time orthogonal P/R projection in float64."""
    values = np.asarray(values, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if values.shape != template.shape or values.ndim != 2:
        raise ValueError("values and template must have the same [T,D] shape")
    u = values - values.mean(axis=0, keepdims=True)
    qq = float(np.sum(template * template))
    if qq <= EPS:
        raise ValueError("template has zero norm")
    p = (float(np.sum(u * template)) / qq) * template
    r = u - p
    return {"mu": values.mean(axis=0), "U": u, "P": p, "R": r, "p_norm": float(np.linalg.norm(p)), "r_norm": float(np.linalg.norm(r))}


def _intervention_arrays(z_n: np.ndarray, z_t: np.ndarray, occurrences: Sequence[Mapping[str, Any]], templates: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Create N/T ID, R_DOWN and P_DOWN arrays and verify local identities."""
    z_n = np.asarray(z_n, dtype=np.float64)
    z_t = np.asarray(z_t, dtype=np.float64)
    if z_n.shape != z_t.shape or z_n.ndim != 2 or z_n.shape[1] != FEATURE_DIM:
        raise ProtocolError(f"natural and aligned TTS feature grids differ: {z_n.shape} vs {z_t.shape}")
    arrays = {
        "N_ID": z_n.copy(), "N_R_DOWN": z_n.copy(), "N_P_DOWN": z_n.copy(),
        "T_ID": z_t.copy(), "T_R_DOWN": z_t.copy(), "T_P_DOWN": z_t.copy(),
    }
    operations: list[dict[str, Any]] = []
    active_indices: set[int] = set()
    no_op_reasons: defaultdict[str, int] = defaultdict(int)
    max_errors = defaultdict(float)
    max_float32_errors = defaultdict(float)
    relative_changes: defaultdict[str, list[float]] = defaultdict(list)
    component_reductions: defaultdict[str, list[float]] = defaultdict(list)
    for occurrence in occurrences:
        indices = [int(i) for i in occurrence.get("natural_frame_indices", [])]
        label = str(occurrence.get("label", ""))
        if occurrence.get("eligible") is not True or len(indices) < MIN_OCCURRENCE_FRAMES or label not in templates:
            reason = "ineligible" if occurrence.get("eligible") is not True else "no_template"
            no_op_reasons[reason] += 1
            operations.append({"occurrence_index": int(occurrence.get("occurrence_index", -1)), "label": label, "frame_indices": indices, "internal_indices": [], "active": False, "reason": reason})
            continue
        internal = indices[EDGE_FRAMES:-EDGE_FRAMES]
        if len(internal) < MIN_INTERNAL_FRAMES:
            no_op_reasons["internal_short"] += 1
            operations.append({"occurrence_index": int(occurrence.get("occurrence_index", -1)), "label": label, "frame_indices": indices, "internal_indices": internal, "active": False, "reason": "internal_short"})
            continue
        q = _template_to_length(templates[label], len(internal))
        n_comp = _project_components(z_n[internal], q)
        t_comp = _project_components(z_t[internal], q)
        norms = [float(n_comp["p_norm"]), float(n_comp["r_norm"]), float(t_comp["p_norm"]), float(t_comp["r_norm"])]
        if any(value <= EPS for value in norms):
            no_op_reasons["zero_component"] += 1
            operations.append({"occurrence_index": int(occurrence.get("occurrence_index", -1)), "label": label, "frame_indices": indices, "internal_indices": internal, "active": False, "reason": "zero_component", "norms": norms})
            continue
        delta = 0.5 * min(norms)
        n_mu, t_mu = np.asarray(n_comp["mu"]), np.asarray(t_comp["mu"])
        n_p, n_r = np.asarray(n_comp["P"]), np.asarray(n_comp["R"])
        t_p, t_r = np.asarray(t_comp["P"]), np.asarray(t_comp["R"])
        arrays["N_R_DOWN"][internal] = n_mu + n_p + (1.0 - delta / norms[1]) * n_r
        arrays["N_P_DOWN"][internal] = n_mu + (1.0 - delta / norms[0]) * n_p + n_r
        arrays["T_R_DOWN"][internal] = t_mu + t_p + (1.0 - delta / norms[3]) * t_r
        arrays["T_P_DOWN"][internal] = t_mu + (1.0 - delta / norms[2]) * t_p + t_r
        source_by_arm = {"N_R_DOWN": z_n, "N_P_DOWN": z_n, "T_R_DOWN": z_t, "T_P_DOWN": z_t}
        component_by_arm = {"N_R_DOWN": norms[1], "N_P_DOWN": norms[0], "T_R_DOWN": norms[3], "T_P_DOWN": norms[2]}
        relative_change: dict[str, float] = {}
        for arm, source in source_by_arm.items():
            source_norm = max(float(np.linalg.norm(source[indices])), EPS)
            relative_change[arm] = float(delta / source_norm)
            relative_changes[arm].append(relative_change[arm])
            component_key = "P_N" if arm == "N_P_DOWN" else "R_N" if arm == "N_R_DOWN" else "P_T" if arm == "T_P_DOWN" else "R_T"
            component_reductions[component_key].append(float(delta / component_by_arm[arm]))
        active_indices.update(internal)
        operation = {
            "occurrence_index": int(occurrence.get("occurrence_index", -1)), "label": label,
            "frame_indices": indices, "internal_indices": internal, "active": True, "reason": "active",
            "delta": float(delta), "norms": norms,
            "p_norm_n": float(n_comp["p_norm"]), "r_norm_n": float(n_comp["r_norm"]),
            "p_norm_t": float(t_comp["p_norm"]), "r_norm_t": float(t_comp["r_norm"]),
            "projection_coeff_n": float(np.sum(n_comp["U"] * q) / np.sum(q * q)),
            "projection_coeff_t": float(np.sum(t_comp["U"] * q) / np.sum(q * q)),
            "relative_change": relative_change,
            "component_reduction_fraction": {key: float(delta / value) for key, value in component_by_arm.items()},
        }
        operations.append(operation)
        for arm, source, comp in (("N_R_DOWN", z_n, n_comp["R"]), ("N_P_DOWN", z_n, n_comp["P"]), ("T_R_DOWN", z_t, t_comp["R"]), ("T_P_DOWN", z_t, t_comp["P"])):
            difference = arrays[arm][internal] - source[internal]
            max_errors[f"norm_{arm}"] = max(max_errors[f"norm_{arm}"], abs(float(np.linalg.norm(difference)) - delta))
            max_errors[f"mean_{arm}"] = max(max_errors[f"mean_{arm}"], float(np.max(np.abs(difference.mean(axis=0)))))
            difference32 = arrays[arm].astype(np.float32)[internal] - source.astype(np.float32)[internal]
            max_float32_errors[f"relative_norm_{arm}"] = max(max_float32_errors[f"relative_norm_{arm}"], abs(float(np.linalg.norm(difference32)) - delta) / max(delta, EPS))
            max_float32_errors[f"absolute_mean_{arm}"] = max(max_float32_errors[f"absolute_mean_{arm}"], float(np.max(np.abs(difference32.mean(axis=0)))))
    active_count = sum(bool(row.get("active")) for row in operations)
    active_frames = len(active_indices)
    non_speech = sum(1 for row in occurrences for _ in row.get("natural_frame_indices", []) if not row.get("is_silence", False))
    metadata = {
        "active_occurrence_count": active_count,
        "active_internal_frame_count": active_frames,
        "non_speech_frame_count": int(non_speech),
        "active_internal_frame_fraction": float(active_frames / non_speech) if non_speech else 0.0,
        "coverage_pass": bool(active_count >= MIN_ACTIVE_OCCURRENCES and (active_frames / non_speech if non_speech else 0.0) >= MIN_ACTIVE_FRACTION),
        "no_op_reasons": dict(no_op_reasons),
        "max_float64_error": dict(max_errors),
        "max_float32_error": dict(max_float32_errors),
        "relative_change": {
            arm: {"values": values, "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}
            for arm, values in sorted(relative_changes.items())
            if values
        },
        "component_reduction_fraction": {
            component: {"values": values, "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}
            for component, values in sorted(component_reductions.items())
            if values
        },
        "relative_change_denominator": "source_occurrence_frobenius_norm",
    }
    return arrays, operations, metadata


def _template_build(rows: Sequence[Mapping[str, Any]], raw_features: Mapping[int, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build N/T-shared donor templates with donor sentence equal weights."""
    by_phone_donor: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    occurrence_meta: list[dict[str, Any]] = []
    for row in rows:
        sid = int(row["sample_id"])
        z_n = np.asarray(row["z_n"], dtype=np.float64)
        z_t = np.asarray(raw_features[sid], dtype=np.float64)
        occurrences = _token_occurrences(row["natural_tokens"], row["tts_tokens"], len(z_n), len(z_t))
        for occurrence in occurrences:
            if occurrence["eligible"] is not True:
                continue
            nidx = occurrence["natural_frame_indices"][EDGE_FRAMES:-EDGE_FRAMES]
            tidx = occurrence["tts_frame_indices"][EDGE_FRAMES:-EDGE_FRAMES]
            n_shape = _center_and_interpolate(z_n[nidx])
            t_shape = _center_and_interpolate(z_t[tidx])
            n_norm, t_norm = float(np.linalg.norm(n_shape)), float(np.linalg.norm(t_shape))
            if n_norm <= EPS or t_norm <= EPS:
                continue
            shape = (n_shape / n_norm + t_shape / t_norm) / 2.0
            by_phone_donor[str(occurrence["label"])][str(row["paired_key"])].append(shape)
            occurrence_meta.append({"sample_id": sid, "paired_key": str(row["paired_key"]), "occurrence_index": occurrence["occurrence_index"], "label": occurrence["label"], "n_frames": len(nidx), "t_frames": len(tidx), "n_norm": n_norm, "t_norm": t_norm})
    templates: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for label in sorted(by_phone_donor):
        donor_values = by_phone_donor[label]
        donor_keys = sorted(donor_values)
        if len(donor_keys) < 2:
            continue
        per_donor = [np.mean(np.stack(donor_values[key], axis=0), axis=0) for key in donor_keys]
        q = np.mean(np.stack(per_donor, axis=0), axis=0)
        q -= q.mean(axis=0, keepdims=True)
        norm = float(np.linalg.norm(q))
        if norm <= EPS:
            continue
        templates[label] = q / norm
        details[label] = {"donor_paired_keys": donor_keys, "donor_occurrence_count": sum(len(donor_values[key]) for key in donor_keys), "template_norm": norm}
    return templates, {"templates": details, "occurrences": occurrence_meta, "donor_ids": list(DONOR_IDS)}


def _semantic_and_source_diagnostics(rows: Sequence[Mapping[str, Any]], templates: Mapping[str, np.ndarray], raw_features: Mapping[int, np.ndarray]) -> dict[int, dict[str, Any]]:
    labels = sorted(templates)
    wrong = {label: labels[(index + 1) % len(labels)] for index, label in enumerate(labels)} if len(labels) >= 2 else {}
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sample_id"])
        z_n = np.asarray(row["z_n"], dtype=np.float64)
        z_t_raw = np.asarray(raw_features[sid], dtype=np.float64)
        z_t_aligned = np.asarray(row["z_t"], dtype=np.float64)
        occurrences = _token_occurrences(row["natural_tokens"], row["tts_tokens"], len(z_n), len(z_t_raw))
        semantic: list[float] = []
        f_n: list[float] = []
        f_t: list[float] = []
        f_t_aligned: list[float] = []
        native_energy: dict[str, list[dict[str, float]]] = {"N": [], "T_RAW": []}
        for occurrence in occurrences:
            label = str(occurrence["label"])
            if occurrence["eligible"] is not True or label not in templates or label not in wrong:
                continue
            nidx = occurrence["natural_frame_indices"][EDGE_FRAMES:-EDGE_FRAMES]
            tidx = occurrence["tts_frame_indices"][EDGE_FRAMES:-EDGE_FRAMES]
            n_u = _center_and_interpolate(z_n[nidx])
            t_u = _center_and_interpolate(z_t_raw[tidx])
            aligned_u = _center_and_interpolate(z_t_aligned[nidx])
            true_q = templates[label]
            wrong_q = templates[wrong[label]]
            n_true, n_wrong = _explained(n_u, true_q), _explained(n_u, wrong_q)
            t_true, t_wrong = _explained(t_u, true_q), _explained(t_u, wrong_q)
            aligned_true = _explained(aligned_u, true_q)
            if n_true is not None and n_wrong is not None and t_true is not None and t_wrong is not None:
                n_comp = _project_components(n_u, true_q)
                t_comp = _project_components(t_u, true_q)
                native_energy["N"].append({"p_energy_per_element": float(float(n_comp["p_norm"]) ** 2 / n_u.size), "r_energy_per_element": float(float(n_comp["r_norm"]) ** 2 / n_u.size), "u_energy_per_element": float(float(np.linalg.norm(n_comp["U"])) ** 2 / n_u.size), "projection_coefficient": float(np.sum(n_comp["U"] * true_q) / np.sum(true_q * true_q))})
                native_energy["T_RAW"].append({"p_energy_per_element": float(float(t_comp["p_norm"]) ** 2 / t_u.size), "r_energy_per_element": float(float(t_comp["r_norm"]) ** 2 / t_u.size), "u_energy_per_element": float(float(np.linalg.norm(t_comp["U"])) ** 2 / t_u.size), "projection_coefficient": float(np.sum(t_comp["U"] * true_q) / np.sum(true_q * true_q))})
                semantic.append(float((n_true - n_wrong + t_true - t_wrong) / 2.0))
                f_n.append(1.0 - n_true)
                f_t.append(1.0 - t_true)
                if aligned_true is not None:
                    f_t_aligned.append(1.0 - aligned_true)
        def energy_summary(values: Sequence[Mapping[str, float]]) -> dict[str, Any]:
            if not values:
                return {"count": 0, "reason": "MISSING_VALID_OCCURRENCE"}
            result_summary: dict[str, Any] = {"count": len(values)}
            for field in ("p_energy_per_element", "r_energy_per_element", "u_energy_per_element", "projection_coefficient"):
                entries = [float(item[field]) for item in values]
                result_summary[field] = {"values": entries, "mean": float(np.mean(entries)), "min": float(np.min(entries)), "max": float(np.max(entries))}
            coefficients = np.asarray([float(item["projection_coefficient"]) for item in values], dtype=np.float64)
            result_summary["projection_coefficient_sign"] = {"positive_count": int(np.sum(coefficients > 0)), "negative_count": int(np.sum(coefficients < 0)), "zero_count": int(np.sum(coefficients == 0)), "positive_fraction": float(np.mean(coefficients > 0)), "negative_fraction": float(np.mean(coefficients < 0))}
            return result_summary

        result[sid] = {
            "semantic_A": float(np.mean(semantic)) if semantic else None,
            "f_R_N": float(np.mean(f_n)) if f_n else None,
            "f_R_T_RAW": float(np.mean(f_t)) if f_t else None,
            "f_R_T_ALIGNED": float(np.mean(f_t_aligned)) if f_t_aligned else None,
            "valid_occurrence_count": len(semantic),
            "wrong_label_map": wrong,
            "native_energy": {source: energy_summary(values) for source, values in native_energy.items()},
        }
    return result


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _parent_rows(parent: Path) -> list[dict[str, Any]]:
    inputs_path = parent / "inputs.json"
    manifest_path = parent / "audio_manifest.json"
    if not inputs_path.is_file() or not manifest_path.is_file():
        raise ProtocolError(f"parent run lacks inputs.json/audio_manifest.json: {parent}")
    inputs, manifest = read_json(inputs_path), read_json(manifest_path)
    if inputs.get("status") != "complete" or manifest.get("status") != "complete":
        raise ProtocolError("parent inputs and audio manifest must be complete")
    records = list(inputs.get("records", []))
    parent_models = inputs.get("models", {})
    if not isinstance(parent_models, Mapping):
        raise ProtocolError("parent inputs.json has no model binding")
    manifest_by_id = {int(row["sample_id"]): row for row in manifest.get("records", [])}
    record_ids = [int(row["sample_id"]) for row in records]
    if record_ids != list(range(1, 16)) or set(manifest_by_id) != set(range(1, 16)):
        raise ProtocolError("parent must contain exactly sample IDs 1..15")
    output: list[dict[str, Any]] = []
    # The protocol binds donor/evaluation roles to the parent's ordered
    # records.  Keep that order and reject a reordered parent rather than
    # silently recovering by sample_id.
    for source in records:
        sid = int(source["sample_id"])
        parent_audio = manifest_by_id[sid]
        z_n_path = _resolve_asset(parent_audio["z_n_path"])
        z_t_path = _resolve_asset(parent_audio["z_t_path"])
        occ_path = _resolve_asset(parent_audio["occurrences_path"])
        natural_audio = _resolve_asset(source["natural_audio"], expected_sha256=str(source["natural_audio_sha256"]))
        tts_audio = _resolve_asset(source["tts_audio"], expected_sha256=str(source["tts_audio_sha256"]))
        face_video = _resolve_asset(source["face_video"], expected_sha256=str(source["face_video_sha256"]))
        z_n = _load_features(z_n_path)
        z_t = _load_features(z_t_path)
        if z_n.shape != z_t.shape or list(z_n.shape) != list(source.get("historical_conditioning_shape", [])):
            raise ProtocolError(f"parent natural/aligned TTS shape mismatch for sample {sid}: {z_n.shape}, {z_t.shape}")
        # The parent manifest carries the exact conditioning files that were
        # fed to the frozen vocoder.  Bind their contents to Z_N/Z_T so a
        # stale or substituted feature file cannot silently enter this run.
        try:
            import torch

            conditions = parent_audio.get("conditions", {})
            for condition_key, expected_values in (("N_100", z_n), ("T_100", z_t)):
                condition = conditions.get(condition_key)
                if not isinstance(condition, Mapping):
                    raise ProtocolError(f"parent condition {condition_key} is missing for sample {sid}")
                condition_path = _resolve_asset(condition["conditioning_path"], expected_sha256=str(condition["conditioning_sha256"]))
                payload = torch.load(condition_path, map_location="cpu", weights_only=False)
                if not isinstance(payload, Mapping) or "conditioning" not in payload:
                    raise ProtocolError(f"parent condition payload is malformed: {condition_path}")
                condition_values = np.asarray(torch.as_tensor(payload["conditioning"], dtype=torch.float32).cpu(), dtype=np.float32)
                if condition_values.shape != expected_values.shape or not np.array_equal(condition_values, expected_values):
                    raise ProtocolError(f"parent {condition_key} conditioning differs from Z for sample {sid}")
        except KeyError as exc:
            raise ProtocolError(f"parent condition binding is incomplete for sample {sid}: {exc}") from exc
        occurrences = read_json(occ_path)
        if not isinstance(occurrences, Mapping) or not isinstance(occurrences.get("occurrences"), list):
            raise ProtocolError(f"parent occurrence mask malformed: {occ_path}")
        score_box = source.get("score_box")
        if not isinstance(score_box, Mapping) or canonical_sha256(score_box) != source.get("score_box_sha256"):
            raise ProtocolError(f"parent score box hash mismatch for sample {sid}")
        score_box_source = source.get("score_box_path")
        if score_box_source:
            score_box_file = _resolve_asset(score_box_source)
            stored_box = read_json(score_box_file)
            if any(stored_box.get(key) != score_box.get(key) for key in ("order", "center", "side", "box", "scale_from_detection")):
                raise ProtocolError(f"parent score box contents differ for sample {sid}: {score_box_file}")
        output.append({
            "sample_id": sid, "paired_key": str(source["paired_key"]), "speaker_id": str(source["speaker_id"]), "split": str(source["split"]), "transcript": str(source.get("transcript", "")),
            "natural_audio": str(natural_audio), "natural_audio_sha256": str(source["natural_audio_sha256"]), "tts_audio": str(tts_audio), "tts_audio_sha256": str(source["tts_audio_sha256"]),
            "face_video": str(face_video), "face_video_sha256": str(source["face_video_sha256"]), "score_box": dict(score_box), "score_box_sha256": str(source["score_box_sha256"]),
            "score_box_source": str(source.get("score_box_path", "")), "natural_tokens": list(source["natural_tokens"]), "tts_tokens": list(source["tts_tokens"]),
            "z_n_path": str(z_n_path), "z_n_sha256": file_sha256(z_n_path), "z_t_path": str(z_t_path), "z_t_sha256": file_sha256(z_t_path), "occurrences_path": str(occ_path), "occurrences_sha256": file_sha256(occ_path),
            "z_n": z_n, "z_t": z_t, "parent_occurrences": list(occurrences["occurrences"]),
        })
    return output


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ProtocolError(f"output exists; use --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parent = args.parent.resolve()
    rows = _parent_rows(parent)
    if {row["paired_key"] for row in rows[:5]} & {row["paired_key"] for row in rows[5:]}:
        raise ProtocolError("donor/evaluation paired_key leakage")
    inputs_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        score_box_path = output / "inputs" / f"score_box_{index:04d}.json"
        write_json(score_box_path, {"sample_id": index, "paired_key": row["paired_key"], **row["score_box"]})
        inputs_records.append({key: value for key, value in row.items() if key not in {"z_n", "z_t", "parent_occurrences"}} | {"role": "donor" if index <= 5 else "evaluation", "score_box_path": str(score_box_path), "score_box_sha256": row["score_box_sha256"]})
    parent_inputs = read_json(parent / "inputs.json")
    models = dict(parent_inputs.get("models", {}))
    inputs = {
        "schema_version": 1, "status": "complete", "protocol": PROTOCOL, "run_id": output.name, "sample_count": 15, "donor_sample_ids": list(DONOR_IDS), "evaluation_sample_ids": list(EVALUATION_IDS),
        "arms": list(ARMS), "resynthesis_arms": list(RESYNTH_ARMS), "sample_rate": SAMPLE_RATE, "frame_stride_samples": FRAME_STRIDE, "feature_dim": FEATURE_DIM, "template_points": TEMPLATE_POINTS,
        "min_occurrence_frames": MIN_OCCURRENCE_FRAMES, "edge_frames": EDGE_FRAMES, "min_internal_frames": MIN_INTERNAL_FRAMES, "min_active_occurrences": MIN_ACTIVE_OCCURRENCES, "min_active_fraction": MIN_ACTIVE_FRACTION,
        "score_lag_range": list(range(-VSHIFT, VSHIFT + 1)), "min_common_windows": MIN_COMMON_WINDOWS, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "bonferroni_alpha": BONFERRONI_ALPHA,
        "parent": {"run": str(parent), "inputs_sha256": file_sha256(parent / "inputs.json"), "audio_manifest_sha256": file_sha256(parent / "audio_manifest.json"), "protocol": parent_inputs.get("protocol")},
        "models": models, "spec_path": str(SPEC_PATH), "spec_sha256": file_sha256(SPEC_PATH) if SPEC_PATH.is_file() else None, "runtime": {"python": platform.python_version(), "git_commit": _git_commit(), "argv": sys.argv}, "records": inputs_records,
    }
    write_json(output / "inputs.json", inputs)
    return inputs


def _load_run_inputs(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = read_json(output / "inputs.json")
    if inputs.get("protocol") != PROTOCOL or inputs.get("status") != "complete":
        raise ProtocolError("inputs.json is not a complete acoustic-reorganization run")
    rows = list(inputs.get("records", []))
    if len(rows) != 15 or [int(row.get("sample_id", -1)) for row in rows] != list(range(1, 16)):
        raise ProtocolError("inputs.json must contain ordered sample IDs 1..15")
    expected_roles = ["donor"] * len(DONOR_IDS) + ["evaluation"] * len(EVALUATION_IDS)
    if [str(row.get("role")) for row in rows] != expected_roles:
        raise ProtocolError("inputs.json donor/evaluation roles are not in the frozen order")
    return inputs, rows


def _eval_rows(rows: Sequence[Mapping[str, Any]], smoke: bool) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if str(row.get("role")) == "evaluation"]
    if len(selected) != 10:
        raise ProtocolError("inputs.json must contain exactly 10 evaluation records")
    return selected[:1] if smoke else selected


def features_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output = args.output_dir.resolve()
    inputs, _rows = _load_run_inputs(output)
    parent = Path(str(inputs["parent"]["run"]))
    parent_rows = _parent_rows(parent)
    by_id = {int(row["sample_id"]): row for row in parent_rows}
    try:
        from wavlm_knn_vc_adapter import KNN_VC_REVISION, WavLMKNNVCAdapter
    except Exception as exc:  # pragma: no cover
        raise ProtocolError(f"cannot import frozen WavLM adapter: {exc}") from exc
    expected_revision = inputs.get("models", {}).get("knn_vc_revision")
    if expected_revision and expected_revision != KNN_VC_REVISION:
        raise ProtocolError("parent kNN-VC revision is not the pinned revision")
    adapter = WavLMKNNVCAdapter.load_pretrained(device=args.device, source=args.knn_source.resolve(), revision=KNN_VC_REVISION)
    model_meta = adapter.metadata()
    for field in ("wavlm_checkpoint_sha256", "vocoder_checkpoint_sha256"):
        expected = inputs.get("models", {}).get(field)
        if expected and model_meta.get(field) != expected:
            raise ProtocolError(f"{field} changed: {model_meta.get(field)} != {expected}")
    raw_features: dict[int, np.ndarray] = {}
    for row in parent_rows:
        audio, _ = _audio(Path(row["tts_audio"]))
        raw_features[int(row["sample_id"])] = np.asarray(adapter.extract(torch.from_numpy(audio).unsqueeze(0)).cpu(), dtype=np.float32)
    donor_templates, template_meta = _template_build([by_id[sid] for sid in DONOR_IDS], raw_features)
    if not donor_templates:
        raise ProtocolError("no donor phone has at least two valid paired donor sentences")
    labels = sorted(donor_templates)
    wrong = {label: labels[(index + 1) % len(labels)] for index, label in enumerate(labels)} if len(labels) >= 2 else {}
    template_arrays = np.stack([donor_templates[label] for label in labels], axis=0).astype(np.float64)
    templates_path = output / "templates.npz"
    np.savez_compressed(templates_path, labels=np.asarray(labels), templates=template_arrays)
    template_meta.update({"schema_version": 1, "protocol": PROTOCOL, "labels": labels, "wrong_label_map": wrong, "sha256": file_sha256(templates_path)})
    write_json(output / "templates.json", template_meta)
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    diagnostics = _semantic_and_source_diagnostics([by_id[sid] for sid in EVALUATION_IDS], donor_templates, raw_features)
    failures: list[dict[str, Any]] = []
    for sid in EVALUATION_IDS:
        row = by_id[sid]
        occurrences = _parent_occurrences(row["parent_occurrences"], row["natural_tokens"], row["tts_tokens"], len(row["z_n"]))
        arrays, operations, coverage = _intervention_arrays(row["z_n"], row["z_t"], occurrences, donor_templates)
        source_diag = diagnostics[sid]
        record_meta = {"sample_id": sid, "paired_key": row["paired_key"], "occurrences": occurrences, "operations": operations, "coverage": coverage, "source_diagnostic": source_diag, "arrays": {key: list(value.shape) for key, value in arrays.items()}, "dtype": "float64"}
        feature_path = feature_dir / f"{row['paired_key']}__interventions.npz"
        _save_feature_file(feature_path, {"Z_N": np.asarray(row["z_n"], dtype=np.float64), "Z_T": np.asarray(row["z_t"], dtype=np.float64), "Z_T_RAW": np.asarray(raw_features[sid], dtype=np.float64), **arrays}, record_meta)
        record_meta["feature_path"] = str(feature_path)
        record_meta["feature_sha256"] = file_sha256(feature_path)
        manifest_records.append(record_meta)
        for source, arm_group in (("N", N_ARMS), ("T", T_ARMS)):
            for arm in arm_group:
                rel = coverage.get("relative_change", {}).get(arm, {})
                component = "P_N" if arm == "N_P_DOWN" else "R_N" if arm == "N_R_DOWN" else "P_T" if arm == "T_P_DOWN" else "R_T" if arm == "T_R_DOWN" else ""
                reduction = coverage.get("component_reduction_fraction", {}).get(component, {})
                qc_rows.append({"sample_id": sid, "paired_key": row["paired_key"], "source": source, "arm": arm, **coverage, "feature_rms": float(np.sqrt(np.mean(np.square(arrays[arm] - arrays[f"{source}_ID"])))), "relative_change_mean": rel.get("mean"), "relative_change_min": rel.get("min"), "relative_change_max": rel.get("max"), "component_reduction_mean": reduction.get("mean"), "component_reduction_min": reduction.get("min"), "component_reduction_max": reduction.get("max"), "source_f_R": source_diag.get("f_R_N" if source == "N" else "f_R_T_RAW")})
        if not coverage["coverage_pass"]:
            failures.append({"sample_id": sid, "reason": "INSUFFICIENT_INTERVENTION_COVERAGE", "coverage": coverage})
    status = "complete" if not failures else "incomplete"
    feature_manifest = {"schema_version": 1, "status": status, "protocol": PROTOCOL, "sample_count": 10, "expected_records": 10, "records": manifest_records, "failures": failures, "templates": {"path": str(templates_path), "sha256": file_sha256(templates_path), "label_count": len(labels), "donor_ids": list(DONOR_IDS)}, "model": model_meta}
    write_json(output / "feature_manifest.json", feature_manifest)
    fields = sorted({key for row in qc_rows for key in row})
    with (output / "feature_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(qc_rows)
    if status != "complete":
        raise ProtocolError(f"feature stage incomplete: {failures}")
    return feature_manifest


def _feature_manifest(output: Path) -> dict[str, Any]:
    manifest = read_json(output / "feature_manifest.json")
    if manifest.get("status") != "complete" or len(manifest.get("records", [])) != 10 or manifest.get("failures"):
        raise ProtocolError("feature_manifest.json is incomplete")
    return manifest


def _load_feature_record(record: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve_asset(record["feature_path"], expected_sha256=str(record["feature_sha256"]))
    arrays = {key: np.asarray(value) for key, value in np.load(path, allow_pickle=False).items()}
    metadata = read_json(path.with_suffix(".json"))
    return {"path": path, "arrays": arrays, "metadata": metadata}


def _log_mel(values: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Small dependency-free 40-bin log-mel representation for audio QC."""
    values = np.asarray(values, dtype=np.float64)
    frame = round(0.025 * rate); hop = round(0.010 * rate); nfft = 512
    if values.size < frame:
        values = np.pad(values, (0, frame - values.size))
    count = 1 + max(0, (values.size - frame) // hop)
    window = np.hanning(frame)
    spectrum = []
    for index in range(count):
        chunk = values[index * hop:index * hop + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        spectrum.append(np.abs(np.fft.rfft(chunk * window, n=nfft)) ** 2)
    power = np.asarray(spectrum, dtype=np.float64)
    hz = np.fft.rfftfreq(nfft, 1.0 / rate)
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    inv_mel = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = inv_mel(np.linspace(mel(0.0), mel(rate / 2.0), 42))
    filters = np.zeros((40, len(hz)), dtype=np.float64)
    for band in range(40):
        left, center, right = edges[band:band + 3]
        up = (hz - left) / max(center - left, 1e-9); down = (right - hz) / max(right - center, 1e-9)
        filters[band] = np.maximum(0.0, np.minimum(up, down))
    return np.log(np.maximum(power @ filters.T, 1e-12))


def _exact_length(values: np.ndarray, target: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    raw = int(values.size)
    if raw > target:
        return values[:target].copy(), {"action": "right_crop", "raw_decoder_sample_count": raw, "target_natural_sample_count": target, "adjustment_sample_count": raw - target}
    if raw < target:
        return np.pad(values, (0, target - raw)).astype(np.float32), {"action": "right_zero_pad", "raw_decoder_sample_count": raw, "target_natural_sample_count": target, "adjustment_sample_count": target - raw}
    return values.copy(), {"action": "none", "raw_decoder_sample_count": raw, "target_natural_sample_count": target, "adjustment_sample_count": 0}


def _write_generated_audio(float_path: Path, pcm_path: Path, values: np.ndarray, target: int, reference: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all() or np.max(np.abs(values), initial=0.0) > 1.0:
        raise ProtocolError("vocoder waveform is non-finite or outside [-1,1]; clipping is forbidden")
    adjusted, adjustment = _exact_length(values, target)
    if not np.isfinite(adjusted).all() or np.max(np.abs(adjusted), initial=0.0) > 1.0:
        raise ProtocolError("length-adjusted waveform is invalid")
    float_path.parent.mkdir(parents=True, exist_ok=True)
    pcm_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(float_path), adjusted, SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(pcm_path), adjusted, SAMPLE_RATE, subtype="PCM_16")
    decoded, rate = _audio(pcm_path)
    if rate != SAMPLE_RATE or decoded.size != target:
        raise ProtocolError(f"PCM16 output length/rate mismatch: {pcm_path}")
    qc = {"sample_count": int(decoded.size), "sample_rate": rate, "channels": 1, "finite": True, "peak": float(np.max(np.abs(decoded), initial=0.0)), "rms": float(np.sqrt(np.mean(np.square(decoded)))) if decoded.size else 0.0, "clipping_sample_count": int(np.sum(np.abs(decoded) >= 0.999969)), **adjustment}
    if reference is not None:
        lhs, rhs = _log_mel(decoded), _log_mel(reference)
        count = min(len(lhs), len(rhs))
        qc["log_mel_distance_to_id"] = float(np.sqrt(np.mean(np.square(lhs[:count] - rhs[:count])))) if count else None
    else:
        qc["log_mel_distance_to_id"] = None
    return qc


def audio_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output = args.output_dir.resolve(); inputs, rows = _load_run_inputs(output); feature_manifest = _feature_manifest(output)
    selected = _eval_rows(rows, args.smoke)
    feature_by_id = {int(record["sample_id"]): record for record in feature_manifest["records"]}
    from wavlm_knn_vc_adapter import KNN_VC_REVISION, WavLMKNNVCAdapter
    adapter = WavLMKNNVCAdapter.load_pretrained(device=args.device, source=args.knn_source.resolve(), revision=KNN_VC_REVISION)
    model_meta = adapter.metadata()
    for field in ("wavlm_checkpoint_sha256", "vocoder_checkpoint_sha256"):
        expected = inputs.get("models", {}).get(field)
        if expected and model_meta.get(field) != expected:
            raise ProtocolError(f"{field} changed")
    # A failed GPU process can leave a valid prefix.  With --resume we retain
    # only rows whose files and hashes still pass the PCM contract, then decode
    # the missing arms.  This keeps a device failure from forcing a second pass
    # over already verified audio.
    existing_rows: dict[tuple[int, str], dict[str, Any]] = {}
    existing_path = output / "audio_manifest.json"
    if args.resume and existing_path.is_file():
        try:
            old = read_json(existing_path)
            for old_record in old.get("records", []):
                for old_row in old_record.get("audio", []):
                    key = (int(old_row["sample_id"]), str(old_row["arm"]))
                    pcm = Path(str(old_row.get("audio_pcm16", "")))
                    if pcm.is_file() and file_sha256(pcm) == str(old_row.get("audio_sha256")):
                        _audio(pcm)
                        existing_rows[key] = dict(old_row)
        except Exception:  # noqa: BLE001 - invalid prior manifest is discarded
            existing_rows = {}
    records: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    t_id_reference: dict[int, np.ndarray] = {}
    for row in selected:
        sid, key = int(row["sample_id"]), str(row["paired_key"]); natural, _ = _audio(Path(row["natural_audio"])); target = len(natural)
        feature = _load_feature_record(feature_by_id[sid]); arrays = feature["arrays"]
        for arm in ARMS:
            try:
                reusable = existing_rows.get((sid, arm))
                if reusable is not None:
                    records.append(reusable)
                    if arm == "T_ID" and Path(str(reusable.get("audio_pcm16", ""))).is_file():
                        t_id_reference[sid], _ = _audio(Path(str(reusable["audio_pcm16"])))
                    continue
                if arm == "N_RAW":
                    path = Path(row["natural_audio"]); values, _ = _audio(path); qc = {"sample_count": len(values), "sample_rate": SAMPLE_RATE, "channels": 1, "finite": True, "peak": float(np.max(np.abs(values), initial=0.0)), "rms": float(np.sqrt(np.mean(values * values))), "clipping_sample_count": int(np.sum(np.abs(values) >= 0.999969)), "action": "source_reuse", "raw_decoder_sample_count": len(values), "target_natural_sample_count": target, "adjustment_sample_count": 0, "log_mel_distance_to_id": None}; float_path = pcm_path = path; condition_path = None
                else:
                    condition = np.asarray(arrays[arm], dtype=np.float32)
                    condition_path = output / "features" / f"{key}__{arm}__conditioning.pt"
                    torch.save({"conditioning": torch.from_numpy(condition), "source": "N" if arm.startswith("N_") else "T", "arm": arm, "sample_id": sid, "paired_key": key}, condition_path)
                    generated = np.asarray(adapter.vocode(torch.from_numpy(condition)).cpu(), dtype=np.float32)
                    float_path = output / "audio" / f"{key}__{arm}__float.wav"; pcm_path = output / "audio" / f"{key}__{arm}.wav"
                    reference = natural if arm.startswith("N_") else None
                    if arm.startswith("T_"):
                        # ARMS is ordered with T_ID first.  Cache the actual
                        # T_ID decode (or a verified resumed file) so QC for
                        # T_R_DOWN/T_P_DOWN never triggers a second baseline
                        # vocoder call.
                        if arm == "T_ID":
                            t_id_reference[sid], _ = _exact_length(generated, target)
                        elif sid not in t_id_reference:
                            resumed = existing_rows.get((sid, "T_ID"))
                            if resumed is not None and Path(str(resumed.get("audio_pcm16", ""))).is_file():
                                t_id_reference[sid], _ = _audio(Path(str(resumed["audio_pcm16"])))
                            else:
                                id_condition = np.asarray(arrays["T_ID"], dtype=np.float32)
                                id_wave = np.asarray(adapter.vocode(torch.from_numpy(id_condition)).cpu(), dtype=np.float32)
                                t_id_reference[sid], _ = _exact_length(id_wave, target)
                        reference = t_id_reference[sid]
                    qc = _write_generated_audio(float_path, pcm_path, generated, target, reference)
                records.append({"sample_id": sid, "paired_key": key, "arm": arm, "audio_float": str(float_path), "audio_pcm16": str(pcm_path), "audio_sha256": file_sha256(pcm_path), "audio_float_sha256": file_sha256(float_path), "conditioning_path": str(condition_path) if condition_path else None, "conditioning_sha256": file_sha256(condition_path) if condition_path else None, "source": "N_RAW" if arm == "N_RAW" else ("N" if arm.startswith("N_") else "T"), "qc": qc, "sample_count": target})
            except Exception as exc:  # noqa: BLE001 - preserve per-cell failures
                failures.append({"sample_id": sid, "paired_key": key, "arm": arm, "stage": "audio", "error": str(exc)})
        print(f"audio {sid} arms={sum(r['sample_id'] == sid for r in records)}/7", flush=True)
    expected = len(selected) * len(ARMS); status = "complete" if not failures and len(records) == expected else "incomplete"
    manifest = {"schema_version": 1, "status": status, "protocol": PROTOCOL, "sample_count": len(selected), "expected_audio_rows": expected, "audio_rows": len(records), "arms": list(ARMS), "records": [{"sample_id": sid, "paired_key": next(row["paired_key"] for row in selected if int(row["sample_id"]) == sid), "audio": [r for r in records if int(r["sample_id"]) == sid]} for sid in [int(row["sample_id"]) for row in selected]], "failures": failures, "model": model_meta}
    write_json(output / "audio_manifest.json", manifest)
    fields = sorted({key for row in records for key in row} | {key for row in records for key in row.get("qc", {})})
    with (output / "audio_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records: writer.writerow({key: row.get(key, row.get("qc", {}).get(key, "")) for key in fields})
    if status != "complete":
        raise ProtocolError(f"audio stage incomplete: {failures}")
    return manifest


def _video_info(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open video: {path}")
    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok: break
            count += 1
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if count < 25 or abs(fps - FPS) > 0.01:
        raise ProtocolError(f"video support invalid: {path} frames={count} fps={fps}")
    return {"frame_count": count, "fps": fps, "width": width, "height": height}


def _wav2lip_box(score_box: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Convert the parent square crop to Wav2Lip's fixed box order.

    The parent run already froze a zero-padded square for SyncNet scoring.  We
    pass that same square to Wav2Lip as ``top,bottom,left,right``.  This keeps
    every arm on the parent crop and avoids a fresh, arm-dependent face
    detector pass (which is also important when the detector is unavailable on
    a CPU fallback).
    """
    try:
        x1, y1, x2, y2 = [int(value) for value in score_box["box"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"malformed parent score box: {score_box}") from exc
    if x2 <= x1 or y2 <= y1 or (x2 - x1) != (y2 - y1):
        raise ProtocolError(f"parent score box must be a positive square: {score_box}")
    if min(x1, y1) < 0:
        raise ProtocolError(f"parent score box cannot have negative coordinates: {score_box}")
    return y1, y2, x1, x2


def render_stage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve(); inputs, rows = _load_run_inputs(output); audio_manifest = read_json(output / "audio_manifest.json")
    if audio_manifest.get("status") != "complete": raise ProtocolError("render requires complete audio_manifest.json")
    selected = _eval_rows(rows, args.smoke); audio_by_key = {(str(r["paired_key"]), str(r["arm"])): r for rec in audio_manifest["records"] for r in rec["audio"]}
    # Keep the venv launcher path itself.  Resolving its symlink turns it into
    # /usr/bin/python3.12 and silently drops the Wav2Lip environment packages.
    python = Path(args.wav2lip_python).expanduser()
    if not python.is_file():
        raise ProtocolError(f"missing Wav2Lip Python: {python}")
    inference = _resolve_asset(args.wav2lip / "inference.py"); checkpoint = _resolve_asset(args.wav2lip_checkpoint)
    existing_manifest = read_json(output / "videos_manifest.json") if args.resume and (output / "videos_manifest.json").is_file() else {}
    existing_by_key_arm = {(str(item.get("paired_key")), str(item.get("arm"))): item for item in existing_manifest.get("videos", []) if isinstance(item, Mapping)}
    checkpoint_sha = file_sha256(checkpoint)
    expected_checkpoint_sha = inputs.get("models", {}).get("wav2lip_checkpoint_sha256")
    if expected_checkpoint_sha and checkpoint_sha != expected_checkpoint_sha:
        raise ProtocolError(f"Wav2Lip checkpoint hash differs from parent binding: {checkpoint_sha} != {expected_checkpoint_sha}")
    videos: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    for row in selected:
        sid, key = int(row["sample_id"]), str(row["paired_key"]); face = _resolve_asset(row["face_video"], expected_sha256=row["face_video_sha256"])
        fixed_box = list(_wav2lip_box(row["score_box"]))
        for arm in ARMS:
            try:
                audio_row = audio_by_key[(key, arm)]; video = output / "videos" / arm / f"{sid:04d}.mp4"; work = output / "render_work" / arm / str(sid); work.mkdir(parents=True, exist_ok=True); (work / "temp").mkdir(exist_ok=True); video.parent.mkdir(parents=True, exist_ok=True)
                log = output / "logs" / "render" / arm / f"{sid:04d}.log"; log.parent.mkdir(parents=True, exist_ok=True)
                parameters = {"face_det_batch_size": 4, "wav2lip_batch_size": 4, "nosmooth": True, "box": fixed_box, "box_order": "top,bottom,left,right", "cwd": str(work)}
                prior = existing_by_key_arm.get((key, arm), {})
                reusable = False
                if args.resume and video.is_file() and isinstance(prior, Mapping):
                    try:
                        reusable = (
                            str(prior.get("video")) == str(video)
                            and str(prior.get("audio_sha256")) == str(audio_row["audio_sha256"])
                            and str(prior.get("face_sha256")) == str(row["face_video_sha256"])
                            and str(prior.get("wav2lip_checkpoint_sha256")) == checkpoint_sha
                            and str(prior.get("video_sha256")) == file_sha256(video)
                            and prior.get("parameters", {}).get("box") == fixed_box
                            and prior.get("parameters", {}).get("box_order") == "top,bottom,left,right"
                            and _video_info(video)["frame_count"] >= 25
                        )
                    except Exception:  # noqa: BLE001 - invalid partial output is rerendered
                        reusable = False
                command = [str(python), str(inference), "--checkpoint_path", str(checkpoint), "--face", str(face), "--audio", str(audio_row["audio_pcm16"]), "--outfile", str(video), "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--box", *[str(value) for value in fixed_box], "--nosmooth"]
                if not reusable:
                    with log.open("w", encoding="utf-8") as handle:
                        result = subprocess.run(command, cwd=str(work), stdout=handle, stderr=subprocess.STDOUT, check=False)
                    if result.returncode != 0: raise ProtocolError(f"Wav2Lip returncode={result.returncode}; see {log}")
                info = _video_info(video)
                videos.append({"sample_id": sid, "paired_key": key, "arm": arm, "video": str(video), "video_sha256": file_sha256(video), "face": str(face), "face_sha256": row["face_video_sha256"], "audio": audio_row["audio_pcm16"], "audio_sha256": audio_row["audio_sha256"], "fps": info["fps"], "frame_count": info["frame_count"], "wav2lip_checkpoint_sha256": checkpoint_sha, "parameters": parameters, "reused": reusable})
            except Exception as exc:  # noqa: BLE001 - retain per-cell failures
                failures.append({"sample_id": sid, "paired_key": key, "arm": arm, "stage": "render", "error": str(exc)})
        print(f"render {sid} videos={sum(r['sample_id'] == sid for r in videos)}/7", flush=True)
    expected = len(selected) * len(ARMS); status = "complete" if not failures and len(videos) == expected else "incomplete"; manifest = {"schema_version": 1, "status": status, "protocol": PROTOCOL, "sample_count": len(selected), "expected_videos": expected, "videos": videos, "failures": failures, "wav2lip_checkpoint_sha256": checkpoint_sha, "render_parameters": {"face_box_source": "parent score_box", "box_order": "top,bottom,left,right", "face_det_batch_size": 4, "wav2lip_batch_size": 4, "nosmooth": True}}
    write_json(output / "videos_manifest.json", manifest)
    if status != "complete": raise ProtocolError(f"render stage incomplete: {failures}")
    return manifest


def _crop_video(path: Path, score_box: Mapping[str, Any]) -> np.ndarray:
    import cv2

    x1, y1, x2, y2 = [int(value) for value in score_box["box"]]; side = x2 - x1
    if side <= 0 or y2 - y1 != side: raise ProtocolError("score box must be a positive square")
    capture = cv2.VideoCapture(str(path)); frames: list[np.ndarray] = []
    if not capture.isOpened(): raise ProtocolError(f"cannot open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            canvas = np.zeros((side, side, 3), dtype=np.uint8); sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            if sx2 > sx1 and sy2 > sy1: canvas[sy1-y1:sy2-y1, sx1-x1:sx2-x1] = frame[sy1:sy2, sx1:sx2]
            frames.append(cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_LINEAR))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally: capture.release()
    if len(frames) < 25 or abs(fps - FPS) > 0.01: raise ProtocolError(f"invalid video scoring support: {path}")
    return np.stack(frames, axis=0)


def _sync_visual(cropped: np.ndarray, scorer: Any, torch: Any, *, device: str = "cuda") -> np.ndarray:
    count = cropped.shape[0] - 4; out: list[Any] = []
    for start in range(0, count, 20):
        seq = np.stack([cropped[index:index+5] for index in range(start, min(count, start + 20))], axis=0)
        batch = torch.from_numpy(np.transpose(seq, (0, 4, 1, 2, 3))).float().to(device)
        with torch.no_grad(): out.append(scorer.__S__.forward_lip(batch).detach().cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32)


def _sync_audio(path: Path, scorer: Any, torch: Any, *, device: str = "cuda") -> tuple[np.ndarray, int, int]:
    import python_speech_features
    from scipy.io import wavfile

    rate, values = wavfile.read(str(path)); values = np.asarray(values)
    if int(rate) != SAMPLE_RATE or values.ndim != 1 or values.dtype.kind not in "iu": raise ProtocolError(f"scoring input must be mono PCM16: {path}")
    mfcc = np.asarray(list(zip(*python_speech_features.mfcc(values, rate))), dtype=np.float32); count = (mfcc.shape[1] - 20) // 4 + 1
    if mfcc.ndim != 2 or mfcc.shape[0] != 13 or count <= 0: raise ProtocolError(f"invalid SyncNet MFCC shape: {path}")
    out: list[Any] = []
    for start in range(0, count, 20):
        chunks = [mfcc[:, index*4:index*4+20] for index in range(start, min(count, start+20))]
        batch = torch.from_numpy(np.asarray(chunks, dtype=np.float32))[:, None].to(device)
        with torch.no_grad(): out.append(scorer.__S__.forward_aud(batch).detach().cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32), int(count), int(values.size)


def _curve(visual: np.ndarray, audio: np.ndarray, t_indices: Sequence[int], *, vshift: int = VSHIFT) -> tuple[list[int], list[float]]:
    import torch

    if len(t_indices) < MIN_COMMON_WINDOWS: raise ProtocolError(f"common support {len(t_indices)} < {MIN_COMMON_WINDOWS}")
    lags = list(range(-vshift, vshift + 1)); left = torch.from_numpy(np.asarray(visual[list(t_indices)], dtype=np.float32)); values: list[float] = []
    for lag in lags:
        right = torch.from_numpy(np.asarray(audio[[index + lag for index in t_indices]], dtype=np.float32)); values.append(float(torch.nn.functional.pairwise_distance(left, right).mean().item()))
    return lags, values


def _curve_metrics(lags: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (2 * VSHIFT + 1,) or not np.isfinite(arr).all(): raise ProtocolError("SyncNet curve must have 31 finite values")
    minimum = float(np.min(arr)); index = int(np.flatnonzero(arr == minimum)[0]); return {"C": float(np.median(arr) - minimum), "D": minimum, "B": float(np.median(arr)), "k_star": int(lags[index]), "d_zero": float(arr[VSHIFT])}


def expected_cells(sample_ids: Sequence[int]) -> set[tuple[int, str, str]]:
    cells: set[tuple[int, str, str]] = set()
    for sid in sample_ids:
        sid = int(sid)
        cells.update((sid, arm, arm) for arm in ARMS)
        cells.update((sid, arm, "N_RAW") for arm in RESYNTH_ARMS)
        for source, ids in (("N", N_ARMS), ("T", T_ARMS)):
            identity = ids[0]
            cells.update((sid, arm, identity) for arm in ids[1:])
            cells.update((sid, identity, arm) for arm in ids[1:])
    return cells


def _score_cells_for_sample(row: Mapping[str, Any], audio_by: Mapping[tuple[str, str], Mapping[str, Any]], video_by: Mapping[tuple[str, str], Mapping[str, Any]], scorer: Any, torch: Any, *, device: str = "cuda") -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    key, sid = str(row["paired_key"]), int(row["sample_id"]); visual: dict[str, np.ndarray] = {}; audio: dict[str, np.ndarray] = {}; vcount: dict[str, int] = {}; acount: dict[str, int] = {}
    for arm in ARMS:
        visual[arm] = _sync_visual(_crop_video(Path(video_by[(key, arm)]["video"]), row["score_box"]), scorer, torch, device=device); vcount[arm] = len(visual[arm])
        audio[arm], acount[arm], _ = _sync_audio(Path(audio_by[(key, arm)]["audio_pcm16"]), scorer, torch, device=device)
    support = min(*vcount.values(), *acount.values()); t_indices = list(range(VSHIFT, support - VSHIFT));
    if len(t_indices) < MIN_COMMON_WINDOWS: raise ProtocolError(f"sample {sid} common support {len(t_indices)} < {MIN_COMMON_WINDOWS}")
    cells = sorted(expected_cells([sid]), key=lambda value: (value[1], value[2])); rows: list[dict[str, Any]] = []; curves: list[dict[str, Any]] = []
    for _, video_arm, audio_arm in cells:
        lags, values = _curve(visual[video_arm], audio[audio_arm], t_indices); metrics = _curve_metrics(lags, values); rows.append({"schema_version": 1, "status": "complete", "sample_id": sid, "paired_key": key, "video_arm": video_arm, "audio_arm": audio_arm, **metrics, "common_support_count": len(t_indices), "fixed_k_n": None, "d_fixed_natural_lag": None, "video": video_by[(key, video_arm)]["video"], "video_sha256": video_by[(key, video_arm)]["video_sha256"], "audio": audio_by[(key, audio_arm)]["audio_pcm16"], "audio_sha256": audio_by[(key, audio_arm)]["audio_sha256"], "score_box_sha256": row["score_box_sha256"], "syncnet_model_sha256": None})
        curves.append({"sample_id": sid, "paired_key": key, "video_arm": video_arm, "audio_arm": audio_arm, "lags": lags, "values": values, "common_support_count": len(t_indices)})
    k_n = int(next(r for r in rows if r["video_arm"] == "N_RAW" and r["audio_arm"] == "N_RAW")["k_star"])
    for score, curve in zip(rows, curves, strict=True):
        if score["audio_arm"] == "N_RAW": score["fixed_k_n"], score["d_fixed_natural_lag"] = k_n, float(score["values"][VSHIFT + k_n] if "values" in score else curve["values"][VSHIFT + k_n])
    return rows, curves, {"visual_count": vcount, "audio_count": acount, "common_support_count": len(t_indices), "k_n": k_n}


def score_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    device = str(args.device).lower()
    if device not in {"cpu", "cuda"}:
        raise ProtocolError(f"unsupported SyncNet device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("requested SyncNet CUDA device is unavailable; rerun with --device cpu")
    output = args.output_dir.resolve(); inputs, rows = _load_run_inputs(output); _feature_manifest(output)
    audio_manifest, video_manifest = read_json(output / "audio_manifest.json"), read_json(output / "videos_manifest.json")
    if audio_manifest.get("status") != "complete" or video_manifest.get("status") != "complete": raise ProtocolError("score requires complete audio/videos manifests")
    selected = _eval_rows(rows, args.smoke); audio_by = {(str(r["paired_key"]), str(r["arm"])): r for rec in audio_manifest["records"] for r in rec["audio"]}; video_by = {(str(r["paired_key"]), str(r["arm"])): r for r in video_manifest["videos"]}
    model = _resolve_asset(args.syncnet_model); sync_root = _resolve_asset(args.syncnet / "SyncNetInstance.py").parent
    model_sha = file_sha256(model)
    expected_model_sha = inputs.get("models", {}).get("syncnet_model_sha256")
    if expected_model_sha and model_sha != expected_model_sha:
        raise ProtocolError(f"SyncNet model hash differs from parent binding: {model_sha} != {expected_model_sha}")
    if str(sync_root) not in sys.path: sys.path.insert(0, str(sync_root))
    from SyncNetInstance import SyncNetInstance
    scorer = SyncNetInstance(device=device); scorer.loadParameters(str(model)); scorer.eval()
    scores: list[dict[str, Any]] = []; curves: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    for row in selected:
        try:
            sample_scores, sample_curves, _ = _score_cells_for_sample(row, audio_by, video_by, scorer, torch, device=device)
            for score, curve in zip(sample_scores, sample_curves, strict=True): score["syncnet_model_sha256"] = model_sha; scores.append(score); curves.append(curve)
        except Exception as exc:  # noqa: BLE001 - preserve score-stage failures
            failures.append({"sample_id": int(row["sample_id"]), "paired_key": row["paired_key"], "stage": "score", "error": str(exc)})
    expected = len(selected) * 21; status = "complete" if not failures and len(scores) == expected else "incomplete"
    # Keep CSV-facing SyncNet scalars at the project's three-decimal display
    # precision while retaining the unrounded values for downstream analysis
    # and audit.  Curves.json is always written at full float precision too.
    for score in scores:
        for field in ("C", "D", "B", "d_zero"):
            score[f"{field}_raw"] = float(score[field])
            score[field] = round(float(score[field]), 3)
    fields = sorted({key for row in scores for key in row})
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(scores)
    write_json(output / "curves.json", {"schema_version": 1, "protocol": PROTOCOL, "lags": list(range(-VSHIFT, VSHIFT + 1)), "curves": curves})
    controls = _controls(output, selected, audio_by, video_by, scorer, torch, device=device) if any(int(row["sample_id"]) == 6 for row in selected) else {"status": "NOT_RUN_SMOKE", "controls": []}
    write_json(output / "controls.json", controls)
    manifest = {"schema_version": 1, "status": status, "protocol": PROTOCOL, "sample_count": len(selected), "expected_cells": expected, "score_rows": len(scores), "curve_rows": len(curves), "failures": failures, "syncnet_model_sha256": model_sha, "scoring_device": device, "scoring_python": str(args.syncnet_python), "scoring_protocol": {"vshift": VSHIFT, "min_common_windows": MIN_COMMON_WINDOWS, "t_rule": "range(15,F-15)", "fixed_audio_frontend": True}}
    write_json(output / "scores_manifest.json", manifest)
    if status != "complete": raise ProtocolError(f"score stage incomplete: {failures}")
    return manifest


def _controls(output: Path, selected: Sequence[Mapping[str, Any]], audio_by: Mapping[tuple[str, str], Mapping[str, Any]], video_by: Mapping[tuple[str, str], Mapping[str, Any]], scorer: Any, torch: Any, *, device: str = "cuda") -> dict[str, Any]:
    row = next(row for row in selected if int(row["sample_id"]) == 6); key = str(row["paired_key"])
    visual_n = _sync_visual(_crop_video(Path(video_by[(key, "N_RAW")]["video"]), row["score_box"]), scorer, torch, device=device); visual_t = _sync_visual(_crop_video(Path(video_by[(key, "T_ID")]["video"]), row["score_box"]), scorer, torch, device=device)
    aud_n, n_count, _ = _sync_audio(Path(audio_by[(key, "N_RAW")]["audio_pcm16"]), scorer, torch, device=device); aud_t, t_count, _ = _sync_audio(Path(audio_by[(key, "T_ID")]["audio_pcm16"]), scorer, torch, device=device); support = min(len(visual_n), len(visual_t), n_count, t_count); t_indices = list(range(VSHIFT, support - VSHIFT))
    controls: list[dict[str, Any]] = []
    # Re-extract both embeddings for the repeat.  Reusing the first tensors
    # would only test arithmetic, while the protocol asks for an independent
    # feature/score pass through the frozen evaluator.
    repeat_visual_n = _sync_visual(_crop_video(Path(video_by[(key, "N_RAW")]["video"]), row["score_box"]), scorer, torch, device=device)
    repeat_aud_n, _, _ = _sync_audio(Path(audio_by[(key, "N_RAW")]["audio_pcm16"]), scorer, torch, device=device)
    repeat_visual_t = _sync_visual(_crop_video(Path(video_by[(key, "T_ID")]["video"]), row["score_box"]), scorer, torch, device=device)
    repeat_aud_t, _, _ = _sync_audio(Path(audio_by[(key, "T_ID")]["audio_pcm16"]), scorer, torch, device=device)
    for name, visual, aud, repeat_visual, repeat_aud in (("N_RAW_REPEAT", visual_n, aud_n, repeat_visual_n, repeat_aud_n), ("T_ID_REPEAT", visual_t, aud_t, repeat_visual_t, repeat_aud_t)):
        lags, first = _curve(visual, aud, t_indices); _, second = _curve(repeat_visual, repeat_aud, t_indices); m1, m2 = _curve_metrics(lags, first), _curve_metrics(lags, second); error = float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
        controls.append({"name": name, "status": "PASS" if error <= 1e-4 else "FAIL", "curve_max_abs_error": error, "first": {**m1, "curve": first}, "second": {**m2, "curve": second}, "common_support_count": len(t_indices)})
    values, _ = _audio(Path(audio_by[(key, "N_RAW")]["audio_pcm16"]))
    delayed = np.concatenate([np.zeros(3200, dtype=np.float32), values[:-3200]])
    delay_path = output / "controls" / "id6_n_raw_delay_200ms.wav"; delay_path.parent.mkdir(parents=True, exist_ok=True); sf.write(str(delay_path), delayed, SAMPLE_RATE, subtype="PCM_16")
    aud_delay, _, _ = _sync_audio(delay_path, scorer, torch, device=device); lags, curve = _curve(visual_n, aud_delay, t_indices); old = _curve_metrics(lags, _curve(visual_n, aud_n, t_indices)[1]); new = _curve_metrics(lags, curve); expected = old["k_star"] + 5; edge = abs(old["k_star"]) == VSHIFT; passed = not edge and abs(new["k_star"] - expected) <= 1
    controls.append({"name": "N_RAW_DELAY_200MS", "status": "PASS" if passed else ("UNINTERPRETABLE_EDGE" if edge else "FAIL"), "old_k_star": old["k_star"], "new_k_star": new["k_star"], "expected_new_k_star": expected, "curve": curve, "metrics": new, "common_support_count": len(t_indices), "delay_samples": 3200})
    return {"schema_version": 1, "status": "complete" if all(item["status"] == "PASS" for item in controls) else "incomplete", "controls": controls}


def _read_scores(output: Path) -> list[dict[str, Any]]:
    path = output / "scores.csv"
    if not path.is_file(): raise ProtocolError("scores.csv is missing")
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    numeric = ("sample_id", "C", "D", "B", "C_raw", "D_raw", "B_raw", "k_star", "d_zero", "d_zero_raw", "common_support_count", "fixed_k_n", "d_fixed_natural_lag")
    result = []
    for row in rows:
        copy = dict(row)
        for key in numeric:
            if copy.get(key) in (None, ""): copy[key] = None
            elif key == "sample_id" or key in {"k_star", "common_support_count", "fixed_k_n"}: copy[key] = int(copy[key])
            else: copy[key] = float(copy[key])
        for field in ("C", "D", "B", "d_zero"):
            raw_field = f"{field}_raw"
            if copy.get(raw_field) is not None:
                copy[field] = float(copy[raw_field])
        result.append(copy)
    return result


def _bootstrap_indices(n: int, *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    if n <= 0: raise ProtocolError("bootstrap requires at least one utterance")
    return np.random.default_rng(seed).integers(0, n, size=(draws, n), endpoint=False)


def _stat_summary(values: Sequence[float], indices: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all(): return {"n": int(np.sum(np.isfinite(arr))), "mean": None, "ci95": [None, None], "ci_bonferroni": [None, None], "reason": "MISSING_VALID_OCCURRENCE"}
    estimates = np.mean(arr[indices], axis=1)
    ci95 = np.quantile(estimates, [0.025, 0.975], method="linear"); cib = np.quantile(estimates, [BONFERRONI_ALPHA / 2.0, 1.0 - BONFERRONI_ALPHA / 2.0], method="linear")
    return {"n": int(arr.size), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "positive_count": int(np.sum(arr > 0)), "negative_count": int(np.sum(arr < 0)), "values": arr.tolist(), "ci95": ci95.tolist(), "ci_bonferroni": cib.tolist(), "ci_bonferroni_coverage": 1.0 - BONFERRONI_ALPHA}


def _score_map(scores: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in scores:
        key = (int(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result: raise ProtocolError(f"duplicate score cell: {key}")
        if not math.isfinite(float(row["C"])) or not math.isfinite(float(row["D"])): raise ProtocolError(f"non-finite score: {key}")
        result[key] = row
    return result


def _c_delta_decomposition(new_score: Mapping[str, Any], old_score: Mapping[str, Any]) -> dict[str, float]:
    """Split a C difference into median and minimum-distance contributions."""
    b_change = float(new_score["B"]) - float(old_score["B"])
    d_improvement = float(old_score["D"]) - float(new_score["D"])
    delta_c = float(new_score["C"]) - float(old_score["C"])
    return {"delta_C": delta_c, "B_new_minus_old": b_change, "D_old_minus_new": d_improvement, "identity_error": delta_c - (b_change + d_improvement)}


def _describe(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        return {"count": int(np.sum(np.isfinite(array))), "mean": None, "median": None, "values": [], "reason": "NONFINITE"}
    return {"count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)), "min": float(np.min(array)), "max": float(np.max(array)), "positive_count": int(np.sum(array > 0)), "negative_count": int(np.sum(array < 0)), "values": array.tolist()}


def analyze_stage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve(); _inputs, rows = _load_run_inputs(output); feature_manifest = _feature_manifest(output); scores_manifest = read_json(output / "scores_manifest.json")
    if scores_manifest.get("status") != "complete": raise ProtocolError("analysis requires complete scores")
    selected = _eval_rows(rows, False); scores = _read_scores(output); expected = expected_cells(EVALUATION_IDS); smap = _score_map(scores)
    actual = {(int(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])) for row in scores}
    if actual != expected: raise ProtocolError(f"score cells differ from 10x21 protocol: expected {len(expected)}, got {len(actual)}")
    paired: list[dict[str, Any]] = []; feature_by_id = {int(rec["sample_id"]): rec for rec in feature_manifest["records"]}
    for row in selected:
        sid = int(row["sample_id"]); q = {arm: float(smap[(sid, arm, "N_RAW")]["C"]) for arm in ARMS}; source = feature_by_id[sid]["source_diagnostic"]
        b_n, b_t = q["N_R_DOWN"] - q["N_ID"], q["T_R_DOWN"] - q["T_ID"]
        comparison_specs = (("N_R_DOWN", "N_ID"), ("N_P_DOWN", "N_ID"), ("T_R_DOWN", "T_ID"), ("T_P_DOWN", "T_ID"))
        d_improvements: dict[str, float] = {}; fixed_lag_improvements: dict[str, float] = {}; c_decompositions: dict[str, dict[str, float]] = {}
        for candidate, baseline in comparison_specs:
            label = f"{candidate}_vs_{baseline}"
            candidate_score, baseline_score = smap[(sid, candidate, "N_RAW")], smap[(sid, baseline, "N_RAW")]
            d_improvements[label] = float(baseline_score["D"]) - float(candidate_score["D"])
            if candidate_score.get("d_fixed_natural_lag") is not None and baseline_score.get("d_fixed_natural_lag") is not None:
                fixed_lag_improvements[label] = float(baseline_score["d_fixed_natural_lag"]) - float(candidate_score["d_fixed_natural_lag"])
            c_decompositions[label] = _c_delta_decomposition(candidate_score, baseline_score)
        f_r_n, f_r_t_raw = source.get("f_R_N"), source.get("f_R_T_RAW")
        s_value = None if f_r_n is None or f_r_t_raw is None else float(f_r_n) - float(f_r_t_raw)
        paired.append({"sample_id": sid, "paired_key": row["paired_key"], "f_R_N": f_r_n, "f_R_T_RAW": f_r_t_raw, "f_R_T_ALIGNED": source.get("f_R_T_ALIGNED"), "S": s_value, "A": source.get("semantic_A"), "b_N": b_n, "h_N": q["N_ID"] - q["N_P_DOWN"], "b_T": b_t, "h_T": q["T_ID"] - q["T_P_DOWN"], "J": b_n - b_t, "source_advantage_q_T_ID_minus_N_ID": q["T_ID"] - q["N_ID"], "replace_reference_q_T_ID_minus_N_RAW": q["T_ID"] - q["N_RAW"], "d_improvement": json.dumps(d_improvements, sort_keys=True), "d_fixed_natural_lag_improvement": json.dumps(fixed_lag_improvements, sort_keys=True), "c_decomposition": json.dumps(c_decompositions, sort_keys=True), **{f"q_{arm}": q[arm] for arm in ARMS}})
    with (output / "paired_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(paired[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    indices = _bootstrap_indices(len(paired)); primary: dict[str, Any] = {}
    for name, field in (("S", "S"), ("A", "A"), ("b_N", "b_N"), ("h_N", "h_N"), ("h_T", "h_T"), ("J", "J")):
        values = np.asarray([row[field] if row[field] is not None else np.nan for row in paired], dtype=np.float64)
        primary[name] = _stat_summary(values, indices)
    two_by_two: dict[str, Any] = {}
    for source, ids in (("N", N_ARMS), ("T", T_ARMS)):
        identity = ids[0]; two_by_two[source] = {}
        for intervention in ids[1:]:
            def c(video: str, audio: str) -> np.ndarray: return np.asarray([float(smap[(int(row["sample_id"]), video, audio)]["C"]) for row in selected])
            q00, q10, q01, q11 = c(identity, identity), c(intervention, identity), c(identity, intervention), c(intervention, intervention)
            two_by_two[source][intervention] = {"G": (q10 - q00).tolist(), "E": (q01 - q00).tolist(), "I": (q11 - q10 - q01 + q00).tolist(), "native_delta": (q11 - q00).tolist(), "delta_C": {"G": [_c_delta_decomposition(smap[(int(row["sample_id"]), intervention, identity)], smap[(int(row["sample_id"]), identity, identity)]) for row in selected], "E": [_c_delta_decomposition(smap[(int(row["sample_id"]), identity, intervention)], smap[(int(row["sample_id"]), identity, identity)]) for row in selected], "native": [_c_delta_decomposition(smap[(int(row["sample_id"]), intervention, intervention)], smap[(int(row["sample_id"]), identity, identity)]) for row in selected]}, "identity_check_max_abs": float(np.max(np.abs((q11 - q00) - ((q10 - q00) + (q01 - q00) + (q11 - q10 - q01 + q00)))))}
    source_advantage = [float(item["source_advantage_q_T_ID_minus_N_ID"]) for item in paired]
    replace_reference = [float(item["replace_reference_q_T_ID_minus_N_RAW"]) for item in paired]
    auxiliary = {"source_advantage_q_T_ID_minus_N_ID": _describe(source_advantage), "replace_reference_q_T_ID_minus_N_RAW": _describe(replace_reference), "d_improvement": {}, "d_fixed_natural_lag_improvement": {}}
    for key in ("N_R_DOWN_vs_N_ID", "N_P_DOWN_vs_N_ID", "T_R_DOWN_vs_T_ID", "T_P_DOWN_vs_T_ID"):
        auxiliary["d_improvement"][key] = _describe([json.loads(item["d_improvement"])[key] for item in paired])
        auxiliary["d_fixed_natural_lag_improvement"][key] = _describe([json.loads(item["d_fixed_natural_lag_improvement"])[key] for item in paired])
    analysis = {"schema_version": 1, "status": "complete", "protocol": PROTOCOL, "sample_count": 10, "coverage": {"all_sentences_passed": all(bool(rec["coverage"]["coverage_pass"]) for rec in feature_manifest["records"]), "records": [{"sample_id": int(rec["sample_id"]), **rec["coverage"]} for rec in feature_manifest["records"]]}, "primary": primary, "paired_effects": paired, "auxiliary": auxiliary, "two_by_two": two_by_two, "controls_status": read_json(output / "controls.json").get("status"), "quality_status": "QUALITY_NOT_ASSESSED", "statistics": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "bonferroni_alpha": BONFERRONI_ALPHA}}
    write_json(output / "analysis.json", analysis); _write_report_and_plot(output, analysis); _write_playable_index(output, selected, ARMS)
    return analysis


def _allowed_conclusion(analysis: Mapping[str, Any]) -> str:
    primary = analysis.get("primary", {}); controls_ok = analysis.get("controls_status") == "complete"; coverage_ok = bool(analysis.get("coverage", {}).get("all_sentences_passed")); names = ("S", "A", "b_N", "h_N", "h_T", "J")
    joint = coverage_ok and controls_ok and all(primary.get(name, {}).get("ci_bonferroni", [None, None])[1] is not None and primary[name]["ci_bonferroni"][0] > 0 for name in names)
    if joint and analysis.get("quality_status") == "ASSESSED_PASS": return "允许写为：在该小样本和操作定义下，证据共同支持声学变化重组假设；仍需保留单说话人、处理链、声码器和非真人同步边界。"
    if joint: return "统计条件满足，但人工听检未完成或质量未确认；功能解释必须降级为含音质混杂的结果，不能写成完整机制证明。"
    return "不允许写成共同支持机制；按预注册分支报告各统计量和不可判读/阴性原因。"


def _write_report_and_plot(output: Path, analysis: Mapping[str, Any]) -> None:
    primary = analysis["primary"]; coverage = analysis["coverage"]; quality = "未提供人工听检（QUALITY_NOT_ASSESSED）"
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    def interval(value: Any) -> str:
        return "NA" if not isinstance(value, Sequence) or len(value) != 2 else f"[{fmt(value[0])}, {fmt(value[1])}]"

    stats_text = "; ".join(f"{name}={fmt(row.get('mean'))}" for name, row in primary.items())
    first = f"完整性：210 个科学评分和模板/特征阶段已完成；覆盖：所有 evaluation 句通过覆盖门={coverage['all_sentences_passed']}；六主统计：{stats_text}；质量评估状态：{quality}；允许结论：{_allowed_conclusion(analysis)}"
    lines = ["# TTS 声学变化重组：轨迹与剩余项干预", "", first, "", "本报告只解释当前 5 条 donor、10 条 evaluation 的冻结历史处理链。P 是跨句共享模板方向，R 是相对于该方向的剩余项；两者都未被先验标成发音或噪声。", "", "## 主统计", "", "| 统计 | 均值 | 普通95%区间 | Bonferroni区间 |", "|---|---:|---|---|"]
    for name, row in primary.items(): lines.append(f"| {name} | {fmt(row.get('mean'))} | {interval(row.get('ci95'))} | {interval(row.get('ci_bonferroni'))} |")
    auxiliary = analysis.get("auxiliary", {})
    lines += ["", "## 辅助诊断", "", "| 辅助量 | 均值 |", "|---|---:|", f"| q(T_ID)-q(N_ID) | {fmt(auxiliary.get('source_advantage_q_T_ID_minus_N_ID', {}).get('mean'))} |", f"| q(T_ID)-q(N_RAW) | {fmt(auxiliary.get('replace_reference_q_T_ID_minus_N_RAW', {}).get('mean'))} |"]
    for key, summary in auxiliary.get("d_improvement", {}).items():
        lines.append(f"| D improvement {key} | {fmt(summary.get('mean'))} |")
    lines += ["", "`paired_effects.csv` 和 `analysis.json` 还保存逐句 b_T、J、来源优势、replace 参照、D 改善、固定自然 lag 的距离改善，以及每个 C 差的 B/D 分解；每个 evaluation 的 feature JSON 还保存原生 P/R/U 每元素能量和投影系数符号。`diagnostic_plots.png` 包含逐句 b/h、六主统计区间和 f_R 来源对比。", "", "## 解释边界", "", "固定自然评分音轨 q=C(a,N_RAW) 用于主干预统计；原生 TTS 的 f_R 只作为来源诊断。若音质 QC 恶化，干预效应包含整条声码器链的影响。SyncNet 分数不等于真人口型同步。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        names = list(primary); means = [primary[name].get("mean", np.nan) for name in names]; lo = [primary[name].get("ci95", [np.nan, np.nan])[0] for name in names]; hi = [primary[name].get("ci95", [np.nan, np.nan])[1] for name in names]
        fig, ax = plt.subplots(figsize=(7, 3)); ax.errorbar(range(len(names)), means, yerr=[np.asarray(means)-np.asarray(lo), np.asarray(hi)-np.asarray(means)], fmt="o"); ax.axhline(0, color="black", linewidth=0.7); ax.set_xticks(range(len(names)), names); ax.set_ylabel("paired effect"); fig.tight_layout(); fig.savefig(output / "primary_effects.png", dpi=160); plt.close(fig)
        paired = list(analysis.get("paired_effects", [])); ids = [int(item["sample_id"]) for item in paired]
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        for field, label, marker in (("b_N", "b_N", "o"), ("h_N", "h_N", "s"), ("h_T", "h_T", "^") ):
            axes[0].plot(ids, [float(item[field]) for item in paired], marker=marker, label=label)
        axes[0].axhline(0, color="black", linewidth=0.7); axes[0].set_title("per-sentence b/h"); axes[0].set_xlabel("sample_id"); axes[0].legend(fontsize=8)
        axes[1].errorbar(range(len(names)), means, yerr=[np.asarray(means)-np.asarray(lo), np.asarray(hi)-np.asarray(means)], fmt="o"); axes[1].axhline(0, color="black", linewidth=0.7); axes[1].set_xticks(range(len(names)), names); axes[1].set_title("primary intervals")
        for field, label, marker in (("f_R_N", "N", "o"), ("f_R_T_RAW", "T_RAW", "s"), ("f_R_T_ALIGNED", "T_ALIGNED", "^") ):
            values = [item.get(field) for item in paired]
            if all(value is not None for value in values): axes[2].plot(ids, [float(value) for value in values], marker=marker, label=label)
        axes[2].set_title("f_R source diagnostic"); axes[2].set_xlabel("sample_id"); axes[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output / "diagnostic_plots.png", dpi=160); plt.close(fig)
    except Exception:  # noqa: BLE001 - plotting is supplementary
        (output / "primary_effects.png").write_bytes(b"")
        (output / "diagnostic_plots.png").write_bytes(b"")


def _write_playable_index(output: Path, rows: Sequence[Mapping[str, Any]], arms: Sequence[str]) -> None:
    selected = [row for row in rows if int(row["sample_id"]) in (6, 7, 8)]
    audio_manifest = read_json(output / "audio_manifest.json"); video_manifest = read_json(output / "videos_manifest.json")
    audio_by = {(int(r["sample_id"]), str(r["arm"])): r for rec in audio_manifest.get("records", []) for r in rec.get("audio", [])}; video_by = {(int(r["sample_id"]), str(r["arm"])): r for r in video_manifest.get("videos", [])}
    payload = [{"sample_id": int(row["sample_id"]), "paired_key": row["paired_key"], "arms": {arm: {"audio": audio_by.get((int(row["sample_id"]), arm), {}).get("audio_pcm16"), "video": video_by.get((int(row["sample_id"]), arm), {}).get("video")} for arm in arms}} for row in selected]
    write_json(output / "playable_index.json", {"status": "complete" if len(payload) == 3 else "QUALITY_NOT_ASSESSED", "records": payload})


def validate_stage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve(); inputs, rows = _load_run_inputs(output); checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    # Re-read all parent-bound assets.  This makes validation detect a changed
    # parent file even when the derived run files themselves still exist.
    parent_failures: list[str] = []
    try:
        parent_rows = _parent_rows(Path(str(inputs["parent"]["run"])))
        parent_by_id = {int(item["sample_id"]): item for item in parent_rows}
        for row in rows:
            sid = int(row["sample_id"]); parent = parent_by_id.get(sid)
            if parent is None or parent["paired_key"] != row["paired_key"] or parent["natural_audio_sha256"] != row["natural_audio_sha256"] or parent["tts_audio_sha256"] != row["tts_audio_sha256"]:
                parent_failures.append(str(sid))
    except Exception as exc:  # noqa: BLE001 - report as a validation failure
        parent_failures.append(str(exc))
    check("parent_identity", not parent_failures, parent_failures)
    check("partition", [int(r["sample_id"]) for r in rows if r["role"] == "donor"] == list(DONOR_IDS) and [int(r["sample_id"]) for r in rows if r["role"] == "evaluation"] == list(EVALUATION_IDS))

    templates = read_json(output / "templates.json")
    donor_keys = {str(key) for item in templates.get("templates", {}).values() for key in item.get("donor_paired_keys", [])}
    eval_keys = {str(r["paired_key"]) for r in rows if r["role"] == "evaluation"}
    template_rows = templates.get("templates", {})
    check("template_no_evaluation_leak", donor_keys.isdisjoint(eval_keys), {"donor_keys": sorted(donor_keys), "evaluation_keys": sorted(eval_keys)})
    check("template_support", all(len(item.get("donor_paired_keys", [])) >= 2 for item in template_rows.values()), {"template_count": len(template_rows)})

    feature_manifest = _feature_manifest(output); feature_failures: list[str] = []
    with np.load(output / "templates.npz", allow_pickle=False) as template_file:
        template_labels = [str(value) for value in template_file["labels"].tolist()]
        template_by_label = {label: np.asarray(template_file["templates"][index], dtype=np.float64) for index, label in enumerate(template_labels)}
    expected_feature_keys = {"Z_N", "Z_T", "Z_T_RAW", "N_ID", "N_R_DOWN", "N_P_DOWN", "T_ID", "T_R_DOWN", "T_P_DOWN"}
    for record in feature_manifest["records"]:
        sid = str(record["sample_id"])
        try:
            loaded = _load_feature_record(record); arrays, meta = loaded["arrays"], loaded["metadata"]; z_n, z_t = arrays["Z_N"], arrays["Z_T"]
            if set(arrays) != expected_feature_keys:
                feature_failures.append(f"{sid}:array_keys")
            if z_n.ndim != 2 or z_t.shape != z_n.shape or arrays["Z_T_RAW"].ndim != 2 or arrays["Z_T_RAW"].shape[1] != FEATURE_DIM:
                feature_failures.append(f"{sid}:shape")
            for arm in ("N_ID", "N_R_DOWN", "N_P_DOWN"):
                if arrays[arm].shape != z_n.shape:
                    feature_failures.append(f"{sid}:{arm}_shape")
            for arm in ("T_ID", "T_R_DOWN", "T_P_DOWN"):
                if arrays[arm].shape != z_t.shape:
                    feature_failures.append(f"{sid}:{arm}_shape")
            if not np.array_equal(arrays["N_ID"], z_n) or not np.array_equal(arrays["T_ID"], z_t):
                feature_failures.append(f"{sid}:ID")
            active_indices: set[int] = set()
            occurrence_groups: list[list[int]] = []
            for op in meta.get("operations", []):
                frame_indices = [int(i) for i in op.get("frame_indices", [])]
                internal = [int(i) for i in op.get("internal_indices", [])]
                if frame_indices:
                    occurrence_groups.append(frame_indices)
                    # Ineligible/no-template occurrences deliberately store an
                    # empty internal list.  Check the edge convention only for
                    # operations that actually carry an internal segment.
                    if (op.get("active") or op.get("reason") == "internal_short") and internal != frame_indices[EDGE_FRAMES:-EDGE_FRAMES]:
                        feature_failures.append(f"{sid}:edge_definition_{op.get('occurrence_index')}")
                if not op.get("active"):
                    continue
                label = str(op["label"])
                if label not in template_by_label or not internal:
                    feature_failures.append(f"{sid}:missing_template_{op.get('occurrence_index')}"); continue
                active_indices.update(internal)
                q = _template_to_length(template_by_label[label], len(internal))
                n_comp, t_comp = _project_components(z_n[internal], q), _project_components(z_t[internal], q)
                norms = [float(n_comp["p_norm"]), float(n_comp["r_norm"]), float(t_comp["p_norm"]), float(t_comp["r_norm"])]
                delta = 0.5 * min(norms)
                expected = {
                    "N_R_DOWN": np.asarray(n_comp["mu"]) + n_comp["P"] + (1.0 - delta / norms[1]) * n_comp["R"],
                    "N_P_DOWN": np.asarray(n_comp["mu"]) + (1.0 - delta / norms[0]) * n_comp["P"] + n_comp["R"],
                    "T_R_DOWN": np.asarray(t_comp["mu"]) + t_comp["P"] + (1.0 - delta / norms[3]) * t_comp["R"],
                    "T_P_DOWN": np.asarray(t_comp["mu"]) + (1.0 - delta / norms[2]) * t_comp["P"] + t_comp["R"],
                }
                for arm, expected_values in expected.items():
                    if not np.allclose(arrays[arm][internal], expected_values, atol=FLOAT_TOL, rtol=FLOAT_TOL):
                        feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_{arm}")
                    source = z_n if arm.startswith("N_") else z_t
                    if abs(float(np.linalg.norm(arrays[arm][internal] - source[internal])) - delta) > FLOAT_TOL:
                        feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_{arm}_delta")
                    if np.max(np.abs(arrays[arm][internal].mean(axis=0) - source[internal].mean(axis=0))) > FLOAT_TOL:
                        feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_{arm}_mean")
                    source32, actual32 = source.astype(np.float32), arrays[arm].astype(np.float32)
                    if abs(float(np.linalg.norm(actual32[internal] - source32[internal])) - delta) / max(delta, EPS) > FLOAT32_TOL:
                        feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_{arm}_float32_delta")
                    if np.max(np.abs((actual32[internal] - source32[internal]).mean(axis=0))) > FLOAT32_TOL:
                        feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_{arm}_float32_mean")
                for component, other in ((n_comp["P"], n_comp["R"]), (t_comp["P"], t_comp["R"])):
                    if abs(float(np.sum(component * other))) > FLOAT_TOL or np.max(np.abs(np.asarray(component).mean(axis=0))) > FLOAT_TOL or np.max(np.abs(np.asarray(other).mean(axis=0))) > FLOAT_TOL:
                        feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_orthogonality")
                if np.max(np.abs(np.asarray(n_comp["U"]) - n_comp["P"] - n_comp["R"])) > FLOAT_TOL or np.max(np.abs(np.asarray(t_comp["U"]) - t_comp["P"] - t_comp["R"])) > FLOAT_TOL:
                    feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_reconstruction")
                factors = [1.0 - delta / norms[index] for index in range(4)]
                if any(value < 0.5 - FLOAT_TOL or value > 1.0 + FLOAT_TOL for value in factors):
                    feature_failures.append(f"{sid}:op{op.get('occurrence_index')}_factor")
            # Non-active frames, occurrence endpoints, internal means and
            # cross-occurrence boundaries are all frozen by the protocol.
            for arm, source in (("N_R_DOWN", z_n), ("N_P_DOWN", z_n), ("T_R_DOWN", z_t), ("T_P_DOWN", z_t)):
                inactive = sorted(set(range(source.shape[0])) - active_indices)
                if inactive and not np.array_equal(arrays[arm][inactive], source[inactive]):
                    feature_failures.append(f"{sid}:{arm}_inactive")
                for group in occurrence_groups:
                    if group and (not np.array_equal(arrays[arm][group[0]], source[group[0]]) or not np.array_equal(arrays[arm][group[-1]], source[group[-1]])):
                        feature_failures.append(f"{sid}:{arm}_endpoint")
                    internal = group[EDGE_FRAMES:-EDGE_FRAMES]
                    if internal and np.max(np.abs(arrays[arm][internal].mean(axis=0) - source[internal].mean(axis=0))) > FLOAT_TOL:
                        feature_failures.append(f"{sid}:{arm}_occurrence_mean")
                for left, right in pairwise(occurrence_groups):
                    if left and right and left[-1] + 1 == right[0] and not np.allclose(arrays[arm][right[0]] - arrays[arm][left[-1]], source[right[0]] - source[left[-1]], atol=FLOAT_TOL, rtol=FLOAT_TOL):
                        feature_failures.append(f"{sid}:{arm}_boundary")
            check(f"feature_{sid}", not any(value.startswith(f"{sid}:") for value in feature_failures), meta.get("coverage"))
        except Exception as exc:  # noqa: BLE001 - retain the exact record failure
            feature_failures.append(f"{sid}:exception:{exc}")
            check(f"feature_{sid}", False, str(exc))
    check("feature_invariants", not feature_failures, feature_failures[:20] if len(feature_failures) > 20 else feature_failures)

    # Validate all 70 PCM records and their conditioning hashes.
    audio_failures: list[str] = []; audio_manifest = read_json(output / "audio_manifest.json")
    audio_rows = [item for record in audio_manifest.get("records", []) for item in record.get("audio", [])]
    audio_keys = {(int(item.get("sample_id", -1)), str(item.get("arm"))) for item in audio_rows}
    expected_audio_keys = {(sid, arm) for sid in EVALUATION_IDS for arm in ARMS}
    feature_by_id = {int(item["sample_id"]): item for item in feature_manifest["records"]}
    try:
        import torch
        for item in audio_rows:
            sid, arm = int(item["sample_id"]), str(item["arm"]); path = _resolve_asset(item["audio_pcm16"], expected_sha256=str(item["audio_sha256"]))
            values, rate = _audio(path); target = len(_audio(Path(next(row for row in rows if int(row["sample_id"]) == sid)["natural_audio"]))[0])
            if rate != SAMPLE_RATE or len(values) != target or int(item.get("sample_count", -1)) != target or not bool(item.get("qc", {}).get("finite", False)):
                audio_failures.append(f"{sid}:{arm}:shape_qc")
            if arm == "N_RAW":
                source = next(row for row in rows if int(row["sample_id"]) == sid)
                if str(path) != str(_resolve_asset(source["natural_audio"], expected_sha256=str(source["natural_audio_sha256"]))) or item.get("audio_sha256") != source.get("natural_audio_sha256"):
                    audio_failures.append(f"{sid}:N_RAW:source")
            else:
                condition_path = _resolve_asset(item.get("conditioning_path"), expected_sha256=str(item.get("conditioning_sha256")))
                payload = torch.load(condition_path, map_location="cpu", weights_only=False)
                condition = np.asarray(torch.as_tensor(payload["conditioning"], dtype=torch.float32).cpu(), dtype=np.float32)
                expected_condition = np.asarray(_load_feature_record(feature_by_id[sid])["arrays"][arm], dtype=np.float32)
                if not np.array_equal(condition, expected_condition):
                    audio_failures.append(f"{sid}:{arm}:conditioning")
    except Exception as exc:  # noqa: BLE001
        audio_failures.append(str(exc))
    check("audio_70_cells", audio_manifest.get("status") == "complete" and len(audio_rows) == 70 and audio_keys == expected_audio_keys and not audio_failures, audio_failures[:20] if audio_failures else len(audio_rows))

    # Validate all 70 videos, including the fixed parent crop in the command.
    video_failures: list[str] = []; video_manifest = read_json(output / "videos_manifest.json"); video_rows = list(video_manifest.get("videos", [])); video_keys = {(int(item.get("sample_id", -1)), str(item.get("arm"))) for item in video_rows}
    audio_by = {(int(item["sample_id"]), str(item["arm"])): item for item in audio_rows}; row_by_id = {int(row["sample_id"]): row for row in rows}
    for item in video_rows:
        sid, arm = int(item.get("sample_id", -1)), str(item.get("arm")); source = row_by_id.get(sid)
        try:
            path = _resolve_asset(item["video"], expected_sha256=str(item["video_sha256"])); info = _video_info(path); expected_box = list(_wav2lip_box(source["score_box"])) if source else None
            parameters = item.get("parameters", {})
            if sid not in EVALUATION_IDS or arm not in ARMS or source is None or info["frame_count"] < 25 or abs(info["fps"] - FPS) > 0.01 or item.get("audio_sha256") != audio_by[(sid, arm)]["audio_sha256"] or item.get("face_sha256") != source["face_video_sha256"] or parameters.get("box") != expected_box or parameters.get("box_order") != "top,bottom,left,right":
                video_failures.append(f"{sid}:{arm}:identity")
        except Exception as exc:  # noqa: BLE001
            video_failures.append(f"{sid}:{arm}:{exc}")
    check("video_70_cells", video_manifest.get("status") == "complete" and len(video_rows) == 70 and video_keys == expected_audio_keys and not video_failures, video_failures[:20] if video_failures else len(video_rows))

    # Curves are independently checked against the scalar score CSV, then the
    # primary analysis is recomputed directly from those checked scalars.
    score_failures: list[str] = []
    try:
        scores = _read_scores(output); smap = _score_map(scores); expected_scores = expected_cells(EVALUATION_IDS)
        curve_payload = read_json(output / "curves.json"); curve_rows = {(int(item["sample_id"]), str(item["video_arm"]), str(item["audio_arm"])): item for item in curve_payload.get("curves", [])}
        if set(smap) != expected_scores or len(scores) != 210 or set(curve_rows) != expected_scores:
            score_failures.append(f"cell_set:{len(scores)}/{len(curve_rows)}")
        for key, score in smap.items():
            curve = curve_rows.get(key)
            if curve is None or list(curve.get("lags", [])) != list(range(-VSHIFT, VSHIFT + 1)) or int(curve.get("common_support_count", 0)) < MIN_COMMON_WINDOWS:
                score_failures.append(f"{key}:curve_header"); continue
            metrics = _curve_metrics(curve["lags"], curve["values"])
            for field in ("C", "D", "B", "d_zero"):
                if abs(float(score[field]) - float(metrics[field])) > 1e-6:
                    score_failures.append(f"{key}:{field}")
            if int(score["k_star"]) != int(metrics["k_star"]):
                score_failures.append(f"{key}:k_star")
        check("score_210_curves", read_json(output / "scores_manifest.json").get("status") == "complete" and not score_failures, score_failures[:20] if score_failures else 210)
    except Exception as exc:  # noqa: BLE001 - retain validation failure detail
        scores, smap = [], {}; score_failures.append(str(exc)); check("score_210_curves", False, score_failures)

    controls = read_json(output / "controls.json") if (output / "controls.json").is_file() else {}
    control_names = {str(item.get("name")) for item in controls.get("controls", [])}
    check("three_controls", control_names == {"N_RAW_REPEAT", "T_ID_REPEAT", "N_RAW_DELAY_200MS"} and controls.get("status") == "complete" and all(item.get("status") == "PASS" for item in controls.get("controls", [])), controls.get("status"))

    analysis_failures: list[str] = []
    try:
        analysis = read_json(output / "analysis.json"); paired_by_id = {int(item["sample_id"]): item for item in analysis.get("paired_effects", [])}
        for row in (row_by_id[sid] for sid in EVALUATION_IDS):
            sid = int(row["sample_id"]); q = {arm: float(smap[(sid, arm, "N_RAW")]["C"]) for arm in ARMS}; diag = feature_by_id[sid]["source_diagnostic"]
            expected = {"f_R_N": diag.get("f_R_N"), "f_R_T_RAW": diag.get("f_R_T_RAW"), "f_R_T_ALIGNED": diag.get("f_R_T_ALIGNED"), "S": None if diag.get("f_R_N") is None or diag.get("f_R_T_RAW") is None else float(diag["f_R_N"]) - float(diag["f_R_T_RAW"]), "A": diag.get("semantic_A"), "b_N": q["N_R_DOWN"] - q["N_ID"], "h_N": q["N_ID"] - q["N_P_DOWN"], "b_T": q["T_R_DOWN"] - q["T_ID"], "h_T": q["T_ID"] - q["T_P_DOWN"], "J": (q["N_R_DOWN"] - q["N_ID"]) - (q["T_R_DOWN"] - q["T_ID"]) }
            actual = paired_by_id.get(sid, {})
            for field, value in expected.items():
                if value is None:
                    if actual.get(field) is not None: analysis_failures.append(f"{sid}:{field}:expected_null")
                elif actual.get(field) is None or abs(float(actual[field]) - float(value)) > 1e-10:
                    analysis_failures.append(f"{sid}:{field}")
        indices = _bootstrap_indices(10)
        for name, field in (("S", "S"), ("A", "A"), ("b_N", "b_N"), ("h_N", "h_N"), ("h_T", "h_T"), ("J", "J")):
            values = [float(paired_by_id[sid][field]) for sid in EVALUATION_IDS]
            expected_summary = _stat_summary(values, indices); actual_summary = analysis.get("primary", {}).get(name, {})
            if abs(float(actual_summary.get("mean")) - float(expected_summary["mean"])) > 1e-12 or not np.allclose(actual_summary.get("ci95"), expected_summary["ci95"], atol=1e-12, rtol=1e-12) or not np.allclose(actual_summary.get("ci_bonferroni"), expected_summary["ci_bonferroni"], atol=1e-12, rtol=1e-12):
                analysis_failures.append(f"primary:{name}")
        if any(float(item.get("identity_check_max_abs", 1.0)) > 1e-12 for source in analysis.get("two_by_two", {}).values() for item in source.values()):
            analysis_failures.append("two_by_two_identity")
    except Exception as exc:  # noqa: BLE001 - retain validation failure detail
        analysis_failures.append(str(exc))
    check("independent_analysis_recompute", not analysis_failures, analysis_failures[:20] if analysis_failures else "paired effects, bootstrap, and G/E/I")

    # Run the separate dependency-light auditor as a second implementation of
    # the saved-curve and paired-statistic checks.  It deliberately does not
    # import this module, so a shared helper bug cannot make both checks pass.
    independent_script = Path(__file__).with_name("check_tts_acoustic_reorganization_independent.py")
    try:
        result = subprocess.run([sys.executable, str(independent_script), "--output-dir", str(output)], cwd=str(REPO), capture_output=True, text=True, check=False)
        payload = json.loads(result.stdout) if result.stdout.strip() else {"status": "incomplete", "stderr": result.stderr[-2000:]}
        check("independent_short_script", result.returncode == 0 and payload.get("status") == "complete", payload)
    except Exception as exc:  # noqa: BLE001 - retain validation failure detail
        check("independent_short_script", False, str(exc))

    validation = {"schema_version": 1, "status": "complete" if all(item["status"] == "PASS" for item in checks) else "incomplete", "protocol": PROTOCOL, "checks": checks}
    write_json(output / "validation.json", validation)
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("stage", choices=("prepare", "features", "audio", "render", "score", "analyze", "validate")); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT); parser.add_argument("--knn-source", type=Path, default=DEFAULT_KNN_SOURCE); parser.add_argument("--device", default="cuda"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--wav2lip", type=Path, default=DEFAULT_WAV2LIP); parser.add_argument("--wav2lip-python", type=Path, default=DEFAULT_WAV2LIP_PYTHON); parser.add_argument("--wav2lip-checkpoint", type=Path, default=DEFAULT_WAV2LIP_CHECKPOINT); parser.add_argument("--syncnet", type=Path, default=DEFAULT_SYNCNET); parser.add_argument("--syncnet-python", type=Path, default=DEFAULT_SYNCNET_PYTHON); parser.add_argument("--syncnet-model", type=Path, default=DEFAULT_SYNCNET_MODEL); return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = {"prepare": prepare, "features": features_stage, "audio": audio_stage, "render": render_stage, "score": score_stage, "analyze": analyze_stage, "validate": validate_stage}[args.stage](args)
    except (ProtocolError, OSError, ValueError, RuntimeError) as exc:
        print(f"{args.stage} FAILED: {exc}", file=sys.stderr); return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
