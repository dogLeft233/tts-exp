#!/usr/bin/env python3
"""Train a minimal train-only phone-prototype feature adapter.

This script intentionally trains on precomputed SSL features only.  It never
uses waveform weak targets, TFG/SyncNet scores, validation/test prototypes, or
random frame splits.  The CLI is suitable for a small CPU smoke run and its
helper functions are deliberately dependency-light for unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

try:
    from tts_feature_head import ResidualFeatureAdapter, feature_statistics, parameter_statistics
except ImportError:  # pragma: no cover - supports direct package-style imports
    from scripts.tts_feature_head import ResidualFeatureAdapter, feature_statistics, parameter_statistics

DEFAULT_MANIFEST = Path("results/rhythm_style_500/source_manifests/aishell1_test_400_condition_mfa.json")
DEFAULT_EMBEDDING_DIR = Path("results/rhythm_style_500/aishell1_test_400/embeddings_full")
DEFAULT_LAYER = 11
DEFAULT_ALPHA = 0.5
DEFAULT_MIN_SUPPORT = 20
DEFAULT_SEED = 42

LOSS_WEIGHTS = {
    "prototype_shift_smooth_l1": 1.0,
    "tts_centroid_mse": 0.2,
    "identity_mse": 0.05,
    "delta_l2": 0.01,
}


class DataContractError(ValueError):
    """Raised when an input manifest violates the fail-closed data contract."""


@dataclass(frozen=True)
class PrototypeTable:
    """Train-only natural/TTS phone centroids and support counts."""

    natural: dict[str, np.ndarray]
    tts: dict[str, np.ndarray]
    delta: dict[str, np.ndarray]
    support_natural: dict[str, int]
    support_tts: dict[str, int]
    dim: int
    min_support: int
    alpha: float

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.delta))

    def labels_with_indices(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    def to_json(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "dim": self.dim,
            "min_support": self.min_support,
            "alpha": self.alpha,
            "support_natural": dict(sorted(self.support_natural.items())),
            "support_tts": dict(sorted(self.support_tts.items())),
            "natural_centroid": {
                label: self.natural[label].astype(float).tolist() for label in self.labels
            },
            "tts_centroid": {
                label: self.tts[label].astype(float).tolist() for label in self.labels
            },
            "delta": {label: self.delta[label].astype(float).tolist() for label in self.labels},
        }


@dataclass
class FrameBatch:
    """One variable-length utterance of natural feature frames."""

    key: str
    speaker_id: str
    split: str
    features: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    frame_times: np.ndarray
    diagnostics: dict[str, Any]

    def tensor(self, device: torch.device) -> tuple[Tensor, Tensor, list[str]]:
        valid = self.valid_mask.astype(bool, copy=False)
        return (
            torch.from_numpy(self.features).to(device=device, dtype=torch.float32).unsqueeze(0),
            torch.from_numpy(valid).to(device=device, dtype=torch.bool).unsqueeze(0),
            self.labels.tolist(),
        )


@dataclass(frozen=True)
class RunConfig:
    """Serializable training configuration."""

    layer: int = DEFAULT_LAYER
    min_support: int = DEFAULT_MIN_SUPPORT
    alpha: float = DEFAULT_ALPHA
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_channels: int = 128
    residual_scale: float = 0.25
    seed: int = DEFAULT_SEED
    device: str = "cpu"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON without allowing non-finite values."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_json_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")


def write_json(path: Path, value: Any) -> None:
    """Write strict finite JSON atomically enough for a single-process run."""
    _assert_json_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _record_key(record: Mapping[str, Any]) -> str:
    value = record.get("paired_key") or record.get("pair_key") or record.get("utterance_id")
    if value is None:
        raise DataContractError("record is missing paired_key/pair_key/utterance_id")
    return str(value)


def _speaker(record: Mapping[str, Any]) -> str:
    value = record.get("speaker_id")
    if value is None or str(value) == "":
        raise DataContractError(f"record {_record_key(record)} is missing speaker_id")
    return str(value)


def _split(record: Mapping[str, Any]) -> str:
    value = record.get("split")
    if value not in {"train", "valid", "test"}:
        raise DataContractError(f"record {_record_key(record)} has invalid split {value!r}")
    return str(value)


def _condition(record: Mapping[str, Any]) -> str:
    value = str(record.get("condition", ""))
    if value not in {"natural", "tts"}:
        raise DataContractError(f"record {_record_key(record)} has invalid condition {value!r}")
    return value


def validate_pair_splits(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate paired records and return immutable split diagnostics.

    Validation happens before embeddings are read.  It rejects pair-arm split
    mismatch, duplicate arms, pair overlap, and speaker overlap.  A natural and
    TTS record may have independently different token timings; only their key,
    condition, split, and speaker contract is shared here.
    """
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        key = _record_key(record)
        condition = _condition(record)
        if condition in grouped[key]:
            raise DataContractError(f"duplicate {condition} arm for paired key {key}")
        grouped[key][condition] = record
        if record.get("alignment_source") != "mfa":
            raise DataContractError(
                f"paired key {key} condition {condition} must use alignment_source=mfa"
            )
        method = str(record.get("alignment_method", ""))
        if "uniform" in method.lower() or record.get("uniform_fallback"):
            raise DataContractError(f"uniform alignment fallback is not allowed for {key}/{condition}")
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise DataContractError(f"missing MFA tokens for {key}/{condition}")
        for token in tokens:
            try:
                start = float(token["start_s"])
                end = float(token["end_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataContractError(f"invalid token interval for {key}/{condition}") from exc
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise DataContractError(f"invalid token bounds for {key}/{condition}: {token!r}")
            if not str(token.get("token", "")).strip():
                raise DataContractError(f"empty token label for {key}/{condition}")
        _split(record)
        _speaker(record)
    missing = [key for key, arms in grouped.items() if set(arms) != {"natural", "tts"}]
    if missing:
        raise DataContractError(f"paired keys missing one condition arm: {missing[:5]}")
    pair_splits: dict[str, str] = {}
    pair_speakers: dict[str, str] = {}
    occurrence_mismatch_keys: list[str] = []
    for key, arms in grouped.items():
        nat, tts = arms["natural"], arms["tts"]
        if _split(nat) != _split(tts):
            raise DataContractError(f"natural/TTS split mismatch for {key}")
        if _speaker(nat) != _speaker(tts):
            raise DataContractError(f"natural/TTS speaker mismatch for {key}")
        natural_labels = [_token_label(token) for token in nat["tokens"] if _token_label(token)]
        tts_labels = [_token_label(token) for token in tts["tokens"] if _token_label(token)]
        if natural_labels != tts_labels:
            occurrence_mismatch_keys.append(key)
        pair_splits[key] = _split(nat)
        pair_speakers[key] = _speaker(nat)
    split_keys = {
        split: sorted(key for key, value in pair_splits.items() if value == split)
        for split in ("train", "valid", "test")
    }
    overlaps = {
        f"{left}:{right}": sorted(set(split_keys[left]) & set(split_keys[right]))
        for left in split_keys
        for right in split_keys
        if left < right and set(split_keys[left]) & set(split_keys[right])
    }
    if overlaps:
        raise DataContractError(f"paired key appears in multiple splits: {overlaps}")
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for key, split in pair_splits.items():
        speaker_splits[pair_speakers[key]].add(split)
    speaker_overlap = {speaker: sorted(splits) for speaker, splits in speaker_splits.items() if len(splits) > 1}
    if speaker_overlap:
        raise DataContractError(f"speaker appears in multiple splits: {speaker_overlap}")
    return {
        "pair_count": len(grouped),
        "record_count": len(records),
        "split_keys": split_keys,
        "split_pair_counts": {split: len(keys) for split, keys in split_keys.items()},
        "speaker_ids": {
            split: sorted({pair_speakers[key] for key in keys}) for split, keys in split_keys.items()
        },
        "speaker_count": {
            split: len({pair_speakers[key] for key in keys}) for split, keys in split_keys.items()
        },
        "leakage_checks": {
            "pair_overlap": False,
            "speaker_overlap": False,
            "natural_tts_pair_complete": True,
            "mfa_only": True,
            "uniform_fallback_rejected": True,
            "phone_occurrence_mismatch_keys": sorted(occurrence_mismatch_keys),
            "phone_occurrence_mismatch_count": len(occurrence_mismatch_keys),
        },
    }


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a condition-level manifest and perform structural validation."""
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        records = value.get("records")
        if not isinstance(records, list):
            raise DataContractError("manifest dictionary must contain records list")
        metadata = {key: item for key, item in value.items() if key != "records"}
    elif isinstance(value, list):
        records, metadata = value, {}
    else:
        raise DataContractError("manifest must be a JSON list or object with records")
    diagnostics = validate_pair_splits(records)
    mismatch_keys = _occurrence_mismatch_keys(records)
    if mismatch_keys:
        records = [record for record in records if _record_key(record) not in mismatch_keys]
        diagnostics = validate_pair_splits(records)
        diagnostics["excluded_phone_occurrence_mismatch_keys"] = sorted(mismatch_keys)
        diagnostics["excluded_phone_occurrence_mismatch_count"] = len(mismatch_keys)
    metadata = {key: item for key, item in metadata.items()}
    metadata["manifest_sha256"] = sha256_file(path)
    metadata["pair_validation"] = diagnostics
    metadata["excluded_phone_occurrence_mismatch_keys"] = sorted(mismatch_keys)
    return [dict(record) for record in records], metadata


def _embedding_stem(sample_id: Any, condition: str) -> str:
    try:
        identifier = int(sample_id)
        return f"{identifier}_{condition}_raw_hubert"
    except (TypeError, ValueError):
        return f"{sample_id}_{condition}_raw_hubert"


def resolve_embedding_path(record: Mapping[str, Any], embedding_dir: Path) -> Path:
    """Resolve an embedding from explicit metadata or the project cache."""
    explicit = record.get("embedding_path") or record.get("filepath_embedding")
    if explicit:
        candidate = Path(str(explicit))
        if not candidate.is_absolute():
            candidate = embedding_dir / candidate
        return candidate
    condition = _condition(record)
    provider = str(record.get("tts_provider") or condition)
    candidates = [
        embedding_dir / f"{_embedding_stem(record.get('sample_id'), condition)}.npy",
        embedding_dir / f"{record.get('sample_id')}_{provider}_raw_hubert.npy",
        embedding_dir / f"{record.get('sample_id')}:{condition}:raw_faster_qwen3_raw_hubert.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def load_embedding(record: Mapping[str, Any], embedding_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one embedding, selecting the requested metadata layer exactly."""
    path = resolve_embedding_path(record, embedding_dir)
    sidecar_path = _resolve_sidecar(path)
    if not path.exists():
        raise DataContractError(f"missing embedding for {_record_key(record)}: {path}")
    if not sidecar_path.exists():
        raise DataContractError(f"missing embedding sidecar for {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    layers = [int(item) for item in sidecar.get("layers", [])]
    if layer not in layers:
        raise DataContractError(
            f"embedding {path} does not contain requested layer {layer}; metadata layers={layers}"
        )
    array = np.asarray(np.load(path, allow_pickle=False))
    if array.ndim == 2:
        selected = array
    elif array.ndim == 3:
        if len(layers) == array.shape[0]:
            selected = array[layers.index(layer)]
        elif array.shape[0] == 1 and len(layers) == 1:
            selected = array[0]
        else:
            raise DataContractError(f"cannot map metadata layers={layers} to embedding shape={array.shape}")
    else:
        raise DataContractError(f"embedding must have 2 or 3 dimensions, got {array.shape}")
    selected = np.asarray(selected, dtype=np.float32)
    if selected.ndim != 2 or selected.shape[0] == 0 or selected.shape[1] <= 0:
        raise DataContractError(f"invalid selected embedding shape {selected.shape} for {path}")
    if not np.isfinite(selected).all():
        raise DataContractError(f"embedding contains NaN/Inf: {path}")
    frame_times = np.asarray(sidecar.get("frame_times_s", []), dtype=np.float32)
    if frame_times.ndim != 1 or len(frame_times) != selected.shape[0]:
        # Some compact synthetic sidecars omit frame times; use the documented
        # 320-sample/16-kHz convention only when metadata explicitly permits it.
        if not sidecar.get("allow_implicit_frame_times", False):
            raise DataContractError(
                f"frame_times_s length mismatch for {path}: {len(frame_times)} vs {selected.shape[0]}"
            )
        frame_times = (np.arange(selected.shape[0], dtype=np.float32) + 0.5) * 320 / 16000
    if not np.isfinite(frame_times).all() or np.any(np.diff(frame_times) < 0):
        raise DataContractError(f"invalid frame times for {path}")
    expected_dim = sidecar.get("embedding_dim")
    if expected_dim is not None and int(expected_dim) != selected.shape[1]:
        raise DataContractError(f"embedding dimension metadata mismatch for {path}")
    return selected, frame_times, {
        "path": str(path),
        "path_sha256": sha256_file(path),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "layers": layers,
        "selected_layer": layer,
        "shape": list(selected.shape),
        "embedding_dim": int(selected.shape[1]),
        "sample_rate": sidecar.get("sample_rate"),
        "frame_stride_samples": sidecar.get("frame_stride_samples"),
        "audio_sha256": sidecar.get("audio_sha256"),
    }


def _token_label(token: Mapping[str, Any]) -> str:
    value = token.get("token") or token.get("phone") or token.get("label")
    label = str(value).strip() if value is not None else ""
    if not label or label.lower() in {"sil", "sp", "spn", "pau", "<eps>"}:
        return ""
    return label


def assign_frame_labels(frame_times: np.ndarray, tokens: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    """Assign each frame center to its MFA phone, leaving silence as empty."""
    labels = np.full(len(frame_times), "", dtype="<U64")
    counters = Counter()
    for token in tokens:
        label = _token_label(token)
        if not label:
            counters["silence_or_empty_tokens"] += 1
            continue
        start, end = float(token["start_s"]), float(token["end_s"])
        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            counters["no_frame_center_tokens"] += 1
            continue
        labels[mask] = label
        counters["speech_tokens_with_frames"] += 1
    counters["frames_total"] = int(len(frame_times))
    counters["frames_labeled"] = int(np.count_nonzero(labels))
    counters["frames_unlabeled"] = int(len(frame_times) - np.count_nonzero(labels))
    return labels, dict(counters)


def _occurrence_mismatch_keys(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return paired keys whose independent MFA phone sequences differ."""
    grouped = _group_records(records)
    mismatches: set[str] = set()
    for key, arms in grouped.items():
        if set(arms) != {"natural", "tts"}:
            continue
        natural_labels = [_token_label(token) for token in arms["natural"]["tokens"] if _token_label(token)]
        tts_labels = [_token_label(token) for token in arms["tts"]["tokens"] if _token_label(token)]
        if natural_labels != tts_labels:
            mismatches.add(key)
    return mismatches


def _group_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[_record_key(record)][_condition(record)] = record
    return grouped


def _record_frame_data(record: Mapping[str, Any], embedding_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    features, frame_times, embedding_meta = load_embedding(record, embedding_dir, layer)
    labels, label_diag = assign_frame_labels(frame_times, record["tokens"])
    diagnostics = {**embedding_meta, **label_diag, "paired_key": _record_key(record), "condition": _condition(record)}
    return features, frame_times, labels, diagnostics


def _iter_frame_data(records: Sequence[Mapping[str, Any]], embedding_dir: Path, layer: int) -> Iterable[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    for record in records:
        features, frame_times, labels, diagnostics = _record_frame_data(record, embedding_dir, layer)
        yield record, features, frame_times, labels, diagnostics


def build_phone_prototypes(
    records: Sequence[Mapping[str, Any]],
    embedding_dir: Path,
    *,
    layer: int = DEFAULT_LAYER,
    min_support: int = DEFAULT_MIN_SUPPORT,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[PrototypeTable, dict[str, Any]]:
    """Estimate natural/TTS centroids from the supplied records only.

    Callers must pass records from the train split.  The function checks this
    explicitly to prevent accidentally fitting prototypes on held-out data.
    """
    if min_support <= 0:
        raise ValueError("min_support must be positive")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0, 1]")
    if not records:
        raise DataContractError("cannot build prototypes from an empty record set")
    non_train = sorted({_split(record) for record in records if _split(record) != "train"})
    if non_train:
        raise DataContractError(f"prototype builder received non-train records: {non_train}")
    grouped = _group_records(records)
    validate_pair_splits(records)
    sums: dict[str, dict[str, np.ndarray]] = {"natural": {}, "tts": {}}
    counts: dict[str, Counter[str]] = {"natural": Counter(), "tts": Counter()}
    diagnostics = Counter()
    dim: int | None = None
    for record, features, _times, labels, info in _iter_frame_data(records, embedding_dir, layer):
        if dim is None:
            dim = int(features.shape[1])
        if features.shape[1] != dim:
            raise DataContractError("embedding dimensions differ across records")
        condition = _condition(record)
        diagnostics.update({key: int(value) for key, value in info.items() if isinstance(value, (int, np.integer))})
        for label in sorted(set(labels.tolist())):
            if not label:
                continue
            values = features[labels == label].astype(np.float64)
            if label not in sums[condition]:
                sums[condition][label] = np.zeros(dim, dtype=np.float64)
            sums[condition][label] += np.sum(values, axis=0)
            counts[condition][label] += len(values)
    usable = sorted(
        set(sums["natural"]) & set(sums["tts"])
        & {label for label, count in counts["natural"].items() if count >= min_support}
        & {label for label, count in counts["tts"].items() if count >= min_support}
    )
    if not usable:
        raise DataContractError("no phone labels pass two-sided minimum support gate")
    natural = {label: (sums["natural"][label] / counts["natural"][label]).astype(np.float32) for label in usable}
    tts = {label: (sums["tts"][label] / counts["tts"][label]).astype(np.float32) for label in usable}
    delta = {label: (tts[label] - natural[label]).astype(np.float32) for label in usable}
    table = PrototypeTable(
        natural=natural,
        tts=tts,
        delta=delta,
        support_natural={label: int(counts["natural"][label]) for label in usable},
        support_tts={label: int(counts["tts"][label]) for label in usable},
        dim=int(dim or 0),
        min_support=int(min_support),
        alpha=float(alpha),
    )
    diagnostics.update({
        "usable_phone_count": len(usable),
        "all_natural_phone_count": len(sums["natural"]),
        "all_tts_phone_count": len(sums["tts"]),
        "discarded_phone_count": len(set(sums["natural"]) | set(sums["tts"])) - len(usable),
    })
    return table, {"support": table.to_json(), "counts": dict(diagnostics)}


def build_frame_batches(
    records: Sequence[Mapping[str, Any]],
    embedding_dir: Path,
    prototypes: PrototypeTable,
    *,
    layer: int = DEFAULT_LAYER,
    require_supported: bool = True,
    input_condition: str = "natural",
) -> tuple[list[FrameBatch], dict[str, Any]]:
    """Load one input condition's frames and mask train-supported labels.

    The adapter MVP is a natural-to-TTS transform, so callers must explicitly
    select the natural arm for training and held-out evaluation.  Prototype
    construction remains the only operation that reads both arms.
    """
    if input_condition not in {"natural", "tts"}:
        raise ValueError("input_condition must be natural or tts")
    records = [record for record in records if _condition(record) == input_condition]
    if not records:
        raise DataContractError(f"no {input_condition} records supplied for feature batches")
    batches: list[FrameBatch] = []
    summary = Counter()
    for record, features, frame_times, labels, diagnostics in _iter_frame_data(records, embedding_dir, layer):
        supported = np.array([label in prototypes.delta for label in labels], dtype=bool)
        valid = (labels != "") & supported
        if require_supported and not np.any(valid):
            summary["records_without_supported_frames"] += 1
            continue
        summary.update({key: int(value) for key, value in diagnostics.items() if isinstance(value, (int, np.integer))})
        summary["supported_frames"] += int(valid.sum())
        summary["candidate_frames"] += int((labels != "").sum())
        batches.append(
            FrameBatch(
                key=_record_key(record),
                speaker_id=_speaker(record),
                split=_split(record),
                features=features,
                labels=labels,
                valid_mask=valid,
                frame_times=frame_times,
                diagnostics={**diagnostics, "supported_frames": int(valid.sum())},
            )
        )
    if not batches:
        raise DataContractError("no usable frame batches after support filtering")
    summary["batch_count"] = len(batches)
    summary["coverage"] = float(summary["supported_frames"] / max(1, summary["candidate_frames"]))
    return batches, dict(summary)


def _prototype_tensors(table: PrototypeTable, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    return (
        {label: torch.from_numpy(value).to(device=device) for label, value in table.natural.items()},
        {label: torch.from_numpy(value).to(device=device) for label, value in table.tts.items()},
        {label: torch.from_numpy(value).to(device=device) for label, value in table.delta.items()},
    )


def compute_feature_loss(
    prediction: Tensor,
    source: Tensor,
    labels: Sequence[str],
    valid_mask: Tensor,
    prototypes: PrototypeTable,
    *,
    weights: Mapping[str, float] = LOSS_WEIGHTS,
) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
    """Compute the interpretable phone-prototype loss for one batch."""
    if prediction.shape != source.shape or prediction.ndim != 3:
        raise ValueError("prediction and source must have matching [batch, frames, dim] shapes")
    if valid_mask.shape != prediction.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [batch, frames]")
    if prediction.shape[0] != 1 or len(labels) != prediction.shape[1]:
        raise ValueError("this minimal loss expects one utterance and one label per frame")
    if not torch.isfinite(prediction).all() or not torch.isfinite(source).all():
        raise ValueError("loss inputs must be finite")
    _natural, tts_centroids, delta_centroids = _prototype_tensors(prototypes, prediction.device)
    valid_indices = valid_mask[0].nonzero(as_tuple=False).flatten().tolist()
    if not valid_indices:
        raise DataContractError("loss received no valid supported frames")
    target_rows: list[Tensor] = []
    tts_rows: list[Tensor] = []
    delta_rows: list[Tensor] = []
    selected = []
    for index in valid_indices:
        label = str(labels[index])
        if label not in delta_centroids:
            continue
        selected.append(index)
        target_rows.append(source[0, index] + prototypes.alpha * delta_centroids[label])
        tts_rows.append(tts_centroids[label])
        delta_rows.append(delta_centroids[label])
    if not selected:
        raise DataContractError("loss received no labels in prototype table")
    pred = prediction[0, selected]
    src = source[0, selected]
    target = torch.stack(target_rows)
    tts_target = torch.stack(tts_rows)
    delta_target = torch.stack(delta_rows)
    terms = {
        "prototype_shift_smooth_l1": F.smooth_l1_loss(pred, target),
        "tts_centroid_mse": F.mse_loss(pred, tts_target),
        "identity_mse": F.mse_loss(pred, src),
        "delta_l2": torch.mean((pred - src) ** 2),
    }
    total = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    for name, term in terms.items():
        total = total + float(weights[name]) * term
    if not torch.isfinite(total):
        raise FloatingPointError("feature loss is NaN/Inf")
    debug = {
        "valid_frames": len(selected),
        "target_shift_norm_mean": float(torch.linalg.vector_norm(target - src, dim=-1).mean().item()),
        "prototype_delta_norm_mean": float(torch.linalg.vector_norm(delta_target, dim=-1).mean().item()),
        "source_stats": feature_statistics(src),
        "target_stats": feature_statistics(target),
        "prediction_stats": feature_statistics(pred),
    }
    return total, terms, debug


def _tensor_terms_to_float(terms: Mapping[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().item()) for key, value in terms.items()}


def evaluate_batches(
    model: ResidualFeatureAdapter,
    batches: Sequence[FrameBatch],
    prototypes: PrototypeTable,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate adapter and identity baselines without updating parameters."""
    model.eval()
    totals: Counter[str] = Counter()
    counts = 0
    residual_norms: list[float] = []
    identity_terms: Counter[str] = Counter()
    adapter_terms: Counter[str] = Counter()
    per_phone: dict[str, Counter[str]] = defaultdict(Counter)
    with torch.no_grad():
        for batch in batches:
            source, mask, labels = batch.tensor(device)
            prediction = model(source, mask)
            adapter_loss, terms, debug = compute_feature_loss(prediction, source, labels, mask, prototypes)
            identity = source.clone()
            _identity_loss, identity_loss_terms, _ = compute_feature_loss(identity, source, labels, mask, prototypes)
            adapter_terms.update({name: float(value.item()) for name, value in terms.items()})
            identity_terms.update({name: float(value.item()) for name, value in identity_loss_terms.items()})
            totals["adapter_total"] += float(adapter_loss.item())
            residual = prediction - source
            selected = mask[0]
            residual_norms.append(float(torch.linalg.vector_norm(residual[0, selected], dim=-1).mean().item()))
            for index in selected.nonzero(as_tuple=False).flatten().tolist():
                label = labels[index]
                per_phone[label]["frames"] += 1
                p = prediction[0, index]
                s = source[0, index]
                t = torch.from_numpy(prototypes.tts[label]).to(device=device)
                n = torch.from_numpy(prototypes.natural[label]).to(device=device)
                per_phone[label]["adapter_to_tts"] += float(torch.mean((p - t) ** 2).item())
                per_phone[label]["identity_to_tts"] += float(torch.mean((s - t) ** 2).item())
                per_phone[label]["adapter_to_natural"] += float(torch.mean((p - n) ** 2).item())
                per_phone[label]["identity_to_natural"] += float(torch.mean((s - n) ** 2).item())
            counts += 1
    if counts == 0:
        raise DataContractError("cannot evaluate an empty batch list")
    phone_rows: dict[str, Any] = {}
    for label, values in sorted(per_phone.items()):
        frames = max(1, int(values["frames"]))
        phone_rows[label] = {
            "frames": frames,
            "adapter_to_tts_mse": values["adapter_to_tts"] / frames,
            "identity_to_tts_mse": values["identity_to_tts"] / frames,
            "adapter_to_natural_mse": values["adapter_to_natural"] / frames,
            "identity_to_natural_mse": values["identity_to_natural"] / frames,
        }
    return {
        "batch_count": counts,
        "adapter_loss": totals["adapter_total"] / counts,
        "identity_loss": sum(float(value) * float(LOSS_WEIGHTS[name]) for name, value in identity_terms.items()) / counts,
        "adapter_loss_terms_sum": {name: value / counts for name, value in adapter_terms.items()},
        "identity_loss_terms_sum": {name: value / counts for name, value in identity_terms.items()},
        "adapter_residual_norm_mean": float(np.mean(residual_norms)),
        "per_phone": phone_rows,
        "coverage": float(sum(int(values["frames"]) for values in per_phone.values()) / max(1, sum(len(batch.labels) for batch in batches))),
    }


def _environment() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        commit = "unavailable"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": commit or "unavailable",
    }


def _records_for_split(records: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    return [dict(record) for record in records if _split(record) == split]


def train_model(
    train_batches: Sequence[FrameBatch],
    valid_batches: Sequence[FrameBatch],
    prototypes: PrototypeTable,
    config: RunConfig,
) -> tuple[ResidualFeatureAdapter, list[dict[str, Any]], dict[str, Any]]:
    """Train only on train batches and select the best checkpoint by validation loss."""
    set_seed(config.seed)
    device = torch.device(config.device)
    model = ResidualFeatureAdapter(
        input_dim=prototypes.dim,
        hidden_channels=config.hidden_channels,
        residual_scale=config.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_valid = float("inf")
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = Counter()
        debug_rows: list[dict[str, Any]] = []
        for batch in train_batches:
            source, mask, labels = batch.tensor(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source, mask)
            total, terms, debug = compute_feature_loss(prediction, source, labels, mask, prototypes)
            total.backward()
            debug_rows.append(debug)
            gradient_info = parameter_statistics(model)
            if not all(math.isfinite(float(value)) for value in gradient_info.values()):
                raise FloatingPointError("non-finite gradient/parameter diagnostics")
            optimizer.step()
            totals["total"] += float(total.detach().item())
            for name, value in terms.items():
                totals[name] += float(value.detach().item())
        train_row = {"total": totals["total"] / max(1, len(train_batches))}
        train_row.update({name: totals[name] / max(1, len(train_batches)) for name in LOSS_WEIGHTS})
        valid_row = evaluate_batches(model, valid_batches, prototypes, device)
        row = {
            "epoch": epoch,
            "train": train_row,
            "valid": valid_row,
            "gradient_norm_mean": float(np.mean([parameter_statistics(model)["gradient_norm"] for _ in [0]])),
            "parameter_norm": parameter_statistics(model)["parameter_norm"],
            "input_output_debug": {
                "train_prediction_mean": float(np.mean([item["prediction_stats"]["mean"] for item in debug_rows])),
                "train_target_mean": float(np.mean([item["target_stats"]["mean"] for item in debug_rows])),
                "train_target_shift_norm_mean": float(np.mean([item["target_shift_norm_mean"] for item in debug_rows])),
            },
        }
        history.append(row)
        if float(valid_row["adapter_loss"]) < best_valid:
            best_valid = float(valid_row["adapter_loss"])
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    return model, history, {"best_valid_loss": best_valid, "epochs": len(history)}


def save_checkpoint(
    path: Path,
    model: ResidualFeatureAdapter,
    config: RunConfig,
    prototypes: PrototypeTable,
    metadata: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    """Save a torch checkpoint with all target/provenance policy metadata."""
    payload = {
        "schema_version": 1,
        "model_config": model.config(),
        "run_config": asdict(config),
        "loss_weights": dict(LOSS_WEIGHTS),
        "prototype": prototypes.to_json(),
        "metadata": dict(metadata),
        "history": list(history),
        "target_definition": "train-only phone centroid shift: x + alpha*(mu_tts-mu_natural)",
        "warnings": [
            "representation-only weak target; not verified TFG/SyncNet ground truth",
            "checkpoint is not a waveform enhancer",
        ],
        "state_dict": model.state_dict(),
    }
    # torch.save can serialize tensors, but metadata must still be strict JSON.
    _assert_json_finite({key: value for key, value in payload.items() if key != "state_dict"})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-pairs", type=int, default=0, help="limit each split for a deterministic smoke run")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records, manifest_meta = load_manifest(args.manifest)
    validation = manifest_meta["pair_validation"]
    train_records = _records_for_split(records, "train")
    valid_records = _records_for_split(records, "valid")
    test_records = _records_for_split(records, "test")
    if args.smoke_pairs > 0:
        def limit_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            keys = sorted({_record_key(item) for item in items})[: args.smoke_pairs]
            return [item for item in items if _record_key(item) in keys]
        train_records, valid_records, test_records = map(limit_pairs, (train_records, valid_records, test_records))
    if not train_records or not valid_records:
        raise DataContractError("training requires non-empty train and valid splits")
    table, prototype_meta = build_phone_prototypes(
        train_records,
        args.embedding_dir,
        layer=args.layer,
        min_support=args.min_support,
        alpha=args.alpha,
    )
    train_batches, train_diag = build_frame_batches(train_records, args.embedding_dir, table, layer=args.layer, input_condition="natural")
    valid_batches, valid_diag = build_frame_batches(valid_records, args.embedding_dir, table, layer=args.layer, input_condition="natural")
    # Load test only to validate provenance and produce a final report later; it
    # is not passed to train_model and cannot affect checkpoint selection.
    test_batches, test_diag = ([], {"not_evaluated_during_training": True})
    if test_records:
        test_batches, test_diag = build_frame_batches(test_records, args.embedding_dir, table, layer=args.layer, input_condition="natural")
    config = RunConfig(
        layer=args.layer,
        min_support=args.min_support,
        alpha=args.alpha,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        residual_scale=args.residual_scale,
        seed=args.seed,
        device=args.device,
    )
    model, history, train_summary = train_model(train_batches, valid_batches, table, config)
    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_meta["manifest_sha256"],
        "embedding_dir": str(args.embedding_dir),
        "split_validation": validation,
        "split_keys": validation["split_keys"],
        "train_speakers": validation["speaker_ids"]["train"],
        "valid_speakers": validation["speaker_ids"]["valid"],
        "test_speakers": validation["speaker_ids"]["test"],
        "train_diagnostics": train_diag,
        "valid_diagnostics": valid_diag,
        "test_diagnostics": test_diag,
        "prototype_hash": sha256_json(table.to_json()),
        "environment": _environment(),
        "source_target_warning": "representation-only weak target; no TFG/SyncNet objective",
    }
    save_checkpoint(output_dir / "checkpoint.pt", model, config, table, metadata, history)
    write_json(output_dir / "prototypes.json", {"schema_version": 1, **table.to_json(), **prototype_meta})
    write_json(output_dir / "training_debug.json", {"schema_version": 1, "config": asdict(config), "metadata": metadata, "history": history, "summary": train_summary})
    write_json(output_dir / "metrics.json", {"schema_version": 1, "validation": evaluate_batches(model, valid_batches, table, torch.device(args.device)), "identity_only": evaluate_identity(valid_batches, table, torch.device(args.device)), "test_not_used_for_selection": True})
    with (output_dir / "epoch_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            _assert_json_finite(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "train_pairs": validation["split_pair_counts"]["train"], "valid_pairs": validation["split_pair_counts"]["valid"], "test_pairs": validation["split_pair_counts"]["test"], "prototype_phones": len(table.labels), "best_valid_loss": train_summary["best_valid_loss"]}, ensure_ascii=False, sort_keys=True))


def evaluate_identity(batches: Sequence[FrameBatch], prototypes: PrototypeTable, device: torch.device) -> dict[str, Any]:
    """Evaluate the natural identity/no-op baseline using the same loss contract."""
    rows = []
    for batch in batches:
        source, mask, labels = batch.tensor(device)
        _loss, terms, debug = compute_feature_loss(source, source, labels, mask, prototypes)
        rows.append({"total": float(sum(float(LOSS_WEIGHTS[name]) * float(value.item()) for name, value in terms.items())), "terms": _tensor_terms_to_float(terms), "debug": debug})
    if not rows:
        raise DataContractError("identity baseline received no batches")
    return {
        "batch_count": len(rows),
        "loss": float(np.mean([row["total"] for row in rows])),
        "terms": {name: float(np.mean([row["terms"][name] for row in rows])) for name in LOSS_WEIGHTS},
        "identity_baseline": True,
    }


if __name__ == "__main__":
    main()
