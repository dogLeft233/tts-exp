#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Feasibility validation for a Ditto-native 1024-D feature adapter.

This runner is intentionally separate from the representation-only 768-D
experiment. It extracts features through Ditto's native streaming HuBERT
frontend, trains a small phone-prototype residual adapter on supplied paired
records, and can inject the frozen adapter into the PyTorch Ditto pipeline
without changing Ditto source or TensorRT assets.

The resulting downstream result is training-set feasibility evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

try:
    from tts_feature_head import ResidualFeatureAdapter
except ImportError:  # pragma: no cover
    from scripts.tts_feature_head import ResidualFeatureAdapter


FRAME_RATE = 25.0
DITTO_DIM = 1024


NON_SPEECH_LABELS = {
    "sil", "sp", "spn", "pau", "<eps>", "<unk>", "unknown", "<unknown>",
    "<sil>", "h#", "noise", "<noise>", "[noise]", "nsn", "br", "laugh",
    "<laugh>", "vocalization", "<vocalization>", "cough", "<cough>",
    "click", "<click>", "",
}


def finite_array(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != DITTO_DIM:
        raise ValueError(f"{name} must have shape [frames, {DITTO_DIM}], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    base = base_dir.resolve()
    resolved = (base / path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path escapes base directory: {value}") from exc
    return resolved


def _normalise_label(value: Any) -> str:
    """Return the canonical case-insensitive phone label used by prototypes."""
    return str(value or "").strip().casefold()


def _explicit_record_key(record: dict[str, Any]) -> str:
    """Resolve an explicit pair key and reject contradictory identity fields."""
    values: list[tuple[str, str]] = []
    for field in ("pair_key", "paired_key", "sample_key"):
        value = record.get(field)
        if value is not None and str(value).strip():
            values.append((field, str(value).strip()))
    utterance_id = str(record.get("utterance_id", "")).strip()
    derived = _normalise_manifest_key({"utterance_id": utterance_id}) if utterance_id else None
    if derived:
        values.append(("utterance_id", derived))
    if not values:
        raise ValueError("record is missing pair_key/paired_key/sample_key/utterance_id")
    keys = {value for _, value in values}
    if len(keys) != 1:
        details = ", ".join(f"{field}={value!r}" for field, value in values)
        raise ValueError(f"conflicting pair identity fields: {details}")
    return values[0][1]


def _record_key(record: dict[str, Any]) -> str:
    return _explicit_record_key(record)


def _token_label_value(token: dict[str, Any]) -> str:
    for field in ("phoneme", "token", "label"):
        value = token.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _validate_pair(record: dict[str, Any]) -> None:
    key = _record_key(record)
    natural_tokens = record.get("natural_tokens")
    tts_tokens = record.get("tts_tokens")
    if not isinstance(natural_tokens, list) or not isinstance(tts_tokens, list):
        raise ValueError(f"{key} is missing natural/TTS token lists")
    if not natural_tokens or not tts_tokens:
        raise ValueError(f"{key} has no tokens on one or both arms")
    durations = record.get("audio_durations_s", {})
    if not isinstance(durations, dict):
        durations = {}
    for arm, tokens in (("natural", natural_tokens), ("tts", tts_tokens)):
        previous_end = 0.0
        duration_value = durations.get(arm, record.get(f"{arm}_duration_s"))
        duration = None if duration_value is None else float(duration_value)
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError(f"{key} has invalid {arm} audio duration")
        for index, token in enumerate(tokens):
            if not isinstance(token, dict):
                raise ValueError(f"{key} has non-object {arm} token {index}")
            label = _token_label_value(token)
            if not label:
                # MFA uses an empty text field for ordinary silence intervals.
                label = "<sil>"
            try:
                start = float(token["start_s"])
                end = float(token["end_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{key} has malformed {arm} token interval {index}") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"invalid token interval for {key} ({arm} token {index})")
            if start < previous_end:
                raise ValueError(f"overlapping token intervals for {key} ({arm})")
            if start > previous_end + 1e-6:
                raise ValueError(f"gap between token intervals for {key} ({arm})")
            if duration is not None and end > duration + 1e-6:
                raise ValueError(f"{arm} token exceeds audio duration for {key}: {end} > {duration}")
            previous_end = end
        if duration is not None:
            _validate_arm_alignment(tokens, duration, key, arm)


def _normalise_manifest_key(source: dict[str, Any]) -> str | None:
    values: list[tuple[str, str]] = []
    for field in ("pair_key", "paired_key", "sample_key"):
        value = source.get(field)
        if value is not None and str(value).strip():
            values.append((field, str(value).strip()))
    utterance_id = str(source.get("utterance_id", "")).strip().strip("/")
    parts = utterance_id.split("/") if utterance_id else []
    if len(parts) >= 3 and parts[-1] in {"natural", "tts"}:
        values.append(("utterance_id", parts[-2]))
    else:
        for suffix in (":natural:raw", ":tts:raw"):
            if utterance_id.endswith(suffix):
                values.append(("utterance_id", utterance_id[: -len(suffix)]))
                break
    if not values:
        return None
    keys = {value for _, value in values}
    if len(keys) != 1:
        details = ", ".join(f"{field}={value!r}" for field, value in values)
        raise ValueError(f"conflicting pair identity fields: {details}")
    return values[0][1]


def _resolve_manifest_audio(value: str | Path, manifest_path: Path, repo_root: Path | None) -> tuple[Path, str]:
    raw = Path(value)
    if raw.is_absolute():
        candidates = [(raw, "absolute")]
    else:
        candidates = [(manifest_path.parent / raw, "manifest_parent")]
        if repo_root is not None:
            candidates.append((repo_root / raw, "repo_root"))
    existing: list[tuple[Path, str]] = []
    for candidate, base in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and (repo_root is None or resolved.is_relative_to(repo_root.resolve())):
            if all(resolved != prior[0] for prior in existing):
                existing.append((resolved, base))
    if len(existing) != 1:
        if not existing:
            raise FileNotFoundError(f"manifest audio path does not resolve: {value}")
        raise ValueError(f"manifest audio path is ambiguous: {value}")
    return existing[0]


def _validate_manifest_text(natural: dict[str, Any], tts: dict[str, Any], key: str) -> None:
    values = []
    for source in (natural, tts):
        for field in ("text", "transcript"):
            if source.get(field) is not None and str(source[field]).strip():
                values.append(str(source[field]).strip())
                break
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError(f"natural/TTS transcript mismatch for {key}")


def _manifest_to_pairs(
    path: Path,
    *,
    repo_root: Path | None = None,
    training_split: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records", payload.get("manifest"))
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("records JSON must contain a records or manifest list")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("manifest records must be objects")
        condition = str(source.get("condition", "")).strip().lower()
        if condition == "faster_qwen3":
            raise ValueError("manifest contains faster_qwen3; provide an explicit TTS condition or filter it before training")
        if condition not in {"natural", "tts"}:
            if any(field in source for field in ("natural_audio", "tts_audio")):
                raise ValueError("mixed explicit-pair and arm-level manifest records are unsupported")
            continue
        key = _normalise_manifest_key(source)
        if not key:
            raise ValueError("manifest record is missing pair_key/sample_key/utterance_id")
        if condition in grouped[key]:
            raise ValueError(f"duplicate {condition} arm for pair {key}")
        grouped[key][condition] = source
    pairs: list[dict[str, Any]] = []
    split_values: set[str] = set()
    for key, arms in grouped.items():
        if set(arms) != {"natural", "tts"}:
            continue
        natural, tts = arms["natural"], arms["tts"]
        natural_split = str(natural.get("split", "")).strip()
        tts_split = str(tts.get("split", "")).strip()
        if natural_split != tts_split:
            raise ValueError(f"natural/TTS split mismatch for {key}")
        if natural_split != training_split:
            continue
        _validate_manifest_text(natural, tts, key)
        natural_value = natural.get("audio_path") or natural.get("filepath")
        tts_value = tts.get("audio_path") or tts.get("filepath")
        if natural_value is None or tts_value is None:
            raise ValueError(f"missing audio path for {key}")
        natural_audio, natural_base = _resolve_manifest_audio(natural_value, path, repo_root)
        tts_audio, tts_base = _resolve_manifest_audio(tts_value, path, repo_root)
        pair = {
            "pair_key": key,
            "natural_audio": str(natural_audio),
            "tts_audio": str(tts_audio),
            "natural_tokens": natural.get("tokens", []),
            "tts_tokens": tts.get("tokens", []),
            "audio_durations_s": {
                "natural": natural.get("duration_s"),
                "tts": tts.get("duration_s"),
            },
            "split": natural_split,
            "natural_text": natural.get("text", natural.get("transcript")),
            "tts_text": tts.get("text", tts.get("transcript")),
            "speaker_id": natural.get("speaker_id"),
            "natural_speaker_id": natural.get("speaker_id"),
            "tts_speaker_id": tts.get("speaker_id"),
            "tts_provider": tts.get("tts_provider") or tts.get("provider"),
            "speaker_scope": {"natural": "source_author", "tts": "provider_condition"},
            "text": natural.get("text", natural.get("transcript")),
            "manifest_path_base": {"natural": natural_base, "tts": tts_base},
            "source_record_ids": {
                "natural": natural.get("utterance_id", natural.get("sample_id")),
                "tts": tts.get("utterance_id", tts.get("sample_id")),
            },
        }
        _validate_pair(pair)
        pairs.append(pair)
        split_values.add(natural_split)
    if not pairs:
        raise ValueError(f"no complete natural/TTS pairs found for training_split={training_split!r}")
    metadata = {
        "records_path": str(path),
        "records_sha256": sha256_file(path),
        "pair_count": len(pairs),
        "splits": sorted(split_values),
        "training_split": training_split,
        "manifest_schema": "arm_level_v2",
    }
    return pairs, metadata


def _normalise_pairs(records: list[dict[str, Any]], base_dir: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for record in records:
        pair = dict(record)
        if "natural_audio" not in pair or "tts_audio" not in pair:
            raise ValueError("explicit pair records require natural_audio and tts_audio")
        pair["pair_key"] = _record_key(pair)
        if pair["pair_key"] in seen_keys:
            raise ValueError(f"duplicate pair key: {pair['pair_key']}")
        seen_keys.add(pair["pair_key"])
        _validate_pair(pair)
        pair_texts = [str(pair.get(field, "")).strip() for field in ("natural_text", "tts_text") if str(pair.get(field, "")).strip()]
        if len(pair_texts) == 2 and pair_texts[0] != pair_texts[1]:
            raise ValueError(f"natural/TTS transcript mismatch for {pair['pair_key']}")
        pair_text = str(pair.get("text", pair.get("transcript", ""))).strip()
        if pair_text and any(value != pair_text for value in pair_texts):
            raise ValueError(f"explicit pair transcript mismatch for {pair['pair_key']}")
        pair["natural_audio"] = str(_resolve_path(pair["natural_audio"], base_dir))
        pair["tts_audio"] = str(_resolve_path(pair["tts_audio"], base_dir))
        if not Path(pair["natural_audio"]).is_file() or not Path(pair["tts_audio"]).is_file():
            raise FileNotFoundError(f"missing audio for {pair['pair_key']}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("no pairs supplied")
    return pairs


def _validate_arm_alignment(tokens: list[dict[str, Any]], duration: float | None, key: str, arm: str) -> None:
    """Require ordered, non-overlapping token spans with meaningful coverage."""
    if not tokens:
        raise ValueError(f"{key} has no {arm} tokens")
    previous_end = 0.0
    for index, token in enumerate(tokens):
        start = float(token["start_s"])
        end = float(token["end_s"])
        if start < previous_end:
            raise ValueError(f"overlapping token intervals for {key} ({arm})")
        if index == 0 and start > 1e-6:
            raise ValueError(f"{arm} token alignment starts after zero for {key}")
        if duration is not None and end > duration + 1e-6:
            raise ValueError(f"{arm} token exceeds audio duration for {key}: {end} > {duration}")
        previous_end = end
    if duration is not None and previous_end < duration - 1e-3:
        raise ValueError(f"{arm} token alignment does not cover audio duration for {key}: {previous_end} < {duration}")


def residual_target_diagnostics(
    delta: dict[str, np.ndarray], residual_scale: float, alpha: float = 0.5,
) -> dict[str, Any]:
    """Report centroid shifts that exceed the adapter's bounded residual."""
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale must be finite and positive")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    bound = float(residual_scale)
    component_limit = bound / alpha if alpha else math.inf
    unreachable: dict[str, int] = {}
    for label, value in delta.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"delta for {label!r} is not finite one-dimensional data")
        unreachable[label] = int(np.sum(np.abs(alpha * array) > bound + 1e-8))
    return {
        "residual_scale": bound,
        "target_alpha": float(alpha),
        "component_limit_on_delta": component_limit,
        "unreachable_labels": sorted(label for label, count in unreachable.items() if count),
        "unreachable_component_count": int(sum(unreachable.values())),
        "label_count": len(unreachable),
    }


def _validate_training_pairs(pairs: list[dict[str, Any]], training_split: str = "train") -> None:
    """Reject unlabelled or held-out records before building any prototype."""
    seen_keys: set[str] = set()
    for pair in pairs:
        key = str(pair.get("pair_key", _record_key(pair)))
        if key in seen_keys:
            raise ValueError(f"duplicate training pair key: {key}")
        seen_keys.add(key)
        split = str(pair.get("split", "")).strip().lower()
        if split != training_split:
            raise ValueError(
                f"training records must have split={training_split!r}; {key} has {split or 'missing split'!r}"
            )
        speaker = pair.get("speaker_id")
        if speaker is None or not str(speaker).strip():
            raise ValueError(f"training record {key} is missing speaker_id")


def feature_frames(wav2feat: Any, path: Path) -> np.ndarray:
    audio, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    features = wav2feat.wav2feat(audio.astype(np.float32), sr=sample_rate)
    return finite_array(features, str(path))


def token_label(frame_index: int, tokens: list[dict[str, Any]]) -> str | None:
    centre = (frame_index + 0.5) / FRAME_RATE
    for token in tokens:
        start = float(token["start_s"])
        end = float(token["end_s"])
        if start <= centre < end:
            return _token_label(token)


def _token_label(token: dict[str, Any]) -> str | None:
    if any(bool(token.get(field, False)) for field in ("is_silence", "is_noise", "is_unknown", "is_non_speech")):
        return None
    label = _normalise_label(_token_label_value(token))
    return None if label in NON_SPEECH_LABELS else (label or None)


def labels_for(features: np.ndarray, tokens: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([token_label(i, tokens) for i in range(len(features))], dtype=object)


def train_adapter(
    pairs: list[dict[str, Any]],
    wav2feat: Any,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    seed: int,
    records_metadata: dict[str, Any] | None = None,
) -> tuple[ResidualFeatureAdapter, dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    examples: list[tuple[np.ndarray, np.ndarray]] = []
    nat_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(DITTO_DIM, dtype=np.float64))
    tts_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(DITTO_DIM, dtype=np.float64))
    nat_count: dict[str, int] = defaultdict(int)
    tts_count: dict[str, int] = defaultdict(int)
    for pair in pairs:
        if "pair_key" not in pair:
            pair["pair_key"] = _record_key(pair)
        natural = finite_array(feature_frames(wav2feat, Path(pair["natural_audio"])), "natural features")
        tts = finite_array(feature_frames(wav2feat, Path(pair["tts_audio"])), "tts features")
        nat_labels = labels_for(natural, pair["natural_tokens"])
        tts_labels = labels_for(tts, pair["tts_tokens"])
        for frame, label in zip(natural, nat_labels):
            if label is not None:
                nat_sum[str(label)] += frame
                nat_count[str(label)] += 1
        for frame, label in zip(tts, tts_labels):
            if label is not None:
                tts_sum[str(label)] += frame
                tts_count[str(label)] += 1
        examples.append((natural, nat_labels))

    labels = sorted(set(nat_sum) & set(tts_sum))
    if not labels:
        raise ValueError("no phone labels have support on both arms")
    nat_centroid = {label: (nat_sum[label] / nat_count[label]).astype(np.float32) for label in labels}
    tts_centroid = {label: (tts_sum[label] / tts_count[label]).astype(np.float32) for label in labels}
    delta = {label: tts_centroid[label] - nat_centroid[label] for label in labels}

    model = ResidualFeatureAdapter(
        input_dim=DITTO_DIM,
        hidden_channels=64,
        dilations=(1, 2, 4),
        residual_scale=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history: list[dict[str, float]] = []
    label_ids = {label: index for index, label in enumerate(labels)}
    tts_matrix = torch.from_numpy(np.stack([tts_centroid[label] for label in labels])).to(device)
    delta_matrix = torch.from_numpy(np.stack([delta[label] for label in labels])).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_terms: list[dict[str, float]] = []
        random.shuffle(examples)
        for features, frame_labels in examples:
            valid = np.asarray([label in label_ids for label in frame_labels], dtype=bool)
            if not valid.any():
                continue
            x = torch.from_numpy(features).to(device).unsqueeze(0)
            y_ids = torch.tensor([label_ids[str(label)] if valid[i] else 0 for i, label in enumerate(frame_labels)], device=device)
            y = model(x)
            xf = x[0, valid]
            yf = y[0, valid]
            target_shift = xf + 0.5 * delta_matrix[y_ids[valid]]
            target_tts = tts_matrix[y_ids[valid]]
            shift_loss = torch.nn.functional.smooth_l1_loss(yf, target_shift)
            centroid_loss = torch.nn.functional.mse_loss(yf, target_tts)
            identity_loss = torch.nn.functional.mse_loss(yf, xf)
            residual_loss = torch.mean((yf - xf) ** 2)
            loss = shift_loss + 0.2 * centroid_loss + 0.05 * identity_loss + 0.01 * residual_loss
            if not torch.isfinite(loss):
                raise ValueError("non-finite Ditto-native adapter loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_terms.append({
                "loss": float(loss.detach().cpu()),
                "shift": float(shift_loss.detach().cpu()),
                "tts_centroid": float(centroid_loss.detach().cpu()),
                "identity": float(identity_loss.detach().cpu()),
                "residual": float(residual_loss.detach().cpu()),
                "frames": int(valid.sum()),
            })
        if not epoch_terms:
            raise ValueError("no labelled frames were available for training")
        total_frames = float(sum(term["frames"] for term in epoch_terms))
        history.append({
            "epoch": epoch + 1,
            "loss": float(sum(term["loss"] * term["frames"] for term in epoch_terms) / total_frames),
            "shift": float(sum(term["shift"] * term["frames"] for term in epoch_terms) / total_frames),
            "tts_centroid": float(sum(term["tts_centroid"] * term["frames"] for term in epoch_terms) / total_frames),
            "identity": float(sum(term["identity"] * term["frames"] for term in epoch_terms) / total_frames),
            "residual": float(sum(term["residual"] * term["frames"] for term in epoch_terms) / total_frames),
            "frames": int(total_frames),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "interface": {
            "input_dim": DITTO_DIM,
            "frame_rate": FRAME_RATE,
            "frontend": "Ditto Wav2FeatHubert / hubert_streaming_fix_kv.onnx",
            "aggregation": "mean of two 20ms HuBERT outputs -> 25 fps",
        },
        "model_config": {"input_dim": DITTO_DIM, "hidden_channels": 64, "dilations": [1, 2, 4], "residual_scale": 0.1},
        "state_dict": model.state_dict(),
        "labels": labels,
        "natural_centroid": {label: value.tolist() for label, value in nat_centroid.items()},
        "tts_centroid": {label: value.tolist() for label, value in tts_centroid.items()},
        "delta": {label: value.tolist() for label, value in delta.items()},
        "support_natural": dict(nat_count),
        "support_tts": dict(tts_count),
        "history": history,
        "training_pairs": [str(pair["pair_key"]) for pair in pairs],
        "training_audio_sha256": {
            str(pair["pair_key"]): {
                "natural": sha256_file(Path(pair["natural_audio"])),
                "tts": sha256_file(Path(pair["tts_audio"])),
            }
            for pair in pairs
        },
        "seed": seed,
        "validation_classification": "training-set feasibility only",
        "target_definition": {
            "alpha": 0.5,
            "loss_weights": {"shift": 1.0, "tts_centroid": 0.2, "identity": 0.05, "residual": 0.01},
            "residual_scale": 0.1,
        },
        "residual_bound_diagnostic": residual_target_diagnostics(delta, residual_scale=0.1, alpha=0.5),
        "records_metadata": records_metadata or {},
    }
    torch.save(checkpoint, output_dir / "ditto_native_adapter.pt")
    report = {
        "schema_version": 1,
        "interface": checkpoint["interface"],
        "training_pair_count": len(pairs),
        "prototype_phone_count": len(labels),
        "support_natural": dict(sorted(nat_count.items())),
        "support_tts": dict(sorted(tts_count.items())),
        "history": history,
        "checkpoint": str(output_dir / "ditto_native_adapter.pt"),
        "validation_classification": "training-set feasibility only",
    }
    (output_dir / "adapter_training.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return model.eval(), checkpoint


def load_adapter(path: Path, device: torch.device) -> ResidualFeatureAdapter:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported Ditto-native adapter checkpoint")
    interface = payload.get("interface", {})
    expected_interface = {
        "input_dim": DITTO_DIM,
        "frame_rate": FRAME_RATE,
        "frontend": "Ditto Wav2FeatHubert / hubert_streaming_fix_kv.onnx",
        "aggregation": "mean of two 20ms HuBERT outputs -> 25 fps",
    }
    for key, expected in expected_interface.items():
        if interface.get(key) != expected:
            raise ValueError(f"checkpoint interface mismatch for {key}: {interface.get(key)!r}")
    config = payload.get("model_config", {})
    if not isinstance(config, dict):
        raise ValueError("adapter checkpoint model_config must be an object")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("adapter checkpoint state_dict is missing or empty")
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("adapter checkpoint state_dict contains an invalid entry")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"adapter checkpoint tensor is non-finite: {name}")
    try:
        input_dim = int(config["input_dim"])
        hidden_channels = int(config["hidden_channels"])
        dilations = tuple(int(value) for value in config["dilations"])
        residual_scale = float(config["residual_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("adapter checkpoint has incomplete model_config") from exc
    if any(not math.isfinite(float(value)) for value in (input_dim, hidden_channels, residual_scale)):
        raise ValueError("adapter checkpoint model_config contains non-finite values")
    if input_dim != DITTO_DIM or hidden_channels <= 0 or not dilations or not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("adapter checkpoint has invalid model_config")
    model = ResidualFeatureAdapter(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        dilations=dilations,
        residual_scale=residual_scale,
    ).to(device)
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    observed_keys = set(state_dict)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(f"adapter checkpoint state_dict keys mismatch (missing={missing}, extra={extra})")
    for name, expected in expected_state.items():
        observed = state_dict[name]
        if tuple(observed.shape) != tuple(expected.shape):
            raise ValueError(
                f"adapter checkpoint tensor shape mismatch for {name}: "
                f"{tuple(observed.shape)} != {tuple(expected.shape)}"
            )
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


class RandomResidualAdapter(torch.nn.Module):
    """Deterministic-seed-controlled random residual baseline."""

    def __init__(self, scale: float, seed: int) -> None:
        super().__init__()
        if scale < 0 or not math.isfinite(scale):
            raise ValueError("random residual scale must be finite and non-negative")
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)), requires_grad=False)
        self._seed = int(seed)
        self._calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=features.device)
        generator.manual_seed(self._seed + self._calls)
        self._calls += 1
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        )
        return features + self.scale.to(dtype=features.dtype) * torch.tanh(noise)


class DittoFeatureHook:
    """Forward-compatible wrapper for Ditto's Wav2Feat facade.

    The adapter body is non-causal, so online calls are rejected by default
    rather than silently producing chunk-local features that differ from the
    offline full-sequence result.  ``allow_online_context_mismatch`` exists only
    for explicitly exploratory runs and must not be interpreted as equivalence.
    """

    def __init__(
        self,
        original_frontend: Any,
        adapter: torch.nn.Module | None,
        device: torch.device,
        allow_online_context_mismatch: bool = False,
    ) -> None:
        if int(getattr(original_frontend, "feat_dim", -1)) != DITTO_DIM:
            raise ValueError("Ditto frontend must expose feat_dim=1024")
        self._original = original_frontend
        self._adapter = adapter
        self._device = device
        self._allow_online_context_mismatch = allow_online_context_mismatch

    @property
    def feat_dim(self) -> int:
        return int(self._original.feat_dim)

    @property
    def support_streaming(self) -> bool:
        # The non-causal adapter cannot satisfy Ditto's chunk-local contract.
        return bool(self._original.support_streaming) if self._adapter is None else False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def _apply(self, features: np.ndarray) -> np.ndarray:
        native = finite_array(features, "Ditto native features")
        if self._adapter is None:
            return native
        with torch.no_grad():
            tensor = torch.from_numpy(native).to(self._device).unsqueeze(0)
            result = self._adapter(tensor).squeeze(0).cpu().numpy()
        return finite_array(result, "adapted Ditto features")

    def __call__(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        norm_mean_std: Any = None,
        chunksize: tuple[int, int, int] = (3, 5, 2),
    ) -> np.ndarray:
        if self._adapter is not None and not self._allow_online_context_mismatch:
            raise RuntimeError(
                "online Ditto injection is disabled for the non-causal adapter; "
                "use offline mode or explicitly opt into chunk-local context mismatch"
            )
        features = self._original(
            audio,
            sr=sr,
            norm_mean_std=norm_mean_std,
            chunksize=chunksize,
        )
        return self._apply(features)

    def wav2feat(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        norm_mean_std: Any = None,
        chunksize: tuple[int, int, int] = (3, 5, 2),
    ) -> np.ndarray:
        features = self._original.wav2feat(
            audio,
            sr=sr,
            norm_mean_std=norm_mean_std,
            chunksize=chunksize,
        )
        return self._apply(features)


def _load_ditto_inference(ditto_dir: Path) -> Any:
    ditto_dir = ditto_dir.resolve()
    inference_path = ditto_dir / "inference.py"
    if not inference_path.is_file():
        raise FileNotFoundError(inference_path)
    spec = importlib.util.spec_from_file_location("ditto_native_inference", inference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {inference_path}")
    inference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = inference
    original_path = list(sys.path)
    sys.path.insert(0, str(ditto_dir))
    try:
        spec.loader.exec_module(inference)
    finally:
        sys.path[:] = original_path
    return inference


def run_ditto_with_adapter(
    args: argparse.Namespace,
    adapter: torch.nn.Module | None,
    install_hook: bool = False,
) -> None:
    ditto_dir = Path(args.ditto_dir).resolve()
    inference = _load_ditto_inference(ditto_dir)

    inference.seed_everything(args.seed)
    sdk = inference.StreamSDK(args.cfg_pkl, args.data_root)
    if install_hook:
        original_frontend = sdk.wav2feat
        device = next(adapter.parameters()).device if adapter is not None else torch.device("cpu")
        sdk.wav2feat = DittoFeatureHook(
            original_frontend,
            adapter,
            device,
            allow_online_context_mismatch=bool(getattr(args, "allow_online_context_mismatch", False)),
        )
    inference.run(sdk, str(args.audio_path), str(args.source_path), str(args.output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "ordinary", "identity", "adapted", "random"), required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--ditto-dir", type=Path)
    parser.add_argument("--data-root")
    parser.add_argument("--cfg-pkl")
    parser.add_argument("--hubert-model", type=Path)
    parser.add_argument("--audio-path", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--training-split", default="train")
    parser.add_argument("--random-scale", type=float, default=0.1)
    parser.add_argument("--allow-online-context-mismatch", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        if args.records is None or args.hubert_model is None:
            raise ValueError("train mode requires --records and --hubert-model")
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.hubert_model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        class NativeFrontend:
            def wav2feat(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                num_f = math.ceil(len(audio) / 16000 * FRAME_RATE)
                if num_f <= 0:
                    raise ValueError("audio must contain at least one sample")
                chunksize = (3, 5, 2)
                split_len = int(sum(chunksize) * 0.04 * 16000) + 80
                pad = np.concatenate([np.zeros((split_len - int(sum(chunksize[1:]) * 0.04 * 16000),), dtype=np.float32), audio.astype(np.float32), np.zeros((split_len,), dtype=np.float32)])
                result: list[np.ndarray] = []
                for index in range(0, num_f, chunksize[1]):
                    start = int(index * 0.04 * 16000)
                    chunk = pad[start:start + split_len]
                    if len(chunk) < split_len:
                        chunk = np.pad(chunk, (0, split_len - len(chunk)))
                    encoding = np.asarray(session.run(None, {"input_values": chunk.reshape(1, -1)})[0])
                    valid = encoding[-sum(chunksize[1:]) * 2:-chunksize[2] * 2]
                    result.append(valid.reshape(chunksize[1], 2, DITTO_DIM).mean(1))
                return np.concatenate(result, axis=0)[:num_f]

        payload = json.loads(args.records.read_text(encoding="utf-8"))
        raw_records = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_records, list):
            raise ValueError("records JSON must contain a list")
        if raw_records and all("natural_audio" in record and "tts_audio" in record for record in raw_records):
            pairs = _normalise_pairs([dict(record) for record in raw_records], args.records.parent)
            records_meta = {"records_path": str(args.records), "records_sha256": sha256_file(args.records), "pair_count": len(pairs), "training_split": args.training_split}
        else:
            pairs, records_meta = _manifest_to_pairs(
                args.records,
                repo_root=Path(__file__).resolve().parent.parent,
                training_split=args.training_split,
            )
            for pair in pairs:
                if not Path(pair["natural_audio"]).is_file() or not Path(pair["tts_audio"]).is_file():
                    raise FileNotFoundError(f"missing audio for {_record_key(pair)}")
        _validate_training_pairs(pairs, args.training_split)
        train_adapter(pairs, NativeFrontend(), args.output_dir, device, args.epochs, args.seed, records_meta)
        print(json.dumps({"mode": "train", "checkpoint": str(args.output_dir / "ditto_native_adapter.pt"), "records": records_meta}, sort_keys=True))
        return

    adapter: torch.nn.Module | None
    if args.mode in {"ordinary", "identity"}:
        adapter = None
    elif args.mode == "random":
        adapter = RandomResidualAdapter(scale=args.random_scale, seed=args.seed)
    elif args.mode == "adapted":
        if args.adapter is None:
            raise ValueError("adapted mode requires --adapter")
        adapter = load_adapter(args.adapter, device)
    else:
        raise ValueError(f"unsupported inference mode: {args.mode}")
    if args.ditto_dir is None or args.data_root is None or args.cfg_pkl is None or args.audio_path is None or args.source_path is None or args.output_path is None:
        raise ValueError("inference modes require Ditto paths, audio, source, and output")
    if args.mode == "adapted" and args.adapter is None:
        raise ValueError("adapted mode requires --adapter")
    run_ditto_with_adapter(args, adapter, install_hook=args.mode in {"identity", "adapted", "random"})
    print(json.dumps({"mode": args.mode, "output": str(args.output_path), "hook_installed": args.mode in {"identity", "adapted", "random"}, "validation": "training-set feasibility only"}, sort_keys=True))


if __name__ == "__main__":
    main()
