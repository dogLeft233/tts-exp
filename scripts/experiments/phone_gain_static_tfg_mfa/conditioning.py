"""MFA-derived phone/time conditioning with explicit arm permissions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.lrs3_phone_rules_metrics import normalize_phone


MODES = frozenset({"AUDIO_FEATURES", "BOUNDARY_TIME", "MFA_PHONE_TIME"})


@dataclass(frozen=True)
class PhoneVocabulary:
    labels: tuple[str, ...]
    ids: Mapping[str, int]
    sil_id: int
    unk_id: int
    hash: str

    @classmethod
    def from_labels(cls, labels: Sequence[str]) -> "PhoneVocabulary":
        import hashlib
        normalized = sorted({normalize_phone(label) for label in labels if normalize_phone(label) and normalize_phone(label) not in {"SIL", "UNK"}})
        # Keep the two non-speech/unknown rows at stable tail positions.  The
        # model uses these positions to hard-mask their embedding output.
        ordered = [*normalized, "SIL", "UNK"]
        ids = {label: index for index, label in enumerate(ordered)}
        digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
        return cls(tuple(ordered), ids, ids["SIL"], ids["UNK"], digest)

    def encode(self, label: str, *, speech: bool) -> int:
        if not speech:
            return self.sil_id
        return int(self.ids.get(normalize_phone(label), self.unk_id))


@dataclass(frozen=True)
class Conditioning:
    mode: str
    phone_ids: np.ndarray
    timing: np.ndarray
    protected_mask: np.ndarray
    vocabulary_hash: str
    edit_mask: np.ndarray | None = None
    schema_version: int = 1

    def validate(self, *, frames: int | None = None, samples: int | None = None) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown conditioning mode: {self.mode}")
        if self.phone_ids.ndim != 1 or self.timing.shape != (3, self.phone_ids.size):
            raise ValueError("conditioning has invalid frame shapes")
        if frames is not None and self.phone_ids.size != int(frames):
            raise ValueError("conditioning frame count differs from audio features")
        if self.protected_mask.ndim != 1 or (samples is not None and self.protected_mask.size != int(samples)):
            raise ValueError("conditioning protected mask differs from PCM")
        if self.edit_mask is not None and (self.edit_mask.ndim != 1 or (samples is not None and self.edit_mask.size != int(samples))):
            raise ValueError("conditioning edit mask differs from PCM")
        if self.edit_mask is not None and (not np.isfinite(self.edit_mask).all() or np.any(self.edit_mask < 0.0) or np.any(self.edit_mask > 1.0)):
            raise ValueError("conditioning edit mask must be finite and in [0, 1]")
        if not np.isfinite(self.timing).all() or not np.isfinite(self.protected_mask.astype(np.float32)).all():
            raise ValueError("conditioning contains non-finite values")


def fit_duration_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: list[float] = []
    for row in rows:
        for token in row.get("tokens", []):
            if bool(token.get("speech", not token.get("silence", False))):
                duration = float(token["end_s"]) - float(token["start_s"])
                if duration > 0 and np.isfinite(duration):
                    values.append(float(np.clip(np.log(duration), np.log(0.01), np.log(2.0))))
    if not values:
        return {"mean": 0.0, "std": 1.0}
    return {"mean": float(np.mean(values)), "std": max(float(np.std(values)), 1e-6)}


def _token_at(tokens: Sequence[Mapping[str, Any]], time_s: float) -> Mapping[str, Any] | None:
    for token in tokens:
        start = float(token["start_s"])
        end = float(token["end_s"])
        if start <= time_s < end:
            return token
    return None


def build_conditioning(
    tokens: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    frame_count: int,
    vocabulary: PhoneVocabulary,
    duration_stats: Mapping[str, float],
    protected_mask: np.ndarray,
    edit_mask: np.ndarray | None = None,
    mode: str,
    sample_rate: int = 16_000,
    hop_length: int = 128,
) -> Conditioning:
    if mode not in MODES:
        raise ValueError(f"unknown conditioning mode: {mode}")
    mask = np.asarray(protected_mask, dtype=bool).reshape(-1)
    if mask.size != int(sample_count):
        raise ValueError("protected mask length mismatch")
    ids = np.full(int(frame_count), vocabulary.sil_id, dtype=np.int64)
    timing = np.zeros((3, int(frame_count)), dtype=np.float32)
    mean = float(duration_stats.get("mean", 0.0))
    std = max(float(duration_stats.get("std", 1.0)), 1e-6)
    for index in range(int(frame_count)):
        time_s = index * int(hop_length) / float(sample_rate)
        token = _token_at(tokens, time_s)
        if token is None or not bool(token.get("speech", not token.get("silence", False))):
            continue
        start = float(token["start_s"])
        end = float(token["end_s"])
        duration = max(end - start, 1e-6)
        ids[index] = vocabulary.encode(str(token.get("label", "")), speech=True)
        timing[0, index] = float(np.clip((time_s - start) / duration, 0.0, 1.0))
        timing[1, index] = (float(np.clip(np.log(duration), np.log(0.01), np.log(2.0))) - mean) / std
        timing[2, index] = 1.0
    if mode == "AUDIO_FEATURES":
        ids.fill(vocabulary.sil_id)
        timing.fill(0.0)
    elif mode == "BOUNDARY_TIME":
        ids.fill(vocabulary.sil_id)
    if edit_mask is None:
        editable = (~mask).astype(np.float32)
    else:
        editable = np.asarray(edit_mask, dtype=np.float32).reshape(-1)
        if editable.size != int(sample_count):
            raise ValueError("edit mask length mismatch")
        editable[mask] = 0.0
    condition = Conditioning(mode, ids, timing, mask, vocabulary.hash, editable)
    condition.validate(frames=frame_count, samples=sample_count)
    return condition


def permute_ids(condition: Conditioning, vocabulary: PhoneVocabulary) -> Conditioning:
    if condition.mode != "MFA_PHONE_TIME":
        raise ValueError("ID permutation is only defined for MFA_PHONE_TIME")
    ids = condition.phone_ids.copy()
    speech_ids = [index for label, index in vocabulary.ids.items() if label not in {"SIL", "UNK"}]
    if speech_ids:
        mapping = {old: speech_ids[(pos + 1) % len(speech_ids)] for pos, old in enumerate(speech_ids)}
        for old, new in mapping.items():
            ids[condition.phone_ids == old] = new
    return Conditioning(condition.mode, ids, condition.timing.copy(), condition.protected_mask.copy(), condition.vocabulary_hash, None if condition.edit_mask is None else condition.edit_mask.copy())


def jitter_boundaries(tokens: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Return a fixed occurrence-hash-like ±20 ms timing perturbation."""
    import hashlib
    result = [dict(token) for token in tokens]
    for index in range(len(result) - 1):
        left, right = result[index], result[index + 1]
        if not bool(left.get("speech", not left.get("silence", False))) or not bool(right.get("speech", not right.get("silence", False))):
            continue
        boundary = float(left["end_s"])
        if boundary <= float(left["start_s"]) + 0.01 or float(right["end_s"]) <= boundary + 0.01:
            continue
        digest = hashlib.sha256(f"{seed}|{index}|{left.get('label')}|{right.get('label')}".encode()).digest()
        delta = 0.02 if digest[0] & 1 else -0.02
        new_boundary = float(np.clip(boundary + delta, float(left["start_s"]) + 0.01, float(right["end_s"]) - 0.01))
        left["end_s"] = new_boundary
        right["start_s"] = new_boundary
    return result


__all__ = ["Conditioning", "MODES", "PhoneVocabulary", "build_conditioning", "fit_duration_stats", "jitter_boundaries", "permute_ids"]
