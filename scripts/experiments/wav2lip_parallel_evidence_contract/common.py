from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class ProtocolError(RuntimeError): pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict): raise ProtocolError(f"expected object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value); body.pop("artifact_sha256", None); body["artifact_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"); os.replace(temporary, path)
    return body


def load_self_hashed(path: Path) -> dict[str, Any]:
    value = read_json(path); actual = value.get("artifact_sha256"); body = dict(value); body.pop("artifact_sha256", None)
    if not isinstance(actual, str) or actual != canonical_hash(body): raise ProtocolError(f"self-hash mismatch: {path}")
    return value


def load_worker(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    worker = read_json(Path(str(row["score"]["worker"])))
    visual = np.asarray(np.load(Path(str(worker["visual"])), allow_pickle=False), dtype=np.float64)
    audio = np.asarray(np.load(Path(str(worker["audio_embedding"])), allow_pickle=False), dtype=np.float64)
    matrix = np.asarray(np.load(Path(str(worker["matrix"])), allow_pickle=False), dtype=np.float64)
    if visual.ndim != 2 or audio.ndim != 2 or visual.shape[1] != audio.shape[1] or matrix.shape != (88, 31): raise ProtocolError("embedding/matrix shape mismatch")
    if not np.isfinite(visual).all() or not np.isfinite(audio).all() or not np.isfinite(matrix).all(): raise ProtocolError("non-finite score cache")
    return visual, audio, matrix


def matrix_from_embeddings(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if visual.ndim != 2 or audio.ndim != 2 or visual.shape[1] != audio.shape[1] or visual.shape[0] < 88 or audio.shape[0] < 88: raise ProtocolError("embedding support is too short or has incompatible width")
    padded = np.pad(audio, ((15, 15), (0, 0)), mode="constant")
    result = np.empty((88, 31), dtype=np.float64)
    for r in range(88):
        for j in range(31):
            result[r, j] = np.sqrt(np.sum((visual[r] - padded[r + j] + 1e-6) ** 2, dtype=np.float64))
    return result


def metrics(matrix: np.ndarray) -> dict[str, Any]:
    u = np.asarray(matrix, dtype=np.float64)[list(range(30, 58))]
    curve = np.mean(u, axis=0, dtype=np.float64); index = int(np.argmin(curve)); minimum = float(curve[index])
    return {"curve": curve.tolist(), "min_index": index, "offset": 15 - index, "D": minimum, "C": float(np.median(curve) - minimum)}


def gain(candidate: dict[str, Any], natural: dict[str, Any]) -> dict[str, float]:
    k = int(natural["min_index"]); return {"C": float(candidate["C"] - natural["C"]), "D": float(natural["D"] - candidate["D"]), "A": float(natural["curve"][k] - candidate["curve"][k])}


def group_bootstrap(rows: list[dict[str, Any]], field: str, *, seed: int = 20260910, draws: int = 20_000) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows: grouped.setdefault(str(row["source_group"]), []).append(float(row[field]))
    labels = sorted(grouped); means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    if len(labels) != 8 or any(len(grouped[label]) != 2 for label in labels): raise ProtocolError("expected 8 groups x 2 records")
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, 8, size=(draws, 8))
    estimates = means[indices].mean(axis=1)
    return {"mean": float(means.mean()), "ci99": [float(np.quantile(estimates, .005, method="linear")), float(np.quantile(estimates, .995, method="linear"))], "ci95": [float(np.quantile(estimates, .025, method="linear")), float(np.quantile(estimates, .975, method="linear"))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0)), "draws": draws, "seed": seed, "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "indices": indices.tolist()}


def official_chunks() -> list[np.ndarray]:
    chunks = []; index = 0
    while True:
        start = int(index * 80.0 / 25.0)
        if start + 16 > 308:
            chunks.append(np.arange(292, 308, dtype=np.int64)); break
        chunks.append(np.arange(start, start + 16, dtype=np.int64)); index += 1
    if len(chunks) != 93 or int(chunks[-1][0]) != 292: raise ProtocolError("official chunk contract changed")
    return chunks


def supports() -> dict[int, np.ndarray]:
    chunks = official_chunks(); return {r: np.unique(np.concatenate(chunks[r:r + 5])).astype(np.int64) for r in range(30, 58)}
