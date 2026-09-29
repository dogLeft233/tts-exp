"""Frozen SSL extraction, occurrence matching, and auditable pooling views."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_phone_rules_metrics import normalize_phone, pool_phone_tokens
from scripts.experiments.lrs3_phone_rules_worker import (
    WorkerError,
    derive_frame_times,
    extract_features,
    frontend_config_from_model,
    load_ssl_bundle,
    processor_fingerprint,
    read_pcm16,
    sha256_file,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_compact_token_store(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Store vectors once; JSONL keeps only auditable row metadata/indexes."""

    dimensions = next((len(row["embedding"]) for row in rows if row.get("embedding") is not None), 0)
    vectors = np.zeros((len(rows), dimensions), dtype=np.float32)
    compact: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        value = dict(row)
        embedding = value.pop("embedding", None)
        if embedding is not None:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vector.size != dimensions or not np.all(np.isfinite(vector)):
                raise ValueError("inconsistent token embedding dimension")
            vectors[index] = vector
        value["embedding_index"] = index
        value["embedding_store"] = "embeddings.npz"
        compact.append(value)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".embeddings.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, vectors=vectors, dtype="float32", row_count=np.asarray([len(rows)], dtype=np.int64))
    temporary.replace(output_dir / "embeddings.npz")
    return compact


def cache_key(
    *,
    pcm_sha256: str,
    model_name: str,
    revision: str | None,
    layers: Sequence[int],
    frontend: Mapping[str, Any],
    textgrid_sha256: str | None,
    view: str,
    context_policy: str = "full_sentence",
) -> str:
    payload = {
        "pcm_sha256": pcm_sha256,
        "model_name": model_name,
        "revision": revision,
        "layers": [int(value) for value in layers],
        "frontend": dict(frontend),
        "textgrid_sha256": textgrid_sha256,
        "view": view,
        "context_policy": context_policy,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalise_features(features: np.ndarray, frame_times: np.ndarray, tokens: Sequence[Mapping[str, Any]], *, view: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = pool_phone_tokens(features, frame_times, tokens)
    for index, token in enumerate(base):
        raw = dict(token)
        indices = np.asarray(raw.get("frame_indices", []), dtype=np.int64)
        start = float(raw["start_s"])
        end = float(raw["end_s"])
        duration = end - start
        if view == "full":
            chosen = indices
        elif view == "core":
            chosen = indices[(frame_times[indices] >= start + 0.20 * duration) & (frame_times[indices] < end - 0.20 * duration)] if indices.size else indices
        elif view == "boundary_start":
            chosen = indices[frame_times[indices] < start + 0.20 * duration] if indices.size else indices
        elif view == "boundary_end":
            chosen = indices[frame_times[indices] >= end - 0.20 * duration] if indices.size else indices
        elif view == "matched_1frame":
            if indices.size:
                chosen = np.asarray([int(indices[np.argmin(np.abs(frame_times[indices] - (start + end) / 2.0))])], dtype=np.int64)
            else:
                chosen = indices
        else:
            raise ValueError(f"unknown pooling view: {view}")
        row = {key: value for key, value in raw.items() if key not in {"embedding", "frame_indices", "valid", "reason"}}
        row["view"] = view
        row["frame_indices"] = chosen.astype(int).tolist()
        row["embedding"] = None
        row["valid"] = False
        row["reason"] = "non_speech" if not bool(raw.get("speech", True)) else "no_frame"
        if bool(raw.get("speech", True)) and chosen.size:
            vector = np.asarray(features[chosen].mean(axis=0), dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if norm > 0.0 and np.all(np.isfinite(vector)):
                row["embedding"] = (vector / norm).astype(np.float32).tolist()
                row["valid"] = True
                row["reason"] = "ok"
            else:
                row["reason"] = "zero_or_nonfinite_embedding"
        rows.append(row)
    return rows


def pool_views(
    layer_embeddings: np.ndarray,
    frame_times: np.ndarray,
    tokens: Sequence[Mapping[str, Any]],
    *,
    views: Sequence[str] = ("full", "core", "boundary_start", "boundary_end", "matched_1frame"),
) -> dict[str, list[dict[str, Any]]]:
    """Produce fixed pooling views without inventing frames for short phones."""

    return {str(view): _normalise_features(layer_embeddings, frame_times, tokens, view=str(view)) for view in views}


def _edit_tables(source: list[str], target: list[str], source_words: list[Any] | None, target_words: list[Any] | None) -> tuple[np.ndarray, np.ndarray]:
    n, m = len(source), len(target)
    cost = np.full((n + 1, m + 1), 10**9, dtype=np.int32)
    count = np.zeros((n + 1, m + 1), dtype=np.int8)
    cost[0, 0] = 0
    count[0, 0] = 1
    for i in range(1, n + 1):
        cost[i, 0] = i
        count[i, 0] = 1
    for j in range(1, m + 1):
        cost[0, j] = j
        count[0, j] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                (int(cost[i - 1, j]) + 1, int(count[i - 1, j])),
                (int(cost[i, j - 1]) + 1, int(count[i, j - 1])),
            ]
            word_ok = source_words is None or target_words is None or source_words[i - 1] == target_words[j - 1]
            substitution = 0 if source[i - 1] == target[j - 1] and word_ok else 1
            candidates.append((int(cost[i - 1, j - 1]) + substitution, int(count[i - 1, j - 1])))
            best = min(value[0] for value in candidates)
            cost[i, j] = best
            count[i, j] = min(2, sum(value[1] for value in candidates if value[0] == best))
    return cost, count


def match_occurrences(
    source_tokens: Sequence[Mapping[str, Any]],
    target_tokens: Sequence[Mapping[str, Any]],
    *,
    speech_only: bool = True,
) -> dict[str, Any]:
    """Globally match phone occurrences and refuse non-unique alignments."""

    source_index = [index for index, token in enumerate(source_tokens) if not speech_only or bool(token.get("speech", not token.get("silence", False)))]
    target_index = [index for index, token in enumerate(target_tokens) if not speech_only or bool(token.get("speech", not token.get("silence", False)))]
    source = [normalize_phone(source_tokens[index].get("label", source_tokens[index].get("token", ""))) for index in source_index]
    target = [normalize_phone(target_tokens[index].get("label", target_tokens[index].get("token", ""))) for index in target_index]
    source_words = [source_tokens[index].get("word_index") for index in source_index] if any("word_index" in source_tokens[index] for index in source_index) else None
    target_words = [target_tokens[index].get("word_index") for index in target_index] if any("word_index" in target_tokens[index] for index in target_index) else None
    cost, count = _edit_tables(source, target, source_words, target_words)
    i, j = len(source), len(target)
    path: list[dict[str, Any]] = []
    while i or j:
        current = int(cost[i, j])
        options: list[tuple[str, int, int, int]] = []
        if i and j:
            word_ok = source_words is None or target_words is None or source_words[i - 1] == target_words[j - 1]
            step = 0 if source[i - 1] == target[j - 1] and word_ok else 1
            if int(cost[i - 1, j - 1]) + step == current:
                options.append(("match" if step == 0 else "substitute", i - 1, j - 1, step))
        if i and int(cost[i - 1, j]) + 1 == current:
            options.append(("delete", i - 1, j, 1))
        if j and int(cost[i, j - 1]) + 1 == current:
            options.append(("insert", i, j - 1, 1))
        if not options:
            raise ValueError("edit table backtrace failed")
        # deterministic tie order: identity match, substitution, delete, insert
        options.sort(key=lambda item: {"match": 0, "substitute": 1, "delete": 2, "insert": 3}[item[0]])
        operation, previous_i, previous_j, step = options[0]
        row = {"op": operation, "source_index": source_index[previous_i] if i and operation != "insert" else None, "target_index": target_index[previous_j] if j and operation != "delete" else None}
        if operation == "match":
            row["label"] = source[previous_i]
            row["unique"] = bool(count[i, j] == 1)
        path.append(row)
        i, j = (previous_i, previous_j)
    path.reverse()
    matches = [row for row in path if row["op"] == "match" and bool(row.get("unique", False))]
    return {
        "cost": int(cost[-1, -1]),
        "edit_rate": float(cost[-1, -1] / max(len(source), 1)),
        "source_count": len(source),
        "target_count": len(target),
        "optimal_path_count_capped": int(count[-1, -1]),
        "ambiguous": bool(count[-1, -1] > 1),
        "path": path,
        "matches": matches,
        "matched_count": len(matches),
    }


def _load_bundle_remote(model_cfg: Mapping[str, Any], *, device: str, proxy: str | None) -> dict[str, Any]:
    """Download a fixed revision only after the local cache attempt failed."""

    import torch
    from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

    old = {key: os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")}
    if proxy:
        for key in old:
            os.environ[key] = proxy
    try:
        kwargs: dict[str, Any] = {"local_files_only": False, "trust_remote_code": False}
        if model_cfg.get("revision"):
            kwargs["revision"] = str(model_cfg["revision"])
        model_name = str(model_cfg["model_name"])
        processor_name = str(model_cfg.get("processor_name") or model_name)
        try:
            processor = AutoProcessor.from_pretrained(processor_name, **kwargs)
        except (OSError, ValueError, RuntimeError, KeyError):
            processor = AutoFeatureExtractor.from_pretrained(processor_name, **kwargs)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True, **kwargs)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise WorkerError(f"requested device is unavailable: {device}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "model_name": model_name,
        "processor_name": processor_name,
        "revision": model_cfg.get("revision"),
        "device": device,
        "model": model,
        "processor": processor,
        "frontend": frontend_config_from_model(model),
        "processor_fingerprint": processor_fingerprint(processor),
        "num_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
    }


def load_bundle(model_cfg: Mapping[str, Any], *, device: str, allow_download: bool = False, proxy: str | None = None) -> dict[str, Any]:
    try:
        return load_ssl_bundle(
            str(model_cfg["model_name"]),
            processor_name=str(model_cfg.get("processor_name") or model_cfg["model_name"]),
            revision=str(model_cfg.get("revision")) if model_cfg.get("revision") else None,
            device=device,
        )
    except Exception as local_error:
        if not allow_download:
            raise WorkerError(f"fixed model is not available locally: {model_cfg.get('model_name')}: {local_error}") from local_error
        try:
            return _load_bundle_remote(model_cfg, device=device, proxy=proxy)
        except Exception as remote_error:
            raise WorkerError(f"fixed model load/download failed: {model_cfg.get('model_name')}: {remote_error}") from remote_error


def extract_asset_views(
    bundle: Mapping[str, Any],
    asset: Mapping[str, Any],
    *,
    layer: int,
    sample_rate: int = 16_000,
    views: Sequence[str] = ("full", "core", "boundary_start", "boundary_end", "matched_1frame"),
) -> list[dict[str, Any]]:
    path = Path(str(asset["audio_path"]))
    pcm, meta = read_pcm16(path, sample_rate=sample_rate)
    tokens = asset.get("tokens") or asset.get("parsed_tokens")
    if not isinstance(tokens, list):
        raise WorkerError(f"asset has no frozen tokens: {asset.get('asset_id')}")
    selected, frame_times, feature_meta = extract_features(
        bundle["model"], bundle["processor"], pcm.astype(np.float32) / 32768.0, sample_rate, [int(layer)], device=str(bundle["device"]), frontend=bundle["frontend"]
    )
    pooled = pool_views(selected[int(layer)], frame_times, tokens, views=views)
    feature_meta_sha256 = hashlib.sha256(json.dumps(feature_meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output: list[dict[str, Any]] = []
    for view, rows in pooled.items():
        for row in rows:
            output.append({
                **row,
                "asset_id": asset["asset_id"],
                "pair_id": asset["paired_key"],
                "sample_id": asset["sample_id"],
                "source_group": asset["source_group"],
                "condition": asset["condition"],
                "analysis_split": asset["analysis_split"],
                "input_mode": asset["input_mode"],
                "clock_owner": asset["clock_owner"],
                "pcm_sha256": meta["pcm_sha256"],
                "container_sha256": meta["container_sha256"],
                # Full feature metadata is stored once in feature_meta.json;
                # repeating a processor config in every token row can consume
                # gigabytes for large Wav2Vec2 processors.
                "feature_meta_sha256": feature_meta_sha256,
                "cache_key": cache_key(pcm_sha256=str(meta["pcm_sha256"]), model_name=str(bundle["model_name"]), revision=bundle.get("revision"), layers=[layer], frontend=bundle["frontend"], textgrid_sha256=asset.get("textgrid_sha256"), view=view),
            })
    return output


def extract_atlas(
    registry: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device: str,
    allow_download: bool = False,
    proxy: str | None = None,
    selected_pair_ids: Sequence[str] | None = None,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Extract one encoder at a time and write pooled token rows."""

    model_key = str(model_cfg["key"])
    layer = int(model_cfg["layer"])
    layers = [layer]
    if include_diagnostics:
        layers.extend(int(value) for value in model_cfg.get("diagnostic_layers", []))
    bundle = load_bundle(model_cfg, device=device, allow_download=allow_download, proxy=proxy)
    if selected_pair_ids is None:
        selected_pair_ids = [str(row["pair_id"]) for row in registry.get("pairs", [])]
    wanted = set(str(value) for value in selected_pair_ids)
    pairs = [row for row in registry.get("pairs", []) if str(row["pair_id"]) in wanted]
    rows: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    views = ("full", "core", "boundary_start", "boundary_end", "matched_1frame")
    for pair in pairs:
        for condition in ("natural", "tts"):
            asset = pair["sides"][condition]
            if asset.get("audit_status") != "OK":
                continue
            extracted = extract_asset_views(bundle, asset, layer=layer, views=views)
            # One JSONL record per token/view is intentionally verbose but
            # self-contained for the independent checker.
            rows.extend(extracted)
            index.append({"asset_id": asset["asset_id"], "pair_id": pair["pair_id"], "condition": condition, "views": list(views), "row_count": len(extracted), "pcm_sha256": asset.get("pcm_sha256")})
    out = Path(output_dir) / model_key
    compact_rows = _write_compact_token_store(out, rows)
    _write_jsonl(out / "token_records.jsonl", compact_rows)
    _write_jsonl(out / "cache_index.jsonl", index)
    meta = {
        "schema_version": 1,
        "model_key": model_key,
        "model_name": bundle["model_name"],
        "processor_name": bundle["processor_name"],
        "revision": bundle.get("revision"),
        "layer": layer,
        "layers": layers,
        "device": device,
        "frontend": bundle["frontend"],
        "processor_fingerprint": bundle["processor_fingerprint"],
        "row_count": len(rows),
        "asset_count": len(index),
        "scope": "selected_pairs",
    }
    _write_json(out / "feature_meta.json", meta)
    return {"model_key": model_key, "rows": rows, "index": index, "meta": meta}


def load_token_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"token record is not an object at {path}:{line_number}")
        rows.append(value)
    stores: dict[str, np.ndarray] = {}
    for row in rows:
        store_name = row.get("embedding_store")
        if not store_name or str(store_name) in stores:
            continue
        store_path = Path(path).parent / str(store_name)
        with np.load(store_path, allow_pickle=False) as archive:
            stores[str(store_name)] = np.asarray(archive["vectors"], dtype=np.float32)
    for row in rows:
        store_name = row.get("embedding_store")
        index = row.get("embedding_index")
        if store_name is None or index is None:
            continue
        vectors = stores[str(store_name)]
        if bool(row.get("valid", False)):
            row["embedding"] = vectors[int(index)].astype(np.float32).tolist()
        else:
            row["embedding"] = None
    return rows


__all__ = [
    "cache_key",
    "extract_asset_views",
    "extract_atlas",
    "load_bundle",
    "load_token_records",
    "match_occurrences",
    "pool_views",
]
