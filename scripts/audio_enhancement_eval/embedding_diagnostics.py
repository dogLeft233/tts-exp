#!/usr/bin/env python3
"""Descriptive SyncNet forward-vector diagnostics, separate from primary scores.

The script consumes an optional `score.syncnet_embeddings` NPZ captured from
the official `forward_lip` and `forward_aud` outputs. It recomputes vector
norms and unit-vector cosine on the same source-clock FULL/GUARD20 common
supports. These cosine/norm summaries are diagnostics; they never replace or
reweight the official distance-matrix endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "cross-tfg-syncnet-embedding-diagnostic-v1"
SCORE_ARRAYS = (
    "query_frame_index", "query_timestamp_num", "query_timestamp_den",
    "track_frame_index", "valid_query_mask", "distance_by_query_lag", "lag_offsets_frames",
)
EMBEDDING_ARRAYS = (
    "video_embedding_forward_float32", "audio_embedding_forward_float32", "query_frame_index_int32",
)


class DiagnosticError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(value: Mapping[str, Any], field: str) -> str:
    return _sha_bytes(_canonical_json({key: item for key, item in value.items() if key != field}))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DiagnosticError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _artifact_path(root: Path, ref: Mapping[str, Any], where: str) -> Path:
    text = ref.get("path")
    if not isinstance(text, str) or not text:
        raise DiagnosticError(f"{where}.path must be nonempty")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _checked_ref(root: Path, ref: Mapping[str, Any], where: str) -> Path:
    path = _artifact_path(root, ref, where)
    expected = ref.get("file_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or _sha_file(path) != expected:
        raise DiagnosticError(f"{where} file SHA mismatch: {path}")
    return path


def _load_npz_ref(root: Path, ref: Mapping[str, Any], arrays: Sequence[str], *, where: str) -> dict[str, np.ndarray]:
    path = _checked_ref(root, ref, where)
    meta = ref.get("arrays")
    if not isinstance(meta, dict):
        raise DiagnosticError(f"{where}.arrays metadata is missing")
    try:
        with np.load(path, allow_pickle=False) as archive:
            output = {name: np.asarray(archive[name]) for name in arrays}
    except Exception as exc:
        raise DiagnosticError(f"cannot read {where}: {exc}") from exc
    for name, array in output.items():
        item = meta.get(name)
        if not isinstance(item, dict) or ("key" in item and item.get("key") != name):
            raise DiagnosticError(f"{where} has no exact array seal for {name}")
        if item.get("dtype") not in (array.dtype.str, array.dtype.name) or item.get("shape") != list(array.shape):
            raise DiagnosticError(f"{where} dtype/shape mismatch for {name}")
        digest = _sha_bytes(np.ascontiguousarray(array).tobytes(order="C"))
        if item.get("array_sha256") != digest:
            raise DiagnosticError(f"{where} array SHA mismatch for {name}")
    return output


def _query_keys(arrays: Mapping[str, np.ndarray]) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for frame, num, den, track in zip(
        arrays["query_frame_index"], arrays["query_timestamp_num"],
        arrays["query_timestamp_den"], arrays["track_frame_index"], strict=True,
    ):
        if int(den) <= 0:
            raise DiagnosticError("score timestamp denominator must be positive")
        divisor = math.gcd(int(num), int(den))
        result.append((int(frame), int(num) // divisor, int(den) // divisor, int(track)))
    if len(result) != len(set(result)):
        raise DiagnosticError("score NPZ contains duplicate source-clock query keys")
    return result


def _unit_cosine_curves(
    video: np.ndarray,
    audio: np.ndarray,
    keys: Sequence[tuple[int, int, int, int]],
    row_indices: Mapping[tuple[int, int, int, int], int],
    selected_keys: Sequence[tuple[int, int, int, int]],
    lag_offsets: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    video64 = video.astype(np.float64, copy=False)
    audio64 = audio.astype(np.float64, copy=False)
    video_norm = np.linalg.norm(video64, axis=1)
    audio_norm = np.linalg.norm(audio64, axis=1)
    index_by_frame: dict[int, int] = {}
    for index, key in enumerate(keys):
        frame = key[0]
        if frame in index_by_frame:
            raise DiagnosticError("embedding frame map is not unique")
        index_by_frame[frame] = index
    cosines = np.zeros((len(selected_keys), len(lag_offsets)), dtype=np.float64)
    zero_audio_norm_pairs = 0
    padded_audio_pairs = 0
    for qi, key in enumerate(selected_keys):
        video_index = row_indices[key]
        vnorm = float(video_norm[video_index])
        for column, lag in enumerate(lag_offsets.tolist()):
            audio_index = index_by_frame.get(key[0] + int(lag))
            if audio_index is None:
                # The official distance routine pads the audio sequence with
                # zero vectors at its edges; cosine with that padded vector is 0.
                padded_audio_pairs += 1
                continue
            anorm = float(audio_norm[audio_index])
            if vnorm <= 0.0 or anorm <= 0.0:
                zero_audio_norm_pairs += 1
                continue
            value = float(np.dot(video64[video_index] / vnorm, audio64[audio_index] / anorm))
            cosines[qi, column] = min(1.0, max(-1.0, value))
    return cosines, {
        "query_count": int(len(selected_keys)),
        "padded_audio_pair_count": padded_audio_pairs,
        "zero_norm_pair_count": zero_audio_norm_pairs,
        "video_norm_median": float(np.median(video_norm[[row_indices[key] for key in selected_keys]])) if selected_keys else None,
        "video_norm_p05": float(np.quantile(video_norm[[row_indices[key] for key in selected_keys]], 0.05)) if selected_keys else None,
        "video_norm_p95": float(np.quantile(video_norm[[row_indices[key] for key in selected_keys]], 0.95)) if selected_keys else None,
        "audio_norm_median_aligned": float(np.median(audio_norm[[row_indices[key] for key in selected_keys]])) if selected_keys else None,
        "audio_norm_p05_aligned": float(np.quantile(audio_norm[[row_indices[key] for key in selected_keys]], 0.05)) if selected_keys else None,
        "audio_norm_p95_aligned": float(np.quantile(audio_norm[[row_indices[key] for key in selected_keys]], 0.95)) if selected_keys else None,
    }


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if out.exists():
        raise DiagnosticError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    plan_path, results_path, manifest_path = (Path(args.plan).resolve(strict=True), Path(args.results).resolve(strict=True), Path(args.result_manifest).resolve(strict=True))
    root = Path(args.artifact_root).resolve(strict=True)
    plan, manifest = _load_json(plan_path), _load_json(manifest_path)
    plan_raw_sha, results_raw_sha = _sha_file(plan_path), _sha_file(results_path)
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN" or plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise DiagnosticError("frozen plan schema/fingerprint is invalid")
    if manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or manifest.get("run_id") != plan.get("run_id") or manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise DiagnosticError("result manifest schema/fingerprint is invalid")
    if manifest.get("plan_raw_sha256") != plan_raw_sha or manifest.get("plan_sha256") != plan.get("plan_sha256") or manifest.get("results_jsonl_sha256") != results_raw_sha:
        raise DiagnosticError("result manifest does not bind exact plan/results inputs")
    syncnet_protocol = plan.get("protocol", {}).get("syncnet", {})
    pinned_sources = syncnet_protocol.get("source_sha256") if isinstance(syncnet_protocol, dict) else None
    if not isinstance(pinned_sources, dict):
        raise DiagnosticError("frozen plan lacks SyncNet source hashes")
    expected_instance_sha = pinned_sources.get("SyncNetInstance.py")
    expected_model_source_sha = pinned_sources.get("SyncNetModel.py")
    for field, value in (("SyncNetInstance.py", expected_instance_sha), ("SyncNetModel.py", expected_model_source_sha)):
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise DiagnosticError(f"frozen plan lacks a valid source SHA for {field}")
    instance_source_path = Path(args.syncnet_instance_source).resolve(strict=True)
    model_source_path = Path(args.syncnet_model_source).resolve(strict=True)
    actual_instance_sha = _sha_file(instance_source_path)
    actual_model_source_sha = _sha_file(model_source_path)
    if actual_instance_sha != expected_instance_sha or actual_model_source_sha != expected_model_source_sha:
        raise DiagnosticError("actual SyncNet source files differ from the source hashes frozen in the plan")
    capture_wrapper_path = Path(args.capture_wrapper_source).resolve(strict=True)
    capture_wrapper_sha = _sha_file(capture_wrapper_path)
    expected_capture_wrapper_sha = args.expected_capture_wrapper_sha256
    if not isinstance(expected_capture_wrapper_sha, str) or len(expected_capture_wrapper_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_capture_wrapper_sha):
        raise DiagnosticError("expected capture-wrapper SHA must be a lowercase SHA-256")
    if capture_wrapper_sha != expected_capture_wrapper_sha:
        raise DiagnosticError("actual embedding capture wrapper differs from its explicit expected SHA")
    planned_rows = plan.get("rows")
    results_rows = _load_jsonl(results_path)
    if not isinstance(planned_rows, list) or len(planned_rows) != len(results_rows) or manifest.get("planned_rows") != len(planned_rows) or manifest.get("result_rows") != len(results_rows):
        raise DiagnosticError("result row counts differ from frozen plan/manifest")
    planned_ids = [row.get("row_id") for row in planned_rows]
    result_ids = [row.get("row_id") for row in results_rows]
    if planned_ids != result_ids or len(set(planned_ids)) != len(planned_ids):
        raise DiagnosticError("result rows do not preserve frozen plan order")
    models_raw = plan.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise DiagnosticError("frozen plan has no model registry")
    model_registry = {
        str(item.get("model_id")): item for item in models_raw
        if isinstance(item, dict) and isinstance(item.get("model_id"), str)
    }
    if len(model_registry) != len(models_raw):
        raise DiagnosticError("frozen model IDs must be valid and unique")

    row_state: dict[str, dict[str, Any]] = {}
    missing_sidecars: list[str] = []
    for planned, result in zip(planned_rows, results_rows, strict=True):
        row_id = str(planned["row_id"])
        for key in ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed", "method_id"):
            if result.get(key) != planned.get(key):
                raise DiagnosticError(f"result identity mismatch for {row_id}.{key}")
        if result.get("status") not in ("PASS", "FAILED", "INCOMPLETE"):
            raise DiagnosticError(f"invalid producer status for {row_id}: {result.get('status')!r}")
        if result.get("status") != "PASS":
            row_state[row_id] = {"planned": planned, "result": result, "status": "SOURCE_ROW_NOT_PASS"}
            continue
        score = result.get("score")
        if not isinstance(score, dict) or score.get("status") != "PASS":
            raise DiagnosticError(f"PASS row has no score object: {row_id}")
        if score.get("score_audio_sha256") != planned.get("natural_pcm", {}).get("pcm_sha256"):
            raise DiagnosticError(f"score audio is not the frozen natural PCM for {row_id}")
        model = model_registry.get(str(planned.get("model_id")))
        if model is None or score.get("score_model_sha256") != model.get("score_model_sha256"):
            raise DiagnosticError(f"score model SHA differs from the frozen model registry for {row_id}")
        expected_preprocessing = model.get("preprocessing_sha256")
        if score.get("preprocessing_sha256") != expected_preprocessing:
            raise DiagnosticError(f"score preprocessing SHA differs from the frozen model registry for {row_id}")
        embedding_ref = score.get("syncnet_embeddings")
        score_ref = score.get("raw_matrix_npz")
        if not isinstance(embedding_ref, dict):
            missing_sidecars.append(row_id)
            row_state[row_id] = {"planned": planned, "result": result, "status": "EMBEDDING_SIDECAR_NOT_AVAILABLE"}
            continue
        score_arrays = _load_npz_ref(root, score_ref, SCORE_ARRAYS, where=f"{row_id}.score.raw_matrix_npz")
        embedding_arrays = _load_npz_ref(root, embedding_ref, EMBEDDING_ARRAYS, where=f"{row_id}.score.syncnet_embeddings")
        n = len(score_arrays["query_frame_index"])
        video = embedding_arrays["video_embedding_forward_float32"]
        audio = embedding_arrays["audio_embedding_forward_float32"]
        qframes = embedding_arrays["query_frame_index_int32"]
        if video.dtype != np.dtype("<f4") or audio.dtype != np.dtype("<f4") or video.ndim != 2 or audio.ndim != 2 or video.shape != audio.shape or video.shape != (n, 1024):
            raise DiagnosticError(f"forward embedding arrays must share the official SyncNet float32 shape (N,1024): {row_id}")
        if qframes.dtype != np.dtype("<i4") or qframes.shape != (n,) or not np.array_equal(qframes, score_arrays["query_frame_index"]):
            raise DiagnosticError(f"embedding vectors are not aligned to score NPZ query rows: {row_id}")
        if not np.isfinite(video).all() or not np.isfinite(audio).all():
            raise DiagnosticError(f"non-finite forward embedding value: {row_id}")
        expected_score_dtypes = {
            "query_frame_index": np.dtype("<i4"),
            "query_timestamp_num": np.dtype("<i8"),
            "query_timestamp_den": np.dtype("<i8"),
            "track_frame_index": np.dtype("<i4"),
            "valid_query_mask": np.dtype("bool"),
            "distance_by_query_lag": np.dtype("<f4"),
            "lag_offsets_frames": np.dtype("<i4"),
        }
        if any(score_arrays[name].dtype != dtype for name, dtype in expected_score_dtypes.items()):
            raise DiagnosticError(f"raw score NPZ dtype differs from frozen diagnostic contract: {row_id}")
        if score_arrays["distance_by_query_lag"].shape != (n, 31) or score_arrays["valid_query_mask"].shape != (n,):
            raise DiagnosticError(f"raw score NPZ shape differs from frozen diagnostic contract: {row_id}")
        if not np.array_equal(score_arrays["lag_offsets_frames"], np.arange(-15, 16, dtype=np.int32)):
            raise DiagnosticError(f"score lag axis is not the frozen −15…+15 axis: {row_id}")
        if embedding_ref.get("forward_stage") != "official_forward_outputs_before_distance_no_extra_normalization":
            raise DiagnosticError(f"embedding capture stage is not frozen/explicit: {row_id}")
        expected_sidecar_sources = {
            "syncnet_instance_sha256": expected_instance_sha,
            "syncnet_model_source_sha256": expected_model_source_sha,
            "capture_wrapper_sha256": capture_wrapper_sha,
        }
        for source_field, expected_source_sha in expected_sidecar_sources.items():
            if embedding_ref.get(source_field) != expected_source_sha:
                raise DiagnosticError(f"embedding capture {source_field} differs from the pinned source file: {row_id}")
        if embedding_ref.get("capture_wrapper_path") != str(capture_wrapper_path):
            raise DiagnosticError(f"embedding capture wrapper path differs from the explicitly pinned file: {row_id}")
        if embedding_ref.get("model_sha256") != score.get("score_model_sha256") or embedding_ref.get("preprocessing_sha256") != score.get("preprocessing_sha256"):
            raise DiagnosticError(f"embedding capture model/preprocessing differs from official score: {row_id}")
        keys = _query_keys(score_arrays)
        valid = score_arrays["valid_query_mask"].astype(bool)
        valid_keys = [key for key, keep in zip(keys, valid.tolist(), strict=True) if keep]
        if len(valid_keys) != len(set(valid_keys)):
            raise DiagnosticError(f"duplicate valid query keys in {row_id}")
        row_state[row_id] = {
            "planned": planned, "result": result, "status": "PASS",
            "score_arrays": score_arrays, "embedding_arrays": embedding_arrays,
            "keys": keys, "valid_keys": valid_keys,
        }

    methods = plan.get("methods")
    supports = plan.get("supports")
    if not isinstance(methods, list) or not isinstance(supports, list):
        raise DiagnosticError("plan methods/supports are missing")
    method_ids = [str(item["method_id"]) for item in methods]
    if not method_ids or len(method_ids) != len(set(method_ids)):
        raise DiagnosticError("plan method IDs must be nonempty and unique")
    method_kinds = {str(item["method_id"]): item.get("kind") for item in methods}
    identity_ids = [method_id for method_id, kind in method_kinds.items() if kind == "identity"]
    if len(identity_ids) != 1:
        raise DiagnosticError("plan must declare exactly one identity method for paired diagnostics")
    support_ids = [str(item.get("support_id")) for item in supports if isinstance(item, dict)]
    if len(support_ids) != len(supports) or not support_ids or len(support_ids) != len(set(support_ids)):
        raise DiagnosticError("plan support IDs must be nonempty, unique objects")
    if not any(item.get("support_id") == "full" and item.get("trim_each_end") == 0 for item in supports):
        raise DiagnosticError("plan must include the frozen FULL support")
    if not any(item.get("support_id") == "guard20" and item.get("trim_each_end") == 20 for item in supports):
        raise DiagnosticError("plan must include the frozen GUARD20 support")
    lag_offsets = np.asarray(plan["protocol"]["syncnet"]["lag_offsets_frames"], dtype=np.int32)
    zero_column = int(plan["protocol"]["syncnet"]["zero_lag_column"])
    if lag_offsets.shape != (31,) or not np.array_equal(lag_offsets, np.arange(-15, 16, dtype=np.int32)) or zero_column != 15:
        raise DiagnosticError("plan lag axis/zero-lag column differs from the frozen 31-column protocol")
    grouped: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for row_id, state in row_state.items():
        row = state["planned"]
        group_key = tuple(row.get(field) for field in ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed"))
        method_id = str(row["method_id"])
        if method_id in grouped[group_key]:
            raise DiagnosticError(f"duplicate method row in planned context {group_key}: {method_id}")
        grouped[group_key][method_id] = row_id

    diagnostics: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    incomplete_groups = 0
    for group_key, by_method in grouped.items():
        if set(by_method) != set(method_ids):
            incomplete_groups += 1
            diagnostics.append({"context": list(group_key), "status": "INCOMPLETE_PLANNED_METHOD_GROUP", "planned_method_ids": method_ids, "present_method_ids": sorted(by_method)})
            continue
        if any(row_state[row_id]["status"] != "PASS" for row_id in by_method.values()):
            incomplete_groups += 1
            diagnostics.append({"context": list(group_key), "status": "INCOMPLETE_EMBEDDING_OR_SCORE_ROW", "planned_method_ids": method_ids})
            continue
        first = row_state[by_method[method_ids[0]]]
        common_set = set(first["valid_keys"])
        for method_id in method_ids[1:]:
            common_set.intersection_update(row_state[by_method[method_id]]["valid_keys"])
        common_keys = sorted(common_set, key=lambda key: (key[0], key[1] / key[2], key[3]))
        row_metrics_by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for support in supports:
            support_id = str(support["support_id"])
            trim = int(support["trim_each_end"])
            start = min(trim, len(common_keys))
            stop = max(start, len(common_keys) - trim)
            selected_keys = common_keys[start:stop]
            for method_id, row_id in by_method.items():
                state = row_state[row_id]
                score_arrays = state["score_arrays"]
                embedding_arrays = state["embedding_arrays"]
                keys = state["keys"]
                row_indices = {key: index for index, key in enumerate(keys)}
                video = embedding_arrays["video_embedding_forward_float32"]
                audio = embedding_arrays["audio_embedding_forward_float32"]
                if not selected_keys:
                    diagnostic = {"context": list(group_key), "row_id": row_id, "method_id": method_id, "support_id": support_id, "status": "EMPTY_COMMON_SUPPORT", "query_count": 0}
                    diagnostics.append(diagnostic)
                    row_metrics_by_method[method_id][support_id] = diagnostic
                    continue
                cosine_rows, norm_stats = _unit_cosine_curves(video, audio, keys, row_indices, selected_keys, lag_offsets)
                cosine_curve = np.mean(cosine_rows, axis=0, dtype=np.float64)
                median_cos = float(np.median(cosine_curve))
                best_column = int(np.argmax(cosine_curve))
                result_metrics = state["result"]["score"].get("metrics_by_support", {}).get(support_id, {})
                diagnostic = {
                    "context": list(group_key), "row_id": row_id, "method_id": method_id, "support_id": support_id,
                    "status": "PASS" if len(selected_keys) >= int(support["minimum_queries"]) else "INELIGIBLE_SHORT_SUPPORT",
                    "query_count": int(len(selected_keys)), "support_mask_definition": "all-planned-method valid source-clock key intersection before endpoint reduction",
                    "primary_syncnet": {key: result_metrics.get(key) for key in ("B", "Dmin", "D0", "Cbest", "Canchor", "best_lag_frames", "official_av_offset_frames")},
                    "embedding": {
                        **norm_stats,
                        "unit_cosine_curve": cosine_curve.tolist(),
                        "unit_cosine_zero_lag": float(cosine_curve[zero_column]),
                        "unit_cosine_lag_median": median_cos,
                        "unitnorm_canchor": float(cosine_curve[zero_column] - median_cos),
                        "unitnorm_cbest": float(cosine_curve[best_column] - median_cos),
                        "unitnorm_best_lag_frames": int(lag_offsets[best_column]),
                        "unitnorm_official_av_offset_frames": int(-lag_offsets[best_column]),
                    },
                }
                diagnostics.append(diagnostic)
                row_metrics_by_method[method_id][support_id] = diagnostic
        if len(identity_ids) == 1:
            identity_id = identity_ids[0]
            for method_id in method_ids:
                if method_id == identity_id:
                    continue
                for support in supports:
                    support_id = str(support["support_id"])
                    left = row_metrics_by_method.get(identity_id, {}).get(support_id)
                    right = row_metrics_by_method.get(method_id, {}).get(support_id)
                    if not left or not right or left.get("query_count") != right.get("query_count"):
                        paired.append({"context": list(group_key), "method_id": method_id, "support_id": support_id, "status": "UNPAIRED"})
                        continue
                    left_emb, right_emb = left.get("embedding", {}), right.get("embedding", {})
                    left_primary, right_primary = left.get("primary_syncnet", {}), right.get("primary_syncnet", {})
                    delta_keys = ("unitnorm_cbest", "unitnorm_canchor")
                    primary_keys = ("B", "Dmin", "D0", "Cbest", "Canchor")
                    paired.append({
                        "context": list(group_key), "method_id": method_id, "identity_method_id": identity_id,
                        "support_id": support_id, "status": "PASS",
                        "primary_syncnet_deltas_vs_identity": {key: (None if left_primary.get(key) is None or right_primary.get(key) is None else float(right_primary[key]) - float(left_primary[key])) for key in primary_keys},
                        "unit_embedding_deltas_vs_identity": {key: (None if left_emb.get(key) is None or right_emb.get(key) is None else float(right_emb[key]) - float(left_emb[key])) for key in delta_keys},
                        "embedding_norm_deltas_vs_identity": {
                            key: (None if left_emb.get(key) is None or right_emb.get(key) is None else float(right_emb[key]) - float(left_emb[key]))
                            for key in ("video_norm_median", "audio_norm_median_aligned")
                        },
                    })

    def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")

    write_jsonl(out / "embedding_diagnostics.jsonl", diagnostics)
    write_jsonl(out / "paired_embedding_deltas.jsonl", paired)
    accounted_rows = sum(row_state[row_id]["status"] in ("PASS", "SOURCE_ROW_NOT_PASS", "EMBEDDING_SIDECAR_NOT_AVAILABLE") for row_id in row_state)
    sidecar_count = sum(row_state[row_id]["status"] == "PASS" for row_id in row_state)
    output_status = "PASS" if sidecar_count == len([row for row in results_rows if row.get("status") == "PASS"]) and incomplete_groups == 0 else "INCOMPLETE"
    audit = {
        "schema_version": SCHEMA_VERSION, "status": output_status, "run_id": plan.get("run_id"),
        "plan_path": str(plan_path), "plan_raw_sha256": plan_raw_sha, "plan_payload_sha256": plan["plan_sha256"],
        "results_path": str(results_path), "results_raw_sha256": results_raw_sha,
        "result_manifest_path": str(manifest_path), "result_manifest_raw_sha256": _sha_file(manifest_path),
        "result_manifest_payload_sha256": manifest["manifest_sha256"], "planned_rows": len(planned_rows),
        "syncnet_source_files": {
            "SyncNetInstance.py": {"path": str(instance_source_path), "sha256": actual_instance_sha},
            "SyncNetModel.py": {"path": str(model_source_path), "sha256": actual_model_source_sha},
            "capture_wrapper": {"path": str(capture_wrapper_path), "sha256": capture_wrapper_sha},
        },
        "accounted_rows": accounted_rows, "embedding_sidecar_rows": sidecar_count,
        "missing_embedding_sidecar_row_ids": missing_sidecars, "incomplete_context_groups": incomplete_groups,
        "diagnostic_rows": len(diagnostics), "paired_delta_rows": len(paired),
        "primary_score_changed": False, "confirmatory_family_changed": False,
        "interpretation": "cosine and vector norms are descriptive mechanism diagnostics; primary SyncNet distance-matrix endpoints remain authoritative",
    }
    (out / "audit.json").write_bytes(_canonical_json(audit) + b"\n")
    output_files = []
    for path in sorted(item for item in out.iterdir() if item.is_file()):
        output_files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    output_manifest = {"schema_version": SCHEMA_VERSION, "status": output_status, "run_id": plan.get("run_id"), "files": output_files}
    output_manifest["manifest_sha256"] = _payload_sha(output_manifest, "manifest_sha256")
    (out / "manifest.json").write_bytes(_canonical_json(output_manifest) + b"\n")
    return 0 if output_status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--syncnet-instance-source", required=True)
    parser.add_argument("--syncnet-model-source", required=True)
    parser.add_argument("--capture-wrapper-source", required=True)
    parser.add_argument("--expected-capture-wrapper-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        out = Path(args.out).resolve()
        if out.exists():
            error = {"schema_version": SCHEMA_VERSION, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "argv": sys.argv}
            (out / "audit.json").write_bytes(_canonical_json(error) + b"\n")
        print(f"embedding_diagnostic_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
