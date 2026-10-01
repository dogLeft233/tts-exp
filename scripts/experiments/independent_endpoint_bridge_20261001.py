"""Read-only artifact bridge from the sealed retry3 run to MEL and norm endpoints.

This compares already-rendered endpoint artifacts per sample. It computes no
cohort contrast, bootstrap, or model forward and imports no experiment producer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MAIN_PROTOCOL_SHA = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
MAIN_ROWS_SHA = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
MAIN_RESULTS_SHA = "fd1f9ce1b23861de5eea7d87470f74e52451f1cc8bb050cbe85141a62975d3fa"
MAIN_LINEAGE_SHA = "0aa21a38aa7b645fb556a589ff132ad71b2083d443df73ecf795f073a522c39c"

MEL_PLAN_SHA = "86f0656e31d4a3e87af28b40d059dc7f52945f6b1650cef166a67b235a8e7290"
MEL_MANIFEST_SHA = "cd5c1d4cc21dd32eccdd7111db18b88de57d0aec3b6079214602a7279f3996b6"
MEL_RESULTS_SHA = "d6baf57517e39316d3baf317ac6c9a06872d409987bdb25824bda8cf6d7b56b0"
MEL_RUNTIME_SHA = "39dea35a068c7cdedbdd93571ef04353d607fd20432af5f8053e536ae8186480"
MEL_INPUT_IDENTITY_SHA = "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27"

NORM_PROTOCOL_SHA = "f1c082f7c255345ce027f231275dcf2917b4c390895517a8ed8aeca8d65c6562"
NORM_ROWS_SHA = "36727fde4ffb652ac592dd37a11a8a112cfb430c37cb7b4c96b3c28fd57571a3"
NORM_RESULTS_SHA = "5c389dfc421080b00e478b455bb5ec919b9ae7b5e22748835a3df84945d7a024"
NORM_STATUS_SHA = "28543b6a0cefd5a065cf05d368c912d9e61f2be43ab15bd0d444f92a7b2bec0f"
NORM_CODE_SEAL_SHA = "a0c066f43e0f4e0223698d4d561e7b35a7c0efae75f16c5f92126c794b86458a"
NORM_ALL_FILES_SHA = "ba11e903d18868781f562554b867d5fe062f58f3b65b1b9eab414518c2bcb50a"

GEOMETRIES = ("raw", "unit")
SUPPORT_KEYS = ("full", "valid", "guard15", "guard20")
VSHIFT = 15


class BridgeError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BridgeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise BridgeError(f"non-object JSONL row: {path}")
    return rows


def _check_sha(path: Path, expected: str, label: str) -> Path:
    path = path.resolve(strict=True)
    actual = _sha(path)
    if actual != expected:
        raise BridgeError(f"{label} SHA mismatch: {actual} != {expected}")
    return path


def _resolve(root: Path, ref: str | Path, label: str) -> Path:
    candidate = Path(ref)
    if not candidate.is_absolute():
        candidate = root / candidate
    path = candidate.resolve(strict=True)
    root_real = root.resolve(strict=True)
    if path != root_real and root_real not in path.parents:
        raise BridgeError(f"{label} escapes sealed root: {path}")
    return path


def _verified_ref(root: Path, ref: str | Path, expected: str, label: str) -> Path:
    return _check_sha(_resolve(root, ref, label), expected, label)


def _read_npz(path: Path, expected_sha: str, label: str) -> dict[str, np.ndarray]:
    _check_sha(path, expected_sha, label)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _read_npy(path: Path, expected_sha: str | None, label: str) -> np.ndarray:
    if expected_sha:
        _check_sha(path, expected_sha, label)
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise BridgeError(f"expected ndarray: {label}")
    return value


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(array.dtype.str.encode("ascii"))
    h.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def _array_data_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _array_delta(left: np.ndarray, right: np.ndarray, *, overlap: bool = False) -> dict[str, Any]:
    left = np.asarray(left)
    right = np.asarray(right)
    result: dict[str, Any] = {
        "left_shape": list(left.shape), "right_shape": list(right.shape),
        "left_dtype": str(left.dtype), "right_dtype": str(right.dtype),
        "shape_match": left.shape == right.shape, "dtype_match": left.dtype == right.dtype,
        "left_array_sha256": _array_hash(left), "right_array_sha256": _array_hash(right),
    }
    if left.shape == right.shape:
        a, b = left, right
    elif overlap and left.ndim == right.ndim and left.ndim > 0:
        common = tuple(min(x, y) for x, y in zip(left.shape, right.shape))
        if len(common) != left.ndim:
            result["overlap_status"] = "unsupported_rank"
            return result
        slices = tuple(slice(0, n) for n in common)
        a, b = left[slices], right[slices]
        result["overlap_shape"] = list(common)
        result["overlap_status"] = "compared_prefix"
    else:
        result["comparison_status"] = "shape_mismatch"
        return result
    result["compared_elements"] = int(a.size)
    result["bit_exact"] = bool(left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right))
    if a.size:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result["overlap_bit_exact"] = bool(a.dtype == b.dtype and np.array_equal(a, b))
        result["max_abs_difference"] = float(np.max(diff, initial=0.0))
        result["mean_abs_difference"] = float(np.mean(diff, dtype=np.float64))
        result["exact_element_fraction"] = float(np.count_nonzero(a == b) / a.size)
    else:
        result["max_abs_difference"] = None
        result["mean_abs_difference"] = None
        result["exact_element_fraction"] = None
    return result


def _lag_from_cross(cross: np.ndarray, zero: np.ndarray, query_n: int, audio_n: int) -> np.ndarray:
    cross = np.asarray(cross, dtype=np.float32)
    zero = np.asarray(zero, dtype=np.float32)
    if cross.ndim != 2 or zero.shape != (cross.shape[0],):
        raise BridgeError(f"bad cross/zero shapes: {cross.shape}/{zero.shape}")
    if query_n < 1 or query_n > cross.shape[0] or audio_n < 1 or audio_n > cross.shape[1]:
        raise BridgeError(f"bad lag domains q={query_n}, a={audio_n}, cross={cross.shape}")
    result = np.empty((query_n, 2 * VSHIFT + 1), dtype=np.float32)
    for i in range(query_n):
        for column in range(2 * VSHIFT + 1):
            j = i + column - VSHIFT
            result[i, column] = cross[i, j] if 0 <= j < audio_n else zero[i]
    return result


def _video_pixels(left_path: Path, right_path: Path) -> dict[str, Any]:
    import cv2

    left = cv2.VideoCapture(str(left_path))
    right = cv2.VideoCapture(str(right_path))
    if not left.isOpened() or not right.isOpened():
        left.release()
        right.release()
        raise BridgeError(f"cannot open endpoint videos: {left_path} / {right_path}")
    left_reported = int(left.get(cv2.CAP_PROP_FRAME_COUNT))
    right_reported = int(right.get(cv2.CAP_PROP_FRAME_COUNT))
    common_frames = 0
    equal_pixels = total_pixels = abs_sum = 0
    max_abs = 0
    shape_match = True
    left_decoded = right_decoded = 0
    try:
        while True:
            ok_l, frame_l = left.read()
            ok_r, frame_r = right.read()
            if ok_l:
                left_decoded += 1
            if ok_r:
                right_decoded += 1
            if not ok_l or not ok_r:
                if ok_l:
                    while True:
                        ok, _ = left.read()
                        if not ok:
                            break
                        left_decoded += 1
                if ok_r:
                    while True:
                        ok, _ = right.read()
                        if not ok:
                            break
                        right_decoded += 1
                break
            common_frames += 1
            if frame_l.shape != frame_r.shape:
                shape_match = False
                continue
            diff = np.abs(frame_l.astype(np.int16) - frame_r.astype(np.int16))
            equal_pixels += int(np.count_nonzero(diff == 0))
            total_pixels += int(diff.size)
            abs_sum += int(diff.sum(dtype=np.int64))
            max_abs = max(max_abs, int(diff.max(initial=0)))
    finally:
        left.release()
        right.release()
    return {
        "left_path": str(left_path), "left_file_sha256": _sha(left_path),
        "right_path": str(right_path), "right_file_sha256": _sha(right_path),
        "left_reported_frames": left_reported, "right_reported_frames": right_reported,
        "left_decoded_frames": left_decoded, "right_decoded_frames": right_decoded,
        "common_frames_compared": common_frames, "frame_count_match": left_decoded == right_decoded == common_frames,
        "frame_shape_match": shape_match, "pixels_compared": total_pixels,
        "pixel_exact_fraction": equal_pixels / total_pixels if total_pixels else None,
        "pixel_mean_abs_difference": abs_sum / total_pixels if total_pixels else None,
        "pixel_max_abs_difference": max_abs if total_pixels else None,
    }


def _support_arrays(arrays: Mapping[str, np.ndarray], family: str) -> dict[str, np.ndarray]:
    if family == "main":
        names = {"full": "full_support_indices_int32", "valid": "valid_support_indices_int32",
                 "guard15": "guard15_indices_int32", "guard20": "guard20_indices_int32"}
    elif family == "mel":
        names = {name: f"support_{name}_int32" for name in SUPPORT_KEYS}
    else:
        names = {"full": "common_support_full_int32", "valid": "common_support_valid_int32",
                 "guard15": "common_support_guard15_int32", "guard20": "common_support_guard20_int32"}
    output: dict[str, np.ndarray] = {}
    for key, actual in names.items():
        if actual not in arrays:
            raise BridgeError(f"missing support array {actual} for {family}")
        output[key] = np.asarray(arrays[actual])
    return output


def _load_main(root: Path, plan: Mapping[str, Any], result: Mapping[str, Any], z_key: str) -> dict[str, Any]:
    context_path = _verified_ref(root, str(plan["context_npz"]), str(result["context_npz_sha256"]), "main context NPZ")
    visual_path = _verified_ref(root, str(plan["visual_npz"]), str(result["visual_npz_sha256"]), "main visual NPZ")
    score_path = _verified_ref(root, str(plan["score_npz"]), str(result["score_npz_sha256"]), "main score NPZ")
    context = _read_npz(context_path, str(result["context_npz_sha256"]), "main context NPZ")
    visual = _read_npz(visual_path, str(result["visual_npz_sha256"]), "main visual NPZ")
    score = _read_npz(score_path, str(result["score_npz_sha256"]), "main score NPZ")
    if z_key not in context or "natural_audio_embeddings_raw_float32" not in context or "lip_embeddings_float32" not in visual:
        raise BridgeError(f"main endpoint NPZ lacks expected arrays: {plan['row_id']}")
    z_value = context[z_key]
    a_value = context["natural_audio_embeddings_raw_float32"]
    v_value = visual["lip_embeddings_float32"]
    video_meta = result.get("visual_metadata", {}).get("video")
    if not isinstance(video_meta, Mapping) or not video_meta.get("sha256"):
        raise BridgeError(f"main endpoint row has no sealed video metadata: {plan['row_id']}")
    visual_meta = result.get("visual_metadata")
    if _array_data_sha(a_value) != result.get("audio_embedding_sha256"):
        raise BridgeError(f"main RAW-N A array hash differs from result row: {plan['row_id']}")
    if _array_data_sha(v_value) != result.get("lip_embedding_sha256"):
        raise BridgeError(f"main V array hash differs from result row: {plan['row_id']}")
    if not isinstance(visual_meta, Mapping) or _array_data_sha(z_value) != visual_meta.get("latent_sha256"):
        raise BridgeError(f"main latent array hash differs from visual metadata: {plan['row_id']}")
    if visual_meta.get("lip_embedding_sha256") != _array_data_sha(v_value):
        raise BridgeError(f"main V array hash differs from visual metadata: {plan['row_id']}")
    video_path = _verified_ref(root, str(plan["video_path"]), str(video_meta["sha256"]), "main rendered video")
    score_arrays = {
        "raw_cross": score["cross_distance_raw_float32"],
        "unit_cross": score["cross_distance_unit_float32"],
        "raw_zero": score["cross_distance_to_zero_raw_float32"],
        "unit_zero": score["cross_distance_to_zero_unit_float32"],
        "raw_lag": score["distance_matrix_raw_float32"],
        "unit_lag": score["distance_matrix_unit_float32"],
    }
    supports = _support_arrays(score, "main")
    for value in score_arrays.values():
        if value.dtype != np.float32:
            raise BridgeError(f"main score matrix is not float32: {plan['row_id']}")
    return {
        "sample_id": str(plan["sample_id"]), "speaker_id": str(plan["speaker_id"]),
        "row_id": str(plan["row_id"]), "condition": str(plan["condition"]),
        "context_path": str(context_path), "visual_path": str(visual_path), "score_path": str(score_path),
        "video_path": str(video_path), "video_sha256": str(video_meta["sha256"]),
        "z": z_value, "A": a_value,
        "pcm16": context.get("natural_pcm16"), "V": v_value,
        "support": supports, "score": score_arrays,
        "common_support_n": int(result["common_support_n"]),
        "audio_count": int(result["audio_row_domain_n"]),
        "visual_count": int(result["visual_query_domain_n"]),
        "artifact_sha256": {"context": str(result["context_npz_sha256"]), "visual": str(result["visual_npz_sha256"]), "score": str(result["score_npz_sha256"])},
    }


def _load_mel(root: Path, artifact_root: Path, result: Mapping[str, Any], condition: str, manifest_ledger: Mapping[str, str]) -> dict[str, Any]:
    sample_id = str(result["id"])
    cell = result.get("cells", {}).get(condition)
    shared_a = result.get("shared_A")
    if not isinstance(cell, Mapping) or not isinstance(shared_a, Mapping):
        raise BridgeError(f"missing MEL endpoint cell/shared A: {sample_id}/{condition}")
    latent_meta, video_meta, v_meta = cell.get("latents"), cell.get("video"), cell.get("V")
    if not all(isinstance(item, Mapping) for item in (latent_meta, video_meta, v_meta)):
        raise BridgeError(f"MEL endpoint cell lacks latent/video/V records: {sample_id}/{condition}")
    z_path = _verified_ref(artifact_root, str(latent_meta["path"]), str(latent_meta["file_sha256"]), "MEL latent")
    video_path = _verified_ref(artifact_root, str(video_meta["path"]), str(video_meta.get("sha256", video_meta.get("file_sha256"))), "MEL video")
    v_path = _verified_ref(artifact_root, str(v_meta["path"]), str(v_meta["file_sha256"]), "MEL V array")
    a_path = _verified_ref(artifact_root, str(shared_a["path"]), str(shared_a["file_sha256"]), "MEL shared A array")
    for path in (z_path, video_path, v_path, a_path):
        rel = path.relative_to(artifact_root.resolve(strict=True)).as_posix()
        if rel not in manifest_ledger or manifest_ledger[rel] != _sha(path):
            raise BridgeError(f"MEL artifact is not matched by run-manifest ledger: {rel}")
    z = _read_npy(z_path, str(latent_meta["file_sha256"]), "MEL latent")
    v = _read_npy(v_path, str(v_meta["file_sha256"]), "MEL V array")
    a = _read_npy(a_path, str(shared_a["file_sha256"]), "MEL shared A array")
    for value, record, label in ((z, latent_meta, "MEL latent"), (v, v_meta, "MEL V"), (a, shared_a, "MEL A")):
        if _array_data_sha(value) != record.get("array_sha256"):
            raise BridgeError(f"{label} ndarray hash differs from sealed record: {sample_id}/{condition}")
    score_map: dict[str, np.ndarray] = {}
    support_arrays: dict[str, np.ndarray] | None = None
    score_paths: dict[str, str] = {}
    for geom in GEOMETRIES:
        record = result["scores"]["geometries"][geom][condition]["distance_artifact"]
        score_path = _verified_ref(artifact_root, str(record["path"]), str(record["file_sha256"]), f"MEL {geom} distance NPZ")
        rel = score_path.relative_to(artifact_root.resolve(strict=True)).as_posix()
        if rel not in manifest_ledger or manifest_ledger[rel] != _sha(score_path):
            raise BridgeError(f"MEL score artifact is not matched by run-manifest ledger: {rel}")
        score = _read_npz(score_path, str(record["file_sha256"]), f"MEL {geom} distance NPZ")
        prefix = "raw" if geom == "raw" else "unit"
        score_map[f"{prefix}_cross"] = score["cross_v_to_a_float32"]
        score_map[f"{prefix}_zero"] = score["zero_boundary_float32"]
        score_map[f"{prefix}_lag"] = score["lag_distance_float32"]
        if support_arrays is None:
            support_arrays = _support_arrays(score, "mel")
        else:
            for name in SUPPORT_KEYS:
                if not np.array_equal(support_arrays[name], _support_arrays(score, "mel")[name]):
                    raise BridgeError(f"MEL raw/unit support arrays differ: {sample_id}/{condition}/{name}")
        score_paths[geom] = str(score_path)
    if z.dtype != np.float32 or v.dtype != np.float32 or a.dtype != np.float32:
        raise BridgeError(f"MEL endpoint arrays are not float32: {sample_id}/{condition}")
    return {
        "sample_id": sample_id, "speaker_id": str(result["speaker"]), "row_id": sample_id + "__" + condition,
        "condition": condition, "context_path": "", "visual_path": str(v_path),
        "score_path": score_paths, "video_path": str(video_path), "video_sha256": str(video_meta.get("sha256", video_meta.get("file_sha256"))),
        "z": z, "A": a, "pcm16": None, "V": v, "support": support_arrays,
        "score": score_map,
        "common_support_n": int(result["scores"]["common_support_n"]),
        "audio_count": int(result["scores"]["A_count"]),
        "visual_count": int(result["scores"]["V_counts"][condition]),
        "artifact_sha256": {"latent": str(latent_meta["file_sha256"]), "V": str(v_meta["file_sha256"]), "A": str(shared_a["file_sha256"]), "video": str(video_meta.get("sha256", video_meta.get("file_sha256")))},
    }


def _load_norm(root: Path, row: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    condition = str(row["condition"])
    sample_id = str(row["sample_id"])
    context_path = _verified_ref(root, str(result["context_npz"]), str(result["context_npz_sha256"]), "norm context")
    visual_path = _verified_ref(root, str(result["visual_npz"]), str(result["visual_npz_sha256"]), "norm visual")
    score_path = _verified_ref(root, str(result["score_npz"]), str(result["score_npz_sha256"]), "norm score")
    context = _read_npz(context_path, str(result["context_npz_sha256"]), "norm context")
    visual = _read_npz(visual_path, str(result["visual_npz_sha256"]), "norm visual")
    score = _read_npz(score_path, str(result["score_npz_sha256"]), "norm score")
    z_key = "z0_saved_float32" if condition == "q00" else "z1_saved_float32"
    if z_key not in context or "natural_audio_embeddings_raw_float32" not in context or "lip_embeddings_float32" not in visual:
        raise BridgeError(f"norm endpoint NPZ lacks arrays: {sample_id}/{condition}")
    video_path = _verified_ref(root, str(result["video_path"]), str(result["video_sha256"]), "norm video")
    z_value, a_value, v_value = context[z_key], context["natural_audio_embeddings_raw_float32"], visual["lip_embeddings_float32"]
    expected_z_sha = row.get("z0_sha256") if condition == "q00" else row.get("z1_sha256")
    if _array_data_sha(z_value) != expected_z_sha:
        raise BridgeError(f"norm endpoint latent hash differs from row plan: {sample_id}/{condition}")
    if _array_data_sha(a_value) != result.get("new_audio_embedding_sha256"):
        raise BridgeError(f"norm RAW-N A array hash differs from result row: {sample_id}/{condition}")
    if _array_data_sha(v_value) != result.get("new_visual_embedding_sha256"):
        raise BridgeError(f"norm V array hash differs from result row: {sample_id}/{condition}")
    score_arrays = {
        "raw_cross": score["cross_distance_raw_float32"], "unit_cross": score["cross_distance_unit_float32"],
        "raw_zero": score["cross_distance_to_zero_raw_float32"], "unit_zero": score["cross_distance_to_zero_unit_float32"],
        "raw_lag": score["distance_matrix_raw_float32"], "unit_lag": score["distance_matrix_unit_float32"],
    }
    return {
        "sample_id": sample_id, "speaker_id": str(row["speaker_id"]), "row_id": str(row["row_id"]),
        "condition": condition, "context_path": str(context_path), "visual_path": str(visual_path),
        "score_path": str(score_path), "video_path": str(video_path), "video_sha256": str(result["video_sha256"]),
        "z": z_value, "A": a_value,
        "pcm16": context.get("natural_pcm16"), "V": v_value,
        "support": _support_arrays(score, "norm"), "score": score_arrays,
        "common_support_n": int(result["common_support_n"]), "audio_count": int(result["audio_row_domain_n"]),
        "visual_count": int(result["visual_query_domain_n"]),
        "artifact_sha256": {"context": str(result["context_npz_sha256"]), "visual": str(result["visual_npz_sha256"]), "score": str(result["score_npz_sha256"]), "video": str(result["video_sha256"])},
    }


def _pair_report(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> dict[str, Any]:
    if left["sample_id"] != right["sample_id"] or left["speaker_id"] != right["speaker_id"]:
        raise BridgeError(f"endpoint row identity mismatch for {label}")
    for side in (left, right):
        for field in ("z", "A", "V"):
            value = np.asarray(side[field])
            if value.dtype != np.float32 or value.ndim != 2 or not np.isfinite(value).all():
                raise BridgeError(f"{label} {side['condition']} {field} violates finite float32 matrix contract")
    if left["pcm16"] is not None and right["pcm16"] is not None:
        pcm_delta = _array_delta(left["pcm16"], right["pcm16"])
    else:
        pcm_delta = {"comparison_status": "not_saved_in_both_endpoint_contexts"}
    support_delta = {
        key: _array_delta(np.asarray(left["support"][key]), np.asarray(right["support"][key]))
        for key in SUPPORT_KEYS
    }
    bridge_q = min(int(left["common_support_n"]), int(right["common_support_n"]), len(left["V"]), len(right["V"]))
    bridge_a = min(len(left["A"]), len(right["A"]))
    matrices: dict[str, Any] = {}
    for geom in GEOMETRIES:
        for kind in ("cross", "zero", "lag"):
            key = f"{geom}_{kind}"
            matrices[key] = _array_delta(np.asarray(left["score"][key]), np.asarray(right["score"][key]), overlap=True)
        harmonized: dict[str, Any] = {}
        for padding, audio_n in (("full_A", bridge_a), ("legacy_query_cutoff", bridge_q)):
            left_lag = _lag_from_cross(left["score"][f"{geom}_cross"], left["score"][f"{geom}_zero"], bridge_q, audio_n)
            right_lag = _lag_from_cross(right["score"][f"{geom}_cross"], right["score"][f"{geom}_zero"], bridge_q, audio_n)
            harmonized[padding] = _array_delta(left_lag, right_lag)
        matrices[f"{geom}_harmonized_lag"] = harmonized
    video_report = _video_pixels(Path(str(left["video_path"])), Path(str(right["video_path"])))
    return {
        "mapping": label, "sample_id": left["sample_id"], "speaker_id": left["speaker_id"],
        "left_condition": left["condition"], "right_condition": right["condition"],
        "path_hashes": {"left": left["artifact_sha256"], "right": right["artifact_sha256"]},
        "latent_z": _array_delta(np.asarray(left["z"]), np.asarray(right["z"]), overlap=True),
        "raw_N_audio_A": _array_delta(np.asarray(left["A"]), np.asarray(right["A"]), overlap=True),
        "lip_V": _array_delta(np.asarray(left["V"]), np.asarray(right["V"]), overlap=True),
        "natural_pcm16": pcm_delta,
        "clock_and_support": {
            "left": {"latent_frames": int(len(left["z"])), "video_path": left["video_path"], "V_windows": int(len(left["V"])), "A_rows": int(len(left["A"])), "common_support_n": int(left["common_support_n"]), "reported_video_frames": left.get("video_frame_count")},
            "right": {"latent_frames": int(len(right["z"])), "video_path": right["video_path"], "V_windows": int(len(right["V"])), "A_rows": int(len(right["A"])), "common_support_n": int(right["common_support_n"]), "reported_video_frames": right.get("video_frame_count")},
            "matched_bridge_query_prefix_n": bridge_q, "matched_audio_prefix_n": bridge_a,
            "support_index_comparisons": support_delta,
        },
        "rendered_video_pixels": video_report,
        "score_matrices": matrices,
        "interpretation": "row-level endpoint reproducibility only; exact old/new runtime parity is not assumed and score differences are not cohort effects",
    }


def _load_main_rows(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol_path, rows_path, results_path = (root / "evaluation_protocol.json", root / "evaluation_rows.jsonl", root / "evaluation_results.jsonl")
    _check_sha(protocol_path, MAIN_PROTOCOL_SHA, "retry3 protocol")
    _check_sha(rows_path, MAIN_ROWS_SHA, "retry3 row plan")
    _check_sha(results_path, MAIN_RESULTS_SHA, "retry3 results")
    plan, results = _jsonl(rows_path), _jsonl(results_path)
    if len(plan) != 2560 or len(results) != 2560:
        raise BridgeError(f"retry3 intended row count differs: {len(plan)}/{len(results)}")
    p = {str(row["row_id"]): row for row in plan}
    r = {str(row["row_id"]): row for row in results}
    if len(p) != 2560 or len(r) != 2560 or set(p) != set(r):
        raise BridgeError("retry3 row plan/results identities are not a complete exact match")
    if any(row.get("status") != "scored" for row in results):
        raise BridgeError("retry3 contains a non-scored row in the bridge contract")
    return p, r


def _main_selected(plan: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    return plan.get("split") == "evaluation" and plan.get("face_id") == "primary" and plan.get("family") == "fourcell" and plan.get("source") == "natural" and plan.get("condition") in {"q00", "q10"} or plan.get("split") == "evaluation" and plan.get("face_id") == "primary" and plan.get("family") == "source_axis" and plan.get("source") == "natural" and plan.get("condition") in {"baseline", "orth_plus"}


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    main_root = args.main_run.resolve(strict=True)
    main_plan, main_results = _load_main_rows(main_root)
    lineage_path = args.main_lineage.resolve(strict=True)
    _check_sha(lineage_path, MAIN_LINEAGE_SHA, "retry3 inherited lineage verification")

    mel_control = args.mel_control.resolve(strict=True)
    mel_artifact = args.mel_artifact_root.resolve(strict=True)
    mel_plan_path = mel_control / "execution_plan.json"
    mel_manifest_path = mel_control / "run_manifest.json"
    mel_results_path = mel_control / "execution_results.jsonl"
    mel_runtime_path = mel_control / "execution_runtime.json"
    _check_sha(mel_plan_path, MEL_PLAN_SHA, "MEL execution plan")
    _check_sha(mel_manifest_path, MEL_MANIFEST_SHA, "MEL run manifest")
    _check_sha(mel_results_path, MEL_RESULTS_SHA, "MEL execution rows")
    _check_sha(mel_runtime_path, MEL_RUNTIME_SHA, "MEL execution runtime")
    mel_plan = _json(mel_plan_path)
    mel_manifest = _json(mel_manifest_path)
    mel_rows = _jsonl(mel_results_path)
    ledger = mel_manifest.get("artifact_ledger")
    if not isinstance(ledger, Mapping) or len(ledger) != int(mel_manifest.get("artifact_count", -1)):
        raise BridgeError("MEL run manifest artifact ledger is malformed")
    if mel_manifest.get("status") != "complete" or (mel_manifest.get("planned_rows"), mel_manifest.get("completed_rows"), mel_manifest.get("planned_cells"), mel_manifest.get("failed_cells_or_rows")) != (80, 80, 320, 0):
        raise BridgeError("MEL GPU manifest is not complete 80×4")
    if len(mel_rows) != 80 or any(row.get("status") != "complete" for row in mel_rows):
        raise BridgeError("MEL GPU result rows are incomplete")
    mel_by_id = {str(row["id"]): row for row in mel_rows}
    if len(mel_by_id) != 80:
        raise BridgeError("MEL result IDs are not unique")
    if mel_plan.get("input_identity_sha256") != MEL_INPUT_IDENTITY_SHA:
        raise BridgeError("MEL input identity digest differs from the frozen selected cohort")
    planned_samples = mel_plan.get("planned_samples")
    if not isinstance(planned_samples, list) or len(planned_samples) != 80:
        raise BridgeError("MEL execution plan does not retain all 80 source samples")
    mel_source_by_id = {str(row["id"]): row for row in planned_samples}
    if len(mel_source_by_id) != 80:
        raise BridgeError("MEL planned sample IDs are not unique")
    for sample_id, mel_row in mel_by_id.items():
        if sample_id not in mel_source_by_id or mel_row.get("speaker") != mel_source_by_id[sample_id].get("speaker"):
            raise BridgeError(f"MEL result identity differs from frozen input identity: {sample_id}")

    norm_root = args.norm_run.resolve(strict=True)
    norm_checks = {
        "evaluation_protocol.json": NORM_PROTOCOL_SHA, "evaluation_rows.jsonl": NORM_ROWS_SHA,
        "evaluation_results.jsonl": NORM_RESULTS_SHA, "run_status.json": NORM_STATUS_SHA,
        "code_seal.json": NORM_CODE_SEAL_SHA, "all_files_sha256.json": NORM_ALL_FILES_SHA,
    }
    for name, expected in norm_checks.items():
        _check_sha(norm_root / name, expected, f"norm {name}")
    norm_status = _json(norm_root / "run_status.json")
    if (norm_status.get("status"), norm_status.get("completed_samples"), norm_status.get("completed_rows"), norm_status.get("failed_samples"), norm_status.get("failed_rows")) != ("complete", 80, 320, 0, 0):
        raise BridgeError("norm run status is incomplete")
    norm_rows, norm_results = _jsonl(norm_root / "evaluation_rows.jsonl"), _jsonl(norm_root / "evaluation_results.jsonl")
    if len(norm_rows) != 320 or len(norm_results) != 320:
        raise BridgeError("norm row/result counts are not 320")
    norm_plan = {str(row["row_id"]): row for row in norm_rows}
    norm_by_key = {(str(row["sample_id"]), str(row["condition"])): row for row in norm_results}
    if len(norm_plan) != 320 or len(norm_by_key) != 320 or set(norm_plan) != {str(row["row_id"]) for row in norm_results}:
        raise BridgeError("norm plan/result row identities are not an exact 320-row match")
    if any(row.get("status") != "scored" for row in norm_results):
        raise BridgeError("norm has a non-scored result row")

    selected_main = [(plan, main_results[rid]) for rid, plan in main_plan.items() if _main_selected(plan, main_results[rid])]
    main_four = {(str(p["sample_id"]), str(p["condition"])): (p, r) for p, r in selected_main if p.get("family") == "fourcell"}
    main_axis = {(str(p["sample_id"]), str(p["condition"])): (p, r) for p, r in selected_main if p.get("family") == "source_axis"}
    if len(main_four) != 160 or len(main_axis) != 160:
        raise BridgeError(f"main selected rows incomplete: fourcell={len(main_four)}, axis={len(main_axis)}")
    if set(mel_by_id) != {sample for sample, _ in main_four} or set(mel_by_id) != {sample for sample, _ in main_axis}:
        raise BridgeError("MEL and retry3 sample IDs differ")
    norm_plan_by_key = {(str(row["sample_id"]), str(row["condition"])): row for row in norm_rows}
    for sample_id in mel_by_id:
        main_hashes = {
            main_four[(sample_id, "q00")][0].get("natural_input_sha256"),
            main_four[(sample_id, "q10")][0].get("natural_input_sha256"),
            main_axis[(sample_id, "baseline")][0].get("natural_input_sha256"),
            main_axis[(sample_id, "orth_plus")][0].get("natural_input_sha256"),
        }
        mel_hash = mel_source_by_id[sample_id].get("source_wav_file_sha256")
        norm_hashes = {norm_plan_by_key[(sample_id, condition)].get("natural_input_sha256") for condition in ("q00", "q11")}
        if len(main_hashes) != 1 or None in main_hashes or mel_hash not in main_hashes or norm_hashes != main_hashes:
            raise BridgeError(f"RAW natural waveform identity differs across bridge runs: {sample_id}")
    return {"main_root": main_root, "main_plan": main_plan, "main_results": main_results,
            "main_four": main_four, "main_axis": main_axis, "mel_control": mel_control,
            "mel_artifact": mel_artifact, "mel_plan": mel_plan, "mel_manifest": mel_manifest,
            "mel_ledger": ledger, "mel_by_id": mel_by_id, "mel_source_by_id": mel_source_by_id,
            "norm_root": norm_root, "norm_plan": norm_plan, "norm_by_key": norm_by_key,
            "lineage_path": lineage_path}


def _bridge_one(
    main_root: Path, main_pair: tuple[Mapping[str, Any], Mapping[str, Any]],
    endpoint: Mapping[str, Any], destination: str, label: str,
) -> dict[str, Any]:
    plan, result = main_pair
    if plan.get("audio_key") != "natural_audio_embeddings_raw_float32":
        raise BridgeError(f"main bridge endpoint is not RAW-N A: {plan.get('row_id')}")
    if result.get("status") != "scored":
        raise BridgeError(f"main bridge endpoint is not scored: {plan.get('row_id')}")
    z_key = str(plan.get("visual_id"))
    if z_key == "natural_raw":
        latent_key = "natural_z_raw_float32"
    elif z_key == "natural_fixed":
        latent_key = "natural_z_fixed_float32"
    elif z_key == "axis_orth_plus":
        latent_key = "axis_z_orth_plus_float32"
    else:
        raise BridgeError(f"unexpected main bridge visual ID: {z_key}")
    main = _load_main(main_root, plan, result, latent_key)
    return _pair_report(main, endpoint, label)


def _consumer(args: argparse.Namespace) -> None:
    prepared = _prepare(args)
    rows: list[dict[str, Any]] = []
    main_root: Path = prepared["main_root"]
    mel_artifact: Path = prepared["mel_artifact"]
    norm_root: Path = prepared["norm_root"]
    mel_by_id: Mapping[str, Any] = prepared["mel_by_id"]
    norm_by_key: Mapping[tuple[str, str], Any] = prepared["norm_by_key"]
    mel_ledger: Mapping[str, str] = prepared["mel_ledger"]

    for sample_id in sorted(mel_by_id):
        mel_row = mel_by_id[sample_id]
        for old_condition, mel_condition in (("q00", "q00"), ("q10", "q11")):
            main_pair = prepared["main_four"][(sample_id, old_condition)]
            endpoint = _load_mel(mel_artifact, mel_artifact, mel_row, mel_condition, mel_ledger)
            rows.append(_bridge_one(main_root, main_pair, endpoint, mel_condition, f"retry3_fourcell_{old_condition}_to_mel_{mel_condition}"))
        for old_condition, norm_condition in (("baseline", "q00"), ("orth_plus", "q11")):
            main_pair = prepared["main_axis"][(sample_id, old_condition)]
            norm_result = norm_by_key[(sample_id, norm_condition)]
            norm_plan = prepared["norm_plan"][str(norm_result["row_id"])]
            if norm_plan.get("condition") != norm_condition or norm_plan.get("face_id") != "primary":
                raise BridgeError(f"norm bridge row identity mismatch: {sample_id}/{norm_condition}")
            endpoint = _load_norm(norm_root, norm_plan, norm_result)
            rows.append(_bridge_one(main_root, main_pair, endpoint, norm_condition, f"retry3_axis_{old_condition}_to_norm_{norm_condition}"))

    expected = 320
    if len(rows) != expected or len({(row["sample_id"], row["mapping"]) for row in rows}) != expected:
        raise BridgeError(f"endpoint bridge rows incomplete/duplicated: {len(rows)}/{expected}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["mapping"], []).append(row)
    diagnostics: dict[str, Any] = {}
    for label, group in groups.items():
        summaries: dict[str, Any] = {}
        for field in ("latent_z", "raw_N_audio_A", "lip_V"):
            values = [item[field] for item in group]
            summaries[field] = {
                "rows": len(values),
                "bit_exact_rows": sum(bool(item.get("bit_exact")) for item in values),
                "shape_mismatch_rows": sum(not bool(item.get("shape_match")) for item in values),
                "max_abs_difference_max": max((float(item["max_abs_difference"]) for item in values if item.get("max_abs_difference") is not None), default=None),
                "mean_abs_difference_mean": float(np.mean([item["mean_abs_difference"] for item in values if item.get("mean_abs_difference") is not None], dtype=np.float64)) if any(item.get("mean_abs_difference") is not None for item in values) else None,
            }
        video_values = [item["rendered_video_pixels"] for item in group]
        summaries["rendered_video_pixels"] = {
            "rows": len(video_values),
            "frame_count_match_rows": sum(bool(item["frame_count_match"]) for item in video_values),
            "pixel_exact_fraction_mean": float(np.mean([item["pixel_exact_fraction"] for item in video_values if item["pixel_exact_fraction"] is not None], dtype=np.float64)) if any(item["pixel_exact_fraction"] is not None for item in video_values) else None,
            "pixel_mean_abs_difference_mean": float(np.mean([item["pixel_mean_abs_difference"] for item in video_values if item["pixel_mean_abs_difference"] is not None], dtype=np.float64)) if any(item["pixel_mean_abs_difference"] is not None for item in video_values) else None,
            "pixel_max_abs_difference_max": max((int(item["pixel_max_abs_difference"]) for item in video_values if item["pixel_max_abs_difference"] is not None), default=None),
        }
        diagnostics[label] = summaries

    output = {
        "schema_version": 1,
        "analysis_type": "independent_retry3_endpoint_artifact_bridge",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "effects_read": False,
        "model_forward_performed": False,
        "cohort_contrasts_or_inference_computed": False,
        "consumer_source": {"path": str(Path(__file__).resolve()), "sha256": _sha(Path(__file__).resolve())},
        "input_seals": {
            "main_protocol_sha256": MAIN_PROTOCOL_SHA, "main_rows_sha256": MAIN_ROWS_SHA,
            "main_results_sha256": MAIN_RESULTS_SHA, "main_lineage_sha256": MAIN_LINEAGE_SHA,
            "mel_execution_plan_sha256": MEL_PLAN_SHA, "mel_run_manifest_sha256": MEL_MANIFEST_SHA,
            "mel_execution_results_sha256": MEL_RESULTS_SHA, "mel_runtime_sha256": MEL_RUNTIME_SHA,
            "norm_protocol_sha256": NORM_PROTOCOL_SHA, "norm_rows_sha256": NORM_ROWS_SHA,
            "norm_results_sha256": NORM_RESULTS_SHA, "norm_status_sha256": NORM_STATUS_SHA,
            "norm_code_seal_sha256": NORM_CODE_SEAL_SHA, "norm_all_files_sha256": NORM_ALL_FILES_SHA,
            "main_run_root": str(prepared["main_root"]), "mel_artifact_root": str(prepared["mel_artifact"]),
            "mel_control_root": str(prepared["mel_control"]), "norm_run_root": str(prepared["norm_root"]),
            "main_lineage_path": str(prepared["lineage_path"]),
        },
        "protocol": {
            "mapping": [
                {"old": "retry3 natural fourcell q00", "new": "MEL q00=M0 exact RAW endpoint"},
                {"old": "retry3 natural fourcell q10", "new": "MEL q11=M1 exact FIXED endpoint"},
                {"old": "retry3 source-axis baseline", "new": "norm q00=exact saved z0"},
                {"old": "retry3 source-axis orth_plus", "new": "norm q11=exact saved z1"},
            ],
            "face": "primary", "audio": "RAW natural A throughout",
            "comparisons": ["latent z arrays", "RAW-N audio A arrays", "lip V arrays", "support indices/clocks", "full V×A raw/unit/zero/lag arrays", "decoded rendered-video pixels"],
            "padding": "stored lag matrices retained with each run's own policy; additive matched-query/full-A and matched-query/legacy-cutoff matrices are reconstructed from each saved full cross matrix",
            "inference": "descriptive artifact bridge per sample; no group effect test or runtime-causality claim",
        },
        "verification": {"planned_bridge_rows": expected, "verified_bridge_rows": len(rows), "samples": 80,
                         "main_fourcell_rows": 160, "main_axis_rows": 160, "mel_rows": 80, "norm_rows": 320,
                         "all_referenced_artifact_hashes_verified": True},
        "group_diagnostics": diagnostics,
        "rows": rows,
    }
    out = args.output.resolve()
    if out.exists():
        raise BridgeError(f"output already exists; choose fresh path: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "output": str(out), "sha256": _sha(out), "effects_read": False}, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--main-lineage", type=Path, required=True)
    parser.add_argument("--mel-control", type=Path, required=True)
    parser.add_argument("--mel-artifact-root", type=Path, required=True)
    parser.add_argument("--norm-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", help="validate sealed row manifests and identities without reading score artifacts")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.preflight_only:
            prepared = _prepare(args)
            print(json.dumps({"status": "PREFLIGHT_PASS", "main_fourcell_rows": len(prepared["main_four"]),
                              "main_axis_rows": len(prepared["main_axis"]), "mel_rows": len(prepared["mel_by_id"]),
                              "norm_rows": len(prepared["norm_by_key"]), "input_identity_rows": len(prepared["mel_source_by_id"]),
                              "effects_read": False}, separators=(",", ":")))
        else:
            _consumer(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
