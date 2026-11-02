"""Read-only algebraic decomposition of sealed proxy/native SyncNet embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_SHA256 = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
COLLECTION_RECEIPT_SHA256 = "68aa2942c9a5b9aa3722d8a65993aa3609b010f8eee8f866f9158eaede229432"
ARRAY_KEYS = (
    "proxy_distance_by_query_lag_float32",
    "native_distance_by_query_lag_float32",
    "proxy_video_embedding_float32",
    "native_video_embedding_center_float32",
    "proxy_audio_embedding_full_float32",
    "proxy_audio_embedding_aligned_segment_float32",
    "native_audio_embedding_full_float32",
)
LAGS = np.arange(-15, 16, dtype=np.int64)
EPS = 1e-6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_float32(name: str, value: np.ndarray, ndim: int = 2) -> np.ndarray:
    if value.dtype != np.dtype("float32") or value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite float32 ndim={ndim}; got {value.dtype} {value.shape}")
    return np.ascontiguousarray(value)


def metric_summary(matrix: np.ndarray) -> dict[str, Any]:
    """Diagnostic float64 reduction of a reconstructed Nquery x 31 surface."""
    curve = matrix.astype(np.float64).mean(axis=0)
    best_index = int(np.argmin(curve))
    zero_index = int(np.flatnonzero(LAGS == 0)[0])
    return {
        "B_median_f64_diagnostic": float(np.median(curve)),
        "Dmin_f64_diagnostic": float(np.min(curve)),
        "D0_f64_diagnostic": float(curve[zero_index]),
        "Cbest_f64_diagnostic": float(np.median(curve) - np.min(curve)),
        "Canchor_f64_diagnostic": float(np.median(curve) - curve[zero_index]),
        "best_lag_frames_first_argmin": int(LAGS[best_index]),
        "far_lag_mean_minus_D0_abs_ge_4": float(curve[np.abs(LAGS) >= 4].mean() - curve[zero_index]),
    }


def pairwise_surface(video: np.ndarray, audio: np.ndarray, q0: int) -> np.ndarray:
    q = np.arange(video.shape[0], dtype=np.int64)
    audio_idx = q0 + q[:, None] + LAGS[None, :]
    if int(audio_idx.min()) < 0 or int(audio_idx.max()) >= audio.shape[0]:
        raise ValueError("source-clock query/lag window exceeds aligned audio embedding support")
    candidates = audio[audio_idx]
    delta = video[:, None, :].astype(np.float64) - candidates.astype(np.float64) + EPS
    return np.sqrt(np.sum(delta * delta, axis=-1, dtype=np.float64))


def cosine_surface(video: np.ndarray, audio: np.ndarray, q0: int) -> np.ndarray:
    q = np.arange(video.shape[0], dtype=np.int64)
    audio_idx = q0 + q[:, None] + LAGS[None, :]
    candidates = audio[audio_idx].astype(np.float64)
    vectors = video.astype(np.float64)[:, None, :]
    numerator = np.sum(vectors * candidates, axis=-1)
    denom = np.linalg.norm(vectors, axis=-1) * np.linalg.norm(candidates, axis=-1)
    return numerator / np.maximum(denom, np.finfo(np.float64).tiny)


def process_row(
    row: dict[str, Any],
    receipt_row: dict[str, Any],
    local_npz: Path,
) -> dict[str, Any]:
    if not local_npz.is_file() or local_npz.stat().st_size != int(receipt_row["output_bytes"]):
        raise ValueError(f"missing or wrong-sized compact array file: {row['sample_id']}")
    local_sha = sha256_file(local_npz)
    if local_sha != receipt_row["output_sha256"]:
        raise ValueError(f"compact array file SHA mismatch: {row['sample_id']}")

    with np.load(local_npz, allow_pickle=False) as archive:
        if set(ARRAY_KEYS) - set(archive.files):
            raise ValueError(f"compact array archive lacks required keys: {row['sample_id']}")
        arrays = {key: require_float32(key, archive[key]) for key in ARRAY_KEYS}
    array_receipts = receipt_row["arrays"]
    for key, value in arrays.items():
        expected = array_receipts[key]
        if list(value.shape) != expected["shape"] or array_sha256(value) != expected["array_sha256"]:
            raise ValueError(f"sealed array digest/shape mismatch for {row['sample_id']}/{key}")

    proxy_m = arrays["proxy_distance_by_query_lag_float32"]
    native_m = arrays["native_distance_by_query_lag_float32"]
    proxy_v = arrays["proxy_video_embedding_float32"]
    native_v = arrays["native_video_embedding_center_float32"]
    proxy_a_full = arrays["proxy_audio_embedding_full_float32"]
    proxy_a = arrays["proxy_audio_embedding_aligned_segment_float32"]
    native_a = arrays["native_audio_embedding_full_float32"]
    if proxy_m.shape != (27, 31) or native_m.shape != (27, 31):
        raise ValueError(f"distance matrix shape mismatch: {row['sample_id']}")
    if proxy_v.shape != (27, 1024) or native_v.shape != (27, 1024):
        raise ValueError(f"visual embedding shape mismatch: {row['sample_id']}")
    if proxy_a.shape != native_a.shape or proxy_a.shape[1] != 1024:
        raise ValueError(f"audio source-clock common support mismatch: {row['sample_id']}")
    if proxy_a.shape[0] > proxy_a_full.shape[0]:
        raise ValueError(f"aligned proxy audio segment exceeds full sequence: {row['sample_id']}")

    q0 = int(row["center_query_start_source_frame"])
    if int(row["center_query_start_selected_track_local"]) != q0 or int(row["selected_track_source_frame_start"]) != 0:
        raise ValueError(f"unexpected center/source-clock mapping: {row['sample_id']}")
    if not np.array_equal(proxy_a, proxy_a_full[: proxy_a.shape[0]]):
        raise ValueError(f"proxy aligned audio is not the source-clock-zero prefix: {row['sample_id']}")
    q = np.arange(27, dtype=np.int64)
    audio_index = q0 + q[:, None] + LAGS[None, :]
    if audio_index.min() < 0 or audio_index.max() >= native_a.shape[0]:
        raise ValueError(f"center 27x31 lag windows leave common audio support: {row['sample_id']}")

    # Rebuild four raw-distance surfaces to verify clock alignment and isolate
    # the very small native/proxy audio-feature change from the visual change.
    pp = pairwise_surface(proxy_v, proxy_a, q0)
    nn = pairwise_surface(native_v, native_a, q0)
    pn = pairwise_surface(proxy_v, native_a, q0)
    np_cross = pairwise_surface(native_v, proxy_a, q0)
    saved_residuals = {
        "proxy_saved_vs_embedding_rebuild_max_abs": float(np.max(np.abs(pp - proxy_m.astype(np.float64)))),
        "native_saved_vs_embedding_rebuild_max_abs": float(np.max(np.abs(nn - native_m.astype(np.float64)))),
    }

    # Two-factor Shapley on visual embedding norm and direction, holding the
    # scored audio vector fixed to the native source-clock feature at each lag.
    p_norm = np.linalg.norm(proxy_v.astype(np.float64), axis=1)
    n_norm = np.linalg.norm(native_v.astype(np.float64), axis=1)
    p_hat = proxy_v.astype(np.float64) / np.maximum(p_norm[:, None], np.finfo(np.float64).tiny)
    n_hat = native_v.astype(np.float64) / np.maximum(n_norm[:, None], np.finfo(np.float64).tiny)
    candidates_native = native_a[audio_index].astype(np.float64)
    d00 = np.sqrt(np.sum((p_norm[:, None, None] * p_hat[:, None, :] - candidates_native + EPS) ** 2, axis=-1))
    d10 = np.sqrt(np.sum((n_norm[:, None, None] * p_hat[:, None, :] - candidates_native + EPS) ** 2, axis=-1))
    d01 = np.sqrt(np.sum((p_norm[:, None, None] * n_hat[:, None, :] - candidates_native + EPS) ** 2, axis=-1))
    d11 = np.sqrt(np.sum((n_norm[:, None, None] * n_hat[:, None, :] - candidates_native + EPS) ** 2, axis=-1))
    norm_shapley = 0.5 * ((d10 - d00) + (d11 - d01))
    direction_shapley = 0.5 * ((d01 - d00) + (d11 - d10))
    visual_total_on_native_audio = d11 - d00
    additive_residual = float(np.max(np.abs(norm_shapley + direction_shapley - visual_total_on_native_audio)))

    lag_masks = {
        "zero_lag": LAGS == 0,
        "near_lags_abs_1_2": (np.abs(LAGS) >= 1) & (np.abs(LAGS) <= 2),
        "far_lags_abs_ge_4": np.abs(LAGS) >= 4,
        "all_lags": np.ones_like(LAGS, dtype=bool),
    }
    lag_group_shapley: dict[str, Any] = {}
    for name, mask in lag_masks.items():
        lag_group_shapley[name] = {
            "norm_change_contribution": float(norm_shapley[:, mask].mean()),
            "direction_change_contribution": float(direction_shapley[:, mask].mean()),
            "total_visual_change_holding_native_audio": float(visual_total_on_native_audio[:, mask].mean()),
            "query_lag_cells": int(norm_shapley[:, mask].size),
        }
    lag_group_shapley["far_minus_zero_lag"] = {
        key: lag_group_shapley["far_lags_abs_ge_4"][key] - lag_group_shapley["zero_lag"][key]
        for key in (
            "norm_change_contribution",
            "direction_change_contribution",
            "total_visual_change_holding_native_audio",
        )
    }

    p_cos = cosine_surface(proxy_v, native_a, q0)
    n_cos = cosine_surface(native_v, native_a, q0)
    zero_col = int(np.flatnonzero(LAGS == 0)[0])
    far_mask = np.abs(LAGS) >= 4
    p_audio_norm = np.linalg.norm(proxy_a.astype(np.float64), axis=1)
    n_audio_norm = np.linalg.norm(native_a.astype(np.float64), axis=1)
    video_angle = np.arccos(np.clip(np.sum(p_hat * n_hat, axis=1), -1.0, 1.0)) * (180.0 / math.pi)

    summary_row: dict[str, Any] = {
        "row_id": row["row_id"],
        "sample_id": row["sample_id"],
        "speaker_id": row["speaker_id"],
        "split": row["split"],
        "historical_exposed": bool(row["historical_exposed"]),
        "method_id": row["method_id"],
        "compact_npz_path": str(local_npz),
        "compact_npz_bytes": local_npz.stat().st_size,
        "compact_npz_sha256": local_sha,
        "q0_source_clock_frame": q0,
        "center_query_source_clock_frames": [q0, q0 + 26],
        "lagged_audio_source_clock_frame_bounds_inclusive": [int(audio_index.min()), int(audio_index.max())],
        "proxy_audio_full_embedding_count": int(proxy_a_full.shape[0]),
        "proxy_audio_aligned_embedding_count": int(proxy_a.shape[0]),
        "native_audio_embedding_count": int(native_a.shape[0]),
        "source_clock_aligned_audio_max_abs_delta": float(np.max(np.abs(proxy_a.astype(np.float64) - native_a.astype(np.float64)))),
        "source_clock_aligned_audio_mean_abs_delta": float(np.mean(np.abs(proxy_a.astype(np.float64) - native_a.astype(np.float64)))),
        "proxy_video_norm_mean": float(p_norm.mean()),
        "native_video_norm_mean": float(n_norm.mean()),
        "native_minus_proxy_video_norm_mean": float(n_norm.mean() - p_norm.mean()),
        "proxy_video_norm_min_max": [float(p_norm.min()), float(p_norm.max())],
        "native_video_norm_min_max": [float(n_norm.min()), float(n_norm.max())],
        "proxy_audio_norm_mean": float(p_audio_norm.mean()),
        "native_audio_norm_mean": float(n_audio_norm.mean()),
        "native_minus_proxy_audio_norm_mean": float(n_audio_norm.mean() - p_audio_norm.mean()),
        "native_proxy_visual_direction_angle_degrees_mean": float(video_angle.mean()),
        "native_proxy_visual_direction_angle_degrees_max": float(video_angle.max()),
        "cosine_to_native_audio_proxy_zero_lag_mean": float(p_cos[:, zero_col].mean()),
        "cosine_to_native_audio_native_zero_lag_mean": float(n_cos[:, zero_col].mean()),
        "cosine_to_native_audio_proxy_far_lag_mean_abs_ge_4": float(p_cos[:, far_mask].mean()),
        "cosine_to_native_audio_native_far_lag_mean_abs_ge_4": float(n_cos[:, far_mask].mean()),
        "cosine_far_minus_zero_proxy": float(p_cos[:, far_mask].mean() - p_cos[:, zero_col].mean()),
        "cosine_far_minus_zero_native": float(n_cos[:, far_mask].mean() - n_cos[:, zero_col].mean()),
        "cosine_native_minus_proxy_zero_lag": float(n_cos[:, zero_col].mean() - p_cos[:, zero_col].mean()),
        "raw_distance_proxy_saved_float32": row["proxy"]["metrics"],
        "raw_distance_native_saved_float32": row["official_native"]["metrics"],
        "raw_distance_reconstruction_residuals": saved_residuals,
        "raw_surface_proxy_visual_proxy_audio_numpy_f64": metric_summary(pp),
        "raw_surface_proxy_visual_native_audio_numpy_f64": metric_summary(pn),
        "raw_surface_native_visual_proxy_audio_numpy_f64": metric_summary(np_cross),
        "raw_surface_native_visual_native_audio_numpy_f64": metric_summary(nn),
        "visual_norm_direction_shapley_raw_distance_mean": {
            "norm_change_contribution": float(norm_shapley.mean()),
            "direction_change_contribution": float(direction_shapley.mean()),
            "total_visual_change_holding_native_audio": float(visual_total_on_native_audio.mean()),
            "max_additivity_residual": additive_residual,
            "scope": "descriptive algebraic decomposition over 27x31 raw distances; native audio held fixed; not a causal attribution",
        },
        "visual_norm_direction_shapley_by_lag_group": lag_group_shapley,
    }
    return summary_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--collection-receipt", required=True, type=Path)
    parser.add_argument("--compact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit("output root already exists; refusing overwrite")
    for path, expected_sha, label in (
        (args.summary, SUMMARY_SHA256, "parity summary"),
        (args.collection_receipt, COLLECTION_RECEIPT_SHA256, "compact collection receipt"),
    ):
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise SystemExit(f"{label} missing or raw SHA mismatch")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    receipt = json.loads(args.collection_receipt.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or len(summary.get("rows", [])) != 8:
        raise SystemExit("full512 summary is not the sealed eight-row completion")
    if receipt.get("status") != "PASS" or len(receipt.get("rows", [])) != 8:
        raise SystemExit("compact-array collection receipt is not PASS for eight rows")
    if receipt.get("summary_raw_sha256") != SUMMARY_SHA256:
        raise SystemExit("compact array receipt does not bind the expected summary")
    # The parity summary's row_id includes the full-context method suffix,
    # while the compact collector names files by sample_id. Join through the
    # unique sample identity and independently bind the source NPZ SHA below.
    by_sample = {str(item["row_id"]): item for item in receipt["rows"]}
    if len(by_sample) != 8:
        raise SystemExit("compact receipt sample identities are not unique")
    summary_samples = [str(item["sample_id"]) for item in summary["rows"]]
    if len(set(summary_samples)) != 8 or set(summary_samples) != set(by_sample):
        raise SystemExit("summary and compact receipt sample identities differ")

    args.output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in summary["rows"]:
        compact = by_sample.get(str(row["sample_id"]))
        if compact is None or compact.get("source_sha256") != row["artifacts"]["parity_arrays_sha256"]:
            raise SystemExit(f"compact arrays do not bind the source NPZ: {row['row_id']}")
        if compact.get("source_after_sha256") != row["artifacts"]["parity_arrays_sha256"] or not compact.get("source_unchanged"):
            raise SystemExit(f"source array was not preserved by compact collection: {row['row_id']}")
        local_npz = args.compact_root / f"{row['sample_id']}_compact_arrays.npz"
        rows.append(process_row(row, compact, local_npz))

    numeric_fields: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row["visual_norm_direction_shapley_raw_distance_mean"].items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric_fields.setdefault(key, []).append(float(value))
    aggregate = {
        key: {
            "equal_clip_mean": float(statistics.mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "n_clips": len(values),
        }
        for key, values in numeric_fields.items()
    }
    lag_group_fields: dict[str, list[float]] = {}
    for row in rows:
        for group, group_values in row["visual_norm_direction_shapley_by_lag_group"].items():
            for field, value in group_values.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    lag_group_fields.setdefault(f"{group}/{field}", []).append(float(value))
    lag_group_aggregate = {
        key: {
            "equal_clip_mean": float(statistics.mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "n_clips": len(values),
        }
        for key, values in lag_group_fields.items()
    }
    result: dict[str, Any] = {
        "schema": "cross_tfg_full512_embedding_norm_direction_decomposition_v2",
        "status": "PASS",
        "scope": "eight historical exposed fit/rms_fixed rows only; read-only sealed embedding/matrix algebra; no model forward, no dev/confirm audio, no score-source modification",
        "summary_path": str(args.summary),
        "summary_bytes": args.summary.stat().st_size,
        "summary_raw_sha256": SUMMARY_SHA256,
        "collection_receipt_path": str(args.collection_receipt),
        "collection_receipt_bytes": args.collection_receipt.stat().st_size,
        "collection_receipt_raw_sha256": COLLECTION_RECEIPT_SHA256,
        "array_formula": "SyncNet pairwise_distance convention ||video-audio+1e-6||_2; angular diagnostic uses cosine of L2-normalized saved embeddings; distance-surface reductions are explicitly NumPy float64 diagnostics, while frozen saved float32 metrics remain authoritative",
        "shapley_formula": "two-factor norm/direction Shapley on each raw distance cell, holding the native source-clock audio embedding fixed; averaged equally over each clip's 27x31 matrix",
        "rows": rows,
        "equal_clip_shapley_aggregate": aggregate,
        "equal_clip_lag_group_shapley_aggregate": lag_group_aggregate,
    }
    result["payload_sha256"] = canonical_sha256(result)
    output_path = args.output_root / "embedding_norm_direction_decomposition.json"
    output_path.write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "payload_sha256": result["payload_sha256"],
        "rows": len(rows),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
