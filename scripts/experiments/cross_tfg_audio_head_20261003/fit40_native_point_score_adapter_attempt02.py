"""Readback-only adapter for native SPSA point rows with verified raw evidence.

The six-method initial-family ingest stays on selector06's strict producer
summary contract. This adapter is only for a separately validated one-method
native point: if both producer summary maps are absent, it derives row-local
support and metrics from the pinned official NPZ plus the already verified
source-frame map, and returns an explicit derivation receipt. It never writes
back to the producer ledger.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit_only_spsa_selection_attempt06 as selector,
)


DERIVATION_SCHEMA = "fit40_native_point_support_derivation_v1"
CONTEXT_SCHEMA = "fit40_native_point_support_context_v1"
STAGEA_CORE_BYTES = 96537
STAGEA_CORE_SHA256 = "6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e"
ARRAY_DTYPES = {
    "query_frame_index": np.dtype("<i4"),
    "query_timestamp_num": np.dtype("<i8"),
    "query_timestamp_den": np.dtype("<i8"),
    "track_frame_index": np.dtype("<i4"),
    "valid_query_mask": np.dtype("bool"),
    "distance_by_query_lag": np.dtype("<f4"),
    "lag_offsets_frames": np.dtype("<i4"),
}


class NativePointScoreError(ValueError):
    """A native point's raw evidence cannot support a canonical score row."""


def _canonical(value: Any) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _pin(value: Any, label: str, *, self_pin: bool = False) -> dict[str, Any]:
    required = {"path", "bytes", "sha256", "self_sha256"} if self_pin else {"path", "bytes", "sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise NativePointScoreError(f"{label} pin does not have its exact reviewed shape")
    if (not isinstance(value.get("path"), str) or not value["path"]
            or type(value.get("bytes")) is not int or value["bytes"] <= 0
            or not selector._is_sha256(value.get("sha256"))):
        raise NativePointScoreError(f"{label} pin is malformed")
    if self_pin and not selector._is_sha256(value.get("self_sha256")):
        raise NativePointScoreError(f"{label} self pin is malformed")
    return dict(value)


def _support_sha(support_id: str, keys: tuple[tuple[int, int, int, int], ...]) -> str:
    return selector._sha256(selector._canonical({
        "support_id": support_id,
        "keys": [list(key) for key in keys],
    }))


def _support_keys(row: selector.FitScoreRow, support_id: str) -> tuple[tuple[int, int, int, int], ...]:
    full = tuple(sorted(
        (key for key, valid in zip(row.query_keys, row.valid_query_mask) if valid),
        key=selector._query_sort_key,
    ))
    if support_id == "full":
        return full
    if support_id == "guard20":
        return full[20:-20] if len(full) >= 41 else ()
    raise NativePointScoreError(f"unsupported derived support {support_id!r}")


def _metric_payload(row: selector.FitScoreRow,
                    keys: tuple[tuple[int, int, int, int], ...],
                    *, anchor_column: int) -> tuple[list[float], dict[str, Any]]:
    if not keys:
        return [], {
            "status": "EMPTY", "reason": "no common valid queries", "B": None,
            "Dmin": None, "D0": None, "Cbest": None, "Canchor": None,
            "best_lag_frames": None, "offset_frames": None,
            "official_av_offset_frames": None, "anchor_column": None,
            "zero_lag_column": None,
        }
    assert row.distance_by_query_lag is not None
    index = {key: i for i, key in enumerate(row.query_keys)}
    selected = np.ascontiguousarray(
        row.distance_by_query_lag[[index[key] for key in keys]], dtype=np.float32)
    try:
        import torch
    except ImportError as exc:
        raise NativePointScoreError("canonical torch_cpu summary is unavailable") from exc
    torch.set_num_threads(1)
    curve_tensor = torch.mean(torch.from_numpy(np.ascontiguousarray(selected.T)), dim=1)
    minimum = torch.min(curve_tensor, dim=0)
    median = torch.median(curve_tensor)
    dmin = minimum.values
    d0 = curve_tensor[selector.ZERO_LAG_COLUMN]
    danchor = curve_tensor[anchor_column]
    best_column = int(minimum.indices.item())
    best_lag = int(row.lag_offsets_frames[best_column])
    curve = curve_tensor.detach().numpy().astype(np.float32, copy=False).tolist()
    return curve, {
        "B": float(np.float32(median.item())),
        "Dmin": float(np.float32(dmin.item())),
        "D0": float(np.float32(d0.item())),
        "Cbest": float(np.float32((median - dmin).item())),
        "Canchor": float(np.float32((median - danchor).item())),
        "best_lag_frames": best_lag,
        "offset_frames": -best_lag,
        "official_av_offset_frames": -best_lag,
        "anchor_column": anchor_column,
        "zero_lag_column": selector.ZERO_LAG_COLUMN,
    }


def _validate_present_summaries(score: Mapping[str, Any],
                                supports: Mapping[str, Mapping[str, Any]],
                                metrics: Mapping[str, Mapping[str, Any]],
                                curves: Mapping[str, list[float]]) -> None:
    support_doc = score.get("support_by_support")
    metric_doc = score.get("metrics_by_support")
    curve_doc = score.get("lag_curve_by_support")
    if (not isinstance(support_doc, Mapping) or not isinstance(metric_doc, Mapping)
            or not isinstance(curve_doc, Mapping)
            or set(support_doc) != {"full", "guard20"}
            or set(metric_doc) != {"full", "guard20"}
            or set(curve_doc) != {"full", "guard20"}):
        raise NativePointScoreError("present producer summary maps are incomplete or malformed")
    for support_id in ("full", "guard20"):
        expected_support = dict(supports[support_id])
        if support_doc.get(support_id) != expected_support:
            raise NativePointScoreError(f"producer {support_id} support does not match raw-array derivation")
        expected_metric = dict(metrics[support_id])
        if dict(metric_doc.get(support_id, {})) != expected_metric:
            raise NativePointScoreError(f"producer {support_id} metric does not match raw-array derivation")
        if curve_doc.get(support_id) != curves[support_id]:
            raise NativePointScoreError(f"producer {support_id} lag curve does not match raw-array derivation")


def _score_row_without_producer_summaries(
    record: Mapping[str, Any], arrays: Mapping[str, Any], *,
    artifact_sha256: str, binding_sha256: str, input_pins_sha256: str,
    expected_anchor_column: int,
) -> selector.FitScoreRow:
    if record.get("schema_version") != "cross_tfg_audio_head_stagea_v1" or record.get("status") != "PASS":
        raise NativePointScoreError("native-derived conversion accepts only passing frozen Stage-A rows")
    required = ("row_id", "split", "sample_id", "speaker_id", "model_id", "face_id", "method_id")
    if any(not isinstance(record.get(key), str) or not record[key] for key in required):
        raise NativePointScoreError("native candidate row lacks a required identity field")
    if type(record.get("seed")) is not int or record["seed"] != selector.RENDERER_SEED:
        raise NativePointScoreError("native candidate renderer seed differs from the frozen seed")
    if record.get("split") != "fit":
        raise NativePointScoreError("native candidate row is outside the frozen fit split")
    if type(expected_anchor_column) is not int or expected_anchor_column != selector.ZERO_LAG_COLUMN:
        raise NativePointScoreError("native candidate anchor differs from the plan-bound zero-lag column")
    natural = record.get("natural_pcm_sha256")
    if not selector._is_sha256(natural) or record.get("score_audio_sha256") != natural:
        raise NativePointScoreError("native SyncNet row does not score the exact original natural PCM")
    source_media_kind = record.get("source_media_kind", "video")
    if source_media_kind not in ("video", "static_portrait"):
        raise NativePointScoreError("native source media kind is unsupported")
    raw_score = record.get("score")
    if not isinstance(raw_score, Mapping):
        raise NativePointScoreError("native score row lacks its official score record")
    raw_ref = raw_score.get("raw_matrix_npz")
    if (not isinstance(raw_ref, Mapping) or not isinstance(raw_ref.get("path"), str)
            or not raw_ref.get("path") or not selector._is_sha256(raw_ref.get("file_sha256"))):
        raise NativePointScoreError("native score row lacks an exact raw NPZ file pin")
    arrays_required = set(ARRAY_DTYPES)
    if set(arrays) != arrays_required:
        raise NativePointScoreError("native NPZ array names differ from the frozen seven-array set")
    array_meta = raw_ref.get("arrays")
    if not isinstance(array_meta, Mapping) or set(array_meta) != arrays_required:
        raise NativePointScoreError("native NPZ array-reference metadata differs from the seven-array set")
    matrix = np.asarray(arrays["distance_by_query_lag"])
    if matrix.ndim != 2 or matrix.shape[1] != 31 or matrix.dtype != np.dtype("<f4"):
        raise NativePointScoreError("native raw distance matrix must be finite little-endian float32 [N,31]")
    if not np.isfinite(matrix).all():
        raise NativePointScoreError("native raw distance matrix contains non-finite scores")
    fields: dict[str, np.ndarray] = {}
    for key, dtype in ARRAY_DTYPES.items():
        value = np.asarray(arrays[key])
        if key == "distance_by_query_lag":
            if value.ndim != 2:
                raise NativePointScoreError("native distance matrix rank is invalid")
        elif value.ndim != 1:
            raise NativePointScoreError(f"native NPZ {key} must be rank one")
        if value.dtype != dtype:
            raise NativePointScoreError(f"native NPZ {key} has a noncanonical dtype")
        meta = array_meta.get(key)
        array_sha = selector._sha256(np.ascontiguousarray(value).tobytes(order="C"))
        if (not isinstance(meta, Mapping) or meta.get("dtype") != value.dtype.str
                or meta.get("shape") != list(value.shape)
                or meta.get("array_sha256") != array_sha):
            raise NativePointScoreError(f"native per-array metadata differs from verified {key} bytes")
        fields[key] = value
    count = matrix.shape[0]
    for name in ("query_frame_index", "query_timestamp_num", "query_timestamp_den",
                 "track_frame_index", "valid_query_mask"):
        if len(fields[name]) != count:
            raise NativePointScoreError(f"native query-array length differs for {name}")
    if fields["lag_offsets_frames"].tolist() != list(selector.LAG_OFFSETS):
        raise NativePointScoreError("native raw lag axis differs from the frozen -15..15 contract")
    query_keys = tuple(
        selector._normalize_query_key(int(frame), int(num), int(den), int(track))
        for frame, num, den, track in zip(
            fields["query_frame_index"], fields["query_timestamp_num"],
            fields["query_timestamp_den"], fields["track_frame_index"])
    )
    return selector.FitScoreRow(
        row_id=record["row_id"], split=record["split"], sample_id=record["sample_id"],
        speaker_id=record["speaker_id"], model_id=record["model_id"], face_id=record["face_id"],
        seed=record["seed"], method_id=record["method_id"], status="PASS",
        natural_pcm_sha256=natural, source_video_sha256=record.get("source_video_sha256"),
        source_face_sha256=record.get("source_face_sha256"), artifact_sha256=artifact_sha256,
        binding_sha256=binding_sha256, input_pins_sha256=input_pins_sha256,
        query_keys=query_keys, valid_query_mask=tuple(bool(value) for value in fields["valid_query_mask"]),
        distance_by_query_lag=matrix,
        lag_offsets_frames=tuple(int(value) for value in fields["lag_offsets_frames"]),
        failure_reason=record.get("failure_reason"), anchor_column=expected_anchor_column,
        source_media_kind=source_media_kind,
    )


def native_point_stagea_score_row(
    record: Mapping[str, Any], arrays: Mapping[str, Any], *,
    artifact_sha256: str, binding_sha256: str, input_pins_sha256: str,
    expected_anchor_column: int, native_point_context: Mapping[str, Any],
) -> tuple[selector.FitScoreRow, dict[str, Any]]:
    """Convert one candidate after caller verifies plan, NPZ file, and map bytes."""
    if not isinstance(native_point_context, Mapping) or native_point_context.get("schema") != CONTEXT_SCHEMA:
        raise NativePointScoreError("native support derivation lacks its versioned readback context")
    for key in ("row_id", "point_id", "method_id", "run_id", "point_kind"):
        if not isinstance(native_point_context.get(key), str) or not native_point_context[key]:
            raise NativePointScoreError(f"native point context lacks {key}")
    if native_point_context["point_kind"] not in {"spsa_proposal", "spsa_center"}:
        raise NativePointScoreError("native support derivation is restricted to SPSA point rows")
    if native_point_context["row_id"] != record.get("row_id") or native_point_context["method_id"] != record.get("method_id"):
        raise NativePointScoreError("native support context does not identify the exact score row/method")
    plan_pin = _pin(native_point_context.get("point_plan_pin"), "point plan", self_pin=True)
    npz_pin = _pin(native_point_context.get("raw_matrix_npz_pin"), "raw NPZ")
    map_pin = _pin(native_point_context.get("official_source_frame_map_pin"), "official frame map")
    core_pin = _pin(native_point_context.get("stagea_core_pin"), "frozen Stage-A core")
    verifier_pin = _pin(native_point_context.get("final_consumer_pin"), "final consumer")
    map_check = native_point_context.get("frame_map_validation")
    if (not isinstance(map_check, Mapping) or map_check.get("row_id") != record.get("row_id")
            or map_check.get("frame_map_raw_sha256") != map_pin["sha256"]
            or map_check.get("frame_map_file_id") != native_point_context.get("frame_map_file_id")
            or type(map_check.get("query_count")) is not int
            or type(map_check.get("frame_count")) is not int
            or map_check["query_count"] != len(arrays.get("query_frame_index", ()))
            or map_check["query_count"] > map_check["frame_count"]):
        raise NativePointScoreError("official frame-map verification does not bind this row/map pin")
    if (not core_pin["path"].endswith("/run_stagea_dev_w2l_block_attempt02.py")
            or core_pin["bytes"] != STAGEA_CORE_BYTES or core_pin["sha256"] != STAGEA_CORE_SHA256):
        raise NativePointScoreError("support derivation is not bound to the invoked pinned Stage-A core")
    if raw_ref := (record.get("score", {}).get("raw_matrix_npz")
                   if isinstance(record.get("score"), Mapping) else None):
        if (raw_ref.get("path") != npz_pin["path"]
                or raw_ref.get("file_sha256") != npz_pin["sha256"]
                or raw_ref.get("arrays") != native_point_context.get("raw_matrix_array_refs")):
            raise NativePointScoreError("raw NPZ context differs from the actual official score reference")
    else:
        raise NativePointScoreError("native score row lacks its official raw NPZ reference")
    if (map_pin["path"] != native_point_context.get("official_frame_map_declared_path")
            or map_pin["sha256"] != map_check.get("frame_map_raw_sha256")):
        raise NativePointScoreError("frame map/raw query count does not bind the exact candidate arrays")

    score = record.get("score")
    if not isinstance(score, Mapping):
        raise NativePointScoreError("native candidate row has no score mapping")
    summary_fields = ("lag_curve_by_support", "support_by_support", "metrics_by_support")
    summary_presence = tuple(name in score for name in summary_fields)
    if any(summary_presence) and not all(summary_presence):
        raise NativePointScoreError("native producer summary triplet is partial; refusing fallback")
    has_producer_summaries = all(summary_presence)
    if has_producer_summaries:
        row = selector.official_stagea_score_row(
            record, arrays, artifact_sha256=artifact_sha256,
            binding_sha256=binding_sha256, input_pins_sha256=input_pins_sha256,
            expected_anchor_column=expected_anchor_column,
        )
        derivation_status = "PRODUCER_SUMMARY_VERIFIED_AGAINST_PINNED_ARRAYS"
    else:
        row = _score_row_without_producer_summaries(
            record, arrays, artifact_sha256=artifact_sha256,
            binding_sha256=binding_sha256, input_pins_sha256=input_pins_sha256,
            expected_anchor_column=expected_anchor_column,
        )
        derivation_status = "DERIVED_FROM_VERIFIED_RAW_NPZ_AND_SOURCE_FRAME_MAP"

    if row.status != "PASS":
        raise NativePointScoreError("native support derivation cannot admit a non-PASS score row")
    full = _support_keys(row, "full")
    guard20 = _support_keys(row, "guard20")
    keys_by_support = {"full": full, "guard20": guard20}
    supports: dict[str, dict[str, Any]] = {}
    metrics_by_support: dict[str, dict[str, Any]] = {}
    curves: dict[str, list[float]] = {}
    for support_id, keys in keys_by_support.items():
        curve, metric = _metric_payload(row, keys, anchor_column=expected_anchor_column)
        curves[support_id] = curve
        supports[support_id] = {
            "query_count": len(keys),
            "support_mask_sha256": _support_sha(support_id, keys),
            "eligible": len(keys) >= (1 if support_id == "full" else 41),
        }
        metrics_by_support[support_id] = metric
    if has_producer_summaries:
        _validate_present_summaries(score, supports, metrics_by_support, curves)

    converter_raw = Path(__file__).resolve(strict=True).read_bytes()
    derivation = {
        "schema": DERIVATION_SCHEMA,
        "status": derivation_status,
        "row_id": row.row_id,
        "point_id": native_point_context["point_id"],
        "point_kind": native_point_context["point_kind"],
        "method_id": row.method_id,
        "model_id": row.model_id,
        "run_id": native_point_context["run_id"],
        "source_pins": {
            "point_plan": plan_pin,
            "stagea_core": core_pin,
            "raw_matrix_npz": {
                **npz_pin,
                "array_refs": native_point_context["raw_matrix_array_refs"],
            },
            "official_source_frame_map": map_pin,
            "final_consumer": verifier_pin,
            "native_point_score_adapter": {
                "path": str(Path(__file__).resolve(strict=True)),
                "bytes": len(converter_raw), "sha256": _sha(converter_raw),
            },
        },
        "frame_map_validation": dict(map_check),
        "source_clock": {
            "query_count": len(row.query_keys),
            "valid_query_count": int(np.count_nonzero(np.asarray(row.valid_query_mask, dtype=bool))),
            "query_clock_keys_sha256": selector._sha256(selector._canonical(
                [list(key) for key in row.query_keys])),
            "valid_query_mask_array_sha256": native_point_context["raw_matrix_array_refs"]["valid_query_mask"]["array_sha256"],
        },
        "supports": supports,
        "metrics_by_support": metrics_by_support,
        "lag_curve_by_support": curves,
        "anchor": {
            "source": "point_plan.protocol.syncnet.anchor_column",
            "column": expected_anchor_column,
            "zero_lag_column": selector.ZERO_LAG_COLUMN,
            "lag_offsets_frames": list(row.lag_offsets_frames),
        },
        "guard20_rule": {
            "source": "run_stagea_dev_w2l_block_attempt02._finish_supports",
            "slice": "full_keys[20:-20] when full_count >= 41; otherwise empty",
            "eligibility_minimum_after_trim": 41,
        },
    }
    return row, derivation
