#!/usr/bin/env python3
"""Independent NumPy-only paired audit of frozen native encoder feature exports.

For each backend, conditions are compared with that backend's own natural feature
sequence on its official query clock. No cross-backend embedding similarity or
mouth-motion/synchronization claim is computed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

CONDITIONS = (
    "natural",
    "global_minus_6db",
    "global_plus_6db",
    "fixed_rms_0p0376838173",
    "local_r090_accent015_tilt_p025",
    "local_r090_flatten015_tilt_m025",
    "local_r090_accent_0.30_tilt0",
)
METRICS = (
    "raw_frame_cosine_mean",
    "raw_frame_cosine_min",
    "feature_norm_ratio_mean",
    "feature_norm_ratio_median",
    "query_time_centered_frame_cosine_mean",
    "query_time_centered_global_cosine",
    "query_time_centered_relative_rms_change",
    "transition_delta_cosine_mean",
    "transition_delta_norm_ratio_mean",
    "transition_delta_query_time_centered_cosine",
    "transition_delta_relative_rms_change",
)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.einsum("ij,ij->i", a, b, optimize=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-20)


def cosine_global(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / den) if den > 1e-20 else float("nan")


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def metrics(natural_raw: np.ndarray, condition_raw: np.ndarray) -> dict[str, float | int]:
    n = np.asarray(natural_raw).reshape(natural_raw.shape[0], -1).astype(np.float64)
    a = np.asarray(condition_raw).reshape(condition_raw.shape[0], -1).astype(np.float64)
    if n.shape != a.shape or n.ndim != 2 or n.shape[0] < 3:
        raise ValueError(f"paired feature shapes differ or support is too short: {n.shape} vs {a.shape}")
    if not np.isfinite(n).all() or not np.isfinite(a).all():
        raise ValueError("feature array contains nonfinite values")
    nn = np.linalg.norm(n, axis=1)
    an = np.linalg.norm(a, axis=1)
    ratio = an / np.maximum(nn, 1e-20)
    nc = n - n.mean(axis=0, keepdims=True)
    ac = a - a.mean(axis=0, keepdims=True)
    nd, ad = np.diff(n, axis=0), np.diff(a, axis=0)
    ndn, adn = np.linalg.norm(nd, axis=1), np.linalg.norm(ad, axis=1)
    ndc = nd - nd.mean(axis=0, keepdims=True)
    adc = ad - ad.mean(axis=0, keepdims=True)
    rawcos = cosine_rows(n, a)
    centeredcos = cosine_rows(nc, ac)
    delcos = cosine_rows(nd, ad)
    dencos = cosine_global(ndc, adc)
    den_global = cosine_global(nc, ac)
    return {
        "query_count": int(n.shape[0]),
        "feature_dimension_after_flatten": int(n.shape[1]),
        "raw_frame_cosine_mean": finite_mean(rawcos),
        "raw_frame_cosine_min": finite_or_none(float(np.nanmin(rawcos))) if np.isfinite(rawcos).any() else None,
        "feature_norm_ratio_mean": float(np.mean(ratio)),
        "feature_norm_ratio_median": float(np.median(ratio)),
        "query_time_centered_frame_cosine_mean": finite_mean(centeredcos),
        "query_time_centered_global_cosine": finite_or_none(den_global),
        "query_time_centered_relative_rms_change": float(np.linalg.norm(ac - nc) / max(float(np.linalg.norm(nc)), 1e-20)),
        "transition_delta_cosine_mean": finite_mean(delcos),
        "transition_delta_norm_ratio_mean": float(np.mean(adn / np.maximum(ndn, 1e-20))),
        "transition_delta_query_time_centered_cosine": finite_or_none(dencos),
        "transition_delta_relative_rms_change": float(np.linalg.norm(ad - nd) / max(float(np.linalg.norm(nd)), 1e-20)),
        "raw_feature_mean_offset_ratio": float(np.linalg.norm(a.mean(axis=0) - n.mean(axis=0)) / max(float(np.linalg.norm(n.mean(axis=0))), 1e-20)),
        "transition_delta_uncentered_global_cosine": finite_or_none(cosine_global(nd, ad)),
    }


def load_npz(path: Path, expected_fps: int) -> tuple[np.ndarray, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"feature path is not regular: {path}")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"features", "query_index", "query_timestamp_num", "query_timestamp_den"}:
            raise ValueError(f"unexpected NPZ keys: {path}")
        features = np.asarray(z["features"])
        idx = np.asarray(z["query_index"])
        num = np.asarray(z["query_timestamp_num"])
        den = np.asarray(z["query_timestamp_den"])
    if features.ndim < 2 or features.shape[0] < 3 or not np.issubdtype(features.dtype, np.number):
        raise ValueError(f"invalid feature shape/type: {path} {features.shape} {features.dtype}")
    if idx.shape != (features.shape[0],) or num.shape != idx.shape or den.shape != idx.shape:
        raise ValueError(f"query arrays do not match features: {path}")
    target = np.arange(features.shape[0], dtype=idx.dtype)
    if not np.array_equal(idx, target) or not np.array_equal(num, target) or not np.all(den == expected_fps):
        raise ValueError(f"query clock is not exactly q/25: {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"feature matrix has NaN/Inf: {path}")
    return features, np.stack((idx.astype(np.int64), num.astype(np.int64), den.astype(np.int64)), axis=1)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups = sorted({(r["backend"], r["anchor"]) for r in rows})
    for backend, anchor in groups:
        backend_rows = [r for r in rows if r["backend"] == backend and r["anchor"] == anchor]
        if len(backend_rows) != 8 * len(CONDITIONS):
            raise ValueError(f"{backend}/{anchor}: expected 8×7 rows, got {len(backend_rows)}")
        for condition in CONDITIONS:
            selected = [r for r in backend_rows if r["condition"] == condition]
            if len(selected) != 8 or len({r["sample_id"] for r in selected}) != 8:
                raise ValueError(f"{backend}/{anchor}/{condition}: expected eight unique clip rows")
            out: dict[str, Any] = {"backend": backend, "anchor": anchor, "condition": condition, "n_clips": len(selected)}
            for metric in METRICS + ("raw_feature_mean_offset_ratio", "transition_delta_uncentered_global_cosine"):
                raw_values = [r[metric] for r in selected]
                v = np.asarray([np.nan if value is None else float(value) for value in raw_values], dtype=np.float64)
                valid = v[np.isfinite(v)]
                out[f"{metric}_valid_clips"] = int(valid.size)
                out[f"{metric}_missing_clips"] = int(len(v) - valid.size)
                out[f"{metric}_equal_clip_mean"] = float(np.mean(valid)) if valid.size else None
                out[f"{metric}_clip_min"] = float(np.min(valid)) if valid.size else None
                out[f"{metric}_clip_max"] = float(np.max(valid)) if valid.size else None
            result.append(out)
    return result


def raw_pin(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {path}")
    digest = sha_file(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def verify_pin_record(path: Path, record: dict[str, Any], label: str) -> dict[str, Any]:
    pin = raw_pin(path)
    if (pin["bytes"], pin["sha256"]) != (record.get("bytes"), record.get("sha256")):
        raise ValueError(f"{label}: local bytes/SHA differ from archived record")
    return pin


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-plan", type=Path, required=True)
    ap.add_argument("--frozen-fit-plan", type=Path, required=True)
    ap.add_argument("--wav2lip-root", type=Path, required=True, help="restored W/M archive frontends/wav2lip")
    ap.add_argument("--muse-root", type=Path, required=True, help="restored W/M archive frontends/muse")
    ap.add_argument("--ditto-root", type=Path, required=True, help="strictly restored Ditto output root")
    ap.add_argument("--wm-root-qc", type=Path, required=True, help="root's independent W/M whole-archive/restore QC")
    ap.add_argument("--ditto-collection-receipt", type=Path, required=True, help="local strict collector receipt")
    ap.add_argument("--ditto-stream-receipt", type=Path, required=True, help="direct-stream receipt binding collector and raw tar")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        raise FileExistsError(args.out_dir)
    plan = json.loads(args.sample_plan.read_text(encoding="utf-8"))
    frozen_plan = json.loads(args.frozen_fit_plan.read_text(encoding="utf-8"))
    wm_qc = json.loads(args.wm_root_qc.read_text(encoding="utf-8"))
    ditto_collection = json.loads(args.ditto_collection_receipt.read_text(encoding="utf-8"))
    stream_receipt = json.loads(args.ditto_stream_receipt.read_text(encoding="utf-8"))
    frozen_pin = raw_pin(args.frozen_fit_plan)
    if plan.get("source_plan", {}).get("sha256") != frozen_pin["sha256"] or plan.get("source_plan", {}).get("bytes") != frozen_pin["bytes"]:
        raise ValueError("probe roster does not bind the exact frozen historical fit plan")
    if plan.get("source_plan", {}).get("run_id") != frozen_plan.get("run_id"):
        raise ValueError("probe roster and frozen plan run IDs differ")

    wm_archive = wm_qc.get("archive", {})
    if wm_qc.get("schema") != "cross_tfg_root_native_wm_archive_and_restore_qc_v1" or wm_qc.get("status") != "PASS_WHOLE_ARCHIVE_AND_ALL_153_RESTORED_FILE_SHA256" or wm_qc.get("completed_encoder_units") != 112 or wm_qc.get("planned_encoder_units") != 168:
        raise ValueError("Wav2Lip/Muse archive lacks its independent whole-archive/file restore PASS")
    if Path(wm_qc.get("restored_path", "")).resolve() != args.wav2lip_root.resolve().parent.parent or args.muse_root.resolve().parent != args.wav2lip_root.resolve().parent:
        raise ValueError("Wav2Lip/Muse feature roots do not match the independently QC'd restoration")
    wm_archive_path = Path(wm_archive.get("path", ""))
    wm_archive_pin = raw_pin(wm_archive_path)
    if (wm_archive_pin["sha256"], wm_archive_pin["bytes"]) != (wm_archive.get("sha256"), wm_archive.get("bytes")):
        raise ValueError("Wav2Lip/Muse archive on disk differs from independent root QC")
    strict_record = wm_qc.get("strict_report", {})
    strict_path = Path(strict_record.get("path", ""))
    strict_pin = verify_pin_record(strict_path, strict_record, "W/M strict restore report")
    strict_report = json.loads(strict_path.read_text(encoding="utf-8"))
    if strict_report.get("schema") != "cross_tfg_native_wm_archive_local_readback_qc_v1" or strict_report.get("status") != "PASS_LOCAL_STRICT_WM_ARCHIVE_READBACK_INCOMPLETE_RUN_PRESERVED" or strict_report.get("archive") != wm_archive:
        raise ValueError("W/M strict report does not bind the root-QC'd archive")
    wm_member_map = {r.get("path"): r for r in strict_report.get("member_checks", [])}
    if len(wm_member_map) != 153 or len(strict_report.get("member_checks", [])) != 153:
        raise ValueError("W/M strict report does not contain 153 unique raw members")

    if ditto_collection.get("schema") != "cross_tfg_native_ditto_feature_collection_receipt_v1" or ditto_collection.get("status") != "PASS" or ditto_collection.get("remote_run_id") != "native_encoder_ditto_retry_execution_attempt04":
        raise ValueError("Ditto collection receipt is not the expected actual terminal collection")
    if ditto_collection.get("file_count") != 58 or len(ditto_collection.get("restored_files", [])) != 58:
        raise ValueError("Ditto strict restore receipt does not contain all 58 actual output files")
    collection_pin = raw_pin(args.ditto_collection_receipt)
    archive_record = ditto_collection.get("archive", {})
    archive_path = Path(archive_record.get("path", ""))
    ditto_archive_pin = raw_pin(archive_path)
    if (ditto_archive_pin["bytes"], ditto_archive_pin["sha256"]) != (archive_record.get("bytes"), archive_record.get("sha256")):
        raise ValueError("Ditto raw stream archive differs from its strict collection receipt")
    if stream_receipt.get("schema") != "cross_tfg_native_ditto_direct_stream_receipt_v1" or stream_receipt.get("status") != "DIRECT_STREAM_AND_STRICT_RESTORE_PASS":
        raise ValueError("Ditto direct-stream receipt is not a completed strict restore")
    stream_body = dict(stream_receipt)
    stream_self_hash = stream_body.pop("receipt_sha256", None)
    stream_self_hash_actual = hashlib.sha256(json.dumps(stream_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    if stream_self_hash != stream_self_hash_actual:
        raise ValueError("Ditto direct-stream receipt self-hash mismatch")
    stream_pin = raw_pin(args.ditto_stream_receipt)
    if stream_receipt.get("remote_archive_copy_created") is not False or stream_receipt.get("source_mutation_performed") is not False or stream_receipt.get("remote_stderr_bytes") != 0:
        raise ValueError("Ditto direct stream does not satisfy the read-only clean-stream contract")
    if (stream_receipt.get("stream_bytes"), stream_receipt.get("stream_sha256")) != (archive_record.get("bytes"), archive_record.get("sha256")):
        raise ValueError("Ditto streamed tar differs from the strict collector archive record")
    if stream_receipt.get("collection_receipt", {}).get("sha256") != collection_pin["sha256"] or stream_receipt.get("collection_receipt", {}).get("bytes") != collection_pin["bytes"]:
        raise ValueError("Ditto direct-stream receipt does not bind the local collection receipt")
    ditto_member_map = {r.get("path"): r for r in ditto_collection.get("restored_files", [])}
    if len(ditto_member_map) != 58:
        raise ValueError("Ditto local receipt does not contain 58 unique restored members")
    ditto_summary_path = args.ditto_root / "summary.json"
    ditto_summary_pin = raw_pin(ditto_summary_path)
    if ditto_summary_pin["sha256"] != ditto_collection.get("terminal_summary_sha256"):
        raise ValueError("Ditto restored summary raw SHA differs from terminal collection receipt")
    ditto_summary_member = ditto_member_map.get("summary.json")
    if not ditto_summary_member or (ditto_summary_member.get("bytes"), ditto_summary_member.get("sha256")) != (ditto_summary_pin["bytes"], ditto_summary_pin["sha256"]):
        raise ValueError("Ditto summary raw bytes are not bound by the local strict restore receipt")

    input_rows = plan.get("input_rows", [])
    if len(input_rows) != 8 or len({r.get("sample_id") for r in input_rows}) != 8:
        raise ValueError("sample plan is not the exact eight-clip frozen roster")
    sample_by_id = {r["sample_id"]: r for r in input_rows}
    if len(sample_by_id) != 8:
        raise ValueError("duplicate source clip IDs")
    rows: list[dict[str, Any]] = []
    producer_numeric_check_count = 0
    producer_numeric_max_abs_diff = 0.0
    producer_numeric_max_location: dict[str, Any] | None = None
    producer_numeric_mismatch_count = 0
    producer_numeric_tolerance = 1e-9
    source_pins: dict[str, Any] = {
        "sample_plan": raw_pin(args.sample_plan),
        "frozen_fit_plan": frozen_pin,
        "wav2lip_muse_root_qc": {**raw_pin(args.wm_root_qc), "archive": wm_archive, "strict_report": strict_pin},
        "wav2lip_muse_archive": wm_archive_pin,
        "ditto_collection_receipt": collection_pin,
        "ditto_stream_receipt": stream_pin,
        "ditto_archive": ditto_archive_pin,
        "ditto_terminal_summary_sha256": ditto_collection.get("terminal_summary_sha256"),
    }
    backends = {
        "wav2lip": (args.wav2lip_root, "wav2lip", "wav2lip_gan_bac9a81e", wm_member_map, None),
        "muse": (args.muse_root, "muse", "musetalk_fixed_portrait_attempt16", wm_member_map, None),
        "ditto": (args.ditto_root, "ditto", "ditto_hubert_fixed_portrait_attempt18", ditto_member_map, ditto_collection),
    }
    root_feature_maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for record in strict_report.get("feature_checks", []):
        backend_name = record.get("frontend")
        if backend_name in ("wav2lip", "muse"):
            root_feature_maps[backend_name] = {(r.get("sample_id"), r.get("condition")): r for r in record.get("rows", [])}
    if set(root_feature_maps) != {"wav2lip", "muse"} or any(len(v) != 56 for v in root_feature_maps.values()):
        raise ValueError("W/M independent feature archive QC lacks both exact 56-row digests")

    for backend, (base, expected_frontend, expected_model, member_map, backend_collection) in backends.items():
        feature_dir = base / "features"
        if base.is_symlink() or feature_dir.is_symlink() or not feature_dir.is_dir():
            raise FileNotFoundError(feature_dir)
        files = sorted(feature_dir.glob("*.npz"))
        expected_names = {f"{sid}__{c}.npz" for sid in sample_by_id for c in CONDITIONS}
        if {p.name for p in files} != expected_names or len(files) != 56:
            raise ValueError(f"{backend}: feature denominator mismatch; got {len(files)} files")
        summary_path = base / "summary.json"
        summary_pin = raw_pin(summary_path)
        if backend in ("wav2lip", "muse"):
            rel_summary = f"frontends/{backend}/summary.json"
            record = wm_member_map.get(rel_summary)
            if not record or (record.get("bytes"), record.get("sha256")) != (summary_pin["bytes"], summary_pin["sha256"]):
                raise ValueError(f"{backend}: summary raw bytes are not bound by the W/M archive member ledger")
        else:
            summary_member = member_map.get("summary.json")
            if not summary_member or (summary_member.get("bytes"), summary_member.get("sha256")) != (summary_pin["bytes"], summary_pin["sha256"]):
                raise ValueError("Ditto summary raw bytes do not match its collection member ledger")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "cross_tfg_native_audio_encoder_probe_v1" or summary.get("status") != "COMPLETE" or summary.get("frontend") != expected_frontend or summary.get("model_id") != expected_model:
            raise ValueError(f"{backend}: restored producer summary identity/status/model mismatch")
        if summary.get("row_count") != 56 or summary.get("plan", {}).get("sha256") != frozen_pin["sha256"] or summary.get("plan", {}).get("bytes") != frozen_pin["bytes"]:
            raise ValueError(f"{backend}: producer summary plan hash or row count differs from the frozen fit plan")
        scope = summary.get("scope", {})
        if scope.get("split") != "historical_fit_only" or scope.get("development_audio_opened") is not False or scope.get("confirmation_audio_opened") is not False or scope.get("completed_source_condition_units") != 56 or scope.get("planned_source_condition_units") != 56:
            raise ValueError(f"{backend}: producer summary scope/denominator mismatch")
        summary_list = summary.get("rows", [])
        summary_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for record in summary_list:
            key = (record.get("sample_id"), record.get("condition"))
            if key in summary_rows:
                raise ValueError(f"{backend}: duplicate summary row {key}")
            summary_rows[key] = record
        expected_keys = {(sid, c) for sid in sample_by_id for c in CONDITIONS}
        if len(summary_rows) != 56 or set(summary_rows) != expected_keys:
            raise ValueError(f"{backend}: summary does not contain exact 8×7 rows")
        source_pins[backend] = {
            "summary": summary_pin,
            "archive_member_summary": {"path": f"frontends/{backend}/summary.json" if backend != "ditto" else "summary.json", "bytes": summary_pin["bytes"], "sha256": summary_pin["sha256"]},
            "feature_count": len(files),
            "model_id": summary.get("model_id"),
            "runtime": summary.get("runtime"),
        }

        arrays: dict[tuple[str, str], np.ndarray] = {}
        clocks: dict[tuple[str, str], np.ndarray] = {}
        feature_pins: dict[tuple[str, str], dict[str, Any]] = {}
        for sid in sample_by_id:
            source_item = sample_by_id[sid]
            source_pcm = source_item["natural_pcm"]
            for condition in CONDITIONS:
                key = (sid, condition)
                feature_path = feature_dir / f"{sid}__{condition}.npz"
                if feature_path.is_symlink():
                    raise ValueError(f"feature array is a symlink: {feature_path}")
                file_pin = raw_pin(feature_path)
                summary_row = summary_rows[key]
                if summary_row.get("status") != "PASS" or summary_row.get("speaker_id") != source_item["speaker_id"]:
                    raise ValueError(f"{backend}/{sid}/{condition}: source summary row did not PASS or speaker differs")
                sr_source = summary_row.get("source_pcm", {})
                if sr_source.get("pcm_float32_sha256") != source_pcm.get("pcm_sha256") or sr_source.get("wav_sha256") != source_pcm.get("file_sha256") or sr_source.get("sample_count") != source_pcm.get("sample_count") or sr_source.get("sample_rate_hz") != 16000:
                    raise ValueError(f"{backend}/{sid}/{condition}: original PCM/file/sample clock binding mismatch")
                array_meta = summary_row.get("feature_array", {})
                if array_meta.get("path", "").split("/")[-1].split("\\")[-1] != feature_path.name:
                    raise ValueError(f"{backend}/{sid}/{condition}: summary feature path points at another file")
                if (array_meta.get("bytes"), array_meta.get("sha256")) != (file_pin["bytes"], file_pin["sha256"]):
                    raise ValueError(f"{backend}/{sid}/{condition}: actual feature file differs from summary.feature_array pin")
                rel_member = f"frontends/{backend}/features/{feature_path.name}" if backend != "ditto" else f"features/{feature_path.name}"
                member = member_map.get(rel_member)
                if not member or (member.get("bytes"), member.get("sha256")) != (file_pin["bytes"], file_pin["sha256"]):
                    raise ValueError(f"{backend}/{sid}/{condition}: raw feature bytes differ from its verified archive member")
                if backend in ("wav2lip", "muse"):
                    root_feature = root_feature_maps[backend].get(key)
                    if not root_feature:
                        raise ValueError(f"{backend}/{sid}/{condition}: root independent per-array QC row is absent")
                features, clock = load_npz(feature_path, 25)
                if list(features.shape) != array_meta.get("shape") or str(features.dtype) != array_meta.get("dtype"):
                    raise ValueError(f"{backend}/{sid}/{condition}: actual NPZ array shape/dtype differs from its summary metadata")
                if backend in ("wav2lip", "muse"):
                    # The root archive QC's features_sha256 binds the uncompressed
                    # contiguous ndarray payload, not the enclosing compressed NPZ.
                    feature_payload = np.ascontiguousarray(features).tobytes(order="C")
                    feature_payload_sha256 = hashlib.sha256(feature_payload).hexdigest()
                    if (root_feature.get("features_sha256"), root_feature.get("features_shape"), root_feature.get("features_dtype")) != (feature_payload_sha256, list(features.shape), str(features.dtype)):
                        raise ValueError(f"{backend}/{sid}/{condition}: root independent ndarray digest/shape/dtype differs")
                if backend == "ditto":
                    expected_queries = (source_pcm["sample_count"] + 639) // 640
                    if features.shape != (expected_queries, 1024):
                        raise ValueError(f"Ditto support/dimension mismatch for {sid}/{condition}: {features.shape}")
                clock_map = summary_row.get("clock_map", {})
                if clock_map.get("fps_num") != 25 or clock_map.get("fps_den") != 1 or clock_map.get("query_count") != features.shape[0] or len(clock_map.get("queries", [])) != features.shape[0]:
                    raise ValueError(f"{backend}/{sid}/{condition}: summary query clock descriptor shape/FPS mismatch")
                for q, query in enumerate(clock_map["queries"]):
                    stop = min(source_pcm["sample_count"], (q + 1) * 640)
                    if (query.get("query_index"), query.get("query_timestamp_num"), query.get("query_timestamp_den"), query.get("source_clock_frame_index"), query.get("source_sample_start"), query.get("source_sample_stop_exclusive")) != (q, q, 25, q, q * 640, stop):
                        raise ValueError(f"{backend}/{sid}/{condition}: source/query clock descriptor mismatch at query {q}")
                arrays[key] = features
                clocks[key] = clock
                feature_pins[key] = file_pin

        for sid in sample_by_id:
            natural_clock = clocks[(sid, "natural")]
            rms_clock = clocks[(sid, "fixed_rms_0p0376838173")]
            if not np.array_equal(natural_clock, rms_clock):
                raise ValueError(f"{backend}/{sid}: natural and fixed-RMS query clock arrays differ")
            for condition in CONDITIONS:
                if not np.array_equal(clocks[(sid, condition)], natural_clock):
                    raise ValueError(f"{backend}/{sid}/{condition}: query clock changed across conditions")
                summary_row = summary_rows[(sid, condition)]
                for anchor in ("natural", "fixed_rms_0p0376838173"):
                    m = metrics(arrays[(sid, anchor)], arrays[(sid, condition)])
                    m.update({
                        "backend": backend,
                        "sample_id": sid,
                        "speaker_id": sample_by_id[sid]["speaker_id"],
                        "anchor": anchor,
                        "condition": condition,
                        "query_count": int(arrays[(sid, condition)].shape[0]),
                        "feature_shape": list(arrays[(sid, condition)].shape),
                        "feature_dtype": str(arrays[(sid, condition)].dtype),
                        "query_clock": "query_index/25 seconds; source samples [640q,min(N,640(q+1)))",
                        "anchor_feature_sha256": feature_pins[(sid, anchor)]["sha256"],
                        "condition_feature_sha256": feature_pins[(sid, condition)]["sha256"],
                    })
                    if anchor == "natural":
                        producer_metrics = summary_row.get("metrics", {})
                        for metric_name in METRICS:
                            expected = producer_metrics.get(metric_name)
                            if expected is None or not np.isfinite(float(expected)):
                                raise ValueError(f"{backend}/{sid}/{condition}: producer omitted/nonfinite required metric {metric_name}")
                            diff = abs(float(expected) - float(m[metric_name]))
                            producer_numeric_check_count += 1
                            if diff > producer_numeric_max_abs_diff:
                                producer_numeric_max_abs_diff = diff
                                producer_numeric_max_location = {"backend": backend, "sample_id": sid, "condition": condition, "metric": metric_name, "producer_value": float(expected), "independent_value": float(m[metric_name]), "absolute_difference": diff}
                            if diff > producer_numeric_tolerance:
                                producer_numeric_mismatch_count += 1
                    rows.append(m)

    expected_rows = 3 * 8 * 7 * 2
    if producer_numeric_check_count != 3 * 8 * 7 * len(METRICS) or len(rows) != expected_rows:
        raise ValueError(f"numeric denominator incomplete: checks={producer_numeric_check_count}, rows={len(rows)}, expected_rows={expected_rows}")
    aggregates = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "clip_condition_metrics.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "equal_clip_condition_summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)
    report = {
        "schema": "cross_tfg_native_encoder_condition_numeric_audit_v2",
        "status": "PASS" if producer_numeric_mismatch_count == 0 else "COMPLETE_WITH_NUMERIC_MISMATCHES",
        "scope": "historical_fit_only_8_clips_4_speakers; 3 frontend-native feature spaces; each of 7 conditions compared separately with its own natural and fixed-RMS anchors",
        "denominator": {"backends": 3, "clips_per_backend": 8, "conditions": 7, "anchors": ["natural", "fixed_rms_0p0376838173"], "paired_rows": len(rows), "expected_rows": expected_rows, "producer_metric_checks": producer_numeric_check_count},
        "source_pins": source_pins,
        "metrics": list(METRICS) + ["raw_feature_mean_offset_ratio", "transition_delta_uncentered_global_cosine"],
        "interpretation_limit": "Within a frontend and clip, features are paired by the exact source query index/timestamp. Natural-anchored and fixed-RMS-anchored metrics are separate and must not be differenced across anchor groups. These are encoder representation changes; they do not measure generator output, lip motion, SyncNet, phoneme alignment, or synchronization benefit. Frontend feature spaces are not compared directly.",
        "producer_summary_numeric_qc": {"checks": producer_numeric_check_count, "absolute_tolerance": producer_numeric_tolerance, "mismatches": producer_numeric_mismatch_count, "max_absolute_difference": producer_numeric_max_abs_diff, "max_difference_location": producer_numeric_max_location, "envelope_correlation_fields_recomputed": False},
        "clip_condition_metrics_path": "clip_condition_metrics.csv",
        "equal_clip_condition_summary_path": "equal_clip_condition_summary.csv",
        "rows": rows,
        "equal_clip_summary": aggregates,
    }
    with (args.out_dir / "numeric_audit.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
