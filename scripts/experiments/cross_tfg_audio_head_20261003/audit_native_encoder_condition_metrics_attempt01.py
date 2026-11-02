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
        "raw_frame_cosine_mean": float(np.nanmean(rawcos)),
        "raw_frame_cosine_min": float(np.nanmin(rawcos)),
        "feature_norm_ratio_mean": float(np.mean(ratio)),
        "feature_norm_ratio_median": float(np.median(ratio)),
        "query_time_centered_frame_cosine_mean": float(np.nanmean(centeredcos)),
        "query_time_centered_global_cosine": den_global,
        "query_time_centered_relative_rms_change": float(np.linalg.norm(ac - nc) / max(float(np.linalg.norm(nc)), 1e-20)),
        "transition_delta_cosine_mean": float(np.nanmean(delcos)),
        "transition_delta_norm_ratio_mean": float(np.mean(adn / np.maximum(ndn, 1e-20))),
        "transition_delta_query_time_centered_cosine": dencos,
        "transition_delta_relative_rms_change": float(np.linalg.norm(ad - nd) / max(float(np.linalg.norm(nd)), 1e-20)),
        "raw_feature_mean_offset_ratio": float(np.linalg.norm(a.mean(axis=0) - n.mean(axis=0)) / max(float(np.linalg.norm(n.mean(axis=0))), 1e-20)),
        "raw_feature_temporal_direction_cosine": den_global,
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
    backends = sorted({r["backend"] for r in rows})
    for backend in backends:
        backend_rows = [r for r in rows if r["backend"] == backend]
        if len(backend_rows) != 8 * len(CONDITIONS):
            raise ValueError(f"{backend}: expected 8×7 rows, got {len(backend_rows)}")
        for condition in CONDITIONS:
            selected = [r for r in backend_rows if r["condition"] == condition]
            if len(selected) != 8 or len({r["sample_id"] for r in selected}) != 8:
                raise ValueError(f"{backend}/{condition}: expected eight unique clip rows")
            out: dict[str, Any] = {"backend": backend, "condition": condition, "n_clips": len(selected)}
            for metric in METRICS + ("raw_feature_mean_offset_ratio", "raw_feature_temporal_direction_cosine"):
                v = np.asarray([r[metric] for r in selected], dtype=np.float64)
                out[f"{metric}_equal_clip_mean"] = float(np.mean(v))
                out[f"{metric}_clip_min"] = float(np.min(v))
                out[f"{metric}_clip_max"] = float(np.max(v))
            result.append(out)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-plan", type=Path, required=True)
    ap.add_argument("--wav2lip-root", type=Path, required=True, help="restored W/M archive frontends/wav2lip")
    ap.add_argument("--muse-root", type=Path, required=True, help="restored W/M archive frontends/muse")
    ap.add_argument("--ditto-root", type=Path, required=True, help="strictly restored Ditto output root")
    ap.add_argument("--wm-root-qc", type=Path, required=True, help="root's independent W/M whole-archive/restore QC")
    ap.add_argument("--ditto-collection-receipt", type=Path, required=True, help="local strict collector receipt")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        raise FileExistsError(args.out_dir)
    plan = json.loads(args.sample_plan.read_text(encoding="utf-8"))
    wm_qc = json.loads(args.wm_root_qc.read_text(encoding="utf-8"))
    ditto_collection = json.loads(args.ditto_collection_receipt.read_text(encoding="utf-8"))
    wm_archive = wm_qc.get("archive", {})
    if wm_qc.get("status") != "PASS_WHOLE_ARCHIVE_AND_ALL_153_RESTORED_FILE_SHA256" or wm_qc.get("completed_encoder_units") != 112 or wm_qc.get("planned_encoder_units") != 168:
        raise ValueError("Wav2Lip/Muse archive lacks its independent whole-archive/file restore PASS")
    if Path(wm_qc.get("restored_path", "")).resolve() != args.wav2lip_root.resolve().parent.parent:
        raise ValueError("Wav2Lip/Muse feature roots do not match independently QC'd restoration")
    if (wm_archive.get("sha256"), wm_archive.get("bytes")) != ("ad4824f6a4b8027fb7c2c4bb3b1d5112c5c980920671523aedfa09ae0c433d48", 50204055):
        raise ValueError("Wav2Lip/Muse archive pin differs from the root-QC'd archive")
    if ditto_collection.get("schema") != "cross_tfg_native_ditto_feature_collection_receipt_v1" or ditto_collection.get("status") != "PASS" or ditto_collection.get("remote_run_id") != "native_encoder_ditto_retry_execution_attempt04":
        raise ValueError("Ditto collection receipt is not the expected actual terminal collection")
    if ditto_collection.get("file_count") != 58 or len(ditto_collection.get("restored_files", [])) != 58:
        raise ValueError("Ditto strict restore receipt does not contain all 58 actual output files")
    ditto_archive = ditto_collection.get("archive", {})
    archive_path = Path(ditto_archive.get("path", ""))
    if not archive_path.is_file() or archive_path.stat().st_size != ditto_archive.get("bytes") or sha_file(archive_path) != ditto_archive.get("sha256"):
        raise ValueError("Ditto raw stream archive no longer matches its collection receipt")
    input_rows = plan.get("input_rows", [])
    if len(input_rows) != 8 or len({r.get("sample_id") for r in input_rows}) != 8:
        raise ValueError("sample plan is not the exact eight-clip frozen roster")
    sample_by_id = {r["sample_id"]: r for r in input_rows}
    rows: list[dict[str, Any]] = []
    producer_numeric_check_count = 0
    source_pins: dict[str, Any] = {
        "sample_plan": {"path": str(args.sample_plan), "bytes": args.sample_plan.stat().st_size, "sha256": sha_file(args.sample_plan)},
        "wav2lip_muse_archive_qc": {"path": str(args.wm_root_qc), "bytes": args.wm_root_qc.stat().st_size, "sha256": sha_file(args.wm_root_qc), "archive": wm_archive},
        "ditto_collection_receipt": {"path": str(args.ditto_collection_receipt), "bytes": args.ditto_collection_receipt.stat().st_size, "sha256": sha_file(args.ditto_collection_receipt), "archive": ditto_archive},
    }
    backends = {
        "wav2lip": (args.wav2lip_root, 0),
        "muse": (args.muse_root, 0),
        "ditto": (args.ditto_root, 0),
    }
    for backend, (base, _) in backends.items():
        feature_dir = base / "features"
        if not feature_dir.is_dir():
            raise FileNotFoundError(feature_dir)
        files = sorted(feature_dir.glob("*.npz"))
        expected_names = {f"{sid}__{c}.npz" for sid in sample_by_id for c in CONDITIONS}
        if {p.name for p in files} != expected_names or len(files) != 56:
            raise ValueError(f"{backend}: feature denominator mismatch; got {len(files)} files")
        summary_path = base / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_frontend = {"wav2lip": "wav2lip", "muse": "muse", "ditto": "ditto"}[backend]
        if summary.get("schema") != "cross_tfg_native_audio_encoder_probe_v1" or summary.get("status") != "COMPLETE" or summary.get("frontend") != expected_frontend:
            raise ValueError(f"{backend}: restored producer summary identity/status mismatch")
        summary_rows = {(r.get("sample_id"), r.get("condition")): r for r in summary.get("rows", [])}
        if len(summary_rows) != 56 or set(summary_rows) != {(sid, c) for sid in sample_by_id for c in CONDITIONS}:
            raise ValueError(f"{backend}: summary does not contain exact 8×7 rows")
        source_pins[backend] = {
            "summary": {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": sha_file(summary_path)},
            "feature_count": len(files),
        }
        natural: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        arrays: dict[tuple[str, str], np.ndarray] = {}
        for sid in sample_by_id:
            for condition in CONDITIONS:
                feature_path = feature_dir / f"{sid}__{condition}.npz"
                features, clock = load_npz(feature_path, 25)
                arrays[(sid, condition)] = features
                source_row = sample_by_id[sid]["natural_pcm"]
                if backend == "ditto":
                    expected_rows = (source_row["sample_count"] + 639) // 640
                    if features.shape != (expected_rows, 1024):
                        raise ValueError(f"Ditto support/dimension mismatch for {sid}/{condition}: {features.shape}")
                if condition == "natural":
                    natural[sid] = (features, clock)
                elif not np.array_equal(clock, natural[sid][1]):
                    raise ValueError(f"{backend}: query support changed across conditions for {sid}")
                if features.shape[0] != natural.get(sid, (features, None))[0].shape[0]:
                    raise ValueError(f"{backend}: feature time support changed across conditions for {sid}")
                summary_row = summary_rows[(sid, condition)]
                if summary_row.get("status") != "PASS":
                    raise ValueError(f"{backend}/{sid}/{condition}: source summary row did not PASS")
                sr_source = summary_row.get("source_pcm", {})
                if sr_source.get("pcm_float32_sha256") != source_row["natural_pcm"]["pcm_sha256"] or sr_source.get("wav_sha256") != source_row["natural_pcm"]["file_sha256"]:
                    raise ValueError(f"{backend}/{sid}/{condition}: original PCM binding mismatch")
                m = metrics(natural[sid][0], features)
                m.update({
                    "backend": backend,
                    "sample_id": sid,
                    "speaker_id": source_row["speaker_id"],
                    "condition": condition,
                    "query_count": int(features.shape[0]),
                    "feature_shape": list(features.shape),
                    "feature_dtype": str(features.dtype),
                    "query_clock": "query_index/25 seconds, exact array equality across conditions",
                    "feature_sha256": sha_file(feature_path),
                })
                # Independent reproduction of finite representation metrics in
                # the producer summary. Envelope correlations need condition PCM
                # arrays, which the producer intentionally did not persist.
                producer_metrics = summary_row.get("metrics", {})
                for key in METRICS:
                    expected = producer_metrics.get(key)
                    if expected is None or not np.isfinite(float(expected)):
                        raise ValueError(f"{backend}/{sid}/{condition}: producer omitted/nonfinite required metric {key}")
                    producer_numeric_check_count += 1
                    if abs(float(expected) - float(m[key])) > 2e-5:
                        raise ValueError(f"{backend}/{sid}/{condition}: independently recomputed {key} differs ({m[key]} vs {expected})")
                rows.append(m)
    if producer_numeric_check_count != len(rows) * len(METRICS) or len(rows) != 168:
        raise ValueError(f"numeric reproduction denominator incomplete: {producer_numeric_check_count}/{len(rows) * len(METRICS)}")
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
        "schema": "cross_tfg_native_encoder_condition_numeric_audit_v1",
        "status": "PASS",
        "scope": "historical_fit_only_8_clips_4_speakers; three frontend-native feature spaces; seven paired audio conditions",
        "denominator": {"backends": 3, "clips_per_backend": 8, "conditions": 7, "paired_rows": len(rows)},
        "source_pins": source_pins,
        "metrics": list(METRICS) + ["raw_feature_mean_offset_ratio", "raw_feature_temporal_direction_cosine"],
        "interpretation_limit": "Within each backend, these are audio-encoder representation changes versus that backend's natural waveform on the same indexed 25 Hz query clock. They do not measure generator output, lip motion, SyncNet, phoneme alignment, or synchronization benefit. Backend feature spaces are never compared directly.",
        "producer_summary_numeric_qc": {"checks": producer_numeric_check_count, "absolute_tolerance": 2e-5, "envelope_correlation_fields_recomputed": False},
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
