#!/usr/bin/env python3
"""Recompute transition-delta norm ratios with explicit zero-denominator support.

This source-only numeric supplement reads the already-QC'd fit8 feature NPZs.
It does not call a model, renderer, scorer, or API. The previous paired audit's
11 producer-parity fields are left untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_AUDIT_BYTES = 677048
PRIMARY_AUDIT_SHA256 = "22ed1ba05e30cdf2cfd2aa6e4cb49944a5070371f9642b61b10c5b40ac9d43f4"
PRIMARY_SOURCE_BYTES = 31397
PRIMARY_SOURCE_SHA256 = "30b253ab6fe4845d92ca8209ade530a440a20329caa80125224a00bcf7f10f0b"
BACKENDS = ("wav2lip", "muse", "ditto")
CONDITIONS = (
    "natural",
    "global_minus_6db",
    "global_plus_6db",
    "fixed_rms_0p0376838173",
    "local_r090_accent015_tilt_p025",
    "local_r090_flatten015_tilt_m025",
    "local_r090_accent_0.30_tilt0",
)
ANCHORS = ("natural", "fixed_rms_0p0376838173")
BASELINE_DELTA_FLOOR = 1e-20


def sha_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    count = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
            count += len(block)
    return h.hexdigest(), count


def raw_pin(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular non-symlink file: {path}")
    digest, count = sha_file(path)
    return {"path": str(path.resolve()), "bytes": count, "sha256": digest}


def load_feature(path: Path, expected_file_sha: str, expected_shape: list[int], expected_dtype: str):
    file_pin = raw_pin(path)
    if file_pin["sha256"] != expected_file_sha:
        raise ValueError(f"feature-file raw SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"features", "query_index", "query_timestamp_num", "query_timestamp_den"}:
            raise ValueError(f"unexpected NPZ members: {path}")
        x = np.asarray(z["features"])
        q = np.asarray(z["query_index"])
        num = np.asarray(z["query_timestamp_num"])
        den = np.asarray(z["query_timestamp_den"])
    if list(x.shape) != expected_shape or str(x.dtype) != expected_dtype:
        raise ValueError(f"feature shape/dtype changed: {path}")
    if x.ndim < 2 or x.shape[0] < 3 or not np.isfinite(x).all():
        raise ValueError(f"feature array invalid: {path}")
    idx = np.arange(x.shape[0], dtype=q.dtype)
    if q.shape != idx.shape or num.shape != idx.shape or den.shape != idx.shape:
        raise ValueError(f"clock-array shape mismatch: {path}")
    if not np.array_equal(q, idx) or not np.array_equal(num, idx) or not np.all(den == 25):
        raise ValueError(f"feature clock is not exact q/25: {path}")
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-report", type=Path, required=True)
    ap.add_argument("--primary-audit-source", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        raise FileExistsError(args.out_dir)

    audit_pin = raw_pin(args.audit_report)
    if (audit_pin["bytes"], audit_pin["sha256"]) != (PRIMARY_AUDIT_BYTES, PRIMARY_AUDIT_SHA256):
        raise ValueError("primary numeric audit raw bytes differ from the reviewed actual run")
    source_pin = raw_pin(args.primary_audit_source)
    if (source_pin["bytes"], source_pin["sha256"]) != (PRIMARY_SOURCE_BYTES, PRIMARY_SOURCE_SHA256):
        raise ValueError("primary auditor source differs from the reviewed attempt04 implementation")
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("denominator") != {
        "anchors": list(ANCHORS),
        "backends": 3,
        "clips_per_backend": 8,
        "conditions": 7,
        "expected_rows": 336,
        "paired_rows": 336,
        "producer_metric_checks": 1848,
    }:
        raise ValueError("primary audit status/denominator is not the reviewed complete fit8 result")

    # Every feature path is derived from the primary audit's raw-pinned summary path.
    backend_roots: dict[str, Path] = {}
    for backend in BACKENDS:
        summary_record = audit.get("source_pins", {}).get(backend, {}).get("summary", {})
        summary_path = Path(summary_record.get("path", ""))
        pin = raw_pin(summary_path)
        if (pin["bytes"], pin["sha256"]) != (summary_record.get("bytes"), summary_record.get("sha256")):
            raise ValueError(f"{backend} summary differs from primary audit source pin")
        backend_roots[backend] = summary_path.parent

    rows = audit.get("rows", [])
    if len(rows) != 336:
        raise ValueError(f"primary audit row count changed: {len(rows)}")
    sample_ids = sorted({r.get("sample_id") for r in rows})
    if len(sample_ids) != 8 or any(not isinstance(s, str) or not s for s in sample_ids):
        raise ValueError("primary audit does not contain eight clip IDs")
    expected_keys = {
        (b, sid, c, a)
        for b in BACKENDS
        for sid in sample_ids
        for c in CONDITIONS
        for a in ANCHORS
    }
    row_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("backend"), row.get("sample_id"), row.get("condition"), row.get("anchor"))
        if key in row_map:
            raise ValueError(f"duplicate primary audit row {key}")
        row_map[key] = row
    if set(row_map) != expected_keys:
        raise ValueError("primary audit row identities differ from exact 3×8×7×2 support")

    feature_cache: dict[tuple[str, str, str], np.ndarray] = {}
    feature_raw_shas: dict[tuple[str, str, str], str] = {}
    for backend in BACKENDS:
        for sid in sample_ids:
            for condition in CONDITIONS:
                path = backend_roots[backend] / "features" / f"{sid}__{condition}.npz"
                references = [
                    row_map[(backend, sid, condition, anchor)]
                    for anchor in ANCHORS
                ]
                expected_shas = {r["condition_feature_sha256"] for r in references}
                shapes = {tuple(r["feature_shape"]) for r in references}
                dtypes = {r["feature_dtype"] for r in references}
                if len(expected_shas) != 1 or len(shapes) != 1 or len(dtypes) != 1:
                    raise ValueError(f"primary audit pair rows disagree on feature pin: {backend}/{sid}/{condition}")
                x = load_feature(path, next(iter(expected_shas)), list(next(iter(shapes))), next(iter(dtypes)))
                feature_cache[(backend, sid, condition)] = x
                feature_raw_shas[(backend, sid, condition)] = next(iter(expected_shas))

    ratio_rows: list[dict[str, Any]] = []
    self_max_abs_error = 0.0
    for backend in BACKENDS:
        for sid in sample_ids:
            for anchor in ANCHORS:
                ref = feature_cache[(backend, sid, anchor)]
                ref2d = ref.reshape(ref.shape[0], -1).astype(np.float64)
                ref_delta_norm = np.linalg.norm(np.diff(ref2d, axis=0), axis=1)
                if ref_delta_norm.size == 0 or not np.isfinite(ref_delta_norm).all():
                    raise ValueError(f"invalid reference transitions: {backend}/{sid}/{anchor}")
                valid = ref_delta_norm > BASELINE_DELTA_FLOOR
                total = int(ref_delta_norm.size)
                valid_count = int(np.count_nonzero(valid))
                if valid_count == 0:
                    raise ValueError(f"no nonzero reference transitions: {backend}/{sid}/{anchor}")
                for condition in CONDITIONS:
                    candidate = feature_cache[(backend, sid, condition)]
                    if candidate.shape != ref.shape:
                        raise ValueError(f"paired feature shapes differ: {backend}/{sid}/{anchor}/{condition}")
                    candidate2d = candidate.reshape(candidate.shape[0], -1).astype(np.float64)
                    delta_norm = np.linalg.norm(np.diff(candidate2d, axis=0), axis=1)
                    if not np.isfinite(delta_norm).all():
                        raise ValueError(f"nonfinite candidate transition: {backend}/{sid}/{condition}")
                    ratio = delta_norm[valid] / ref_delta_norm[valid]
                    if not np.isfinite(ratio).all():
                        raise ValueError(f"nonfinite valid transition ratio: {backend}/{sid}/{anchor}/{condition}")
                    if condition == anchor:
                        err = float(np.max(np.abs(ratio - 1.0)))
                        self_max_abs_error = max(self_max_abs_error, err)
                    ratio_rows.append({
                        "backend": backend,
                        "sample_id": sid,
                        "anchor": anchor,
                        "condition": condition,
                        "query_count": int(ref.shape[0]),
                        "total_adjacent_query_pairs": total,
                        "valid_nonzero_anchor_delta_pairs": valid_count,
                        "excluded_zero_or_floor_anchor_delta_pairs": total - valid_count,
                        "valid_transition_delta_norm_ratio_mean": float(np.mean(ratio)),
                        "valid_transition_delta_norm_ratio_median": float(np.median(ratio)),
                        "anchor_feature_npz_sha256": feature_raw_shas[(backend, sid, anchor)],
                        "condition_feature_npz_sha256": feature_raw_shas[(backend, sid, condition)],
                    })

    if len(ratio_rows) != 336 or self_max_abs_error != 0.0:
        status = "COMPLETE_WITH_SELF_CONTROL_MISMATCH"
    else:
        status = "PASS"
    aggregates: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for anchor in ANCHORS:
            selected_base = [r for r in ratio_rows if r["backend"] == backend and r["anchor"] == anchor]
            for condition in CONDITIONS:
                selected = [r for r in selected_base if r["condition"] == condition]
                if len(selected) != 8 or len({r["sample_id"] for r in selected}) != 8:
                    raise ValueError(f"aggregate denominator mismatch: {backend}/{anchor}/{condition}")
                means = [r["valid_transition_delta_norm_ratio_mean"] for r in selected]
                pooled_sum = 0.0
                pooled_count = 0
                total_pairs = 0
                excluded_pairs = 0
                for r in selected:
                    count = r["valid_nonzero_anchor_delta_pairs"]
                    pooled_sum += r["valid_transition_delta_norm_ratio_mean"] * count
                    pooled_count += count
                    total_pairs += r["total_adjacent_query_pairs"]
                    excluded_pairs += r["excluded_zero_or_floor_anchor_delta_pairs"]
                aggregates.append({
                    "backend": backend,
                    "anchor": anchor,
                    "condition": condition,
                    "planned_clips": 8,
                    "clips_with_valid_transitions": len(means),
                    "clips_missing_all_transitions": 0,
                    "equal_clip_mean_of_valid_interval_means": float(np.mean(means)),
                    "pooled_valid_interval_mean": pooled_sum / pooled_count,
                    "valid_transition_pairs": pooled_count,
                    "all_transition_pairs": total_pairs,
                    "excluded_zero_or_floor_anchor_delta_pairs": excluded_pairs,
                })

    source_path = Path(__file__).resolve()
    result = {
        "schema": "cross_tfg_native_encoder_transition_ratio_support_audit_v1",
        "status": status,
        "scope": "historical_fit_only_8_clips_4_speakers; existing frozen feature arrays only; no model or score calls",
        "denominator": {
            "backends": 3,
            "clips": 8,
            "conditions": 7,
            "anchors": list(ANCHORS),
            "paired_rows": len(ratio_rows),
        },
        "ratio_definition": "||condition[q+1]-condition[q]|| / ||anchor[q+1]-anchor[q]|| only where anchor-delta L2 norm > 1e-20; zero/floor anchor transitions are explicitly excluded and counted per clip and aggregate",
        "self_control": {"max_abs_error_from_one_on_valid_support": self_max_abs_error, "required_exact_identity": True},
        "primary_numeric_audit": audit_pin,
        "primary_auditor_source": source_pin,
        "baseline_delta_norm_floor": BASELINE_DELTA_FLOOR,
        "interpretation_limit": "This describes encoder transition-vector magnitudes on a selected nonzero-baseline support. It is not a generator, SyncNet, lip-motion, phoneme, or synchronization result. Natural- and RMS-anchored summaries are separate and must not be differenced across anchors.",
        "per_clip": ratio_rows,
        "equal_clip_summary": aggregates,
        "source": {"path": str(source_path), "bytes": source_path.stat().st_size, "sha256": sha_file(source_path)[0]},
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "per_clip_transition_ratio_support.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ratio_rows)
    with (args.out_dir / "equal_clip_transition_ratio_support.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)
    with (args.out_dir / "transition_ratio_support_audit.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
