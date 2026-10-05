#!/usr/bin/env python3
"""Independent fail-fast numeric QC for the 336-row transition-ratio audit.

This verifier does not import the producer, the paired audit, or the ratio
supplement. It reloads the already-sealed encoder arrays and recomputes the
support, per-clip values, self controls, and equal-clip aggregates.
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
SUPPLEMENT_BYTES = 247060
SUPPLEMENT_SHA256 = "7ebbce1151f16d5c5486ecd808b97ccfb6018e5b1ffbfb9b9bc27ab5b2ba449c"
SUPPLEMENT_SOURCE_BYTES = 14062
SUPPLEMENT_SOURCE_SHA256 = "bf4eba698cd15c10139ff90291c3b9e5dbf64dc3536904f7b9a35f483624cde9"
PER_CLIP_CSV_BYTES = 82101
PER_CLIP_CSV_SHA256 = "0881becdc2d32f88caca1d478f30599a5d8c441df222ea2c6ab31983c5ffa968"
EQUAL_CLIP_CSV_BYTES = 4225
EQUAL_CLIP_CSV_SHA256 = "abf08b01a433cf9e5be77d5691f33bf2660570f907acaf1cde84a4bc66cb85ff"
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
DENOMINATOR_FLOOR = 1e-20
NUMERIC_ATOL = 1e-12


def file_pin(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input is not regular/non-symlink: {path}")
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            h.update(block)
    return {"path": str(path.resolve()), "bytes": size, "sha256": h.hexdigest()}


def assert_pin(path: Path, size: int, digest: str, label: str) -> dict[str, Any]:
    pin = file_pin(path)
    if (pin["bytes"], pin["sha256"]) != (size, digest):
        raise ValueError(f"{label} raw size/SHA mismatch")
    return pin


def load_x(path: Path, expected_sha: str) -> np.ndarray:
    pin = file_pin(path)
    if pin["sha256"] != expected_sha:
        raise ValueError(f"feature raw file SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"features", "query_index", "query_timestamp_num", "query_timestamp_den"}:
            raise ValueError(f"unexpected NPZ members: {path}")
        x = np.asarray(z["features"])
        q = np.asarray(z["query_index"])
        tn = np.asarray(z["query_timestamp_num"])
        td = np.asarray(z["query_timestamp_den"])
    if x.ndim < 2 or x.shape[0] < 3 or not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError(f"invalid feature array: {path}")
    expected = np.arange(x.shape[0], dtype=q.dtype)
    if q.shape != expected.shape or tn.shape != expected.shape or td.shape != expected.shape:
        raise ValueError(f"query-clock shape mismatch: {path}")
    if not np.array_equal(q, expected) or not np.array_equal(tn, expected) or not np.all(td == 25):
        raise ValueError(f"query clock differs from q/25: {path}")
    return x


def require_close(actual: float, expected: float, label: str, maxdiff: list[float]) -> None:
    diff = abs(float(actual) - float(expected))
    maxdiff[0] = max(maxdiff[0], diff)
    if diff > NUMERIC_ATOL:
        raise ValueError(f"{label} mismatch: actual={actual!r}, expected={expected!r}, absdiff={diff!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary-audit", type=Path, required=True)
    ap.add_argument("--primary-audit-source", type=Path, required=True)
    ap.add_argument("--supplement-report", type=Path, required=True)
    ap.add_argument("--supplement-source", type=Path, required=True)
    ap.add_argument("--per-clip-csv", type=Path, required=True)
    ap.add_argument("--equal-clip-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        raise FileExistsError(args.out_dir)

    pins = {
        "primary_audit": assert_pin(args.primary_audit, PRIMARY_AUDIT_BYTES, PRIMARY_AUDIT_SHA256, "primary audit"),
        "primary_source": assert_pin(args.primary_audit_source, PRIMARY_SOURCE_BYTES, PRIMARY_SOURCE_SHA256, "primary source"),
        "supplement_report": assert_pin(args.supplement_report, SUPPLEMENT_BYTES, SUPPLEMENT_SHA256, "supplement report"),
        "supplement_source": assert_pin(args.supplement_source, SUPPLEMENT_SOURCE_BYTES, SUPPLEMENT_SOURCE_SHA256, "supplement source"),
        "per_clip_csv": assert_pin(args.per_clip_csv, PER_CLIP_CSV_BYTES, PER_CLIP_CSV_SHA256, "per-clip CSV"),
        "equal_clip_csv": assert_pin(args.equal_clip_csv, EQUAL_CLIP_CSV_BYTES, EQUAL_CLIP_CSV_SHA256, "equal-clip CSV"),
    }
    audit = json.loads(args.primary_audit.read_text(encoding="utf-8"))
    supplement = json.loads(args.supplement_report.read_text(encoding="utf-8"))
    expected_denominator = {
        "anchors": list(ANCHORS), "backends": 3, "clips_per_backend": 8,
        "conditions": 7, "expected_rows": 336, "paired_rows": 336,
        "producer_metric_checks": 1848,
    }
    if audit.get("status") != "PASS" or audit.get("denominator") != expected_denominator:
        raise ValueError("primary audit is not the sealed 336-row/1848-check PASS")
    if supplement.get("status") != "PASS" or supplement.get("denominator") != {
        "backends": 3, "clips": 8, "conditions": 7,
        "anchors": list(ANCHORS), "paired_rows": 336,
    }:
        raise ValueError("supplement is not the sealed complete 336-row PASS")
    if supplement.get("primary_numeric_audit", {}).get("sha256") != pins["primary_audit"]["sha256"]:
        raise ValueError("supplement does not bind the pinned primary audit")
    if supplement.get("baseline_delta_norm_floor") != DENOMINATOR_FLOOR:
        raise ValueError("supplement denominator floor differs")
    if supplement.get("self_control") != {
        "max_abs_error_from_one_on_valid_support": 0.0,
        "required_exact_identity": True,
    }:
        raise ValueError("supplement self-control receipt differs")

    row_key = lambda r: (r.get("backend"), r.get("sample_id"), r.get("condition"), r.get("anchor"))
    primary_rows = {row_key(r): r for r in audit.get("rows", [])}
    supplement_rows = {row_key(r): r for r in supplement.get("per_clip", [])}
    samples = sorted({key[1] for key in primary_rows})
    expected_keys = {
        (b, sid, c, a)
        for b in BACKENDS for sid in samples for c in CONDITIONS for a in ANCHORS
    }
    if len(samples) != 8 or len(primary_rows) != 336 or len(supplement_rows) != 336:
        raise ValueError("primary/supplement row count or clip count mismatch")
    if set(primary_rows) != expected_keys or set(supplement_rows) != expected_keys:
        raise ValueError("primary/supplement row keys differ from exact 3×8×7×2 roster")

    roots: dict[str, Path] = {}
    summary_pins: dict[str, dict[str, Any]] = {}
    for backend in BACKENDS:
        summary = audit.get("source_pins", {}).get(backend, {}).get("summary", {})
        summary_path = Path(summary.get("path", ""))
        pin = file_pin(summary_path)
        if (pin["bytes"], pin["sha256"]) != (summary.get("bytes"), summary.get("sha256")):
            raise ValueError(f"{backend} summary file differs from primary audit input pin")
        roots[backend] = summary_path.parent
        summary_pins[backend] = pin

    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    feature_file_pins: dict[tuple[str, str, str], dict[str, Any]] = {}
    for backend in BACKENDS:
        for sid in samples:
            for condition in CONDITIONS:
                refs = [primary_rows[(backend, sid, condition, anchor)] for anchor in ANCHORS]
                digests = {r["condition_feature_sha256"] for r in refs}
                shapes = {tuple(r["feature_shape"]) for r in refs}
                dtypes = {r["feature_dtype"] for r in refs}
                if len(digests) != 1 or len(shapes) != 1 or len(dtypes) != 1:
                    raise ValueError(f"primary audit contains inconsistent file pin: {backend}/{sid}/{condition}")
                path = roots[backend] / "features" / f"{sid}__{condition}.npz"
                x = load_x(path, next(iter(digests)))
                if tuple(x.shape) != next(iter(shapes)) or str(x.dtype) != next(iter(dtypes)):
                    raise ValueError(f"actual array shape/dtype differs from primary audit: {backend}/{sid}/{condition}")
                arrays[(backend, sid, condition)] = x
                feature_file_pins[(backend, sid, condition)] = file_pin(path)

    per_clip_checked = 0
    self_rows = 0
    zero_support_counts: dict[str, dict[str, int]] = {b: {a: 0 for a in ANCHORS} for b in BACKENDS}
    total_interval_counts: dict[str, dict[str, int]] = {b: {a: 0 for a in ANCHORS} for b in BACKENDS}
    expected_recomputed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    max_abs_diff = [0.0]
    for backend in BACKENDS:
        for sid in samples:
            for anchor in ANCHORS:
                x = arrays[(backend, sid, anchor)].reshape(arrays[(backend, sid, anchor)].shape[0], -1).astype(np.float64)
                base_delta_norm = np.linalg.norm(np.diff(x, axis=0), axis=1)
                valid = base_delta_norm > DENOMINATOR_FLOOR
                total = int(base_delta_norm.size)
                valid_count = int(np.count_nonzero(valid))
                excluded = total - valid_count
                if not valid_count:
                    raise ValueError(f"no valid anchor transitions: {backend}/{sid}/{anchor}")
                zero_support_counts[backend][anchor] += excluded
                total_interval_counts[backend][anchor] += total
                for condition in CONDITIONS:
                    candidate = arrays[(backend, sid, condition)]
                    if candidate.shape != arrays[(backend, sid, anchor)].shape:
                        raise ValueError(f"pair shape mismatch: {backend}/{sid}/{anchor}/{condition}")
                    y = candidate.reshape(candidate.shape[0], -1).astype(np.float64)
                    candidate_delta_norm = np.linalg.norm(np.diff(y, axis=0), axis=1)
                    ratio = candidate_delta_norm[valid] / base_delta_norm[valid]
                    if not np.isfinite(ratio).all():
                        raise ValueError(f"nonfinite valid ratio: {backend}/{sid}/{anchor}/{condition}")
                    observed = supplement_rows[(backend, sid, condition, anchor)]
                    expected_metadata = {
                        "query_count": int(x.shape[0]),
                        "total_adjacent_query_pairs": total,
                        "valid_nonzero_anchor_delta_pairs": valid_count,
                        "excluded_zero_or_floor_anchor_delta_pairs": excluded,
                        "anchor_feature_npz_sha256": feature_file_pins[(backend, sid, anchor)]["sha256"],
                        "condition_feature_npz_sha256": feature_file_pins[(backend, sid, condition)]["sha256"],
                    }
                    for name, expected in expected_metadata.items():
                        if observed.get(name) != expected:
                            raise ValueError(f"support metadata mismatch at {backend}/{sid}/{anchor}/{condition}/{name}")
                    require_close(float(np.mean(ratio)), observed.get("valid_transition_delta_norm_ratio_mean"), "clip mean", max_abs_diff)
                    require_close(float(np.median(ratio)), observed.get("valid_transition_delta_norm_ratio_median"), "clip median", max_abs_diff)
                    if condition == anchor:
                        self_rows += 1
                        if not np.array_equal(ratio, np.ones_like(ratio)):
                            raise ValueError(f"self comparison is not exactly one on valid support: {backend}/{sid}/{anchor}")
                    expected_recomputed[(backend, sid, condition, anchor)] = {
                        "mean": float(np.mean(ratio)), "count": valid_count,
                        "total": total, "excluded": excluded,
                    }
                    per_clip_checked += 1

    if per_clip_checked != 336 or self_rows != 3 * 8 * 2 * 1:
        raise ValueError(f"numeric denominator mismatch: rows={per_clip_checked}, self_rows={self_rows}")

    equal_rows = supplement.get("equal_clip_summary", [])
    equal_map = {row_key(r): r for r in equal_rows}
    expected_equal_keys = {
        (b, "", c, a)
        for b in BACKENDS for c in CONDITIONS for a in ANCHORS
    }
    if len(equal_rows) != 42 or len(equal_map) != 42:
        raise ValueError("equal-clip aggregate row count is not 42")
    # Equal-clip rows use no sample ID; their tuple key is (backend, null, condition, anchor).
    equal_map = {(r.get("backend"), "", r.get("condition"), r.get("anchor")): r for r in equal_rows}
    if set(equal_map) != expected_equal_keys:
        raise ValueError("equal-clip aggregate identities differ from exact 3×7×2 roster")
    equal_checked = 0
    for backend in BACKENDS:
        for anchor in ANCHORS:
            for condition in CONDITIONS:
                per_clip = [expected_recomputed[(backend, sid, condition, anchor)] for sid in samples]
                aggregate = equal_map[(backend, "", condition, anchor)]
                valid_count = sum(r["count"] for r in per_clip)
                total_count = sum(r["total"] for r in per_clip)
                excluded = sum(r["excluded"] for r in per_clip)
                equal_mean = float(np.mean([r["mean"] for r in per_clip]))
                pooled_mean = float(sum(r["mean"] * r["count"] for r in per_clip) / valid_count)
                for name, expected in (
                    ("planned_clips", 8),
                    ("clips_with_valid_transitions", 8),
                    ("clips_missing_all_transitions", 0),
                    ("valid_transition_pairs", valid_count),
                    ("all_transition_pairs", total_count),
                    ("excluded_zero_or_floor_anchor_delta_pairs", excluded),
                ):
                    if aggregate.get(name) != expected:
                        raise ValueError(f"aggregate support mismatch at {backend}/{anchor}/{condition}/{name}")
                require_close(equal_mean, aggregate.get("equal_clip_mean_of_valid_interval_means"), "equal-clip mean", max_abs_diff)
                require_close(pooled_mean, aggregate.get("pooled_valid_interval_mean"), "pooled mean", max_abs_diff)
                equal_checked += 1
    if equal_checked != 42:
        raise ValueError(f"equal-clip aggregation denominator mismatch: {equal_checked}")

    with args.per_clip_csv.open("r", newline="", encoding="utf-8") as f:
        per_clip_csv = list(csv.DictReader(f))
    with args.equal_clip_csv.open("r", newline="", encoding="utf-8") as f:
        equal_csv = list(csv.DictReader(f))
    if len(per_clip_csv) != 336 or len(equal_csv) != 42:
        raise ValueError("CSV row denominator mismatch")
    if {row_key(r) for r in per_clip_csv} != expected_keys:
        raise ValueError("per-clip CSV keys differ from exact expected rows")
    if {(r.get("backend"), "", r.get("condition"), r.get("anchor")) for r in equal_csv} != expected_equal_keys:
        raise ValueError("equal-clip CSV keys differ from exact expected rows")

    verifier = file_pin(Path(__file__))
    result = {
        "schema": "cross_tfg_root_native_encoder_transition_ratio_support_qc_v1",
        "status": "PASS",
        "scope": "fit8-only array-level independent numeric recomputation; no frontend/model/scorer calls",
        "pins": pins,
        "verifier_source": verifier,
        "denominator": {"feature_npz_files": 168, "per_clip_pair_rows": per_clip_checked, "equal_clip_rows": equal_checked, "interval_ratio_comparisons": sum(total_interval_counts[b][a] for b in BACKENDS for a in ANCHORS) * len(CONDITIONS)},
        "zero_or_floor_anchor_delta_intervals": zero_support_counts,
        "total_anchor_intervals": total_interval_counts,
        "self_comparisons": {"rows": self_rows, "max_abs_error": 0.0, "exact_one_on_valid_support": True},
        "comparison_tolerance_absolute": NUMERIC_ATOL,
        "max_abs_difference_to_supplement": max_abs_diff[0],
        "equal_clip_aggregation_rows_checked": equal_checked,
        "interpretation_limit": "Ratios describe encoder transition-vector norms only on nonzero anchor-transition support. They are not generator, SyncNet, lip-motion, phoneme, or synchronization effects. Natural and fixed-RMS anchors remain separate.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out_path = args.out_dir / "root_transition_ratio_support_qc.json"
    with out_path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "verifier": verifier, "qc_path": str(out_path), "qc_bytes": out_path.stat().st_size}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
