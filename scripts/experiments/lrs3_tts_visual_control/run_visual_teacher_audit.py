#!/usr/bin/env python3
"""Run the duration-compatible TTS visual-teacher audit."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
VISUAL_MODULE_DIR = REPO / "scripts/experiments/lrs3_tts_visual_advantage"
if str(VISUAL_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(VISUAL_MODULE_DIR))

from scripts.experiments.lrs3_tts_visual_control.protocol import (  # noqa: E402
    assert_finite_json,
    canonical_sha256,
    file_sha256,
    git_info,
    load_json,
    validate_test_lock,
    write_json,
)
from video_features import create_landmarker, extract_video_features, save_visual_sequence  # noqa: E402
from visual_metrics import pair_metrics  # noqa: E402

STAGE_PREFIX = "01_visual_teacher_audit_retry"
STAGE_ID = "01_visual_teacher_audit_retry1"
SCHEMA_VERSION = 1
FROZEN_PROTOCOL_MANIFEST = REPO / "runs/lrs3_tts_visual_control_20260825/00_protocol_lock_retry2/manifest.json"
FROZEN_TEST_LOCK = FROZEN_PROTOCOL_MANIFEST.with_name("test_lock.json")
FROZEN_PROTOCOL_SHA256 = "e1c970542eb90a2dac787463ebe4e8708ad3f2c4b4f1c4379f22e8382a29ec38"
FROZEN_TEST_LOCK_SHA256 = "295ccba71fb50d873a3ddf1b5cfd895418fd5bd95d242ec6441c57ef7e9660d4"
DEFAULT_CODE_REVIEW = REPO / "_reviews/lrs3_tts_visual_control/01_visual_teacher_audit/code_review.json"
MIN_VALID_FRACTION = 0.90
MIN_EXACT_COVERAGE = 0.85
DTW_BAND_FRACTION = 0.20
DTW_MAX_RUN = 2
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 10000


def _stage_id(output_dir: Path) -> str:
    value = output_dir.name
    if not value.startswith(STAGE_PREFIX) or not value.removeprefix(STAGE_PREFIX).isdigit():
        raise ValueError(f"unexpected Stage01 output directory name: {value}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _source_manifest(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in paths]


def _validate_review(path: Path, source_paths: Sequence[Path]) -> dict[str, Any]:
    review = load_json(path)
    if review.get("stage_id") != "01_visual_teacher_audit" or review.get("status") != "PASS":
        raise ValueError("Stage01 code review is not PASS")
    if review.get("test_status") != "PASS":
        raise ValueError("Stage01 code review tests are not PASS")
    if review.get("source_manifest_sha256") != canonical_sha256(_source_manifest(source_paths)):
        raise ValueError("Stage01 code review source hash mismatch")
    return review


def _validate_stage00() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = FROZEN_PROTOCOL_MANIFEST.resolve()
    lock_path = FROZEN_TEST_LOCK.resolve()
    if file_sha256(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("Stage00 protocol hash does not match the frozen Stage00 artifact")
    if file_sha256(lock_path) != FROZEN_TEST_LOCK_SHA256:
        raise ValueError("Stage00 test-lock hash does not match the frozen Stage00 artifact")
    protocol = load_json(protocol_path)
    lock = load_json(lock_path)
    if protocol.get("stage_id") != "00_protocol_lock_retry2":
        raise ValueError("unexpected frozen Stage00 protocol stage")
    if protocol.get("status") != "complete" or protocol.get("engineering_decision") != "GO":
        raise ValueError("frozen Stage00 protocol is not complete GO")
    if protocol.get("next_allowed_stage") != STAGE_ID:
        raise ValueError("frozen Stage00 protocol does not authorize Stage01")
    validate_test_lock(lock)
    if lock.get("status") != "sealed_unvisited":
        raise ValueError("Stage00 test lock is not sealed_unvisited")
    for key, value in lock.items():
        if key.endswith(("_media_opened", "_derived_features_created", "_scores_created")) and value is not False:
            raise ValueError(f"Stage00 test lock records access: {key}")
    split = protocol.get("split")
    cohort = protocol.get("cohort")
    if not isinstance(split, Mapping) or not isinstance(cohort, Mapping):
        raise ValueError("Stage00 split/cohort is missing")
    if split.get("policy_id") != "pre_score_common_support_23_v1":
        raise ValueError("Stage00 cohort policy is not the frozen common-support policy")
    if len(split.get("effective_fit_groups", [])) != 23:
        raise ValueError("Stage00 effective fit group count is not 23")
    if len(split.get("fit_train_groups", [])) != 17 or len(split.get("fit_selection_groups", [])) != 6:
        raise ValueError("Stage00 effective split is not 17+6")
    records = cohort.get("records")
    if not isinstance(records, list) or len(records) != 133:
        raise ValueError("Stage00 cohort is not the frozen 133-record cohort")
    if any(row.get("protocol_split") != "fit" for row in records):
        raise ValueError("Stage00 cohort contains a non-fit record")
    return protocol, lock


def _snapshot_asset(asset: Path, target: Path) -> tuple[Path, str]:
    resolved = asset.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = file_sha256(resolved)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite asset snapshot: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolved, target)
    if file_sha256(target) != digest:
        raise ValueError("landmarker asset snapshot hash mismatch")
    return target, digest


def _record_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "sample_id",
        "source_group",
        "real_video",
        "real_video_sha256",
        "natural_video",
        "natural_video_sha256",
        "candidate_video",
        "candidate_video_sha256",
    )
    if any(not record.get(key) for key in required):
        raise ValueError(f"Stage00 visual record has missing fields: {record.get('sample_id')}")
    return {key: record[key] for key in required}


def _metric_gate(sequences: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    exact = metrics["exact_time"]
    dtw = metrics["visual_dtw"]
    valid_fraction = {arm: float(sequences[arm].valid.mean()) for arm in ("real", "natural", "candidate")}
    exact_coverage = {arm: float(exact[arm]["coverage"]) for arm in ("natural", "candidate")}
    checks = {
        "valid_fraction_at_least_0_90": all(value >= MIN_VALID_FRACTION for value in valid_fraction.values()),
        "exact_time_coverage_at_least_0_85": all(value >= MIN_EXACT_COVERAGE for value in exact_coverage.values()),
        "finite_primary_metrics": all(
            _finite(exact[arm].get("distance")) for arm in ("natural", "candidate")
        ) and _finite(exact.get("benefit")),
        "finite_secondary_metrics": all(
            _finite(dtw[arm].get(field))
            for arm in ("natural", "candidate")
            for field in ("distance", "coverage", "path_length", "warp_burden")
        ) and _finite(dtw.get("benefit")),
    }
    return {
        "eligible": bool(all(checks.values())),
        "checks": checks,
        "valid_fraction": valid_fraction,
        "exact_time_coverage": exact_coverage,
    }


def _record_result(
    record: Mapping[str, Any],
    asset_snapshot: Path,
    feature_root: Path,
    landmarker_handle: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    contract = _record_contract(record)
    paths = {
        "real": Path(str(record["real_video"])).resolve(),
        "natural": Path(str(record["natural_video"])).resolve(),
        "candidate": Path(str(record["candidate_video"])).resolve(),
    }
    expected_hashes = {
        "real": str(record["real_video_sha256"]),
        "natural": str(record["natural_video_sha256"]),
        "candidate": str(record["candidate_video_sha256"]),
    }
    for arm in paths:
        if not paths[arm].is_file():
            raise FileNotFoundError(f"{arm} video is missing: {paths[arm]}")
        if file_sha256(paths[arm]) != expected_hashes[arm]:
            raise ValueError(f"{arm} video hash mismatch: {record['sample_id']}")
    extraction_kwargs = {} if landmarker_handle is None else {"landmarker_handle": landmarker_handle}
    sequences = {
        arm: extract_video_features(paths[arm], asset_snapshot, **extraction_kwargs)
        for arm in paths
    }
    for arm in paths:
        if file_sha256(paths[arm]) != expected_hashes[arm]:
            raise ValueError(f"{arm} video changed during extraction: {record['sample_id']}")
    feature_paths = {}
    for arm, sequence in sequences.items():
        target = feature_root / str(record["sample_id"]) / f"{arm}.npz"
        save_visual_sequence(target, sequence)
        feature_paths[arm] = str(target.resolve())
    metrics = pair_metrics(
        sequences["real"],
        sequences["natural"],
        sequences["candidate"],
        band_fraction=DTW_BAND_FRACTION,
        max_run=DTW_MAX_RUN,
    )
    for metric_name in ("exact_time", "visual_dtw"):
        metrics[metric_name]["candidate"] = metrics[metric_name].pop("tts")
    gate = _metric_gate(sequences, metrics)
    return {
        **contract,
        "feature_paths": feature_paths,
        "feature_metadata": {arm: sequences[arm].metadata for arm in paths},
        "metrics": _json_safe(metrics),
        "eligibility": _json_safe(gate),
    }


def _group_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["eligibility"]["eligible"]:
            by_group.setdefault(str(row["source_group"]), []).append(row)
    group_rows = []
    for group in sorted(by_group):
        selected = by_group[group]
        exact = [float(row["metrics"]["exact_time"]["benefit"]) for row in selected]
        dtw = [float(row["metrics"]["visual_dtw"]["benefit"]) for row in selected]
        group_rows.append(
            {
                "source_group": group,
                "eligible_record_count": len(selected),
                "exact_time_benefit_mean": float(np.mean(exact)),
                "visual_dtw_benefit_mean": float(np.mean(dtw)),
            }
        )
    return {"group_rows": group_rows, "group_count": len(group_rows)}


def _exact_sign_flip(values: Sequence[float]) -> dict[str, Any]:
    observed = float(np.mean(np.asarray(values, dtype=np.float64)))
    sums = np.asarray([0.0], dtype=np.float64)
    for value in values:
        sums = np.concatenate((sums + float(value), sums - float(value)))
    p_value = float(np.count_nonzero(sums / len(values) >= observed - 1e-15) / len(sums))
    return {"mean": observed, "p_value": p_value, "n_groups": len(values), "enumerated_signs": int(len(sums))}


def _bootstrap_interval(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "lower_2_5": float(np.quantile(means, 0.025)),
        "upper_97_5": float(np.quantile(means, 0.975)),
    }


def _statistics(rows: Sequence[Mapping[str, Any]], effective_groups: Sequence[str]) -> dict[str, Any]:
    group = _group_statistics(rows)
    observed_groups = {row["source_group"] for row in group["group_rows"]}
    missing = [value for value in effective_groups if value not in observed_groups]
    if missing:
        return {
            "complete": False,
            "group_statistics": group,
            "missing_eligible_groups": missing,
        }
    exact_values = [row["exact_time_benefit_mean"] for row in group["group_rows"]]
    dtw_values = [row["visual_dtw_benefit_mean"] for row in group["group_rows"]]
    return {
        "complete": True,
        "group_statistics": group,
        "missing_eligible_groups": [],
        "primary": {
            "endpoint": "D_exact(G,V_N)-D_exact(G,V_M)",
            "benefit_direction": "positive_is_better",
            "sign_flip": _exact_sign_flip(exact_values),
            "bootstrap_ci": _bootstrap_interval(exact_values),
            "positive_group_count": int(sum(value > 0 for value in exact_values)),
        },
        "secondary": {
            "endpoint": "D_DTW(G,V_N)-D_DTW(G,V_M)",
            "benefit_direction": "positive_is_better",
            "mean": float(np.mean(dtw_values)),
            "positive_group_count": int(sum(value > 0 for value in dtw_values)),
            "bootstrap_ci": _bootstrap_interval(dtw_values),
        },
    }


def _decision(statistics: Mapping[str, Any], eligible_count: int, minimum_count: int) -> tuple[str, str, str]:
    if not statistics.get("complete") or eligible_count < minimum_count:
        return "BLOCKED", "not_available", "eligible denominator or effective-group coverage is incomplete"
    primary = statistics["primary"]["sign_flip"]
    if primary["mean"] > 0.0 and primary["p_value"] <= 0.05:
        return "GO", "GO", "primary exact-time visual benefit is positive and passes the preregistered group sign-flip gate"
    return "GO", "NO_GO", "engineering complete but the preregistered primary exact-time visual-benefit gate is not met"


def _extract_records(
    records: Sequence[Mapping[str, Any]],
    asset_snapshot: Path,
    feature_root: Path,
    record_root: Path,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1 and len(records) > 1:
        chunk_size = math.ceil(len(records) / workers)
        chunks = [list(records[start : start + chunk_size]) for start in range(0, len(records), chunk_size)]
        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as pool:
            futures = [
                pool.submit(_extract_records, chunk, asset_snapshot, feature_root, record_root, 1)
                for chunk in chunks
            ]
            for chunk, future in zip(chunks, futures, strict=True):
                try:
                    chunk_rows, chunk_failures = future.result()
                except Exception as error:
                    chunk_rows = []
                    chunk_failures = [
                        {
                            "sample_id": str(record.get("sample_id", "")),
                            "source_group": str(record.get("source_group", "")),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        for record in chunk
                    ]
                results.extend(chunk_rows)
                failures.extend(chunk_failures)
        return results, failures

    landmarker_handle = create_landmarker(asset_snapshot)
    record_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        for record in records:
            sample_id = str(record["sample_id"])
            try:
                result = _record_result(record, asset_snapshot, feature_root, landmarker_handle)
                record_rows.append(result)
                write_json(record_root / f"{sample_id}.json", result)
            except Exception as error:
                failure = {
                    "sample_id": sample_id,
                    "source_group": str(record.get("source_group", "")),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(failure)
                write_json(record_root / f"{sample_id}.failure.json", failure)
    finally:
        landmarker_handle[1].close()
    return record_rows, failures


def run(
    output_dir: Path,
    *,
    landmarker_asset: Path,
    code_review_path: Path,
    source_paths: Sequence[Path],
    workers: int,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output}")
    stage_id = _stage_id(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol, test_lock = _validate_stage00()
    review = _validate_review(code_review_path.resolve(), source_paths)
    asset_snapshot, asset_sha256 = _snapshot_asset(landmarker_asset, output / "snapshots/mediapipe_landmarker.task")
    records = protocol["cohort"]["records"]
    feature_root = output / "features"
    record_root = output / "records"
    record_rows, failures = _extract_records(records, asset_snapshot, feature_root, record_root, workers)
    effective_groups = list(protocol["split"]["effective_fit_groups"])
    eligible_count = sum(bool(row["eligibility"]["eligible"]) for row in record_rows)
    minimum_count = int(protocol["visual_contract"]["minimum_eligible_record_count"])
    statistics = _statistics(record_rows, effective_groups) if not failures else {
        "complete": False,
        "group_statistics": {"group_rows": [], "group_count": 0},
        "missing_eligible_groups": effective_groups,
    }
    engineering, scientific, reason = _decision(statistics, eligible_count, minimum_count) if not failures else ("BLOCKED", "not_available", "one or more frozen records failed visual extraction")
    own_sources = _source_manifest(source_paths)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "lrs3_duration_compatible_tts_visual_teacher_audit",
        "protocol_id": protocol["protocol_id"],
        "stage_id": stage_id,
        "status": "complete" if engineering == "GO" else "blocked",
        "engineering_decision": engineering,
        "scientific_decision": scientific,
        "reason": reason,
        "protocol_manifest": {"path": str(FROZEN_PROTOCOL_MANIFEST.resolve()), "sha256": FROZEN_PROTOCOL_SHA256},
        "test_lock": {"path": str(FROZEN_TEST_LOCK.resolve()), "sha256": FROZEN_TEST_LOCK_SHA256},
        "landmarker_asset": {"path": str(landmarker_asset.resolve()), "sha256": asset_sha256, "snapshot": str(asset_snapshot.resolve())},
        "extraction_workers": workers,
        "record_count": len(records),
        "feature_record_count": len(record_rows),
        "failure_count": len(failures),
        "eligible_record_count": eligible_count,
        "minimum_eligible_record_count": minimum_count,
        "effective_fit_group_count": len(effective_groups),
        "effective_fit_groups": effective_groups,
        "statistics": statistics,
        "visual_contract": {
            "primary": "natural_clock_exact_time_canonical_mouth_landmark_distance",
            "benefit": "D_exact(G,V_N)-D_exact(G,V_M)",
            "secondary": "visual_only_constrained_dtw_canonical_mouth_shape_distance",
            "dtw_band_fraction": DTW_BAND_FRACTION,
            "dtw_max_run": DTW_MAX_RUN,
            "valid_fraction_min": MIN_VALID_FRACTION,
            "exact_time_coverage_min": MIN_EXACT_COVERAGE,
            "dtw_can_rescue_primary": False,
            "syncnet_used_for_selection": False,
            "syncnet_used_for_decision": False,
            "audio_scores_used": False,
        },
        "media_access": {
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
            "syncnet_scores_created": False,
        },
        "failures": failures,
        "source_manifest": own_sources,
        "source_manifest_sha256": canonical_sha256(own_sources),
        "code_review": {"path": str(code_review_path.resolve()), "sha256": file_sha256(code_review_path)},
        "code_review_payload_sha256": canonical_sha256(review),
        "git": git_info(),
    }
    assert_finite_json(manifest)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "stage_id": stage_id,
        "status": manifest["status"],
        "engineering_decision": engineering,
        "scientific_decision": scientific,
        "record_count": len(records),
        "feature_record_count": len(record_rows),
        "failure_count": len(failures),
        "eligible_record_count": eligible_count,
        "minimum_eligible_record_count": minimum_count,
        "effective_fit_group_count": len(effective_groups),
        "primary_mean": statistics.get("primary", {}).get("sign_flip", {}).get("mean"),
        "primary_p_value": statistics.get("primary", {}).get("sign_flip", {}).get("p_value"),
        "next_allowed_stage": "02_identity_chain_retry1" if scientific == "GO" else None,
    }
    decision = {**summary, "reason": reason}
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", decision)
    if failures:
        write_json(output / "failure.json", {"status": "blocked", "failures": failures})
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landmarker-asset", type=Path, required=True)
    parser.add_argument("--code-review", type=Path, default=DEFAULT_CODE_REVIEW)
    parser.add_argument("--test-source", type=Path, default=REPO / "tests/experiments/lrs3_tts_visual_control/test_visual_teacher_audit.py")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = (
        Path(__file__).resolve(),
        VISUAL_MODULE_DIR / "video_features.py",
        VISUAL_MODULE_DIR / "visual_metrics.py",
        args.test_source.resolve(),
        REPO / "tests/experiments/lrs3_tts_visual_advantage/test_visual_features_metrics.py",
    )
    result = run(
        args.output_dir,
        landmarker_asset=args.landmarker_asset,
        code_review_path=args.code_review,
        source_paths=source_paths,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("engineering_decision") == "GO" else 2


if __name__ == "__main__":
    sys.exit(main())
