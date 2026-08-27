#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
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
    write_json,
)
from scripts.experiments.lrs3_tts_visual_control.run_visual_teacher_audit import (  # noqa: E402
    _decision,
    _json_safe,
    _metric_gate,
    _record_result,
    _statistics,
)
from video_features import create_landmarker  # noqa: E402

RUN_ROOT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
STAGE00_PATH = RUN_ROOT / "00_protocol_lock_expanded_retry3/manifest.json"
ALIGNMENT_PATH = RUN_ROOT / "01_mfa3_screen_expanded_retry2/alignment_manifest.json"
CANDIDATE_PATH = RUN_ROOT / "02_candidate_audio_visual_expanded_retry3/candidate_manifest.json"
FACE_PREFLIGHT_PATH = RUN_ROOT / "03_visual_expanded_face_preflight_retry1/face_preflight_manifest.json"
RENDER_PATH = RUN_ROOT / "03_visual_expanded_wav2lip_render_retry3/render_manifest.json"
RENDER_PROTOCOL_PATH = RENDER_PATH.with_name("protocol_manifest.json")
LANDMARKER_ASSET = REPO / "checkpoints/mediapipe/face_landmarker.task"
DEFAULT_CODE_REVIEW = REPO / "_reviews/lrs3_mfa_linear_replacement_mfa3_exploratory/01_visual_teacher_audit_expanded_face_ready/code_review.json"
STAGE_ID = "01_visual_teacher_audit_expanded_face_ready_retry1"
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
SCHEMA_VERSION = 1
INPUT_CLEAN_RECORD_COUNT = 62
INPUT_FACE_READY_RECORD_COUNT = 59
MINIMUM_ELIGIBLE_RECORD_COUNT = math.ceil(INPUT_CLEAN_RECORD_COUNT * 44 / 48)
EXPECTED_GROUP_COUNT = 23
MIN_VALID_FRACTION = 0.90
MIN_EXACT_COVERAGE = 0.85
DTW_BAND_FRACTION = 0.20
DTW_MAX_RUN = 2
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 10000


def _source_manifest(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in paths]


def _validate_review(path: Path, source_paths: Sequence[Path]) -> dict[str, Any]:
    review = load_json(path)
    if review.get("stage_id") != STAGE_ID or review.get("status") != "PASS":
        raise ValueError("expanded Stage01 code review is not PASS")
    if review.get("test_status") != "PASS":
        raise ValueError("expanded Stage01 code review tests are not PASS")
    if review.get("source_manifest_sha256") != canonical_sha256(_source_manifest(source_paths)):
        raise ValueError("expanded Stage01 code review source hash mismatch")
    return review


def _binding(path: Path, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(expected, Mapping):
        raise ValueError(f"{label} parent binding is missing")
    if Path(str(expected.get("path"))).resolve() != path.resolve():
        raise ValueError(f"{label} parent path mismatch")
    if str(expected.get("sha256")) != file_sha256(path):
        raise ValueError(f"{label} parent hash mismatch")


def _validate_stage00(path: Path) -> dict[str, Any]:
    if path.resolve() != STAGE00_PATH.resolve():
        raise ValueError("expanded Stage01 requires the frozen expanded Stage00 path")
    stage00 = load_json(path)
    if (
        stage00.get("protocol_id") != PROTOCOL_ID
        or stage00.get("stage_id") != "00_protocol_lock_expanded_retry3"
        or stage00.get("status") != "complete"
        or stage00.get("engineering_decision") != "GO"
        or stage00.get("scientific_decision") != "not_available"
    ):
        raise ValueError("expanded Stage00 is not the complete engineering GO artifact")
    cohort = stage00.get("cohort")
    records = cohort.get("records") if isinstance(cohort, Mapping) else None
    if (
        not isinstance(records, list)
        or len(records) != 122
        or cohort.get("record_count") != 122
        or cohort.get("source_group_count") != EXPECTED_GROUP_COUNT
        or len({str(row.get("sample_id")) for row in records}) != 122
        or any(row.get("protocol_split") != "fit" for row in records)
    ):
        raise ValueError("expanded Stage00 cohort contract mismatch")
    test_lock = stage00.get("test_lock")
    if not isinstance(test_lock, Mapping) or test_lock.get("status") != "sealed_unvisited":
        raise ValueError("expanded Stage00 test lock is not sealed_unvisited")
    for key, value in test_lock.items():
        if key.endswith(("_media_opened", "_derived_features_created", "_scores_created")) and value is not False:
            raise ValueError(f"expanded Stage00 test lock records access: {key}")
    media_access = stage00.get("media_access")
    if not isinstance(media_access, Mapping):
        raise ValueError("expanded Stage00 media access record is missing")
    for key in ("internal_dev_media_opened", "validation_media_opened", "test_media_opened", "scores_created"):
        if media_access.get(key) is not False:
            raise ValueError(f"expanded Stage00 records forbidden media access: {key}")
    return stage00


def _validate_alignment(path: Path, stage00_path: Path, stage00: Mapping[str, Any]) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if path.resolve() != ALIGNMENT_PATH.resolve():
        raise ValueError("expanded Stage01 requires the frozen MFA3 alignment path")
    alignment = load_json(path)
    if (
        alignment.get("protocol_id") != PROTOCOL_ID
        or alignment.get("stage_id") != "01_mfa3_screen_expanded_retry2"
        or alignment.get("status") != "partial"
        or alignment.get("clean_record_count") != INPUT_CLEAN_RECORD_COUNT
        or alignment.get("ordered_input_count") != 122
    ):
        raise ValueError("expanded MFA3 alignment screen contract mismatch")
    _binding(stage00_path, alignment.get("parents", {}).get("stage00"), "alignment Stage00")
    records = alignment.get("records")
    if (
        not isinstance(records, list)
        or len(records) != INPUT_CLEAN_RECORD_COUNT
        or len({str(row.get("sample_id")) for row in records}) != INPUT_CLEAN_RECORD_COUNT
    ):
        raise ValueError("expanded alignment clean records are incomplete")
    stage_ids = [str(row["sample_id"]) for row in stage00["cohort"]["records"]]
    clean_ids = [str(row["sample_id"]) for row in records]
    if any(sample_id not in set(stage_ids) for sample_id in clean_ids):
        raise ValueError("expanded alignment contains a non-Stage00 sample")
    if alignment.get("clean_sample_ids_sha256") != canonical_sha256(clean_ids):
        raise ValueError("expanded alignment clean sample ID hash mismatch")
    if alignment.get("ordered_input_sample_ids_sha256") != canonical_sha256(stage_ids):
        raise ValueError("expanded alignment input sample ID hash mismatch")
    return alignment, records


def _validate_candidate(path: Path, stage00_path: Path, alignment_path: Path, alignment: Mapping[str, Any]) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if path.resolve() != CANDIDATE_PATH.resolve():
        raise ValueError("expanded Stage01 requires the frozen candidate path")
    candidate = load_json(path)
    if (
        candidate.get("protocol_id") != PROTOCOL_ID
        or candidate.get("stage_id") != "02_candidate_audio_visual_expanded_retry3"
        or candidate.get("status") != "complete"
        or candidate.get("engineering_decision") != "GO"
        or candidate.get("record_count") != INPUT_CLEAN_RECORD_COUNT
    ):
        raise ValueError("expanded candidate audio contract mismatch")
    parents = candidate.get("parents")
    _binding(stage00_path, parents.get("stage00") if isinstance(parents, Mapping) else None, "candidate Stage00")
    _binding(alignment_path, parents.get("alignment") if isinstance(parents, Mapping) else None, "candidate alignment")
    results = candidate.get("results")
    if not isinstance(results, list) or len(results) != INPUT_CLEAN_RECORD_COUNT:
        raise ValueError("expanded candidate results are incomplete")
    alignment_ids = [str(row["sample_id"]) for row in alignment["records"]]
    candidate_ids = [str(row["sample_id"]) for row in results]
    if candidate_ids != alignment_ids or len(set(candidate_ids)) != INPUT_CLEAN_RECORD_COUNT:
        raise ValueError("expanded candidate order does not match clean alignment")
    for row in results:
        if int(row.get("candidate", {}).get("waveform_qc", {}).get("samples", -1)) != int(row.get("candidate_samples", -2)):
            raise ValueError(f"candidate waveform QC mismatch: {row.get('sample_id')}")
        if int(row.get("mapping", {}).get("speech_fallback_frames", -1)) != 0:
            raise ValueError(f"candidate speech fallback detected: {row.get('sample_id')}")
    return candidate, results


def _validate_preflight(path: Path, stage00_path: Path, alignment_path: Path, candidate_path: Path) -> tuple[dict[str, Any], list[str]]:
    if path.resolve() != FACE_PREFLIGHT_PATH.resolve():
        raise ValueError("expanded Stage01 requires the frozen face preflight path")
    preflight = load_json(path)
    if (
        preflight.get("protocol_id") != PROTOCOL_ID
        or preflight.get("stage_id") != "03_visual_expanded_face_preflight_retry1"
        or preflight.get("status") != "complete"
        or preflight.get("engineering_decision") != "GO"
        or preflight.get("input_record_count") != INPUT_CLEAN_RECORD_COUNT
        or preflight.get("face_ready_record_count") != INPUT_FACE_READY_RECORD_COUNT
        or preflight.get("face_ready_group_count") != EXPECTED_GROUP_COUNT
    ):
        raise ValueError("expanded face preflight contract mismatch")
    parents = preflight.get("parents")
    _binding(stage00_path, parents.get("stage00") if isinstance(parents, Mapping) else None, "face preflight Stage00")
    _binding(alignment_path, parents.get("alignment") if isinstance(parents, Mapping) else None, "face preflight alignment")
    _binding(candidate_path, parents.get("candidate") if isinstance(parents, Mapping) else None, "face preflight candidate")
    if preflight.get("selection", {}).get("score_based_selection") is not False or preflight.get("selection", {}).get("visual_metric_used_for_selection") is not False or preflight.get("selection", {}).get("syncnet_used_for_selection") is not False:
        raise ValueError("expanded face preflight selection is not score-free")
    results = preflight.get("results")
    ready_ids = [str(row["sample_id"]) for row in results if row.get("face_ready") is True] if isinstance(results, list) else []
    all_ids = [str(row["sample_id"]) for row in results] if isinstance(results, list) else []
    if len(all_ids) != INPUT_CLEAN_RECORD_COUNT or len(set(all_ids)) != INPUT_CLEAN_RECORD_COUNT or len(ready_ids) != INPUT_FACE_READY_RECORD_COUNT or len(set(ready_ids)) != INPUT_FACE_READY_RECORD_COUNT:
        raise ValueError("expanded face preflight IDs/counts are invalid")
    if preflight.get("face_ready_sample_ids") != ready_ids:
        raise ValueError("expanded face preflight ready ID order mismatch")
    return preflight, ready_ids


def _validate_render(path: Path, protocol_path: Path, stage00_path: Path, alignment_path: Path, candidate_path: Path, preflight_path: Path, ready_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.resolve() != RENDER_PATH.resolve() or protocol_path.resolve() != RENDER_PROTOCOL_PATH.resolve():
        raise ValueError("expanded Stage01 requires the frozen retry2 render paths")
    protocol = load_json(protocol_path)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("stage_id") != "03_visual_expanded_wav2lip_render_retry3"
        or protocol.get("status") != "locked"
        or protocol.get("cohort", {}).get("record_count") != INPUT_FACE_READY_RECORD_COUNT
        or protocol.get("cohort", {}).get("source_group_count") != EXPECTED_GROUP_COUNT
    ):
        raise ValueError("expanded render protocol contract mismatch")
    parents = protocol.get("parents")
    _binding(stage00_path, parents.get("stage00") if isinstance(parents, Mapping) else None, "render Stage00")
    _binding(alignment_path, parents.get("alignment") if isinstance(parents, Mapping) else None, "render alignment")
    _binding(candidate_path, parents.get("candidate") if isinstance(parents, Mapping) else None, "render candidate")
    _binding(preflight_path, parents.get("face_preflight") if isinstance(parents, Mapping) else None, "render face preflight")
    if protocol.get("selection", {}).get("score_based_selection") is not False or protocol.get("selection", {}).get("visual_metric_used_for_selection") is not False or protocol.get("selection", {}).get("syncnet_used_for_selection") is not False:
        raise ValueError("expanded render selection is not score-free")
    cohort_records = protocol.get("cohort", {}).get("records")
    protocol_ids = [str(row["sample_id"]) for row in cohort_records] if isinstance(cohort_records, list) else []
    if protocol_ids != list(ready_ids):
        raise ValueError("expanded render protocol IDs do not match face-ready order")
    render = load_json(path)
    if (
        render.get("protocol_id") != PROTOCOL_ID
        or render.get("stage_id") != "03_visual_expanded_wav2lip_render_retry3"
        or render.get("record_count") != INPUT_FACE_READY_RECORD_COUNT
        or render.get("render_count") != INPUT_FACE_READY_RECORD_COUNT * 2
        or render.get("face_ready_sample_ids") != list(ready_ids)
    ):
        raise ValueError("expanded render result contract mismatch")
    _binding(protocol_path, {"path": str(protocol_path), "sha256": file_sha256(protocol_path)}, "render protocol")
    render_rows = render.get("renders")
    if not isinstance(render_rows, list) or len(render_rows) != INPUT_FACE_READY_RECORD_COUNT * 2:
        raise ValueError("expanded render rows are incomplete")
    expected_keys = {(sample_id, arm) for sample_id in ready_ids for arm in ("natural", "candidate")}
    actual_keys = {(str(row.get("sample_id")), str(row.get("arm"))) for row in render_rows}
    if actual_keys != expected_keys:
        raise ValueError("expanded render IDs/arms do not match face-ready cohort")
    for row in render_rows:
        video = Path(str(row.get("video", ""))).resolve()
        if not video.is_file() or file_sha256(video) != str(row.get("video_sha256")):
            raise ValueError(f"expanded render hash mismatch: {row.get('sample_id')} {row.get('arm')}")
    return protocol, render


def _build_records(stage00: Mapping[str, Any], alignment_records: Sequence[Mapping[str, Any]], candidate_results: Sequence[Mapping[str, Any]], render_protocol: Mapping[str, Any], render: Mapping[str, Any], ready_ids: Sequence[str]) -> list[dict[str, Any]]:
    stage_by_id = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
    alignment_by_id = {str(row["sample_id"]): row for row in alignment_records}
    candidate_by_id = {str(row["sample_id"]): row for row in candidate_results}
    render_by_key = {(str(row["sample_id"]), str(row["arm"])): row for row in render["renders"]}
    protocol_by_id = {str(row["sample_id"]): row for row in render_protocol["cohort"]["records"]}
    records: list[dict[str, Any]] = []
    ready_set = set(ready_ids)
    for stage_row in stage00["cohort"]["records"]:
        sample_id = str(stage_row["sample_id"])
        if sample_id not in ready_set:
            continue
        alignment_row = alignment_by_id.get(sample_id)
        candidate_row = candidate_by_id.get(sample_id)
        render_row = protocol_by_id.get(sample_id)
        if alignment_row is None or candidate_row is None or render_row is None:
            raise ValueError(f"expanded Stage01 record join missing: {sample_id}")
        if str(render_row["face_video"]) != str(stage_row["source_video"]):
            raise ValueError(f"expanded render/source video mismatch: {sample_id}")
        if str(candidate_row["natural_audio_sha256"]) != str(stage_row["natural_audio_sha256"]):
            raise ValueError(f"expanded candidate/natural audio mismatch: {sample_id}")
        paths = {
            "real": (render_row["face_video"], render_row["face_video_sha256"]),
            "natural": (render_by_key[(sample_id, "natural")]["video"], render_by_key[(sample_id, "natural")]["video_sha256"]),
            "candidate": (render_by_key[(sample_id, "candidate")]["video"], render_by_key[(sample_id, "candidate")]["video_sha256"]),
        }
        records.append({
            "sample_id": sample_id,
            "source_group": str(stage_row["source_group"]),
            "protocol_split": str(stage_row["protocol_split"]),
            "real_video": str(paths["real"][0]),
            "real_video_sha256": str(paths["real"][1]),
            "natural_video": str(paths["natural"][0]),
            "natural_video_sha256": str(paths["natural"][1]),
            "candidate_video": str(paths["candidate"][0]),
            "candidate_video_sha256": str(paths["candidate"][1]),
            "natural_audio": str(render_row["natural_audio"]),
            "natural_audio_sha256": str(render_row["natural_audio_sha256"]),
            "candidate_audio": str(render_row["candidate_audio"]),
            "candidate_audio_sha256": str(render_row["candidate_audio_sha256"]),
            "natural_samples": int(render_row["natural_samples"]),
            "candidate_samples": int(render_row["candidate_samples"]),
            "alignment_record_sha256": canonical_sha256(alignment_row),
            "candidate_result_sha256": canonical_sha256(candidate_row),
        })
    if len(records) != INPUT_FACE_READY_RECORD_COUNT or [row["sample_id"] for row in records] != list(ready_ids):
        raise ValueError("expanded Stage01 joined record order/count mismatch")
    if len({row["source_group"] for row in records}) != EXPECTED_GROUP_COUNT:
        raise ValueError("expanded Stage01 joined group coverage mismatch")
    return records


def _extract_records(records: Sequence[Mapping[str, Any]], asset_snapshot: Path, feature_root: Path, record_root: Path, workers: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1 and len(records) > 1:
        chunk_size = math.ceil(len(records) / workers)
        chunks = [list(records[start : start + chunk_size]) for start in range(0, len(records), chunk_size)]
        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as pool:
            futures = [pool.submit(_extract_records, chunk, asset_snapshot, feature_root, record_root, 1) for chunk in chunks]
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

    handle = create_landmarker(asset_snapshot)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        for record in records:
            sample_id = str(record["sample_id"])
            try:
                result = _record_result(record, asset_snapshot, feature_root, handle)
                rows.append(result)
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
        handle[1].close()
    return rows, failures


def run(output_dir: Path, *, landmarker_asset: Path, code_review_path: Path, source_paths: Sequence[Path], workers: int) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        stage00_path = STAGE00_PATH.resolve()
        alignment_path = ALIGNMENT_PATH.resolve()
        candidate_path = CANDIDATE_PATH.resolve()
        preflight_path = FACE_PREFLIGHT_PATH.resolve()
        render_path = RENDER_PATH.resolve()
        render_protocol_path = RENDER_PROTOCOL_PATH.resolve()
        stage00 = _validate_stage00(stage00_path)
        alignment, alignment_records = _validate_alignment(alignment_path, stage00_path, stage00)
        candidate, candidate_results = _validate_candidate(candidate_path, stage00_path, alignment_path, alignment)
        preflight, ready_ids = _validate_preflight(preflight_path, stage00_path, alignment_path, candidate_path)
        render_protocol, render = _validate_render(render_path, render_protocol_path, stage00_path, alignment_path, candidate_path, preflight_path, ready_ids)
        review = _validate_review(code_review_path.resolve(), source_paths)
        asset = landmarker_asset.resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        asset_sha256 = file_sha256(asset)
        asset_snapshot = output / "snapshots/mediapipe_landmarker.task"
        asset_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset, asset_snapshot)
        if file_sha256(asset_snapshot) != asset_sha256:
            raise ValueError("landmarker asset snapshot hash mismatch")
        records = _build_records(stage00, alignment_records, candidate_results, render_protocol, render, ready_ids)
        record_rows, failures = _extract_records(records, asset_snapshot, output / "features", output / "records", workers)
        eligible_count = sum(bool(row["eligibility"]["eligible"]) for row in record_rows)
        effective_groups = list(dict.fromkeys(str(row["source_group"]) for row in records))
        statistics = _statistics(record_rows, effective_groups) if not failures else {
            "complete": False,
            "group_statistics": {"group_rows": [], "group_count": 0},
            "missing_eligible_groups": effective_groups,
        }
        engineering, scientific, reason = _decision(statistics, eligible_count, MINIMUM_ELIGIBLE_RECORD_COUNT) if not failures else ("BLOCKED", "not_available", "one or more face-ready records failed visual extraction")
        own_sources = _source_manifest(source_paths)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "lrs3_mfa3_duration_compatible_tts_visual_teacher_face_ready_audit",
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "complete" if engineering == "GO" else "blocked",
            "engineering_decision": engineering,
            "scientific_decision": scientific,
            "reason": reason,
            "parents": {
                "stage00": {"path": str(stage00_path), "sha256": file_sha256(stage00_path)},
                "alignment": {"path": str(alignment_path), "sha256": file_sha256(alignment_path)},
                "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
                "face_preflight": {"path": str(preflight_path), "sha256": file_sha256(preflight_path)},
                "render_protocol": {"path": str(render_protocol_path), "sha256": file_sha256(render_protocol_path)},
                "render": {"path": str(render_path), "sha256": file_sha256(render_path)},
            },
            "landmarker_asset": {"path": str(asset), "sha256": asset_sha256, "snapshot": str(asset_snapshot.resolve())},
            "cohort": {
                "input_clean_record_count": INPUT_CLEAN_RECORD_COUNT,
                "face_ready_record_count": INPUT_FACE_READY_RECORD_COUNT,
                "record_count": len(records),
                "source_group_count": len(effective_groups),
                "ordered_sample_ids": [str(row["sample_id"]) for row in records],
                "ordered_sample_ids_sha256": canonical_sha256([str(row["sample_id"]) for row in records]),
                "records": records,
            },
            "extraction_workers": workers,
            "feature_record_count": len(record_rows),
            "failure_count": len(failures),
            "eligible_record_count": eligible_count,
            "minimum_eligible_record_count": MINIMUM_ELIGIBLE_RECORD_COUNT,
            "minimum_denominator_basis": "62 clean MFA3 records before score-free Wav2Lip face preflight",
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
            "selection": {
                "face_preflight_used": True,
                "score_based_selection": False,
                "visual_metric_used_for_selection": False,
                "syncnet_used_for_selection": False,
                "source_order_preserved": True,
            },
            "media_access": {
                "fit_media_opened": True,
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
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": manifest["status"],
            "engineering_decision": engineering,
            "scientific_decision": scientific,
            "record_count": len(records),
            "input_clean_record_count": INPUT_CLEAN_RECORD_COUNT,
            "face_ready_record_count": INPUT_FACE_READY_RECORD_COUNT,
            "feature_record_count": len(record_rows),
            "failure_count": len(failures),
            "eligible_record_count": eligible_count,
            "minimum_eligible_record_count": MINIMUM_ELIGIBLE_RECORD_COUNT,
            "effective_fit_group_count": len(effective_groups),
            "primary_mean": statistics.get("primary", {}).get("sign_flip", {}).get("mean"),
            "primary_p_value": statistics.get("primary", {}).get("sign_flip", {}).get("p_value"),
            "next_allowed_stage": "02_identity_chain_retry1" if scientific == "GO" else None,
        }
        write_json(output / "manifest.json", manifest)
        write_json(output / "summary.json", summary)
        write_json(output / "decision.json", {**summary, "reason": reason})
        if failures:
            write_json(output / "failure.json", {"status": "blocked", "failures": failures})
        return summary
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "error": str(exc),
        }
        write_json(output / "failure.json", payload)
        write_json(output / "summary.json", payload)
        write_json(output / "decision.json", {**payload, "next_allowed_stage": None})
        return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landmarker-asset", type=Path, default=LANDMARKER_ASSET)
    parser.add_argument("--code-review", type=Path, default=DEFAULT_CODE_REVIEW)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = (
        Path(__file__).resolve(),
        REPO / "scripts/experiments/lrs3_tts_visual_control/run_visual_teacher_audit.py",
        VISUAL_MODULE_DIR / "video_features.py",
        VISUAL_MODULE_DIR / "visual_metrics.py",
        REPO / "scripts/experiments/lrs3_mfa_linear_replacement_mfa3_exploratory/run_visual_expanded_face_ready_render_retry3.py",
        REPO / "tests/experiments/lrs3_tts_visual_control/test_visual_teacher_audit_face_ready.py",
        REPO / "tests/experiments/lrs3_tts_visual_control/test_visual_teacher_audit.py",
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
    raise SystemExit(main())
