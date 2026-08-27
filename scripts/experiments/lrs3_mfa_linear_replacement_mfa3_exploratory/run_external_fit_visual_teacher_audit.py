from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    canonical_sha256,
    file_sha256,
    load_json,
    write_json,
)

REPO = Path(__file__).resolve().parents[3]
VISUAL_MODULE_DIR = REPO / "scripts/experiments/lrs3_tts_visual_advantage"
if str(VISUAL_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(VISUAL_MODULE_DIR))

from scripts.experiments.lrs3_tts_visual_control.run_visual_teacher_audit import (  # noqa: E402
    _decision,
    _extract_records,
    _statistics,
)

PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE_ID = "05_external_fit_visual_teacher_audit_retry7"
STAGE00_ID = "00_external_fit_supplement_protocol_retry4"
ALIGNMENT_ID = "01_external_fit_mfa3_screen_retry4"
CANDIDATE_ID = "02_external_fit_candidate_audio_retry4"
FACE_PREFLIGHT_ID = "03_external_fit_face_preflight_retry4"
RENDER_ID = "04_external_fit_wav2lip_render_retry6"
ROOT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE00_PATH = ROOT / STAGE00_ID / "manifest.json"
ALIGNMENT_PATH = ROOT / ALIGNMENT_ID / "alignment_manifest.json"
CANDIDATE_PATH = ROOT / CANDIDATE_ID / "candidate_manifest.json"
FACE_PREFLIGHT_PATH = ROOT / FACE_PREFLIGHT_ID / "face_preflight_manifest.json"
RENDER_PROTOCOL_PATH = ROOT / RENDER_ID / "render_protocol.json"
RENDER_PATH = ROOT / RENDER_ID / "render_manifest.json"
EXPECTED_OUTPUT_DIR = ROOT / STAGE_ID
LANDMARKER_ASSET = REPO / "checkpoints/mediapipe/face_landmarker.task"
EXPECTED_INPUT_COUNT = 266
EXPECTED_CLEAN_COUNT = 208
EXPECTED_FACE_READY_COUNT = 194
EXPECTED_GROUP_COUNT = 23
MINIMUM_ELIGIBLE_RECORD_COUNT = 191
MIN_VALID_FRACTION = 0.90
MIN_EXACT_COVERAGE = 0.85
DTW_BAND_FRACTION = 0.20
DTW_MAX_RUN = 2


def _binding(binding: Mapping[str, Any], path: Path, label: str) -> None:
    resolved = path.resolve()
    if Path(str(binding.get("path", ""))).resolve() != resolved:
        raise ValueError(f"{label} path mismatch")
    if not resolved.is_file() or file_sha256(resolved) != str(binding.get("sha256", "")):
        raise ValueError(f"{label} hash mismatch")


def _assert_sealed_media(media: Mapping[str, Any], label: str) -> None:
    for key in (
        "internal_dev_media_opened",
        "validation_media_opened",
        "test_media_opened",
        "syncnet_scores_created",
        "scores_created",
    ):
        if key in media and media.get(key) is not False:
            raise ValueError(f"{label} records forbidden access: {key}")


def _validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    stage00 = load_json(STAGE00_PATH)
    if (
        stage00.get("protocol_id") != PROTOCOL_ID
        or stage00.get("stage_id") != STAGE00_ID
        or stage00.get("status") != "complete"
        or stage00.get("engineering_decision") != "GO"
        or stage00.get("scientific_decision") != "not_available"
    ):
        raise ValueError("external-fit Stage00 is not the complete retry4 engineering GO")
    if (STAGE00_PATH.parent / "manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(STAGE00_PATH):
        raise ValueError("external-fit Stage00 companion hash mismatch")
    stage_records = stage00.get("cohort", {}).get("records")
    if not isinstance(stage_records, list) or len(stage_records) != EXPECTED_INPUT_COUNT:
        raise ValueError("external-fit Stage00 cohort is incomplete")
    stage_by_id = {str(row["sample_id"]): row for row in stage_records}
    if len(stage_by_id) != EXPECTED_INPUT_COUNT:
        raise ValueError("external-fit Stage00 IDs are not unique")
    effective_groups = {str(value) for value in stage00.get("split", {}).get("effective_fit_groups", [])}
    if len(effective_groups) != EXPECTED_GROUP_COUNT or "6W2dsnhC18Q" in effective_groups:
        raise ValueError("external-fit Stage00 effective groups are invalid")
    _assert_sealed_media(stage00.get("media_access", {}), "Stage00")

    alignment = load_json(ALIGNMENT_PATH)
    if (
        alignment.get("protocol_id") != PROTOCOL_ID
        or alignment.get("stage_id") != ALIGNMENT_ID
        or alignment.get("status") != "partial"
        or alignment.get("engineering_decision") != "GO"
        or alignment.get("scientific_decision") != "not_available"
        or alignment.get("ordered_input_count") != EXPECTED_INPUT_COUNT
        or alignment.get("clean_record_count") != EXPECTED_CLEAN_COUNT
        or alignment.get("minimum_eligible_record_count") != MINIMUM_ELIGIBLE_RECORD_COUNT
        or alignment.get("clean_source_group_count") != EXPECTED_GROUP_COUNT
        or alignment.get("missing_clean_groups") != []
    ):
        raise ValueError("external-fit alignment contract does not authorize visual audit")
    if (ALIGNMENT_PATH.parent / "alignment_manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(ALIGNMENT_PATH):
        raise ValueError("external-fit alignment companion hash mismatch")
    _binding(alignment.get("parents", {}).get("stage00", {}), STAGE00_PATH, "alignment Stage00")
    alignment_records = alignment.get("records")
    if not isinstance(alignment_records, list) or len(alignment_records) != EXPECTED_CLEAN_COUNT:
        raise ValueError("external-fit clean alignment is incomplete")
    alignment_ids = [str(row["sample_id"]) for row in alignment_records]
    stage_ids = [str(row["sample_id"]) for row in stage_records]
    if alignment_ids != [sample_id for sample_id in stage_ids if sample_id in set(alignment_ids)]:
        raise ValueError("external-fit alignment order is not Stage00 source order")
    if alignment.get("clean_sample_ids_sha256") != canonical_sha256(alignment_ids):
        raise ValueError("external-fit alignment clean ID hash mismatch")
    if len(alignment.get("failures", [])) != 115 or alignment.get("failed_record_count") != 58:
        raise ValueError("external-fit alignment failures are not retained")

    candidate = load_json(CANDIDATE_PATH)
    if (
        candidate.get("protocol_id") != PROTOCOL_ID
        or candidate.get("stage_id") != CANDIDATE_ID
        or candidate.get("status") != "complete"
        or candidate.get("engineering_decision") != "GO"
        or candidate.get("scientific_decision") != "not_available"
        or candidate.get("record_count") != EXPECTED_CLEAN_COUNT
    ):
        raise ValueError("external-fit candidate is not the complete retry4 artifact")
    if (CANDIDATE_PATH.parent / "candidate_manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(CANDIDATE_PATH):
        raise ValueError("external-fit candidate companion hash mismatch")
    _binding(candidate.get("parents", {}).get("stage00", {}), STAGE00_PATH, "candidate Stage00")
    _binding(candidate.get("parents", {}).get("alignment", {}), ALIGNMENT_PATH, "candidate alignment")
    candidate_results = candidate.get("results")
    if not isinstance(candidate_results, list) or [str(row["sample_id"]) for row in candidate_results] != alignment_ids:
        raise ValueError("external-fit candidate order does not match alignment")
    candidate_by_id = {str(row["sample_id"]): row for row in candidate_results}
    if len(candidate_by_id) != EXPECTED_CLEAN_COUNT:
        raise ValueError("external-fit candidate IDs are not unique")
    _assert_sealed_media(candidate.get("media_access", {}), "candidate")

    preflight = load_json(FACE_PREFLIGHT_PATH)
    if (
        preflight.get("protocol_id") != PROTOCOL_ID
        or preflight.get("stage_id") != FACE_PREFLIGHT_ID
        or preflight.get("status") != "complete"
        or preflight.get("engineering_decision") != "GO"
        or preflight.get("scientific_decision") != "not_available"
        or preflight.get("input_record_count") != EXPECTED_CLEAN_COUNT
        or preflight.get("face_ready_record_count") != EXPECTED_FACE_READY_COUNT
        or preflight.get("face_ready_record_count") < MINIMUM_ELIGIBLE_RECORD_COUNT
        or preflight.get("face_ready_group_count") != EXPECTED_GROUP_COUNT
    ):
        raise ValueError("external-fit face preflight does not authorize visual audit")
    if (FACE_PREFLIGHT_PATH.parent / "face_preflight_manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(FACE_PREFLIGHT_PATH):
        raise ValueError("external-fit face preflight companion hash mismatch")
    for key, path in (("stage00", STAGE00_PATH), ("alignment", ALIGNMENT_PATH), ("candidate", CANDIDATE_PATH)):
        _binding(preflight.get("parents", {}).get(key, {}), path, f"face preflight {key}")
    _assert_sealed_media(preflight.get("media_access", {}), "face preflight")
    preflight_rows = preflight.get("results")
    ready_ids = [str(row["sample_id"]) for row in preflight_rows if row.get("face_ready") is True] if isinstance(preflight_rows, list) else []
    if not isinstance(preflight_rows, list) or len(preflight_rows) != EXPECTED_CLEAN_COUNT or [str(row["sample_id"]) for row in preflight_rows] != alignment_ids or ready_ids != preflight.get("face_ready_sample_ids"):
        raise ValueError("external-fit face preflight IDs are invalid")
    if len(ready_ids) != EXPECTED_FACE_READY_COUNT or len(set(ready_ids)) != EXPECTED_FACE_READY_COUNT:
        raise ValueError("external-fit face-ready IDs are invalid")
    preflight_by_id = {str(row["sample_id"]): row for row in preflight_rows}

    render_protocol = load_json(RENDER_PROTOCOL_PATH)
    if (
        render_protocol.get("protocol_id") != PROTOCOL_ID
        or render_protocol.get("stage_id") != RENDER_ID
        or render_protocol.get("status") != "locked"
        or render_protocol.get("engineering_decision") != "GO"
        or render_protocol.get("cohort", {}).get("record_count") != EXPECTED_FACE_READY_COUNT
        or render_protocol.get("cohort", {}).get("ordered_sample_ids") != ready_ids
    ):
        raise ValueError("external-fit render protocol does not match face-ready cohort")
    if (RENDER_PROTOCOL_PATH.parent / "render_protocol.sha256").read_text(encoding="utf-8").strip() != file_sha256(RENDER_PROTOCOL_PATH):
        raise ValueError("external-fit render protocol companion hash mismatch")
    for key, path in (("stage00", STAGE00_PATH), ("alignment", ALIGNMENT_PATH), ("candidate", CANDIDATE_PATH), ("face_preflight", FACE_PREFLIGHT_PATH)):
        _binding(render_protocol.get("parents", {}).get(key, {}), path, f"render protocol {key}")
    if any(render_protocol.get("selection", {}).get(key) is not False for key in ("score_based_selection", "visual_metric_used_for_selection", "syncnet_used_for_selection", "audio_quality_used_for_selection", "alignment_result_used_for_selection")):
        raise ValueError("external-fit render protocol selection is not score-free")
    _assert_sealed_media(render_protocol.get("media_access", {}), "render protocol")

    render = load_json(RENDER_PATH)
    if (
        render.get("protocol_id") != PROTOCOL_ID
        or render.get("stage_id") != RENDER_ID
        or render.get("status") != "complete"
        or render.get("engineering_decision") != "GO"
        or render.get("record_count") != EXPECTED_FACE_READY_COUNT
        or render.get("render_count") != EXPECTED_FACE_READY_COUNT * 2
        or render.get("face_ready_sample_ids") != ready_ids
    ):
        raise ValueError("external-fit render result contract is incomplete")
    if (RENDER_PATH.parent / "render_manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(RENDER_PATH):
        raise ValueError("external-fit render companion hash mismatch")
    for key, path in (("stage00", STAGE00_PATH), ("alignment", ALIGNMENT_PATH), ("candidate", CANDIDATE_PATH), ("face_preflight", FACE_PREFLIGHT_PATH)):
        _binding(render.get("parents", {}).get(key, {}), path, f"render {key}")
    _assert_sealed_media(render.get("media_access", {}), "render")
    render_rows = render.get("renders")
    expected_keys = {(sample_id, arm) for sample_id in ready_ids for arm in ("natural", "candidate")}
    actual_keys = {(str(row.get("sample_id")), str(row.get("arm"))) for row in render_rows} if isinstance(render_rows, list) else set()
    if actual_keys != expected_keys:
        raise ValueError("external-fit render IDs/arms are incomplete")
    render_by_key = {(str(row["sample_id"]), str(row["arm"])): row for row in render_rows}

    records: list[dict[str, Any]] = []
    for sample_id in ready_ids:
        stage_row = stage_by_id[sample_id]
        face_row = preflight_by_id[sample_id]
        natural_render = render_by_key[(sample_id, "natural")]
        candidate_render = render_by_key[(sample_id, "candidate")]
        paths = {
            "real": (Path(str(face_row["face_video"])).resolve(), str(face_row["face_video_sha256"])),
            "natural": (Path(str(natural_render["video"])).resolve(), str(natural_render["video_sha256"])),
            "candidate": (Path(str(candidate_render["video"])).resolve(), str(candidate_render["video_sha256"])),
        }
        if paths["real"][0] != Path(str(stage_row["source_video"])).resolve() or paths["real"][1] != str(stage_row["source_video_sha256"]):
            raise ValueError(f"real source video binding mismatch: {sample_id}")
        for arm, (video, expected_hash) in paths.items():
            if not video.is_file() or file_sha256(video) != expected_hash:
                raise ValueError(f"{arm} video hash mismatch: {sample_id}")
        if natural_render.get("audio_sha256") != str(stage_row["natural_audio_sha256"]):
            raise ValueError(f"natural render audio binding mismatch: {sample_id}")
        candidate_row = candidate_by_id[sample_id]
        if candidate_render.get("audio_sha256") != str(candidate_row["candidate_audio_sha256"]):
            raise ValueError(f"candidate render audio binding mismatch: {sample_id}")
        records.append({
            "sample_id": sample_id,
            "source_group": str(stage_row["source_group"]),
            "real_video": str(paths["real"][0]),
            "real_video_sha256": paths["real"][1],
            "natural_video": str(paths["natural"][0]),
            "natural_video_sha256": paths["natural"][1],
            "candidate_video": str(paths["candidate"][0]),
            "candidate_video_sha256": paths["candidate"][1],
        })
    if len(records) != EXPECTED_FACE_READY_COUNT or len({str(row["source_group"]) for row in records}) != EXPECTED_GROUP_COUNT:
        raise ValueError("external-fit visual audit cohort count/group coverage mismatch")
    return stage00, alignment, candidate, preflight, render, records


def _write_failure(output: Path, error: Exception, record_rows: list[Mapping[str, Any]], failures: list[Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_external_fit_visual_teacher_audit_failure",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "error_type": type(error).__name__,
        "error": str(error),
        "observed_record_count": len(record_rows),
        "observed_records": list(record_rows),
        "failure_count": len(failures),
        "failures": list(failures),
    }
    write_json(output / "visual_teacher_failure.json", payload)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "input_record_count": EXPECTED_FACE_READY_COUNT,
        "feature_record_count": len(record_rows),
        "failure_count": len(failures),
        "eligible_record_count": 0,
        "minimum_eligible_record_count": MINIMUM_ELIGIBLE_RECORD_COUNT,
        "next_allowed_stage": None,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", {**summary, "reason": f"visual extraction did not complete: {type(error).__name__}: {error}"})
    return summary


def run(output_dir: Path, *, workers: int = 1) -> dict[str, Any]:
    output = output_dir.resolve()
    if output != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(f"external-fit visual audit output must be canonical: {EXPECTED_OUTPUT_DIR}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty external-fit visual audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    try:
        stage00, alignment, candidate, preflight, render, records = _validate_inputs()
        if not LANDMARKER_ASSET.is_file():
            raise FileNotFoundError(LANDMARKER_ASSET)
        asset_sha256 = file_sha256(LANDMARKER_ASSET)
        asset_snapshot = output / "snapshots/mediapipe_landmarker.task"
        asset_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LANDMARKER_ASSET, asset_snapshot)
        if file_sha256(asset_snapshot) != asset_sha256:
            raise ValueError("MediaPipe landmarker asset snapshot hash mismatch")
        record_rows, failures = _extract_records(records, asset_snapshot, output / "features", output / "records", workers)
        if failures:
            statistics: dict[str, Any] = {
                "complete": False,
                "group_statistics": {"group_rows": [], "group_count": 0},
                "missing_eligible_groups": list(stage00["split"]["effective_fit_groups"]),
            }
            engineering, scientific, reason = "BLOCKED", "not_available", "one or more face-ready records failed visual extraction"
        else:
            effective_groups = [str(value) for value in stage00["split"]["effective_fit_groups"]]
            eligible_count = sum(bool(row["eligibility"]["eligible"]) for row in record_rows)
            statistics = _statistics(record_rows, effective_groups)
            engineering, scientific, reason = _decision(statistics, eligible_count, MINIMUM_ELIGIBLE_RECORD_COUNT)
        eligible_count = sum(bool(row.get("eligibility", {}).get("eligible")) for row in record_rows)
        effective_groups = [str(value) for value in stage00["split"]["effective_fit_groups"]]
        manifest = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa3_external_fit_visual_teacher_audit",
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "complete" if engineering == "GO" else "blocked",
            "engineering_decision": engineering,
            "scientific_decision": scientific,
            "reason": reason,
            "parents": {
                "stage00": {"path": str(STAGE00_PATH.resolve()), "sha256": file_sha256(STAGE00_PATH)},
                "alignment": {"path": str(ALIGNMENT_PATH.resolve()), "sha256": file_sha256(ALIGNMENT_PATH)},
                "candidate": {"path": str(CANDIDATE_PATH.resolve()), "sha256": file_sha256(CANDIDATE_PATH)},
                "face_preflight": {"path": str(FACE_PREFLIGHT_PATH.resolve()), "sha256": file_sha256(FACE_PREFLIGHT_PATH)},
                "render_protocol": {"path": str(RENDER_PROTOCOL_PATH.resolve()), "sha256": file_sha256(RENDER_PROTOCOL_PATH)},
                "render": {"path": str(RENDER_PATH.resolve()), "sha256": file_sha256(RENDER_PATH)},
            },
            "landmarker_asset": {
                "path": str(LANDMARKER_ASSET.resolve()),
                "sha256": asset_sha256,
                "snapshot": str(asset_snapshot.resolve()),
            },
            "cohort": {
                "input_clean_record_count": EXPECTED_CLEAN_COUNT,
                "face_ready_record_count": EXPECTED_FACE_READY_COUNT,
                "record_count": len(records),
                "feature_record_count": len(record_rows),
                "source_group_count": len(effective_groups),
                "effective_fit_groups": effective_groups,
                "ordered_sample_ids": [str(row["sample_id"]) for row in records],
                "ordered_sample_ids_sha256": canonical_sha256([str(row["sample_id"]) for row in records]),
            },
            "extraction_workers": workers,
            "visual_contract": {
                "primary": "natural_clock_exact_time_canonical_mouth_landmark_distance",
                "primary_benefit": "D_exact(G,V_N)-D_exact(G,V_M)",
                "secondary": "visual_only_constrained_dtw_canonical_mouth_shape_distance",
                "dtw_band_fraction": DTW_BAND_FRACTION,
                "dtw_max_run": DTW_MAX_RUN,
                "valid_fraction_min": MIN_VALID_FRACTION,
                "exact_time_coverage_min": MIN_EXACT_COVERAGE,
                "minimum_eligible_record_count": MINIMUM_ELIGIBLE_RECORD_COUNT,
                "minimum_eligible_ratio": {"numerator": 44, "denominator": 48},
                "dtw_can_rescue_primary": False,
                "syncnet_used_for_selection": False,
                "syncnet_used_for_decision": False,
                "audio_scores_used": False,
            },
            "statistics": statistics,
            "eligible_record_count": eligible_count,
            "failures": failures,
            "records": record_rows,
            "selection": {
                "rule": "all face-ready records from the frozen clean MFA3 screen",
                "face_preflight_structural_only": True,
                "score_based_selection": False,
                "visual_metric_used_for_selection": False,
                "syncnet_used_for_selection": False,
                "audio_quality_used_for_selection": False,
                "alignment_result_used_for_selection": False,
                "source_order_preserved": True,
                "sample_substitution_allowed": False,
            },
            "media_access": {
                "fit_video_opened": True,
                "features_created": bool(record_rows),
                "scores_created": False,
                "internal_dev_media_opened": False,
                "validation_media_opened": False,
                "test_media_opened": False,
                "syncnet_scores_created": False,
            },
        }
        json.dumps(manifest, allow_nan=False)
        write_json(output / "visual_teacher_manifest.json", manifest)
        (output / "visual_teacher_manifest.sha256").write_text(file_sha256(output / "visual_teacher_manifest.json") + "\n", encoding="utf-8")
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": manifest["status"],
            "engineering_decision": engineering,
            "scientific_decision": scientific,
            "input_record_count": EXPECTED_FACE_READY_COUNT,
            "feature_record_count": len(record_rows),
            "failure_count": len(failures),
            "eligible_record_count": eligible_count,
            "minimum_eligible_record_count": MINIMUM_ELIGIBLE_RECORD_COUNT,
            "effective_fit_group_count": len(effective_groups),
            "primary_mean": statistics.get("primary", {}).get("sign_flip", {}).get("mean"),
            "primary_p_value": statistics.get("primary", {}).get("sign_flip", {}).get("p_value"),
            "next_allowed_stage": "06_external_fit_artifact_review_retry7" if scientific in ("GO", "NO_GO") else None,
            "media_access": manifest["media_access"],
        }
        write_json(output / "summary.json", summary)
        write_json(output / "decision.json", {**summary, "reason": reason})
        return summary
    except Exception as exc:
        return _write_failure(output, exc, record_rows, failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPECTED_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run(args.output, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
