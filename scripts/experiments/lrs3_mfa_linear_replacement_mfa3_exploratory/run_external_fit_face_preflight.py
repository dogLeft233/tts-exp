from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    canonical_sha256,
    file_sha256,
    load_json,
    write_json,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_external_fit_candidate_audio import (
    _assert_exploratory_gpu_ready,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_face_preflight import (
    FACE_DET_BATCH_SIZE,
    _detect_all,
    _read_frames,
)

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE_ID = "03_external_fit_face_preflight_retry4"
STAGE00_ID = "00_external_fit_supplement_protocol_retry4"
ALIGNMENT_ID = "01_external_fit_mfa3_screen_retry4"
CANDIDATE_ID = "02_external_fit_candidate_audio_retry4"
ROOT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE00_PATH = ROOT / STAGE00_ID / "manifest.json"
ALIGNMENT_PATH = ROOT / ALIGNMENT_ID / "alignment_manifest.json"
CANDIDATE_PATH = ROOT / CANDIDATE_ID / "candidate_manifest.json"
EXPECTED_OUTPUT_DIR = ROOT / STAGE_ID
EXPECTED_INPUT_COUNT = 266
EXPECTED_CLEAN_COUNT = 208
EXPECTED_GROUP_COUNT = 23
MIN_FACE_READY_RECORDS = 191


def _binding(binding: Mapping[str, Any], path: Path, label: str) -> None:
    resolved = path.resolve()
    if Path(str(binding.get("path", ""))).resolve() != resolved:
        raise ValueError(f"{label} path mismatch")
    if not resolved.is_file() or file_sha256(resolved) != str(binding.get("sha256", "")):
        raise ValueError(f"{label} hash mismatch")


def _validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    stage00 = load_json(STAGE00_PATH)
    if (
        stage00.get("protocol_id") != PROTOCOL_ID
        or stage00.get("stage_id") != STAGE00_ID
        or stage00.get("status") != "complete"
        or stage00.get("engineering_decision") != "GO"
        or stage00.get("scientific_decision") != "not_available"
        or stage00.get("next_allowed_stage") != ALIGNMENT_ID
    ):
        raise ValueError("external-fit Stage00 is not the complete retry4 engineering GO")
    stage00_companion = STAGE00_PATH.parent / "manifest.sha256"
    if stage00_companion.read_text(encoding="utf-8").strip() != file_sha256(STAGE00_PATH):
        raise ValueError("external-fit Stage00 companion hash mismatch")
    stage00_records = stage00.get("cohort", {}).get("records")
    if (
        not isinstance(stage00_records, list)
        or len(stage00_records) != EXPECTED_INPUT_COUNT
        or stage00.get("cohort", {}).get("record_count") != EXPECTED_INPUT_COUNT
        or len({str(row.get("sample_id")) for row in stage00_records}) != EXPECTED_INPUT_COUNT
        or any(row.get("protocol_split") != "fit" for row in stage00_records)
    ):
        raise ValueError("external-fit Stage00 cohort is incomplete")
    split = stage00.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("external-fit Stage00 split is missing")
    effective_groups = {str(value) for value in split.get("effective_fit_groups", [])}
    sealed_groups = {
        str(value)
        for key in ("internal_dev_groups", "validation_groups", "test_groups")
        for value in split.get(key, [])
    }
    if (
        len(effective_groups) != EXPECTED_GROUP_COUNT
        or "6W2dsnhC18Q" in effective_groups
        or effective_groups & sealed_groups
        or [len({str(value) for value in split.get(key, [])}) for key in ("internal_dev_groups", "validation_groups", "test_groups")] != [6, 6, 7]
    ):
        raise ValueError("external-fit Stage00 split contract is invalid")
    stage00_media = stage00.get("media_access")
    if not isinstance(stage00_media, Mapping) or any(
        stage00_media.get(key) is not False
        for key in ("fit_media_decoded", "internal_dev_media_opened", "validation_media_opened", "test_media_opened", "mfa_run", "features_created", "scores_created")
    ):
        raise ValueError("external-fit Stage00 media access ledger is unsafe")

    alignment = load_json(ALIGNMENT_PATH)
    if (
        alignment.get("protocol_id") != PROTOCOL_ID
        or alignment.get("stage_id") != ALIGNMENT_ID
        or alignment.get("status") != "partial"
        or alignment.get("engineering_decision") != "GO"
        or alignment.get("scientific_decision") != "not_available"
        or alignment.get("ordered_input_count") != EXPECTED_INPUT_COUNT
        or alignment.get("clean_record_count") != EXPECTED_CLEAN_COUNT
        or alignment.get("clean_source_group_count") != EXPECTED_GROUP_COUNT
        or alignment.get("missing_clean_groups") != []
        or alignment.get("minimum_eligible_record_count") != MIN_FACE_READY_RECORDS
        or alignment.get("next_allowed_stage") != CANDIDATE_ID
    ):
        raise ValueError("external-fit retry4 alignment contract does not authorize face preflight")
    alignment_companion = ALIGNMENT_PATH.parent / "alignment_manifest.sha256"
    if alignment_companion.read_text(encoding="utf-8").strip() != file_sha256(ALIGNMENT_PATH):
        raise ValueError("external-fit alignment companion hash mismatch")
    _binding(alignment.get("parents", {}).get("stage00", {}), STAGE00_PATH, "alignment Stage00")
    alignment_records = alignment.get("records")
    if not isinstance(alignment_records, list) or len(alignment_records) != EXPECTED_CLEAN_COUNT:
        raise ValueError("external-fit clean alignment records are incomplete")
    alignment_ids = [str(row.get("sample_id")) for row in alignment_records]
    stage00_ids = [str(row.get("sample_id")) for row in stage00_records]
    if len(set(alignment_ids)) != EXPECTED_CLEAN_COUNT or alignment_ids != [sample_id for sample_id in stage00_ids if sample_id in set(alignment_ids)]:
        raise ValueError("external-fit clean alignment order is not a Stage00 subset")
    if alignment.get("clean_sample_ids_sha256") != canonical_sha256(alignment_ids):
        raise ValueError("external-fit clean alignment hash mismatch")
    if len(alignment.get("failures", [])) != 115 or alignment.get("failed_record_count") != 58:
        raise ValueError("external-fit alignment failure rows are not fully retained")

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
    candidate_companion = CANDIDATE_PATH.parent / "candidate_manifest.sha256"
    if candidate_companion.read_text(encoding="utf-8").strip() != file_sha256(CANDIDATE_PATH):
        raise ValueError("external-fit candidate companion hash mismatch")
    _binding(candidate.get("parents", {}).get("stage00", {}), STAGE00_PATH, "candidate Stage00")
    _binding(candidate.get("parents", {}).get("alignment", {}), ALIGNMENT_PATH, "candidate alignment")
    if candidate.get("candidate_contract") != stage00.get("candidate_contract") or candidate.get("assets") != stage00.get("assets"):
        raise ValueError("external-fit candidate contract/assets differ from Stage00")
    candidate_results = candidate.get("results")
    if not isinstance(candidate_results, list) or [str(row.get("sample_id")) for row in candidate_results] != alignment_ids:
        raise ValueError("external-fit candidate order does not match clean alignment")
    candidate_by_id = {str(row["sample_id"]): row for row in candidate_results}
    if len(candidate_by_id) != EXPECTED_CLEAN_COUNT:
        raise ValueError("external-fit candidate IDs are not unique")
    stage00_by_id = {str(row["sample_id"]): row for row in stage00_records}
    for sample_id in alignment_ids:
        stage_row = stage00_by_id[sample_id]
        candidate_row = candidate_by_id[sample_id]
        if (
            candidate_row.get("natural_audio_sha256") != stage_row.get("natural_audio_sha256")
            or candidate_row.get("natural_samples") != stage_row.get("natural_audio_samples")
            or candidate_row.get("candidate_samples") != stage_row.get("natural_audio_samples")
            or candidate_row.get("mapping", {}).get("speech_fallback_frames") != 0
        ):
            raise ValueError(f"external-fit candidate contract mismatch: {sample_id}")
        candidate_audio = Path(str(candidate_row.get("candidate_audio", ""))).resolve()
        if not candidate_audio.is_file() or file_sha256(candidate_audio) != str(candidate_row.get("candidate_audio_sha256")):
            raise ValueError(f"external-fit candidate audio hash mismatch: {sample_id}")
    candidate_media = candidate.get("media_access")
    if not isinstance(candidate_media, Mapping) or any(
        candidate_media.get(key) is not False
        for key in ("fit_video_opened", "internal_dev_media_opened", "validation_media_opened", "test_media_opened", "syncnet_scores_created")
    ):
        raise ValueError("external-fit candidate media access ledger is unsafe")
    return stage00, alignment, candidate, alignment_records, stage00_by_id


def _write_failure(output: Path, error: Exception, results: list[Mapping[str, Any]], failures: list[Mapping[str, Any]], gpu_gate: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_external_fit_face_preflight_failure",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "error_type": type(error).__name__,
        "error": str(error),
        "observed_result_count": len(results),
        "observed_results": list(results),
        "failure_count": len(failures),
        "failures": list(failures),
        "gpu_gate": gpu_gate,
        "parents": {
            "stage00": {"path": str(STAGE00_PATH), "sha256": file_sha256(STAGE00_PATH) if STAGE00_PATH.is_file() else None},
            "alignment": {"path": str(ALIGNMENT_PATH), "sha256": file_sha256(ALIGNMENT_PATH) if ALIGNMENT_PATH.is_file() else None},
            "candidate": {"path": str(CANDIDATE_PATH), "sha256": file_sha256(CANDIDATE_PATH) if CANDIDATE_PATH.is_file() else None},
        },
    }
    write_json(output / "face_preflight_failure.json", payload)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "input_record_count": EXPECTED_CLEAN_COUNT,
        "observed_result_count": len(results),
        "processing_failure_count": len(failures),
        "gpu_used": gpu_gate is not None,
        "next_allowed_stage": None,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", {**summary, "reason": f"face preflight did not complete: {type(error).__name__}: {error}"})
    return summary


def run(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if output != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(f"external-fit face preflight output must be canonical: {EXPECTED_OUTPUT_DIR}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty external-fit face preflight output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    gpu_gate: dict[str, Any] | None = None
    try:
        stage00, alignment, candidate, alignment_records, stage00_by_id = _validate_inputs()
        gpu_gate = _assert_exploratory_gpu_ready()
        wav2lip_root = REPO / "third_party/Wav2Lip"
        if str(wav2lip_root) not in sys.path:
            sys.path.insert(0, str(wav2lip_root))
        import face_detection

        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device="cuda",
        )
        try:
            for alignment_row in alignment_records:
                sample_id = str(alignment_row["sample_id"])
                source = stage00_by_id[sample_id]
                face = Path(str(source["source_video"])).resolve()
                try:
                    if not face.is_file() or file_sha256(face) != str(source["source_video_sha256"]):
                        raise ValueError("source video hash mismatch")
                    frames = _read_frames(face)
                    failed_frames = _detect_all(detector, frames)
                    result = {
                        "sample_id": sample_id,
                        "source_group": str(source["source_group"]),
                        "face_video": str(face),
                        "face_video_sha256": str(source["source_video_sha256"]),
                        "frame_count": len(frames),
                        "failed_frame_indices": failed_frames,
                        "face_ready": not failed_frames,
                    }
                    results.append(result)
                    write_json(output / "records" / f"{sample_id}.json", result)
                except Exception as error:
                    failure = {
                        "sample_id": sample_id,
                        "source_group": str(source["source_group"]),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    failures.append(failure)
                    write_json(output / "records" / f"{sample_id}.failure.json", failure)
        finally:
            del detector
        if failures:
            raise RuntimeError("one or more face-ready source videos failed preflight")
        ready_ids = [str(row["sample_id"]) for row in results if row["face_ready"]]
        ready_groups = {str(row["source_group"]) for row in results if row["face_ready"]}
        engineering_go = len(ready_ids) >= MIN_FACE_READY_RECORDS and len(ready_groups) == EXPECTED_GROUP_COUNT
        reason = (
            "frozen Wav2Lip S3FD face preflight retained enough clean records and all effective fit groups"
            if engineering_go
            else "frozen Wav2Lip S3FD face preflight did not retain the fixed record denominator and group coverage"
        )
        manifest = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa3_external_fit_wav2lip_face_preflight",
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "complete" if engineering_go else "blocked",
            "engineering_decision": "GO" if engineering_go else "BLOCKED",
            "scientific_decision": "not_available",
            "reason": reason,
            "parents": {
                "stage00": {"path": str(STAGE00_PATH.resolve()), "sha256": file_sha256(STAGE00_PATH)},
                "alignment": {"path": str(ALIGNMENT_PATH.resolve()), "sha256": file_sha256(ALIGNMENT_PATH)},
                "candidate": {"path": str(CANDIDATE_PATH.resolve()), "sha256": file_sha256(CANDIDATE_PATH)},
            },
            "input_record_count": len(results),
            "face_ready_record_count": len(ready_ids),
            "face_failed_record_count": len(results) - len(ready_ids),
            "minimum_face_ready_record_count": MIN_FACE_READY_RECORDS,
            "face_ready_group_count": len(ready_groups),
            "expected_group_count": EXPECTED_GROUP_COUNT,
            "face_ready_sample_ids": ready_ids,
            "results": results,
            "failures": failures,
            "face_det_batch_size": FACE_DET_BATCH_SIZE,
            "gpu_gate": gpu_gate,
            "selection": {
                "rule": "retain all clean alignment records whose frozen Wav2Lip S3FD detector finds a face in every source-video frame",
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
                "internal_dev_media_opened": False,
                "validation_media_opened": False,
                "test_media_opened": False,
                "syncnet_scores_created": False,
            },
        }
        write_json(output / "face_preflight_manifest.json", manifest)
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": manifest["status"],
            "engineering_decision": manifest["engineering_decision"],
            "scientific_decision": "not_available",
            "input_record_count": len(results),
            "face_ready_record_count": len(ready_ids),
            "face_failed_record_count": len(results) - len(ready_ids),
            "face_ready_group_count": len(ready_groups),
            "minimum_face_ready_record_count": MIN_FACE_READY_RECORDS,
            "expected_group_count": EXPECTED_GROUP_COUNT,
            "gpu_used": True,
            "syncnet_used": False,
            "next_allowed_stage": "04_external_fit_wav2lip_render_retry4" if engineering_go else None,
            "media_access": manifest["media_access"],
        }
        write_json(output / "summary.json", summary)
        write_json(output / "decision.json", {**summary, "reason": reason})
        (output / "face_preflight_manifest.sha256").write_text(file_sha256(output / "face_preflight_manifest.json") + "\n", encoding="utf-8")
        return summary
    except Exception as exc:
        return _write_failure(output, exc, results, failures, gpu_gate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPECTED_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
