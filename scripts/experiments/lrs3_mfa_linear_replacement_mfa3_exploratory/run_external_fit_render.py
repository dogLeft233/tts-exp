from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    canonical_sha256,
    file_sha256,
    load_json,
    write_json,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_external_fit_candidate_audio import (
    _assert_exploratory_gpu_ready,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_strict_replacement import (
    WAV2LIP_CHECKPOINT,
    WAV2LIP_INFERENCE,
    WAV2LIP_PY,
    _run_logged,
)

REPO = Path(__file__).resolve().parents[3]
PROTOCOL_ID = "lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE_ID = "04_external_fit_wav2lip_render_retry6"
STAGE00_ID = "00_external_fit_supplement_protocol_retry4"
ALIGNMENT_ID = "01_external_fit_mfa3_screen_retry4"
CANDIDATE_ID = "02_external_fit_candidate_audio_retry4"
FACE_PREFLIGHT_ID = "03_external_fit_face_preflight_retry4"
ROOT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE00_PATH = ROOT / STAGE00_ID / "manifest.json"
ALIGNMENT_PATH = ROOT / ALIGNMENT_ID / "alignment_manifest.json"
CANDIDATE_PATH = ROOT / CANDIDATE_ID / "candidate_manifest.json"
FACE_PREFLIGHT_PATH = ROOT / FACE_PREFLIGHT_ID / "face_preflight_manifest.json"
EXPECTED_OUTPUT_DIR = ROOT / STAGE_ID
EXPECTED_INPUT_COUNT = 266
EXPECTED_CLEAN_COUNT = 208
EXPECTED_FACE_READY_COUNT = 194
EXPECTED_GROUP_COUNT = 23
EXPECTED_MIN_FACE_READY_COUNT = 191


def _binding(binding: Mapping[str, Any], path: Path, label: str) -> None:
    resolved = path.resolve()
    if Path(str(binding.get("path", ""))).resolve() != resolved:
        raise ValueError(f"{label} path mismatch")
    if not resolved.is_file() or file_sha256(resolved) != str(binding.get("sha256", "")):
        raise ValueError(f"{label} hash mismatch")


def _wav_contract(path: Path, expected_samples: int, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != 16_000
            or handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getcomptype() != "NONE"
            or handle.getnframes() != expected_samples
        ):
            raise ValueError(f"{label} WAV contract mismatch: {path}")


def _validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
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
    if (STAGE00_PATH.parent / "manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(STAGE00_PATH):
        raise ValueError("external-fit Stage00 companion hash mismatch")
    stage_records = stage00.get("cohort", {}).get("records")
    if not isinstance(stage_records, list) or len(stage_records) != EXPECTED_INPUT_COUNT:
        raise ValueError("external-fit Stage00 record pool is incomplete")
    stage_by_id = {str(row["sample_id"]): row for row in stage_records}
    if len(stage_by_id) != EXPECTED_INPUT_COUNT:
        raise ValueError("external-fit Stage00 IDs are not unique")
    split = stage00.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("external-fit Stage00 split is missing")
    effective_groups = {str(value) for value in split.get("effective_fit_groups", [])}
    sealed_groups = {
        str(value)
        for key in ("internal_dev_groups", "validation_groups", "test_groups")
        for value in split.get(key, [])
    }
    if len(effective_groups) != EXPECTED_GROUP_COUNT or effective_groups & sealed_groups or "6W2dsnhC18Q" in effective_groups:
        raise ValueError("external-fit Stage00 group boundary is invalid")
    stage_media = stage00.get("media_access")
    if not isinstance(stage_media, Mapping) or any(
        stage_media.get(key) is not False
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
        or alignment.get("minimum_eligible_record_count") != 191
        or alignment.get("next_allowed_stage") != CANDIDATE_ID
    ):
        raise ValueError("external-fit alignment does not authorize render")
    if (ALIGNMENT_PATH.parent / "alignment_manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(ALIGNMENT_PATH):
        raise ValueError("external-fit alignment companion hash mismatch")
    _binding(alignment.get("parents", {}).get("stage00", {}), STAGE00_PATH, "alignment Stage00")
    alignment_records = alignment.get("records")
    if not isinstance(alignment_records, list) or len(alignment_records) != EXPECTED_CLEAN_COUNT:
        raise ValueError("external-fit clean alignment records are incomplete")
    alignment_ids = [str(row["sample_id"]) for row in alignment_records]
    stage_ids = [str(row["sample_id"]) for row in stage_records]
    if alignment_ids != [sample_id for sample_id in stage_ids if sample_id in set(alignment_ids)]:
        raise ValueError("external-fit alignment order is not the Stage00 source order")
    if alignment.get("clean_sample_ids_sha256") != canonical_sha256(alignment_ids):
        raise ValueError("external-fit alignment clean ID hash mismatch")

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
    if candidate.get("candidate_contract") != stage00.get("candidate_contract") or candidate.get("assets") != stage00.get("assets"):
        raise ValueError("external-fit candidate contract/assets differ from Stage00")
    candidate_results = candidate.get("results")
    if not isinstance(candidate_results, list) or [str(row["sample_id"]) for row in candidate_results] != alignment_ids:
        raise ValueError("external-fit candidate order does not match alignment")
    candidate_by_id = {str(row["sample_id"]): row for row in candidate_results}
    if len(candidate_by_id) != EXPECTED_CLEAN_COUNT:
        raise ValueError("external-fit candidate IDs are not unique")
    for sample_id in alignment_ids:
        stage_row = stage_by_id[sample_id]
        candidate_row = candidate_by_id[sample_id]
        candidate_audio = Path(str(candidate_row["candidate_audio"])).resolve()
        if (
            candidate_row.get("natural_audio_sha256") != stage_row.get("natural_audio_sha256")
            or candidate_row.get("natural_samples") != stage_row.get("natural_audio_samples")
            or candidate_row.get("candidate_samples") != stage_row.get("natural_audio_samples")
            or candidate_row.get("mapping", {}).get("speech_fallback_frames") != 0
            or not candidate_audio.is_file()
            or file_sha256(candidate_audio) != str(candidate_row.get("candidate_audio_sha256"))
        ):
            raise ValueError(f"external-fit candidate binding mismatch: {sample_id}")
        _wav_contract(candidate_audio, int(stage_row["natural_audio_samples"]), "candidate")

    preflight = load_json(FACE_PREFLIGHT_PATH)
    if (
        preflight.get("protocol_id") != PROTOCOL_ID
        or preflight.get("stage_id") != FACE_PREFLIGHT_ID
        or preflight.get("status") != "complete"
        or preflight.get("engineering_decision") != "GO"
        or preflight.get("scientific_decision") != "not_available"
        or preflight.get("input_record_count") != EXPECTED_CLEAN_COUNT
        or preflight.get("face_ready_record_count") != EXPECTED_FACE_READY_COUNT
        or preflight.get("face_ready_record_count") < EXPECTED_MIN_FACE_READY_COUNT
        or preflight.get("face_ready_group_count") != EXPECTED_GROUP_COUNT
    ):
        raise ValueError("external-fit face preflight does not authorize render")
    if (FACE_PREFLIGHT_PATH.parent / "face_preflight_manifest.sha256").read_text(encoding="utf-8").strip() != file_sha256(FACE_PREFLIGHT_PATH):
        raise ValueError("external-fit face preflight companion hash mismatch")
    for key, path in (("stage00", STAGE00_PATH), ("alignment", ALIGNMENT_PATH), ("candidate", CANDIDATE_PATH)):
        _binding(preflight.get("parents", {}).get(key, {}), path, f"face preflight {key}")
    selection = preflight.get("selection")
    if not isinstance(selection, Mapping) or any(
        selection.get(key) is not False
        for key in ("score_based_selection", "visual_metric_used_for_selection", "syncnet_used_for_selection", "audio_quality_used_for_selection", "alignment_result_used_for_selection")
    ):
        raise ValueError("external-fit face preflight selection is not score-free")
    preflight_rows = preflight.get("results")
    ready_ids = [str(row["sample_id"]) for row in preflight_rows if row.get("face_ready") is True] if isinstance(preflight_rows, list) else []
    if (
        not isinstance(preflight_rows, list)
        or len(preflight_rows) != EXPECTED_CLEAN_COUNT
        or [str(row["sample_id"]) for row in preflight_rows] != alignment_ids
        or len(ready_ids) != EXPECTED_FACE_READY_COUNT
        or preflight.get("face_ready_sample_ids") != ready_ids
    ):
        raise ValueError("external-fit face preflight IDs/counts are invalid")
    records: list[dict[str, Any]] = []
    for sample_id in ready_ids:
        stage_row = stage_by_id[sample_id]
        candidate_row = candidate_by_id[sample_id]
        face = Path(str(stage_row["source_video"])).resolve()
        natural = Path(str(stage_row["natural_audio"])).resolve()
        candidate_audio = Path(str(candidate_row["candidate_audio"])).resolve()
        if file_sha256(face) != str(stage_row["source_video_sha256"]):
            raise ValueError(f"source video hash mismatch: {sample_id}")
        if file_sha256(natural) != str(stage_row["natural_audio_sha256"]):
            raise ValueError(f"natural audio hash mismatch: {sample_id}")
        _wav_contract(natural, int(stage_row["natural_audio_samples"]), "natural")
        records.append({
            "sample_id": sample_id,
            "source_group": str(stage_row["source_group"]),
            "transcript": str(stage_row["transcript"]),
            "face_video": str(face),
            "face_video_sha256": str(stage_row["source_video_sha256"]),
            "natural_audio": str(natural),
            "natural_audio_sha256": str(stage_row["natural_audio_sha256"]),
            "candidate_audio": str(candidate_audio),
            "candidate_audio_sha256": str(candidate_row["candidate_audio_sha256"]),
            "natural_samples": int(stage_row["natural_audio_samples"]),
            "candidate_samples": int(candidate_row["candidate_samples"]),
        })
    if len(records) != EXPECTED_FACE_READY_COUNT or len({str(row["source_group"]) for row in records}) != EXPECTED_GROUP_COUNT:
        raise ValueError("external-fit render cohort count/group coverage mismatch")
    return stage00, alignment, candidate, preflight, records


def _render(records: Sequence[Mapping[str, Any]], output: Path) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    renders: list[dict[str, Any]] = []
    gpu_checks: list[dict[str, Any]] = [_assert_exploratory_gpu_ready()]
    for record in records:
        sample_id = str(record["sample_id"])
        for arm, audio_key, audio_hash_key in (
            ("natural", "natural_audio", "natural_audio_sha256"),
            ("candidate", "candidate_audio", "candidate_audio_sha256"),
        ):
            video = output / "renders" / arm / f"{sample_id}.mp4"
            work = output / "wav2lip_work" / arm / sample_id
            video.parent.mkdir(parents=True, exist_ok=True)
            (work / "temp").mkdir(parents=True, exist_ok=True)
            command = [
                str(WAV2LIP_PY),
                str(WAV2LIP_INFERENCE),
                "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
                "--face", str(record["face_video"]),
                "--audio", str(record[audio_key]),
                "--outfile", str(video),
                "--face_det_batch_size", "4",
                "--wav2lip_batch_size", "4",
                "--nosmooth",
            ]
            log = output / "logs" / "wav2lip" / arm / f"{sample_id}.log"
            return_code = _run_logged(command, cwd=work, log_path=log)
            if return_code != 0 or not video.is_file():
                raise RuntimeError(f"Wav2Lip failed for {arm}/{sample_id}: exit {return_code}")
            renders.append({
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "arm": arm,
                "face_video": str(record["face_video"]),
                "face_video_sha256": str(record["face_video_sha256"]),
                "audio": str(record[audio_key]),
                "audio_sha256": str(record[audio_hash_key]),
                "natural_samples": int(record["natural_samples"]),
                "candidate_samples": int(record["candidate_samples"]),
                "video": str(video),
                "video_sha256": file_sha256(video),
                "command": command,
                "log": str(log),
            })
    return renders, gpu_checks


def _write_failure(output: Path, error: Exception, renders: Sequence[Mapping[str, Any]], gpu_checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_external_fit_wav2lip_render_failure",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "error_type": type(error).__name__,
        "error": str(error),
        "observed_render_count": len(renders),
        "observed_renders": list(renders),
        "gpu_gate_check_count": len(gpu_checks),
        "gpu_gate_checks": list(gpu_checks),
        "parents": {
            "stage00": {"path": str(STAGE00_PATH), "sha256": file_sha256(STAGE00_PATH) if STAGE00_PATH.is_file() else None},
            "alignment": {"path": str(ALIGNMENT_PATH), "sha256": file_sha256(ALIGNMENT_PATH) if ALIGNMENT_PATH.is_file() else None},
            "candidate": {"path": str(CANDIDATE_PATH), "sha256": file_sha256(CANDIDATE_PATH) if CANDIDATE_PATH.is_file() else None},
            "face_preflight": {"path": str(FACE_PREFLIGHT_PATH), "sha256": file_sha256(FACE_PREFLIGHT_PATH) if FACE_PREFLIGHT_PATH.is_file() else None},
        },
    }
    write_json(output / "render_failure.json", payload)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "expected_record_count": EXPECTED_FACE_READY_COUNT,
        "observed_render_count": len(renders),
        "gpu_used": bool(gpu_checks),
        "syncnet_used": False,
        "next_allowed_stage": None,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "decision.json", {**summary, "reason": f"Wav2Lip render did not complete: {type(error).__name__}: {error}"})
    return summary


def run(output_dir: Path) -> dict[str, Any]:
    output = output_dir.resolve()
    if output != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(f"external-fit render output must be canonical: {EXPECTED_OUTPUT_DIR}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty external-fit render output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    renders: list[dict[str, Any]] = []
    gpu_checks: list[dict[str, Any]] = []
    try:
        stage00, alignment, candidate, preflight, records = _validate_inputs()
        renders, gpu_checks = _render(records, output)
        expected_render_count = EXPECTED_FACE_READY_COUNT * 2
        if len(renders) != expected_render_count:
            raise ValueError(f"expected {expected_render_count} renders, found {len(renders)}")
        expected_keys = {(str(record["sample_id"]), arm) for record in records for arm in ("natural", "candidate")}
        actual_keys = {(str(row["sample_id"]), str(row["arm"])) for row in renders}
        if actual_keys != expected_keys:
            raise ValueError("render IDs/arms do not match face-ready cohort")
        parents = {
            "stage00": {"path": str(STAGE00_PATH.resolve()), "sha256": file_sha256(STAGE00_PATH)},
            "alignment": {"path": str(ALIGNMENT_PATH.resolve()), "sha256": file_sha256(ALIGNMENT_PATH)},
            "candidate": {"path": str(CANDIDATE_PATH.resolve()), "sha256": file_sha256(CANDIDATE_PATH)},
            "face_preflight": {"path": str(FACE_PREFLIGHT_PATH.resolve()), "sha256": file_sha256(FACE_PREFLIGHT_PATH)},
        }
        protocol = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa3_external_fit_wav2lip_render_protocol",
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "locked",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "parents": parents,
            "cohort": {
                "input_clean_record_count": EXPECTED_CLEAN_COUNT,
                "face_ready_record_count": len(records),
                "record_count": len(records),
                "source_group_count": len({str(record["source_group"]) for record in records}),
                "ordered_sample_ids": [str(record["sample_id"]) for record in records],
                "ordered_sample_ids_sha256": canonical_sha256([str(record["sample_id"]) for record in records]),
                "records": list(records),
            },
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
            "render_contract": {
                "arms": ["natural", "candidate"],
                "renderer": "frozen Wav2Lip inference.py",
                "face_det_batch_size": 4,
                "wav2lip_batch_size": 4,
                "nosmooth": True,
                "strict_mux_or_syncnet_run": False,
                "candidate_audio_exact_natural_length": True,
            },
            "assets": {
                "wav2lip_python": {"path": str(WAV2LIP_PY.resolve()), "sha256": file_sha256(WAV2LIP_PY)},
                "wav2lip_inference": {"path": str(WAV2LIP_INFERENCE.resolve()), "sha256": file_sha256(WAV2LIP_INFERENCE)},
                "wav2lip_checkpoint": {"path": str(WAV2LIP_CHECKPOINT.resolve()), "sha256": file_sha256(WAV2LIP_CHECKPOINT)},
            },
            "gpu_gate": {
                "initial": gpu_checks[0],
                "check_count": len(gpu_checks),
                "all_checks_zero_utilization": all(not any(gpu["utilization_gpu_percent"] for gpu in check["gpus"]) for check in gpu_checks),
                "all_checks_no_active_compute_processes": all(not check["active_compute_processes"] for check in gpu_checks),
            },
            "media_access": {
                "fit_video_opened": True,
                "fit_audio_opened": True,
                "internal_dev_media_opened": False,
                "validation_media_opened": False,
                "test_media_opened": False,
                "syncnet_scores_created": False,
            },
        }
        write_json(output / "render_protocol.json", protocol)
        (output / "render_protocol.sha256").write_text(file_sha256(output / "render_protocol.json") + "\n", encoding="utf-8")
        render_manifest = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa3_external_fit_wav2lip_render_results",
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "complete",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "record_count": len(records),
            "render_count": len(renders),
            "face_ready_sample_ids": [str(record["sample_id"]) for record in records],
            "renders": renders,
            "parents": parents,
            "gpu_gate": protocol["gpu_gate"],
            "media_access": protocol["media_access"],
        }
        write_json(output / "render_manifest.json", render_manifest)
        (output / "render_manifest.sha256").write_text(file_sha256(output / "render_manifest.json") + "\n", encoding="utf-8")
        summary = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "complete",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "record_count": len(records),
            "render_count": len(renders),
            "expected_render_count": expected_render_count,
            "arms": ["natural", "candidate"],
            "gpu_used": True,
            "syncnet_used": False,
            "next_allowed_stage": "05_external_fit_visual_teacher_audit_retry6",
            "media_access": protocol["media_access"],
        }
        write_json(output / "summary.json", summary)
        write_json(output / "decision.json", {**summary, "reason": "frozen Wav2Lip natural/candidate renders completed for every face-ready fit record without SyncNet scoring"})
        return summary
    except Exception as exc:
        return _write_failure(output, exc, renders, gpu_checks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPECTED_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
