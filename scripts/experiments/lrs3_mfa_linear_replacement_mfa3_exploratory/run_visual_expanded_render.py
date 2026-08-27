from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    PROTOCOL_ID,
    file_sha256,
    load_json,
    validate_stage00,
    write_json,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_strict_replacement import (
    _assert_gpu_ready,
    _render,
)

REPO = Path(__file__).resolve().parents[3]
STAGE_ID = "03_visual_expanded_wav2lip_render_retry1"
STAGE00_ID = "00_protocol_lock_expanded_retry3"
STAGE00_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / STAGE00_ID
    / "manifest.json"
)
CANDIDATE_STAGE_ID = "02_candidate_audio_visual_expanded_retry3"
CANDIDATE_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / CANDIDATE_STAGE_ID
    / "candidate_manifest.json"
)
ALIGNMENT_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / "01_mfa3_screen_expanded_retry2"
    / "alignment_manifest.json"
)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _validate_inputs(
    stage00_path: Path,
    candidate_path: Path,
    alignment_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if stage00_path != STAGE00_PATH.resolve():
        raise ValueError("visual-expanded render requires the frozen expanded Stage00 path")
    if candidate_path != CANDIDATE_PATH.resolve():
        raise ValueError("visual-expanded render requires the frozen retry3 candidate path")
    if alignment_path != ALIGNMENT_PATH.resolve():
        raise ValueError("visual-expanded render requires the frozen expanded alignment path")

    stage00 = load_json(stage00_path)
    validate_stage00(
        stage00_path,
        stage00,
        expected_stage_id=STAGE00_ID,
        expected_path=STAGE00_PATH,
    )
    candidate = load_json(candidate_path)
    if (
        candidate.get("protocol_id") != PROTOCOL_ID
        or candidate.get("stage_id") != CANDIDATE_STAGE_ID
        or candidate.get("status") != "complete"
        or candidate.get("engineering_decision") != "GO"
        or candidate.get("record_count") != 62
    ):
        raise ValueError("candidate manifest is not the complete 62-record retry3 artifact")
    if candidate.get("assets") != stage00.get("assets") or candidate.get("candidate_contract") != stage00.get("candidate_contract"):
        raise ValueError("candidate contract/assets do not match expanded Stage00")

    alignment = load_json(alignment_path)
    if (
        alignment.get("protocol_id") != PROTOCOL_ID
        or alignment.get("stage_id") != "01_mfa3_screen_expanded_retry2"
        or alignment.get("clean_record_count") != 62
        or len(alignment.get("records", [])) != 62
    ):
        raise ValueError("expanded alignment does not contain the frozen 62-record clean cohort")
    candidate_ids = [str(row["sample_id"]) for row in candidate.get("results", [])]
    alignment_ids = [str(row["sample_id"]) for row in alignment["records"]]
    if candidate_ids != alignment_ids:
        raise ValueError("candidate order does not match clean alignment order")
    return stage00, candidate, alignment


def _build_render_manifest(
    stage00_path: Path,
    candidate_path: Path,
    alignment_path: Path,
    stage00: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    parents = stage00.get("parents")
    if not isinstance(parents, Mapping) or not isinstance(parents.get("source_manifest"), Mapping):
        raise ValueError("expanded Stage00 source manifest binding is missing")
    source_path = _resolve_repo_path(str(parents["source_manifest"]["path"]))
    if source_path != Path(str(parents["source_manifest"]["path"])).resolve():
        raise ValueError("source manifest path normalization mismatch")
    if file_sha256(source_path) != str(parents["source_manifest"]["sha256"]):
        raise ValueError("source manifest hash mismatch")
    source = load_json(source_path)
    source_by_id = {str(row["sample_id"]): row for row in source.get("records", [])}
    stage_by_id = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
    records: list[dict[str, Any]] = []
    for item in candidate["results"]:
        sample_id = str(item["sample_id"])
        stage_row = stage_by_id.get(sample_id)
        source_row = source_by_id.get(sample_id)
        if stage_row is None or source_row is None:
            raise ValueError(f"source/stage record missing: {sample_id}")
        source_group = str(source_row["source_group"])
        if source_group != str(stage_row["source_group"]):
            raise ValueError(f"source group mismatch: {sample_id}")
        face = _resolve_repo_path(str(source_row["video_local_path"]))
        natural = Path(str(stage_row["natural_audio"])).resolve()
        candidate_audio = Path(str(item["candidate_audio"])).resolve()
        if not face.is_file() or file_sha256(face) != str(source_row["video_sha256"]):
            raise ValueError(f"source video hash mismatch: {sample_id}")
        if not natural.is_file() or file_sha256(natural) != str(stage_row["natural_audio_sha256"]):
            raise ValueError(f"natural audio hash mismatch: {sample_id}")
        if not candidate_audio.is_file() or file_sha256(candidate_audio) != str(item["candidate_audio_sha256"]):
            raise ValueError(f"candidate audio hash mismatch: {sample_id}")
        if int(item["natural_samples"]) != int(stage_row["natural_audio_samples"]) or int(item["candidate_samples"]) != int(stage_row["natural_audio_samples"]):
            raise ValueError(f"audio length identity mismatch: {sample_id}")
        if int(item.get("mapping", {}).get("speech_fallback_frames", -1)) != 0:
            raise ValueError(f"candidate speech fallback detected: {sample_id}")
        records.append(
            {
                "sample_id": sample_id,
                "source_group": source_group,
                "transcript": str(stage_row["transcript"]),
                "face_video": str(face),
                "face_video_sha256": str(source_row["video_sha256"]),
                "natural_audio": str(natural),
                "natural_audio_sha256": str(stage_row["natural_audio_sha256"]),
                "candidate_audio": str(candidate_audio),
                "candidate_audio_sha256": str(item["candidate_audio_sha256"]),
                "natural_samples": int(stage_row["natural_audio_samples"]),
                "candidate_samples": int(item["candidate_samples"]),
            }
        )
    groups = {str(row["source_group"]) for row in records}
    if len(records) != 62 or len(groups) != 23:
        raise ValueError("render cohort count/group coverage mismatch")
    return {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_visual_expanded_wav2lip_render",
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "locked",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "parents": {
            "stage00": {"path": str(stage00_path), "sha256": file_sha256(stage00_path)},
            "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
            "alignment": {"path": str(alignment_path), "sha256": file_sha256(alignment_path)},
            "source_manifest": {"path": str(source_path), "sha256": file_sha256(source_path)},
        },
        "cohort": {
            "record_count": len(records),
            "source_group_count": len(groups),
            "records": records,
        },
        "selection": {
            "rule": "all records from the pre-score frozen clean MFA3 screen",
            "score_based_selection": False,
            "visual_metric_used_for_selection": False,
            "syncnet_used_for_selection": False,
            "source_order_preserved": True,
        },
        "render_contract": {
            "arms": ["natural", "candidate"],
            "renderer": "frozen Wav2Lip inference.py",
            "face_det_batch_size": 4,
            "wav2lip_batch_size": 4,
            "nosmooth": True,
            "strict_mux_or_syncnet_run": False,
        },
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }


def run(
    stage00_path: Path = STAGE00_PATH,
    candidate_path: Path = CANDIDATE_PATH,
    alignment_path: Path = ALIGNMENT_PATH,
    output_dir: Path = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_visual_expanded_wav2lip_render_retry1",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty visual render output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        stage00_path = stage00_path.resolve()
        candidate_path = candidate_path.resolve()
        alignment_path = alignment_path.resolve()
        stage00, candidate, alignment = _validate_inputs(stage00_path, candidate_path, alignment_path)
        manifest = _build_render_manifest(stage00_path, candidate_path, alignment_path, stage00, candidate)
        gpu_gate = _assert_gpu_ready()
        renders = _render(manifest, output_dir)
        if len(renders) != 124:
            raise ValueError(f"expected 124 rendered videos, found {len(renders)}")
        write_json(output_dir / "protocol_manifest.json", manifest)
        (output_dir / "protocol_manifest.sha256").write_text(file_sha256(output_dir / "protocol_manifest.json") + "\n", encoding="utf-8")
        write_json(
            output_dir / "render_manifest.json",
            {
                "schema_version": 1,
                "manifest_type": "lrs3_mfa3_visual_expanded_wav2lip_render_results",
                "stage_id": STAGE_ID,
                "protocol_id": PROTOCOL_ID,
                "record_count": 62,
                "render_count": len(renders),
                "renders": renders,
                "gpu_gate": gpu_gate,
                "parents": manifest["parents"],
            },
        )
        summary = {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "record_count": 62,
            "render_count": len(renders),
            "expected_render_count": 124,
            "arms": ["natural", "candidate"],
            "gpu_used": True,
            "syncnet_used": False,
            "media_access": manifest["media_access"],
        }
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "decision.json",
            {
                **summary,
                "next_allowed_stage": "01_visual_teacher_audit",
                "reason": "62-record natural/candidate frozen Wav2Lip renders completed without SyncNet scoring",
            },
        )
        return summary
    except Exception as exc:
        partial = len(list((output_dir / "renders").glob("*/*.mp4")))
        payload = {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "error": str(exc),
            "partial_render_count": partial,
            "gpu_used": partial > 0,
            "syncnet_used": False,
        }
        write_json(output_dir / "failure.json", payload)
        write_json(output_dir / "summary.json", payload)
        write_json(output_dir / "decision.json", {**payload, "next_allowed_stage": None})
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, default=STAGE00_PATH)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_visual_expanded_wav2lip_render_retry1",
    )
    args = parser.parse_args()
    result = run(args.stage00, args.candidate, args.alignment, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
