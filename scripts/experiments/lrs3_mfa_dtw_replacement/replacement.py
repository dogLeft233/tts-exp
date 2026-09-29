from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify

from . import config
from .diagonal import _parse_syncnet, _run_logged
from .protocol import (
    ProtocolError,
    load_json,
    sha256_file,
    write_json_once,
)


def run_replacement(stage00_dir: str | Path = config.STAGE00, diagonal_dir: str | Path = config.STAGE02, analysis_dir: str | Path = config.STAGE03, output_dir: str | Path = config.STAGE04) -> dict[str, Any]:
    stage00_root = Path(stage00_dir).resolve()
    diagonal_root = Path(diagonal_dir).resolve()
    analysis_root = Path(analysis_dir).resolve()
    output = Path(output_dir).resolve()
    decision = load_json(analysis_root / "decision.json")
    decision_hash_sidecar = load_json(analysis_root / "decision.sha256")
    decision_path = analysis_root / "decision.json"
    if decision_hash_sidecar.get("sha256") != sha256_file(decision_path):
        raise ProtocolError("diagonal decision self-hash changed")
    if decision.get("protocol_id") != config.PROTOCOL_ID or decision.get("cohort_sha256") != config.EXPECTED_COHORT_HASH:
        raise ProtocolError("diagonal decision binding is invalid")
    if decision.get("replacement_authorized") is not True or decision.get("scientific_decision") != "DTW_TFG_ADVANTAGE":
        if output.exists():
            raise ProtocolError("replacement output exists despite sealed decision")
        return {
            "schema_version": 1,
            "stage_id": "04_replacement",
            "protocol_id": config.PROTOCOL_ID,
            "status": "sealed_not_run",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "replacement_authorized": False,
            "reason": "Stage 03 did not authorize strict replacement",
        }
    stage00 = load_json(stage00_root / "manifest.json")
    diagonal = load_json(diagonal_root / "diagonal_manifest.json")
    if diagonal.get("status") != "complete" or len(diagonal.get("scores", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("diagonal matrix is incomplete")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"replacement stage is already populated: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stage00_rows = stage00["cohort"]["records"]
    diagonal_rows = diagonal["scores"]
    if [str(row["sample_id"]) for row in diagonal_rows] != [str(row["sample_id"]) for row in stage00_rows]:
        raise ProtocolError("diagonal order differs from frozen cohort")
    scores: list[dict[str, Any]] = []
    for index, (diagonal_row, source) in enumerate(zip(diagonal_rows, stage00_rows, strict=True), 1):
        sample_id = str(diagonal_row["sample_id"])
        video = Path(diagonal_row["video"])
        natural_audio = Path(source["natural_audio"]["path"])
        if not video.is_file() or sha256_file(video) != diagonal_row["video_sha256"]:
            raise ProtocolError(f"DTW video hash changed: {sample_id}")
        muxed = output / "mux" / f"{sample_id}.mkv"
        muxed.parent.mkdir(parents=True, exist_ok=True)
        verification = mux_and_verify(
            source_video=video,
            expected_audio=natural_audio,
            output_path=muxed,
            ffmpeg=str(config.FFMPEG),
            ffprobe=str(config.FFPROBE),
        )
        sync_dir = output / "syncnet" / sample_id
        pipeline_log = output / "logs" / f"{sample_id}.pipeline.log"
        score_log = output / "logs" / f"{sample_id}.score.log"
        reference = f"lrs3_mfa_dtw_replacement_{sample_id}"
        pipeline_command = [
            str(config.SYNCNET_PYTHON), "run_pipeline.py", "--videofile", str(muxed),
            "--reference", reference, "--data_dir", str(sync_dir), "--min_track", str(config.MIN_TRACK), "--overwrite",
        ]
        if _run_logged(pipeline_command, cwd=config.SYNCNET_ROOT, log_path=pipeline_log) != 0:
            raise RuntimeError(f"SyncNet pipeline failed for replacement {sample_id}")
        score_command = [
            str(config.SYNCNET_PYTHON), "run_syncnet.py", "--videofile", str(muxed),
            "--reference", reference, "--data_dir", str(sync_dir), "--initial_model", str(config.SYNCNET_MODEL),
        ]
        if _run_logged(score_command, cwd=config.SYNCNET_ROOT, log_path=score_log) != 0:
            raise RuntimeError(f"SyncNet score failed for replacement {sample_id}")
        parsed = _parse_syncnet(score_log)
        row = {
            "schema_version": 1,
            "sample_id": sample_id,
            "source_group": source["source_group"],
            "video": str(video),
            "video_sha256": diagonal_row["video_sha256"],
            "audio": str(natural_audio),
            "audio_sha256": source["natural_audio"]["sha256"],
            "muxed_file": str(muxed),
            "muxed_file_sha256": sha256_file(muxed),
            "mux_verification": verification,
            "syncnet_model_sha256": sha256_file(config.SYNCNET_MODEL),
            "min_track": config.MIN_TRACK,
            "pipeline_log": str(pipeline_log),
            "score_log": str(score_log),
            "sync_c": float(parsed["sync_c"]),
            "sync_d": float(parsed["sync_d"]),
            "av_offset": int(parsed["av_offset"]),
        }
        write_json_once(output / "scores" / f"{sample_id}.json", row)
        scores.append(row)
        print(f"OK {index}/{len(diagonal_rows)} {sample_id} C={row['sync_c']:.3f} D={row['sync_d']:.3f}", flush=True)
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa_dtw_comparison_replacement",
        "stage_id": "04_replacement",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "replacement_authorization_sha256": sha256_file(analysis_root / "decision.json"),
        "cohort_sha256": config.EXPECTED_COHORT_HASH,
        "record_count": len(scores),
        "scores": scores,
    }
    write_json_once(output / "replacement_manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "stage_id": "04_replacement",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(scores),
        "replacement_pcm_verified": True,
        "media_access": {
            "fit_media_opened": True,
            "fit_scores_created": True,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json_once(output / "summary.json", summary)
    write_json_once(output / "decision.json", {
        "schema_version": 1,
        "stage_id": "04_replacement",
        "protocol_id": config.PROTOCOL_ID,
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": "05_final",
        "reason": "all 133 DTW video plus untouched natural PCM cells passed engineering checks",
    })
    return summary


def replacement_contrast_rows(stage00: dict[str, Any], replacement: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source_rows = stage00["cohort"]["records"]
    replacement_rows = replacement["scores"]
    if len(replacement_rows) != len(source_rows):
        raise ProtocolError("replacement matrix count differs from cohort")
    contrasts = {"dtw_vs_natural": [], "linear_vs_natural": [], "dtw_vs_linear_replacement": []}
    for dtw, source in zip(replacement_rows, source_rows, strict=True):
        if dtw["sample_id"] != source["sample_id"]:
            raise ProtocolError("replacement order differs from cohort")
        natural = source["historical_scores"]["G_N_E_N"]
        linear = source["historical_scores"]["G_M_E_N"]
        common = {"sample_id": dtw["sample_id"], "source_group": source["source_group"]}
        contrasts["dtw_vs_natural"].append({**common, "delta_C": dtw["sync_c"] - natural["sync_c"], "delta_D": natural["sync_d"] - dtw["sync_d"]})
        contrasts["linear_vs_natural"].append({**common, "delta_C": linear["sync_c"] - natural["sync_c"], "delta_D": natural["sync_d"] - linear["sync_d"]})
        contrasts["dtw_vs_linear_replacement"].append({**common, "delta_C": dtw["sync_c"] - linear["sync_c"], "delta_D": linear["sync_d"] - dtw["sync_d"]})
    return contrasts
