from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify

from . import config
from .candidates import gpu_gate
from .protocol import ProtocolError, load_json, sha256_file, write_json_once


def _run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(result.returncode)


def _parse_syncnet(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([0-9.]+)", text)
    distance = re.search(r"Min dist:\s+([0-9.]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        raise ProtocolError(f"official SyncNet output is missing scores: {log_path}")
    result = {
        "sync_c": float(confidence.group(1)),
        "sync_d": float(distance.group(1)),
        "av_offset": int(offset.group(1)) if offset else 0,
    }
    if not all(math.isfinite(float(result[key])) for key in ("sync_c", "sync_d", "av_offset")):
        raise ProtocolError(f"official SyncNet output is non-finite: {log_path}")
    return result


def _validate_inputs(stage00_dir: Path, candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage00 = load_json(stage00_dir / "manifest.json")
    candidates = load_json(candidate_dir / "candidate_manifest.json")
    if stage00.get("protocol_id") != config.PROTOCOL_ID or stage00.get("cohort", {}).get("ordered_sample_ids_sha256") != config.EXPECTED_COHORT_HASH:
        raise ProtocolError("Stage 00 binding is invalid")
    if candidates.get("protocol_id") != config.PROTOCOL_ID or candidates.get("status") != "complete":
        raise ProtocolError("candidate manifest is invalid")
    if candidates.get("cohort_sha256") != config.EXPECTED_COHORT_HASH or len(candidates.get("results", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("candidate cohort binding is invalid")
    if candidates.get("dtw_contract") != stage00.get("dtw_contract"):
        raise ProtocolError("candidate DTW contract changed")
    return stage00, candidates


def run_diagonal(stage00_dir: str | Path = config.STAGE00, candidate_dir: str | Path = config.STAGE01, output_dir: str | Path = config.STAGE02) -> dict[str, Any]:
    stage00_root = Path(stage00_dir).resolve()
    candidate_root = Path(candidate_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "diagonal_manifest.json").exists():
        raise FileExistsError(f"diagonal stage is already finalized: {output}")
    stage00, candidates = _validate_inputs(stage00_root, candidate_root)
    gate = gpu_gate()
    candidate_rows = candidates["results"]
    stage00_by_id = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
    if [str(row["sample_id"]) for row in candidate_rows] != [str(row["sample_id"]) for row in stage00["cohort"]["records"]]:
        raise ProtocolError("candidate order differs from Stage 00 cohort")
    scores: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidate_rows, 1):
        sample_id = str(candidate["sample_id"])
        source = stage00_by_id[sample_id]
        candidate_audio = Path(candidate["candidate_audio"])
        if not candidate_audio.is_file() or sha256_file(candidate_audio) != candidate["candidate_audio_sha256"]:
            raise ProtocolError(f"candidate audio hash changed: {sample_id}")
        face = Path(source["face_video"]["path"])
        video = output / "videos" / f"{sample_id}.mp4"
        wav2lip_log = output / "logs" / "wav2lip" / f"{sample_id}.log"
        video.parent.mkdir(parents=True, exist_ok=True)
        if not video.exists():
            command = [
                str(config.WAV2LIP_PYTHON), str(config.WAV2LIP_ROOT / "inference.py"),
                "--checkpoint_path", str(config.WAV2LIP_CHECKPOINT),
                "--face", str(face), "--audio", str(candidate_audio), "--outfile", str(video),
                "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth",
            ]
            returncode = _run_logged(command, cwd=config.WAV2LIP_ROOT, log_path=wav2lip_log)
            if returncode != 0 or not video.is_file():
                raise RuntimeError(f"Wav2Lip failed for {sample_id}: exit {returncode}")
        video_hash = sha256_file(video)
        muxed = output / "mux" / f"{sample_id}.mkv"
        muxed.parent.mkdir(parents=True, exist_ok=True)
        verification = mux_and_verify(
            source_video=video,
            expected_audio=candidate_audio,
            output_path=muxed,
            ffmpeg=str(config.FFMPEG),
            ffprobe=str(config.FFPROBE),
        )
        sync_dir = output / "syncnet" / sample_id
        pipeline_log = output / "logs" / "syncnet" / f"{sample_id}.pipeline.log"
        score_log = output / "logs" / "syncnet" / f"{sample_id}.score.log"
        reference = f"lrs3_mfa_dtw_{sample_id}"
        pipeline_command = [
            str(config.SYNCNET_PYTHON), "run_pipeline.py", "--videofile", str(muxed),
            "--reference", reference, "--data_dir", str(sync_dir), "--min_track", str(config.MIN_TRACK), "--overwrite",
        ]
        if _run_logged(pipeline_command, cwd=config.SYNCNET_ROOT, log_path=pipeline_log) != 0:
            raise RuntimeError(f"SyncNet pipeline failed for {sample_id}")
        score_command = [
            str(config.SYNCNET_PYTHON), "run_syncnet.py", "--videofile", str(muxed),
            "--reference", reference, "--data_dir", str(sync_dir), "--initial_model", str(config.SYNCNET_MODEL),
        ]
        if _run_logged(score_command, cwd=config.SYNCNET_ROOT, log_path=score_log) != 0:
            raise RuntimeError(f"SyncNet score failed for {sample_id}")
        parsed = _parse_syncnet(score_log)
        row = {
            "schema_version": 1,
            "sample_id": sample_id,
            "source_group": source["source_group"],
            "candidate_audio": str(candidate_audio),
            "candidate_audio_sha256": candidate["candidate_audio_sha256"],
            "video": str(video),
            "video_sha256": video_hash,
            "muxed_file": str(muxed),
            "muxed_file_sha256": sha256_file(muxed),
            "wav2lip_checkpoint_sha256": sha256_file(config.WAV2LIP_CHECKPOINT),
            "syncnet_model_sha256": sha256_file(config.SYNCNET_MODEL),
            "min_track": config.MIN_TRACK,
            "wav2lip_log": str(wav2lip_log),
            "pipeline_log": str(pipeline_log),
            "score_log": str(score_log),
            "mux_verification": verification,
            "sync_c": float(parsed["sync_c"]),
            "sync_d": float(parsed["sync_d"]),
            "av_offset": int(parsed["av_offset"]),
            "historical_linear": source["historical_scores"]["G_M_E_M"],
        }
        write_json_once(output / "scores" / f"{sample_id}.json", row)
        scores.append(row)
        print(f"OK {index}/{len(candidate_rows)} {sample_id} C={row['sync_c']:.3f} D={row['sync_d']:.3f}", flush=True)
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa_dtw_comparison_diagonal",
        "stage_id": "02_diagonal",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(scores),
        "cohort_sha256": config.EXPECTED_COHORT_HASH,
        "gpu_gate": gate,
        "scores": scores,
    }
    write_json_once(output / "diagonal_manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "stage_id": "02_diagonal",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(scores),
        "score_count": len(scores),
        "media_access": {
            "fit_media_opened": True,
            "fit_features_created": False,
            "fit_scores_created": True,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json_once(output / "summary.json", summary)
    write_json_once(output / "decision.json", {
        "schema_version": 1,
        "stage_id": "02_diagonal",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": "03_diagonal_analysis",
        "reason": "133 DTW video/audio diagonal cells passed mux and official SyncNet engineering checks",
    })
    return summary
