from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer
from scripts.experiments.wav2lip_roi_retiming_oracle.common import extract_pcm_from_media, read_pcm16_wav

from . import analysis, config, media, protocol, validate
from .common import ExperimentError, bytes_sha256, canonical_sha256, file_sha256, verify_self_hashed_json, write_self_hashed_json


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def _load(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_runtime() -> dict[str, Any]:
    if not config.WAV2LIP_PYTHON.is_file():
        raise ExperimentError(f"Wav2Lip interpreter is missing: {config.WAV2LIP_PYTHON}")
    if not config.WAV2LIP_CHECKPOINT.is_file() or file_sha256(config.WAV2LIP_CHECKPOINT) != config.WAV2LIP_CHECKPOINT_SHA256:
        raise ExperimentError("Wav2Lip checkpoint is missing or hash changed")
    free = int(__import__("shutil").disk_usage(config.REPO).free)
    if free < config.DISK_FREE_MIN_BYTES:
        raise ExperimentError(f"free disk is below 15 GiB: {free} bytes")
    command = [
        str(config.WAV2LIP_PYTHON),
        "-c",
        "import torch; assert torch.cuda.is_available(); x=torch.ones(1024,device='cuda'); y=x+1; torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), float(y[0]))",
    ]
    result = subprocess.run(command, cwd=str(config.WAV2LIP_ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        detail = (result.stderr or result.stdout)[-2000:]
        raise ExperimentError(f"host CUDA check failed; no CPU fallback: {detail}")
    return {"free_bytes_before_generation": free, "cuda_check_command": command, "cuda_check_stdout": result.stdout.strip(), "cuda_check_stderr": result.stderr.strip(), "device": "cuda", "wav2lip_python": str(config.WAV2LIP_PYTHON)}


def _protocol_records(prepared: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = prepared["protocol"].get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ExperimentError("prepared protocol record count differs")
    return sorted(records, key=lambda row: (str(row["source_group"]), str(row["sample_id"])))


def _face_index(record: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    modes = record.get("face_modes")
    if not isinstance(modes, Mapping) or not isinstance(modes.get(mode), Mapping):
        raise ExperimentError(f"face mode is missing: {record.get('sample_id')}/{mode}")
    return modes[mode]


def _audio_index(record: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    audio = record.get("audio")
    if not isinstance(audio, Mapping) or not isinstance(audio.get("arms"), Mapping) or not isinstance(audio["arms"].get(arm), Mapping):
        raise ExperimentError(f"audio arm is missing: {record.get('sample_id')}/{arm}")
    return audio["arms"][arm]


def _generation_arm(video_arm: str) -> str:
    return {config.VIDEO_N: config.AUDIO_N, config.VIDEO_N_REPEAT: config.AUDIO_N_REPEAT, config.VIDEO_DELAY: config.AUDIO_DELAY, config.VIDEO_ADVANCE: config.AUDIO_ADVANCE}[video_arm]


def _generate_one(record: Mapping[str, Any], mode: str, video_arm: str, paths: config.RunPaths, runtime: Mapping[str, Any]) -> dict[str, Any]:
    sid = str(record["sample_id"])
    face = _face_index(record, mode)
    audio = _audio_index(record, _generation_arm(video_arm))
    output = paths.root / "videos" / f"{sid}__{mode}__{video_arm}.mkv"
    work = paths.root / "videos" / "work" / f"{sid}__{mode}__{video_arm}"
    work.mkdir(parents=True, exist_ok=True)
    audio_plan = work / "audio.json"
    output_plan = work / "outputs.json"
    _atomic_json(audio_plan, {"audio": str(Path(str(audio["path"])).resolve())})
    _atomic_json(output_plan, {"audio": str(output.resolve())})
    log = paths.root / "videos" / "logs" / f"{sid}__{mode}__{video_arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.WAV2LIP_PYTHON),
        str(config.GENERATION_WORKER),
        "--face", str(Path(str(face["video"]["path"])).resolve()),
        "--boxes", str(Path(str(face["boxes"]["path"])).resolve()),
        "--audio-json", str(audio_plan),
        "--outputs-json", str(output_plan),
        "--checkpoint", str(config.WAV2LIP_CHECKPOINT),
        "--ffmpeg", str(config.FFMPEG),
        "--batch-size", str(config.GENERATION_BATCH_SIZE),
        "--expected-frame-count", str(record["frame_count"]),
    ]
    if output.is_file() and (work / "generation_result.json").is_file():
        generation = verify_self_hashed_json(work / "generation_result.json")
        item = generation.get("rows", {}).get("audio") if isinstance(generation.get("rows"), Mapping) else None
        if isinstance(item, Mapping) and str(item.get("output_sha256")) == file_sha256(output) and item.get("device") == "cuda":
            return {"sample_id": sid, "source_group": str(record["source_group"]), "face_mode": mode, "video_arm": video_arm, "generation_audio_arm": _generation_arm(video_arm), "output": str(output.resolve()), "output_sha256": file_sha256(output), "frame_count": int(record["frame_count"]), "face": dict(face["video"]), "boxes": dict(face["boxes"]), "audio": dict(audio), "generation_result": str((work / "generation_result.json").resolve()), "generation_result_sha256": file_sha256(work / "generation_result.json"), "command": command, "log": str(log.resolve()), "device": "cuda", "batch_size": config.GENERATION_BATCH_SIZE, "resumed": True}
        raise ExperimentError(f"existing generation artifact does not match frozen binding: {output}")
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = "1"
    env["NUMBA_CACHE_DIR"] = str(work / "numba_cache")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(work), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ExperimentError(f"Wav2Lip generation failed: {sid}/{mode}/{video_arm}; see {log}")
    generation_path = work / "generation_result.json"
    generation = verify_self_hashed_json(generation_path)
    item = generation.get("rows", {}).get("audio") if isinstance(generation.get("rows"), Mapping) else None
    if not isinstance(item, Mapping) or item.get("device") != "cuda":
        raise ExperimentError(f"Wav2Lip did not run on CUDA: {sid}/{mode}/{video_arm}")
    if not output.is_file() or str(item.get("output_sha256")) != file_sha256(output):
        raise ExperimentError(f"generated video binding is incomplete: {sid}/{mode}/{video_arm}")
    expected_face = np.asarray([], dtype=np.uint8)
    # The worker already verifies the source face/box hashes; use ffmpeg decode here
    # to freeze the generated frame/timeline identity for the manifest.
    decoded, evidence = media.decode_frames(output)
    if decoded.shape[0] != int(record["frame_count"]):
        raise ExperimentError(f"generated frame count changed: {sid}/{mode}/{video_arm}")
    return {"sample_id": sid, "source_group": str(record["source_group"]), "face_mode": mode, "video_arm": video_arm, "generation_audio_arm": _generation_arm(video_arm), "output": str(output.resolve()), "output_sha256": file_sha256(output), "frame_count": int(decoded.shape[0]), "pixel_sha256": bytes_sha256(np.ascontiguousarray(decoded).tobytes()), "probe": evidence.get("probe"), "face": dict(face["video"]), "boxes": dict(face["boxes"]), "audio": dict(audio), "generation_result": str(generation_path.resolve()), "generation_result_sha256": file_sha256(generation_path), "command": command, "log": str(log.resolve()), "device": "cuda", "batch_size": config.GENERATION_BATCH_SIZE, "resumed": False, "runtime": dict(runtime)}


def _write_video_manifest(paths: config.RunPaths, rows: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": 1, "stage_id": "videos", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "video_count": len(rows), "rows": [dict(row) for row in rows], "runtime": dict(runtime)}
    write_self_hashed_json(paths.videos_manifest, payload)
    return verify_self_hashed_json(paths.videos_manifest)


def _video_index(payload: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            raise ExperimentError("video manifest row is malformed")
        key = (str(row["sample_id"]), str(row["face_mode"]), str(row["video_arm"]))
        if key in result:
            raise ExperimentError(f"duplicate video manifest row: {key}")
        result[key] = row
    return result


def _score_one(record: Mapping[str, Any], mode: str, video_arm: str, audio_arm: str, video: Mapping[str, Any], paths: config.RunPaths, scorer: SyncNetScorer, protocol_sha: str) -> dict[str, Any]:
    sid = str(record["sample_id"])
    audio = _audio_index(record, audio_arm)
    audio_path = Path(str(audio["path"])).resolve()
    pcm, values, params = read_pcm16_wav(audio_path)
    if params["sample_rate"] != config.SAMPLE_RATE or params["channels"] != 1 or params["sample_width"] != 2:
        raise ExperimentError(f"score audio format changed: {sid}/{audio_arm}")
    cell = config.cell_key(sid, mode, video_arm, audio_arm)
    output = paths.root / "scores" / "media" / f"{cell}.mkv"
    if output.is_file():
        raise ExperimentError(f"existing score media requires explicit hash-resume support: {output}")
    evidence = media.mux_audio(Path(str(video["output"])), audio_path, output, pcm)
    sidecar = {**evidence, "protocol_id": "wav2lip_global_shift_response", "protocol_sha256": protocol_sha, "sample_id": sid, "source_group": str(record["source_group"]), "face_mode": mode, "video_arm": video_arm, "audio_arm": audio_arm, "cell_key": cell, "audio_path": str(audio_path), "audio_sha256": str(audio["container_sha256"]), "audio_pcm_sha256": str(audio["decoded_pcm_sha256"]), "video_path": str(video["output"]), "video_sha256": str(video["output_sha256"])}
    sidecar_path = output.with_suffix(".json")
    write_self_hashed_json(sidecar_path, sidecar)
    cell_dir = paths.root / "scores" / "cells" / cell
    worker = scorer.score(output, audio_path, cell_dir, str(evidence["output_sha256"]), str(evidence["audio_pcm_sha256"]))
    worker_path = cell_dir / "worker.json"
    write_self_hashed_json(worker_path, worker)
    return {"schema_version": 1, "protocol_id": "wav2lip_global_shift_response", "protocol_sha256": protocol_sha, "sample_id": sid, "source_group": str(record["source_group"]), "face_mode": mode, "video_arm": video_arm, "audio_arm": audio_arm, "cell_key": cell, "media": str(output.resolve()), "media_sha256": file_sha256(output), "media_sidecar": str(sidecar_path.resolve()), "media_sidecar_sha256": file_sha256(sidecar_path), "audio": str(audio_path), "audio_container_sha256": file_sha256(audio_path), "audio_pcm_sha256": bytes_sha256(pcm), "audio_sample_count": int(values.size), "video": str(video["output"]), "video_sha256": file_sha256(Path(str(video["output"]))), "worker": str(worker_path.resolve()), "worker_sha256": file_sha256(worker_path), "visual": str(worker["visual"]), "visual_sha256": str(worker["visual_sha256"]), "audio_embedding": str(worker["audio_embedding"]), "audio_embedding_sha256": str(worker["audio_embedding_sha256"]), "matrix": str(worker["matrix"]), "matrix_sha256": str(worker["matrix_sha256"]), "matrix_shape": list(worker["matrix_shape"]), "new_forward": True}


def _write_scores_manifest(paths: config.RunPaths, rows: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": 1, "stage_id": "scores", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "score_count": len(rows), "scores": [dict(row) for row in rows], "scoring_runtime": dict(runtime)}
    write_self_hashed_json(paths.scores_manifest, payload)
    return verify_self_hashed_json(paths.scores_manifest)


def _run_generation_and_scores(prepared: Mapping[str, Any], paths: config.RunPaths, runtime: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_payload = prepared["protocol"]
    protocol_sha = str(protocol_payload["artifact_sha256"])
    records = _protocol_records(prepared)
    generated: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    # Stage A: generate N/N_REPEAT first; score V_N against all three audio arms
    # and V_N_REPEAT against N before touching shifted-driver generations.
    scorer = SyncNetScorer(config.SYNCNET_MODEL, device="cpu", batch_size=config.SYNCNET_BATCH_SIZE, threads=config.TORCH_THREADS)
    for record in records:
        for mode in config.FACE_MODES:
            for video_arm in (config.VIDEO_N, config.VIDEO_N_REPEAT):
                generated.append(_generate_one(record, mode, video_arm, paths, runtime))
    video_manifest = _write_video_manifest(paths, generated, runtime)
    vindex = _video_index(video_manifest)
    for record in records:
        for mode in config.FACE_MODES:
            for video_arm, audio_arms in ((config.VIDEO_N, (config.AUDIO_N, config.AUDIO_DELAY, config.AUDIO_ADVANCE)), (config.VIDEO_N_REPEAT, (config.AUDIO_N,))):
                for audio_arm in audio_arms:
                    scores.append(_score_one(record, mode, video_arm, audio_arm, vindex[(str(record["sample_id"]), mode, video_arm)], paths, scorer, protocol_sha))
    # Stage B: shifted-driver generation and the registered same/different-audio cells.
    for record in records:
        for mode in config.FACE_MODES:
            for video_arm in (config.VIDEO_DELAY, config.VIDEO_ADVANCE):
                generated.append(_generate_one(record, mode, video_arm, paths, runtime))
    video_manifest = _write_video_manifest(paths, generated, runtime)
    vindex = _video_index(video_manifest)
    for record in records:
        for mode in config.FACE_MODES:
            for video_arm, audio_arm in ((config.VIDEO_DELAY, config.AUDIO_N), (config.VIDEO_DELAY, config.AUDIO_DELAY), (config.VIDEO_ADVANCE, config.AUDIO_N), (config.VIDEO_ADVANCE, config.AUDIO_ADVANCE)):
                scores.append(_score_one(record, mode, video_arm, audio_arm, vindex[(str(record["sample_id"]), mode, video_arm)], paths, scorer, protocol_sha))
    score_manifest = _write_scores_manifest(paths, scores, {"python": str(config.SYNCNET_PYTHON), "device": "cpu", "batch_size": config.SYNCNET_BATCH_SIZE, "threads": config.TORCH_THREADS, "model": str(config.SYNCNET_MODEL), "model_sha256": config.SYNCNET_MODEL_SHA256})
    if len(generated) != config.EXPECTED_VIDEO_COUNT or len(scores) != config.EXPECTED_SCORE_COUNT:
        raise ExperimentError(f"fresh count mismatch: videos={len(generated)}, scores={len(scores)}")
    return video_manifest, score_manifest


def _write_review(paths: config.RunPaths, *, independent: bool = False) -> dict[str, Any]:
    payload = {"schema_version": 1, "status": "complete", "reviewer": "independent-validator" if independent else "self-review", "method": "focused contract tests, OpenSpec strict validation, independent artifact validator", "scope": ["PCM shift direction/length/no-wrap", "face mode/frame-box binding", "absolute historical track support", "matrix endpoint reconstruction", "cell denominator and hash binding"], "limitations": ["No subagent review was claimed", "Scientific interpretation remains diagnostic only"]}
    write_self_hashed_json(paths.review, payload)
    return verify_self_hashed_json(paths.review)


def _result_markdown(final: Mapping[str, Any], analysis_payload: Mapping[str, Any]) -> str:
    lines = ["# Wav2Lip global shift response diagnostic", "", f"- Engineering: `{final.get('engineering_decision')}`", f"- Diagnostic: `{final.get('diagnostic_decision')}`", f"- Scientific: `{final.get('scientific_decision')}`", f"- Generated videos: `{final.get('counts', {}).get('generated_videos', 0)}`; scored cells: `{final.get('counts', {}).get('score_cells', 0)}`", "", "本轮仅是 seen-fit 的全局偏移机制诊断；不授权训练、不建立泛化、不声称纯 mouth leakage，也不修复历史 gate。"]
    for mode in config.FACE_MODES:
        item = analysis_payload.get("face_modes", {}).get(mode, {})
        lines.append(f"- {mode}: repeat=`{item.get('flags', {}).get('repeat_stable')}`, audio-shift=`{item.get('flags', {}).get('audio_shift_detectable')}`, generated-shift=`{item.get('flags', {}).get('generated_shift_follows')}`")
        for shift in (config.VIDEO_DELAY, config.VIDEO_ADVANCE):
            gain = item.get("comparisons", {}).get(shift, {}).get("I", {})
            lines.append(f"  - {shift}: ΔC={gain.get('C')}, ΔD={gain.get('D')}, Δanchor={gain.get('anchor')}, offsetΔ={gain.get('offset_delta')}")
    lines.extend(["", "下一步固定为 `STOP_AND_REVIEW`；语义辅助实验另立 spec。", ""])
    return "\n".join(lines)


def _blocked(paths: config.RunPaths, stage: str, reason: str, counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    if paths.final.is_file():
        return _load(paths.final)
    payload = {"schema_version": 1, "protocol_id": "wav2lip_global_shift_response", "protocol_revision": "global_shift_response_v1", "status": "blocked", "engineering_decision": "BLOCKED", "diagnostic_decision": "INCOMPLETE", "scientific_decision": "not_available", "blocked_stage": stage, "block_reason": str(reason), "counts": dict(counts or {}), "training_authorized": False, "generalization_established": False, "historical_gate_repaired": False, "next_action": "STOP_AND_REVIEW"}
    paths.result.write_text(f"# Wav2Lip global shift response diagnostic\n\n- Engineering: `BLOCKED`\n- Stage: `{stage}`\n- Reason: {reason}\n", encoding="utf-8")
    payload["result_sha256"] = file_sha256(paths.result)
    write_self_hashed_json(paths.final, payload)
    return _load(paths.final)


def run(run_id: str, stage: str, resume: bool = False) -> dict[str, Any]:
    paths = _paths(run_id)
    if paths.final.is_file():
        raise ExperimentError(f"terminal run cannot be resumed or overwritten: {paths.final}")
    if stage == "prepare":
        prepared = protocol.prepare(run_id, paths)
        print(json.dumps({"status": "prepared", "protocol": str(paths.protocol), "history": str(paths.history)}, ensure_ascii=False), flush=True)
        return prepared["protocol"]
    if stage != "all":
        raise ValueError(f"unknown stage: {stage}")
    if resume and paths.protocol.is_file() and paths.history.is_file() and paths.input_audit.is_file() and paths.audio_manifest.is_file() and paths.faces_manifest.is_file():
        prepared = protocol.load_prepared(paths)
    else:
        prepared = protocol.prepare(run_id, paths)
    runtime = _check_runtime()
    videos, scores = _run_generation_and_scores(prepared, paths, runtime)
    analysis_payload = analysis.analyze(prepared["protocol"], prepared["history"], videos, scores, paths)
    review = _write_review(paths)
    validation_payload = validate.validate_complete(paths.root)
    validation_sha = write_self_hashed_json(paths.validation, validation_payload)
    if validation_payload.get("status") != "valid":
        raise ExperimentError(f"independent validator failed: {validation_payload}")
    final: dict[str, Any] = {"schema_version": 1, "protocol_id": "wav2lip_global_shift_response", "protocol_revision": "global_shift_response_v1", "status": "complete", "engineering_decision": "GO", "diagnostic_decision": "COMPLETE", "scientific_decision": "NOT_A_CONFIRMATION", "next_action": "STOP_AND_REVIEW", "training_authorized": False, "generalization_established": False, "historical_gate_repaired": False, "protocol_sha256": str(prepared["protocol"]["artifact_sha256"]), "history_sha256": file_sha256(paths.history), "analysis_sha256": file_sha256(paths.analysis), "validation_sha256": validation_sha, "review_sha256": file_sha256(paths.review), "visual_response_sha256": file_sha256(paths.visual_response), "endpoints_sha256": file_sha256(paths.endpoints), "counts": {"records": config.EXPECTED_RECORD_COUNT, "generated_videos": len(videos.get("rows", [])), "score_cells": len(scores.get("scores", [])), "network_forwards": len(scores.get("scores", [])), "cached_forwards": 0, "derived_face_streams": config.EXPECTED_RECORD_COUNT * len(config.FACE_MODES), "muxes": len(scores.get("scores", []))}, "flags": analysis_payload.get("flags", {}), "runtime": runtime, "diagnostic_interpretation": "All new intervals are exploratory; this is not a confirmation of replacement, mouth leakage, generalization, or a trainable head."}
    paths.result.write_text(_result_markdown(final, analysis_payload), encoding="utf-8")
    final["result_sha256"] = file_sha256(paths.result)
    write_self_hashed_json(paths.final, final)
    print(json.dumps({"status": "complete", "final": str(paths.final), "validation": validation_payload}, ensure_ascii=False), flush=True)
    return _load(paths.final)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Wav2Lip global shift response diagnostic")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args.run_id, args.stage, args.resume)
    except Exception as exc:  # noqa: BLE001
        paths = _paths(args.run_id)
        if not paths.final.exists():
            try:
                _blocked(paths, args.stage, str(exc))
            except Exception as block_exc:  # noqa: BLE001
                print(json.dumps({"status": "invalid", "error": str(exc), "block_write_error": str(block_exc)}, ensure_ascii=False), flush=True)
        else:
            print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
