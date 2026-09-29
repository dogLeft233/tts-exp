from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

from . import analysis, audio, config, media, protocol, validate
from .common import (
    ExperimentError,
    bytes_sha256,
    file_sha256,
    free_bytes,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _audio_manifest_index(paths: config.RunPaths) -> dict[str, Mapping[str, Any]]:
    payload = verify_self_hashed_json(paths.audio_manifest)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ExperimentError("audio manifest rows are missing")
    result = {str(row["sample_id"]): row for row in rows if isinstance(row, Mapping)}
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise ExperimentError("audio manifest record count differs")
    return result


def _cell_row(
    *,
    row: Mapping[str, Any],
    video_arm: str,
    audio_arm: str,
    media_info: Mapping[str, Any],
    audio_info: Mapping[str, Any],
    worker: Mapping[str, Any],
    protocol_sha256: str,
    cell_dir: Path,
) -> dict[str, Any]:
    worker_path = cell_dir / "worker.json"
    worker_sha = write_self_hashed_json(worker_path, worker)
    return {
        "schema_version": 1,
        "stage_id": "fresh_syncnet_forward",
        "protocol_id": "wav2lip_spectral_structure_replacement",
        "protocol_sha256": protocol_sha256,
        "sample_id": str(row["sample_id"]),
        "source_group": str(row["source_group"]),
        "video_arm": video_arm,
        "audio_arm": audio_arm,
        "media": str(media_info["output"]),
        "media_sha256": str(media_info["output_sha256"]),
        "media_pcm_sha256": str(media_info["audio_pcm_sha256"]),
        "audio": str(audio_info["path"]),
        "audio_container_sha256": str(audio_info["container_sha256"]),
        "audio_pcm_sha256": str(audio_info["decoded_pcm_sha256"]),
        "cell_dir": str(cell_dir.resolve()),
        "worker_result": str(worker_path.resolve()),
        "worker_result_sha256": worker_sha,
        "visual": str(worker["visual"]),
        "visual_sha256": str(worker["visual_sha256"]),
        "audio_embedding": str(worker["audio_embedding"]),
        "audio_embedding_sha256": str(worker["audio_embedding_sha256"]),
        "matrix": str(worker["matrix"]),
        "matrix_sha256": str(worker["matrix_sha256"]),
        "matrix_shape": list(worker["matrix_shape"]),
    }


def _check_host_runtime() -> dict[str, Any]:
    free = free_bytes(config.REPO)
    if free < config.DISK_FREE_MIN_BYTES:
        raise ExperimentError(f"free disk is below 15 GiB: {free} bytes")
    command = [
        str(config.WAV2LIP_PYTHON),
        "-c",
        "import torch; assert torch.cuda.is_available(); x=torch.ones(1024,device='cuda'); y=x+1; torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), float(y[0]))",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or "cuda" not in result.stdout.lower() and "tesla" not in result.stdout.lower():
        detail = (result.stderr or result.stdout)[-2000:]
        raise ExperimentError(f"host CUDA check failed; use host namespace, no CPU fallback: {detail}")
    return {"free_bytes_before_generation": free, "cuda_check_command": command, "cuda_check_stdout": result.stdout.strip(), "cuda_check_stderr": result.stderr.strip()}


def _prepare(paths: config.RunPaths, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    inputs = protocol.load_frozen_inputs()
    rows = inputs["rows"]
    audit = protocol.build_input_audit(inputs)
    audit_sha = write_self_hashed_json(paths.input_audit, audit)
    bindings = config.spec_bindings(file_sha256)
    frozen = protocol.build_protocol(run_id, inputs, audit_sha, bindings)
    protocol_sha = write_self_hashed_json(paths.protocol, frozen)
    audio_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        candidates = audio.construct_candidates(row["_natural_pcm"], row["_mfa_pcm"])
        sid = str(row["sample_id"])
        arm_meta: dict[str, Any] = {}
        for arm in (config.ARM_N, config.ARM_N_REPEAT, config.ARM_RT, config.ARM_MAG, config.ARM_ENV):
            pcm, construction = candidates[arm]
            target = paths.audio_dir / f"{sid}__{arm}.wav"
            written = audio.write_candidate_wav(target, pcm)
            arm_meta[arm] = {**written, **construction, "source_pcm_sha256": bytes_sha256(row["_natural_pcm"]), "protocol_sha256": protocol_sha}
        p_path = Path(str(row["plateau_audio"]["path"]))
        p_info = media.verify_audio(p_path, row["_p_pcm"], row["plateau_audio"]["container_sha256"])
        arm_meta[config.ARM_P] = {**p_info, "construction": "frozen plateau P audio; reused only for N/P mux control", "protocol_sha256": protocol_sha}
        diagnostics = audio.candidate_diagnostics({arm: Path(str(arm_meta[arm]["path"])) for arm in (config.ARM_RT, config.ARM_MAG, config.ARM_ENV)}, Path(str(arm_meta[config.ARM_N]["path"])), Path(str(row["mfa_linear_audio"]["path"])))
        audio_rows.append({"sample_id": sid, "source_group": str(row["source_group"]), "arms": arm_meta, "diagnostics": diagnostics})
        print(f"AUDIO_DONE {index}/{len(rows)} {sid}", flush=True)
    audio_sha = write_self_hashed_json(
        paths.audio_manifest,
        {
            "schema_version": 1,
            "stage_id": "audio",
            "protocol_id": frozen["protocol_id"],
            "protocol_sha256": protocol_sha,
            "status": "complete",
            "record_count": len(audio_rows),
            "candidate_arms": list(config.ALL_AUDIO_ARMS),
            "alpha": config.ALPHA,
            "rows": audio_rows,
        },
    )
    return frozen, rows, {"inputs": inputs, "protocol_sha256": protocol_sha, "audio_sha256": audio_sha}


def _generate_video(row: Mapping[str, Any], arm: str, audio_path: Path, output: Path, paths: config.RunPaths, *, invocation: str) -> dict[str, Any]:
    face = Path(str(row["face_video"]["path"])).resolve()
    boxes = Path(str(row["roi"]["boxes_path"])).resolve()
    if file_sha256(face) != str(row["face_video"]["sha256"]):
        raise ExperimentError(f"face input hash changed: {row['sample_id']}")
    if file_sha256(boxes) != str(row["roi"]["boxes_sha256"]):
        raise ExperimentError(f"ROI box hash changed: {row['sample_id']}")
    if file_sha256(audio_path) == "":
        raise ExperimentError(f"candidate audio is missing: {audio_path}")
    work = paths.videos_dir / "work" / str(row["sample_id"]) / invocation
    work.mkdir(parents=True, exist_ok=True)
    audio_plan = work / "audio.json"
    output_plan = work / "outputs.json"
    _atomic_text(audio_plan, json.dumps({"audio": str(audio_path.resolve())}, indent=2) + "\n")
    _atomic_text(output_plan, json.dumps({"audio": str(output.resolve())}, indent=2) + "\n")
    log = paths.videos_dir / "logs" / f"{row['sample_id']}__{arm}__{invocation}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.WAV2LIP_PYTHON), str(config.GENERATION_WORKER),
        "--face", str(face), "--boxes", str(boxes),
        "--audio-json", str(audio_plan), "--outputs-json", str(output_plan),
        "--checkpoint", str(config.WAV2LIP_CHECKPOINT), "--ffmpeg", str(config.FFMPEG),
        "--batch-size", "4", "--expected-frame-count", str(row["frame_count"]),
    ]
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = "1"
    env["NUMBA_CACHE_DIR"] = str(work / "numba_cache")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(work), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ExperimentError(f"Wav2Lip generation failed; see {log}")
    generation_path = work / "generation_result.json"
    generation = verify_self_hashed_json(generation_path)
    item = generation.get("rows", {}).get("audio")
    if not isinstance(item, Mapping) or item.get("device") != "cuda":
        raise ExperimentError(f"Wav2Lip did not run on CUDA: {row['sample_id']}/{arm}")
    if not output.is_file() or file_sha256(output) != str(item.get("output_sha256")):
        raise ExperimentError(f"generated video binding is incomplete: {row['sample_id']}/{arm}")
    video = media.video_identity(output, int(row["frame_count"]))
    return {
        "arm": arm,
        "invocation": invocation,
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "frame_count": int(row["frame_count"]),
        "pixel_sha256": video["pixel_sha256"],
        "face": str(face),
        "face_sha256": file_sha256(face),
        "boxes": str(boxes),
        "boxes_sha256": file_sha256(boxes),
        "audio": str(audio_path.resolve()),
        "audio_sha256": file_sha256(audio_path),
        "checkpoint": str(config.WAV2LIP_CHECKPOINT.resolve()),
        "checkpoint_sha256": config.WAV2LIP_CHECKPOINT_SHA256,
        "generation_result": str(generation_path.resolve()),
        "generation_result_sha256": file_sha256(generation_path),
        "command": command,
        "log": str(log.resolve()),
        "device": "cuda",
        "batch_size": 4,
        "probe": video["probe"],
    }


def _score_one(
    scorer: SyncNetScorer,
    row: Mapping[str, Any],
    video_arm: str,
    audio_arm: str,
    video_path: Path,
    audio_info: Mapping[str, Any],
    paths: config.RunPaths,
    protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pcm, _values, _params = read_pcm16_wav(Path(str(audio_info["path"])))
    cell = config.cell_key(str(row["sample_id"]), video_arm, audio_arm)
    output = paths.root / "media" / "cells" / f"{cell}.mkv"
    evidence = media.mux_audio(video_path, Path(str(audio_info["path"])), output, pcm)
    sidecar = {**evidence, "protocol_id": "wav2lip_spectral_structure_replacement", "protocol_sha256": protocol_sha256, "sample_id": row["sample_id"], "source_group": row["source_group"], "video_arm": video_arm, "audio_arm": audio_arm}
    write_self_hashed_json(output.with_suffix(".json"), sidecar)
    cell_dir = paths.scores_dir / "cells" / cell
    worker = scorer.score(output, Path(str(audio_info["path"])), cell_dir, evidence["output_sha256"], evidence["audio_pcm_sha256"])
    worker.update({"protocol_id": "wav2lip_spectral_structure_replacement", "protocol_sha256": protocol_sha256, "sample_id": row["sample_id"], "source_group": row["source_group"], "video_arm": video_arm, "audio_arm": audio_arm, "new_forward": True})
    score = _cell_row(row=row, video_arm=video_arm, audio_arm=audio_arm, media_info=evidence, audio_info=audio_info, worker=worker, protocol_sha256=protocol_sha256, cell_dir=cell_dir)
    return {"cell": cell, "media": evidence, "audio": dict(audio_info)}, score


def _run_stage(paths: config.RunPaths, rows: Sequence[Mapping[str, Any]], audio_index: Mapping[str, Mapping[str, Any]], stage: str, protocol_sha256: str, runtime: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if stage == "A":
        arms = config.STAGE_A_ARMS
    else:
        arms = config.STAGE_B_ARMS
    generated: list[dict[str, Any]] = []
    media_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    scorer = SyncNetScorer(config.SYNCNET_MODEL, device="cpu", batch_size=config.BATCH_SIZE, threads=config.TORCH_THREADS)
    for index, row in enumerate(rows, 1):
        sid = str(row["sample_id"])
        arm_audio = audio_index[sid]["arms"]
        row_media: dict[str, Any] = {"sample_id": sid, "source_group": row["source_group"], "videos": {}, "cells": {}}
        for arm in arms:
            audio_info = arm_audio[arm]
            video_path = paths.videos_dir / "generated" / f"{sid}__{arm}.mkv"
            video_info = _generate_video(row, arm, Path(str(audio_info["path"])), video_path, paths, invocation=arm)
            generated.append(video_info)
            row_media["videos"][arm] = video_info
            score_audio_arms = (config.ARM_N, config.ARM_P) if stage == "A" and arm == config.ARM_N else (config.ARM_N,)
            for audio_arm in score_audio_arms:
                audio_meta = arm_audio[audio_arm]
                media_info, score = _score_one(scorer, row, arm, audio_arm, video_path, audio_meta, paths, protocol_sha256)
                row_media["cells"][f"{arm}__{audio_arm}"] = media_info
                score_rows.append(score)
                print(f"CELL_DONE {index}/{len(rows)} {sid} {arm}/{audio_arm}", flush=True)
        media_rows.append(row_media)
        print(f"STAGE_{stage}_DONE {index}/{len(rows)} {sid}", flush=True)
    return generated, media_rows, score_rows


def _write_videos_manifest(paths: config.RunPaths, protocol_sha256: str, old: Mapping[str, Any] | None, generated: Sequence[Mapping[str, Any]], media_rows: Sequence[Mapping[str, Any]], stage: str) -> str:
    existing_rows = list(old.get("rows", [])) if isinstance(old, Mapping) and isinstance(old.get("rows"), list) else []
    all_rows = existing_rows + [dict(row) for row in media_rows]
    payload = {
        "schema_version": 1,
        "stage_id": "videos",
        "protocol_id": "wav2lip_spectral_structure_replacement",
        "protocol_sha256": protocol_sha256,
        "status": "complete",
        "completed_stage": stage,
        "record_count": len(all_rows),
        "generated_video_count": sum(len(row.get("videos", {})) for row in all_rows),
        "rows": all_rows,
        "runtime": {"generation_python": str(config.WAV2LIP_PYTHON), "device": "cuda", "batch_size": 4},
    }
    return write_self_hashed_json(paths.videos_manifest, payload)


def _write_scores_manifest(paths: config.RunPaths, protocol_sha256: str, old: Mapping[str, Any] | None, scores: Sequence[Mapping[str, Any]], stage: str) -> str:
    existing = list(old.get("scores", [])) if isinstance(old, Mapping) and isinstance(old.get("scores"), list) else []
    all_scores = existing + [dict(row) for row in scores]
    payload = {
        "schema_version": 1,
        "stage_id": "scores",
        "protocol_id": "wav2lip_spectral_structure_replacement",
        "protocol_sha256": protocol_sha256,
        "status": "complete",
        "completed_stage": stage,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "cell_count": len(all_scores),
        "scores": all_scores,
        "scoring_runtime": {"python": str(config.SYNCNET_PYTHON), "device": "cpu", "batch_size": config.BATCH_SIZE, "threads": config.TORCH_THREADS, "model_sha256": config.SYNCNET_MODEL_SHA256},
    }
    return write_self_hashed_json(paths.scores_manifest, payload)


def _result_text(final: Mapping[str, Any], analysis_payload: Mapping[str, Any] | None) -> str:
    lines = [
        "# Wav2Lip spectral structure replacement",
        "",
        f"- 工程终态：`{final.get('engineering_decision')}`",
        f"- 科学终态：`{final.get('scientific_decision')}`",
        f"- Stage A：{final.get('counts', {}).get('stage_a_generated_videos', 0)} videos / {final.get('counts', {}).get('stage_a_score_cells', 0)} cells",
        f"- Stage B：{final.get('counts', {}).get('stage_b_generated_videos', 0)} videos / {final.get('counts', {}).get('stage_b_score_cells', 0)} cells",
        "",
    ]
    if analysis_payload and "controls" in analysis_payload:
        controls = analysis_payload["controls"]
        lines.append(f"- Stage A controls：`{'PASS' if analysis_payload.get('passes') else 'FAIL'}`；baseline={controls['baseline_clarity']['count']}/22，sensitivity={controls['sensitivity']['count']}/22")
        lines.append(f"- Stage A damage CI lower：C={controls['damage']['c']['ci'][0]:.6f}，D={controls['damage']['d']['ci'][0]:.6f}")
    if analysis_payload and "gain" in analysis_payload:
        for arm in config.STAGE_B_ARMS:
            lines.append(f"- {arm} gain：`{'PASS' if analysis_payload['gain'][arm]['passes'] else 'FAIL'}`")
        lines.append(f"- MAG−ENV：C CI=[{analysis_payload['comparisons']['MAG_vs_ENV']['c']['ci'][0]:.6f},{analysis_payload['comparisons']['MAG_vs_ENV']['c']['ci'][1]:.6f}]，D CI=[{analysis_payload['comparisons']['MAG_vs_ENV']['d']['ci'][0]:.6f},{analysis_payload['comparisons']['MAG_vs_ENV']['d']['ci'][1]:.6f}]")
    lines.extend([
        "",
        "本轮是 seen-fit 的固定 U 局部 endpoint 机制探索；不等同于独立 replication，不授权训练、不证明泛化、不修复历史 bridge gate，也不执行旧 bridge runner。",
        "",
    ])
    return "\n".join(lines)


def _fixed_flags() -> dict[str, Any]:
    return {"training_authorized": False, "generalization_established": False, "historical_gate_repaired": False, "legacy_bridge_executed": False}


def _finish(paths: config.RunPaths, *, scientific_decision: str | None, engineering_decision: str, counts: Mapping[str, int], analysis_sha256: str | None, validation_sha256: str | None, blocked_reason: str | None = None, analysis_payload: Mapping[str, Any] | None = None, runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    final: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "wav2lip_spectral_structure_replacement",
        "protocol_revision": "spectral_structure_v1",
        "status": "complete" if engineering_decision != "BLOCKED" else "blocked",
        "engineering_decision": engineering_decision,
        "scientific_decision": scientific_decision,
        "classification": "seen_fit_mechanism_exploration",
        "protocol_sha256": file_sha256(paths.protocol) if paths.protocol.is_file() else None,
        "input_audit_sha256": file_sha256(paths.input_audit) if paths.input_audit.is_file() else None,
        "audio_manifest_sha256": file_sha256(paths.audio_manifest) if paths.audio_manifest.is_file() else None,
        "videos_manifest_sha256": file_sha256(paths.videos_manifest) if paths.videos_manifest.is_file() else None,
        "scores_manifest_sha256": file_sha256(paths.scores_manifest) if paths.scores_manifest.is_file() else None,
        "analysis_sha256": analysis_sha256,
        "validation_sha256": validation_sha256,
        "counts": dict(counts),
        "budget": {"max_videos": 110, "max_score_cells": 132},
        "runtime": dict(runtime or {}),
        **_fixed_flags(),
    }
    if blocked_reason:
        final["blocked_reason"] = blocked_reason
    _atomic_text(paths.result, _result_text(final, analysis_payload))
    final["result_sha256"] = file_sha256(paths.result)
    write_self_hashed_json(paths.final, final)
    return final


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    return verify_self_hashed_json(path) if path.is_file() else None


def run_experiment(run_id: str, stage: str = "all", *, resume: bool = False) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.root.exists() and any(paths.root.iterdir()) and not resume:
        raise ExperimentError(f"run root already contains artifacts; choose a new run-id or use --resume: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, Any] = {}
    try:
        if resume and paths.protocol.is_file() and paths.audio_manifest.is_file():
            frozen = verify_self_hashed_json(paths.protocol)
            inputs = protocol.load_frozen_inputs()
            rows = inputs["rows"]
            protocol_sha = file_sha256(paths.protocol)
            runtime["resumed"] = True
        else:
            frozen, rows, prep = _prepare(paths, run_id)
            inputs = prep["inputs"]
            protocol_sha = prep["protocol_sha256"]
        audio_index = _audio_manifest_index(paths)
        if stage == "prepare":
            free = free_bytes(config.REPO)
            if free < config.DISK_FREE_MIN_BYTES:
                raise ExperimentError(f"free disk is below 15 GiB: {free} bytes")
            runtime["free_bytes_before_generation"] = free
            return _finish(paths, scientific_decision=None, engineering_decision="GO", counts={"stage_a_generated_videos": 0, "stage_a_score_cells": 0, "stage_b_generated_videos": 0, "stage_b_score_cells": 0}, analysis_sha256=None, validation_sha256=None, runtime=runtime)
        runtime.update(_check_host_runtime())
        old_videos = _load_existing_manifest(paths.videos_manifest) if resume else None
        old_scores = _load_existing_manifest(paths.scores_manifest) if resume else None
        generated_a, media_a, scores_a = _run_stage(paths, rows, audio_index, "A", protocol_sha, runtime)
        videos_sha = _write_videos_manifest(paths, protocol_sha, old_videos, generated_a, media_a, "A")
        scores_sha = _write_scores_manifest(paths, protocol_sha, old_scores, scores_a, "A")
        control = analysis.analyze_stage_a(frozen, scores_a)
        control["videos_manifest_sha256"] = videos_sha
        control["scores_manifest_sha256"] = scores_sha
        control_sha = write_self_hashed_json(paths.control_analysis, control)
        control_validation = validate.validate_stage_a(paths.root)
        control_validation["control_analysis_sha256"] = control_sha
        control_validation_sha = write_self_hashed_json(paths.control_validation, control_validation)
        counts = {"stage_a_generated_videos": len(generated_a), "stage_a_score_cells": len(scores_a), "stage_b_generated_videos": 0, "stage_b_score_cells": 0}
        if not bool(control_validation.get("valid")) or not bool(control.get("passes")):
            final = _finish(paths, scientific_decision="CONTROL_FAILED", engineering_decision="GO", counts=counts, analysis_sha256=control_sha, validation_sha256=control_validation_sha, analysis_payload=control, runtime=runtime)
            return final
        if stage == "stage_a":
            final = _finish(paths, scientific_decision="STAGE_A_CONTROLS_PASSED_STAGE_B_NOT_RUN", engineering_decision="GO", counts=counts, analysis_sha256=control_sha, validation_sha256=control_validation_sha, analysis_payload=control, runtime=runtime)
            return final
        generated_b, media_b, scores_b = _run_stage(paths, rows, audio_index, "B", protocol_sha, runtime)
        videos_sha = _write_videos_manifest(paths, protocol_sha, {"rows": media_a}, generated_b, media_b, "B")
        scores_sha = _write_scores_manifest(paths, protocol_sha, {"scores": scores_a}, scores_b, "B")
        all_scores = [*scores_a, *scores_b]
        stage_b = analysis.analyze_stage_b(frozen, all_scores, control)
        stage_b["control_analysis_sha256"] = control_sha
        stage_b["videos_manifest_sha256"] = videos_sha
        stage_b["scores_manifest_sha256"] = scores_sha
        analysis_sha = write_self_hashed_json(paths.analysis, {"stage_a": control, "stage_b": stage_b, "decision": stage_b["decision"], **_fixed_flags()})
        validation = validate.validate_complete(paths.root)
        validation["analysis_sha256"] = analysis_sha
        validation_sha = write_self_hashed_json(paths.validation, validation)
        counts.update({"stage_b_generated_videos": len(generated_b), "stage_b_score_cells": len(scores_b)})
        if not bool(validation.get("valid")):
            return _finish(paths, scientific_decision=None, engineering_decision="BLOCKED", counts=counts, analysis_sha256=analysis_sha, validation_sha256=validation_sha, blocked_reason="independent complete validation failed", analysis_payload=stage_b, runtime=runtime)
        return _finish(paths, scientific_decision=str(stage_b["decision"]), engineering_decision="GO", counts=counts, analysis_sha256=analysis_sha, validation_sha256=validation_sha, analysis_payload=stage_b, runtime=runtime)
    except (ExperimentError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        completed_a = 0
        completed_b = 0
        if paths.scores_manifest.is_file():
            try:
                payload = verify_self_hashed_json(paths.scores_manifest)
                completed_a = min(int(payload.get("cell_count", 0)), config.EXPECTED_RECORD_COUNT * 4)
            except (OSError, KeyError, ValueError, ExperimentError):
                pass
        return _finish(paths, scientific_decision=None, engineering_decision="BLOCKED", counts={"stage_a_generated_videos": 0, "stage_a_score_cells": completed_a, "stage_b_generated_videos": 0, "stage_b_score_cells": completed_b}, analysis_sha256=file_sha256(paths.control_analysis) if paths.control_analysis.is_file() else None, validation_sha256=file_sha256(paths.control_validation) if paths.control_validation.is_file() else None, blocked_reason=f"{type(exc).__name__}: {exc}", runtime=runtime)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Wav2Lip spectral structure replacement experiment")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "stage_a", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_experiment(args.run_id, args.stage, resume=args.resume)
    print(json.dumps({"status": result.get("status"), "engineering_decision": result.get("engineering_decision"), "scientific_decision": result.get("scientific_decision"), "run_root": str(config.run_root_for(args.run_id))}, ensure_ascii=False))
    return 0 if result.get("engineering_decision") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
