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

from . import analysis, config, media, protocol
from .common import (
    PlateauError,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _cell_row(
    *,
    sample_id: str,
    source_group: str,
    video_arm: str,
    audio_arm: str,
    media_info: dict[str, Any],
    audio_info: dict[str, Any],
    worker: dict[str, Any],
    protocol_sha256: str,
    cell_dir: Path,
) -> dict[str, Any]:
    worker_path = cell_dir / "worker.json"
    worker_sha = write_self_hashed_json(worker_path, worker)
    return {
        "schema_version": 1,
        "stage_id": "fresh_syncnet_forward",
        "protocol_id": "wav2lip_integer_plateau_control",
        "protocol_sha256": protocol_sha256,
        "sample_id": sample_id,
        "source_group": source_group,
        "video_arm": video_arm,
        "audio_arm": audio_arm,
        "media": media_info["output"],
        "media_sha256": media_info["output_sha256"],
        "media_pcm_sha256": media_info["audio_pcm_sha256"],
        "audio": audio_info["path"],
        "audio_container_sha256": audio_info["container_sha256"],
        "audio_pcm_sha256": audio_info["decoded_pcm_sha256"],
        "cell_dir": str(cell_dir.resolve()),
        "worker_result": str(worker_path.resolve()),
        "worker_result_sha256": worker_sha,
        "visual": worker["visual"],
        "visual_sha256": worker["visual_sha256"],
        "audio_embedding": worker["audio_embedding"],
        "audio_embedding_sha256": worker["audio_embedding_sha256"],
        "matrix": worker["matrix"],
        "matrix_sha256": worker["matrix_sha256"],
        "matrix_shape": worker["matrix_shape"],
    }


def _write_result(paths: config.RunPaths, final: dict[str, Any], oracle: dict[str, Any] | None = None, generated: dict[str, Any] | None = None) -> None:
    lines = [
        "# Wav2Lip integer plateau control 2026-09-07",
        "",
        f"- 工程终态：`{final.get('engineering_decision')}`",
        f"- 科学终态：`{final.get('scientific_decision')}`",
        f"- Stage A：{final.get('counts', {}).get('stage_a_fresh_score_cells', 0)}/66 fresh cells；GPU生成视频={final.get('counts', {}).get('stage_b_generated_videos', 0)}",
        "",
    ]
    if oracle:
        lines.extend(
            [
                f"- A transport parity：{oracle['transport_parity']['pass_count']}/{oracle['transport_parity']['expected_count']}，最大差={oracle['transport_parity']['max_abs_difference']:.9f}",
                f"- A baseline/timing B-C-O：{oracle['baseline_count']}/{oracle['record_count']}；{oracle['timing_counts']['B']}/{oracle['timing_counts']['C']}/{oracle['timing_counts']['O']}",
                f"- A matched-own CI下界：C={oracle['own_bootstrap']['c']['ci95'][0]:.6f}，D={oracle['own_bootstrap']['d']['ci95'][0]:.6f}",
                f"- A damage CI下界：C={oracle['damage_bootstrap']['c']['ci95'][0]:.6f}，D={oracle['damage_bootstrap']['d']['ci95'][0]:.6f}",
            ]
        )
    if generated:
        lines.extend(
            [
                f"- B timing E_P/N-vs-V_P/N、E_P/P-vs-V_P/P：{generated['timing_counts']['E_P_N_vs_V_P_N']}/{generated['record_count']}、{generated['timing_counts']['E_P_P_vs_V_P_P']}/{generated['record_count']}",
                f"- B generated-own CI下界：C={generated['own_bootstrap']['c']['ci95'][0]:.6f}，D={generated['own_bootstrap']['d']['ci95'][0]:.6f}",
                f"- B generated-damage CI下界：C={generated['damage_bootstrap']['c']['ci95'][0]:.6f}，D={generated['damage_bootstrap']['d']['ci95'][0]:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "本轮仅验证固定整数平台控制；不修复历史 CONTROL_FAILED，不构成 replacement 收益、训练授权或跨模型泛化证据。",
            "",
        ]
    )
    paths.result.write_text("\n".join(lines), encoding="utf-8")


def _blocked(paths: config.RunPaths, reason: str, completed_cells: int = 0) -> dict[str, Any]:
    paths.root.mkdir(parents=True, exist_ok=True)
    final = {
        "schema_version": 1,
        "protocol_id": "wav2lip_integer_plateau_control",
        "protocol_revision": "integer_plateau_v1",
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": None,
        "blocked_reason": reason,
        "counts": {"stage_a_fresh_score_cells": completed_cells, "stage_a_cached_score_cells": config.EXPECTED_RECORD_COUNT, "stage_b_generated_videos": 0, "stage_b_fresh_score_cells": 0},
        **analysis._fixed_flags(),
    }
    _write_result(paths, final)
    final["result_sha256"] = file_sha256(paths.result)
    write_self_hashed_json(paths.final, final)
    return final


def _prepare_stage_a(paths: config.RunPaths, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    inputs = protocol.load_frozen_inputs()
    rows = protocol.build_parent_rows(inputs)
    audit = protocol.build_input_audit(rows)
    audit_sha = write_self_hashed_json(paths.input_audit, audit)
    bindings = config.spec_bindings(file_sha256)
    frozen = protocol.build_protocol(run_id, rows, audit_sha, bindings)
    protocol_sha = write_self_hashed_json(paths.protocol, frozen)

    audio_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    media_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    scorer = SyncNetScorer(config.SYNCNET_MODEL, device="cpu", batch_size=config.BATCH_SIZE, threads=config.TORCH_THREADS)
    for index, row in enumerate(rows, 1):
        sid = str(row["sample_id"])
        p_pcm, p_meta = media.make_plateau_pcm(row["_pcm"], row["_audio_source_indices"])
        audio_meta = media.copy_or_write_audio(Path(row["natural_audio"]["path"]), row["_pcm_bytes"], p_pcm, paths.root / "audio", sid)
        audio_rows.append({"sample_id": sid, "source_group": row["source_group"], "arms": audio_meta, "plateau": p_meta})
        source_frames = row["_frames"]
        p_frames = np.ascontiguousarray(source_frames[row["_q"]], dtype=np.uint8)
        p_video = paths.media / "streams" / f"{sid}__{config.VIDEO_P}.mkv"
        p_evidence = media.encode_video(p_frames, p_video)
        frame_rows.extend(
            [
                {"sample_id": sid, "source_group": row["source_group"], "video_arm": config.VIDEO_ID, "parent": row["parent_stream"]},
                {"sample_id": sid, "source_group": row["source_group"], "video_arm": config.VIDEO_P, "q": row["q"], "evidence": p_evidence, "pixel_identity_verified": True},
            ]
        )
        parent_vid = Path(str(row["parent_stream"]["path"])).resolve()
        cell_media: dict[str, dict[str, Any]] = {}
        for video_arm, video_path in ((config.VIDEO_ID, parent_vid), (config.VIDEO_P, p_video)):
            for audio_arm in (config.AUDIO_N, config.AUDIO_P):
                if (video_arm, audio_arm) not in config.FRESH_CELL_SPECS:
                    continue
                audio_info = audio_meta[audio_arm]
                pcm = row["_pcm_bytes"] if audio_arm == config.AUDIO_N else p_pcm
                output = paths.media / "cells" / f"{config.cell_key(sid, video_arm, audio_arm)}.mkv"
                evidence = media.mux_audio(video_path, Path(audio_info["path"]), output, pcm)
                sidecar = {**evidence, "protocol_id": "wav2lip_integer_plateau_control", "protocol_sha256": protocol_sha, "sample_id": sid, "video_arm": video_arm, "audio_arm": audio_arm}
                write_self_hashed_json(output.with_suffix(".json"), sidecar)
                cell_media[f"{video_arm}__{audio_arm}"] = {**evidence, "audio": audio_info}
                cell_dir = paths.scores / "cells" / config.cell_key(sid, video_arm, audio_arm)
                print(f"CELL_START {index}/{len(rows)} {sid} {video_arm}/{audio_arm}", flush=True)
                worker = scorer.score(Path(evidence["output"]), Path(audio_info["path"]), cell_dir, evidence["output_sha256"], evidence["audio_pcm_sha256"])
                worker.update({"protocol_id": "wav2lip_integer_plateau_control", "protocol_sha256": protocol_sha, "sample_id": sid, "source_group": row["source_group"], "video_arm": video_arm, "audio_arm": audio_arm, "new_forward": True})
                score_rows.append(_cell_row(sample_id=sid, source_group=str(row["source_group"]), video_arm=video_arm, audio_arm=audio_arm, media_info=evidence, audio_info=audio_info, worker=worker, protocol_sha256=protocol_sha, cell_dir=cell_dir))
                print(f"CELL_DONE {index}/{len(rows)} {sid} {video_arm}/{audio_arm}", flush=True)
        media_rows.append(
            {
                "sample_id": sid,
                "source_group": row["source_group"],
                "streams": {"V_ID": {**row["parent_stream"], "parent": True}, "V_P": {"output": p_evidence["output"], "output_sha256": p_evidence["output_sha256"], "frame_count": row["frame_count"], "evidence": p_evidence}},
                "cells": cell_media,
            }
        )
    frames_sha = write_self_hashed_json(paths.frames_manifest, {"schema_version": 1, "stage_id": "frame_materialization", "protocol_id": "wav2lip_integer_plateau_control", "protocol_sha256": protocol_sha, "status": "complete", "record_count": len(rows), "stream_count": len(frame_rows), "new_stream_count": len(rows), "rows": frame_rows})
    audio_sha = write_self_hashed_json(paths.audio_manifest, {"schema_version": 1, "stage_id": "audio_binding", "protocol_id": "wav2lip_integer_plateau_control", "protocol_sha256": protocol_sha, "status": "complete", "record_count": len(rows), "rows": audio_rows})
    media_sha = write_self_hashed_json(paths.media_manifest, {"schema_version": 1, "stage_id": "media", "protocol_id": "wav2lip_integer_plateau_control", "protocol_sha256": protocol_sha, "status": "complete", "record_count": len(rows), "stream_count": len(frame_rows), "new_stream_count": len(rows), "cell_count": len(score_rows), "new_cell_count": len(score_rows), "rows": media_rows})
    score_sha = write_self_hashed_json(paths.score_manifest, {"schema_version": 1, "stage_id": "score", "protocol_id": "wav2lip_integer_plateau_control", "protocol_sha256": protocol_sha, "media_manifest_sha256": media_sha, "status": "complete", "record_count": len(rows), "cell_count": len(score_rows), "expected_cell_count": len(rows) * len(config.FRESH_CELL_SPECS), "scores": score_rows})
    analysis_payload = analysis.analyze_oracle(frozen, score_rows)
    analysis_payload.update({"protocol_sha256": protocol_sha, "input_audit_sha256": audit_sha, "frames_manifest_sha256": frames_sha, "audio_manifest_sha256": audio_sha, "media_manifest_sha256": media_sha, "score_manifest_sha256": score_sha})
    write_self_hashed_json(paths.oracle_analysis, analysis_payload)
    return frozen, rows, analysis_payload, score_rows


def _run_generation(record: Mapping[str, Any], p_audio: Path, output: Path, paths: config.RunPaths) -> dict[str, Any]:
    roi_record = record["_roi_record"]
    face = Path(str(roi_record["face_video"]["path"])).resolve()
    boxes = Path(str(roi_record["roi"]["boxes_path"])).resolve()
    if not face.is_file() or file_sha256(face) != str(roi_record["face_video"]["sha256"]):
        raise PlateauError(f"face input hash changed: {record['sample_id']}")
    if not boxes.is_file() or file_sha256(boxes) != str(roi_record["roi"]["boxes_sha256"]):
        raise PlateauError(f"ROI box hash changed: {record['sample_id']}")
    work = paths.videos / "work" / str(record["sample_id"])
    work.mkdir(parents=True, exist_ok=True)
    audio_plan = work / "audio.json"
    output_plan = work / "outputs.json"
    audio_plan.write_text(json.dumps({"P": str(p_audio.resolve())}, indent=2) + "\n", encoding="utf-8")
    output_plan.write_text(json.dumps({"P": str(output.resolve())}, indent=2) + "\n", encoding="utf-8")
    log = paths.videos / "logs" / f"{record['sample_id']}__G_P.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(config.WAV2LIP_PYTHON), str(config.GENERATION_WORKER), "--face", str(face), "--boxes", str(boxes), "--audio-json", str(audio_plan), "--outputs-json", str(output_plan), "--checkpoint", str(config.WAV2LIP_CHECKPOINT), "--ffmpeg", str(config.FFMPEG), "--batch-size", "4", "--expected-frame-count", str(record["frame_count"])]
    env = dict(os.environ)
    env["NUMBA_DISABLE_JIT"] = "1"
    env["NUMBA_CACHE_DIR"] = str(work / "numba_cache")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(work), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise PlateauError(f"Wav2Lip generation failed; see {log}")
    generation_result = verify_self_hashed_json(work / "generation_result.json")
    item = generation_result.get("rows", {}).get("P")
    if not isinstance(item, dict) or not output.is_file() or file_sha256(output) != str(item.get("output_sha256")):
        raise PlateauError(f"generated G_P artifact is incomplete: {record['sample_id']}")
    frames, evidence = media.decode_frames(output)
    if frames.shape[0] != int(record["frame_count"]) or frames.shape[1:] != (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3):
        raise PlateauError(f"generated G_P frame contract failed: {record['sample_id']}")
    return {"output": str(output.resolve()), "output_sha256": file_sha256(output), "frame_count": int(frames.shape[0]), "face": str(face), "face_sha256": file_sha256(face), "boxes": str(boxes), "boxes_sha256": file_sha256(boxes), "checkpoint_sha256": config.WAV2LIP_CHECKPOINT_SHA256, "generation_result": str((work / "generation_result.json").resolve()), "generation_result_sha256": file_sha256(work / "generation_result.json"), "command": command, "log": str(log.resolve()), "probe": evidence["probe"]}


def _run_stage_b(paths: config.RunPaths, frozen: dict[str, Any], rows: list[dict[str, Any]], oracle_score_rows: list[dict[str, Any]], inputs: dict[str, Any]) -> dict[str, Any]:
    roi_index = {str(row["sample_id"]): row for row in inputs["roi_protocol"]["records"]}
    for row in rows:
        row["_roi_record"] = roi_index.get(str(row["sample_id"]))
        if not isinstance(row["_roi_record"], dict):
            raise PlateauError(f"ROI parent record is missing: {row['sample_id']}")
    generated_media: list[dict[str, Any]] = []
    generated_scores: list[dict[str, Any]] = []
    scorer = SyncNetScorer(config.SYNCNET_MODEL, device="cpu", batch_size=config.BATCH_SIZE, threads=config.TORCH_THREADS)
    audio_index = {str(row["sample_id"]): row for row in json.loads(paths.audio_manifest.read_text(encoding="utf-8"))["rows"]}
    for index, row in enumerate(rows, 1):
        sid = str(row["sample_id"])
        p_audio = Path(str(audio_index[sid]["arms"][config.AUDIO_P]["path"]))
        generated_video = paths.videos / "G_P" / f"{sid}.mkv"
        video_info = _run_generation(row, p_audio, generated_video, paths)
        cells: dict[str, Any] = {}
        for audio_arm, pcm in ((config.AUDIO_N, row["_pcm_bytes"]), (config.AUDIO_P, p_audio.read_bytes()[44:] if False else None)):
            audio_info = audio_index[sid]["arms"][audio_arm]
            if audio_arm == config.AUDIO_N:
                expected_pcm = row["_pcm_bytes"]
            else:
                expected_pcm, _values, _params = read_pcm16_wav(p_audio)
            output = paths.media / "generated_cells" / f"{sid}__E_P__{audio_arm}.mkv"
            evidence = media.mux_audio(generated_video, Path(str(audio_info["path"])), output, expected_pcm)
            sidecar = {**evidence, "protocol_id": "wav2lip_integer_plateau_control", "stage_id": "generated", "sample_id": sid, "video_arm": "E_P", "audio_arm": audio_arm}
            write_self_hashed_json(output.with_suffix(".json"), sidecar)
            cells[f"E_P__{audio_arm}"] = {**evidence, "audio": audio_info}
            cell_dir = paths.scores / "generated_cells" / config.cell_key(sid, "E_P", audio_arm)
            worker = scorer.score(Path(evidence["output"]), Path(str(audio_info["path"])), cell_dir, evidence["output_sha256"], evidence["audio_pcm_sha256"])
            worker.update({"protocol_id": "wav2lip_integer_plateau_control", "stage_id": "generated", "sample_id": sid, "source_group": row["source_group"], "video_arm": "E_P", "audio_arm": audio_arm, "new_forward": True})
            generated_scores.append(_cell_row(sample_id=sid, source_group=str(row["source_group"]), video_arm="E_P", audio_arm=audio_arm, media_info=evidence, audio_info=audio_info, worker=worker, protocol_sha256=file_sha256(paths.protocol), cell_dir=cell_dir))
        generated_media.append({"sample_id": sid, "source_group": row["source_group"], "generated_video": video_info, "cells": cells})
        print(f"GENERATED_DONE {index}/{len(rows)} {sid}", flush=True)
    generated_media_sha = write_self_hashed_json(paths.root / "generated_media_manifest.json", {"schema_version": 1, "stage_id": "generated_media", "protocol_id": "wav2lip_integer_plateau_control", "status": "complete", "record_count": len(rows), "video_count": len(generated_media), "cell_count": len(generated_scores), "rows": generated_media})
    generated_score_sha = write_self_hashed_json(paths.root / "generated_score_manifest.json", {"schema_version": 1, "stage_id": "generated_score", "protocol_id": "wav2lip_integer_plateau_control", "status": "complete", "record_count": len(rows), "cell_count": len(generated_scores), "scores": generated_scores, "generated_media_manifest_sha256": generated_media_sha})
    generated_analysis = analysis.analyze_generated(frozen, oracle_score_rows, generated_scores)
    generated_analysis.update({"generated_media_manifest_sha256": generated_media_sha, "generated_score_manifest_sha256": generated_score_sha})
    write_self_hashed_json(paths.generated_analysis, generated_analysis)
    return {"generated_analysis": generated_analysis, "generated_scores": generated_scores, "generated_media_sha256": generated_media_sha, "generated_score_sha256": generated_score_sha}


def run_experiment(run_id: str, stage: str) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.root.exists() and any(paths.root.iterdir()):
        raise PlateauError(f"run root already contains artifacts; use a new run-id: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    try:
        frozen, rows, oracle_analysis, oracle_score_rows = _prepare_stage_a(paths, run_id)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _blocked(paths, f"Stage A engineering failure: {type(exc).__name__}: {exc}")

    from . import validate

    oracle_validation = validate.validate_oracle(paths.root, final_sha256=None, write=True)
    if not bool(oracle_validation.get("valid")):
        final = _blocked(paths, "independent Stage A validation failed", len(oracle_score_rows))
        final["oracle_decision"] = oracle_analysis["decision"]
        write_self_hashed_json(paths.final, final)
        return final
    final = {
        "schema_version": 1,
        "protocol_id": "wav2lip_integer_plateau_control",
        "protocol_revision": "integer_plateau_v1",
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": oracle_analysis["decision"],
        "classification": "seen_fit_diagnostic",
        "protocol_sha256": file_sha256(paths.protocol),
        "input_audit_sha256": file_sha256(paths.input_audit),
        "oracle_analysis_sha256": file_sha256(paths.oracle_analysis),
        "counts": {"stage_a_generated_videos": config.EXPECTED_RECORD_COUNT, "stage_a_fresh_score_cells": len(oracle_score_rows), "stage_a_cached_score_cells": config.EXPECTED_RECORD_COUNT, "stage_b_generated_videos": 0, "stage_b_fresh_score_cells": 0},
        "expected_counts": {"stage_a_generated_videos": config.EXPECTED_RECORD_COUNT, "stage_a_fresh_score_cells": 66, "stage_a_cached_score_cells": 22, "stage_b_generated_videos": config.EXPECTED_RECORD_COUNT, "stage_b_fresh_score_cells": 44},
        "bridge_executed": False,
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
        "oracle_validation_valid": True,
        "oracle_decision": oracle_analysis["decision"],
        **analysis._fixed_flags(),
    }
    write_self_hashed_json(paths.final, final)
    if oracle_analysis["decision"] != "INTEGER_PLATEAU_CONTROL_SUPPORTED" or stage == "oracle":
        oracle_validation = dict(oracle_validation)
        oracle_validation["final_sha256"] = file_sha256(paths.final)
        write_self_hashed_json(paths.oracle_validation, oracle_validation)
        _write_result(paths, final, oracle_analysis)
        return final

    try:
        inputs = protocol.load_frozen_inputs()
        generated = _run_stage_b(paths, frozen, rows, oracle_score_rows, inputs)
        generated_validation = validate.validate_generated(paths.root, write=True)
        if not bool(generated_validation.get("valid")):
            final["engineering_decision"] = "BLOCKED"
            final["scientific_decision"] = None
        else:
            final["scientific_decision"] = generated["generated_analysis"]["decision"]
        final["generated_analysis_sha256"] = file_sha256(paths.generated_analysis)
        final["generated_validation_sha256"] = file_sha256(paths.generated_validation)
        final["counts"].update({"stage_b_generated_videos": config.EXPECTED_RECORD_COUNT, "stage_b_fresh_score_cells": len(generated["generated_scores"])})
        final["generated_validation_valid"] = bool(generated_validation.get("valid"))
        write_self_hashed_json(paths.final, final)
        oracle_validation = dict(oracle_validation)
        oracle_validation["final_sha256"] = file_sha256(paths.final)
        write_self_hashed_json(paths.oracle_validation, oracle_validation)
        _write_result(paths, final, oracle_analysis, generated["generated_analysis"])
        return final
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        final["engineering_decision"] = "BLOCKED"
        final["scientific_decision"] = None
        final["blocked_reason"] = f"Stage B engineering failure: {type(exc).__name__}: {exc}"
        write_self_hashed_json(paths.final, final)
        oracle_validation = dict(oracle_validation)
        oracle_validation["final_sha256"] = file_sha256(paths.final)
        write_self_hashed_json(paths.oracle_validation, oracle_validation)
        _write_result(paths, final, oracle_analysis)
        return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Wav2Lip integer plateau control")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("oracle", "generated", "all"), default="all")
    args = parser.parse_args(argv)
    result = run_experiment(args.run_id, args.stage)
    print(json.dumps({"status": result.get("status"), "engineering_decision": result.get("engineering_decision"), "scientific_decision": result.get("scientific_decision"), "run_root": str(config.run_root_for(args.run_id))}, ensure_ascii=False))
    return 0 if result.get("engineering_decision") != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
