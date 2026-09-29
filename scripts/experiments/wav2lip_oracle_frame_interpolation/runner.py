from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer
from scripts.experiments.wav2lip_roi_retiming_oracle.common import (
    OracleError,
    bytes_sha256,
    extract_bgr24_frames,
    extract_pcm_from_media,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from scripts.experiments.wav2lip_roi_retiming_oracle.media import mux_audio

from . import config
from .analysis import analyze, result_markdown
from .media import build_linear_indices, encode_video, interpolate_frames


def _load_parent() -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in config.PARENT_FILES.items():
        payloads[name] = verify_self_hashed_json(path, config.PARENT_HASHES[name])
    if payloads["validation"].get("valid") is not True:
        raise OracleError("fixed parent validation is not valid")
    if payloads["final"].get("diagnostic_decision") != "ORACLE_OWN_AUDIO_UNRESOLVED":
        raise OracleError("fixed parent is not the expected oracle own-audio result")
    protocol = payloads["protocol"]
    if protocol.get("classification") != "seen_fit_diagnostic" or len(protocol.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise OracleError("fixed parent record cohort is incomplete")
    records = protocol["records"]
    ids = [str(row["sample_id"]) for row in records]
    if bytes_sha256("\n".join(ids).encode("utf-8")) != config.ORDERED_SAMPLE_ID_SHA256:
        raise OracleError("fixed parent ordered sample-id hash changed")
    score_rows = payloads["score_manifest"].get("scores")
    if not isinstance(score_rows, list) or len(score_rows) != config.EXPECTED_CACHED_SCORE_CELL_COUNT:
        raise OracleError("fixed parent cached score count is not 88")
    frame_rows = payloads["frame_manifest"].get("rows")
    if not isinstance(frame_rows, list) or len(frame_rows) != 44:
        raise OracleError("fixed parent frame manifest is not 44 streams")
    audio_rows = payloads["audio_manifest"].get("rows")
    if not isinstance(audio_rows, list) or len(audio_rows) != config.EXPECTED_RECORD_COUNT:
        raise OracleError("fixed parent audio manifest is incomplete")
    return {
        "payloads": payloads,
        "protocol": protocol,
        "records": records,
        "score_rows": score_rows,
        "analysis": payloads["analysis"],
        "frame_by_key": {(str(row["sample_id"]), str(row["video_arm"])): row for row in frame_rows},
        "audio_by_id": {str(row["sample_id"]): row for row in audio_rows},
    }


def _environment(device: str, threads: int) -> dict[str, Any]:
    try:
        import cv2
        import python_speech_features
        import torch
    except Exception as exc:  # pragma: no cover - environment failure is reported by the runner
        raise OracleError(f"scoring environment import failed: {exc}") from exc
    ffmpeg = subprocess.run([str(config.FFMPEG), "-version"], capture_output=True, text=True, check=False)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cv2": cv2.__version__,
            "python_speech_features": getattr(python_speech_features, "__version__", "unknown"),
        },
        "device_requested": device,
        "torch_threads": int(threads),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ffmpeg_first_line": (ffmpeg.stdout.splitlines() or [""])[0],
        "syncnet_python": str(config.SYNCNET_PYTHON),
        "syncnet_model": str(config.SYNCNET_MODEL.resolve()),
        "syncnet_model_sha256": config.SYNCNET_MODEL_SHA256,
    }


def _source_bindings() -> dict[str, dict[str, str]]:
    package = Path(__file__).resolve().parent
    paths: list[tuple[str, Path]] = [(f"new/{path.name}", path) for path in sorted(package.glob("*.py"))]
    paths.extend(
        [
            ("official/SyncNetModel.py", config.SYNCNET_ROOT / "SyncNetModel.py"),
            ("official/SyncNetInstance.py", config.SYNCNET_ROOT / "SyncNetInstance.py"),
            ("official/syncnet_v2.model", config.SYNCNET_MODEL),
            ("scorer/worker.py", config.REPO / "scripts/experiments/wav2lip_roi_peak_recheck/worker.py"),
            ("shared/oracle_common.py", config.REPO / "scripts/experiments/wav2lip_roi_retiming_oracle/common.py"),
            ("shared/oracle_media.py", config.REPO / "scripts/experiments/wav2lip_roi_retiming_oracle/media.py"),
        ]
    )
    paths.extend((f"change/{name}", path) for name, path in (("proposal.md", config.PROPOSAL), ("design.md", config.DESIGN), ("spec.md", config.SPEC)))
    result: dict[str, dict[str, str]] = {}
    for name, path in paths:
        if not path.is_file():
            raise OracleError(f"source binding is missing: {path}")
        result[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return result


def _score_index(rows: Sequence[Mapping[str, Any]], sample_ids: Sequence[str]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result:
            raise OracleError(f"duplicate parent score cell: {key}")
        result[key] = row
    expected = {
        (str(sample_id), video, audio)
        for sample_id in sample_ids
        for video in ("V_ID", "V_ORACLE")
        for audio in config.AUDIO_ARMS
    }
    if set(result) != expected:
        raise OracleError(f"parent score cell set differs: {len(result)}/{len(expected)}")
    return result


def _audit_inputs(parent: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    score_by_key = _score_index(parent["score_rows"], [str(item["sample_id"]) for item in parent["records"]])
    audit_records: list[dict[str, Any]] = []
    work_rows: list[dict[str, Any]] = []
    for record in parent["records"]:
        sample_id = str(record["sample_id"])
        frame_row = parent["frame_by_key"].get((sample_id, "V_ID"))
        if not isinstance(frame_row, Mapping):
            raise OracleError(f"parent V_ID frame row is missing: {sample_id}")
        source_stream = frame_row.get("source_stream")
        if not isinstance(source_stream, Mapping):
            raise OracleError(f"parent V_ID source stream is missing: {sample_id}")
        source_path = Path(str(source_stream["path"])).resolve()
        if not source_path.is_file() or file_sha256(source_path) != str(source_stream["sha256"]):
            raise OracleError(f"parent V_ID source stream hash changed: {sample_id}")
        source_frames, source_evidence = extract_bgr24_frames(source_path)
        frame_count = int(record["frame_count"])
        if source_frames.shape[0] != frame_count or frame_row.get("q") != list(range(frame_count)):
            raise OracleError(f"parent V_ID identity stream is not frozen: {sample_id}")
        natural = record.get("natural_audio")
        if not isinstance(natural, Mapping):
            raise OracleError(f"natural audio binding is missing: {sample_id}")
        natural_path = Path(str(natural["path"])).resolve()
        natural_pcm, natural_values, _ = read_pcm16_wav(natural_path)
        if file_sha256(natural_path) != str(natural["container_sha256"]) or bytes_sha256(natural_pcm) != str(natural["decoded_pcm_sha256"]):
            raise OracleError(f"natural audio hash changed: {sample_id}")
        indices = build_linear_indices(int(natural_values.size), frame_count)
        parent_q = np.asarray(record["indices"]["q_float"], dtype=np.float64)
        if parent_q.shape != np.asarray(indices["u"], dtype=np.float64).shape or not np.allclose(parent_q, indices["u"], atol=1e-9, rtol=0.0):
            raise OracleError(f"parent q_float differs from the frozen mapping: {sample_id}")
        linear_frames, linear_evidence = interpolate_frames(source_frames, indices)
        audio_row = parent["audio_by_id"].get(sample_id)
        if not isinstance(audio_row, Mapping):
            raise OracleError(f"parent audio row is missing: {sample_id}")
        audio_dir = Path(str(audio_row["audio_dir"])).resolve()
        audio_paths: dict[str, Path] = {}
        audio_pcm: dict[str, bytes] = {}
        for audio in config.AUDIO_ARMS:
            path = audio_dir / f"{sample_id}__{audio}.wav"
            pcm, _values, _params = read_pcm16_wav(path)
            audio_paths[audio] = path
            audio_pcm[audio] = pcm
        if audio_pcm["N"] != natural_pcm:
            raise OracleError(f"parent N audio is not byte-identical: {sample_id}")
        parent_w_media = record.get("parent_w_media")
        if not isinstance(parent_w_media, Mapping):
            raise OracleError(f"parent W media binding is missing: {sample_id}")
        w_media_path = Path(str(parent_w_media["path"])).resolve()
        if file_sha256(w_media_path) != str(parent_w_media["sha256"]):
            raise OracleError(f"parent W media hash changed: {sample_id}")
        if extract_pcm_from_media(w_media_path) != audio_pcm["W"]:
            raise OracleError(f"parent W audio differs from the frozen mux: {sample_id}")
        parent_cells = {
            f"{video}__{audio}": {
                "media": str(score_by_key[(sample_id, video, audio)]["media"]),
                "media_sha256": str(score_by_key[(sample_id, video, audio)]["media_sha256"]),
                "matrix": str(score_by_key[(sample_id, video, audio)]["matrix"]),
                "matrix_sha256": str(score_by_key[(sample_id, video, audio)]["matrix_sha256"]),
            }
            for video in ("V_ID", "V_ORACLE")
            for audio in config.AUDIO_ARMS
        }
        masks = record["masks"]
        if len(masks["plus_rows"]) < config.MIN_LOCAL_ROWS or len(masks["minus_rows"]) < config.MIN_LOCAL_ROWS:
            raise OracleError(f"parent local mask is too small: {sample_id}")
        work_rows.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "frame_count": frame_count,
                "sample_count": int(natural_values.size),
                "source_frames": source_frames,
                "linear_frames": linear_frames,
                "linear_indices": indices,
                "linear_evidence": linear_evidence,
                "source_evidence": source_evidence,
                "natural_pcm": natural_pcm,
                "warped_pcm": audio_pcm["W"],
                "audio_paths": audio_paths,
                "masks": masks,
                "natural_audio": dict(natural),
                "parent_stream": dict(source_stream),
                "parent_cells": parent_cells,
            }
        )
        audit_records.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "frame_count": frame_count,
                "sample_count": int(natural_values.size),
                "source_stream": dict(source_stream),
                "source_frame_sha256": source_evidence["frame_sha256"],
                "linear_indices": indices,
                "linear_pixel_evidence": linear_evidence,
                "natural_audio": dict(natural),
                "audio_paths": {audio: str(path) for audio, path in audio_paths.items()},
                "audio_pcm_sha256": {audio: bytes_sha256(pcm) for audio, pcm in audio_pcm.items()},
                "masks": masks,
                "parent_cells": parent_cells,
            }
        )
    audit = {
        "schema_version": 1,
        "stage_id": "input_audit",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "parent_root": str(config.PARENT_ROOT.resolve()),
        "parent_hashes": dict(config.PARENT_HASHES),
        "record_count": len(audit_records),
        "source_group_count": len({row["source_group"] for row in audit_records}),
        "records": audit_records,
    }
    return audit, work_rows


def _protocol(parent: Mapping[str, Any], audit: Mapping[str, Any], audit_sha: str, run_id: str, device: str, threads: int, work_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    for row in work_rows:
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "frame_count": row["frame_count"],
                "sample_count": row["sample_count"],
                "source_stream": row["parent_stream"],
                "natural_audio": row["natural_audio"],
                "audio_paths": {audio: str(path) for audio, path in row["audio_paths"].items()},
                "linear_indices": row["linear_indices"],
                "masks": row["masks"],
                "parent_cells": row["parent_cells"],
            }
        )
    fresh_cells = [
        {"sample_id": row["sample_id"], "source_group": row["source_group"], "video_arm": config.VIDEO_ARM, "audio_arm": audio, "origin": "fresh"}
        for row in work_rows
        for audio in config.AUDIO_ARMS
    ]
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "run_id": run_id,
        "status": "frozen",
        "classification": "seen_fit_diagnostic",
        "parent": {
            "root": str(config.PARENT_ROOT.resolve()),
            "fixed_hashes": dict(config.PARENT_HASHES),
            "scientific_decision": "CONTROL_FAILED",
            "oracle_decision": "ORACLE_OWN_AUDIO_UNRESOLVED",
        },
        "change_bindings": config.spec_bindings(),
        "source_bindings": _source_bindings(),
        "frozen_config": config.FrozenConfig().to_dict(),
        "environment": _environment(device, threads),
        "input_audit_sha256": audit_sha,
        "parent_score_manifest_sha256": config.PARENT_HASHES["score_manifest"],
        "records": records,
        "fresh_cells": fresh_cells,
        "cached_score_cell_count": config.EXPECTED_CACHED_SCORE_CELL_COUNT,
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "expected_fresh_stream_count": config.EXPECTED_FRESH_STREAM_COUNT,
        "expected_fresh_media_count": config.EXPECTED_FRESH_MEDIA_COUNT,
        "expected_fresh_score_cell_count": config.EXPECTED_FRESH_SCORE_CELL_COUNT,
    }


def _write_blocked(paths: config.RunPaths, reason: str, completed_cells: int = 0) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(
        "# Wav2Lip oracle frame interpolation 2026-09-07\n\n"
        f"终态：`BLOCKED`；fresh cell {completed_cells}/{config.EXPECTED_FRESH_SCORE_CELL_COUNT}。\n\n原因：{reason}\n",
        encoding="utf-8",
    )
    write_self_hashed_json(
        paths.final,
        {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_revision": config.PROTOCOL_REVISION,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "diagnostic_decision": None,
            "historical_scientific_decision": "CONTROL_FAILED",
            "parent_oracle_decision": "ORACLE_OWN_AUDIO_UNRESOLVED",
            "parent_own_audio_retested": False,
            "oracle_own_audio_tested": completed_cells > 0,
            "new_generated_videos": 0,
            "bridge_executed": False,
            "training_authorized": False,
            "reference_conditioned_audio_head_spec_eligible": False,
            "generalization_established": False,
            "new_score_cells": completed_cells,
            "blocked_reason": reason,
            "result_sha256": file_sha256(paths.result),
        },
    )


def run_experiment(run_id: str, device: str = "cpu", threads: int = config.TORCH_THREADS) -> dict[str, Any]:
    if device != "cpu":
        raise OracleError("the protocol is frozen to CPU")
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.root.exists() and any(paths.root.iterdir()):
        raise OracleError(f"run root already contains artifacts: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    completed_cells = 0
    try:
        parent = _load_parent()
        audit, work_rows = _audit_inputs(parent)
        audit_sha = write_self_hashed_json(paths.input_audit, audit)
        protocol = _protocol(parent, audit, audit_sha, run_id, device, threads, work_rows)
        protocol_sha = write_self_hashed_json(paths.protocol, protocol)
        frame_rows: list[dict[str, Any]] = []
        media_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        scorer = SyncNetScorer(config.SYNCNET_MODEL, device=device, batch_size=config.BATCH_SIZE, threads=threads)
        for index, row in enumerate(work_rows, 1):
            sample_id = str(row["sample_id"])
            video_path = paths.root / "media/streams" / f"{sample_id}__{config.VIDEO_ARM}.mkv"
            video_evidence = encode_video(row["linear_frames"], video_path)
            frame_rows.append(
                {
                    "sample_id": sample_id,
                    "source_group": row["source_group"],
                    "video_arm": config.VIDEO_ARM,
                    "source_stream": row["parent_stream"],
                    "indices": row["linear_indices"],
                    "interpolation": row["linear_evidence"],
                    "evidence": video_evidence,
                    "pixel_identity_verified": True,
                }
            )
            cells: dict[str, dict[str, Any]] = {}
            for audio in config.AUDIO_ARMS:
                audio_path = Path(str(row["audio_paths"][audio]))
                media_path = paths.root / "media/cells" / f"{config.cell_key(sample_id, config.VIDEO_ARM, audio)}.mkv"
                expected_pcm = row["natural_pcm"] if audio == "N" else row["warped_pcm"]
                mux_evidence = mux_audio(video_path, audio_path, media_path, expected_pcm)
                cells[audio] = {
                    **mux_evidence,
                    "audio": {"path": str(audio_path.resolve()), "container_sha256": file_sha256(audio_path), "decoded_pcm_sha256": bytes_sha256(expected_pcm)},
                }
            media_rows.append(
                {
                    "sample_id": sample_id,
                    "source_group": row["source_group"],
                    "stream": {"output": str(video_path.resolve()), "output_sha256": file_sha256(video_path), "frame_count": row["frame_count"]},
                    "cells": {audio: {**value, "origin": "fresh"} for audio, value in cells.items()},
                }
            )
            for audio in config.AUDIO_ARMS:
                cell = cells[audio]
                cell_dir = paths.root / "scores/cells" / config.cell_key(sample_id, config.VIDEO_ARM, audio)
                print(f"CELL_START {index}/{len(work_rows)} {sample_id} {config.VIDEO_ARM}/{audio}", flush=True)
                worker = scorer.score(
                    Path(str(cell["output"])),
                    Path(str(cell["audio"]["path"])),
                    cell_dir,
                    str(cell["output_sha256"]),
                    str(cell["audio_pcm_sha256"]),
                )
                worker.update(
                    {
                        "protocol_id": config.PROTOCOL_ID,
                        "protocol_sha256": protocol_sha,
                        "sample_id": sample_id,
                        "source_group": row["source_group"],
                        "video_arm": config.VIDEO_ARM,
                        "audio_arm": audio,
                        "new_forward": True,
                    }
                )
                worker_path = cell_dir / "worker.json"
                worker_sha = write_self_hashed_json(worker_path, worker)
                score_rows.append(
                    {
                        "schema_version": 1,
                        "stage_id": "fresh_syncnet_forward",
                        "protocol_id": config.PROTOCOL_ID,
                        "protocol_sha256": protocol_sha,
                        "sample_id": sample_id,
                        "source_group": row["source_group"],
                        "video_arm": config.VIDEO_ARM,
                        "audio_arm": audio,
                        "origin": "fresh",
                        "media": str(cell["output"]),
                        "media_sha256": str(cell["output_sha256"]),
                        "media_pcm_sha256": str(cell["audio_pcm_sha256"]),
                        "audio": str(cell["audio"]["path"]),
                        "audio_container_sha256": str(cell["audio"]["container_sha256"]),
                        "audio_pcm_sha256": str(cell["audio"]["decoded_pcm_sha256"]),
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
                )
                completed_cells += 1
                print(f"CELL_DONE {index}/{len(work_rows)} {sample_id} {config.VIDEO_ARM}/{audio}", flush=True)

        frame_manifest_sha = write_self_hashed_json(
            paths.frames_manifest,
            {
                "schema_version": 1,
                "stage_id": "frame_materialization",
                "protocol_id": config.PROTOCOL_ID,
                "protocol_sha256": protocol_sha,
                "status": "complete",
                "record_count": len(work_rows),
                "stream_count": len(frame_rows),
                "rows": frame_rows,
            },
        )
        media_manifest_sha = write_self_hashed_json(
            paths.media_manifest,
            {
                "schema_version": 1,
                "stage_id": "media",
                "protocol_id": config.PROTOCOL_ID,
                "protocol_sha256": protocol_sha,
                "status": "complete",
                "record_count": len(media_rows),
                "stream_count": len(frame_rows),
                "cell_count": len(score_rows),
                "expected_stream_count": config.EXPECTED_FRESH_STREAM_COUNT,
                "expected_cell_count": config.EXPECTED_FRESH_MEDIA_COUNT,
                "rows": media_rows,
            },
        )
        score_manifest_sha = write_self_hashed_json(
            paths.scores_manifest,
            {
                "schema_version": 1,
                "stage_id": "score",
                "protocol_id": config.PROTOCOL_ID,
                "protocol_sha256": protocol_sha,
                "media_manifest_sha256": media_manifest_sha,
                "parent_score_manifest_sha256": config.PARENT_HASHES["score_manifest"],
                "status": "complete",
                "record_count": len(work_rows),
                "cell_count": len(score_rows),
                "expected_cell_count": config.EXPECTED_FRESH_SCORE_CELL_COUNT,
                "cached_cell_count": config.EXPECTED_CACHED_SCORE_CELL_COUNT,
                "scores": score_rows,
            },
        )
        analysis = analyze(parent, protocol, score_rows)
        analysis.update(
            {
                "input_audit_sha256": audit_sha,
                "protocol_sha256": protocol_sha,
                "frame_manifest_sha256": frame_manifest_sha,
                "media_manifest_sha256": media_manifest_sha,
                "score_manifest_sha256": score_manifest_sha,
            }
        )
        analysis_sha = write_self_hashed_json(paths.analysis, analysis)
        paths.result.write_text(result_markdown(analysis), encoding="utf-8")
        final = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_revision": config.PROTOCOL_REVISION,
            "status": "complete",
            "engineering_decision": "GO",
            "diagnostic_decision": analysis["decision"],
            "historical_scientific_decision": "CONTROL_FAILED",
            "parent_oracle_decision": "ORACLE_OWN_AUDIO_UNRESOLVED",
            "parent_own_audio_retested": False,
            "oracle_own_audio_tested": True,
            "new_generated_videos": 0,
            "new_video_stream_count": len(frame_rows),
            "new_media_count": len(score_rows),
            "new_score_cells": len(score_rows),
            "cached_score_cells": config.EXPECTED_CACHED_SCORE_CELL_COUNT,
            "interpolation_improvement_supported": analysis["interpolation_improvement_supported"],
            "bridge_executed": False,
            "training_authorized": False,
            "reference_conditioned_audio_head_spec_eligible": False,
            "generalization_established": False,
            "protocol_sha256": protocol_sha,
            "input_audit_sha256": audit_sha,
            "analysis_sha256": analysis_sha,
            "result_sha256": file_sha256(paths.result),
            "frame_manifest_sha256": frame_manifest_sha,
            "media_manifest_sha256": media_manifest_sha,
            "score_manifest_sha256": score_manifest_sha,
        }
        write_self_hashed_json(paths.final, final)
        return final
    except Exception as exc:
        _write_blocked(paths, str(exc), completed_cells)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Wav2Lip linear-pixel oracle diagnostic")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--threads", type=int, default=config.TORCH_THREADS)
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args.run_id, args.device, args.threads)
    except Exception as exc:  # noqa: BLE001 - a blocked artifact is written for inspection
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
