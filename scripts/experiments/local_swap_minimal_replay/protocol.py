from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import DiagnosticError, file_sha256, git_commit, read_json, sample_ids_sha256, write_self_hashed_json


def _history_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"history artifact is missing: {path}")
    return read_json(path)


def load_history() -> dict[str, Any]:
    return {
        "cohort": _history_json(config.HISTORY_COHORT),
        "audio": _history_json(config.HISTORY_AUDIO),
        "videos": _history_json(config.HISTORY_VIDEOS),
        "scores": _history_json(config.HISTORY_SCORES),
        "final": _history_json(config.HISTORY_FINAL),
    }


def _runtime_binding(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "label": label}
    return {"path": str(path.resolve()), "exists": True, "sha256": file_sha256(path), "label": label}


def _git_status(repo: Path) -> str:
    result = subprocess.run(["git", "status", "--porcelain=v1"], cwd=str(repo), capture_output=True, text=True, check=False)
    return "dirty" if result.returncode == 0 and result.stdout.strip() else "clean"


def _asset(path_value: str, expected_sha: str, label: str) -> dict[str, Any]:
    path = Path(path_value).resolve()
    exists = path.is_file()
    actual = file_sha256(path) if exists else None
    return {"path": str(path), "expected_sha256": expected_sha, "actual_sha256": actual, "exists": exists, "label": label}


def freeze_inputs(history: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    cohort = history["cohort"]
    audio = history["audio"]
    videos = history["videos"]
    scores = history["scores"]
    records = list(cohort.get("records", []))
    if len(records) != config.EXPECTED_HISTORY_COUNT:
        raise DiagnosticError(f"history cohort count changed: {len(records)}")
    audio_rows = {str(row["sample_id"]): row for row in audio.get("rows", [])}
    video_rows = {str(row["sample_id"]): row for row in videos.get("rows", [])}
    score_rows = {str(row["sample_id"]): row for row in scores.get("scores", [])}
    all_bindings: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        audio_row = audio_rows.get(sample_id)
        video_row = video_rows.get(sample_id)
        if not isinstance(audio_row, Mapping) or not isinstance(video_row, Mapping):
            raise DiagnosticError(f"history row missing for {sample_id}")
        arms = {str(row["arm"]): row for row in audio_row.get("arms", [])}
        video_arm = video_row.get("arms", {}).get("LOCAL_SWAP")
        if not isinstance(video_arm, Mapping) or "N" not in arms or "LOCAL_SWAP" not in arms:
            raise DiagnosticError(f"history LOCAL_SWAP binding is incomplete: {sample_id}")
        score_sample = [row for row in scores.get("scores", []) if str(row.get("sample_id")) == sample_id and str(row.get("cell")) in {"V_LOCAL_SWAP/A_N", "V_LOCAL_SWAP/A_LOCAL_SWAP"}]
        all_bindings.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "face_video": _asset(str(record["face_video"]["path"]), str(record["face_video"]["sha256"]), "face_video"),
                "natural_audio": _asset(str(arms["N"]["output"]), str(arms["N"]["output_sha256"]), "history_N_audio"),
                "local_swap_audio": _asset(str(arms["LOCAL_SWAP"]["output"]), str(arms["LOCAL_SWAP"]["output_sha256"]), "history_LOCAL_SWAP_audio"),
                "generated_local_swap_video": _asset(str(video_arm["output"]), str(video_arm["output_sha256"]), "history_LOCAL_SWAP_video"),
                "video_audio_binding_sha256": str(video_arm.get("audio_sha256", "")),
                "historical_score_rows": [
                    {
                        "cell": str(row.get("cell")),
                        "media": str(row.get("media")),
                        "media_sha256": str(row.get("media_sha256", "")),
                        "score_log": str(row.get("score_log", "")),
                    }
                    for row in score_sample
                ],
            }
        )
    selected = [row for row in all_bindings if row["sample_id"] in config.SAMPLE_IDS]
    if [row["sample_id"] for row in records[:3]] != list(config.SAMPLE_IDS):
        raise DiagnosticError("fixed sample IDs are not the first three cohort records")
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "history_root": str(config.HISTORY_ROOT.resolve()),
        "history_artifacts": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in {
                "cohort": config.HISTORY_COHORT,
                "audio": config.HISTORY_AUDIO,
                "videos": config.HISTORY_VIDEOS,
                "scores": config.HISTORY_SCORES,
                "final": config.HISTORY_FINAL,
            }.items()
        },
        "selection": {
            "all_history_records": config.EXPECTED_HISTORY_COUNT,
            "new_score_records": len(config.SAMPLE_IDS),
            "ordered_sample_ids": list(config.SAMPLE_IDS),
            "ordered_sample_id_sha256": sample_ids_sha256(config.SAMPLE_IDS),
            "score_based_selection": False,
        },
        "all_history_bindings": all_bindings,
        "selected_bindings": selected,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_commit(config.REPO),
            "git_worktree": _git_status(config.REPO),
            "syncnet_python": _runtime_binding(config.SYNCNET_PYTHON, "syncnet_python"),
            "syncnet_model": _runtime_binding(config.SYNCNET_MODEL, "syncnet_model"),
            "syncnet_worker": _runtime_binding(config.SYNCNET_WORKER, "syncnet_worker"),
            "run_pipeline": _runtime_binding(config.SYNCNET_PIPELINE, "run_pipeline"),
            "run_syncnet": _runtime_binding(config.SYNCNET_SCORE, "run_syncnet"),
            "ffmpeg": _runtime_binding(config.FFMPEG, "ffmpeg"),
            "ffprobe": _runtime_binding(config.FFPROBE, "ffprobe"),
            "experiment_code": {
                str(path.resolve().relative_to(config.REPO.resolve())): _runtime_binding(path, "experiment_code")
                for path in config.EXPERIMENT_CODE_FILES
            },
        },
        "new_run_root": str(run_root.resolve()),
        "forbidden_operations": ["wav2lip_inference", "tts_generation", "training", "bridge_rescoring", "score_based_retry", "history_overwrite"],
    }
    return payload


def write_protocol(paths: config.RunPaths, inputs: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    protocol = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "config": {
            "sample_rate": config.SAMPLE_RATE,
            "fps": config.FPS,
            "samples_per_frame": config.SAMPLES_PER_FRAME,
            "window_frames": config.WINDOW_FRAMES,
            "vshift": config.VSHIFT,
            "fixed_cell_specs": [list(item) for item in config.CELL_SPECS],
            "minimum_k": config.MIN_K,
            "local_margin_frames": config.LOCAL_MARGIN_FRAMES,
        },
        "inputs_sha256": inputs.get("artifact_sha256"),
        "historical_audit_sha256": audit.get("artifact_sha256"),
        "historical_run_unchanged": True,
        "new_run_root": str(paths.root.resolve()),
    }
    write_self_hashed_json(paths.protocol, protocol)
    return protocol
