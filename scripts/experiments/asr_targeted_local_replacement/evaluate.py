from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.asr_sync_error_correlation.io import atomic_write_json, atomic_write_npz, canonical_hash, file_sha256, read_npz
from scripts.experiments.asr_sync_error_correlation.local_sync import compute_local_scores, map_local_scores, validate_paired_track_metadata
from scripts.experiments.asr_sync_error_correlation.syncnet_adapter import build_adapter_command, build_pipeline_command, run_subprocess
from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify

from .config import CONDITIONS, SYNCNET_CHECKPOINT_SHA256


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _code_hash(package_dir: Path) -> str:
    return canonical_hash({path.name: file_sha256(path) for path in sorted(package_dir.glob("*.py"))})


def _safe_reference(sample_id: str, condition: str) -> str:
    value = f"{sample_id}_{condition}"
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise ValueError("sample/condition cannot be used as a SyncNet reference")
    return value


def validate_isolated_work_paths(cells: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for cell in cells:
        for key in ("work_dir", "temp_dir"):
            value = str(Path(str(cell[key])).resolve())
            if value in seen:
                raise ValueError(f"shared Wav2Lip path is forbidden: {value}")
            seen.add(value)
        work = Path(str(cell["work_dir"]))
        temp = Path(str(cell["temp_dir"]))
        if temp.parent.resolve() != work.resolve() or temp.name != "temp":
            raise ValueError("Wav2Lip temp directory must be the cell-local work_dir/temp")


def render_wav2lip_cell(
    *,
    repo_root: Path,
    run_dir: Path,
    sample_id: str,
    condition: str,
    face_path: Path,
    driver_audio: Path,
    device: str,
    resume: bool = False,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unregistered condition: {condition}")
    work_dir = (run_dir / "02_renders" / "work" / sample_id / condition).resolve()
    temp_dir = work_dir / "temp"
    output_video = (run_dir / "02_renders" / "videos" / sample_id / f"{condition}.avi").resolve()
    log_path = (run_dir / "logs" / "render" / sample_id / f"{condition}.log").resolve()
    record_path = (run_dir / "02_renders" / "records" / sample_id / f"{condition}.json").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not face_path.is_file() or not driver_audio.is_file():
        raise FileNotFoundError(f"render input missing for {sample_id}/{condition}")
    wav2lip_dir = repo_root / "third_party/Wav2Lip"
    wav2lip_python = Path.home() / ".venvs/wav2lip/bin/python"
    checkpoint = wav2lip_dir / "checkpoints/wav2lip_gan.pth"
    if not wav2lip_python.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("frozen Wav2Lip executable or checkpoint is missing")
    command = [
        str(wav2lip_python), str(wav2lip_dir / "inference.py"),
        "--checkpoint_path", str(checkpoint), "--face", str(face_path),
        "--audio", str(driver_audio), "--outfile", str(output_video),
        "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth",
    ]
    bindings = {
        "sample_id": sample_id,
        "condition": condition,
        "face_sha256": file_sha256(face_path),
        "driver_audio_sha256": file_sha256(driver_audio),
        "wav2lip_checkpoint_sha256": file_sha256(checkpoint),
        "wav2lip_code_hash": canonical_hash({path.name: file_sha256(path) for path in sorted(wav2lip_dir.glob("*.py"))}),
        "device": device,
        "command": command,
        "work_dir": str(work_dir),
        "temp_dir": str(temp_dir),
    }
    if resume and output_video.is_file():
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing.get("bindings") == bindings and existing.get("video_sha256") == file_sha256(output_video):
                return {**existing, "resumed": True}
        except (OSError, ValueError, TypeError, KeyError):
            pass
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(work_dir), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not output_video.is_file():
        raise RuntimeError(f"Wav2Lip failed for {sample_id}/{condition}: returncode={result.returncode}")
    return {
        "sample_id": sample_id,
        "condition": condition,
        "video_path": str(output_video),
        "video_sha256": file_sha256(output_video),
        "bindings": bindings,
        "log_path": str(log_path),
        "returncode": int(result.returncode),
        "resumed": False,
    }


def strict_replacement_cell(
    *,
    repo_root: Path,
    run_dir: Path,
    sample_id: str,
    condition: str,
    rendered_video: Path,
    natural_audio: Path,
    resume: bool = False,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unregistered condition: {condition}")
    output = (run_dir / "03_replacement" / sample_id / f"{condition}.mkv").resolve()
    record_path = (run_dir / "03_replacement" / "records" / sample_id / f"{condition}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if not rendered_video.is_file() or not natural_audio.is_file():
        raise FileNotFoundError(f"replacement input missing for {sample_id}/{condition}")
    rendered_hash = file_sha256(rendered_video)
    natural_hash = file_sha256(natural_audio)
    if resume and output.is_file() and record_path.is_file():
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if (existing.get("copy_video_sha256") == rendered_hash and existing.get("untouched_natural_audio_sha256") == natural_hash and existing.get("output_sha256") == file_sha256(output) and existing.get("strict_contract", {}).get("audio_pcm_verified") is True and existing.get("strict_contract", {}).get("video_stream_copy_verified") is True):
                return {**existing, "resumed": True}
        except (OSError, ValueError, TypeError, KeyError):
            pass
    verification = mux_and_verify(source_video=rendered_video, expected_audio=natural_audio, output_path=output)
    return {
        "sample_id": sample_id,
        "condition": condition,
        "copy_video": str(rendered_video),
        "copy_video_sha256": rendered_hash,
        "untouched_natural_audio": str(natural_audio),
        "untouched_natural_audio_sha256": natural_hash,
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        "strict_contract": verification,
        "resumed": False,
    }


def _local_diagnostic(
    natural_record: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    natural_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    patches: Sequence[Mapping[str, Any]],
    *,
    audio_duration_s: float,
) -> dict[str, Any] | None:
    try:
        validate_paired_track_metadata(natural_record, candidate_record)
    except ValueError:
        return {"status": "null", "reason": "paired_track_mismatch"}
    if natural_matrix.shape != candidate_matrix.shape:
        return {"status": "null", "reason": "distance_shape_mismatch"}
    natural_scores = compute_local_scores(natural_matrix)
    mapped = map_local_scores(
        natural_scores,
        track_start_frame=int(natural_record["track_start_frame"]),
        track_end_frame=int(natural_record["track_end_frame"]),
        audio_duration_s=float(audio_duration_s),
    )
    if mapped["retained_count"] == 0:
        return {"status": "null", "reason": "no_common_valid_rows"}
    rows = np.asarray(mapped["source_rows"], dtype=np.int64)
    improvement = natural_matrix[rows, int(natural_scores["j_star"])] - candidate_matrix[rows, int(natural_scores["j_star"])]
    timestamps = np.asarray(mapped["timestamps_s"], dtype=np.float64)
    support = np.zeros(timestamps.shape, dtype=bool)
    for patch in patches:
        start = float(patch["destination_start_s"]) - 0.20
        end = float(patch["destination_end_s"]) + 0.20
        support |= (timestamps >= max(0.0, start)) & (timestamps < min(float(audio_duration_s), end))
    def summarize(values: np.ndarray) -> dict[str, Any]:
        if values.size == 0:
            return {"count": 0, "mean": None, "median": None}
        return {"count": int(values.size), "mean": float(np.mean(values)), "median": float(np.median(values))}
    return {
        "status": "ok",
        "coordinate": {"offset_column": int(natural_scores["j_star"]), "av_offset_frames": int(natural_scores["av_offset_frames"]), "guard_s": 0.20},
        "inside": summarize(improvement[support]),
        "outside": summarize(improvement[~support]),
        "common_valid_rows": int(rows.size),
        "support_rows": int(np.count_nonzero(support)),
    }


def score_syncnet_cell(
    *,
    repo_root: Path,
    run_dir: Path,
    sample_id: str,
    condition: str,
    replacement_media: Path,
    natural_sync_record: Mapping[str, Any] | None = None,
    patches: Sequence[Mapping[str, Any]] = (),
    audio_duration_s: float,
    device: str,
    resume: bool = False,
    sync_stage: str = "04_sync",
    media_label: str = "replacement_media",
    log_stage: str = "sync",
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unregistered condition: {condition}")
    if not replacement_media.is_file():
        raise FileNotFoundError(f"replacement media missing: {replacement_media}")
    syncnet_dir = repo_root / "third_party/syncnet_python"
    syncnet_python = Path.home() / ".venvs/syncnet/bin/python"
    model = syncnet_dir / "data/syncnet_v2.model"
    if file_sha256(model) != SYNCNET_CHECKPOINT_SHA256:
        raise ValueError("SyncNet checkpoint hash differs from frozen lock")
    reference = _safe_reference(sample_id, condition)
    cell_dir = (run_dir / sync_stage / sample_id / condition).resolve()
    data_dir = cell_dir / "data"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs" / log_stage / sample_id
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = log_dir / f"{condition}.pipeline.log"
    adapter_log = log_dir / f"{condition}.adapter.log"
    distance_path = cell_dir / "distances.npz"
    record_path = cell_dir / "record.json"
    if resume and distance_path.is_file() and record_path.is_file():
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing.get(f"{media_label}_sha256") == file_sha256(replacement_media) and existing.get("distance_sha256") == file_sha256(distance_path) and existing.get("parity", {}).get("passed") is True:
                read_npz(distance_path, required=("dists",))
                existing["sample_id"] = sample_id
                existing["condition"] = condition
                atomic_write_json(record_path, existing)
                return {**existing, "resumed": True}
        except (OSError, ValueError, TypeError, KeyError):
            pass
    pipeline_command = build_pipeline_command(syncnet_python=syncnet_python, syncnet_dir=syncnet_dir, video_path=replacement_media, reference=reference, data_dir=data_dir, min_track=50)
    adapter_path = Path(__file__).resolve().parents[3] / "scripts/experiments/asr_sync_error_correlation/syncnet_adapter.py"
    adapter_command = build_adapter_command(syncnet_python=syncnet_python, adapter_path=adapter_path, video_path=replacement_media, reference=reference, data_dir=data_dir, model_path=model, output_npz=distance_path, output_json=record_path, batch_size=20, vshift=15, device=device)
    run_subprocess(pipeline_command, cwd=syncnet_dir, log_path=pipeline_log)
    run_subprocess(adapter_command, cwd=syncnet_dir, log_path=adapter_log)
    upstream = json.loads(record_path.read_text(encoding="utf-8"))
    arrays = read_npz(distance_path, required=("dists",))
    matrix = np.asarray(arrays["dists"], dtype=np.float32)
    scores = compute_local_scores(matrix, vshift=15, median_width=9)
    parity = {
        "offset_exact": int(scores["av_offset_frames"]) == int(upstream["upstream_offset"]),
        "confidence_abs_error": abs(float(scores["sync_c"]) - float(upstream["upstream_sync_c"])),
        "distance_abs_error": abs(float(scores["sync_d"]) - float(upstream["upstream_sync_d"])),
    }
    parity["confidence_within_1e-6"] = parity["confidence_abs_error"] <= 1e-6
    parity["distance_within_5e-4"] = parity["distance_abs_error"] <= 5e-4
    parity["passed"] = bool(parity["offset_exact"] and parity["confidence_within_1e-6"] and parity["distance_within_5e-4"])
    upstream["distance_sha256"] = file_sha256(distance_path)
    upstream["sample_id"] = sample_id
    upstream["condition"] = condition
    upstream[media_label] = str(replacement_media)
    upstream[f"{media_label}_sha256"] = file_sha256(replacement_media)
    upstream["official"] = {"sync_c": float(scores["sync_c"]), "sync_d": float(scores["sync_d"]), "av_offset_frames": int(scores["av_offset_frames"])}
    upstream["parity"] = parity
    if not parity["passed"]:
        raise ValueError(f"SyncNet parity failed for {sample_id}/{condition}")
    if condition != "natural" and natural_sync_record is not None:
        natural_matrix_path = Path(str(natural_sync_record["distance_path"]))
        natural_matrix = read_npz(natural_matrix_path, required=("dists",))["dists"]
        upstream["fixed_coordinate_local"] = _local_diagnostic(natural_sync_record, upstream, np.asarray(natural_matrix, dtype=np.float32), matrix, patches, audio_duration_s=audio_duration_s)
    else:
        upstream["fixed_coordinate_local"] = None
    atomic_write_json(record_path, upstream)
    return upstream
