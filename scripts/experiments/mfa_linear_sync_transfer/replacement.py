"""Frozen Wav2Lip driver arms and complete natural-audio replacement matrix."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..mfa_linear_real_video_sync.evaluate import curve_metrics, mux_condition_once, official_file_curve
from ..mfa_linear_real_video_sync.protocol import extract_official_bgr_frames, frame_hashes, materialize_ffv1_once, read_fixed_video_frames, sha256_file, write_json_once
from .config import (
    ARMS,
    EVAL_ARMS,
    EXPECTED_MATRIX_CELLS,
    GAIN_MIN,
    NO_REPLACEMENT,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_OBSERVED,
    RENDER_INCOMPLETE,
    TARGET_GAP,
    WAV2LIP_CHECKPOINT_SHA256,
)
from .protocol import TransferProtocolError


def _wav2lip_python() -> str:
    candidate = Path.home() / ".venvs/wav2lip/bin/python"
    return str(candidate if candidate.is_file() else Path(sys.executable))


def _video_code_hash(repo: Path) -> str:
    return sha256_file(repo / "third_party/Wav2Lip/inference.py")


def render_arm_once(
    row: Mapping[str, Any],
    arm: str,
    driver_wav: str | Path,
    output_root: str | Path,
    *,
    repo: str | Path,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown driver arm: {arm}")
    root = Path(output_root).resolve()
    repo_path = Path(repo).resolve()
    sid = str(row["sample_id"])
    work = root / "04_replacement/renders" / sid / arm
    generated = work / "renderer.mp4"
    log = work / "renderer.log"
    canonical_path = work / "video.avi"
    if generated.exists() or canonical_path.exists() or log.exists():
        raise FileExistsError(f"render arm already exists: {sid}/{arm}")
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = repo_path / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
    if sha256_file(checkpoint) != WAV2LIP_CHECKPOINT_SHA256:
        raise TransferProtocolError("Wav2Lip checkpoint hash changed")
    command = [
        _wav2lip_python(), str(repo_path / "third_party/Wav2Lip/inference.py"),
        "--checkpoint_path", str(checkpoint), "--face", str(Path(row["visual"]["path"]).resolve()),
        "--audio", str(Path(driver_wav).resolve()), "--outfile", str(generated),
        "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth",
    ]
    completed = subprocess.run(command, cwd=str(work), check=False, capture_output=True, text=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not generated.is_file():
        raise RuntimeError(f"Wav2Lip render failed for {sid}/{arm}: returncode={completed.returncode}")
    frames = read_fixed_video_frames(generated)
    visual = materialize_ffv1_once(canonical_path, frames)
    scoring = extract_official_bgr_frames(canonical_path, work / "scoring_frames")
    visual["scoring_bgr_frame_sha256"] = frame_hashes(scoring)
    result = {
        "status": "complete",
        "sample_id": sid,
        "driver_arm": arm,
        "driver_wav": str(Path(driver_wav).resolve()),
        "driver_wav_sha256": sha256_file(driver_wav),
        "renderer_video": str(generated.resolve()),
        "renderer_video_sha256": sha256_file(generated),
        "canonical_video": visual,
        "wav2lip_checkpoint": str(checkpoint.resolve()),
        "wav2lip_checkpoint_sha256": sha256_file(checkpoint),
        "wav2lip_code_sha256": _video_code_hash(repo_path),
        "python": command[0],
        "command": command,
        "frame_count": visual["frame_count"],
        "geometry": [visual["height"], visual["width"]],
        "renderer_audio_discarded": True,
    }
    write_json_once(work / "render.json", result)
    return result


def render_p0_seam(row: Mapping[str, Any], driver_wavs: Mapping[str, str | Path], output_root: str | Path, *, repo: str | Path) -> dict[str, Any]:
    results = [render_arm_once(row, arm, driver_wavs[arm], output_root, repo=repo) for arm in ARMS]
    if len({int(item["frame_count"]) for item in results}) != 1 or len({tuple(item["geometry"]) for item in results}) != 1:
        raise TransferProtocolError(f"{RENDER_INCOMPLETE}: P0 arm support mismatch")
    return {"status": "GO", "fixture_sample_id": row["sample_id"], "arms": results, "engineering_only": True}


def materialize_matrix(
    rows: Sequence[Mapping[str, Any]],
    render_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    eval_wavs: Mapping[str, Mapping[str, str | Path]],
    syncnet: nn.Module,
    output_root: str | Path,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["sample_id"])
        for driver in ARMS:
            visual = render_results[sid][driver]["canonical_video"]
            for evaluation in EVAL_ARMS:
                key = f"G_{driver}_E_{evaluation}"
                root = Path(output_root).resolve() / "04_replacement/matrix" / sid / key
                audio_path = Path(eval_wavs[sid][evaluation]).resolve()
                condition = root / "condition.avi"
                mux_condition_once(visual["path"], audio_path, condition)
                pcm_sha = _pcm_hash(audio_path)
                score = official_file_curve(
                    syncnet,
                    condition,
                    expected_frame_hashes=visual["decoded_bgr_frame_sha256"],
                    expected_scoring_frame_hashes=visual.get("scoring_bgr_frame_sha256"),
                    expected_pcm_sha256=pcm_sha,
                    device=device,
                )
                metrics = curve_metrics(score["curve"])
                cell = {
                    "status": "complete", "sample_id": sid, "key": key,
                    "driver_arm": driver, "evaluation_arm": evaluation,
                    "video_sha256": visual["sha256"],
                    "video_frame_hashes": visual["decoded_bgr_frame_sha256"],
                    "evaluation_audio": str(audio_path), "evaluation_audio_sha256": sha256_file(audio_path),
                    "evaluation_pcm_sha256": pcm_sha, "score": score, "metrics": metrics,
                    "diagnostic_only": evaluation != "N" or driver == "N",
                    "authoritative_replacement": evaluation == "N" and driver in {"B", "C"},
                }
                write_json_once(root / "cell.json", cell)
                cells.append(cell)
    if len(cells) != len(rows) * len(ARMS) * len(EVAL_ARMS):
        raise TransferProtocolError(f"{REPLACEMENT_NOT_EVALUATED}: matrix denominator changed")
    return cells


def _pcm_hash(path: Path) -> str:
    import scipy.io.wavfile as wavfile
    rate, pcm = wavfile.read(path)
    if int(rate) != 16_000 or pcm.dtype != np.int16 or pcm.ndim != 1 or pcm.size != 61_440:
        raise TransferProtocolError("evaluation audio is not exact signed PCM16")
    return hashlib.sha256(np.ascontiguousarray(pcm).tobytes()).hexdigest()


def replacement_record_evidence(cells: Sequence[Mapping[str, Any]], sample_id: str, oracle_shift: int) -> dict[str, Any]:
    by_key = {str(cell["key"]): cell for cell in cells if str(cell.get("sample_id")) == sample_id}
    expected = {f"G_{driver}_E_{evaluation}" for driver in ARMS for evaluation in EVAL_ARMS}
    if set(by_key) != expected:
        return {"sample_id": sample_id, "engineering_valid": False, "scientific_success": False, "reason": "incomplete cell keys"}
    base = by_key["G_B_E_N"]["metrics"]
    candidate = by_key["G_C_E_N"]["metrics"]
    d_gain = float(base["sync_d"] - candidate["sync_d"])
    c_gain = float(candidate["sync_c"] - base["sync_c"])
    candidate_match = candidate["best_signed_shift"] == int(oracle_shift) and candidate["best_second_gap"] > TARGET_GAP
    engineering = all(cell.get("status") == "complete" and cell.get("score", {}).get("window_count") == 91 for cell in by_key.values())
    success = engineering and d_gain >= GAIN_MIN and c_gain >= GAIN_MIN and candidate_match
    return {
        "sample_id": sample_id,
        "engineering_valid": engineering,
        "scientific_success": success,
        "replacement_d_gain": d_gain,
        "replacement_c_gain": c_gain,
        "oracle_shift": int(oracle_shift),
        "candidate_oracle_match": candidate_match,
        "candidate_best_second_gap": candidate["best_second_gap"],
        "cells": sorted(by_key),
        "diagnostic_diagonal": {
            "baseline_native": by_key["G_B_E_B"]["metrics"],
            "candidate_native": by_key["G_C_E_C"]["metrics"],
        },
    }


def decide_replacement(record_rows: Sequence[Mapping[str, Any]], *, cell_count: int | None = None) -> dict[str, Any]:
    actual_cells = EXPECTED_MATRIX_CELLS if cell_count is None else int(cell_count)
    if len(record_rows) != 8 or actual_cells != EXPECTED_MATRIX_CELLS:
        return {"stage": "P4_STRICT_NATURAL_REPLACEMENT", "status": REPLACEMENT_NOT_EVALUATED, "pass": False, "engineering_status": "incomplete", "records": list(record_rows)}
    if not all(bool(row.get("engineering_valid")) for row in record_rows):
        return {"stage": "P4_STRICT_NATURAL_REPLACEMENT", "status": REPLACEMENT_NOT_EVALUATED, "pass": False, "engineering_status": "invalid_or_incomplete", "records": list(record_rows)}
    d = [float(row["replacement_d_gain"]) for row in record_rows]
    c = [float(row["replacement_c_gain"]) for row in record_rows]
    successes = sum(bool(row.get("scientific_success")) for row in record_rows)
    passed = successes >= 6 and float(np.median(d)) >= GAIN_MIN and float(np.median(c)) >= GAIN_MIN
    return {
        "stage": "P4_STRICT_NATURAL_REPLACEMENT",
        "status": REPLACEMENT_OBSERVED if passed else NO_REPLACEMENT,
        "pass": passed,
        "engineering_status": "complete",
        "record_count": 8,
        "cell_count": EXPECTED_MATRIX_CELLS,
        "success_count": successes,
        "median_replacement_d_gain": float(np.median(d)),
        "median_replacement_c_gain": float(np.median(c)),
        "records": list(record_rows),
        "claim_scope": "empirical strict natural-audio replacement transfer through one frozen Wav2Lip/SyncNet fixed-crop protocol" if passed else "complete strict replacement evidence did not pass the preregistered gate",
    }
