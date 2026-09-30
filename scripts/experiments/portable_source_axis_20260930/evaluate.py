"""Run the frozen source-axis and level four-cell evaluation cohort.

This writer seals the protocol and ordered row manifest before model loading,
then saves per-clip context, visual embeddings, distance matrices, videos, and
row metrics. It does not compute cohort effect summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_AUX_SITE_PACKAGES = os.environ.get("TTS_SOURCE_AXIS_AUX_SITE_PACKAGES", "")
if _AUX_SITE_PACKAGES and Path(_AUX_SITE_PACKAGES).is_dir() and _AUX_SITE_PACKAGES not in sys.path:
    sys.path.append(_AUX_SITE_PACKAGES)

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import ops, scoring, wav2lip

SOURCE_AXIS_CONDITIONS = (
    "baseline", "full_plus", "full_minus", "orth_plus", "orth_minus",
    "random1", "random2", "random3",
)
LEVEL_CELLS = ("q00", "q10", "q01", "q11")
PRODUCTION_CODE_FILES = (
    "__init__.py", "evaluate.py", "ops.py", "scoring.py", "wav2lip.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha_file(path: str | Path) -> str:
    return scoring.sha256_file(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_revision(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _code_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {name: _sha_file(package_dir / name) for name in PRODUCTION_CODE_FILES}


def _read_pcm16(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_rate, pcm = wavfile.read(str(path))
    if sample_rate != ops.SAMPLE_RATE or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"expected mono PCM16 16 kHz WAV: {path} ({sample_rate},{pcm.shape},{pcm.dtype})")
    waveform = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
    return pcm, waveform, {
        "path": str(path.resolve()),
        "sha256": _sha_file(path),
        "sample_rate_hz": int(sample_rate),
        "channels": 1,
        "samples": int(pcm.size),
        "duration_s": float(pcm.size / sample_rate),
        "pcm_dtype": str(pcm.dtype),
        "peak_abs_pcm16": int(np.max(np.abs(pcm.astype(np.int32)))),
        "quantization": "none on RAW PCM; FIXED is retained as float32",
    }


def _mfcc_from_units(units: np.ndarray) -> np.ndarray:
    import python_speech_features

    rows = python_speech_features.mfcc(np.asarray(units, dtype=np.float64), ops.SAMPLE_RATE)
    return np.stack([np.asarray(row) for row in zip(*rows, strict=True)], axis=0).astype(np.float64, copy=False)


def _audio_forward(network: Any, units: np.ndarray, *, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count = int(len(units) // 640 - 5)
    if count < 1:
        raise ValueError("audio is shorter than the official SyncNet window support")
    features, metadata = scoring.audio_embeddings_from_syncnet_units(
        network, units, window_count=count, device=device, batch_size=batch_size,
    )
    mfcc = _mfcc_from_units(units)
    if ops.array_sha256(mfcc) != metadata["mfcc_sha256"]:
        raise AssertionError("saved MFCC differs from the SyncNet scoring frontend")
    return mfcc, features, metadata


def _row_unit(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    norm = np.sqrt(np.sum(value * value, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (value / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def _cross_distance_matrix(lip: np.ndarray, audio: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Save the full V×A row distance domain used by the temporal null."""
    import torch.nn.functional as F

    v = np.asarray(lip, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != a.shape[1] or not len(v) or not len(a):
        raise ValueError("cross-distance embeddings must be nonempty aligned matrices")
    vt = torch.from_numpy(np.ascontiguousarray(v)).to(device)
    at = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    cross = np.empty((len(v), len(a)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(v), 16):
            stop = min(start + 16, len(v))
            v_batch = vt[start:stop, None, :].expand(-1, len(a), -1)
            a_batch = at[None, :, :].expand(stop - start, -1, -1)
            cross[start:stop] = F.pairwise_distance(v_batch, a_batch, p=2, eps=1e-6).detach().cpu().numpy()
    zero = torch.zeros_like(vt)
    with torch.inference_mode():
        to_zero = F.pairwise_distance(vt, zero, p=2, eps=1e-6).detach().cpu().numpy().astype(np.float32)
    return cross, to_zero


def _lag_matrix_from_cross(cross: np.ndarray, to_zero: np.ndarray, query_n: int, audio_n: int, vshift: int = 15) -> np.ndarray:
    value = np.asarray(cross, dtype=np.float32)
    zero = np.asarray(to_zero, dtype=np.float32)
    if query_n < 1 or query_n > value.shape[0] or audio_n < 1 or audio_n > value.shape[1] or zero.shape != (value.shape[0],):
        raise ValueError("invalid query or audio domain for lag gathering")
    result = np.empty((query_n, 2 * vshift + 1), dtype=np.float32)
    for i in range(query_n):
        for column in range(2 * vshift + 1):
            j = i + column - vshift
            result[i, column] = value[i, j] if 0 <= j < audio_n else zero[i]
    return result


def _support_indices(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    guard15 = np.arange(min(15, count), max(min(15, count), count - 15), dtype=np.int32)
    guard20 = np.arange(min(20, count), max(min(20, count), count - 20), dtype=np.int32)
    return {"full": full, "valid": full.copy(), "guard20": guard20, "guard15": guard15}


def _metric_supports(matrix: np.ndarray, anchor_column: int) -> dict[str, Any]:
    supports = _support_indices(len(matrix))
    result: dict[str, Any] = {}
    for name, indices in supports.items():
        if len(indices) == 0:
            result[name] = {"status": "too_short", "support_n": 0, "window_indices": []}
        else:
            result[name] = {
                **scoring.curve_metrics(matrix[indices], anchor_column=anchor_column),
                "status": "ok",
                "window_indices": indices.tolist(),
            }
    return result


def _plan_rows(eval_rows: list[dict[str, Any]], tts_rows: dict[str, dict[str, Any]], face_ids: tuple[str, ...], input_root: Path) -> list[dict[str, Any]]:
    result = []
    for ordered_index, row in enumerate(eval_rows):
        sample_id = row["id"]
        speaker_id = row["speaker"]
        utterance = sample_id.split("_", 1)[1]
        natural_path = input_root / "audio" / speaker_id / f"{utterance}.wav"
        tts_path = input_root / "tts_outputs" / "audio_16k_canonical" / f"{sample_id}.wav"
        natural_hash = row["ref_sha256"]
        tts_hash = tts_rows[sample_id]["qwen_canonical_16k"]["sha256"]
        for face_id in face_ids:
            for source in ("natural", "tts"):
                for cell in LEVEL_CELLS:
                    visual_level = "raw" if cell in ("q00", "q01") else "fixed"
                    audio_level = "raw" if cell in ("q00", "q10") else "fixed"
                    row_id = f"{sample_id}__{face_id}__fourcell__{source}__{cell}"
                    result.append({
                        "ordered_eval_index": ordered_index,
                        "row_id": row_id,
                        "sample_id": sample_id,
                        "speaker_id": speaker_id,
                        "split": "evaluation",
                        "family": "fourcell",
                        "source": source,
                        "condition": cell,
                        "face_id": face_id,
                        "visual_id": f"{source}_{visual_level}",
                        "audio_key": f"{source}_audio_embeddings_{audio_level}_float32",
                        "visual_level": visual_level,
                        "audio_level": audio_level,
                        "natural_input_path": str(natural_path),
                        "natural_input_sha256": natural_hash,
                        "tts_input_path": str(tts_path),
                        "tts_input_sha256": tts_hash,
                        "context_npz": f"samples/{sample_id}_context.npz",
                        "visual_npz": f"visuals/{face_id}/{sample_id}__{source}_{visual_level}.npz",
                        "video_path": f"videos/{face_id}/{sample_id}__{source}_{visual_level}.mkv",
                        "score_crop_path": f"score_crops/{face_id}/{sample_id}__{source}_{visual_level}.mkv",
                        "score_npz": f"scores/{row_id}.npz",
                    })
            for condition in SOURCE_AXIS_CONDITIONS:
                visual_id = "natural_fixed" if condition == "baseline" else f"axis_{condition}"
                row_id = f"{sample_id}__{face_id}__source_axis__natural__{condition}"
                result.append({
                    "ordered_eval_index": ordered_index,
                    "row_id": row_id,
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "split": "evaluation",
                    "family": "source_axis",
                    "source": "natural",
                    "condition": condition,
                    "face_id": face_id,
                    "visual_id": visual_id,
                    "audio_key": "natural_audio_embeddings_raw_float32",
                    "visual_level": "fixed_plus_generator_intervention",
                    "audio_level": "raw",
                    "natural_input_path": str(natural_path),
                    "natural_input_sha256": natural_hash,
                    "tts_input_path": str(tts_path),
                    "tts_input_sha256": tts_hash,
                    "context_npz": f"samples/{sample_id}_context.npz",
                    "visual_npz": f"visuals/{face_id}/{sample_id}__{visual_id}.npz",
                    "video_path": f"videos/{face_id}/{sample_id}__{visual_id}.mkv",
                    "score_crop_path": f"score_crops/{face_id}/{sample_id}__{visual_id}.mkv",
                    "score_npz": f"scores/{row_id}.npz",
                })
    return result


def _seal_plan(args: argparse.Namespace, run_dir: Path, input_manifest: dict[str, Any], tts_manifest: dict[str, Any], frozen_manifest: dict[str, Any], calibration: dict[str, Any], d: np.ndarray, u: np.ndarray, eval_rows: list[dict[str, Any]], tts_rows: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_root = Path(args.input_root).resolve()
    face_paths = {"primary": Path(args.primary_face).resolve(), "secondary": Path(args.secondary_face).resolve()}
    face_entries = {key: {"path": str(path), "sha256": _sha_file(path)} for key, path in face_paths.items()}
    rows = _plan_rows(eval_rows, tts_rows, ("primary", "secondary"), input_root)
    rows_path = run_dir / "evaluation_rows.jsonl"
    protocol_path = run_dir / "evaluation_protocol.json"
    code_hashes = _code_hashes()
    spec_path = Path(args.spec).resolve()
    core = {
        "schema_version": 1,
        "experiment_id": "confirm-tts-source-mechanism-20260930",
        "status": "frozen_pre_effects",
        "created_at_utc": _utc_now(),
        "spec": {"path": str(spec_path), "sha256": _sha_file(spec_path)},
        "input_root": str(input_root),
        "input_manifest": {"path": str(Path(args.input_manifest).resolve()), "raw_sha256": _sha_file(args.input_manifest), "embedded_sha256": input_manifest.get("manifest_sha256")},
        "tts_pair_manifest": {"path": str(Path(args.tts_pair_manifest).resolve()), "raw_sha256": _sha_file(args.tts_pair_manifest), "embedded_sha256": tts_manifest.get("manifest_sha256")},
        "run_frozen_input_manifest": {"path": str(Path(args.frozen_input_manifest).resolve()), "raw_sha256": _sha_file(args.frozen_input_manifest), "embedded_sha256": frozen_manifest.get("manifest_sha256")},
        "input_identity_sha256": tts_manifest["input_identity_sha256"],
        "input_identity_rows_match": True,
        "calibration_summary": {"path": str(Path(args.calibration_summary).resolve()), "sha256": _sha_file(args.calibration_summary)},
        "calibration_axes": {"path": str(Path(args.calibration_axes).resolve()), "sha256": _sha_file(args.calibration_axes), "d_sha256": ops.array_sha256(d), "u_sha256": ops.array_sha256(u)},
        "global_anchor": calibration["global_anchor"],
        "face_images": face_entries,
        "cohort": {"evaluation_rows": len(eval_rows), "evaluation_speakers": len({row['speaker'] for row in eval_rows}), "two_clips_per_speaker": True},
        "row_count": len(rows),
        "production_code_sha256": code_hashes,
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
            "wav2lip_repo_revision": _git_revision(Path(args.wav2lip_root).resolve()),
            "syncnet_repo_revision": _git_revision(Path(args.syncnet_root).resolve()),
            "wav2lip_checkpoint_sha256": _sha_file(args.wav2lip_checkpoint),
            "syncnet_checkpoint_sha256": _sha_file(args.syncnet_checkpoint),
            "ffmpeg_path": str(Path(args.ffmpeg).resolve()), "ffmpeg_sha256": _sha_file(args.ffmpeg),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "unavailable",
        },
        "method": {
            "seed": 20260930,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "wav2lip_batch_size": args.wav2lip_batch_size,
            "syncnet_batch_size": args.syncnet_batch_size,
            "rms_target": ops.TARGET_RMS,
            "mel_frontend": "official Wav2Lip mel chunks; librosa 0.10 compatibility shim, defaults unchanged",
            "speech_mask": {"source": "FIXED waveform", "threshold_dbfs": -45.0, "window_samples": 3200, "mel_hop_samples": 200},
            "syncnet_audio_units": "RAW PCM16 integer amplitude; FIXED float32 promoted to float64 and multiplied by 32768, no round/cast/clipping",
            "syncnet_visual_frontend": "official FFmpeg JPEG extraction and OpenCV BGR decode",
            "lag_domain": {"vshift": 15, "columns": 31, "k_by_column": "column - 15", "official_offset_by_k": "-k", "audio_padding": "zero embedding outside the real A row domain"},
            "global_anchor_column": int(calibration["global_anchor"]["anchor_column"]),
            "level_cells": {"q00": ["RAW", "RAW"], "q10": ["FIXED", "RAW"], "q01": ["RAW", "FIXED"], "q11": ["FIXED", "FIXED"]},
            "fourcell_common_support": "intersection of valid V/A row counts for q00,q10,q01,q11 within each N/T recipient",
            "source_axis_conditions": list(SOURCE_AXIS_CONDITIONS),
            "source_axis_audio": "same full RAW-N A rows and original clock for every source-axis condition",
            "supports": {"full": "all common query rows", "valid": "all common V/A query rows before lag-edge guards", "guard20": "indices 20 through n-21", "guard15": "indices 15 through n-16"},
            "temporal_null_artifact": "score NPZ retains full V×A cross-distance matrix and V-to-zero boundary distances for source-axis rows; null permutations are computed by the independent CPU consumer after all artifacts are sealed",
            "artifact_scope": "all 80 evaluation clips, all four-cell and source-axis rows, primary and secondary faces",
        },
    }
    serialized_rows = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    row_hash = hashlib.sha256(serialized_rows.encode("utf-8")).hexdigest()
    if protocol_path.exists() or rows_path.exists():
        if not protocol_path.is_file() or not rows_path.is_file():
            raise RuntimeError("partially written protocol seal exists; refusing to overwrite")
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        existing_rows_hash = _sha_file(rows_path)
        if existing.get("row_manifest_sha256") != row_hash or existing_rows_hash != row_hash:
            raise RuntimeError("frozen evaluation row manifest differs from the existing seal")
        if existing.get("input_identity_sha256") != core["input_identity_sha256"]:
            raise RuntimeError("inputs differ from the existing evaluation protocol seal")
        if existing.get("production_code_sha256") != code_hashes:
            if not args.continuation_parent:
                raise RuntimeError("code differs from the existing evaluation protocol seal; explicit continuation parent is required")
            parent_dir = Path(args.continuation_parent).resolve()
            parent_protocol = parent_dir / "evaluation_protocol.json"
            parent_rows = parent_dir / "evaluation_rows.jsonl"
            if not parent_protocol.is_file() or not parent_rows.is_file():
                raise RuntimeError("continuation parent is missing its frozen protocol or rows")
            if _sha_file(parent_protocol) != _sha_file(protocol_path) or _sha_file(parent_rows) != row_hash:
                raise RuntimeError("child protocol/rows do not exactly match the continuation parent seal")
            parent_seal = json.loads(parent_protocol.read_text(encoding="utf-8"))
            if parent_seal.get("production_code_sha256") != existing.get("production_code_sha256"):
                raise RuntimeError("continuation parent protocol code hash differs from the child protocol")
        return rows, existing
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(serialized_rows, encoding="utf-8", newline="\n")
    core["row_manifest_sha256"] = row_hash
    core["row_manifest_path"] = str(rows_path)
    _atomic_json(protocol_path, core)
    return rows, core


def _audio_key(source: str, level: str) -> str:
    return f"{source}_audio_embeddings_{level}_float32"


def _encode_audio(model: Any, waveform: np.ndarray, wav2lip_root: Path, batch_size: int, device: str):
    mel, chunks, starts, metadata = wav2lip.mel_chunks(waveform, wav2lip_root)
    z = wav2lip.encode_latents(model, chunks, device=device, batch_size=batch_size)
    metadata.update({"latent_shape": list(z.shape), "latent_dtype": str(z.dtype), "latent_sha256": ops.array_sha256(z)})
    return mel, chunks, starts, z, metadata


def _make_context(
    row: dict[str, Any], natural_path: Path, tts_path: Path, *, model: Any, syncnet: Any,
    wav2lip_root: Path, d: np.ndarray, u: np.ndarray, wav2lip_batch_size: int,
    syncnet_batch_size: int, device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_pcm, n_raw, n_input_meta = _read_pcm16(natural_path)
    t_pcm, t_raw, t_input_meta = _read_pcm16(tts_path)
    if n_input_meta["sha256"] != row["ref_sha256"]:
        raise ValueError(f"natural input hash differs from frozen manifest for {row['id']}")
    if t_input_meta["sha256"] != row["tts_hash"]:
        raise ValueError(f"canonical TTS input hash differs from frozen manifest for {row['id']}")
    n_fixed, n_norm = ops.normalize_rms(n_raw)
    t_fixed, t_norm = ops.normalize_rms(t_raw)
    sources = {"natural": (n_pcm, n_raw, n_fixed, n_input_meta, n_norm), "tts": (t_pcm, t_raw, t_fixed, t_input_meta, t_norm)}
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"sample_id": row["id"], "speaker_id": row["speaker"], "split": row["split"], "inputs": {}, "sources": {}, "axis_diagnostics": {}}
    for source, (pcm, raw, fixed, input_meta, norm_meta) in sources.items():
        raw_mel, _, raw_starts, raw_z, raw_front = _encode_audio(model, raw, wav2lip_root, wav2lip_batch_size, device)
        fixed_mel, _, fixed_starts, fixed_z, fixed_front = _encode_audio(model, fixed, wav2lip_root, wav2lip_batch_size, device)
        if raw_starts != fixed_starts:
            raise AssertionError(f"RMS normalization changed Wav2Lip timing for {row['id']} {source}")
        mask, mask_meta = ops.fixed_speech_mask(fixed, fixed_starts)
        raw_units = pcm.astype(np.float64)
        raw_float_units, raw_identity_meta = ops.pcm_to_syncnet_integer_units(raw)
        if not np.array_equal(raw_units, raw_float_units):
            raise AssertionError(f"PCM16 float identity failed for {row['id']} {source}")
        fixed_units, fixed_units_meta = ops.pcm_to_syncnet_integer_units(fixed)
        raw_mfcc, raw_audio, raw_audio_meta = _audio_forward(syncnet, raw_units, device=device, batch_size=syncnet_batch_size)
        fixed_mfcc, fixed_audio, fixed_audio_meta = _audio_forward(syncnet, fixed_units, device=device, batch_size=syncnet_batch_size)
        prefix = source
        arrays.update({
            f"{prefix}_pcm16": pcm,
            f"{prefix}_raw_float32": raw,
            f"{prefix}_fixed_float32": fixed,
            f"{prefix}_mel_raw_float32": raw_mel.astype(np.float32, copy=False),
            f"{prefix}_mel_fixed_float32": fixed_mel.astype(np.float32, copy=False),
            f"{prefix}_mel_starts_int32": np.asarray(fixed_starts, dtype=np.int32),
            f"{prefix}_z_raw_float32": raw_z.astype(np.float32, copy=False),
            f"{prefix}_z_fixed_float32": fixed_z.astype(np.float32, copy=False),
            f"{prefix}_speech_mask_fixed_uint8": mask.astype(np.uint8),
            f"{prefix}_audio_units_raw_float64": raw_units,
            f"{prefix}_audio_units_fixed_float64": fixed_units.astype(np.float64, copy=False),
            f"{prefix}_audio_mfcc_raw_float64": raw_mfcc,
            f"{prefix}_audio_mfcc_fixed_float64": fixed_mfcc,
            f"{prefix}_audio_embeddings_raw_float32": raw_audio,
            f"{prefix}_audio_embeddings_fixed_float32": fixed_audio,
        })
        metadata["inputs"][source] = input_meta
        metadata["sources"][source] = {
            "rms_normalization": norm_meta,
            "mel_frontend": {"raw": raw_front, "fixed": fixed_front},
            "mask_fixed": mask_meta,
            "syncnet_units": {"raw_pcm16_identity_exact": True, "raw_identity": raw_identity_meta, "fixed": fixed_units_meta},
            "syncnet_audio": {"raw": raw_audio_meta, "fixed": fixed_audio_meta},
            "z_sha256": {"raw": ops.array_sha256(raw_z), "fixed": ops.array_sha256(fixed_z)},
        }

    z_n = arrays["natural_z_fixed_float32"]
    mask_n = arrays["natural_speech_mask_fixed_uint8"].astype(bool)
    arrays["axis_z_baseline_float32"] = z_n.copy()
    arrays["calibration_d_source_float64"] = np.asarray(d, dtype=np.float64)
    arrays["calibration_u_level_float64"] = np.asarray(u, dtype=np.float64)
    plus, plus_meta = ops.apply_source_direction(z_n, mask_n, d, u, sign=1, level_orthogonal=False)
    minus, minus_meta = ops.apply_source_direction(z_n, mask_n, d, u, sign=-1, level_orthogonal=False)
    orth_plus, orth_plus_meta = ops.apply_source_direction(z_n, mask_n, d, u, sign=1, level_orthogonal=True)
    orth_minus, orth_minus_meta = ops.apply_source_direction(z_n, mask_n, d, u, sign=-1, level_orthogonal=True)
    arrays["axis_z_full_plus_float32"] = plus
    arrays["axis_z_full_minus_float32"] = minus
    arrays["axis_z_orth_plus_float32"] = orth_plus
    arrays["axis_z_orth_minus_float32"] = orth_minus
    metadata["axis_diagnostics"].update({
        "baseline": {"z_sha256": ops.array_sha256(z_n), "mask_sha256": ops.array_sha256(mask_n.astype(np.uint8))},
        "full_plus": plus_meta,
        "full_minus": minus_meta,
        "orth_plus": orth_plus_meta,
        "orth_minus": orth_minus_meta,
    })
    directions, direction_meta = ops.random_control_directions(d, u)
    for i, (direction, direction_meta_row) in enumerate(zip(directions, direction_meta, strict=True), start=1):
        output, dose_meta = ops.apply_random_direction(z_n, mask_n, direction, u, orth_plus)
        condition = f"random{i}"
        arrays[f"axis_z_{condition}_float32"] = output
        arrays[f"axis_direction_{condition}_float64"] = direction
        metadata["axis_diagnostics"][condition] = {**direction_meta_row, **dose_meta}
    arrays["axis_z_baseline_float32"] = z_n.copy()
    metadata["array_sha256"] = {key: ops.array_sha256(value) for key, value in arrays.items()}
    metadata["input_hashes"] = {source: info["sha256"] for source, info in metadata["inputs"].items()}
    return arrays, metadata


def _write_context(run_dir: Path, sample_id: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> tuple[Path, Path]:
    npz_path = run_dir / "samples" / f"{sample_id}_context.npz"
    json_path = run_dir / "samples" / f"{sample_id}_context.json"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    metadata["context_npz"] = {"path": str(npz_path), "sha256": _sha_file(npz_path), "size_bytes": npz_path.stat().st_size}
    metadata["context_json"] = str(json_path)
    _atomic_json(json_path, metadata)
    return npz_path, json_path


def _load_context(npz_path: Path, json_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("context_npz", {}).get("sha256") != _sha_file(npz_path):
        raise RuntimeError(f"existing context hash mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, metadata


def _visual_specifications() -> list[tuple[str, str]]:
    return [
        ("natural_raw", "natural_z_raw_float32"), ("natural_fixed", "natural_z_fixed_float32"),
        ("tts_raw", "tts_z_raw_float32"), ("tts_fixed", "tts_z_fixed_float32"),
        ("axis_full_plus", "axis_z_full_plus_float32"), ("axis_full_minus", "axis_z_full_minus_float32"),
        ("axis_orth_plus", "axis_z_orth_plus_float32"), ("axis_orth_minus", "axis_z_orth_minus_float32"),
        ("axis_random1", "axis_z_random1_float32"), ("axis_random2", "axis_z_random2_float32"),
        ("axis_random3", "axis_z_random3_float32"),
    ]


def _load_or_make_visual(
    run_dir: Path, sample_id: str, face_id: str, visual_id: str, z: np.ndarray,
    frame: np.ndarray, face_meta: dict[str, Any], model: Any, face_cache: dict[int, list[Any]],
    syncnet: Any, args: argparse.Namespace, device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_path = run_dir / "visuals" / face_id / f"{sample_id}__{visual_id}.npz"
    meta_path = run_dir / "visuals" / face_id / f"{sample_id}__{visual_id}.json"
    if feature_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("feature_npz_sha256") == _sha_file(feature_path):
            with np.load(feature_path, allow_pickle=False) as archive:
                return archive["lip_embeddings_float32"], meta
    video_path = run_dir / "videos" / face_id / f"{sample_id}__{visual_id}.mkv"
    video_meta = wav2lip.render_video(
        model, z, frame, face_meta["generation_box_xyxy"], face_cache, video_path, args.ffmpeg,
        batch_size=args.wav2lip_batch_size, device=device,
    )
    crop_path = run_dir / "score_crops" / face_id / f"{sample_id}__{visual_id}.mkv"
    crop_meta = wav2lip.crop_score_video(video_path, face_meta["score_box"], crop_path, args.ffmpeg)
    jpeg_dir = run_dir / "official_jpeg" / face_id / sample_id / visual_id
    jpeg_frames, jpeg_meta = scoring.extract_official_jpegs(crop_path, jpeg_dir, args.ffmpeg)
    lip = scoring.lip_embeddings_from_frames(syncnet, jpeg_frames, device=device, batch_size=args.syncnet_batch_size)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, z_condition_float32=z, lip_embeddings_float32=lip)
    metadata = {
        "sample_id": sample_id, "face_id": face_id, "visual_id": visual_id,
        "video": video_meta, "score_crop": crop_meta, "official_jpeg": jpeg_meta,
        "lip_embedding_shape": list(lip.shape), "lip_embedding_sha256": ops.array_sha256(lip),
        "latent_sha256": ops.array_sha256(z), "feature_npz": str(feature_path),
        "feature_npz_sha256": _sha_file(feature_path),
    }
    _atomic_json(meta_path, metadata)
    return lip, metadata


def _build_plan_inputs(args: argparse.Namespace):
    input_manifest_path = Path(args.input_manifest).resolve()
    tts_manifest_path = Path(args.tts_pair_manifest).resolve()
    frozen_manifest_path = Path(args.frozen_input_manifest).resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    tts_manifest = json.loads(tts_manifest_path.read_text(encoding="utf-8"))
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if tts_manifest["source_manifest"]["raw_sha256"] != _sha_file(input_manifest_path):
        raise ValueError("selected input manifest raw hash differs from the frozen TTS pair manifest")
    if tts_manifest["run_frozen_input_manifest"]["raw_sha256"] != _sha_file(frozen_manifest_path):
        raise ValueError("run-frozen manifest raw hash differs from the frozen TTS pair manifest")
    if frozen_manifest.get("manifest_sha256") != tts_manifest["run_frozen_input_manifest"]["embedded_fingerprint"]:
        raise ValueError("run-frozen manifest fingerprint differs from the TTS pair manifest")
    identity_fields = tts_manifest["input_identity_fields"]
    source_identity = [{field: row.get(field) for field in identity_fields} for row in input_manifest["records"]]
    tts_identity = [record["input_identity"] for record in tts_manifest["records"]]
    identity_sha = hashlib.sha256(json.dumps(source_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if source_identity != tts_identity or identity_sha != tts_manifest["input_identity_sha256"]:
        raise ValueError("source/TTS ordered row identities differ from their frozen digest")
    eval_rows = [row for row in input_manifest["records"] if row["split"] == "evaluation"]
    tts_rows = {row["id"]: row for row in tts_manifest["records"] if row["split"] == "evaluation"}
    if len(eval_rows) != 80 or len(tts_rows) != 80 or len({row["speaker"] for row in eval_rows}) != 40:
        raise ValueError("evaluation cohort must contain 80 utterances from 40 speakers")
    for row in eval_rows:
        if row["id"] not in tts_rows:
            raise ValueError(f"missing canonical TTS record for {row['id']}")
        for key in ("id", "speaker", "text", "split", "ref_sha256"):
            if row.get(key) != tts_rows[row["id"]]["input_identity"].get(key):
                raise ValueError(f"input identity mismatch for {row['id']} field {key}")
    calibration_path = Path(args.calibration_summary).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "calibration_complete" or calibration.get("classification") != "sealed calibration; no evaluation effects":
        raise ValueError("calibration summary is not a completed sealed calibration")
    axes_path = Path(args.calibration_axes).resolve()
    if calibration["axes"]["npz_sha256"] != _sha_file(axes_path):
        raise ValueError("calibration axes NPZ hash differs from its summary")
    with np.load(axes_path, allow_pickle=False) as axes:
        d, u = axes["d_source"].astype(np.float64), axes["u_level"].astype(np.float64)
    if ops.array_sha256(d) != calibration["axes"]["d_sha256"] or ops.array_sha256(u) != calibration["axes"]["u_sha256"]:
        raise ValueError("calibration axis arrays differ from their summary hashes")
    return input_manifest, tts_manifest, frozen_manifest, calibration, d, u, eval_rows, tts_rows


def _run_face(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("samples", "visuals", "videos", "score_crops", "official_jpeg", "scores", "faces"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    input_manifest, tts_manifest, frozen_manifest, calibration, d, u, eval_rows, tts_rows = _build_plan_inputs(args)
    rows, protocol = _seal_plan(args, run_dir, input_manifest, tts_manifest, frozen_manifest, calibration, d, u, eval_rows, tts_rows)
    code_hashes = protocol["production_code_sha256"]
    runtime_code_hashes = _code_hashes()
    attempt_id = f"{args.face_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    attempts_path = run_dir / "evaluation_attempts.jsonl"
    completed_path = run_dir / "evaluation_clip_status.jsonl"
    results_path = run_dir / "evaluation_results.jsonl"
    previous_attempts = _read_jsonl(attempts_path)
    attempt_number = 1 + sum(item.get("event") == "run_started" and item.get("face_id") == args.face_id for item in previous_attempts)
    _append_jsonl(attempts_path, {
        "event": "run_started", "attempt_id": attempt_id, "attempt_number": attempt_number,
        "face_id": args.face_id, "started_at_utc": _utc_now(), "code_sha256": code_hashes,
        "runtime_code_sha256": runtime_code_hashes,
        "row_manifest_sha256": protocol["row_manifest_sha256"],
    })
    completed = {item["sample_id"] for item in _read_jsonl(completed_path) if item.get("face_id") == args.face_id and item.get("status") == "complete"}
    existing_results = {item.get("row_id") for item in _read_jsonl(results_path)}

    w2l_root = Path(args.wav2lip_root).resolve()
    sync_root = Path(args.syncnet_root).resolve()
    torch.manual_seed(20260930)
    np.random.seed(20260930)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    frontend = wav2lip.install_librosa_compatibility_shim()
    model, model_meta = wav2lip.load_model(w2l_root, args.wav2lip_checkpoint, device=device)
    syncnet, syncnet_meta = scoring.load_syncnet(sync_root, args.syncnet_checkpoint, device=device)

    face_path = Path(args.primary_face if args.face_id == "primary" else args.secondary_face).resolve()
    frame = cv2.imread(str(face_path))
    if frame is None:
        raise RuntimeError(f"cannot read face image: {face_path}")
    face_meta = wav2lip.detect_static_face(frame, w2l_root, device=device)
    face_meta.update({"path": str(face_path), "sha256": _sha_file(face_path), "width": int(frame.shape[1]), "height": int(frame.shape[0])})
    _atomic_json(run_dir / "faces" / f"{args.face_id}.json", {
        "face_id": args.face_id, "face": face_meta, "wav2lip_checkpoint_sha256": _sha_file(args.wav2lip_checkpoint),
        "code_sha256": code_hashes, "runtime_code_sha256": runtime_code_hashes,
        "continuation_parent": str(Path(args.continuation_parent).resolve()) if args.continuation_parent else None,
        "frontend_compatibility": frontend,
    })
    face_cache = wav2lip.cache_face_features(model, frame, face_meta["generation_box_xyxy"], range(1, args.wav2lip_batch_size + 1), device=device)

    parity_path = run_dir / "faces" / f"{args.face_id}_direct_decode_parity.json"
    if not parity_path.exists():
        first_row = eval_rows[0]
        first_natural_path = Path(args.input_root).resolve() / "audio" / first_row["speaker"] / f"{first_row['id'].split('_', 1)[1]}.wav"
        _, first_waveform, _ = _read_pcm16(first_natural_path)
        _, first_chunks, _, _ = wav2lip.mel_chunks(first_waveform, w2l_root)
        parity = wav2lip.parity_check_batches(
            model, first_chunks, frame, face_meta["generation_box_xyxy"], face_cache,
            batch_size=args.wav2lip_batch_size, device=device,
        )
        if not parity["uint8_prediction_exact"]:
            raise AssertionError(f"direct latent decode/cache changed Wav2Lip pixels: {parity}")
        parity.update({"sample_id": first_row["id"], "face_id": args.face_id, "source_level": "natural_raw", "scope": "head and tail batches only; not all frames"})
        _atomic_json(parity_path, parity)

    method_path = run_dir / "faces" / f"{args.face_id}_runtime.json"
    _atomic_json(method_path, {
        "face_id": args.face_id, "started_at_utc": _utc_now(), "device": torch.cuda.get_device_name(),
        "python": platform.python_version(), "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
        "wav2lip_model": model_meta, "syncnet_model": syncnet_meta, "frontend_compatibility": frontend,
        "runtime_code_sha256": runtime_code_hashes,
        "wav2lip_batch_size": args.wav2lip_batch_size, "syncnet_batch_size": args.syncnet_batch_size,
        "shared_face_encoder_batch_sizes": list(face_cache),
    })

    eval_by_id = {row["id"]: row for row in eval_rows}
    tts_by_id = tts_rows
    plan_by_id: dict[str, list[dict[str, Any]]] = {}
    for plan_row in rows:
        if plan_row["face_id"] == args.face_id:
            plan_by_id.setdefault(plan_row["sample_id"], []).append(plan_row)
    failed_sample = None
    for ordered_index, source_row in enumerate(eval_rows):
        sample_id = source_row["id"]
        if sample_id in completed:
            continue
        clip_started = time.perf_counter()
        failed_sample = sample_id
        try:
            tts_row = tts_by_id[sample_id]
            natural_path = Path(args.input_root).resolve() / "audio" / source_row["speaker"] / f"{sample_id.split('_', 1)[1]}.wav"
            tts_path = Path(args.input_root).resolve() / "tts_outputs" / "audio_16k_canonical" / f"{sample_id}.wav"
            context_npz = run_dir / "samples" / f"{sample_id}_context.npz"
            context_json = run_dir / "samples" / f"{sample_id}_context.json"
            if context_npz.is_file() and context_json.is_file():
                arrays, context_meta = _load_context(context_npz, context_json)
                if context_meta["input_hashes"]["natural"] != source_row["ref_sha256"] or context_meta["input_hashes"]["tts"] != tts_row["qwen_canonical_16k"]["sha256"]:
                    raise RuntimeError(f"existing audio context inputs differ from the sealed cohort: {sample_id}")
            else:
                context_row = {**source_row, "tts_hash": tts_row["qwen_canonical_16k"]["sha256"]}
                arrays, context_meta = _make_context(
                    context_row, natural_path, tts_path, model=model, syncnet=syncnet, wav2lip_root=w2l_root,
                    d=d, u=u, wav2lip_batch_size=args.wav2lip_batch_size, syncnet_batch_size=args.syncnet_batch_size, device=device,
                )
                _write_context(run_dir, sample_id, arrays, context_meta)

            z_map = {key: arrays[key] for _, key in _visual_specifications()}
            lips: dict[str, np.ndarray] = {}
            visual_meta: dict[str, dict[str, Any]] = {}
            for visual_id, latent_key in _visual_specifications():
                lips[visual_id], visual_meta[visual_id] = _load_or_make_visual(
                    run_dir, sample_id, args.face_id, visual_id, z_map[latent_key], frame, face_meta,
                    model, face_cache, syncnet, args, device,
                )
                visual_json = run_dir / "visuals" / args.face_id / f"{sample_id}__{visual_id}.json"
                visual_meta[visual_id]["metadata_path"] = str(visual_json)

            result_rows: list[dict[str, Any]] = []
            source_support: dict[str, int] = {}
            for source in ("natural", "tts"):
                source_support[source] = min(
                    len(lips[f"{source}_raw"]), len(lips[f"{source}_fixed"]),
                    len(arrays[_audio_key(source, "raw")]), len(arrays[_audio_key(source, "fixed")]),
                )
                if source_support[source] < 1:
                    raise ValueError(f"common SyncNet support is empty for {sample_id} {source}: {source_support[source]}")
            axis_support = min(
                *(len(lips["natural_fixed"] if condition == "baseline" else lips[f"axis_{condition}"]) for condition in SOURCE_AXIS_CONDITIONS),
                len(arrays[_audio_key("natural", "raw")]),
            )
            if axis_support < 41:
                raise ValueError(f"source-axis SyncNet support too short for {sample_id}: {axis_support}")

            pair_geometry: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            def geometry(visual_id: str, audio_key: str):
                key = (visual_id, audio_key)
                if key not in pair_geometry:
                    lip = lips[visual_id]
                    audio = arrays[audio_key]
                    cross_raw, zero_raw = _cross_distance_matrix(lip, audio, device)
                    cross_unit, zero_unit = _cross_distance_matrix(_row_unit(lip), _row_unit(audio), device)
                    pair_geometry[key] = cross_raw, zero_raw, cross_unit, zero_unit
                return pair_geometry[key]

            plan_rows = plan_by_id[sample_id]
            for row_plan in plan_rows:
                if row_plan["family"] == "fourcell":
                    support = source_support[row_plan["source"]]
                else:
                    support = axis_support
                audio_key = row_plan["audio_key"]
                visual_id = row_plan["visual_id"]
                cross_raw, zero_raw, cross_unit, zero_unit = geometry(visual_id, audio_key)
                audio_n = len(arrays[audio_key])
                # The score uses the same common truncated support on both V
                # and A. The saved cross matrix still retains the full A row
                # domain for the predeclared temporal permutation null.
                matrix_raw = _lag_matrix_from_cross(cross_raw, zero_raw, support, support)
                matrix_unit = _lag_matrix_from_cross(cross_unit, zero_unit, support, support)
                indices = _support_indices(support)
                score_path = run_dir / row_plan["score_npz"]
                score_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    score_path,
                    distance_matrix_raw_float32=matrix_raw,
                    distance_matrix_unit_float32=matrix_unit,
                    cross_distance_raw_float32=cross_raw,
                    cross_distance_unit_float32=cross_unit,
                    cross_distance_to_zero_raw_float32=zero_raw,
                    cross_distance_to_zero_unit_float32=zero_unit,
                    common_support_indices_int32=np.arange(support, dtype=np.int32),
                    full_support_indices_int32=indices["full"],
                    valid_support_indices_int32=indices["valid"],
                    guard20_indices_int32=indices["guard20"],
                    guard15_indices_int32=indices["guard15"],
                    lag_columns_int32=np.arange(31, dtype=np.int32),
                    lag_k_int32=np.arange(-15, 16, dtype=np.int32),
                )
                anchor_column = int(calibration["global_anchor"]["anchor_column"])
                raw_metrics = _metric_supports(matrix_raw, anchor_column)
                unit_metrics = _metric_supports(matrix_unit, anchor_column)
                row_result = {
                    **row_plan,
                    "status": "scored",
                    "attempt_id": attempt_id,
                    "runtime_code_sha256": runtime_code_hashes,
                    "recorded_at_utc": _utc_now(),
                    "context_npz": str(context_npz),
                    "context_npz_sha256": _sha_file(context_npz),
                    "context_json": str(context_json),
                    "context_json_sha256": _sha_file(context_json),
                    "visual_npz": str(run_dir / row_plan["visual_npz"]),
                    "visual_npz_sha256": _sha_file(run_dir / row_plan["visual_npz"]),
                    "visual_metadata": visual_meta[visual_id],
                    "score_npz": str(score_path),
                    "score_npz_sha256": _sha_file(score_path),
                    "audio_embedding_key": audio_key,
                    "audio_embedding_sha256": ops.array_sha256(arrays[audio_key]),
                    "lip_embedding_sha256": ops.array_sha256(lips[visual_id]),
                    "common_support_n": int(support),
                    "audio_row_domain_n": int(audio_n),
                    "scoring_audio_domain_n": int(support),
                    "visual_query_domain_n": int(len(lips[visual_id])),
                    "support_policy": "same full row indices shared by all conditions within this recipient/family; lags outside real A domain use official zero-row padding",
                    "anchor": {"source": "calibration natural q10", "column": anchor_column, "k": anchor_column - 15, "official_offset_frames": 15 - anchor_column},
                    "raw_geometry": raw_metrics,
                    "row_unit_geometry": unit_metrics,
                    "input_pcm_identity": {
                        "natural_raw_exact": True,
                        "tts_raw_exact": True,
                        "natural_syncnet_units_sha256": ops.array_sha256(arrays["natural_audio_units_raw_float64"]),
                        "tts_syncnet_units_sha256": ops.array_sha256(arrays["tts_audio_units_raw_float64"]),
                    },
                    "axis_diagnostics": context_meta["axis_diagnostics"].get(row_plan["condition"]) if row_plan["family"] == "source_axis" else None,
                }
                result_rows.append(row_result)
                if row_plan["row_id"] not in existing_results:
                    _append_jsonl(results_path, row_result)
                    existing_results.add(row_plan["row_id"])

            completed_record = {
                "event": "clip_complete", "status": "complete", "attempt_id": attempt_id,
                "attempt_number": attempt_number, "face_id": args.face_id,
                "runtime_code_sha256": runtime_code_hashes,
                "ordered_eval_index": ordered_index, "sample_id": sample_id,
                "speaker_id": source_row["speaker"], "score_rows_written": len(result_rows),
                "expected_score_rows": len(plan_rows), "context_npz": str(context_npz),
                "context_npz_sha256": _sha_file(context_npz),
                "elapsed_seconds": float(time.perf_counter() - clip_started), "completed_at_utc": _utc_now(),
            }
            _append_jsonl(completed_path, completed_record)
            _append_jsonl(attempts_path, completed_record)
            completed.add(sample_id)
            print(json.dumps({"event": "clip_complete", "face_id": args.face_id, "sample_id": sample_id, "elapsed_seconds": completed_record["elapsed_seconds"]}), flush=True)
        except BaseException as exc:
            error_record = {
                "event": "clip_failed", "status": "failed", "attempt_id": attempt_id,
                "attempt_number": attempt_number, "face_id": args.face_id,
                "ordered_eval_index": ordered_index, "sample_id": sample_id,
                "speaker_id": source_row.get("speaker"), "completed_score_rows_for_clip": sum(1 for item in _read_jsonl(results_path) if item.get("sample_id") == sample_id and item.get("face_id") == args.face_id),
                "expected_score_rows": len(plan_by_id.get(sample_id, [])),
                "failed_at_utc": _utc_now(), "elapsed_seconds": float(time.perf_counter() - clip_started),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(limit=8), "code_sha256": code_hashes,
                "runtime_code_sha256": runtime_code_hashes,
            }
            _append_jsonl(attempts_path, error_record)
            raise

    final = {
        "status": "face_complete",
        "classification": "sealed row-level scores; no cohort effect summaries",
        "face_id": args.face_id,
        "run_dir": str(run_dir),
        "protocol_path": str(run_dir / "evaluation_protocol.json"),
        "protocol_sha256": _sha_file(run_dir / "evaluation_protocol.json"),
        "row_manifest_path": str(run_dir / "evaluation_rows.jsonl"),
        "row_manifest_sha256": _sha_file(run_dir / "evaluation_rows.jsonl"),
        "results_path": str(results_path),
        "production_code_sha256": code_hashes,
        "runtime_code_sha256": runtime_code_hashes,
        "continuation_parent": str(Path(args.continuation_parent).resolve()) if args.continuation_parent else None,
        "completed_clips": len(completed), "expected_clips": len(eval_rows),
        "expected_rows_for_face": sum(row["face_id"] == args.face_id for row in rows),
        "elapsed_seconds_this_attempt": float(time.perf_counter() - started),
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(run_dir / f"evaluation_summary_{args.face_id}.json", final)
    _atomic_json(run_dir / "evaluation_status.json", {
        "last_face_id": args.face_id, "last_status": "face_complete", "updated_at_utc": _utc_now(),
        "primary_complete": (run_dir / "evaluation_summary_primary.json").is_file(),
        "secondary_complete": (run_dir / "evaluation_summary_secondary.json").is_file(),
    })
    return final


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal evaluation runner")
    return _run_face(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--tts-pair-manifest", required=True)
    parser.add_argument("--frozen-input-manifest", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-axes", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--secondary-face", required=True)
    parser.add_argument("--face-id", choices=("primary", "secondary"), default="primary")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--wav2lip-batch-size", type=int, default=32)
    parser.add_argument("--syncnet-batch-size", type=int, default=20)
    parser.add_argument("--continuation-parent", default=None, help="Parent run directory whose exact frozen protocol and rows this run continues")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
