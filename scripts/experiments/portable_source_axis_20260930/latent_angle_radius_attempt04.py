"""Sealed four-cell direction/radius factorization of saved source-axis latents.

This producer reads only retry3's verified natural baseline and exact
orth_plus latent trajectories, derives the two factorized trajectories, then
freshly renders and scores all four cells against unchanged RAW natural audio.
It writes row-level artifacts only; cohort inference is an independent step.
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
from typing import Any

_AUX_SITE_PACKAGES = os.environ.get("TTS_SOURCE_AXIS_AUX_SITE_PACKAGES", "")
if _AUX_SITE_PACKAGES and Path(_AUX_SITE_PACKAGES).is_dir() and _AUX_SITE_PACKAGES not in sys.path:
    sys.path.append(_AUX_SITE_PACKAGES)

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import evaluate, ops, scoring, wav2lip

EXPECTED_SPEC_SHA256 = "631e76042a523f598e9135de7ff9bf328f3656eabc800a0aa85f38f66bc01252"
EXPECTED_GEOMETRY_SHA256 = "800d0a0d040df2337249958c907b28eb13ef75173392778ce7c674eed1fce17e"
EXPECTED_SOURCE_PROTOCOL_SHA256 = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
EXPECTED_SOURCE_ROWS_SHA256 = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
EXPECTED_INPUT_MANIFEST_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
EXPECTED_ANCHOR = {"k": 3, "column": 18, "official_offset": -3}
CONDITIONS = ("q00", "q10", "q01", "q11")
VISUAL_IDS = {
    "q00": "latent_radius_q00",
    "q10": "latent_radius_q10",
    "q01": "latent_radius_q01",
    "q11": "latent_radius_q11",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _resolve_ref(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("latent_angle_radius.py", "__init__.py", "evaluate.py", "ops.py", "scoring.py", "wav2lip.py")
    return {name: _sha(package / name) for name in names}


def _flags() -> dict[str, bool]:
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _set_flags() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _instrument(events: list[dict[str, Any]], active: dict[str, str | None]):
    originals = []
    for module, name in (
        (wav2lip, "cache_face_features"),
        (wav2lip, "decode_latents"),
        (scoring, "audio_embeddings_from_syncnet_units"),
        (scoring, "lip_embeddings_from_frames"),
        (evaluate, "_cross_distance_matrix"),
    ):
        original = getattr(module, name)
        originals.append((module, name, original))

        def wrapped(*args, __module=module.__name__, __name=name, __original=original, **kwargs):
            _set_flags()
            events.append({
                "operation": __name,
                "module": __module,
                "sample_id": active.get("sample_id"),
                "condition": active.get("condition"),
                "flags_at_call": _flags(),
                "time_monotonic": time.monotonic(),
            })
            return __original(*args, **kwargs)

        setattr(module, name, wrapped)
    return originals


def _per_frame_norm(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return np.linalg.norm(array.reshape(len(array), -1).astype(np.float64), axis=1)


def _ratio_stats(ratio: np.ndarray) -> dict[str, Any]:
    value = np.asarray(ratio, dtype=np.float64)
    return {
        "count": int(value.size),
        "min": float(value.min()),
        "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95)),
        "max": float(value.max()),
        "outside_0_5_to_2_fraction": float(np.mean((value < 0.5) | (value > 2.0))),
    }


def _u_projection(delta: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.sum(
        np.asarray(delta, dtype=np.float64).reshape(len(delta), -1)
        * np.asarray(u, dtype=np.float64).reshape(1, -1), axis=1, dtype=np.float64,
    )


def _frame_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    b = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    numerator = np.sum(a * b, axis=1, dtype=np.float64)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / denominator


def _summarize(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(value)), "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95)), "max_abs": float(np.max(np.abs(value))),
    }


def _build_factorizations(z0: np.ndarray, z1: np.ndarray, u: np.ndarray, mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    z0 = np.asarray(z0, dtype=np.float32)
    z1 = np.asarray(z1, dtype=np.float32)
    if z0.shape != z1.shape or z0.ndim < 2 or len(z0) == 0:
        raise ValueError(f"z0/z1 shapes are incompatible: {z0.shape} vs {z1.shape}")
    if not np.isfinite(z0).all() or not np.isfinite(z1).all() or np.any(z0 < 0) or np.any(z1 < 0):
        raise ValueError("z0/z1 must be finite nonnegative float32 arrays")
    if len(mask) != len(z0):
        raise ValueError(f"original FIXED N speech mask length differs from latent clock: {len(mask)} vs {len(z0)}")
    r0 = _per_frame_norm(z0)
    r1 = _per_frame_norm(z1)
    if np.any(r0 <= 0) or np.any(r1 <= 0):
        raise ValueError("zero-radius frame in saved z0/z1; preserve as planned failure")
    ratio_q10 = r0 / r1
    ratio_q01 = r1 / r0
    scale_shape = (len(r0),) + (1,) * (z0.ndim - 1)
    q10_pre = (z1.astype(np.float64) * ratio_q10.reshape(scale_shape)).astype(np.float64, copy=False)
    q01_pre = (z0.astype(np.float64) * ratio_q01.reshape(scale_shape)).astype(np.float64, copy=False)
    q10 = q10_pre.astype(np.float32)
    q01 = q01_pre.astype(np.float32)
    q10_norm = _per_frame_norm(q10)
    q01_norm = _per_frame_norm(q01)
    rel_q10 = np.abs(q10_norm - r0) / r0
    rel_q01 = np.abs(q01_norm - r1) / r1
    if float(rel_q10.max(initial=0.0)) > 1e-6 or float(rel_q01.max(initial=0.0)) > 1e-6:
        raise ValueError(f"float32 radius target exceeded 1e-6 relative tolerance: q10={rel_q10.max()}, q01={rel_q01.max()}")
    u64 = np.asarray(u, dtype=np.float64)
    if u64.size != int(np.prod(z0.shape[1:])):
        raise ValueError(f"saved level axis shape {u64.shape} does not match latent frame shape {z0.shape[1:]}")
    q10_pre_up = _u_projection(q10_pre - z0.astype(np.float64), u64)
    q10_post_up = _u_projection(q10.astype(np.float64) - z0.astype(np.float64), u64)
    q01_pre_up = _u_projection(q01_pre - z0.astype(np.float64), u64)
    q01_post_up = _u_projection(q01.astype(np.float64) - z0.astype(np.float64), u64)
    result = {"q00": z0.copy(), "q10": q10, "q01": q01, "q11": z1.copy()}
    if not np.array_equal(result["q00"], z0) or not np.array_equal(result["q11"], z1):
        raise AssertionError("q00/q11 endpoint copies are not bit exact")
    diagnostics = {
        "frame_count": len(z0),
        "z0_norm": _summarize(r0), "z1_norm": _summarize(r1),
        "z1_to_z0_radius_ratio": _ratio_stats(r1 / r0),
        "q10_scale_z1_to_z0_radius": _ratio_stats(ratio_q10),
        "q01_scale_z0_to_z1_radius": _ratio_stats(ratio_q01),
        "relative_radius_error": {"q10_max": float(rel_q10.max(initial=0.0)), "q01_max": float(rel_q01.max(initial=0.0))},
        "cosine_z0_z1": _summarize(_frame_cosine(z0, z1)),
        "cosine_z0_q10": _summarize(_frame_cosine(z0, q10)),
        "cosine_z0_q01": _summarize(_frame_cosine(z0, q01)),
        "cosine_z1_q10": _summarize(_frame_cosine(z1, q10)),
        "cosine_z1_q01": _summarize(_frame_cosine(z1, q01)),
        "actual_delta_norms": {
            name: _summarize(_per_frame_norm(value - z0)) for name, value in result.items()
        },
        "u_projection_pre_float32": {"q10": _summarize(q10_pre_up), "q01": _summarize(q01_pre_up)},
        "u_projection_post_float32": {"q10": _summarize(q10_post_up), "q01": _summarize(q01_post_up)},
        "float32_projection_change": {"q10_max_abs": float(np.max(np.abs(q10_post_up - q10_pre_up))), "q01_max_abs": float(np.max(np.abs(q01_post_up - q01_pre_up)))},
        "u_projection_mean_abs": {
            "q00": _summarize(_u_projection(result["q00"] - z0, u64)),
            "q10": _summarize(q10_post_up),
            "q01": _summarize(q01_post_up),
            "q11": _summarize(_u_projection(result["q11"] - z0, u64)),
        },
        "original_fixed_n_activity_mask": {
            "active_frames": int(np.count_nonzero(mask)), "total_frames": int(len(mask)),
            "q0_mean_norm_active": float(np.mean(r0[mask])) if np.any(mask) else None,
            "q1_mean_norm_active": float(np.mean(r1[mask])) if np.any(mask) else None,
            "q10_mean_norm_active": float(np.mean(q10_norm[mask])) if np.any(mask) else None,
            "q01_mean_norm_active": float(np.mean(q01_norm[mask])) if np.any(mask) else None,
        },
    }
    return result, diagnostics


def _read_pcm16(path: Path, expected_sha: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    file_sha = _sha(path)
    if file_sha != expected_sha:
        raise ValueError(f"RAW natural WAV SHA differs from frozen evaluation plan: {path}")
    rate, pcm = wavfile.read(str(path))
    if rate != ops.SAMPLE_RATE or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"expected mono 16k PCM16 natural audio, got rate={rate}, shape={pcm.shape}, dtype={pcm.dtype}")
    waveform = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
    units, units_meta = ops.pcm_to_syncnet_integer_units(waveform)
    if not np.array_equal(units, pcm.astype(np.float64)):
        raise AssertionError("RAW PCM16 to SyncNet float64 identity failed")
    return pcm, units, {
        "path": str(path.resolve()), "sha256": file_sha, "sample_rate_hz": int(rate),
        "samples": int(pcm.size), "duration_s": float(pcm.size / rate), "pcm_dtype": str(pcm.dtype),
        "syncnet_units": units_meta,
    }


def _source_inventory(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_root = Path(args.source_run_dir).resolve(strict=True)
    source_protocol_path = source_root / "evaluation_protocol.json"
    source_plan_path = source_root / "evaluation_rows.jsonl"
    source_results_path = source_root / "evaluation_results.jsonl"
    protocol_sha = _sha(source_protocol_path)
    plan_sha = _sha(source_plan_path)
    if protocol_sha != EXPECTED_SOURCE_PROTOCOL_SHA256 or plan_sha != EXPECTED_SOURCE_ROWS_SHA256:
        raise ValueError(f"retry3 source seals mismatch: protocol={protocol_sha}, rows={plan_sha}")
    protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_plan = _read_jsonl(source_plan_path)
    results = _read_jsonl(source_results_path)
    plan_by_sample = {
        row["sample_id"]: row for row in source_plan
        if row.get("family") == "source_axis" and row.get("source") == "natural" and row.get("face_id") == "primary" and row.get("condition") == "baseline"
    }
    result_by_key = {
        (row.get("sample_id"), row.get("condition")): row for row in results
        if row.get("family") == "source_axis" and row.get("source") == "natural" and row.get("face_id") == "primary" and row.get("condition") in {"baseline", "orth_plus"}
    }
    source_manifest_path = Path(args.input_manifest).resolve(strict=True)
    source_manifest_sha = _sha(source_manifest_path)
    if source_manifest_sha != EXPECTED_INPUT_MANIFEST_SHA256:
        raise ValueError(f"frozen input manifest SHA mismatch: {source_manifest_sha}")
    input_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    natural_rows = [row for row in input_manifest["records"] if row.get("split") == "evaluation"]
    if len(natural_rows) != 80 or len({row["speaker"] for row in natural_rows}) != 40:
        raise ValueError("direction/radius runner requires frozen 80-row / 40-speaker natural cohort")
    if len(plan_by_sample) != 80:
        raise ValueError(f"retry3 source-axis baseline rows are incomplete: {len(plan_by_sample)}/80")
    if len(result_by_key) != 160:
        raise ValueError(f"retry3 source-axis baseline/orth_plus result rows are incomplete: {len(result_by_key)}/160")
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    source_u_hash = None
    source_cal_axis_path = Path(args.calibration_axes).resolve(strict=True)
    axes = np.load(source_cal_axis_path, allow_pickle=False)
    cal_u = np.asarray(axes["u_level"], dtype=np.float64)
    if ops.array_sha256(cal_u) != json.loads(Path(args.calibration_summary).read_text(encoding="utf-8"))["axes"]["u_sha256"]:
        raise ValueError("calibration u axis array does not match sealed summary")
    source_u_hash = ops.array_sha256(cal_u)
    del axes
    for input_row in natural_rows:
        sample_id = input_row["id"]
        source_plan_row = plan_by_sample.get(sample_id)
        baseline = result_by_key.get((sample_id, "baseline"))
        orth = result_by_key.get((sample_id, "orth_plus"))
        if source_plan_row is None or baseline is None or orth is None:
            raise ValueError(f"missing retry3 baseline or orth_plus artifact binding for {sample_id}")
        raw_input_path = _resolve_ref(Path(args.input_root).resolve(), source_plan_row["natural_input_path"])
        if _sha(raw_input_path) != input_row["ref_sha256"] or source_plan_row.get("natural_input_sha256") != input_row["ref_sha256"]:
            raise ValueError(f"RAW natural input path/hash differs from frozen manifest for {sample_id}")
        if baseline.get("status") != "scored" or orth.get("status") != "scored":
            raise ValueError(f"retry3 source artifacts are not scored for {sample_id}")
        ctx_npz = _resolve_ref(source_root, baseline["context_npz"])
        ctx_json = _resolve_ref(source_root, baseline.get("context_json", str(ctx_npz.with_suffix(".json"))))
        expected_ctx_sha = baseline["context_npz_sha256"]
        expected_meta_sha = baseline["context_json_sha256"]
        if _sha(ctx_npz) != expected_ctx_sha or _sha(ctx_json) != expected_meta_sha:
            raise ValueError(f"retry3 context hash mismatch for {sample_id}")
        if orth.get("context_npz_sha256") != expected_ctx_sha or orth.get("context_json_sha256") != expected_meta_sha:
            raise ValueError(f"baseline/orth_plus context refs differ for {sample_id}")
        meta = json.loads(ctx_json.read_text(encoding="utf-8"))
        with np.load(ctx_npz, allow_pickle=False) as archive:
            z0 = np.asarray(archive["natural_z_fixed_float32"], dtype=np.float32)
            z1 = np.asarray(archive["axis_z_orth_plus_float32"], dtype=np.float32)
            mask = np.asarray(archive["natural_speech_mask_fixed_uint8"], dtype=np.uint8)
            saved_u = np.asarray(archive["calibration_u_level_float64"], dtype=np.float64)
        for key, value in (
            ("natural_z_fixed_float32", z0),
            ("axis_z_orth_plus_float32", z1),
            ("natural_speech_mask_fixed_uint8", mask),
            ("calibration_u_level_float64", saved_u),
        ):
            if meta.get("array_sha256", {}).get(key) != ops.array_sha256(value):
                raise ValueError(f"retry3 context array hash mismatch for {sample_id}/{key}")
        if ops.array_sha256(saved_u) != source_u_hash:
            raise ValueError(f"saved calibration u axis differs across context for {sample_id}")
        radius0, radius1 = _per_frame_norm(z0), _per_frame_norm(z1)
        preflight_error = None
        if np.any(radius0 <= 0) or np.any(radius1 <= 0):
            preflight_error = "zero-radius frame in saved z0/z1; no arbitrary replacement direction was generated"
            factorized = {"q00": z0.copy(), "q11": z1.copy()}
            diagnostics = {
                "status": "failed_zero_radius",
                "zero_radius_z0_frames": np.flatnonzero(radius0 <= 0).astype(int).tolist(),
                "zero_radius_z1_frames": np.flatnonzero(radius1 <= 0).astype(int).tolist(),
            }
        else:
            factorized, diagnostics = _build_factorizations(z0, z1, saved_u, mask.astype(bool))
        context_path = Path(args.run_dir).resolve() / "samples" / f"{sample_id}_latent_context.npz"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        arrays_to_save = {
            "z0_saved_float32": factorized["q00"], "z1_saved_float32": factorized["q11"],
            "natural_speech_mask_fixed_uint8": mask,
            "calibration_u_level_float64": saved_u,
        }
        if preflight_error is None:
            arrays_to_save.update({"z_q10_float32": factorized["q10"], "z_q01_float32": factorized["q01"]})
        np.savez_compressed(context_path, **arrays_to_save)
        context_meta = {
            "sample_id": sample_id, "speaker_id": input_row["speaker"],
            "source_context_npz": {"path": str(ctx_npz), "sha256": expected_ctx_sha},
            "source_context_json": {"path": str(ctx_json), "sha256": expected_meta_sha},
            "source_context_arrays": {
                "z0_key": "natural_z_fixed_float32",
                "z0_sha256": ops.array_sha256(z0),
                "z1_key": "axis_z_orth_plus_float32",
                "z1_sha256": ops.array_sha256(z1),
                "u_sha256": source_u_hash,
                "mask_sha256": ops.array_sha256(mask),
            },
            "old_baseline_visual": {
                "path": baseline["visual_npz"], "sha256": baseline["visual_npz_sha256"],
                "video_path": source_plan_row["video_path"],
            },
            "old_orth_plus_visual": {
                "path": orth["visual_npz"], "sha256": orth["visual_npz_sha256"],
                "video_path": next(
                    row["video_path"] for row in source_plan
                    if row.get("sample_id") == sample_id and row.get("family") == "source_axis"
                    and row.get("source") == "natural" and row.get("face_id") == "primary"
                    and row.get("condition") == "orth_plus"
                ),
            },
            "factorization_diagnostics": diagnostics,
            "preflight_error": preflight_error,
            "factorized_context_npz": {"path": str(context_path.relative_to(Path(args.run_dir).resolve())), "sha256": _sha(context_path)},
        }
        _json(context_path.with_suffix(".json"), context_meta)
        artifacts.append({
            "sample_id": sample_id, "speaker_id": input_row["speaker"],
            "split": "evaluation", "input_sha256": input_row["ref_sha256"],
            "natural_input_path": str(raw_input_path),
            "source_context_npz": str(ctx_npz), "source_context_npz_sha256": expected_ctx_sha,
            "source_context_json": str(ctx_json), "source_context_json_sha256": expected_meta_sha,
            "z0_sha256": ops.array_sha256(z0), "z1_sha256": ops.array_sha256(z1),
            "factorized_context_npz": context_meta["factorized_context_npz"],
            "factorized_context_json_sha256": _sha(context_path.with_suffix(".json")),
            "preflight_error": preflight_error,
            "conditions": {
                name: {
                    "visual_id": VISUAL_IDS[name],
                    "video_path": f"videos/primary/{sample_id}__{VISUAL_IDS[name]}.mkv",
                    "visual_npz": f"visuals/primary/{sample_id}__{VISUAL_IDS[name]}.npz",
                    "score_npz": f"scores/{sample_id}__primary__{name}.npz",
                    "context_npz": context_meta["factorized_context_npz"]["path"],
                }
                for name in CONDITIONS
            },
        })
        for condition in CONDITIONS:
            rows.append({
                "ordered_eval_index": len(artifacts) - 1,
                "row_id": f"{sample_id}__primary__{condition}",
                "sample_id": sample_id, "speaker_id": input_row["speaker"],
                "split": "evaluation", "family": "latent_angle_radius",
                "condition": condition, "face_id": "primary", "visual_id": VISUAL_IDS[condition],
                "natural_input_path": str(raw_input_path), "natural_input_sha256": input_row["ref_sha256"],
                "source_context_npz": str(ctx_npz), "source_context_npz_sha256": expected_ctx_sha,
                "source_context_json": str(ctx_json), "source_context_json_sha256": expected_meta_sha,
                "z0_key": "natural_z_fixed_float32", "z0_sha256": ops.array_sha256(z0),
                "z1_key": "axis_z_orth_plus_float32", "z1_sha256": ops.array_sha256(z1),
                "context_npz": context_meta["factorized_context_npz"]["path"],
                "context_json": f"samples/{sample_id}_latent_context.json",
                "video_path": f"videos/primary/{sample_id}__{VISUAL_IDS[condition]}.mkv",
                "visual_npz": f"visuals/primary/{sample_id}__{VISUAL_IDS[condition]}.npz",
                "score_npz": f"scores/{sample_id}__primary__{condition}.npz",
                "status": "planned",
                "preflight_error": preflight_error,
            })
    if len(rows) != 320 or len(artifacts) != 80:
        raise ValueError(f"expected 80 prepared samples x 4 rows, got {len(artifacts)} samples / {len(rows)} rows")
    return rows, artifacts, {
        "source_run_dir": str(source_root), "source_protocol_sha256": protocol_sha,
        "source_row_manifest_sha256": plan_sha, "source_evaluation_results_sha256": _sha(source_results_path),
        "source_calibration_summary_sha256": _sha(Path(args.calibration_summary).resolve()),
        "source_calibration_axes_sha256": _sha(source_cal_axis_path), "source_u_sha256": source_u_hash,
        "source_artifacts": artifacts,
    }


def _row_unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(array * array, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (array / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def _create_protocol(args: argparse.Namespace, rows: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    spec_path = Path(args.spec).resolve(strict=True)
    geometry_path = Path(args.geometry_audit).resolve(strict=True)
    spec_sha, geometry_sha = _sha(spec_path), _sha(geometry_path)
    if spec_sha != EXPECTED_SPEC_SHA256 or geometry_sha != EXPECTED_GEOMETRY_SHA256:
        raise ValueError(f"frozen spec/geometry SHA mismatch: spec={spec_sha}, geometry={geometry_sha}")
    calibration = json.loads(Path(args.calibration_summary).read_text(encoding="utf-8"))
    anchor = calibration.get("global_anchor", {})
    observed = {
        "k": int(anchor.get("selected_k", 10_000)),
        "column": int(anchor.get("anchor_column", -1)),
        "official_offset": int(anchor.get("official_syncnet_offset", 10_000)),
    }
    if observed != EXPECTED_ANCHOR:
        raise ValueError(f"direction/radius runner requires anchor {EXPECTED_ANCHOR}, got {observed}")
    row_bytes = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    row_sha = hashlib.sha256(row_bytes).hexdigest()
    run_dir = Path(args.run_dir).resolve()
    row_path = run_dir / "evaluation_rows.jsonl"
    row_path.parent.mkdir(parents=True, exist_ok=True)
    unexpected = [path.name for path in run_dir.iterdir() if path.name != "samples"]
    if row_path.exists() or unexpected:
        raise FileExistsError(f"refusing to overwrite nonempty direction/radius run: {run_dir}")
    row_path.write_bytes(row_bytes)
    protocol = {
        "schema_version": 1,
        "classification": "sealed direction/radius generator factorization; no cohort effect summary",
        "experiment_id": "confirm-latent-angle-radius-20261001",
        "declared_commit": "fcc4438e9eafa17c78dc49e205c418461bd0aa54",
        "spec": {"path": str(spec_path), "sha256": spec_sha},
        "calibration_geometry_audit": {"path": str(geometry_path), "sha256": geometry_sha},
        "source_retry3": {
            "run_dir": source["source_run_dir"],
            "protocol_sha256": source["source_protocol_sha256"],
            "row_manifest_sha256": source["source_row_manifest_sha256"],
            "evaluation_results_sha256": source["source_evaluation_results_sha256"],
            "calibration_summary_sha256": source["source_calibration_summary_sha256"],
            "calibration_axes_sha256": source["source_calibration_axes_sha256"],
            "u_level_array_sha256": source["source_u_sha256"],
            "context_count": 80,
        },
        "calibration_anchor": {"k": 3, "column": 18, "official_syncnet_offset": -3},
        "input_manifest": {"path": str(Path(args.input_manifest).resolve()), "sha256": _sha(Path(args.input_manifest).resolve()), "required_sha256": EXPECTED_INPUT_MANIFEST_SHA256},
        "input_root": str(Path(args.input_root).resolve()),
        "face": {"id": "primary", "path": str(Path(args.primary_face).resolve()), "sha256": _sha(Path(args.primary_face).resolve())},
        "wav2lip_checkpoint_sha256": _sha(Path(args.wav2lip_checkpoint).resolve()),
        "syncnet_checkpoint_sha256": _sha(Path(args.syncnet_checkpoint).resolve()),
        "runtime_policy": {
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "deterministic_algorithms": True, "reassert_after_detection": True,
            "all_conditions_freshly_rendered": True,
        },
        "frontend": "official JPEG/OpenCV SyncNet visual frontend; RAW PCM16 float64 integer-amplitude MFCC units",
        "conditions": {
            "q00": "exact saved z0 direction and radius; freshly rendered",
            "q10": "float32(z1 * r0/r1), source direction with own z0 radius",
            "q01": "float32(z0 * r1/r0), own direction with source-plus z1 radius",
            "q11": "exact saved z1 direction and radius; freshly rendered",
        },
        "score_audio": "unchanged original RAW natural PCM for all four cells; one new A cache per sample",
        "planned_samples": 80, "planned_speakers": 40, "planned_rows": len(rows),
        "row_manifest_path": str(row_path), "row_manifest_sha256": row_sha,
        "code_sha256": _code_hashes(),
        "source_artifacts": source["source_artifacts"],
        "started_at_utc": _utc_now(),
    }
    _json(run_dir / "evaluation_protocol.json", protocol)
    _json(run_dir / "code_seal.json", {"code_sha256": protocol["code_sha256"], "protocol_sha256": _sha(run_dir / "evaluation_protocol.json")})
    _json(run_dir / "run_status.json", {"status": "sealed_preflight_complete", "planned_samples": 80, "planned_rows": len(rows), "completed_samples": 0})
    return protocol


def _source_visual_compare(source_run: Path, original_row: dict[str, Any], new_lip: np.ndarray, new_video_meta: dict[str, Any]) -> dict[str, Any]:
    old_path = _resolve_ref(source_run, original_row["visual_npz"])
    if _sha(old_path) != original_row["visual_npz_sha256"]:
        raise ValueError(f"source visual NPZ hash mismatch: {old_path}")
    with np.load(old_path, allow_pickle=False) as archive:
        old_lip = np.asarray(archive["lip_embeddings_float32"], dtype=np.float32)
    same_shape = old_lip.shape == new_lip.shape
    difference = np.abs(old_lip.astype(np.float64) - new_lip.astype(np.float64)) if same_shape else np.asarray([np.inf])
    old_meta_path = old_path.with_suffix(".json")
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    old_pixel_sha = old_meta.get("video", {}).get("raw_bgr_pixel_sha256")
    new_pixel_sha = new_video_meta.get("video", {}).get("raw_bgr_pixel_sha256")
    return {
        "old_visual_npz_path": str(old_path), "old_visual_npz_sha256": _sha(old_path),
        "new_shape": list(new_lip.shape), "old_shape": list(old_lip.shape),
        "shape_equal": same_shape,
        "embedding_max_abs_difference": float(difference.max(initial=0.0)),
        "embedding_exact": bool(same_shape and np.array_equal(old_lip, new_lip)),
        "old_video_pixel_sha256": old_pixel_sha, "new_video_pixel_sha256": new_pixel_sha,
        "video_pixels_exact": bool(old_pixel_sha and old_pixel_sha == new_pixel_sha),
    }


def _audio_cache(syncnet: Any, natural_path: Path, expected_sha: str, device: str, batch_size: int):
    pcm, units, audio_meta = _read_pcm16(natural_path, expected_sha)
    count = int(len(units) // 640 - 5)
    if count < 1:
        raise ValueError(f"RAW natural audio has no SyncNet window support: {natural_path}")
    audio, sync_meta = scoring.audio_embeddings_from_syncnet_units(
        syncnet, units, window_count=count, device=device, batch_size=batch_size,
    )
    mfcc = evaluate._mfcc_from_units(units)
    if ops.array_sha256(mfcc) != sync_meta["mfcc_sha256"]:
        raise AssertionError("stored MFCC differs from official SyncNet audio frontend")
    return pcm, units, audio, mfcc, audio_meta, sync_meta


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.preflight_only:
        if _sha(Path(args.spec).resolve(strict=True)) != EXPECTED_SPEC_SHA256:
            raise ValueError("frozen direction/radius spec SHA mismatch")
        if _sha(Path(args.geometry_audit).resolve(strict=True)) != EXPECTED_GEOMETRY_SHA256:
            raise ValueError("frozen calibration geometry SHA mismatch")
        rows, artifacts, source = _source_inventory(args)
        return {
            "status": "preflight_pass", "planned_samples": len(artifacts), "planned_rows": len(rows),
            "source_run_dir": source["source_run_dir"],
            "source_protocol_sha256": source["source_protocol_sha256"],
            "source_row_manifest_sha256": source["source_row_manifest_sha256"],
            "source_context_count": len(source["source_artifacts"]),
        }
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty direction/radius run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if _sha(Path(args.spec).resolve(strict=True)) != EXPECTED_SPEC_SHA256:
        raise ValueError("frozen direction/radius spec SHA mismatch")
    if _sha(Path(args.geometry_audit).resolve(strict=True)) != EXPECTED_GEOMETRY_SHA256:
        raise ValueError("frozen calibration geometry SHA mismatch")
    rows, source_artifacts, source = _source_inventory(args)
    protocol = _create_protocol(args, rows, source)
    _json(run_dir / "faces_primary.json", {"face_id": "primary", "status": "pending_model_load"})
    _json(run_dir / "attempts.json", {"attempts": [{"attempt": 1, "status": "running", "started_at_utc": protocol["started_at_utc"], "code_sha256": protocol["code_sha256"]}]})
    source_root = Path(args.source_run_dir).resolve()
    source_plan = _read_jsonl(source_root / "evaluation_rows.jsonl")
    source_results = _read_jsonl(source_root / "evaluation_results.jsonl")
    source_plan_by_key = {(row["sample_id"], row["condition"]): row for row in source_plan if row.get("family") == "source_axis" and row.get("source") == "natural" and row.get("face_id") == "primary"}
    source_result_by_key = {(row["sample_id"], row["condition"]): row for row in source_results if row.get("family") == "source_axis" and row.get("source") == "natural" and row.get("face_id") == "primary"}

    torch.manual_seed(20261001)
    np.random.seed(20261001)
    _set_flags()
    device = "cuda"
    frontend = wav2lip.install_librosa_compatibility_shim()
    model, model_meta = wav2lip.load_model(Path(args.wav2lip_root).resolve(), args.wav2lip_checkpoint, device=device)
    syncnet, syncnet_meta = scoring.load_syncnet(Path(args.syncnet_root).resolve(), args.syncnet_checkpoint, device=device)
    face_path = Path(args.primary_face).resolve()
    frame = cv2.imread(str(face_path))
    if frame is None:
        raise RuntimeError(f"cannot read primary face image: {face_path}")
    flags_before_detection = _flags()
    face_meta = wav2lip.detect_static_face(frame, Path(args.wav2lip_root).resolve(), device=device)
    flags_after_detection = _flags()
    _set_flags()
    flags_after_override = _flags()
    face_meta.update({"path": str(face_path), "sha256": _sha(face_path), "width": int(frame.shape[1]), "height": int(frame.shape[0])})
    _json(run_dir / "faces_primary.json", {
        "face_id": "primary", "face": face_meta,
        "flags_before_detection": flags_before_detection,
        "flags_after_detection": flags_after_detection,
        "flags_after_explicit_override": flags_after_override,
        "wav2lip_checkpoint_sha256": _sha(Path(args.wav2lip_checkpoint).resolve()),
        "code_sha256": protocol["code_sha256"], "frontend_compatibility": frontend,
    })
    events: list[dict[str, Any]] = []
    active: dict[str, str | None] = {"sample_id": None, "condition": None}
    originals = _instrument(events, active)
    failed_samples: list[dict[str, Any]] = []
    completed_samples = 0
    result_path = run_dir / "evaluation_results.jsonl"
    attempts_path = run_dir / "attempts.jsonl"
    context_root = run_dir / "samples"
    try:
        _set_flags()
        face_cache = wav2lip.cache_face_features(model, frame, face_meta["generation_box_xyxy"], range(1, args.wav2lip_batch_size + 1), device=device)
        for artifact in source_artifacts:
            sample_id = artifact["sample_id"]
            sample_started = time.perf_counter()
            context_path = run_dir / artifact["conditions"]["q00"]["context_npz"]
            context_meta_path = context_path.with_suffix(".json")
            try:
                if artifact.get("preflight_error"):
                    raise ValueError(artifact["preflight_error"])
                with np.load(context_path, allow_pickle=False) as archive:
                    z_by_condition = {
                        "q00": np.asarray(archive["z0_saved_float32"], dtype=np.float32),
                        "q10": np.asarray(archive["z_q10_float32"], dtype=np.float32),
                        "q01": np.asarray(archive["z_q01_float32"], dtype=np.float32),
                        "q11": np.asarray(archive["z1_saved_float32"], dtype=np.float32),
                    }
                    mask = np.asarray(archive["natural_speech_mask_fixed_uint8"], dtype=np.uint8).astype(bool)
                if not context_meta_path.is_file():
                    raise FileNotFoundError(f"preflight latent context metadata missing: {context_meta_path}")
                context_meta = json.loads(context_meta_path.read_text(encoding="utf-8"))
                context_sha = _sha(context_path)
                if context_meta.get("factorized_context_npz", {}).get("sha256") != context_sha:
                    raise ValueError(f"factorized context NPZ hash mismatch for {sample_id}")
                active["sample_id"] = sample_id
                active["condition"] = "shared_A_RAW_N"
                pcm, units, audio, mfcc, audio_input_meta, audio_forward_meta = _audio_cache(
                    syncnet, Path(artifact["natural_input_path"]), artifact["input_sha256"], device, args.syncnet_batch_size,
                )
                active["condition"] = None
                audio_domain = len(audio)
                query_n = min(audio_domain, *(len(z) for z in z_by_condition.values()))
                if query_n < 1:
                    raise ValueError(f"no shared V/A support for {sample_id}")
                support = evaluate._support_indices(query_n)
                batch_sizes = {args.wav2lip_batch_size}
                for z in z_by_condition.values():
                    tail = len(z) % args.wav2lip_batch_size
                    if tail:
                        batch_sizes.add(tail)
                visual_cache: dict[str, np.ndarray] = {}
                visual_meta_cache: dict[str, dict[str, Any]] = {}
                bridge_comparisons: dict[str, Any] = {}
                for condition in CONDITIONS:
                    _set_flags()
                    active["sample_id"] = sample_id
                    active["condition"] = condition
                    lip, visual_meta = evaluate._load_or_make_visual(
                        run_dir, sample_id, "primary", VISUAL_IDS[condition], z_by_condition[condition],
                        frame, face_meta, model,
                        face_cache,
                        syncnet,
                        argparse.Namespace(
                            ffmpeg=args.ffmpeg, wav2lip_batch_size=args.wav2lip_batch_size,
                            syncnet_batch_size=args.syncnet_batch_size,
                        ),
                        device,
                    )
                    if len(lip) < query_n:
                        raise ValueError(f"visual frames shorter than frozen common support for {sample_id}/{condition}")
                    visual_cache[condition] = np.asarray(lip, dtype=np.float32)
                    visual_meta_cache[condition] = visual_meta
                    if condition in {"q00", "q11"}:
                        source_condition = "baseline" if condition == "q00" else "orth_plus"
                        bridge_comparisons[condition] = _source_visual_compare(
                            source_root, source_result_by_key[(sample_id, source_condition)], visual_cache[condition], visual_meta,
                        )
                active["sample_id"] = None
                active["condition"] = None

                context_output = run_dir / "samples" / f"{sample_id}_context.npz"
                source_arrays = {
                    "z0_saved_float32": z_by_condition["q00"],
                    "z1_saved_float32": z_by_condition["q11"],
                    "z_q10_float32": z_by_condition["q10"],
                    "z_q01_float32": z_by_condition["q01"],
                    "natural_pcm16": pcm,
                    "natural_syncnet_units_raw_float64": units,
                    "natural_mfcc_raw_float64": mfcc,
                    "natural_audio_embeddings_raw_float32": audio,
                    "natural_speech_mask_fixed_uint8": mask.astype(np.uint8),
                    "common_support_full_int32": support["full"],
                    "common_support_guard20_int32": support["guard20"],
                    "common_support_guard15_int32": support["guard15"],
                }
                context_output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(context_output, **source_arrays)
                output_context_sha = _sha(context_output)
                output_context_meta = {
                    "sample_id": sample_id, "speaker_id": artifact["speaker_id"],
                    "input_audio": audio_input_meta,
                    "syncnet_audio_forward": audio_forward_meta,
                    "common_query_windows": query_n, "full_audio_rows": audio_domain,
                    "support_indices": {key: value.tolist() for key, value in support.items()},
                    "source_context": context_meta,
                    "engineering_bridge_q00_q11": bridge_comparisons,
                    "visuals": {
                        name: {
                            "metadata": visual_meta_cache[name],
                            "lip_embedding_sha256": ops.array_sha256(visual_cache[name]),
                            "source_visual_npz": artifact["conditions"][name]["visual_npz"],
                        }
                        for name in CONDITIONS
                    },
                    "npz_sha256": output_context_sha,
                }
                _json(context_output.with_suffix(".json"), output_context_meta)

                per_condition_results: dict[str, dict[str, Any]] = {}
                for condition in CONDITIONS:
                    active["sample_id"] = sample_id
                    active["condition"] = condition
                    visual = visual_cache[condition][:query_n]
                    visual_unit = evaluate._row_unit(visual)
                    audio_unit = evaluate._row_unit(audio)
                    _set_flags()
                    cross_raw, zero_raw = evaluate._cross_distance_matrix(visual, audio, device)
                    raw_matrix = evaluate._lag_matrix_from_cross(cross_raw, zero_raw, query_n, audio_domain, vshift=15)
                    _set_flags()
                    cross_unit, zero_unit = evaluate._cross_distance_matrix(visual_unit, audio_unit, device)
                    unit_matrix = evaluate._lag_matrix_from_cross(cross_unit, zero_unit, query_n, audio_domain, vshift=15)
                    score_path = run_dir / artifact["conditions"][condition]["score_npz"]
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        score_path,
                        cross_distance_raw_float32=cross_raw,
                        cross_distance_to_zero_raw_float32=zero_raw,
                        distance_matrix_raw_float32=raw_matrix,
                        cross_distance_unit_float32=cross_unit,
                        cross_distance_to_zero_unit_float32=zero_unit,
                        distance_matrix_unit_float32=unit_matrix,
                        visual_embeddings_raw_float32=visual,
                        audio_embeddings_raw_float32=audio,
                        visual_embeddings_unit_float32=visual_unit,
                        audio_embeddings_unit_float32=audio_unit,
                        common_support_full_int32=support["full"],
                        common_support_valid_int32=support["valid"],
                        common_support_guard20_int32=support["guard20"],
                        common_support_guard15_int32=support["guard15"],
                        lag_columns_int32=np.arange(31, dtype=np.int32),
                        lag_k_int32=np.arange(-15, 16, dtype=np.int32),
                    )
                    result = {
                        **next(row for row in rows if row["row_id"] == f"{sample_id}__primary__{condition}"),
                        "status": "scored", "recorded_at_utc": _utc_now(),
                        "context_npz": str(context_output.relative_to(run_dir)),
                        "context_npz_sha256": output_context_sha,
                        "context_json": str(context_output.with_suffix(".json").relative_to(run_dir)),
                        "context_json_sha256": _sha(context_output.with_suffix(".json")),
                        "visual_npz": artifact["conditions"][condition]["visual_npz"],
                        "visual_npz_sha256": visual_meta_cache[condition]["feature_npz_sha256"],
                        "video_path": artifact["conditions"][condition]["video_path"],
                        "video_sha256": visual_meta_cache[condition]["video"]["sha256"],
                        "score_npz": str(score_path.relative_to(run_dir)),
                        "score_npz_sha256": _sha(score_path),
                        "common_support_n": query_n, "audio_row_domain_n": audio_domain,
                        "anchor": {"k": 3, "column": 18, "official_syncnet_offset": -3},
                        "raw_support_metrics": evaluate._metric_supports(raw_matrix, 18),
                        "unit_support_metrics": evaluate._metric_supports(unit_matrix, 18),
                        "factorization_diagnostics": context_meta["factorization_diagnostics"],
                        "new_audio_embedding_sha256": ops.array_sha256(audio),
                        "new_visual_embedding_sha256": ops.array_sha256(visual),
                        "flags_at_forward": sorted({json.dumps(event["flags_at_call"], sort_keys=True) for event in events if event.get("sample_id") == sample_id and event.get("condition") == condition}),
                    }
                    per_condition_results[condition] = result
                for condition in CONDITIONS:
                    _append_jsonl(result_path, per_condition_results[condition])
                _append_jsonl(attempts_path, {
                    "sample_id": sample_id, "speaker_id": artifact["speaker_id"], "status": "complete",
                    "score_rows_written": 4, "elapsed_seconds": time.perf_counter() - sample_started,
                    "context_npz_sha256": output_context_sha, "recorded_at_utc": _utc_now(),
                })
                completed_samples += 1
                _json(run_dir / "run_status.json", {
                    "status": "running", "planned_samples": 80, "planned_rows": 320,
                    "completed_samples": completed_samples, "failed_samples": len(failed_samples),
                    "last_sample_id": sample_id,
                })
                print(json.dumps({"event": "clip_complete", "sample_id": sample_id, "completed_samples": completed_samples, "elapsed_seconds": time.perf_counter() - sample_started}), flush=True)
            except Exception as exc:
                error = {
                    "sample_id": sample_id, "error_type": type(exc).__name__, "error": str(exc)[:2000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                    "elapsed_seconds": time.perf_counter() - sample_started,
                }
                failed_samples.append(error)
                for condition in CONDITIONS:
                    row = next(row for row in rows if row["row_id"] == f"{sample_id}__primary__{condition}")
                    _append_jsonl(result_path, {**row, "status": "failed", **error, "recorded_at_utc": _utc_now()})
                _append_jsonl(attempts_path, {"status": "failed", **error, "recorded_at_utc": _utc_now()})
                _json(run_dir / "run_status.json", {
                    "status": "running", "planned_samples": 80, "planned_rows": 320,
                    "completed_samples": completed_samples, "failed_samples": len(failed_samples),
                    "last_sample_id": sample_id,
                })
                print(json.dumps({"event": "clip_failed", **error}, ensure_ascii=False), flush=True)
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
    actual_flag_set = sorted({json.dumps(event["flags_at_call"], sort_keys=True) for event in events})
    final = {
        "status": "complete_with_failures" if failed_samples else "complete",
        "classification": protocol["classification"],
        "planned_samples": 80, "completed_samples": completed_samples,
        "failed_samples": len(failed_samples), "planned_rows": 320,
        "completed_rows": sum(1 for row in _read_jsonl(result_path) if row.get("status") == "scored"),
        "failed_rows": sum(1 for row in _read_jsonl(result_path) if row.get("status") == "failed"),
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"),
        "row_manifest_sha256": _sha(run_dir / "evaluation_rows.jsonl"),
        "code_sha256": protocol["code_sha256"],
        "runtime": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name(),
            "numpy": str(np.__version__), "opencv": str(cv2.__version__),
            "frontend_compatibility": frontend, "wav2lip_model": model_meta, "syncnet_model": syncnet_meta,
        },
        "face_flags": {"before_detection": flags_before_detection, "after_detection": flags_after_detection, "after_override": flags_after_override},
        "forward_flag_values": actual_flag_set, "forward_events": events,
        "finished_at_utc": _utc_now(),
    }
    _json(run_dir / "run_status.json", final)
    _json(run_dir / "all_files_sha256.json", {
        str(path.relative_to(run_dir)): _sha(path)
        for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"
    })
    _json(run_dir / "attempts.json", {"attempts": [{"attempt": 1, "status": final["status"], "finished_at_utc": final["finished_at_utc"], "completed_samples": completed_samples, "failed_samples": len(failed_samples)}]})
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-axes", required=True)
    parser.add_argument("--geometry-audit", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true", help="validate all sealed inputs and source contexts without creating a run or loading GPU models")
    parser.add_argument("--wav2lip-batch-size", type=int, default=64)
    parser.add_argument("--syncnet-batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    print(json.dumps(_run(build_parser().parse_args()), ensure_ascii=False, separators=(",", ":")))
