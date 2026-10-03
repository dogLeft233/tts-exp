"""Fit-only v3 objective for the pinned full-natural-context Wav2Lip proxy.

This runtime reuses the immutable v2 model/context/audio-cache implementation,
but defines a new training objective against an explicitly rendered bounded
RMS reference. The historical v2 module and its artifacts are not modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import wav2lip_fullcontext_fit_runtime_v2_attempt02 as v2
from .audio_io import read_canonical_wav
from .baselines import apply_global_rms, waveform_rms
from .training import TrainStepInfo


RUNTIME_SCHEMA = "cross_tfg_wav2lip_fullcontext_fit_runtime_v3_attempt03"
BASELINE_CACHE_SCHEMA = "cross_tfg_bounded_rms_visual_feature_cache_v1"
LAG_RADIUS = v2.LAG_RADIUS
ZERO_LAG_COLUMN = LAG_RADIUS
VIDEO_QUERY_COUNT = v2.VIDEO_QUERY_COUNT


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_key(value: str) -> str:
    import re

    if re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError(f"unsafe v3 baseline cache key: {value!r}")
    return value


def _finite_scalar(value, name: str) -> float:
    scalar = float(value.detach().to(dtype=value.dtype).item()) if hasattr(value, "detach") else float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(f"non-finite v3 fit metric: {name}")
    return scalar


def _flat_waveform(torch, waveform):
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        return waveform[:, 0, :]
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim == 2:
        return waveform
    raise ValueError("v3 fit waveform must be mono [N], [B,N], or [B,1,N]")


def _mask_tv(torch, log_mask):
    if log_mask.numel() == 0 or log_mask.shape[-1] <= 1 or log_mask.shape[-2] <= 1:
        return log_mask.new_zeros(())
    tv_frequency = torch.mean(torch.abs(log_mask[..., 1:, :] - log_mask[..., :-1, :]))
    tv_time = torch.mean(torch.abs(log_mask[..., :, 1:] - log_mask[..., :, :-1]))
    return tv_frequency + tv_time


def _embedding_metrics(torch, functional, visual, audio_center):
    if visual.shape != audio_center.shape or visual.ndim != 2 or visual.shape[1] != 1024:
        raise ValueError("v3 visual/audio embedding query geometry must be equal [N,1024]")
    if not torch.isfinite(visual).all().item() or not torch.isfinite(audio_center).all().item():
        raise FloatingPointError("v3 SyncNet embeddings contain non-finite values")
    visual_norm = torch.linalg.vector_norm(visual, dim=-1)
    audio_norm = torch.linalg.vector_norm(audio_center, dim=-1)
    if (visual_norm <= 1e-8).any().item() or (audio_norm <= 1e-8).any().item():
        raise FloatingPointError("v3 embedding norm is too small for normalized cosine")
    cosine = functional.cosine_similarity(visual, audio_center, dim=-1, eps=1e-8)
    return visual_norm, audio_norm, cosine


def _curve_metrics(torch, curve):
    if tuple(curve.shape) != (2 * LAG_RADIUS + 1,) or not torch.isfinite(curve).all().item():
        raise FloatingPointError("v3 SyncNet curve must contain 31 finite lag distances")
    b_value = torch.median(curve)
    d_min = torch.min(curve)
    d_zero = curve[ZERO_LAG_COLUMN]
    return {
        "B_median_distance": b_value,
        "Dmin_best_lag_distance": d_min,
        "D0_zero_lag_distance": d_zero,
        "Cbest_B_minus_Dmin": b_value - d_min,
        "Canchor_zero_B_minus_D0": b_value - d_zero,
    }


def _baseline_binding(*, plan, row, context, source_pins, rms_config, runtime_sha):
    natural = row["natural_pcm"]
    context_geometry = {
        "query_start_frame": int(context.query_start_frame),
        "query_count": int(context.query_count),
        "frame_count": int(context.frame_count),
        "source_video_sha256": str(context.source_video_sha256),
        "face_crops_bgr_sha256": _array_sha(context.face_crops_bgr),
        "syncnet_crops_bgr_sha256": _array_sha(context.syncnet_crops_bgr),
        "patch_boxes_xyxy_sha256": _array_sha(context.patch_boxes_xyxy),
        "mel_starts_i64_sha256": _array_sha(np.asarray(context.mel_starts, dtype=np.int64)),
        "source_frame_indices_i64_sha256": _array_sha(np.asarray(context.source_video_frame_indices, dtype=np.int64)),
    }
    return {
        "schema": BASELINE_CACHE_SCHEMA,
        "row_id": str(row["row_id"]),
        "sample_id": str(row["sample_id"]),
        "speaker_id": str(row["speaker_id"]),
        "split": str(row["split"]),
        "source_pcm_file_sha256": str(natural["file_sha256"]).lower(),
        "source_pcm_float32_sha256": str(natural["pcm_sha256"]).lower(),
        "source_sample_count": int(natural["sample_count"]),
        "source_pcm16_cache_sha256": str(context.cache_record["cache_binding"]["source_pcm16_sha256"]),
        "natural_audio_embedding_cache_sha256": str(context.cache_record["npz_sha256"]),
        "context_geometry": context_geometry,
        "model_assets": {
            "wav2lip_checkpoint": source_pins.get("wav2lip_checkpoint"),
            "wav2lip_sources": source_pins.get("wav2lip_sources"),
            "syncnet_checkpoint": source_pins.get("syncnet_checkpoint"),
            "syncnet_sources": source_pins.get("syncnet_sources"),
            "python_speech_features": source_pins.get("python_speech_features"),
            "official_frontend": source_pins.get("official_frontend"),
        },
        "frontend_parity_sha256": _canonical_sha(source_pins["full_context_frontend_parity"]),
        "runtime_v2_source_sha256": str(source_pins["v2_runtime"]["sha256"]),
        "runtime_v3_source_sha256": runtime_sha,
        "rms_reference": rms_config,
        "embedding_definition": {
            "visual": "FullContextSyncAdapter.forward_lip outputs before L2 normalization; query windows 0..26 of fixed center-32 render",
            "audio": "unchanged cached original-natural PCM16 SyncNet forward_aud outputs at the same 27 source-clock query indices",
            "distance": "v2 FullContextSyncAdapter.sync_distance_curve; mean float32 pairwise_distance over 27 queries for each lag -15..+15",
            "zero_lag_column": ZERO_LAG_COLUMN,
        },
    }


def _cache_baseline(
    *, torch, functional, cache_root: Path, binding: dict[str, Any], row: dict[str, Any], context,
    natural_audio, enhancement_stack, adapter, rms_config, cache_records: list[dict[str, Any]],
):
    row_key = _safe_key(str(row["row_id"]))
    fingerprint = _canonical_sha(binding)
    npz_path = cache_root / f"{row_key}.{fingerprint[:16]}.npz"
    meta_path = cache_root / f"{row_key}.{fingerprint[:16]}.json"
    if npz_path.exists() != meta_path.exists():
        raise RuntimeError(f"partial v3 bounded-RMS baseline cache entry: {row_key}")

    input_audio = torch.as_tensor(natural_audio, device=next(enhancement_stack.parameters()).device, dtype=torch.float32)
    enhancement_stack.eval()
    with torch.no_grad():
        reference, reference_mask, reference_diag = enhancement_stack.forward_with_mask(
            input_audio,
            use_global_rms=True,
            use_local_envelope=False,
            use_spectral_head=False,
            post_rms_match="target",
            target_rms=float(rms_config["target_rms"]),
            peak_ceiling=float(rms_config["peak_ceiling"]),
            rms_stage_peak_ceiling=float(rms_config["rms_stage_peak_ceiling"]),
            return_diagnostics=True,
        )
    reference_flat = _flat_waveform(torch, reference)
    reference_np = np.ascontiguousarray(reference_flat[0].detach().cpu().numpy(), dtype=np.float32)
    if reference_np.size != int(row["natural_pcm"]["sample_count"]) or not np.isfinite(reference_np).all():
        raise RuntimeError(f"v3 bounded-RMS reference changed sample count or became non-finite: {row_key}")

    if npz_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata_payload = {key: value for key, value in metadata.items() if key != "metadata_payload_sha256"}
        if (
            metadata.get("schema") != BASELINE_CACHE_SCHEMA
            or metadata.get("binding") != binding
            or metadata.get("cache_fingerprint") != fingerprint
            or metadata.get("metadata_payload_sha256") != _canonical_sha(metadata_payload)
            or _sha256_file(npz_path) != metadata.get("npz_file_sha256")
        ):
            raise ValueError(f"v3 baseline cache binding/file integrity mismatch: {row_key}")
        with np.load(npz_path, allow_pickle=False) as archive:
            expected_names = {
                "rms_reference_audio_float32",
                "rms_visual_embedding_float32",
                "rms_distance_curve_float32",
                "query_frame_index_int32",
            }
            if set(archive.files) != expected_names:
                raise ValueError(f"v3 baseline cache array inventory mismatch: {row_key}")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if not np.array_equal(arrays["rms_reference_audio_float32"], reference_np):
            raise ValueError(f"recomputed bounded-RMS reference differs from sealed cache: {row_key}")
        for name, array in arrays.items():
            record = metadata["arrays"].get(name)
            if not isinstance(record, dict) or record.get("shape") != list(array.shape) or record.get("dtype") != str(array.dtype):
                raise ValueError(f"v3 baseline array metadata mismatch: {row_key}:{name}")
            if _array_sha(array) != record.get("array_sha256"):
                raise ValueError(f"v3 baseline array SHA mismatch: {row_key}:{name}")
    else:
        with torch.no_grad():
            rendered = adapter.render_video(context, reference)
            curve = adapter.sync_distance_curve(rendered, context=context)
            visual = adapter.last_visual_embeddings
            audio_center = adapter.last_audio_center_embeddings
            if visual is None or audio_center is None:
                raise RuntimeError(f"v3 baseline render omitted SyncNet embeddings: {row_key}")
            visual_np = np.ascontiguousarray(visual.detach().to(dtype=torch.float32).cpu().numpy())
            curve_np = np.ascontiguousarray(curve.detach().to(dtype=torch.float32).cpu().numpy())
            query_index = np.arange(context.query_start_frame, context.query_start_frame + VIDEO_QUERY_COUNT, dtype=np.int32)
            base_metrics = _curve_metrics(torch, curve)
            base_norm, _, base_cos = _embedding_metrics(torch, functional, visual, audio_center)
            metric_values = {
                key: _finite_float(value) for key, value in base_metrics.items()
            }
            metric_values.update({
                "mean_visual_embedding_l2": _finite_float(base_norm.mean()),
                "mean_zero_lag_unit_cosine": _finite_float(base_cos.mean()),
            })
        arrays = {
            "rms_reference_audio_float32": reference_np,
            "rms_visual_embedding_float32": visual_np,
            "rms_distance_curve_float32": curve_np,
            "query_frame_index_int32": query_index,
        }
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary_npz = npz_path.with_name(npz_path.name + f".tmp.{os.getpid()}.npz")
        with temporary_npz.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_npz, npz_path)
        array_records = {
            name: {"dtype": str(array.dtype), "shape": [int(value) for value in array.shape], "array_sha256": _array_sha(array)}
            for name, array in arrays.items()
        }
        reference_diag_values = {
            name: _finite_float(value) if hasattr(value, "numel") else value
            for name, value in reference_diag.items()
        }
        natural_np = np.asarray(natural_audio, dtype=np.float32)
        legacy_rms, legacy_info = apply_global_rms(
            natural_np,
            float(rms_config["target_rms"]),
            peak_ceiling=None,
        )
        legacy_difference = reference_np.astype(np.float64) - legacy_rms.astype(np.float64)
        source_peak = float(np.max(np.abs(natural_np.astype(np.float64))))
        source_rms = waveform_rms(natural_np)
        requested_global_gain = float(rms_config["target_rms"]) / max(source_rms, 1e-12)
        stage_peak_active = bool(
            rms_config["rms_stage_peak_ceiling"] is not None
            and source_peak * requested_global_gain > float(rms_config["rms_stage_peak_ceiling"])
        )
        post_peak_gain = float(reference_diag["final_peak_gain"].detach().reshape(-1)[0].item())
        metadata = {
            "schema": BASELINE_CACHE_SCHEMA,
            "binding": binding,
            "cache_fingerprint": fingerprint,
            "npz_path": str(npz_path),
            "arrays": array_records,
            "metrics": metric_values,
            "rms_reference_diagnostics": reference_diag_values,
            "legacy_rms_control_parity": {
                "legacy_method": "apply_global_rms(target_rms, peak_ceiling=None)",
                "legacy_output_pcm_float32_sha256": _array_sha(legacy_rms),
                "bounded_reference_output_pcm_float32_sha256": _array_sha(reference_np),
                "max_abs_delta": float(np.max(np.abs(legacy_difference))),
                "rmse": float(np.sqrt(np.mean(np.square(legacy_difference)))),
                "within_1e_7_max_abs": bool(np.max(np.abs(legacy_difference)) <= 1e-7),
                "global_rms_stage_peak_active": stage_peak_active,
                "final_peak_ceiling_active": bool(post_peak_gain < 1.0 - 1e-12),
                "legacy_requested_gain": float(legacy_info.requested_gain),
                "legacy_applied_gain": float(legacy_info.applied_gain),
                "bounded_reference_post_peak_gain": post_peak_gain,
            },
            "npz_bytes": int(npz_path.stat().st_size),
            "npz_file_sha256": _sha256_file(npz_path),
        }
        metadata["metadata_payload_sha256"] = _canonical_sha(metadata)
        _atomic_json(meta_path, metadata)

    reference_array = arrays["rms_reference_audio_float32"]
    visual_array = arrays["rms_visual_embedding_float32"]
    curve_array = arrays["rms_distance_curve_float32"]
    query_array = arrays["query_frame_index_int32"]
    if reference_array.shape != (int(row["natural_pcm"]["sample_count"]),) or reference_array.dtype != np.float32:
        raise ValueError(f"v3 baseline RMS waveform array shape/dtype mismatch: {row_key}")
    if visual_array.shape != (VIDEO_QUERY_COUNT, 1024) or visual_array.dtype != np.float32:
        raise ValueError(f"v3 baseline visual embedding array shape/dtype mismatch: {row_key}")
    if curve_array.shape != (2 * LAG_RADIUS + 1,) or curve_array.dtype != np.float32:
        raise ValueError(f"v3 baseline curve shape/dtype mismatch: {row_key}")
    expected_queries = np.arange(context.query_start_frame, context.query_start_frame + VIDEO_QUERY_COUNT, dtype=np.int32)
    if query_array.dtype != np.int32 or not np.array_equal(query_array, expected_queries):
        raise ValueError(f"v3 baseline query frame map mismatch: {row_key}")
    if not all(np.isfinite(value).all() for value in (reference_array, visual_array, curve_array)):
        raise FloatingPointError(f"v3 baseline cache contains non-finite values: {row_key}")
    base_tensor = torch.from_numpy(reference_array).to(device=input_audio.device, dtype=torch.float32)
    base_visual_tensor = torch.from_numpy(visual_array).to(device=input_audio.device, dtype=torch.float32)
    base_curve_tensor = torch.from_numpy(curve_array).to(device=input_audio.device, dtype=torch.float32)
    audio_center = context.audio_embeddings[
        context.query_start_frame : context.query_start_frame + VIDEO_QUERY_COUNT
    ].detach()
    base_norm, _, base_cos = _embedding_metrics(torch, functional, base_visual_tensor, audio_center)
    metrics = _curve_metrics(torch, base_curve_tensor)
    record = {
        "row_id": row_key,
        "cache_fingerprint": fingerprint,
        "npz_path": str(npz_path),
        "metadata_path": str(meta_path),
        "npz_bytes": int(npz_path.stat().st_size),
        "npz_file_sha256": _sha256_file(npz_path),
        "metadata_file_sha256": _sha256_file(meta_path),
        "array_sha256": {name: _array_sha(array) for name, array in arrays.items()},
        "metrics": {
            **{key: _finite_float(value) for key, value in metrics.items()},
            "mean_visual_embedding_l2": _finite_float(base_norm.mean()),
            "mean_zero_lag_unit_cosine": _finite_float(base_cos.mean()),
        },
        "legacy_rms_control_parity": json.loads(meta_path.read_text(encoding="utf-8"))["legacy_rms_control_parity"],
        "binding_sha256": _canonical_sha(binding),
    }
    cache_records[:] = [item for item in cache_records if item["row_id"] != row_key]
    cache_records.append(record)
    cache_records.sort(key=lambda item: item["row_id"])
    manifest = {
        "schema": "cross_tfg_bounded_rms_visual_feature_cache_manifest_v1",
        "runtime_schema": RUNTIME_SCHEMA,
        "row_count": len(cache_records),
        "rows": cache_records,
    }
    manifest["payload_sha256"] = _canonical_sha(manifest)
    _atomic_json(cache_root / "baseline_cache_manifest.json", manifest)
    return base_tensor, base_visual_tensor, base_curve_tensor, record


def _finite_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().item()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite v3 baseline/training diagnostic")
    return result


def create_runtime(*, plan: dict[str, Any], fit_rows: list[dict[str, Any]], device: str, seed: int) -> dict[str, Any]:
    """Create the v3 fit-only runtime and prebuild all eight RMS baselines."""

    import torch
    import torch.nn.functional as functional

    fit_cfg = plan.get("fit_config", {})
    v3_cfg = fit_cfg.get("v3_objective")
    runtime_cfg = plan.get("runtime", {}).get("fit_v3")
    if not isinstance(v3_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise ValueError("v3 fit plan is missing its objective/runtime contract")
    if v3_cfg.get("schema") != RUNTIME_SCHEMA or runtime_cfg.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("v3 fit objective/runtime schema mismatch")
    expected_rows = int(v3_cfg.get("expected_fit_rows", -1))
    if len(fit_rows) != expected_rows or expected_rows != 8:
        raise ValueError("v3 training is frozen to the eight historical fit-only utterances")
    if any(row.get("split") != "fit" or not row.get("historical_exposed", False) for row in fit_rows):
        raise ValueError("v3 runtime may read only the pinned historical fit rows")

    # Reuse the complete, independently source-pinned v2 loader/context/cache;
    # its serialized plan subtree is identical and keeps the same audio cache.
    base_runtime = v2.create_runtime(plan=plan, fit_rows=fit_rows, device=device, seed=seed)
    stack = base_runtime["enhancement_stack"]
    optimizer = base_runtime["optimizer"]
    adapter = base_runtime["adapter"]
    base_context_for_row = base_runtime["context_for_row"]
    source_pins = base_runtime["source_pins"]
    full = plan["runtime"]["full_context"]
    rms_config = v3_cfg["rms_reference"]
    candidate_config = v3_cfg["candidate_audio_transform"]
    if candidate_config.get("use_global_rms") is not True or rms_config.get("use_global_rms") is not True:
        raise ValueError("candidate and bounded-RMS reference must both enable the same global RMS stage")
    if candidate_config["global_rms_target"] != rms_config["target_rms"]:
        raise ValueError("candidate and bounded-RMS reference must share an exact global RMS target")
    if candidate_config["rms_stage_peak_ceiling"] != rms_config["rms_stage_peak_ceiling"]:
        raise ValueError("candidate and bounded-RMS reference must share the RMS-stage peak bound")
    if candidate_config["post_rms_match"] != rms_config["post_rms_match"] or candidate_config["peak_ceiling"] != rms_config["peak_ceiling"]:
        raise ValueError("candidate and bounded-RMS reference must share final RMS/peak controls")
    if candidate_config["use_local_envelope"] or candidate_config["post_rms_match"] != "target":
        raise ValueError("v3 candidate transform is frozen to global RMS, spectral head, and post-RMS target")

    weights = v3_cfg["objective_weights"]
    if weights != {
        "relative_zero_lag_distance_reduction": 0.5,
        "zero_lag_unit_cosine_gain": 0.5,
        "visual_embedding_log_norm_ratio_squared": 0.1,
        "waveform_relative_mse_to_rms_reference": 0.02,
        "mask_total_variation": 0.002,
    }:
        raise ValueError("v3 objective weights differ from the frozen root specification")
    tolerances = v3_cfg["zero_init_tolerances"]
    cache_root = Path(str(runtime_cfg["baseline_feature_cache_root"]))
    if not str(cache_root).startswith("/dev/shm/cross_tfg_audio_head_20261003/fit_v3_fullcontext_attempt03/"):
        raise ValueError("v3 baseline feature cache must remain in its fresh attempt namespace")
    cache_root.mkdir(parents=True, exist_ok=True)
    runtime_path = Path(__file__).resolve()
    runtime_sha = _sha256_file(runtime_path)
    baseline_contexts: dict[str, Any] = {}
    baseline_tensors: dict[str, Any] = {}
    baseline_visuals: dict[str, Any] = {}
    baseline_curves: dict[str, Any] = {}
    baseline_records: list[dict[str, Any]] = []
    step_diagnostics: list[dict[str, Any]] = []
    zero_init_rows: list[dict[str, Any]] = []

    source_pins["v3_runtime"] = {
        "schema": RUNTIME_SCHEMA,
        "path": str(runtime_path),
        "bytes": int(runtime_path.stat().st_size),
        "sha256": runtime_sha,
        "uses_v2_context_cache_and_renderer": True,
        "baseline_feature_cache_root": str(cache_root),
        "baseline_cache_schema": BASELINE_CACHE_SCHEMA,
    }
    source_pins["v3_objective"] = {
        "schema": RUNTIME_SCHEMA,
        "direction": "maximize_reward; minimize_loss=-reward+regularizers",
        "weights": dict(weights),
        "zero_lag_column": ZERO_LAG_COLUMN,
        "fixed_clock_lag_frames": 0,
        "best_lag_or_B_reward_used": False,
        "baseline_name": "bounded_RMS_reference",
        "diagnostics_scope": "historical fit-only training proxy; not dev selection or official fresh-score replacement",
        "zero_init_tolerances": dict(tolerances),
        "steps": step_diagnostics,
    }
    source_pins["v3_baseline_feature_cache"] = {
        "schema": BASELINE_CACHE_SCHEMA,
        "root": str(cache_root),
        "baseline_rows": baseline_records,
        "legacy_fixed_rms_control": "computed per row from apply_global_rms(target_rms, peak_ceiling=None) for explicit parity/peak-activity diagnostics; no development audio opened",
    }
    source_pins["v3_zero_init_parity"] = {
        "status": "PENDING",
        "fit_rows": zero_init_rows,
        "candidate_vs_reference_audio_max_abs_tolerance": float(tolerances["audio_max_abs"]),
        "visual_embedding_max_abs_tolerance": float(tolerances["visual_embedding_max_abs"]),
        "initial_reward_abs_tolerance": float(tolerances["reward_abs"]),
        "regularizer_abs_tolerance": float(tolerances["regularizer_abs"]),
    }

    def _context_for_row(row: dict[str, Any]):
        row_id = str(row["row_id"])
        if row_id in baseline_contexts:
            return baseline_contexts[row_id]
        context = base_context_for_row(row)
        audio_path = Path(str(row["natural_pcm"]["path"])).resolve(strict=True)
        audio, info = read_canonical_wav(audio_path)
        natural = row["natural_pcm"]
        if (
            info.file_sha256 != str(natural["file_sha256"]).lower()
            or info.pcm_float32_sha256 != str(natural["pcm_sha256"]).lower()
            or info.frames != int(natural["sample_count"])
            or info.sample_rate != 16_000
            or info.source_channels != 1
        ):
            raise ValueError(f"v3 fit audio binding differs from frozen source: {row_id}")
        binding = _baseline_binding(
            plan=plan,
            row=row,
            context=context,
            source_pins=source_pins,
            rms_config=rms_config,
            runtime_sha=runtime_sha,
        )
        baseline, visual, curve, record = _cache_baseline(
            torch=torch,
            functional=functional,
            cache_root=cache_root,
            binding=binding,
            row=row,
            context=context,
            natural_audio=audio,
            enhancement_stack=stack,
            adapter=adapter,
            rms_config=rms_config,
            cache_records=baseline_records,
        )
        baseline_contexts[row_id] = context
        baseline_tensors[row_id] = baseline
        baseline_visuals[row_id] = visual
        baseline_curves[row_id] = curve
        return context

    def _candidate_audio(natural_audio):
        return stack.forward_with_mask(
            natural_audio,
            use_global_rms=True,
            use_local_envelope=False,
            use_spectral_head=True,
            post_rms_match="target",
            target_rms=float(rms_config["target_rms"]),
            peak_ceiling=float(rms_config["peak_ceiling"]),
            rms_stage_peak_ceiling=float(rms_config["rms_stage_peak_ceiling"]),
            return_diagnostics=True,
        )

    # Materialize all historical fit-only reference caches before optimization,
    # then test that the identity-initialized head reproduces each baseline.
    for row in fit_rows:
        context = _context_for_row(row)
        row_id = str(row["row_id"])
        audio, _ = read_canonical_wav(Path(str(row["natural_pcm"]["path"])).resolve(strict=True))
        natural_tensor = torch.from_numpy(audio).to(device=device, dtype=torch.float32)
        base_audio = baseline_tensors[row_id]
        base_visual = baseline_visuals[row_id]
        base_curve = baseline_curves[row_id]
        stack.eval()
        with torch.no_grad():
            candidate_audio, candidate_mask, _ = _candidate_audio(natural_tensor)
            candidate_flat = _flat_waveform(torch, candidate_audio)
            base_flat = _flat_waveform(torch, base_audio)
            audio_max_abs = torch.max(torch.abs(candidate_flat - base_flat))
            candidate_frames = adapter.render_video(context, candidate_audio)
            candidate_curve = adapter.sync_distance_curve(candidate_frames, context=context)
            candidate_visual = adapter.last_visual_embeddings
            audio_center = adapter.last_audio_center_embeddings
            if candidate_visual is None or audio_center is None:
                raise RuntimeError(f"v3 zero-init gate omitted candidate embeddings: {row_id}")
            visual_max_abs = torch.max(torch.abs(candidate_visual - base_visual))
            curve_max_abs = torch.max(torch.abs(candidate_curve - base_curve))
            candidate_norm, _, candidate_cos = _embedding_metrics(torch, functional, candidate_visual, audio_center)
            baseline_norm, _, baseline_cos = _embedding_metrics(torch, functional, base_visual, audio_center)
            candidate_metrics = _curve_metrics(torch, candidate_curve)
            baseline_metrics = _curve_metrics(torch, base_curve)
            initial_reward = (
                0.5 * (baseline_metrics["D0_zero_lag_distance"] - candidate_metrics["D0_zero_lag_distance"])
                / baseline_metrics["D0_zero_lag_distance"].clamp_min(1e-6)
                + 0.5 * (candidate_cos.mean() - baseline_cos.mean())
            )
            norm_reg = torch.mean(torch.log(candidate_norm / baseline_norm).square())
            wave_den = base_flat.square().mean().clamp_min(1e-8)
            waveform_reg = torch.mean((candidate_flat - base_flat).square()) / wave_den
            mask_tv = _mask_tv(torch, candidate_mask)
            checks = {
                "candidate_vs_reference_audio_max_abs": _finite_float(audio_max_abs),
                "candidate_vs_reference_visual_embedding_max_abs": _finite_float(visual_max_abs),
                "candidate_vs_reference_curve_max_abs": _finite_float(curve_max_abs),
                "initial_reward": _finite_float(initial_reward),
                "initial_norm_penalty": _finite_float(norm_reg),
                "initial_relative_waveform_mse": _finite_float(waveform_reg),
                "initial_mask_tv": _finite_float(mask_tv),
                "D0_RMS": _finite_float(baseline_metrics["D0_zero_lag_distance"]),
                "D0_identity_head": _finite_float(candidate_metrics["D0_zero_lag_distance"]),
                "cos0_RMS": _finite_float(baseline_cos.mean()),
                "cos0_identity_head": _finite_float(candidate_cos.mean()),
                "mean_vnorm_RMS": _finite_float(baseline_norm.mean()),
                "mean_vnorm_identity_head": _finite_float(candidate_norm.mean()),
            }
            if checks["candidate_vs_reference_audio_max_abs"] > float(tolerances["audio_max_abs"]):
                raise RuntimeError(f"v3 zero-init candidate does not reproduce bounded-RMS audio: {row_id}")
            if checks["candidate_vs_reference_visual_embedding_max_abs"] > float(tolerances["visual_embedding_max_abs"]):
                raise RuntimeError(f"v3 zero-init Wav2Lip/SyncNet feature parity failed: {row_id}")
            if checks["candidate_vs_reference_curve_max_abs"] > float(tolerances["curve_max_abs"]):
                raise RuntimeError(f"v3 zero-init SyncNet distance-curve parity failed: {row_id}")
            if abs(checks["initial_reward"]) > float(tolerances["reward_abs"]):
                raise RuntimeError(f"v3 zero-init reward is not approximately zero: {row_id}")
            for key in ("initial_norm_penalty", "initial_relative_waveform_mse", "initial_mask_tv"):
                if abs(checks[key]) > float(tolerances["regularizer_abs"]):
                    raise RuntimeError(f"v3 zero-init regularizer is not approximately zero ({key}): {row_id}")
        zero_init_rows.append({
            "row_id": row_id,
            "baseline_cache_sha256": next(record["npz_file_sha256"] for record in baseline_records if record["row_id"] == row_id),
            "status": "PASS",
            **checks,
        })
    source_pins["v3_zero_init_parity"]["status"] = "PASS"

    objective_cfg = v3_cfg["objective"]
    if objective_cfg != {
        "reward": "0.5*(D0_RMS-D0_head)/max(D0_RMS,1e-6)+0.5*(cos0_head-cos0_RMS)",
        "zero_lag_column": ZERO_LAG_COLUMN,
        "distance_curve_reduction": "v2 FullContextSyncAdapter.sync_distance_curve; float32 query mean; all 27 queries",
        "cosine": "mean querywise cosine_similarity of unnormalized forward_lip visual embedding and fixed natural forward_aud embedding at the same source-clock query; eps=1e-8",
        "norm_penalty": "mean((log(vnorm_head/vnorm_RMS))^2) over 27 query frames; both norms from unnormalized forward_lip outputs",
        "waveform_penalty": "mean((h(x)-R(x))^2)/max(mean(R(x)^2),1e-8)",
        "mask_tv": "mean(abs(diff_frequency(log_mask)))+mean(abs(diff_time(log_mask)))",
        "loss": "-reward+0.1*norm_penalty+0.02*waveform_penalty+0.002*mask_tv",
        "no_best_lag_reward": True,
        "no_B_reward": True,
        "post_rms_match": "target",
    }:
        raise ValueError("v3 objective formula differs from the frozen versioned contract")
    learning_rate = float(fit_cfg.get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("v3 fit learning rate must be finite and positive")

    def training_step(*, natural_audio, video_context, row):
        row_id = str(row["row_id"])
        context = video_context
        if row_id not in baseline_contexts:
            raise RuntimeError(f"v3 RMS baseline context was not precomputed before optimization: {row_id}")
        if context is not baseline_contexts[row_id]:
            raise RuntimeError(f"v3 fit CLI returned a context object different from its baseline cache: {row_id}")
        stack.train()
        optimizer.zero_grad(set_to_none=True)
        candidate, log_mask, _ = _candidate_audio(natural_audio)
        candidate_flat = _flat_waveform(torch, candidate)
        baseline_flat = _flat_waveform(torch, baseline_tensors[row_id])
        generated = adapter.render_video(context, candidate)
        curve = adapter.sync_distance_curve(generated, context=context)
        visual = adapter.last_visual_embeddings
        audio_center = adapter.last_audio_center_embeddings
        if visual is None or audio_center is None:
            raise RuntimeError(f"v3 candidate pass omitted SyncNet embeddings: {row_id}")
        base_visual = baseline_visuals[row_id].detach()
        base_curve = baseline_curves[row_id].detach()
        baseline_metrics = _curve_metrics(torch, base_curve)
        head_metrics = _curve_metrics(torch, curve)
        baseline_norm, _, baseline_cos = _embedding_metrics(torch, functional, base_visual, audio_center.detach())
        head_norm, _, head_cos = _embedding_metrics(torch, functional, visual, audio_center.detach())
        d0_reference = baseline_metrics["D0_zero_lag_distance"].detach()
        d0_head = head_metrics["D0_zero_lag_distance"]
        relative_d0_improvement = (d0_reference - d0_head) / d0_reference.clamp_min(1e-6)
        cosine_gain = head_cos.mean() - baseline_cos.mean().detach()
        reward = 0.5 * relative_d0_improvement + 0.5 * cosine_gain
        norm_penalty = torch.mean(torch.log(head_norm / baseline_norm.detach()).square())
        waveform_mse = torch.mean((candidate_flat - baseline_flat).square())
        waveform_energy = baseline_flat.square().mean().clamp_min(1e-8)
        relative_waveform_mse = waveform_mse / waveform_energy
        mask_total_variation = _mask_tv(torch, log_mask)
        loss = (
            -reward
            + weights["visual_embedding_log_norm_ratio_squared"] * norm_penalty
            + weights["waveform_relative_mse_to_rms_reference"] * relative_waveform_mse
            + weights["mask_total_variation"] * mask_total_variation
        )
        for name, value in (("reward", reward), ("loss", loss), ("norm_penalty", norm_penalty),
                            ("relative_waveform_mse", relative_waveform_mse), ("mask_total_variation", mask_total_variation)):
            if not torch.isfinite(value).item():
                raise FloatingPointError(f"v3 training objective {name} is non-finite for {row_id}")
        audio_gradient = torch.autograd.grad(reward, candidate, retain_graph=True, allow_unused=True)[0]
        if audio_gradient is None:
            raise RuntimeError(f"v3 objective has no gradient to enhanced audio: {row_id}")
        audio_gradient_l2_tensor = torch.linalg.vector_norm(audio_gradient.detach().to(torch.float64))
        if not torch.isfinite(audio_gradient_l2_tensor).item() or float(audio_gradient_l2_tensor.item()) <= 0.0:
            raise RuntimeError(f"v3 audio gradient is zero/non-finite: {row_id}")
        loss.backward()
        trainable_parameters = [parameter for parameter in stack.parameters() if parameter.requires_grad]
        gradient_sq = torch.zeros((), dtype=torch.float64, device=loss.device)
        has_gradient = False
        for parameter in trainable_parameters:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError(f"v3 head gradient is non-finite: {row_id}")
                gradient_sq += parameter.grad.detach().to(torch.float64).square().sum()
                has_gradient = True
        if not has_gradient or not torch.isfinite(gradient_sq).item() or float(gradient_sq.item()) <= 0.0:
            raise RuntimeError(f"v3 head gradient is absent/zero/non-finite: {row_id}")
        parameters_before = [parameter.detach().clone() for parameter in trainable_parameters]
        optimizer.step()
        update_sq = torch.zeros((), dtype=torch.float64, device=loss.device)
        for before, after in zip(parameters_before, trainable_parameters):
            update_sq += (after.detach() - before).to(torch.float64).square().sum()
        update_l2_tensor = torch.sqrt(update_sq)
        if not torch.isfinite(update_sq).item() or float(update_l2_tensor.item()) <= 0.0:
            raise RuntimeError(f"v3 optimizer did not make a finite nonzero head update: {row_id}")

        metrics = {
            "objective_name": RUNTIME_SCHEMA,
            "objective_reward": _finite_float(reward),
            "loss": _finite_float(loss),
            "D0_RMS": _finite_float(d0_reference),
            "D0_head": _finite_float(d0_head),
            "relative_D0_improvement": _finite_float(relative_d0_improvement),
            "B_RMS": _finite_float(baseline_metrics["B_median_distance"]),
            "Dmin_RMS": _finite_float(baseline_metrics["Dmin_best_lag_distance"]),
            "Cbest_RMS": _finite_float(baseline_metrics["Cbest_B_minus_Dmin"]),
            "Canchor_RMS": _finite_float(baseline_metrics["Canchor_zero_B_minus_D0"]),
            "B_head": _finite_float(head_metrics["B_median_distance"]),
            "Dmin_head": _finite_float(head_metrics["Dmin_best_lag_distance"]),
            "Cbest_head": _finite_float(head_metrics["Cbest_B_minus_Dmin"]),
            "Canchor_head": _finite_float(head_metrics["Canchor_zero_B_minus_D0"]),
            "cos0_RMS": _finite_float(baseline_cos.mean()),
            "cos0_head": _finite_float(head_cos.mean()),
            "cos0_gain": _finite_float(cosine_gain),
            "mean_vnorm_RMS": _finite_float(baseline_norm.mean()),
            "mean_vnorm_head": _finite_float(head_norm.mean()),
            "mean_log_vnorm_ratio_squared": _finite_float(norm_penalty),
            "waveform_mse_to_RMS": _finite_float(waveform_mse),
            "RMS_reference_mean_square": _finite_float(waveform_energy),
            "relative_waveform_mse_to_RMS": _finite_float(relative_waveform_mse),
            "mask_tv": _finite_float(mask_total_variation),
            "distance_reward_component": _finite_float(0.5 * relative_d0_improvement),
            "cosine_reward_component": _finite_float(0.5 * cosine_gain),
            "enhanced_waveform_max_abs_from_RMS": _finite_float(torch.max(torch.abs(candidate_flat.detach() - baseline_flat))),
            "audio_gradient_l2": _finite_float(audio_gradient_l2_tensor),
            "head_gradient_l2": _finite_float(torch.sqrt(gradient_sq)),
            "parameter_update_l2": _finite_float(update_l2_tensor),
            "baseline_feature_cache_sha256": next(
                record["npz_file_sha256"] for record in baseline_records if record["row_id"] == row_id
            ),
        }
        metrics["step_index"] = len(step_diagnostics) + 1
        metrics["row_id"] = row_id
        step_diagnostics.append(metrics)
        info = TrainStepInfo(
            sync_score_mean=metrics["objective_reward"],
            identity_penalty=metrics["relative_waveform_mse_to_RMS"],
            loss=metrics["loss"],
            gradient_l2=metrics["head_gradient_l2"],
            audio_gradient_l2=metrics["audio_gradient_l2"],
            parameter_update_l2=metrics["parameter_update_l2"],
            batch_size=1,
        )
        return info, metrics

    score_config = {
        "score_proxy": "v3_bounded_RMS_reference_relative_D0_reduction_plus_zero_lag_unit_cosine_gain",
        "anchor_lag_frames_used": False,
        "zero_lag_column": ZERO_LAG_COLUMN,
        "lag_count": 2 * LAG_RADIUS + 1,
        "video_frame_count": v2.VIDEO_FRAMES,
        "visual_query_count": VIDEO_QUERY_COUNT,
        "baseline_name": "bounded_RMS_reference",
        "training_proxy_only": True,
    }
    expected_fit_cli = {
        "use_global_rms": True,
        "use_local_envelope": False,
        "post_rms_match": "target",
        "target_rms": float(rms_config["target_rms"]),
        "peak_ceiling": float(rms_config["peak_ceiling"]),
        "rms_stage_peak_ceiling": float(rms_config["rms_stage_peak_ceiling"]),
        "identity_weight": float(weights["waveform_relative_mse_to_rms_reference"]),
        "spectral_smoothness_weight": float(weights["mask_total_variation"]),
        "anchor_lag_frames": 0,
        "epochs": int(fit_cfg["epochs"]),
        "max_steps": int(fit_cfg["steps_requested"]),
        "seed": int(fit_cfg["fit_seed"]),
    }
    return {
        **base_runtime,
        "context_for_row": _context_for_row,
        "training_step": training_step,
        "step_objective_name": RUNTIME_SCHEMA,
        "expected_fit_cli": expected_fit_cli,
        "score_config": score_config,
        "source_pins": source_pins,
    }


__all__ = ["create_runtime"]
