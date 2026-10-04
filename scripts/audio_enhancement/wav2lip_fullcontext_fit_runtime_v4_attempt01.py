"""Fit-only v4 native-crop objective for the full-natural-context Wav2Lip head.

This runtime reuses the frozen v2 full-audio and model loader, but replaces the
224-pixel proxy crop path with the pinned 512-pixel portrait paste, tracker
crop, and OpenCV resize. The original audio query clock remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import wav2lip_fullcontext_fit_runtime_v2_attempt02 as v2
from .audio_io import read_canonical_wav
from .baselines import apply_global_rms, waveform_rms
from .differentiable_adapter import prepare_wav2lip_face_input
from .differentiable_native_crop import build_native_crop_geometry, render_native_syncnet_crops
from .training import TrainStepInfo
from .wav2lip_frontend import official_chunk_starts


RUNTIME_SCHEMA = "cross_tfg_wav2lip_fullcontext_fit_runtime_v4_attempt01"
BASELINE_CACHE_SCHEMA = "cross_tfg_native512_rms_visual_feature_cache_v1"
LAG_RADIUS = v2.LAG_RADIUS
ZERO_LAG_COLUMN = LAG_RADIUS
VIDEO_QUERY_COUNT = v2.VIDEO_QUERY_COUNT
VIDEO_FRAMES = v2.VIDEO_FRAMES


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
        raise ValueError(f"unsafe v4 baseline cache key: {value!r}")
    return value


def _finite_scalar(value, name: str) -> float:
    scalar = float(value.detach().to(dtype=value.dtype).item()) if hasattr(value, "detach") else float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(f"non-finite v4 fit metric: {name}")
    return scalar


def _flat_waveform(torch, waveform):
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        return waveform[:, 0, :]
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim == 2:
        return waveform
    raise ValueError("v4 fit waveform must be mono [N], [B,N], or [B,1,N]")


def _mask_tv(torch, log_mask):
    if log_mask.numel() == 0 or log_mask.shape[-1] <= 1 or log_mask.shape[-2] <= 1:
        return log_mask.new_zeros(())
    tv_frequency = torch.mean(torch.abs(log_mask[..., 1:, :] - log_mask[..., :-1, :]))
    tv_time = torch.mean(torch.abs(log_mask[..., :, 1:] - log_mask[..., :, :-1]))
    return tv_frequency + tv_time


def _embedding_metrics(torch, functional, visual, audio_center):
    if visual.shape != audio_center.shape or visual.ndim != 2 or visual.shape[1] != 1024:
        raise ValueError("v4 visual/audio embedding query geometry must be equal [N,1024]")
    if not torch.isfinite(visual).all().item() or not torch.isfinite(audio_center).all().item():
        raise FloatingPointError("v4 SyncNet embeddings contain non-finite values")
    visual_norm = torch.linalg.vector_norm(visual, dim=-1)
    audio_norm = torch.linalg.vector_norm(audio_center, dim=-1)
    if (visual_norm <= 1e-8).any().item() or (audio_norm <= 1e-8).any().item():
        raise FloatingPointError("v4 embedding norm is too small for normalized cosine")
    cosine = functional.cosine_similarity(visual, audio_center, dim=-1, eps=1e-8)
    return visual_norm, audio_norm, cosine


def _curve_metrics(torch, curve):
    if tuple(curve.shape) != (2 * LAG_RADIUS + 1,) or not torch.isfinite(curve).all().item():
        raise FloatingPointError("v4 SyncNet curve must contain 31 finite lag distances")
    b_value = torch.median(curve)
    d_min = torch.min(curve)
    d_zero = curve[ZERO_LAG_COLUMN]
    lag_offsets = torch.arange(-LAG_RADIUS, LAG_RADIUS + 1, device=curve.device)
    far_mask = torch.abs(lag_offsets) >= 4
    far_contrast = torch.mean(curve[far_mask]) - d_zero
    return {
        "B_median_distance": b_value,
        "Dmin_best_lag_distance": d_min,
        "D0_zero_lag_distance": d_zero,
        "Cbest_B_minus_Dmin": b_value - d_min,
        "Canchor_zero_B_minus_D0": b_value - d_zero,
        "far_lag_contrast_abs4plus": far_contrast,
    }


def _load_native_geometry_manifest(plan: dict[str, Any], fit_rows: list[dict[str, Any]], runtime_cfg: dict[str, Any]):
    """Load only the sealed eight-row fit geometry and its 32-frame maps."""
    manifest_path = Path(str(runtime_cfg["native_geometry_manifest_path"])).resolve(strict=True)
    expected_sha = str(runtime_cfg["native_geometry_manifest_sha256"]).lower()
    if _sha256_file(manifest_path) != expected_sha:
        raise ValueError("v4 native geometry manifest raw SHA differs from the frozen plan")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1":
        raise ValueError("v4 native geometry manifest schema mismatch")
    manifest_payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if _canonical_sha(manifest_payload) != manifest.get("payload_sha256"):
        raise ValueError("v4 native geometry manifest payload digest mismatch")
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(fit_rows):
        raise ValueError("v4 native geometry manifest row count differs from the eight fit rows")
    by_id = {str(item.get("row_id")): item for item in manifest_rows if isinstance(item, dict)}
    if len(by_id) != len(manifest_rows) or set(by_id) != {str(row["row_id"]) for row in fit_rows}:
        raise ValueError("v4 native geometry manifest IDs are not an exact fit-row match")
    if int(manifest.get("source_clock_fps_num", -1)) != 25 or int(manifest.get("source_clock_fps_den", -1)) != 1:
        raise ValueError("v4 native geometry clock must remain 25/1 fps")
    if manifest.get("audio_clock_policy") != "unchanged_original_natural_pcm_mfcc_queries":
        raise ValueError("v4 manifest may not shift or reindex original audio queries")

    geometries: dict[str, Any] = {}
    map_records: list[dict[str, Any]] = []
    source_portrait_pin = manifest.get("source_portrait")
    if not isinstance(source_portrait_pin, dict):
        raise ValueError("v4 geometry manifest lacks the pinned static source portrait")
    portrait_path = Path(str(source_portrait_pin["path"])).resolve(strict=True)
    if _sha256_file(portrait_path) != str(source_portrait_pin["sha256"]).lower():
        raise ValueError("v4 static portrait file SHA differs from manifest")
    portrait = cv2.imread(str(portrait_path), cv2.IMREAD_COLOR)
    if portrait is None or portrait.dtype != np.uint8 or portrait.shape != (512, 512, 3):
        raise ValueError("v4 static portrait is not the pinned 512x512 uint8 BGR image")

    for row in fit_rows:
        row_id = str(row["row_id"])
        record = by_id[row_id]
        track_ref = record.get("tracks_pickle")
        if not isinstance(track_ref, dict):
            raise ValueError(f"v4 row lacks its selected frozen tracker file: {row_id}")
        track_path = Path(str(track_ref["path"])).resolve(strict=True)
        if _sha256_file(track_path) != str(track_ref["sha256"]).lower():
            raise ValueError(f"v4 tracker pickle SHA differs from manifest: {row_id}")
        with track_path.open("rb") as handle:
            tracks = pickle.load(handle)
        track_index = int(record["selected_track_index"])
        if not isinstance(tracks, list) or not (0 <= track_index < len(tracks)):
            raise ValueError(f"v4 selected track index is absent from pinned pickle: {row_id}")
        selected = tracks[track_index]
        proc_track = selected.get("proc_track") if isinstance(selected, dict) else None
        if not isinstance(proc_track, dict):
            raise ValueError(f"v4 selected track lacks the pinned proc_track mapping: {row_id}")
        output_frames = [int(value) for value in record.get("output_source_clock_frames", [])]
        track_local = [int(value) for value in record.get("track_local_frame_indices", [])]
        prediction_frames = [int(value) for value in record.get("prediction_source_clock_frames", [])]
        if any(len(values) != VIDEO_FRAMES for values in (output_frames, track_local, prediction_frames)):
            raise ValueError(f"v4 selected frame mapping must have 32 entries in every clock: {row_id}")
        if output_frames != list(range(output_frames[0], output_frames[0] + VIDEO_FRAMES)):
            raise ValueError(f"v4 target output frames are not a contiguous source-clock window: {row_id}")
        if track_local != list(range(track_local[0], track_local[0] + VIDEO_FRAMES)):
            raise ValueError(f"v4 tracker-local frame window is not contiguous: {row_id}")
        expected_q0 = (int(row["full_context_fit"]["scoring_audio_embedding_count"]) - VIDEO_QUERY_COUNT) // 2
        if int(record.get("query_start_frame", -1)) != expected_q0:
            raise ValueError(f"v4 video query start differs from the original audio query start: {row_id}")
        if output_frames[0] != int(record["query_start_frame"]):
            raise ValueError(f"v4 video and natural-audio clocks do not share the frozen query origin: {row_id}")
        mapping = record.get("prediction_map_evidence")
        receipt_sha = str(mapping.get("receipt_sha256", "")).lower() if isinstance(mapping, dict) else ""
        if len(receipt_sha) != 64 or any(ch not in "0123456789abcdef" for ch in receipt_sha):
            raise ValueError(f"v4 prediction map lacks the clock-receipt lineage: {row_id}")
        if any(value < 0 for value in prediction_frames):
            raise ValueError(f"v4 prediction mel map contains a negative source-clock frame: {row_id}")
        map_entries = mapping.get("entries")
        if not isinstance(map_entries, list) or len(map_entries) != VIDEO_FRAMES:
            raise ValueError(f"v4 measured frame map does not contain all 32 rendered source frames: {row_id}")
        for offset, entry in enumerate(map_entries):
            if (
                int(entry.get("jpeg_index_zero_based", -1)) != output_frames[offset]
                or int(entry.get("target_source_clock_frame", -1)) != output_frames[offset]
                or int(entry.get("mapped_full512_prediction_mel_index", -1)) != prediction_frames[offset]
            ):
                raise ValueError(f"v4 measured frame-map entry disagrees with its frozen vectors: {row_id}:{offset}")
        if not str(mapping.get("map_kind", "")).startswith("minimum_mae_monotone_correspondence"):
            raise ValueError(f"v4 frame map does not declare its measured minimum-MAE lineage: {row_id}")
        geometry = build_native_crop_geometry(
            portrait,
            record["face_box_y1_y2_x1_x2"],
            proc_track,
            track_local,
            output_frames,
            crop_scale=0.40,
        )
        if geometry.frame_count != VIDEO_FRAMES:
            raise RuntimeError(f"v4 native geometry did not produce 32 fit frames: {row_id}")
        geometries[row_id] = {
            "geometry": geometry,
            "prediction_source_clock_frames": tuple(prediction_frames),
            "output_source_clock_frames": tuple(output_frames),
            "track_local_frame_indices": tuple(track_local),
            "manifest_row": record,
        }
        map_records.append({
            "row_id": row_id,
            "query_start_frame": int(record["query_start_frame"]),
            "output_frame_indices": output_frames,
            "prediction_source_clock_frame_indices": prediction_frames,
            "track_local_frame_indices": track_local,
            "tracks_pickle_sha256": str(track_ref["sha256"]).lower(),
            "mapping_receipt_sha256": str(mapping["receipt_sha256"]).lower(),
            "geometry_sha256": _canonical_sha({
                "face_box": geometry.face_box_y1_y2_x1_x2,
                "crop_boxes": geometry.padded_crop_xyxy.tolist(),
                "padding_bsi": geometry.padding_bsi.tolist(),
                "frame_indices": geometry.frame_indices,
            }),
        })
    return manifest, geometries, map_records


def _render_native_video(adapter, context, enhanced_audio, geometry_record):
    """Render center-32 predictions, then native full512 paste/tracker crops."""
    import torch

    waveform = torch.as_tensor(enhanced_audio)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform[:, 0, :]
    if waveform.ndim == 2:
        if waveform.shape[0] != 1:
            raise ValueError("v4 fit renders one complete natural waveform per optimizer step")
        waveform = waveform[0]
    if waveform.ndim != 1 or waveform.numel() != context.expected_sample_count:
        raise ValueError("v4 enhancement changed the original waveform sample count")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all().item():
        raise ValueError("v4 enhancement waveform must remain finite float32")
    model_device = next(adapter.wav2lip_model.parameters()).device
    mel = adapter.mel_frontend(waveform.to(device=model_device, dtype=torch.float32))
    all_starts = official_chunk_starts(int(mel.shape[-1]), fps_num=25, fps_den=1)
    prediction_frames = list(geometry_record["prediction_source_clock_frames"])
    if any(index >= len(all_starts) for index in prediction_frames):
        raise ValueError("v4 measured native frame map exceeds the full-audio Wav2Lip mel clock")
    selected_starts = [all_starts[index] for index in prediction_frames]
    if any(start < 0 or start + 16 > int(mel.shape[-1]) for start in selected_starts):
        raise ValueError("v4 selected official mel chunk is incomplete on the unchanged full-audio clock")
    mel_chunks = torch.stack([mel[:, start : start + 16] for start in selected_starts], dim=0)
    if tuple(mel_chunks.shape) != (VIDEO_FRAMES, 80, 16):
        raise RuntimeError("v4 mapped prediction mel chunks differ from the 32-frame contract")
    face_input = prepare_wav2lip_face_input(
        torch.as_tensor(context.face_crops_bgr, device=model_device, dtype=torch.float32)
    )
    predictions = []
    for first in range(0, VIDEO_FRAMES, adapter.frame_batch_size):
        stop = min(VIDEO_FRAMES, first + adapter.frame_batch_size)
        prediction = adapter.wav2lip_model(
            mel_chunks[first:stop, None].contiguous(), face_input[first:stop].contiguous()
        )
        if tuple(prediction.shape) != (stop - first, 3, 96, 96) or not torch.isfinite(prediction).all().item():
            raise RuntimeError("v4 mapped Wav2Lip prediction shape/finite validation failed")
        predictions.append(prediction)
    prediction_tensor = torch.cat(predictions, dim=0)
    return render_native_syncnet_crops(prediction_tensor, geometry_record["geometry"])


def _baseline_binding(*, plan, row, context, source_pins, rms_config, runtime_sha, native_geometry_record):
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
        "runtime_v4_source_sha256": runtime_sha,
        "native_geometry_binding_sha256": _canonical_sha(native_geometry_record),
        "rms_reference": rms_config,
        "embedding_definition": {
            "visual": "FullContextSyncAdapter.forward_lip outputs before L2 normalization; query windows 0..26 on unchanged source-clock native tracked crops",
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
        raise RuntimeError(f"partial v4 bounded-RMS baseline cache entry: {row_key}")

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
        raise RuntimeError(f"v4 bounded-RMS reference changed sample count or became non-finite: {row_key}")

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
            raise ValueError(f"v4 baseline cache binding/file integrity mismatch: {row_key}")
        with np.load(npz_path, allow_pickle=False) as archive:
            expected_names = {
                "rms_reference_audio_float32",
                "rms_visual_embedding_float32",
                "rms_distance_curve_float32",
                "query_frame_index_int32",
            }
            if set(archive.files) != expected_names:
                raise ValueError(f"v4 baseline cache array inventory mismatch: {row_key}")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        if not np.array_equal(arrays["rms_reference_audio_float32"], reference_np):
            raise ValueError(f"recomputed bounded-RMS reference differs from sealed cache: {row_key}")
        for name, array in arrays.items():
            record = metadata["arrays"].get(name)
            if not isinstance(record, dict) or record.get("shape") != list(array.shape) or record.get("dtype") != str(array.dtype):
                raise ValueError(f"v4 baseline array metadata mismatch: {row_key}:{name}")
            if _array_sha(array) != record.get("array_sha256"):
                raise ValueError(f"v4 baseline array SHA mismatch: {row_key}:{name}")
    else:
        with torch.no_grad():
            rendered = adapter.render_video(context, reference)
            curve = adapter.sync_distance_curve(rendered, context=context)
            visual = adapter.last_visual_embeddings
            audio_center = adapter.last_audio_center_embeddings
            if visual is None or audio_center is None:
                raise RuntimeError(f"v4 baseline render omitted SyncNet embeddings: {row_key}")
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
        raise ValueError(f"v4 baseline RMS waveform array shape/dtype mismatch: {row_key}")
    if visual_array.shape != (VIDEO_QUERY_COUNT, 1024) or visual_array.dtype != np.float32:
        raise ValueError(f"v4 baseline visual embedding array shape/dtype mismatch: {row_key}")
    if curve_array.shape != (2 * LAG_RADIUS + 1,) or curve_array.dtype != np.float32:
        raise ValueError(f"v4 baseline curve shape/dtype mismatch: {row_key}")
    expected_queries = np.arange(context.query_start_frame, context.query_start_frame + VIDEO_QUERY_COUNT, dtype=np.int32)
    if query_array.dtype != np.int32 or not np.array_equal(query_array, expected_queries):
        raise ValueError(f"v4 baseline query frame map mismatch: {row_key}")
    if not all(np.isfinite(value).all() for value in (reference_array, visual_array, curve_array)):
        raise FloatingPointError(f"v4 baseline cache contains non-finite values: {row_key}")
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
        raise FloatingPointError("non-finite v4 baseline/training diagnostic")
    return result


def create_runtime(*, plan: dict[str, Any], fit_rows: list[dict[str, Any]], device: str, seed: int) -> dict[str, Any]:
    """Create the v4 native-crop fit runtime and prebuild the eight RMS baselines."""

    import torch
    import torch.nn.functional as functional

    fit_cfg = plan.get("fit_config", {})
    v4_cfg = fit_cfg.get("v4_objective")
    runtime_cfg = plan.get("runtime", {}).get("fit_v4")
    if not isinstance(v4_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise ValueError("v4 fit plan is missing its objective/runtime contract")
    if v4_cfg.get("schema") != RUNTIME_SCHEMA or runtime_cfg.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("v4 fit objective/runtime schema mismatch")
    expected_rows = int(v4_cfg.get("expected_fit_rows", -1))
    if len(fit_rows) != expected_rows or expected_rows != 8:
        raise ValueError("v4 training is frozen to the eight historical fit-only utterances")
    if any(row.get("split") != "fit" or not row.get("historical_exposed", False) for row in fit_rows):
        raise ValueError("v4 runtime may read only the pinned historical fit rows")

    # Reuse the complete, independently source-pinned v2 loader/context/cache;
    # its serialized plan subtree is identical and keeps the same audio cache.
    base_runtime = v2.create_runtime(plan=plan, fit_rows=fit_rows, device=device, seed=seed)
    stack = base_runtime["enhancement_stack"]
    optimizer = base_runtime["optimizer"]
    adapter = base_runtime["adapter"]
    base_context_for_row = base_runtime["context_for_row"]
    source_pins = base_runtime["source_pins"]
    full = plan["runtime"]["full_context"]
    rms_config = v4_cfg["rms_reference"]
    candidate_config = v4_cfg["candidate_audio_transform"]
    geometry_manifest, native_geometries, native_geometry_records = _load_native_geometry_manifest(
        plan, fit_rows, runtime_cfg
    )
    geometry_by_row = {record["row_id"]: record for record in native_geometry_records}
    native_geometry_by_row = {}
    for row_id, entry in native_geometries.items():
        native_geometry_by_row[row_id] = {
            "geometry": entry["geometry"],
            "prediction_source_clock_frames": entry["prediction_source_clock_frames"],
        }

    # Replace only the visual render side of the v2 adapter. Its frozen full
    # natural-audio embeddings, query_start, 27-query support, and 31-lag
    # SyncNet reduction stay unchanged.
    def _native_render(context, enhanced_audio):
        row_id = str(context.cache_record["row_id"])
        if row_id not in native_geometry_by_row:
            raise KeyError(f"v4 native geometry manifest has no row: {row_id}")
        return _render_native_video(adapter, context, enhanced_audio, native_geometry_by_row[row_id])

    adapter.render_video = _native_render
    if candidate_config.get("use_global_rms") is not True or rms_config.get("use_global_rms") is not True:
        raise ValueError("candidate and bounded-RMS reference must both enable the same global RMS stage")
    if candidate_config["global_rms_target"] != rms_config["target_rms"]:
        raise ValueError("candidate and bounded-RMS reference must share an exact global RMS target")
    if candidate_config["rms_stage_peak_ceiling"] != rms_config["rms_stage_peak_ceiling"]:
        raise ValueError("candidate and bounded-RMS reference must share the RMS-stage peak bound")
    if candidate_config["post_rms_match"] != rms_config["post_rms_match"] or candidate_config["peak_ceiling"] != rms_config["peak_ceiling"]:
        raise ValueError("candidate and bounded-RMS reference must share final RMS/peak controls")
    if candidate_config["use_local_envelope"] or candidate_config["post_rms_match"] != "target":
        raise ValueError("v4 candidate transform is frozen to global RMS, spectral head, and post-RMS target")

    weights = v4_cfg["objective_weights"]
    if weights != {
        "relative_Cbest_improvement": 1.0,
        "far_lag_contrast_improvement": 0.25,
        "D0_nonworsening_guard": 2.0,
        "Canchor_nonworsening_guard": 1.0,
        "visual_embedding_log_norm_ratio_squared": 0.1,
        "waveform_relative_mse_to_rms_reference": 0.02,
        "mask_total_variation": 0.002,
    }:
        raise ValueError("v4 objective weights differ from the frozen root specification")
    tolerances = v4_cfg["zero_init_tolerances"]
    if v4_cfg.get("one_sided_guard_slack") != {"D0": 0.0, "Canchor": 0.0}:
        raise ValueError("v4 D0/Canchor one-sided guard slacks must remain zero")
    if int(fit_cfg.get("steps_requested", -1)) != 96 or int(fit_cfg.get("epochs", -1)) != 12:
        raise ValueError("v4 fit is frozen to 96 steps over 12 epochs")
    if int(fit_cfg.get("fit_seed", -1)) != 20261003:
        raise ValueError("v4 fit seed differs from the frozen historical fit seed")
    cache_root = Path(str(runtime_cfg["baseline_feature_cache_root"]))
    if not str(cache_root).startswith("/dev/shm/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_attempt01/"):
        raise ValueError("v4 baseline feature cache must remain in its fresh attempt namespace")
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

    source_pins["v4_runtime"] = {
        "schema": RUNTIME_SCHEMA,
        "path": str(runtime_path),
        "bytes": int(runtime_path.stat().st_size),
        "sha256": runtime_sha,
        "uses_v2_full_audio_cache_and_replaces_proxy_with_native_crop_renderer": True,
        "baseline_feature_cache_root": str(cache_root),
        "baseline_cache_schema": BASELINE_CACHE_SCHEMA,
    }
    source_pins["v4_objective"] = {
        "schema": RUNTIME_SCHEMA,
        "direction": "maximize_reward; minimize_loss=-reward+regularizers",
        "weights": dict(weights),
        "zero_lag_column": ZERO_LAG_COLUMN,
        "fixed_clock_lag_frames": 0,
        "Cbest_reward_uses_median_and_best_lag": True,
        "standalone_B_reward_used": False,
        "baseline_name": "bounded_RMS_reference",
        "diagnostics_scope": "historical fit-only training proxy; not dev selection or official fresh-score replacement",
        "zero_init_tolerances": dict(tolerances),
        "steps": step_diagnostics,
    }
    source_pins["v4_native_geometry"] = {
        "manifest_path": str(Path(str(runtime_cfg["native_geometry_manifest_path"])).resolve(strict=True)),
        "manifest_bytes": int(Path(str(runtime_cfg["native_geometry_manifest_path"])).stat().st_size),
        "manifest_sha256": str(runtime_cfg["native_geometry_manifest_sha256"]).lower(),
        "manifest_payload_sha256": geometry_manifest["payload_sha256"],
        "rows": native_geometry_records,
        "prediction_clock_policy": "per-row minimum-MAE source-frame map from the pinned rendered-video to SyncNet JPEG audit; audio MFCC/query timestamps remain original source-clock q",
    }
    source_pins["v4_baseline_feature_cache"] = {
        "schema": BASELINE_CACHE_SCHEMA,
        "root": str(cache_root),
        "baseline_rows": baseline_records,
        "legacy_fixed_rms_control": "computed per row from apply_global_rms(target_rms, peak_ceiling=None) for explicit parity/peak-activity diagnostics; no development audio opened",
    }
    source_pins["v4_zero_init_parity"] = {
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
            raise ValueError(f"v4 fit audio binding differs from frozen source: {row_id}")
        binding = _baseline_binding(
            plan=plan,
            row=row,
            context=context,
            source_pins=source_pins,
            rms_config=rms_config,
            runtime_sha=runtime_sha,
            native_geometry_record=geometry_by_row[row_id],
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
                raise RuntimeError(f"v4 zero-init gate omitted candidate embeddings: {row_id}")
            visual_max_abs = torch.max(torch.abs(candidate_visual - base_visual))
            curve_max_abs = torch.max(torch.abs(candidate_curve - base_curve))
            candidate_norm, _, candidate_cos = _embedding_metrics(torch, functional, candidate_visual, audio_center)
            baseline_norm, _, baseline_cos = _embedding_metrics(torch, functional, base_visual, audio_center)
            candidate_metrics = _curve_metrics(torch, candidate_curve)
            baseline_metrics = _curve_metrics(torch, base_curve)
            delta_cbest = candidate_metrics["Cbest_B_minus_Dmin"] - baseline_metrics["Cbest_B_minus_Dmin"]
            delta_far = candidate_metrics["far_lag_contrast_abs4plus"] - baseline_metrics["far_lag_contrast_abs4plus"]
            d0_guard_violation = torch.relu(
                candidate_metrics["D0_zero_lag_distance"] - baseline_metrics["D0_zero_lag_distance"]
            )
            anchor_guard_violation = torch.relu(
                baseline_metrics["Canchor_zero_B_minus_D0"] - candidate_metrics["Canchor_zero_B_minus_D0"]
            )
            d0_guard_penalty = weights["D0_nonworsening_guard"] * d0_guard_violation.square()
            anchor_guard_penalty = weights["Canchor_nonworsening_guard"] * anchor_guard_violation.square()
            initial_reward = delta_cbest + weights["far_lag_contrast_improvement"] * delta_far - d0_guard_penalty - anchor_guard_penalty
            norm_reg = torch.mean(torch.log(candidate_norm / baseline_norm).square())
            wave_den = base_flat.square().mean().clamp_min(1e-8)
            waveform_reg = torch.mean((candidate_flat - base_flat).square()) / wave_den
            mask_tv = _mask_tv(torch, candidate_mask)
            checks = {
                "candidate_vs_reference_audio_max_abs": _finite_float(audio_max_abs),
                "candidate_vs_reference_visual_embedding_max_abs": _finite_float(visual_max_abs),
                "candidate_vs_reference_curve_max_abs": _finite_float(curve_max_abs),
                "initial_reward": _finite_float(initial_reward),
                "initial_delta_Cbest": _finite_float(delta_cbest),
                "initial_delta_far_lag_contrast_abs4plus": _finite_float(delta_far),
                "initial_D0_guard_violation": _finite_float(d0_guard_violation),
                "initial_Canchor_guard_violation": _finite_float(anchor_guard_violation),
                "initial_D0_guard_penalty": _finite_float(d0_guard_penalty),
                "initial_Canchor_guard_penalty": _finite_float(anchor_guard_penalty),
                "initial_norm_penalty": _finite_float(norm_reg),
                "initial_relative_waveform_mse": _finite_float(waveform_reg),
                "initial_mask_tv": _finite_float(mask_tv),
                "D0_RMS": _finite_float(baseline_metrics["D0_zero_lag_distance"]),
                "D0_identity_head": _finite_float(candidate_metrics["D0_zero_lag_distance"]),
                "Cbest_RMS": _finite_float(baseline_metrics["Cbest_B_minus_Dmin"]),
                "Cbest_identity_head": _finite_float(candidate_metrics["Cbest_B_minus_Dmin"]),
                "Canchor_RMS": _finite_float(baseline_metrics["Canchor_zero_B_minus_D0"]),
                "Canchor_identity_head": _finite_float(candidate_metrics["Canchor_zero_B_minus_D0"]),
                "far_lag_contrast_RMS": _finite_float(baseline_metrics["far_lag_contrast_abs4plus"]),
                "far_lag_contrast_identity_head": _finite_float(candidate_metrics["far_lag_contrast_abs4plus"]),
                "best_lag_RMS": int(torch.argmin(base_curve).item()) - LAG_RADIUS,
                "best_lag_identity_head": int(torch.argmin(candidate_curve).item()) - LAG_RADIUS,
                "cos0_RMS": _finite_float(baseline_cos.mean()),
                "cos0_identity_head": _finite_float(candidate_cos.mean()),
                "mean_vnorm_RMS": _finite_float(baseline_norm.mean()),
                "mean_vnorm_identity_head": _finite_float(candidate_norm.mean()),
            }
            if checks["candidate_vs_reference_audio_max_abs"] > float(tolerances["audio_max_abs"]):
                raise RuntimeError(f"v4 zero-init candidate does not reproduce bounded-RMS audio: {row_id}")
            if checks["candidate_vs_reference_visual_embedding_max_abs"] > float(tolerances["visual_embedding_max_abs"]):
                raise RuntimeError(f"v4 zero-init Wav2Lip/SyncNet feature parity failed: {row_id}")
            if checks["candidate_vs_reference_curve_max_abs"] > float(tolerances["curve_max_abs"]):
                raise RuntimeError(f"v4 zero-init SyncNet distance-curve parity failed: {row_id}")
            if abs(checks["initial_reward"]) > float(tolerances["reward_abs"]):
                raise RuntimeError(f"v4 zero-init reward is not approximately zero: {row_id}")
            for key in ("initial_norm_penalty", "initial_relative_waveform_mse", "initial_mask_tv"):
                if abs(checks[key]) > float(tolerances["regularizer_abs"]):
                    raise RuntimeError(f"v4 zero-init regularizer is not approximately zero ({key}): {row_id}")
        zero_init_rows.append({
            "row_id": row_id,
            "baseline_cache_sha256": next(record["npz_file_sha256"] for record in baseline_records if record["row_id"] == row_id),
            "status": "PASS",
            **checks,
        })
    source_pins["v4_zero_init_parity"]["status"] = "PASS"

    objective_cfg = v4_cfg["objective"]
    if objective_cfg != {
        "reward": "(Cbest_head-Cbest_RMS)+0.25*(far_abs4plus_head-far_abs4plus_RMS)-2*relu(D0_head-D0_RMS)^2-relu(Canchor_RMS-Canchor_head)^2",
        "zero_lag_column": ZERO_LAG_COLUMN,
        "distance_curve_reduction": "v2 FullContextSyncAdapter.sync_distance_curve; float32 query mean; all 27 queries",
        "far_lag_contrast": "mean(D_lag-D0) over lag_frames with absolute value >=4, after float32 per-lag query-mean reduction",
        "Cbest": "median_float32(curve)-min_float32(curve); first argmin is diagnostic only",
        "Canchor": "median_float32(curve)-curve_float32[zero_lag_column]",
        "fixed_clock": "source-clock q is unchanged; per-row measured minimum-MAE map selects only the Wav2Lip conditioning mel frame for output frame q",
        "norm_penalty": "mean((log(vnorm_head/vnorm_RMS))^2) over 27 query frames; both norms from unnormalized forward_lip outputs",
        "waveform_penalty": "mean((h(x)-R(x))^2)/max(mean(R(x)^2),1e-8)",
        "mask_tv": "mean(abs(diff_frequency(log_mask)))+mean(abs(diff_time(log_mask)))",
        "loss": "-reward+0.1*norm_penalty+0.02*waveform_penalty+0.002*mask_tv",
        "no_standalone_B_reward": True,
        "no_audio_retiming": True,
        "D0_guard_slack": 0.0,
        "Canchor_guard_slack": 0.0,
        "post_rms_match": "target",
    }:
        raise ValueError("v4 objective formula differs from the frozen versioned contract")
    learning_rate = float(fit_cfg.get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("v4 fit learning rate must be finite and positive")

    def training_step(*, natural_audio, video_context, row):
        row_id = str(row["row_id"])
        context = video_context
        if row_id not in baseline_contexts:
            raise RuntimeError(f"v4 RMS baseline context was not precomputed before optimization: {row_id}")
        if context is not baseline_contexts[row_id]:
            raise RuntimeError(f"v4 fit CLI returned a context object different from its baseline cache: {row_id}")
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
            raise RuntimeError(f"v4 candidate pass omitted SyncNet embeddings: {row_id}")
        base_visual = baseline_visuals[row_id].detach()
        base_curve = baseline_curves[row_id].detach()
        baseline_metrics = _curve_metrics(torch, base_curve)
        head_metrics = _curve_metrics(torch, curve)
        baseline_norm, _, baseline_cos = _embedding_metrics(torch, functional, base_visual, audio_center.detach())
        head_norm, _, head_cos = _embedding_metrics(torch, functional, visual, audio_center.detach())
        d0_reference = baseline_metrics["D0_zero_lag_distance"].detach()
        d0_head = head_metrics["D0_zero_lag_distance"]
        canchor_reference = baseline_metrics["Canchor_zero_B_minus_D0"].detach()
        canchor_head = head_metrics["Canchor_zero_B_minus_D0"]
        cbest_delta = head_metrics["Cbest_B_minus_Dmin"] - baseline_metrics["Cbest_B_minus_Dmin"].detach()
        far_delta = head_metrics["far_lag_contrast_abs4plus"] - baseline_metrics["far_lag_contrast_abs4plus"].detach()
        d0_guard_violation = torch.relu(d0_head - d0_reference)
        canchor_guard_violation = torch.relu(canchor_reference - canchor_head)
        d0_guard_penalty = weights["D0_nonworsening_guard"] * d0_guard_violation.square()
        canchor_guard_penalty = weights["Canchor_nonworsening_guard"] * canchor_guard_violation.square()
        cbest_reward_component = cbest_delta
        far_reward_component = weights["far_lag_contrast_improvement"] * far_delta
        reward = cbest_reward_component + far_reward_component - d0_guard_penalty - canchor_guard_penalty
        cosine_gain = head_cos.mean() - baseline_cos.mean().detach()
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
        for name, value in (("reward", reward), ("loss", loss), ("D0_guard_penalty", d0_guard_penalty),
                            ("Canchor_guard_penalty", canchor_guard_penalty), ("norm_penalty", norm_penalty),
                            ("relative_waveform_mse", relative_waveform_mse), ("mask_total_variation", mask_total_variation)):
            if not torch.isfinite(value).item():
                raise FloatingPointError(f"v4 training objective {name} is non-finite for {row_id}")
        audio_gradient = torch.autograd.grad(loss, candidate, retain_graph=True, allow_unused=True)[0]
        if audio_gradient is None:
            raise RuntimeError(f"v4 objective has no gradient to enhanced audio: {row_id}")
        audio_gradient_l2_tensor = torch.linalg.vector_norm(audio_gradient.detach().to(torch.float64))
        if not torch.isfinite(audio_gradient_l2_tensor).item() or float(audio_gradient_l2_tensor.item()) <= 0.0:
            raise RuntimeError(f"v4 audio gradient is zero/non-finite: {row_id}")
        loss.backward()
        trainable_parameters = [parameter for parameter in stack.parameters() if parameter.requires_grad]
        gradient_sq = torch.zeros((), dtype=torch.float64, device=loss.device)
        has_gradient = False
        for parameter in trainable_parameters:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError(f"v4 head gradient is non-finite: {row_id}")
                gradient_sq += parameter.grad.detach().to(torch.float64).square().sum()
                has_gradient = True
        if not has_gradient or not torch.isfinite(gradient_sq).item() or float(gradient_sq.item()) <= 0.0:
            raise RuntimeError(f"v4 head gradient is absent/zero/non-finite: {row_id}")
        parameters_before = [parameter.detach().clone() for parameter in trainable_parameters]
        optimizer.step()
        update_sq = torch.zeros((), dtype=torch.float64, device=loss.device)
        for before, after in zip(parameters_before, trainable_parameters):
            update_sq += (after.detach() - before).to(torch.float64).square().sum()
        update_l2_tensor = torch.sqrt(update_sq)
        if not torch.isfinite(update_sq).item() or float(update_l2_tensor.item()) <= 0.0:
            raise RuntimeError(f"v4 optimizer did not make a finite nonzero head update: {row_id}")

        metrics = {
            "objective_name": RUNTIME_SCHEMA,
            "objective_reward": _finite_float(reward),
            "loss": _finite_float(loss),
            "D0_RMS": _finite_float(d0_reference),
            "D0_head": _finite_float(d0_head),
            "delta_Cbest": _finite_float(cbest_delta),
            "delta_far_lag_contrast_abs4plus": _finite_float(far_delta),
            "delta_D0": _finite_float(d0_head - d0_reference),
            "delta_Canchor": _finite_float(canchor_head - canchor_reference),
            "D0_guard_violation": _finite_float(d0_guard_violation),
            "Canchor_guard_violation": _finite_float(canchor_guard_violation),
            "D0_guard_penalty": _finite_float(d0_guard_penalty),
            "Canchor_guard_penalty": _finite_float(canchor_guard_penalty),
            "B_RMS": _finite_float(baseline_metrics["B_median_distance"]),
            "Dmin_RMS": _finite_float(baseline_metrics["Dmin_best_lag_distance"]),
            "Cbest_RMS": _finite_float(baseline_metrics["Cbest_B_minus_Dmin"]),
            "Canchor_RMS": _finite_float(baseline_metrics["Canchor_zero_B_minus_D0"]),
            "far_lag_contrast_RMS": _finite_float(baseline_metrics["far_lag_contrast_abs4plus"]),
            "best_lag_RMS": int(torch.argmin(base_curve).item()) - LAG_RADIUS,
            "B_head": _finite_float(head_metrics["B_median_distance"]),
            "Dmin_head": _finite_float(head_metrics["Dmin_best_lag_distance"]),
            "Cbest_head": _finite_float(head_metrics["Cbest_B_minus_Dmin"]),
            "Canchor_head": _finite_float(head_metrics["Canchor_zero_B_minus_D0"]),
            "far_lag_contrast_head": _finite_float(head_metrics["far_lag_contrast_abs4plus"]),
            "best_lag_head": int(torch.argmin(curve).item()) - LAG_RADIUS,
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
            "Cbest_reward_component": _finite_float(cbest_reward_component),
            "far_lag_reward_component": _finite_float(far_reward_component),
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
        "score_proxy": "v4_bounded_RMS_delta_Cbest_plus_0.25_delta_far_lag_contrast_with_D0_Canchor_guards",
        "anchor_lag_frames_used": True,
        "zero_lag_column": ZERO_LAG_COLUMN,
        "lag_count": 2 * LAG_RADIUS + 1,
        "video_frame_count": v2.VIDEO_FRAMES,
        "visual_query_count": VIDEO_QUERY_COUNT,
        "baseline_name": "bounded_RMS_reference",
        "far_lag_definition": "mean lag distance minus D0 over |lag_frames|>=4",
        "source_clock_policy": "no audio shift; per-row minimum-MAE mapping changes only Wav2Lip conditioning mel indices",
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
