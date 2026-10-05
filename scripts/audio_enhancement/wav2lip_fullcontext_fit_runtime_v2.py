"""Full-natural-audio, center-32 differentiable Wav2Lip fit runtime.

This is an isolated v2 factory for the historical fit-only subset. It keeps
the complete natural waveform and SyncNet audio-embedding sequence, renders
only a centered 32-frame Wav2Lip window, and compares its 27 visual queries to
the corresponding full-audio queries at all 31 unpadded lags. The existing
``wav2lip_fit_runtime`` remains untouched for provenance of the earlier
short-window run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import read_canonical_wav
from .differentiable_adapter import prepare_wav2lip_face_input
from .heads import EnhancementStack, SpectralResidualHead
from .wav2lip_fit_runtime import _load_context_seed, _load_models
from .wav2lip_frontend import check_frontend_parity, official_chunk_starts


RUNTIME_SCHEMA = "cross_tfg_wav2lip_fullcontext_fit_runtime_v2"
CACHE_SCHEMA = "cross_tfg_syncnet_full_audio_embedding_cache_v1"
VIDEO_FRAMES = 32
VIDEO_QUERY_COUNT = VIDEO_FRAMES - 5
LAG_RADIUS = 15
LAG_COUNT = 2 * LAG_RADIUS + 1
SOFTMIN_TEMPERATURE = 0.1
MFCC_WINDOW = 20
MFCC_HOP = 4
FRAME_SAMPLES = 640


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_row_key(row_id: str) -> str:
    if not row_id or re.fullmatch(r"[A-Za-z0-9_.-]+", row_id) is None:
        raise ValueError(f"unsafe fit row id for cache path: {row_id!r}")
    return row_id


def _pcm16_from_float32(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)
    if values.dtype != np.float32 or values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("full-context source must be nonempty finite mono float32")
    pcm16 = np.rint(np.clip(values.astype(np.float64) * 32768.0, -32768.0, 32767.0)).astype("<i2")
    return np.ascontiguousarray(pcm16)


def _straight_through_truncate(value):
    bounded = value.clamp(0.0, 255.0)
    integer = bounded.floor()
    return bounded + (integer - bounded).detach()


def _straight_through_round(value):
    bounded = value.clamp(0.0, 255.0)
    integer = bounded.round()
    return bounded + (integer - bounded).detach()


@dataclass(frozen=True)
class FullContextFitContext:
    face_crops_bgr: np.ndarray
    syncnet_crops_bgr: np.ndarray
    patch_boxes_xyxy: np.ndarray
    mel_starts: tuple[int, ...]
    source_video_frame_indices: tuple[int, ...]
    frame_timestamps_seconds: tuple[float, ...]
    audio_embeddings: Any
    audio_embedding_count: int
    query_start_frame: int
    query_count: int
    expected_sample_count: int
    source_video_sha256: str
    cache_record: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.face_crops_bgr.shape[0])

    def validate(self) -> None:
        import torch

        if self.frame_count != VIDEO_FRAMES:
            raise ValueError("full-context fit requires exactly 32 frozen video frames")
        if self.query_count != VIDEO_QUERY_COUNT:
            raise ValueError("full-context fit requires exactly 27 SyncNet visual queries")
        if self.syncnet_crops_bgr.shape != (VIDEO_FRAMES, 3, 224, 224):
            raise ValueError("full-context SyncNet crop shape mismatch")
        if self.face_crops_bgr.shape != (VIDEO_FRAMES, 3, 96, 96):
            raise ValueError("full-context Wav2Lip face crop shape mismatch")
        if self.patch_boxes_xyxy.shape != (VIDEO_FRAMES, 4):
            raise ValueError("full-context patch geometry shape mismatch")
        if len(self.mel_starts) != VIDEO_FRAMES or len(self.source_video_frame_indices) != VIDEO_FRAMES:
            raise ValueError("full-context frame selectors must contain 32 indices")
        if len(self.frame_timestamps_seconds) != VIDEO_FRAMES:
            raise ValueError("full-context timestamps must contain 32 values")
        if self.source_video_frame_indices != tuple(
            range(self.query_start_frame, self.query_start_frame + VIDEO_FRAMES)
        ):
            raise ValueError("full-context selected video frames are not a contiguous source-clock window")
        timestamps = np.asarray(self.frame_timestamps_seconds, dtype=np.float64)
        if timestamps.size > 1 and not np.allclose(np.diff(timestamps), 1.0 / 25.0, rtol=0.0, atol=1e-12):
            raise ValueError("full-context center window does not preserve the exact 25 fps clock")
        for box in np.asarray(self.patch_boxes_xyxy).tolist():
            x1, y1, x2, y2 = [int(value) for value in box]
            if not (0 <= x1 < x2 <= 224 and 0 <= y1 < y2 <= 224):
                raise ValueError(f"invalid full-context SyncNet patch box: {box}")
        if self.audio_embeddings.shape != (self.audio_embedding_count, 1024):
            raise ValueError("cached full-audio SyncNet embedding dimensions differ")
        if not torch.isfinite(self.audio_embeddings).all().item():
            raise ValueError("cached full-audio SyncNet embeddings are non-finite")
        if self.query_start_frame != (self.audio_embedding_count - VIDEO_QUERY_COUNT) // 2:
            raise ValueError("query start does not use the frozen centered full-audio rule")
        if self.audio_embedding_count < 2 * LAG_RADIUS + VIDEO_QUERY_COUNT:
            raise ValueError("full audio has insufficient support for all 31 unpadded lags")
        min_audio = self.query_start_frame - LAG_RADIUS
        max_audio = self.query_start_frame + self.query_count - 1 + LAG_RADIUS
        if min_audio < 0 or max_audio >= self.audio_embedding_count:
            raise ValueError("full-context lag map leaves the natural-audio embedding sequence")
        if not np.isfinite(np.asarray(self.frame_timestamps_seconds, dtype=np.float64)).all():
            raise ValueError("full-context source timestamps are non-finite")


class FullContextSyncAdapter:
    def __init__(self, *, wav2lip_model, syncnet_model, mel_frontend, frame_batch_size: int, syncnet_batch_size: int):
        import torch

        self.torch = torch
        self.F = torch.nn.functional
        self.wav2lip_model = wav2lip_model
        self.syncnet_model = syncnet_model
        self.mel_frontend = mel_frontend
        self.frame_batch_size = int(frame_batch_size)
        self.syncnet_batch_size = int(syncnet_batch_size)
        self.last_visual_embeddings = None
        self.last_audio_center_embeddings = None
        if self.frame_batch_size <= 0 or self.syncnet_batch_size <= 0:
            raise ValueError("full-context batch sizes must be positive")

    def render_video(self, context: FullContextFitContext, enhanced_audio):
        """Render the fixed center-32 frame window from a full enhanced waveform."""

        torch = self.torch
        context.validate()
        waveform = torch.as_tensor(enhanced_audio)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                raise ValueError("full-context fit currently renders one waveform per optimizer step")
            waveform = waveform[0]
        if waveform.ndim != 1 or waveform.numel() != context.expected_sample_count:
            raise ValueError("enhanced waveform changed the frozen full-context sample count")
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all().item():
            raise ValueError("enhanced full-context waveform must be finite floating point")

        device = next(self.wav2lip_model.parameters()).device
        mel = self.mel_frontend(waveform.to(device=device, dtype=torch.float32))
        all_starts = official_chunk_starts(int(mel.shape[-1]), fps_num=25, fps_den=1)
        expected_indices = list(context.source_video_frame_indices)
        if max(expected_indices) >= len(all_starts):
            raise ValueError("full-audio Wav2Lip mel clock does not cover the frozen center-32 frame range")
        selected_starts = tuple(all_starts[index] for index in expected_indices)
        if selected_starts != context.mel_starts:
            raise ValueError("enhanced mel frame geometry differs from the frozen full-audio Wav2Lip clock")
        mel_chunks = torch.stack([mel[:, start : start + 16] for start in selected_starts], dim=0)
        if tuple(mel_chunks.shape) != (VIDEO_FRAMES, 80, 16):
            raise RuntimeError("center-32 official Wav2Lip mel chunk shape mismatch")

        face_input = prepare_wav2lip_face_input(torch.as_tensor(context.face_crops_bgr, device=device))
        predictions = []
        for first in range(0, VIDEO_FRAMES, self.frame_batch_size):
            stop = min(VIDEO_FRAMES, first + self.frame_batch_size)
            mel_batch = mel_chunks[first:stop, None].contiguous()
            image_batch = face_input[first:stop].contiguous()
            prediction = self.wav2lip_model(mel_batch, image_batch)
            if not torch.is_tensor(prediction) or tuple(prediction.shape) != (stop - first, 3, 96, 96):
                shape = tuple(prediction.shape) if torch.is_tensor(prediction) else type(prediction).__name__
                raise RuntimeError(f"Wav2Lip center-frame output shape mismatch: {shape}")
            if not torch.isfinite(prediction).all().item():
                raise FloatingPointError("Wav2Lip emitted non-finite center frames")
            predictions.append(prediction)

        patches = _straight_through_truncate(torch.cat(predictions, dim=0) * 255.0)
        crops = torch.as_tensor(context.syncnet_crops_bgr, device=device, dtype=torch.float32)
        boxes = torch.as_tensor(context.patch_boxes_xyxy, device=device, dtype=torch.int64)
        output_frames = []
        for index in range(VIDEO_FRAMES):
            x1, y1, x2, y2 = [int(value) for value in boxes[index].tolist()]
            patch = self.F.interpolate(
                patches[index : index + 1],
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )[0]
            patch = _straight_through_round(patch)
            frame = crops[index].clone()
            frame[:, y1:y2, x1:x2] = patch
            output_frames.append(frame)
        rendered = torch.stack(output_frames, dim=0)
        if tuple(rendered.shape) != (VIDEO_FRAMES, 3, 224, 224) or not torch.isfinite(rendered).all().item():
            raise RuntimeError("full-context center-32 paste violated shape/finite contract")
        return rendered

    def sync_distance_curve(self, generated_video_crops, *, context: FullContextFitContext):
        """Score center-video queries against cached full-audio embeddings, no padding."""

        torch = self.torch
        context.validate()
        if tuple(generated_video_crops.shape) != (VIDEO_FRAMES, 3, 224, 224):
            raise ValueError("SyncNet must receive exactly 32 center-window scoring crops")
        visual_windows = [
            generated_video_crops[index : index + 5].permute(1, 0, 2, 3)
            for index in range(VIDEO_QUERY_COUNT)
        ]
        video_input = torch.stack(visual_windows, dim=0).contiguous()
        if tuple(video_input.shape) != (VIDEO_QUERY_COUNT, 3, 5, 224, 224):
            raise RuntimeError("full-context SyncNet visual-window shape mismatch")
        visual_parts = []
        for first in range(0, VIDEO_QUERY_COUNT, self.syncnet_batch_size):
            stop = min(VIDEO_QUERY_COUNT, first + self.syncnet_batch_size)
            # Keep input gradients through the frozen SyncNet weights.
            visual = self.syncnet_model.forward_lip(video_input[first:stop])
            if tuple(visual.shape) != (stop - first, 1024) or not torch.isfinite(visual).all().item():
                raise RuntimeError("SyncNet visual encoder returned unexpected embeddings")
            visual_parts.append(visual)
        visual_embeddings = torch.cat(visual_parts, dim=0)
        audio_embeddings = context.audio_embeddings
        if audio_embeddings.device != visual_embeddings.device:
            audio_embeddings = audio_embeddings.to(device=visual_embeddings.device)
        audio_centers = audio_embeddings[
            context.query_start_frame : context.query_start_frame + VIDEO_QUERY_COUNT
        ]
        self.last_visual_embeddings = visual_embeddings
        self.last_audio_center_embeddings = audio_centers
        rows = []
        for query in range(VIDEO_QUERY_COUNT):
            audio_center = context.query_start_frame + query
            first_audio = audio_center - LAG_RADIUS
            stop_audio = audio_center + LAG_RADIUS + 1
            candidates = audio_embeddings[first_audio:stop_audio]
            if tuple(candidates.shape) != (LAG_COUNT, 1024):
                raise RuntimeError("full-context 31-lag audio window is incomplete")
            visual_query = visual_embeddings[query : query + 1].expand(LAG_COUNT, -1)
            rows.append(self.F.pairwise_distance(visual_query, candidates))
        distance_by_query = torch.stack(rows, dim=1)  # [31 lags, 27 fixed-clock queries]
        curve = distance_by_query.mean(dim=1)
        if tuple(curve.shape) != (LAG_COUNT,) or not torch.isfinite(curve).all().item():
            raise FloatingPointError("full-context 31-lag distance curve is invalid")
        return curve


def _cache_manifest_write(cache_root: Path, rows: list[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {
        "schema": "cross_tfg_syncnet_full_audio_embedding_cache_manifest_v1",
        "row_count": len(rows),
        "rows": rows,
    }
    payload["manifest_sha256"] = _canonical_sha(payload)
    _atomic_json(cache_root / "cache_manifest.json", payload)


def _load_or_build_audio_cache(
    *, cache_root: Path, row: dict[str, Any], audio: np.ndarray, pcm16: np.ndarray,
    psf_mfcc, syncnet_model, device, runtime_source_pins: dict[str, Any],
    row_cache_records: list[dict[str, Any]], batch_size: int,
):
    import torch

    natural = row["natural_pcm"]
    row_id = str(row["row_id"])
    key = _safe_row_key(row_id)
    pcm_sha = hashlib.sha256(pcm16.astype("<i2", copy=False).tobytes()).hexdigest()
    if natural.get("direct_source_pcm16_sha256") != pcm_sha:
        raise ValueError(f"full-context source PCM16 digest mismatch: {row_id}")
    cache_binding = {
        "schema": CACHE_SCHEMA,
        "row_id": row_id,
        "sample_id": str(row["sample_id"]),
        "speaker_id": str(row["speaker_id"]),
        "source_pcm_file_sha256": str(natural["file_sha256"]).lower(),
        "source_pcm_float32_sha256": str(natural["pcm_sha256"]).lower(),
        "source_pcm16_sha256": pcm_sha,
        "sample_count": int(audio.size),
        "sample_rate_hz": 16_000,
        "mfcc_frontend": "pinned python_speech_features.mfcc defaults; int16 PCM; samplerate=16000",
        "mfcc_window_frames": MFCC_WINDOW,
        "mfcc_hop_frames": MFCC_HOP,
        "syncnet_source_sha256": runtime_source_pins["syncnet_sources"],
        "syncnet_checkpoint": runtime_source_pins["syncnet_checkpoint"],
        "python_speech_features": runtime_source_pins["python_speech_features"],
        "torch_version": runtime_source_pins["torch_version"],
        "numpy_version": runtime_source_pins["numpy_version"],
        "embedding_batch_size": int(batch_size),
        "embedding_dtype": "float32",
        "padding": "none; all 31 lag indices must be in full natural-audio sequence",
    }
    fingerprint = _canonical_sha(cache_binding)
    npz_path = cache_root / f"{key}.{fingerprint[:16]}.npz"
    meta_path = cache_root / f"{key}.{fingerprint[:16]}.json"

    if npz_path.exists() or meta_path.exists():
        if not npz_path.is_file() or not meta_path.is_file():
            raise RuntimeError(f"partial full-context audio cache entry: {row_id}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema") != CACHE_SCHEMA
            or metadata.get("cache_binding") != cache_binding
            or metadata.get("cache_fingerprint") != fingerprint
        ):
            raise ValueError(f"full-context cache metadata differs from inputs: {row_id}")
        if _sha256_file(npz_path) != metadata.get("npz_sha256"):
            raise ValueError(f"full-context cache NPZ file hash mismatch: {row_id}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {"mfcc_frames_by_coeff", "audio_embeddings", "audio_window_start_mfcc"}:
                raise ValueError(f"full-context cache NPZ inventory mismatch: {row_id}")
            mfcc_array = np.ascontiguousarray(loaded["mfcc_frames_by_coeff"])
            embedding_array = np.ascontiguousarray(loaded["audio_embeddings"])
            starts_array = np.ascontiguousarray(loaded["audio_window_start_mfcc"])
        for name, array in (("mfcc_frames_by_coeff", mfcc_array), ("audio_embeddings", embedding_array),
                            ("audio_window_start_mfcc", starts_array)):
            expected = metadata["arrays"][name]
            if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
                raise ValueError(f"full-context cache array metadata mismatch: {row_id}:{name}")
            if _array_sha(array) != expected["array_sha256"]:
                raise ValueError(f"full-context cache array SHA mismatch: {row_id}:{name}")
        mfcc = mfcc_array
        embeddings_np = embedding_array
        starts = starts_array
    else:
        features = np.asarray(psf_mfcc(pcm16, samplerate=16_000), dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 13 or not np.isfinite(features).all():
            raise RuntimeError(f"full-context SyncNet MFCC returned invalid values: {row_id}")
        window_starts = np.arange(0, features.shape[0] - MFCC_WINDOW + 1, MFCC_HOP, dtype=np.int32)
        if window_starts.size == 0:
            raise ValueError(f"full-context fit audio has no SyncNet MFCC windows: {row_id}")
        windows = np.stack(
            [features[int(start) : int(start) + MFCC_WINDOW].T for start in window_starts], axis=0
        ).astype(np.float32, copy=False)
        audio_input = torch.from_numpy(np.ascontiguousarray(windows[:, None, :, :])).to(device=device)
        embedding_parts = []
        syncnet_model.eval()
        with torch.no_grad():
            for first in range(0, int(audio_input.shape[0]), batch_size):
                stop = min(int(audio_input.shape[0]), first + batch_size)
                values = syncnet_model.forward_aud(audio_input[first:stop])
                if tuple(values.shape) != (stop - first, 1024) or not torch.isfinite(values).all().item():
                    raise RuntimeError(f"SyncNet audio encoder returned invalid full-context values: {row_id}")
                embedding_parts.append(values.detach().to(dtype=torch.float32).cpu().numpy())
        embeddings_np = np.ascontiguousarray(np.concatenate(embedding_parts, axis=0), dtype=np.float32)
        mfcc = np.ascontiguousarray(features, dtype=np.float32)
        starts = np.ascontiguousarray(window_starts, dtype=np.int32)
        query_count = VIDEO_QUERY_COUNT
        query_start = (int(embeddings_np.shape[0]) - query_count) // 2
        if int(embeddings_np.shape[0]) < LAG_COUNT + query_count - 1:
            raise ValueError(f"fit utterance is too short for a full 31-lag center context: {row_id}")
        if query_start - LAG_RADIUS < 0 or query_start + query_count - 1 + LAG_RADIUS >= embeddings_np.shape[0]:
            raise RuntimeError(f"centered lag support calculation failed: {row_id}")
        if not np.array_equal(starts, np.arange(embeddings_np.shape[0], dtype=np.int32) * MFCC_HOP):
            raise RuntimeError(f"SyncNet MFCC audio-window clock is not contiguous 25fps: {row_id}")

        arrays = {
            "mfcc_frames_by_coeff": mfcc,
            "audio_embeddings": embeddings_np,
            "audio_window_start_mfcc": starts,
        }
        cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", dir=cache_root, prefix=f".{key}.", suffix=".npz.tmp", delete=False) as handle:
            temporary_npz = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_npz, npz_path)
        array_meta = {
            name: {"dtype": str(array.dtype), "shape": [int(value) for value in array.shape],
                   "array_sha256": _array_sha(array)}
            for name, array in arrays.items()
        }
        metadata = {
            "schema": CACHE_SCHEMA,
            "cache_binding": cache_binding,
            "cache_fingerprint": fingerprint,
            "npz_path": str(npz_path),
            "npz_bytes": npz_path.stat().st_size,
            "npz_sha256": _sha256_file(npz_path),
            "arrays": array_meta,
            "audio_embedding_count": int(embeddings_np.shape[0]),
            "query_start_frame": int(query_start),
            "query_count": VIDEO_QUERY_COUNT,
            "all_31_lags_valid": True,
            "created_by": RUNTIME_SCHEMA,
        }
        temporary_meta = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
        temporary_meta.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_meta, meta_path)

    query_count = VIDEO_QUERY_COUNT
    query_start = (int(embeddings_np.shape[0]) - query_count) // 2
    if int(embeddings_np.shape[0]) < LAG_COUNT + query_count - 1:
        raise ValueError(f"fit utterance is too short for a full 31-lag center context: {row_id}")
    if query_start - LAG_RADIUS < 0 or query_start + query_count - 1 + LAG_RADIUS >= embeddings_np.shape[0]:
        raise RuntimeError(f"cached center query lies outside 31-lag support: {row_id}")
    expected_starts = np.arange(embeddings_np.shape[0], dtype=np.int32) * MFCC_HOP
    if not np.array_equal(starts, expected_starts):
        raise ValueError(f"cached audio embeddings are not on the exact 25fps source clock: {row_id}")
    if _array_sha(mfcc) != metadata["arrays"]["mfcc_frames_by_coeff"]["array_sha256"]:
        raise ValueError(f"cached MFCC changed during load: {row_id}")

    record = {
        "row_id": row_id,
        "cache_binding": cache_binding,
        "cache_fingerprint": fingerprint,
        "npz_path": str(npz_path),
        "npz_bytes": int(npz_path.stat().st_size),
        "npz_sha256": _sha256_file(npz_path),
        "metadata_path": str(meta_path),
        "metadata_sha256": _sha256_file(meta_path),
        "mfcc_frame_count": int(mfcc.shape[0]),
        "audio_embedding_count": int(embeddings_np.shape[0]),
        "query_start_frame": int(query_start),
        "query_count": int(query_count),
        "all_31_lags_valid": True,
        "array_sha256": {name: values["array_sha256"] for name, values in metadata["arrays"].items()},
    }
    row_cache_records[:] = [old for old in row_cache_records if old["row_id"] != row_id]
    row_cache_records.append(record)
    row_cache_records.sort(key=lambda item: item["row_id"])
    _cache_manifest_write(cache_root, row_cache_records)
    embeddings_tensor = torch.from_numpy(embeddings_np).to(device=device, dtype=torch.float32)
    return mfcc, embeddings_tensor, int(query_start), record


def create_runtime(*, plan: dict[str, Any], fit_rows: list[dict[str, Any]], device: str, seed: int) -> dict[str, Any]:
    """Factory compatible with the existing fit CLI, using a v2 full context."""

    import torch
    from torch.optim import Adam

    runtime_config = plan.get("runtime")
    if not isinstance(runtime_config, dict) or not runtime_config.get("source_context_is_training_proxy_only"):
        raise ValueError("v2 full-context runtime must remain explicitly training-proxy-only")
    full = runtime_config.get("full_context")
    if not isinstance(full, dict) or full.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("fit plan is missing the frozen full-context v2 runtime contract")
    if int(full.get("video_frame_count", -1)) != VIDEO_FRAMES or int(full.get("visual_query_count", -1)) != VIDEO_QUERY_COUNT:
        raise ValueError("full-context v2 frame/query counts differ from the pinned 32/27 design")
    if int(full.get("lag_radius_frames", -1)) != LAG_RADIUS or full.get("padding_policy") != "none":
        raise ValueError("full-context v2 requires the exact 31-lag no-padding policy")
    cache_root = Path(str(full["audio_embedding_cache_root"])).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    arrays, context_meta, context_pins = _load_context_seed(runtime_config)
    w2l_model, sync_model, mel_frontend, audio_module, mfcc_function, source_pins = _load_models(
        runtime_config, device_obj
    )
    first_audio, first_info = read_canonical_wav(Path(str(fit_rows[0]["natural_pcm"]["path"])).resolve(strict=True))
    if first_info.file_sha256 != str(fit_rows[0]["natural_pcm"]["file_sha256"]).lower():
        raise ValueError("full-context parity input differs from the frozen original PCM WAV")
    parity = check_frontend_parity(mel_frontend, audio_module, first_audio)
    cache_records: list[dict[str, Any]] = []
    cache_policy = {
        "schema": CACHE_SCHEMA,
        "cache_root": str(cache_root),
        "source": "full natural 16 kHz mono PCM16-derived float32 waveform; original source bytes pinned in each plan row",
        "mfcc_frontend": "python_speech_features.mfcc default parameters at samplerate 16000",
        "mfcc_window_frames": MFCC_WINDOW,
        "mfcc_hop_frames": MFCC_HOP,
        "embedding_sequence": "forward_aud over all contiguous 20-MFCC-frame windows at hop 4; source-clock index q maps to 25 fps frame q",
        "query_rule": "q0=floor((N_audio_embeddings-27)/2); 27 video queries; lags -15..+15; require all 31 audio indices valid; no padding",
        "batch_size": int(full["audio_embedding_batch_size"]),
        "cache_rows": cache_records,
    }
    source_pins.update({
        "v2_runtime": {"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size,
                        "sha256": _sha256_file(Path(__file__).resolve()), "schema": RUNTIME_SCHEMA},
        "full_context_portrait_context": context_pins,
        "full_context_audio_embedding_cache": cache_policy,
        "full_context_frontend_parity": parity,
        "full_context_geometry": {
            "video_frames": VIDEO_FRAMES,
            "visual_queries": VIDEO_QUERY_COUNT,
            "audio_query_center_formula": "floor((N_audio_embeddings-27)/2)",
            "frame_rate": "25/1 fps",
            "audio_mfcc_window_frames": MFCC_WINDOW,
            "audio_mfcc_hop_frames": MFCC_HOP,
            "no_lag_padding": True,
        },
        "seed": int(seed),
    })
    first_face_context = arrays["face_crops_bgr_u8"]
    frame_capacity = int(first_face_context.shape[0])
    if frame_capacity < VIDEO_FRAMES:
        raise ValueError("pinned portrait context has fewer than 32 source-clock frames")
    base_video_sha = str(context_meta["source_video_sha256"])
    context_cache: dict[str, FullContextFitContext] = {}
    adapter = FullContextSyncAdapter(
        wav2lip_model=w2l_model,
        syncnet_model=sync_model,
        mel_frontend=mel_frontend,
        frame_batch_size=int(full["video_frame_batch_size"]),
        syncnet_batch_size=int(full["syncnet_batch_size"]),
    )

    def context_for_row(row: dict[str, Any]) -> FullContextFitContext:
        row_id = str(row["row_id"])
        if row_id in context_cache:
            return context_cache[row_id]
        natural = row["natural_pcm"]
        path = Path(str(natural["path"])).resolve(strict=True)
        audio, audio_info = read_canonical_wav(path)
        if (
            audio_info.file_sha256 != str(natural["file_sha256"]).lower()
            or audio_info.pcm_float32_sha256 != str(natural["pcm_sha256"]).lower()
            or audio_info.frames != int(natural["sample_count"])
            or audio_info.sample_rate != 16_000
            or audio_info.source_channels != 1
        ):
            raise ValueError(f"full-context fit audio binding mismatch: {row_id}")
        pcm16 = _pcm16_from_float32(audio)
        mfcc_array, audio_embeddings, query_start, cache_record = _load_or_build_audio_cache(
            cache_root=cache_root,
            row=row,
            audio=audio,
            pcm16=pcm16,
            psf_mfcc=mfcc_function,
            syncnet_model=sync_model,
            device=device_obj,
            runtime_source_pins=source_pins,
            row_cache_records=cache_records,
            batch_size=int(full["audio_embedding_batch_size"]),
        )
        with torch.no_grad():
            full_mel = mel_frontend(torch.from_numpy(audio).to(device=device_obj, dtype=torch.float32))
        all_starts = official_chunk_starts(int(full_mel.shape[-1]), fps_num=25, fps_den=1)
        if query_start < 0 or query_start + VIDEO_FRAMES > len(all_starts):
            raise ValueError(f"full natural Wav2Lip mel clock cannot cover center-32 selection: {row_id}")
        if query_start + VIDEO_FRAMES > frame_capacity:
            raise ValueError(f"fixed portrait proxy context does not cover centered full-audio frame selection: {row_id}")
        indices = np.arange(query_start, query_start + VIDEO_FRAMES, dtype=np.int32)
        face_crops = np.ascontiguousarray(arrays["face_crops_bgr_u8"][indices])
        sync_crops = np.ascontiguousarray(arrays["syncnet_crops_bgr_u8"][indices])
        boxes = np.ascontiguousarray(arrays["patch_boxes_xyxy_i32"][indices])
        timestamps = tuple(float(index) / 25.0 for index in indices.tolist())
        context = FullContextFitContext(
            face_crops_bgr=face_crops,
            syncnet_crops_bgr=sync_crops,
            patch_boxes_xyxy=boxes,
            mel_starts=tuple(int(value) for value in all_starts[query_start : query_start + VIDEO_FRAMES]),
            source_video_frame_indices=tuple(int(value) for value in indices.tolist()),
            frame_timestamps_seconds=timestamps,
            audio_embeddings=audio_embeddings,
            audio_embedding_count=int(audio_embeddings.shape[0]),
            query_start_frame=int(query_start),
            query_count=VIDEO_QUERY_COUNT,
            expected_sample_count=int(audio.size),
            source_video_sha256=base_video_sha,
            cache_record=cache_record,
        )
        context.validate()
        # Independently bind the full MFCC clock and center selection to this
        # utterance before any optimization step is permitted.
        if int(mfcc_array.shape[0]) < MFCC_WINDOW or context.audio_embedding_count <= 0:
            raise RuntimeError("full-context MFCC/audio query support mismatch")
        context_cache[row_id] = context
        return context

    fit_cfg = plan.get("fit_config")
    if not isinstance(fit_cfg, dict) or fit_cfg.get("optimizer") != "Adam":
        raise ValueError("full-context plan requires Adam")
    if fit_cfg.get("fit_seed") is not None and int(fit_cfg["fit_seed"]) != int(seed):
        raise ValueError("fit CLI seed differs from the frozen full-context plan seed")
    learning_rate = float(fit_cfg.get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("full-context learning rate must be finite and positive")
    temperature = float(fit_cfg.get("proxy_loss", {}).get("temperature", SOFTMIN_TEMPERATURE))
    if temperature != SOFTMIN_TEMPERATURE:
        raise ValueError("full-context v2 softmin temperature is frozen at 0.1")

    diagnostics: list[dict[str, Any]] = []
    source_pins["full_context_training_diagnostics"] = {
        "schema": "cross_tfg_fullcontext_fit_step_diagnostics_v1",
        "role": "descriptive fit diagnostics only; embedding norm/cosine are not loss terms or selection criteria",
        "norm_penalty_applied": False,
        "natural_rms_identity_vnorm_reference": "not computed in this first full-context pass; reserve for a separately frozen ablation",
        "steps": diagnostics,
    }

    def sync_score(generated_video, natural_audio, *, context, anchor_lag_frames=0, vshift=LAG_RADIUS):
        del natural_audio, anchor_lag_frames
        if vshift != LAG_RADIUS:
            raise ValueError("full-context runtime lag radius is frozen at 15 frames")
        curve = adapter.sync_distance_curve(generated_video, context=context)
        softmin = -temperature * (torch.logsumexp(-curve / temperature, dim=0) - math.log(LAG_COUNT))
        score = curve.median() - softmin
        visual = adapter.last_visual_embeddings
        audio_center = adapter.last_audio_center_embeddings
        if visual is None or audio_center is None:
            raise RuntimeError("full-context adapter omitted fit diagnostics embeddings")
        visual_norm = torch.linalg.vector_norm(visual.detach(), dim=-1)
        audio_norm = torch.linalg.vector_norm(audio_center.detach(), dim=-1)
        visual_unit = torch.nn.functional.normalize(visual.detach(), dim=-1)
        audio_unit = torch.nn.functional.normalize(audio_center.detach(), dim=-1)
        unit_cosine = (visual_unit * audio_unit).sum(dim=-1)
        diagnostics.append({
            "step_index": len(diagnostics) + 1,
            "B_median_distance": float(curve.median().detach().item()),
            "D_min_distance": float(curve.min().detach().item()),
            "D0_zero_lag_distance": float(curve[LAG_RADIUS].detach().item()),
            "Cbest_B_minus_Dmin": float((curve.median() - curve.min()).detach().item()),
            "Canchor_B_minus_D0": float((curve.median() - curve[LAG_RADIUS]).detach().item()),
            "training_objective_B_minus_normalized_softmin": float(score.detach().item()),
            "mean_visual_embedding_l2": float(visual_norm.mean().item()),
            "mean_natural_audio_embedding_l2_zero_lag": float(audio_norm.mean().item()),
            "mean_zero_lag_unit_cosine": float(unit_cosine.mean().item()),
            "unit_cosine_definition": "cosine(F.normalize(visual_query), F.normalize(original_natural_audio_embedding_at_same_source_clock_query))",
        })
        return score

    head = SpectralResidualHead(
        n_fft=int(full.get("head_n_fft", 512)),
        hop_length=int(full.get("head_hop_length", 128)),
        hidden_channels=int(full.get("head_hidden_channels", 24)),
        max_residual_db=float(full.get("head_max_residual_db", 3.0)),
        level_normalize=bool(full.get("head_level_normalize", True)),
    )
    stack = EnhancementStack(head).to(device_obj)
    optimizer = Adam(stack.parameters(), lr=learning_rate)
    source_pins["full_context_loss"] = {
        "name": "fullaudio_center32_syncnet_median_minus_normalized_softmin_v2",
        "formula": "median(mean_query(D[query,lag], axis=query)) - normalized_softmin_31lags",
        "temperature": temperature,
        "audio_embeddings_from_original_natural_pcm": True,
        "video_frames": VIDEO_FRAMES,
        "video_queries": VIDEO_QUERY_COUNT,
        "lag_offsets_frames": list(range(-LAG_RADIUS, LAG_RADIUS + 1)),
        "zero_padding": False,
        "audio_query_start": "floor((N_audio_embeddings-27)/2)",
    }
    return {
        "enhancement_stack": stack,
        "optimizer": optimizer,
        "adapter": adapter,
        "sync_score": sync_score,
        "context_for_row": context_for_row,
        "score_config": {
            "score_proxy": "fullcontext_syncnet_31lag_median_minus_normalized_softmin_v2",
            "softmin_temperature": temperature,
            "lag_count": LAG_COUNT,
            "anchor_lag_frames_used": False,
            "video_frame_count": VIDEO_FRAMES,
            "visual_query_count": VIDEO_QUERY_COUNT,
            "no_zero_padding": True,
        },
        "source_pins": source_pins,
    }


__all__ = ["create_runtime"]
