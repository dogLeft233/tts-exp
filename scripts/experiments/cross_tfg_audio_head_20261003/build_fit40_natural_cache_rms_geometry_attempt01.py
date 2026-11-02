#!/usr/bin/env python3
"""Build fit40 original-audio caches and RMS full512 geometry, without scoring.

This producer is intentionally independent of the v4 eight-row manifest builder.
It consumes the sealed 40-row historical-fit source manifest, builds the v2
SyncNet audio cache from original PCM16, then generates one official static-face
Wav2Lip RMS baseline per row.  It emits the V4-compatible row geometry fields
for a separate 40-row consumer, plus per-row measured full512→pipeline→JPEG
clock receipts.  It does not train, select, score SyncNet, or open dev/confirm
audio.

The native geometry manifest schema keeps the existing row contract while its
producer schema and scope identify the 40-row extension.  Each row includes:
row/sample/speaker identity; original source and canonical PCM hashes; full
natural MFCC/embedding cache binding; RMS control details; official full512 and
selected-track artifact pins; query q0; 32 source-clock output indices; 32
selected-track-local indices; 32 measured full512 prediction-frame indices;
the V4 field names and 32-entry map evidence required by a 40-row adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Match the literal fit environment before importing NumPy or PyTorch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


EXPECTED_SOURCE_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_V2_PLAN_SHA256 = "e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06"
EXPECTED_V2_RUNTIME_SHA256 = "d086c61cdb993bffa5f9ff7a55bf4b66b1b9127028f8b80eb3b5f1be74464d7b"
EXPECTED_FIT_RUNTIME_SHA256 = "7903fbd9dfebf95bcb388add26a1967bc39ad855d3d995a4730d316dd0e806c4"
EXPECTED_FRONTEND_SHA256 = "3480087daecf8dc295f4a3b20703f393bada2690e1f5ce7839ba1dc4fbdec8ef"
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
EXPECTED_PARITY_HELPER_SHA256 = "6712ad54e58b5302a77b9ee9ba3d3b938d3775d1d9e3f95c757ad8ea2122b570"
EXPECTED_CLOCK_HELPER_SHA256 = "930d8d7e308b4a11c31f7a99ea3776ad8790c9cd4c409185f30197f72257ce8b"
EXPECTED_CONTEXT_NPZ_SHA256 = "6616e649412cdf5c5f78ce51fa7d8403364c56528f93777a2e35eb82fc941ec1"
EXPECTED_CONTEXT_JSON_SHA256 = "46bab1ff647cbaabc3eea247dc315431d6ad976ca142a9c5e331ce4e81094450"
EXPECTED_PRIMARY_FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
EXPECTED_PSF_TREE_SHA256 = "a419b997acfd21da8f4c7324e5be22e32357d085e37ac3376165e676c1e67c17"
EXPECTED_W2L_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
EXPECTED_SYNCNET_CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
EXPECTED_PIPELINE_PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python"
EXPECTED_TORCH_VERSION = "2.5.1+cu124"
EXPECTED_NUMPY_VERSION = "1.26.4"
EXPECTED_FRAME_RATE_NUM = 25
EXPECTED_FRAME_RATE_DEN = 1
EXPECTED_FRAME_SAMPLES = 640
EXPECTED_MFCC_WINDOW = 20
EXPECTED_MFCC_HOP = 4
EXPECTED_VISUAL_FRAMES = 32
EXPECTED_VISUAL_QUERIES = 27
EXPECTED_LAG_RADIUS = 15
TARGET_RMS = 0.0376838172675746
MINIMUM_FREE_BYTES_DEFAULT = 1 << 30


class BuildError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_bytes(body.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pin_file(path: Path, expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise BuildError(f"expected regular file: {resolved}")
    size = resolved.stat().st_size
    digest = sha256_file(resolved)
    if expected_bytes is not None and size != int(expected_bytes):
        raise BuildError(f"file byte count differs from its source binding: {resolved}")
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise BuildError(f"file SHA differs from its source binding: {resolved}")
    return {"path": str(resolved), "bytes": size, "sha256": digest}


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load pinned helper source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def expected_fit_pairs() -> set[tuple[str, int]]:
    return {(f"S{number:04d}", j) for number in range(52, 72) for j in (0, 15)}


def expected_psf_mfcc_frames(sample_count: int) -> int:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count <= 400:
        return 1
    return 1 + int(math.ceil((sample_count - 400) / 160))


def source_rows_and_audio(*, manifest: dict[str, Any], manifest_path: Path, source_root: Path, read_wav):
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise BuildError("fit40 source manifest must contain exactly 40 rows")
    pairs = {(str(row.get("speaker_id")), int(row.get("canonical_j", -1))) for row in rows}
    if pairs != expected_fit_pairs() or len(pairs) != 40:
        raise BuildError("fit40 source roster is not all S0052–S0071 canonical j=0/j=15 rows")
    speaker_ids = sorted({str(row.get("speaker_id")) for row in rows})
    if speaker_ids != [f"S{number:04d}" for number in range(52, 72)]:
        raise BuildError("fit40 manifest speaker roster changed")

    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise BuildError("staged fit40 source root is not a directory")
    prepared: list[dict[str, Any]] = []
    for source_row in rows:
        relative = Path(str(source_row["transfer_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise BuildError(f"unsafe fit40 relative WAV path: {relative}")
        wav_path = (root / relative).resolve(strict=True)
        try:
            wav_path.relative_to(root)
        except ValueError as exc:
            raise BuildError(f"fit40 WAV escapes staged source root: {wav_path}") from exc
        source_pin = pin_file(
            wav_path,
            str(source_row["transfer_file_sha256"]),
            int(source_row["transfer_file_bytes"]),
        )
        audio, info = read_wav(wav_path)
        sample_count = int(source_row["sample_count"])
        canonical_sha = str(source_row["canonical_float32_pcm_sha256"]).lower()
        pcm_payload_sha = str(source_row["source_pcm_payload_sha256"]).lower()
        if (
            info.file_sha256 != source_pin["sha256"]
            or info.pcm_float32_sha256 != canonical_sha
            or int(info.frames) != sample_count
            or int(info.sample_rate) != 16_000
            or int(info.source_channels) != 1
            or str(source_row.get("source_encoding")) != "PCM_16"
        ):
            raise BuildError(f"fit40 source WAV canonical decode differs from manifest: {source_row['row_id']}")
        if audio.dtype != np.float32 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise BuildError(f"fit40 decoded canonical waveform is invalid: {source_row['row_id']}")

        # This exactly matches the pinned v2 runtime's float32→PCM16 conversion.
        pcm16 = np.rint(np.clip(audio.astype(np.float64) * 32768.0, -32768.0, 32767.0)).astype("<i2")
        pcm16 = np.ascontiguousarray(pcm16)
        if sha256_bytes(pcm16.tobytes()) != pcm_payload_sha:
            raise BuildError(f"canonical decode does not round-trip to source PCM16: {source_row['row_id']}")

        mfcc_frames = expected_psf_mfcc_frames(sample_count)
        all_valid_windows = max(0, (mfcc_frames - EXPECTED_MFCC_WINDOW) // EXPECTED_MFCC_HOP + 1)
        official_embedding_count = max(0, sample_count // EXPECTED_FRAME_SAMPLES - 5)
        if all_valid_windows < official_embedding_count:
            raise BuildError(f"PSF MFCC window count is below official SyncNet rows: {source_row['row_id']}")
        row_geometry = {
            "source_sample_count": sample_count,
            "source_duration_seconds": sample_count / 16_000.0,
            "source_clock_frame_count_floor_sample_count_over_640": sample_count // EXPECTED_FRAME_SAMPLES,
            "psf_mfcc_frame_count": mfcc_frames,
            "all_valid_mfcc_window_count": all_valid_windows,
            "scoring_audio_embedding_count": official_embedding_count,
            "valid_mfcc_windows_excluded_after_official_query_end": all_valid_windows - official_embedding_count,
            "audio_query_rule": "q0=floor((N_audio_embeddings-27)/2); 27 center queries and all 31 offsets -15..+15 must remain unpadded and in bounds",
            "lag_offsets_frames": list(range(-EXPECTED_LAG_RADIUS, EXPECTED_LAG_RADIUS + 1)),
            "video_frame_count": EXPECTED_VISUAL_FRAMES,
            "visual_query_count": EXPECTED_VISUAL_QUERIES,
        }
        if official_embedding_count < EXPECTED_VISUAL_QUERIES + 2 * EXPECTED_LAG_RADIUS:
            raise BuildError(f"fit row lacks all 31 natural-audio lag candidates: {source_row['row_id']}")
        q0 = (official_embedding_count - EXPECTED_VISUAL_QUERIES) // 2
        if q0 < EXPECTED_LAG_RADIUS or q0 + EXPECTED_VISUAL_QUERIES - 1 + EXPECTED_LAG_RADIUS >= official_embedding_count:
            raise BuildError(f"fit row centered query lacks full 31-lag support: {source_row['row_id']}")
        natural_pcm = {
            "path": str(wav_path),
            "file_sha256": source_pin["sha256"],
            "pcm_sha256": canonical_sha,
            "direct_source_pcm16_sha256": pcm_payload_sha,
            "sample_count": sample_count,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "source_encoding": "original PCM16 source WAV decoded to canonical float32 for Wav2Lip; exact source PCM16 payload retained for SyncNet MFCC and mux",
        }
        fit_row = {
            "row_id": str(source_row["row_id"]),
            "sample_id": str(source_row["sample_id"]),
            "speaker_id": str(source_row["speaker_id"]),
            "split": "fit",
            "historical_exposed": True,
            "canonical_j": int(source_row["canonical_j"]),
            "fold_index": int(source_row["fold_index"]),
            "natural_pcm": natural_pcm,
            "full_context_fit": row_geometry,
            "source_manifest_row": source_row,
        }
        prepared.append({"row": fit_row, "source": source_pin, "audio": audio, "pcm16": pcm16})
    if len({item["row"]["row_id"] for item in prepared}) != 40 or len({item["row"]["sample_id"] for item in prepared}) != 40:
        raise BuildError("fit40 source manifest has duplicate row or sample IDs")
    return prepared


def decode_and_map_row(*, row_root: Path, render_info: dict[str, Any], q0: int, clock_module, row_id: str):
    video = render_info["video"]
    tracking = render_info["tracking"]
    pipeline_dir = Path(str(tracking["pipeline_dir"])).resolve(strict=True)
    argv = [str(value) for value in video["run_pipeline_argv"]]
    if "--reference" not in argv:
        raise BuildError(f"run_pipeline argv has no declared reference: {row_id}")
    reference = argv[argv.index("--reference") + 1]
    intermediate_video = (pipeline_dir / "pyavi" / reference / "video.avi").resolve(strict=True)
    jpeg_dir = (pipeline_dir / "pyframes" / reference).resolve(strict=True)
    full512_path = Path(str(video["path"])).resolve(strict=True)
    full_pin = pin_file(full512_path, str(video["file_sha256"]), int(video["bytes"]))
    pipeline_video_pin = pin_file(intermediate_video)
    tracks_pin = pin_file(
        Path(str(tracking["tracks_pickle_path"])),
        str(tracking["tracks_pickle_sha256"]),
        int(tracking["tracks_pickle_bytes"]),
    )

    jpeg_paths = sorted(path for path in jpeg_dir.glob("*.jpg") if path.is_file())
    if not jpeg_paths:
        raise BuildError(f"pinned run_pipeline produced no JPEG frames: {row_id}")
    declared_jpegs = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in jpeg_paths
    ]
    checked_paths, jpeg_frames = clock_module.read_jpegs(jpeg_dir, declared_jpegs)
    full_frames = clock_module.decode_video(full512_path)
    pipeline_frames = clock_module.decode_video(intermediate_video)
    if int(video["decoded_frame_count"]) != len(full_frames):
        raise BuildError(f"full512 decoded count differs from renderer receipt: {row_id}")
    if len(pipeline_frames) != len(checked_paths):
        raise BuildError(f"pipeline AVI and JPEG frame counts differ: {row_id}")

    jpeg_to_pipeline = clock_module.monotone_duplicate_map(
        clock_module.downsample(pipeline_frames), clock_module.downsample(jpeg_frames)
    )
    pipeline_to_full512 = clock_module.monotone_duplicate_map(
        clock_module.downsample(full_frames), clock_module.downsample(pipeline_frames)
    )
    pipeline_to_source = [
        int(item["mapped_source_frame_index_zero_based"])
        for item in pipeline_to_full512["per_jpeg"]
    ]
    composed: list[dict[str, Any]] = []
    for item in jpeg_to_pipeline["per_jpeg"]:
        pipeline_index = int(item["mapped_source_frame_index_zero_based"])
        if pipeline_index < 0 or pipeline_index >= len(pipeline_to_source):
            raise BuildError(f"frame map composed to an out-of-range pipeline frame: {row_id}")
        full_index = pipeline_to_source[pipeline_index]
        composed.append({
            "jpeg_index_zero_based": int(item["jpeg_index_zero_based"]),
            "mapped_pipeline_avi_frame_index_zero_based": pipeline_index,
            "mapped_full512_source_frame_index_zero_based": full_index,
            "best_candidate_mae_0_255": float(item["best_candidate_mae_0_255"]),
            "second_candidate_mae_0_255": item["second_candidate_mae_0_255"],
            "pipeline_to_full512_mae_0_255": float(pipeline_to_full512["per_jpeg"][pipeline_index]["downsample_mae_0_255"]),
        })
    if q0 + EXPECTED_VISUAL_FRAMES > len(composed):
        raise BuildError(f"measured full-frame map does not cover the complete q0+32 window: {row_id}")

    with Path(str(tracking["tracks_pickle_path"])).open("rb") as handle:
        tracks = pickle.load(handle)
    track_index = int(video["selected_track_index"])
    if not isinstance(tracks, list) or not (0 <= track_index < len(tracks)):
        raise BuildError(f"selected track index is absent from pinned tracks: {row_id}")
    selected = tracks[track_index]
    if not isinstance(selected, dict) or not isinstance(selected.get("proc_track"), dict):
        raise BuildError(f"selected track has no proc_track geometry: {row_id}")
    source_frames = np.asarray(selected.get("track", {}).get("frame"), dtype=np.int64).reshape(-1)
    if source_frames.size != int(video["selected_track_frame_count"]) or not np.array_equal(
        source_frames, np.arange(int(source_frames[0]), int(source_frames[0]) + source_frames.size, dtype=np.int64)
    ):
        raise BuildError(f"selected track source-frame range differs from its renderer receipt: {row_id}")
    source_frames_sha = sha256_bytes(np.ascontiguousarray(source_frames, dtype="<i8").tobytes())
    if source_frames_sha != str(video["selected_track_source_frames_sha256"]).lower():
        raise BuildError(f"selected track source-frame digest differs from renderer receipt: {row_id}")
    track_source_start = int(source_frames[0])
    track_local_start = q0 - track_source_start
    if track_local_start < 0 or track_local_start + EXPECTED_VISUAL_FRAMES > source_frames.size:
        raise BuildError(f"selected face track does not cover the full q0..q0+31 window: {row_id}")
    if int(video["center_query_start_source_clock_frame"]) != q0 or int(video["center_query_start_local"]) != track_local_start:
        raise BuildError(f"renderer q0 source/local track map differs from source audio q0: {row_id}")

    output_frames = list(range(q0, q0 + EXPECTED_VISUAL_FRAMES))
    track_local_frames = list(range(track_local_start, track_local_start + EXPECTED_VISUAL_FRAMES))
    prediction_frames = [
        int(composed[index]["mapped_full512_source_frame_index_zero_based"])
        for index in output_frames
    ]
    map_entries = []
    for offset, output_frame in enumerate(output_frames):
        measured = composed[output_frame]
        map_entries.append({
            "jpeg_index_zero_based": output_frame,
            "target_source_clock_frame": output_frame,
            "mapped_full512_prediction_mel_index": prediction_frames[offset],
            "best_candidate_mae_0_255": measured["best_candidate_mae_0_255"],
            "second_candidate_mae_0_255": measured["second_candidate_mae_0_255"],
            "mapping_ambiguity_delta_mae_0_255": (
                None if measured["second_candidate_mae_0_255"] is None else
                float(measured["second_candidate_mae_0_255"]) - float(measured["best_candidate_mae_0_255"])
            ),
            "mapped_pipeline_avi_frame_index_zero_based": measured["mapped_pipeline_avi_frame_index_zero_based"],
            "pipeline_to_full512_mae_0_255": measured["pipeline_to_full512_mae_0_255"],
        })

    mapping_payload = {
        "schema": "cross_tfg_fit40_full512_pipeline_jpeg_source_map_v1",
        "row_id": row_id,
        "sample_id": str(render_info["sample_id"]),
        "fps_num": EXPECTED_FRAME_RATE_NUM,
        "fps_den": EXPECTED_FRAME_RATE_DEN,
        "map_method": "64x64 BGR MAE dynamic programming; only repeats or one-frame advances; compose pipeline AVI→full512 AVI with pipeline AVI→JPEG",
        "interpretation": "measured frame correspondence only; not a claim of exact frame identity or audio retiming",
        "q0_source_clock_frame": q0,
        "source_files": {
            "full512_avi": full_pin,
            "run_pipeline_video_avi": pipeline_video_pin,
            "tracks_pickle": tracks_pin,
            "pyframes_directory": str(jpeg_dir),
            "pyframes_count": len(checked_paths),
            "pyframes_files": declared_jpegs,
        },
        "decoded_counts": {
            "full512_avi": len(full_frames),
            "run_pipeline_video_avi": len(pipeline_frames),
            "pyframes_jpeg": len(jpeg_frames),
        },
        "jpeg_to_pipeline_avi": jpeg_to_pipeline,
        "pipeline_avi_to_full512_source": pipeline_to_full512,
        "composed_jpeg_to_full512_source": composed,
        "q0_window_entries": map_entries,
    }
    map_path = row_root / "full512_clock_map.json"
    atomic_json(map_path, mapping_payload)
    map_pin = pin_file(map_path)
    after_full = pin_file(full512_path, full_pin["sha256"], full_pin["bytes"])
    after_pipeline = pin_file(intermediate_video, pipeline_video_pin["sha256"], pipeline_video_pin["bytes"])
    after_tracks = pin_file(Path(str(tracking["tracks_pickle_path"])), tracks_pin["sha256"], tracks_pin["bytes"])
    if (after_full, after_pipeline, after_tracks) != (full_pin, pipeline_video_pin, tracks_pin):
        raise BuildError(f"native geometry sources changed during map creation: {row_id}")
    del full_frames, pipeline_frames, jpeg_frames, tracks

    return {
        "row_id": row_id,
        "sample_id": str(render_info["sample_id"]),
        "speaker_id": str(render_info["speaker_id"]),
        "split": "fit",
        "historical_exposed": True,
        "method_id": "rms_fixed",
        "selected_track_index": track_index,
        "selected_track_id": str(video["selected_track_id"]),
        "tracks_pickle": tracks_pin,
        "track_frame_count": int(video["selected_track_frame_count"]),
        "track_source_start_frame": track_source_start,
        "face_box_y1_y2_x1_x2": [int(value) for value in render_info["face_crop"]["official_face_box_y1_y2_x1_x2"]],
        "query_start_frame": q0,
        "output_source_clock_frames": output_frames,
        "track_local_frame_indices": track_local_frames,
        "prediction_source_clock_frames": prediction_frames,
        "prediction_map_evidence": {
            "schema": mapping_payload["schema"],
            "receipt_path": str(map_pin["path"]),
            "receipt_bytes": int(map_pin["bytes"]),
            "receipt_sha256": str(map_pin["sha256"]),
            "row_sample_id": str(render_info["sample_id"]),
            "map_kind": "minimum_mae_monotone_correspondence_64x64_BGR_composed_pipeline_AVI_and_JPEG; measured association, not exact frame identity",
            "syncnet_query_policy": "original-natural audio query remains q0+offset on the unshifted 25fps sample clock; only the Wav2Lip input mel frame follows the measured full512 correspondence",
            "source_composition": "full512_source_to_pipeline_AVI_to_JPEG",
            "entries": map_entries,
        },
        "full512_artifacts": {
            "avi": full_pin,
            "natural_pcm16_mux": {
                "path": str(video["full512_audio_video_mux_path"]),
                "bytes": int(video["full512_audio_video_mux_bytes"]),
                "sha256": str(video["full512_audio_video_mux_sha256"]),
            },
            "selected_track_crop_avi": {
                "path": str(video["selected_track_crop_path"]),
                "bytes": int(video["selected_track_crop_bytes"]),
                "sha256": str(video["selected_track_crop_sha256"]),
            },
            "decoded_frame_count": int(video["decoded_frame_count"]),
            "decoded_fps": float(video["decoded_fps"]),
            "track_source_frame_start": track_source_start,
            "track_source_frame_stop_exclusive": int(video["selected_track_stop_source_frame_exclusive"]),
            "selected_track_source_frames_sha256": source_frames_sha,
            "selected_track_direct_original_pcm16_mux": {
                "path": str(video["mux_path"]),
                "bytes": int(video["mux_bytes"]),
                "sha256": str(video["mux_sha256"]),
            },
            "run_pipeline_reference": reference,
            "run_pipeline_video_avi": pipeline_video_pin,
            "run_pipeline_directory": str(pipeline_dir),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v2-plan", type=Path, required=True)
    parser.add_argument("--parity-helper", type=Path, required=True)
    parser.add_argument("--clock-helper", type=Path, required=True)
    parser.add_argument("--primary-face", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffmpeg-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-free-bytes", type=int, default=MINIMUM_FREE_BYTES_DEFAULT)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo_root))
    manifest_path = args.source_manifest.resolve(strict=True)
    plan_path = args.v2_plan.resolve(strict=True)
    parity_path = args.parity_helper.resolve(strict=True)
    clock_path = args.clock_helper.resolve(strict=True)
    primary_face = args.primary_face.resolve(strict=True)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    out = args.out.resolve()
    if out.exists():
        raise BuildError(f"refusing to reuse output directory: {out}")
    if int(args.minimum_free_bytes) <= 0:
        raise BuildError("minimum-free-bytes must be positive")

    manifest_pin = pin_file(manifest_path, EXPECTED_SOURCE_MANIFEST_SHA256)
    plan_pin = pin_file(plan_path, EXPECTED_V2_PLAN_SHA256)
    parity_pin = pin_file(parity_path, EXPECTED_PARITY_HELPER_SHA256)
    clock_pin = pin_file(clock_path, EXPECTED_CLOCK_HELPER_SHA256)
    face_pin = pin_file(primary_face, EXPECTED_PRIMARY_FACE_SHA256)
    ffmpeg_pin = pin_file(ffmpeg, str(args.ffmpeg_sha256).lower())
    package_root = repo_root / "scripts" / "audio_enhancement"
    package_sources = {
        "wav2lip_fullcontext_fit_runtime_v2_attempt02.py": EXPECTED_V2_RUNTIME_SHA256,
        "wav2lip_fit_runtime.py": EXPECTED_FIT_RUNTIME_SHA256,
        "wav2lip_frontend.py": EXPECTED_FRONTEND_SHA256,
        "baselines.py": EXPECTED_BASELINES_SHA256,
    }
    package_pins = {
        name: pin_file(package_root / name, expected)
        for name, expected in package_sources.items()
    }
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("runtime", {}).get("full_context", {}).get("schema") != "cross_tfg_wav2lip_fullcontext_fit_runtime_v2_attempt02":
        raise BuildError("pinned V2 plan does not declare the expected full-context runtime")
    runtime_cfg = plan["runtime"]
    execution = runtime_cfg["execution"]
    if (
        execution.get("python_executable_literal") != sys.executable
        or execution.get("torch_version") != EXPECTED_TORCH_VERSION
        or execution.get("numpy_version") != EXPECTED_NUMPY_VERSION
        or str(execution.get("python_speech_features_tree_sha256", "")).lower() != EXPECTED_PSF_TREE_SHA256
        or str(runtime_cfg.get("context", {}).get("npz_sha256", "")).lower() != EXPECTED_CONTEXT_NPZ_SHA256
        or str(runtime_cfg.get("context", {}).get("metadata_sha256", "")).lower() != EXPECTED_CONTEXT_JSON_SHA256
        or str(runtime_cfg.get("wav2lip", {}).get("checkpoint_sha256", "")).lower() != EXPECTED_W2L_CHECKPOINT_SHA256
        or str(runtime_cfg.get("syncnet", {}).get("checkpoint_sha256", "")).lower() != EXPECTED_SYNCNET_CHECKPOINT_SHA256
    ):
        raise BuildError("V2 runtime plan pins differ from the frozen Wav2Lip/SyncNet inputs")
    psf_root = str(execution["python_speech_features_root"])
    existing_psf = os.environ.get("CROSS_TFG_PSF_ROOT")
    if existing_psf not in (None, psf_root):
        raise BuildError("CROSS_TFG_PSF_ROOT differs from the literal source-pinned runtime")
    os.environ["CROSS_TFG_PSF_ROOT"] = psf_root
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise BuildError("CUBLAS_WORKSPACE_CONFIG was not set before model imports")

    # Local byte/canonical decode checks run before loading either CUDA model.
    from scripts.audio_enhancement.wav2lip_fit_runtime import read_canonical_wav
    prepared = source_rows_and_audio(
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path=manifest_path,
        source_root=args.source_root,
        read_wav=read_canonical_wav,
    )
    if len(prepared) != 40:
        raise BuildError("fit40 source reader did not yield all 40 rows")
    disk_free = shutil.disk_usage(out.parent).free
    if disk_free < int(args.minimum_free_bytes):
        raise BuildError(f"insufficient free space before output creation: {disk_free}")

    torch = None
    output_rows: list[dict[str, Any]] = []
    row_cache_records: list[dict[str, Any]] = []
    stage = "runtime_import"
    active_row: str | None = None
    out.mkdir(parents=True, exist_ok=False)
    state_path = out / "worker_state.json"
    atomic_json(state_path, {
        "schema": "cross_tfg_fit40_rms_geometry_worker_state_v1",
        "status": "RUNNING_FIT_ONLY_ENGINEERING_RMS_GEOMETRY",
        "completed_rows": 0,
        "expected_rows": 40,
        "started_unix_seconds": time.time(),
        "output_root": str(out),
    })
    try:
        import cv2
        import torch as torch_module
        from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v2 as v2
        from scripts.audio_enhancement.baselines import apply_global_rms
        from scripts.audio_enhancement.wav2lip_fit_runtime import _load_context_seed, _load_models
        from scripts.audio_enhancement.wav2lip_frontend import check_frontend_parity, official_chunk_starts

        torch = torch_module
        if str(torch.__version__) != EXPECTED_TORCH_VERSION or str(np.__version__) != EXPECTED_NUMPY_VERSION:
            raise BuildError(f"runtime version mismatch: torch={torch.__version__} numpy={np.__version__}")
        if not torch.cuda.is_available():
            raise BuildError("fit40 native baseline builder requires the pinned CUDA runtime")
        if not str(args.device).startswith("cuda"):
            raise BuildError("fit40 native baseline device must be CUDA")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        torch.cuda.manual_seed_all(int(args.seed))
        cv2.setNumThreads(1)

        arrays, context_meta, context_pins = _load_context_seed(runtime_cfg)
        if context_pins["npz_sha256"] != EXPECTED_CONTEXT_NPZ_SHA256 or context_pins["metadata_sha256"] != EXPECTED_CONTEXT_JSON_SHA256:
            raise BuildError("static portrait context pins changed")
        if arrays["face_crops_bgr_u8"].shape[0] < 141:
            raise BuildError("pinned fixed portrait seed does not cover the established 141-frame clock")
        if not np.array_equal(
            arrays["face_crops_bgr_u8"],
            np.broadcast_to(arrays["face_crops_bgr_u8"][:1], arrays["face_crops_bgr_u8"].shape),
        ):
            raise BuildError("fixed portrait seed no longer contains a repeated static face crop")
        source_portrait_image = cv2.imread(str(primary_face), cv2.IMREAD_COLOR)
        if source_portrait_image is None or source_portrait_image.dtype != np.uint8 or source_portrait_image.shape != (512, 512, 3):
            raise BuildError("pinned primary source portrait is not 512x512 BGR uint8")

        stage = "model_load"
        device = torch.device(args.device)
        wav2lip_model, syncnet_model, mel_frontend, official_audio, psf_mfcc, runtime_source_pins = _load_models(
            runtime_cfg, device
        )
        if runtime_source_pins.get("wav2lip_checkpoint", {}).get("sha256") != EXPECTED_W2L_CHECKPOINT_SHA256:
            raise BuildError("loaded Wav2Lip checkpoint pin changed")
        if runtime_source_pins.get("syncnet_checkpoint", {}).get("sha256") != EXPECTED_SYNCNET_CHECKPOINT_SHA256:
            raise BuildError("loaded SyncNet checkpoint pin changed")
        psf_record = runtime_source_pins.get("python_speech_features", {})
        if str(psf_record.get("overlay_root", "")) != psf_root:
            raise BuildError("loaded PSF module came from a different overlay")
        if v2._python_source_tree_sha(psf_record.get("source_files", []), Path(psf_record["module_path"]).resolve(strict=True).parent) != EXPECTED_PSF_TREE_SHA256:
            raise BuildError("loaded PSF source tree differs from the pinned tree")

        parity_module = import_source(parity_path, "fit40_native_full512_parity_helper_attempt01")
        clock_module = import_source(clock_path, "fit40_native_clock_map_helper_attempt01")
        inference = parity_module.import_official_inference(
            Path(runtime_cfg["wav2lip"]["root"]).resolve(strict=True),
            Path(runtime_cfg["wav2lip"]["checkpoint"]).resolve(strict=True),
        )
        if str(inference.args.checkpoint_path) != str(runtime_cfg["wav2lip"]["checkpoint"]):
            raise BuildError("official Wav2Lip inference module did not retain the frozen checkpoint argument")
        syncnet_root = Path(runtime_cfg["syncnet"]["root"]).resolve(strict=True)
        pipeline_python = Path(EXPECTED_PIPELINE_PYTHON)
        if not pipeline_python.is_file() or str(pipeline_python) != EXPECTED_PIPELINE_PYTHON:
            raise BuildError("SyncNet tracker Python differs from the literal qualified environment")
        if sha256_file(ffmpeg) != ffmpeg_pin["sha256"]:
            raise BuildError("ffmpeg binary changed after preflight pin")

        cache_root = out / "natural_audio_embedding_cache"
        cache_root.mkdir(parents=True, exist_ok=False)
        stage = "fit40_row_generation"
        worker_started = time.time()
        for index, item in enumerate(prepared):
            row = item["row"]
            active_row = str(row["row_id"])
            free_now = shutil.disk_usage(out).free
            if free_now < int(args.minimum_free_bytes):
                raise BuildError(f"free-space reserve breached before row {active_row}: {free_now}")

            mfcc_array, audio_embeddings, q0, cache_record = v2._load_or_build_audio_cache(
                cache_root=cache_root,
                row=row,
                audio=item["audio"],
                pcm16=item["pcm16"],
                psf_mfcc=psf_mfcc,
                syncnet_model=syncnet_model,
                device=device,
                runtime_source_pins=runtime_source_pins,
                row_cache_records=row_cache_records,
                batch_size=int(runtime_cfg["full_context"]["audio_embedding_batch_size"]),
            )
            if int(q0) != (int(cache_record["audio_embedding_count"]) - EXPECTED_VISUAL_QUERIES) // 2:
                raise BuildError(f"natural audio cache returned a noncanonical center query: {active_row}")
            if q0 + EXPECTED_VISUAL_FRAMES > int(arrays["face_crops_bgr_u8"].shape[0]):
                raise BuildError(f"center-32 query exceeds the pinned fixed portrait seed: {active_row}")

            natural = item["audio"]
            rms_audio, rms_info = apply_global_rms(natural, target_rms=TARGET_RMS, peak_ceiling=None)
            if rms_audio.dtype != np.float32 or rms_audio.shape != natural.shape or not np.isfinite(rms_audio).all():
                raise BuildError(f"fixed RMS waveform is invalid: {active_row}")
            row_root = out / "rows" / str(row["row_id"])
            row_root.mkdir(parents=True, exist_ok=False)
            rms_wav = row_root / "rms_fixed_wav2lip_input_float32.wav"
            from scipy.io import wavfile
            wavfile.write(rms_wav, 16_000, np.ascontiguousarray(rms_audio, dtype="<f4"))
            rms_wav_pin = pin_file(rms_wav)
            reread_rms, reread_info = read_canonical_wav(rms_wav)
            rms_pcm_sha = sha256_bytes(np.ascontiguousarray(rms_audio, dtype="<f4").tobytes())
            if (
                reread_info.frames != row["natural_pcm"]["sample_count"]
                or reread_info.sample_rate != 16_000
                or reread_info.source_channels != 1
                or reread_info.pcm_float32_sha256 != rms_pcm_sha
                or not np.array_equal(reread_rms, rms_audio)
            ):
                raise BuildError(f"float32 RMS input WAV changed on disk round-trip: {active_row}")

            with torch.no_grad():
                official_mel = np.asarray(
                    inference.audio.melspectrogram(np.asarray(rms_audio, dtype=np.float32)),
                    dtype=np.float32,
                )
            if official_mel.ndim != 2 or official_mel.shape[0] != 80 or not np.isfinite(official_mel).all():
                raise BuildError(f"official Wav2Lip RMS mel is invalid: {active_row}")
            all_mel_starts = official_chunk_starts(int(official_mel.shape[1]), fps_num=25, fps_den=1)
            if q0 + EXPECTED_VISUAL_FRAMES > len(all_mel_starts):
                raise BuildError(f"official RMS Wav2Lip mel does not cover center q0+32: {active_row}")
            selected_mel_starts = tuple(all_mel_starts[q0 : q0 + EXPECTED_VISUAL_FRAMES])
            face_context = np.ascontiguousarray(
                arrays["face_crops_bgr_u8"][q0 : q0 + EXPECTED_VISUAL_FRAMES]
            )
            if face_context.shape != (EXPECTED_VISUAL_FRAMES, 3, 96, 96):
                raise BuildError(f"fixed portrait face context has an unexpected shape: {active_row}")
            context = SimpleNamespace(
                face_crops_bgr=face_context,
                query_start_frame=int(q0),
                mel_starts=selected_mel_starts,
                expected_sample_count=int(row["natural_pcm"]["sample_count"]),
            )
            row_for_renderer = {
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "natural_pcm": row["natural_pcm"],
            }
            media_out = row_root / "native_media"
            media_out.mkdir(parents=True, exist_ok=False)
            pipeline_dir = row_root / "official_syncnet_track_pipeline"
            native_wav2lip_calls: list[dict[str, np.ndarray]] = []
            render_info = parity_module.make_full512_native_media(
                torch=torch,
                inference=inference,
                wav2lip_model=wav2lip_model,
                adapter=SimpleNamespace(mel_frontend=mel_frontend),
                context=context,
                waveform=np.ascontiguousarray(rms_audio, dtype=np.float32),
                score_audio_waveform=np.ascontiguousarray(natural, dtype=np.float32),
                row=row_for_renderer,
                method_id="rms_fixed",
                output_dir=media_out,
                pipeline_dir=pipeline_dir,
                ffmpeg=ffmpeg,
                syncnet_root=syncnet_root,
                pipeline_python=pipeline_python,
                primary_face_path=primary_face,
                native_wav2lip_calls=native_wav2lip_calls,
            )
            del native_wav2lip_calls
            if render_info["video"]["decoded_frame_count"] != len(all_mel_starts):
                raise BuildError(f"official full512 frames do not match official W2L mel chunk count: {active_row}")
            if render_info["video"]["decoded_fps"] != 25.0:
                raise BuildError(f"official full512 render changed the frozen 25/1 clock: {active_row}")
            if render_info["pcm16"]["score_audio_is_frozen_natural_pcm"] is not True:
                raise BuildError(f"native RMS baseline was not remuxed against original natural PCM16: {active_row}")

            geometry_record = decode_and_map_row(
                row_root=row_root,
                render_info={**render_info, "sample_id": row["sample_id"], "speaker_id": row["speaker_id"]},
                q0=int(q0),
                clock_module=clock_module,
                row_id=active_row,
            )
            mel_starts_sha = sha256_bytes(np.asarray(all_mel_starts, dtype="<i8").tobytes())
            row_result = {
                **geometry_record,
                "source_input": {
                    "path": row["natural_pcm"]["path"],
                    "file_bytes": int(item["source"]["bytes"]),
                    "file_sha256": str(row["natural_pcm"]["file_sha256"]),
                    "source_pcm16_sha256": str(row["natural_pcm"]["direct_source_pcm16_sha256"]),
                    "canonical_float32_pcm_sha256": str(row["natural_pcm"]["pcm_sha256"]),
                    "sample_count": int(row["natural_pcm"]["sample_count"]),
                    "sample_rate_hz": 16_000,
                    "source_encoding": "PCM_16",
                },
                "full_context_fit": row["full_context_fit"],
                "natural_audio_embedding_cache": cache_record,
                "rms_fixed_baseline": {
                    "method_id": "rms_fixed",
                    "transform_source": "scripts/audio_enhancement/baselines.py:apply_global_rms",
                    "transform_source_sha256": EXPECTED_BASELINES_SHA256,
                    "target_rms": float(rms_info.target_rms),
                    "input_rms": float(rms_info.input_rms),
                    "requested_gain": float(rms_info.requested_gain),
                    "applied_gain": float(rms_info.applied_gain),
                    "output_rms": float(rms_info.output_rms),
                    "peak_limited": bool(rms_info.peak_limited),
                    "peak_ceiling": None,
                    "output_float32_pcm_sha256": rms_pcm_sha,
                    "rms_input_float32_wav": rms_wav_pin,
                    "waveform_sample_count_unchanged": int(rms_audio.size) == int(natural.size),
                    "audio_time_warped": False,
                },
                "native_mel_clock": {
                    "fps_num": 25,
                    "fps_den": 1,
                    "all_official_mel_chunk_count": len(all_mel_starts),
                    "all_official_mel_starts_i64_sha256": mel_starts_sha,
                    "center32_mel_starts": list(selected_mel_starts),
                    "center32_start_clock_equals_audio_q0": True,
                },
                "native_render_diagnostics": {
                    "official_torch_mel_max_abs": float(render_info["mel"]["official_torch_max_abs"]),
                    "official_torch_mel_mean_abs": float(render_info["mel"]["official_torch_mean_abs"]),
                    "face96_matches_fixed_proxy_context_exact": bool(render_info["face_crop"]["face96_matches_frozen_fit_context_exact"]),
                    "run_pipeline_exit_code": int(render_info["video"]["run_pipeline_exit_code"]),
                    "eligible_full_lag_track_count": int(render_info["video"]["eligible_full_lag_track_count"]),
                    "track_count": int(render_info["tracking"]["track_count"]),
                },
            }
            output_rows.append(row_result)
            stage = "row_checkpoint"
            atomic_json(out / "progress.json", {
                "schema": "cross_tfg_fit40_rms_geometry_progress_v1",
                "status": "RUNNING_FIT_ONLY_ENGINEERING_RMS_GEOMETRY",
                "completed_rows": len(output_rows),
                "expected_rows": 40,
                "last_completed_row_id": active_row,
                "elapsed_seconds": time.time() - worker_started,
                "row_ids": [value["row_id"] for value in output_rows],
            })
            atomic_json(state_path, {
                "schema": "cross_tfg_fit40_rms_geometry_worker_state_v1",
                "status": "RUNNING_FIT_ONLY_ENGINEERING_RMS_GEOMETRY",
                "completed_rows": len(output_rows),
                "expected_rows": 40,
                "last_completed_row_id": active_row,
                "updated_unix_seconds": time.time(),
                "output_root": str(out),
            })
            stage = "fit40_row_generation"

        if len(output_rows) != 40 or {row["row_id"] for row in output_rows} != {item["row"]["row_id"] for item in prepared}:
            raise BuildError("native fit40 output does not match the exact 40-row source roster")
        cache_manifest_path = cache_root / "cache_manifest.json"
        cache_manifest_pin = pin_file(cache_manifest_path)
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if len(cache_manifest.get("rows", [])) != 40:
            raise BuildError("natural audio cache manifest does not contain exactly 40 rows")

        source_pins = {
            "source_manifest": manifest_pin,
            "v2_runtime_plan": plan_pin,
            "parity_helper": parity_pin,
            "clock_map_helper": clock_pin,
            "primary_source_portrait": face_pin,
            "ffmpeg": ffmpeg_pin,
            "fit_runtime_sources": package_pins,
            "runtime_models_and_frontends": runtime_source_pins,
            "fixed_context_seed": context_pins,
        }
        manifest_payload: dict[str, Any] = {
            "schema": "cross_tfg_native_full512_fit_geometry_manifest_v1",
            "producer_schema": "cross_tfg_fit40_natural_cache_rms_geometry_attempt01",
            "created_by": Path(__file__).name,
            "status": "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY",
            "source_clock_fps_num": 25,
            "source_clock_fps_den": 1,
            "audio_clock_policy": "unchanged_original_natural_pcm_mfcc_queries",
            "scope": "40 historical fit-only rows across S0052-S0071 at canonical j=0/j=15; original-natural SyncNet audio cache plus rms_fixed official full512 geometry; no scores, no training, no dev/confirmation access",
            "method_id": "rms_fixed",
            "fit_rows": 40,
            "fit_speakers": [f"S{number:04d}" for number in range(52, 72)],
            "source_pins": source_pins,
            "source_portrait": face_pin,
            "audio_cache": {
                "schema": v2.CACHE_SCHEMA,
                "path": str(cache_manifest_pin["path"]),
                "bytes": int(cache_manifest_pin["bytes"]),
                "sha256": str(cache_manifest_pin["sha256"]),
                "row_count": 40,
                "mfcc_source": "pinned python_speech_features.mfcc defaults over complete original direct-PCM16 payloads",
                "embedding_source": "pinned SyncNet forward_aud over the first floor(N/640)-5 valid 20-frame MFCC windows at hop 4",
                "query_rule": "q0=floor((N_audio_embeddings-27)/2), 27 unshifted source-clock queries and every lag -15..+15 is in bounds; no padding",
            },
            "baseline_control": {
                "source": "scripts/audio_enhancement/baselines.py:apply_global_rms",
                "source_sha256": EXPECTED_BASELINES_SHA256,
                "target_rms": TARGET_RMS,
                "peak_ceiling": None,
                "peak_limited": False,
                "role": "native input/geometry control only; no fit effect estimate or method selection",
            },
            "rows": output_rows,
        }
        manifest_payload["payload_sha256"] = canonical_sha(manifest_payload)
        geometry_path = out / "fit40_native_rms_geometry_manifest.json"
        atomic_json(geometry_path, manifest_payload)
        geometry_pin = pin_file(geometry_path)
        elapsed = time.time() - worker_started
        final_state = {
            "schema": "cross_tfg_fit40_rms_geometry_worker_state_v1",
            "status": "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY",
            "completed_rows": 40,
            "expected_rows": 40,
            "elapsed_seconds": elapsed,
            "geometry_manifest": geometry_pin,
            "audio_cache_manifest": cache_manifest_pin,
            "output_root": str(out),
            "model_score_computed": False,
            "training_performed": False,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
        }
        atomic_json(state_path, final_state)
        print(json.dumps(final_state, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        failure = {
            "schema": "cross_tfg_fit40_rms_geometry_worker_failure_v1",
            "status": "FAILED_FIT_ONLY_ENGINEERING_PREPARATION",
            "stage": stage,
            "active_row_id": active_row,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc()[-8000:],
            "completed_rows": len(output_rows),
            "expected_rows": 40,
            "output_root": str(out),
            "model_score_computed": False,
            "training_performed": False,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
        }
        atomic_json(out / "failure.json", failure)
        atomic_json(state_path, {
            "schema": "cross_tfg_fit40_rms_geometry_worker_state_v1",
            "status": "FAILED_FIT_ONLY_ENGINEERING_PREPARATION",
            "completed_rows": len(output_rows),
            "expected_rows": 40,
            "active_row_id": active_row,
            "failure_path": str(out / "failure.json"),
            "failure_sha256": sha256_file(out / "failure.json"),
        })
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
