"""Engineering-only four-cell Wav2Lip/SyncNet smoke for 2026-09-30.

This pilot validates the portable frontend, direct latent decode, two-face
rendering, float PCM amplitude units, and four-cell artifact production. Its
scores are diagnostics only and are never a scientific result.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_AUX_SITE_PACKAGES = os.environ.get("TTS_SOURCE_AXIS_AUX_SITE_PACKAGES", "")
if _AUX_SITE_PACKAGES and Path(_AUX_SITE_PACKAGES).is_dir() and _AUX_SITE_PACKAGES not in sys.path:
    # Reuse an existing sibling package directory only for missing imports;
    # the active interpreter's own packages keep precedence.
    sys.path.append(_AUX_SITE_PACKAGES)

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import ops, scoring, wav2lip

_SHARED_SYNCNET = None
_SHARED_SYNCNET_META: dict[str, Any] = {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return scoring.sha256_file(path)


def _git_revision(path: Path) -> str:
    completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _read_input(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_rate, pcm = wavfile.read(str(path))
    if sample_rate != ops.SAMPLE_RATE or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"pilot input must be mono PCM16 at 16 kHz: {path} ({sample_rate}, {pcm.shape}, {pcm.dtype})")
    float_pcm = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
    sync_units, sync_meta = ops.pcm_to_syncnet_integer_units(float_pcm)
    identity_expected = pcm.astype(np.float64)
    identity = bool(np.array_equal(sync_units, identity_expected))
    return pcm, float_pcm, {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "sample_rate_hz": int(sample_rate),
        "channels": 1,
        "samples": int(pcm.size),
        "duration_s": float(pcm.size / sample_rate),
        "pcm_dtype": str(pcm.dtype),
        "peak_abs_pcm16": int(np.max(np.abs(pcm.astype(np.int32)))),
        "syncnet_pcm_float_identity_exact": identity,
        "syncnet_identity_max_abs_error": float(np.max(np.abs(sync_units - identity_expected))),
        "syncnet_units": sync_meta,
    }


def _encode_audio(model: Any, waveform: np.ndarray, wav2lip_root: Path, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    mel, chunks, starts, mel_meta = wav2lip.mel_chunks(waveform, wav2lip_root)
    latent = wav2lip.encode_latents(model, chunks, device=device, batch_size=batch_size)
    mel_meta.update({
        "latent_shape": list(latent.shape),
        "latent_dtype": str(latent.dtype),
        "latent_sha256": ops.array_sha256(latent),
    })
    return latent, chunks, starts, {"mel": mel, "metadata": mel_meta}


def _score_supports(matrix: np.ndarray, anchor: int) -> dict[str, Any]:
    windows = len(matrix)
    supports = {
        "full": np.arange(windows, dtype=np.int64),
        "guard15": np.arange(15, max(15, windows - 15), dtype=np.int64),
        "guard20": np.arange(20, max(20, windows - 20), dtype=np.int64),
    }
    result = {}
    for name, indices in supports.items():
        if len(indices) < 1:
            result[name] = {"support_n": 0, "status": "too_short"}
        else:
            result[name] = scoring.curve_metrics(matrix[indices], anchor_column=anchor)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    wav2lip_root = Path(args.wav2lip_root).resolve()
    syncnet_root = Path(args.syncnet_root).resolve()
    wav2lip_checkpoint = Path(args.wav2lip_checkpoint).resolve()
    syncnet_checkpoint = Path(args.syncnet_checkpoint).resolve()
    face_inputs = {
        "primary": Path(args.primary_face).resolve(),
        "secondary": Path(args.secondary_face).resolve(),
    }
    examples = [
        ("pilot_local_1", Path(args.natural_1).resolve(), Path(args.tts_1).resolve()),
        ("pilot_local_2", Path(args.natural_2).resolve(), Path(args.tts_2).resolve()),
    ]

    torch.manual_seed(20260930)
    np.random.seed(20260930)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    method: dict[str, Any] = {
        "classification": "engineering smoke only; excluded from calibration and holdout",
        "seed": 20260930,
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "python": platform.python_version(),
        "aux_site_packages_appended": _AUX_SITE_PACKAGES or None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "wav2lip_repo_revision": _git_revision(wav2lip_root),
        "syncnet_repo_revision": _git_revision(syncnet_root),
        "ffmpeg": str(Path(args.ffmpeg).resolve()),
        "wav2lip_frontend_compatibility": wav2lip.install_librosa_compatibility_shim(),
    }
    model, model_meta = wav2lip.load_model(wav2lip_root, wav2lip_checkpoint, device=device)
    shared_net = _SHARED_SYNCNET
    syncnet_meta = dict(_SHARED_SYNCNET_META)
    if shared_net is None:
        shared_net, syncnet_meta = scoring.load_syncnet(syncnet_root, syncnet_checkpoint, device=device)
    method["wav2lip_model"] = model_meta
    method["syncnet_model"] = syncnet_meta

    face_frames: dict[str, np.ndarray] = {}
    face_meta: dict[str, Any] = {}
    for face_id, path in face_inputs.items():
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cannot read face image {path}")
        detected = wav2lip.detect_static_face(frame, wav2lip_root, device=device)
        detected.update({"path": str(path), "sha256": _sha(path), "width": int(frame.shape[1]), "height": int(frame.shape[0])})
        face_frames[face_id] = frame
        face_meta[face_id] = detected

    artifacts: dict[str, Any] = {}
    scores: list[dict[str, Any]] = []
    parity_rows = []
    total_started = time.perf_counter()
    for clip_id, natural_path, tts_path in examples:
        clip_started = time.perf_counter()
        natural_pcm, natural_float, natural_meta = _read_input(natural_path)
        tts_pcm, tts_float, tts_meta = _read_input(tts_path)
        waveforms: dict[str, np.ndarray] = {}
        levels: dict[str, dict[str, Any]] = {}
        z_by_source_level: dict[str, np.ndarray] = {}
        chunks_by_source_level: dict[str, np.ndarray] = {}
        pcm_by_source_level: dict[str, np.ndarray] = {}
        aud_by_source_level: dict[str, np.ndarray] = {}
        for source_id, pcm, waveform, source_meta in (
            ("natural", natural_pcm, natural_float, natural_meta),
            ("tts", tts_pcm, tts_float, tts_meta),
        ):
            fixed, fixed_meta = ops.normalize_rms(waveform)
            levels[source_id] = {"raw_input": source_meta, "fixed_rms": fixed_meta}
            waveforms[f"{source_id}_raw"] = waveform
            waveforms[f"{source_id}_fixed"] = fixed
            pcm_by_source_level[f"{source_id}_raw"] = pcm
            pcm_by_source_level[f"{source_id}_fixed"] = fixed
            raw_z, raw_chunks, raw_starts, raw_front = _encode_audio(model, waveform, wav2lip_root, device, args.wav2lip_batch_size)
            fixed_z, fixed_chunks, fixed_starts, fixed_front = _encode_audio(model, fixed, wav2lip_root, device, args.wav2lip_batch_size)
            if raw_starts != fixed_starts:
                raise AssertionError("RMS normalization changed mel timing support")
            z_by_source_level[f"{source_id}_raw"] = raw_z
            z_by_source_level[f"{source_id}_fixed"] = fixed_z
            chunks_by_source_level[f"{source_id}_raw"] = raw_chunks
            chunks_by_source_level[f"{source_id}_fixed"] = fixed_chunks
            raw_units = pcm.astype(np.float64)
            fixed_units, fixed_units_meta = ops.pcm_to_syncnet_integer_units(fixed)
            raw_units_from_float, _ = ops.pcm_to_syncnet_integer_units(waveform)
            if not np.array_equal(raw_units, raw_units_from_float):
                raise AssertionError("normalized PCM16 did not round-trip exactly to SyncNet integer units")
            raw_pcm = pcm
            aud_by_source_level[f"{source_id}_raw"] = raw_units
            aud_by_source_level[f"{source_id}_fixed"] = fixed_units
            levels[source_id]["frontend"] = {
                "raw": raw_front["metadata"],
                "fixed": fixed_front["metadata"],
                "raw_syncnet_units_sha256": ops.array_sha256(raw_units),
                "raw_syncnet_float_identity_exact": True,
                "fixed_syncnet_units": fixed_units_meta,
                "raw_pcm16_exact": bool(np.array_equal(raw_pcm, pcm)),
            }
            mask, mask_meta = ops.fixed_speech_mask(fixed, fixed_starts)
            levels[source_id]["fixed_speech_mask"] = mask_meta
            np.savez_compressed(
                run_dir / f"{clip_id}_{source_id}_audio.npz",
                pcm16=pcm,
                raw_float=waveform,
                fixed_float=fixed,
                mel_raw=raw_front["mel"],
                mel_fixed=fixed_front["mel"],
                mel_starts=np.asarray(raw_starts, dtype=np.int32),
                z_raw=raw_z,
                z_fixed=fixed_z,
                speech_mask_fixed=mask,
                syncnet_units_fixed=fixed_units,
            )

        face_records: dict[str, Any] = {}
        for face_id, frame in face_frames.items():
            generation_box = face_meta[face_id]["generation_box_xyxy"]
            count_set = {4, args.wav2lip_batch_size}
            for z in z_by_source_level.values():
                if len(z) % args.wav2lip_batch_size:
                    count_set.add(len(z) % args.wav2lip_batch_size)
            feature_cache = wav2lip.cache_face_features(model, frame, generation_box, sorted(count_set), device=device)
            parity = wav2lip.parity_check_batches(
                model, chunks_by_source_level["natural_raw"], frame, generation_box,
                feature_cache, batch_size=args.wav2lip_batch_size, device=device,
            )
            parity.update({"clip_id": clip_id, "face_id": face_id, "source_level": "natural_raw"})
            parity_rows.append(parity)
            if not parity["uint8_prediction_exact"]:
                raise AssertionError(f"direct latent no-op path changed uint8 output: {parity}")

            visual_paths: dict[str, Path] = {}
            lip_features: dict[str, np.ndarray] = {}
            visual_metadata: dict[str, Any] = {}
            for level in ("raw", "fixed"):
                z = z_by_source_level[f"natural_{level}"]
                video_path = run_dir / f"{clip_id}_{face_id}_{level}.mp4"
                video_meta = wav2lip.render_video(
                    model, z, frame, generation_box, feature_cache, video_path, args.ffmpeg,
                    batch_size=args.wav2lip_batch_size, device=device,
                )
                crop_path = run_dir / f"{clip_id}_{face_id}_{level}_score.mkv"
                crop_meta = wav2lip.crop_score_video(video_path, face_meta[face_id]["score_box"], crop_path, args.ffmpeg)
                jpeg_dir = run_dir / "jpeg" / clip_id / face_id / level
                frames, jpeg_meta = scoring.extract_official_jpegs(crop_path, jpeg_dir, args.ffmpeg)
                visual_paths[level] = crop_path
                visual_metadata[level] = {"video": video_meta, "score_crop": crop_meta, "jpeg": jpeg_meta}
                lip_features[level] = frames  # decoded JPEG frames, converted to embeddings immediately below

            if shared_net is None:
                raise RuntimeError("SyncNet feature model is unavailable")
            for level in ("raw", "fixed"):
                frames = lip_features[level]
                lip_features[level] = scoring.lip_embeddings_from_frames(shared_net, frames, device=device, batch_size=20)
            max_windows = min(len(lip_features["raw"]), len(lip_features["fixed"]))
            audio_features: dict[str, np.ndarray] = {}
            audio_meta: dict[str, Any] = {}
            for level in ("raw", "fixed"):
                units = aud_by_source_level[f"natural_{level}"]
                count = min(max_windows, len(units) // 640 - 5)
                audio_features[level], audio_meta[level] = scoring.audio_embeddings_from_syncnet_units(
                    shared_net, units, window_count=count, device=device, batch_size=20,
                )
            common_support = min(max_windows, len(audio_features["raw"]), len(audio_features["fixed"]))
            matrices = {
                "q00": scoring.pairwise_distance_matrix(lip_features["raw"][:common_support], audio_features["raw"][:common_support], device=device),
                "q10": scoring.pairwise_distance_matrix(lip_features["fixed"][:common_support], audio_features["raw"][:common_support], device=device),
                "q01": scoring.pairwise_distance_matrix(lip_features["raw"][:common_support], audio_features["fixed"][:common_support], device=device),
                "q11": scoring.pairwise_distance_matrix(lip_features["fixed"][:common_support], audio_features["fixed"][:common_support], device=device),
            }
            anchor = int(np.argmin(matrices["q00"].mean(axis=0, dtype=np.float32)))
            for cell, matrix in matrices.items():
                scores.append({
                    "clip_id": clip_id,
                    "speaker_id": "clip_scoped_pilot_label",
                    "face_id": face_id,
                    "cell": cell,
                    "visual_level": "raw" if cell in ("q00", "q01") else "fixed",
                    "audio_level": "raw" if cell in ("q00", "q10") else "fixed",
                    "shared_support_n": int(common_support),
                    "anchor_column_from_q00": anchor,
                    "supports": _score_supports(matrix, anchor),
                    "distance_matrix_sha256": ops.array_sha256(matrix),
                })
            np.savez_compressed(
                run_dir / f"{clip_id}_{face_id}_syncnet_features.npz",
                lip_raw=lip_features["raw"], lip_fixed=lip_features["fixed"],
                audio_raw=audio_features["raw"], audio_fixed=audio_features["fixed"],
                q00=matrices["q00"], q10=matrices["q10"], q01=matrices["q01"], q11=matrices["q11"],
            )
            if clip_id == "pilot_local_1" and face_id == "primary":
                parity_mux = run_dir / f"{clip_id}_{face_id}_q00_official_parity.mkv"
                mux_command = [
                    str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(visual_paths["raw"]), "-i", str(natural_path),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-shortest", str(parity_mux),
                ]
                muxed = subprocess.run(mux_command, capture_output=True, text=True, check=False)
                if muxed.returncode:
                    raise RuntimeError(f"official parity PCM mux failed: {muxed.stderr[-2000:]}")
                official = scoring.official_evaluate_capture(
                    shared_net,
                    SimpleNamespace(
                        tmp_dir=str(run_dir / "official_parity_tmp"),
                        reference="pilot_local_1_primary_q00",
                        batch_size=20,
                        vshift=15,
                    ),
                    parity_mux,
                )
                n_lip = min(len(lip_features["raw"]), len(official["lip"]))
                n_audio = min(len(audio_features["raw"]), len(official["audio"]))
                n_dist = min(len(matrices["q00"]), len(official["dists"]))
                parity_rows.append({
                    "type": "official_syncnet_raw_pcm16_frontend_parity",
                    "clip_id": clip_id,
                    "face_id": face_id,
                    "mux_path": str(parity_mux),
                    "mux_sha256": _sha(parity_mux),
                    "official_offset_frames": official["offset"],
                    "official_confidence": official["confidence"],
                    "lip_windows_compared": n_lip,
                    "audio_windows_compared": n_audio,
                    "distance_windows_compared": n_dist,
                    "lip_max_abs_difference": float(np.max(np.abs(lip_features["raw"][:n_lip] - official["lip"][:n_lip]))),
                    "audio_max_abs_difference": float(np.max(np.abs(audio_features["raw"][:n_audio] - official["audio"][:n_audio]))),
                    "distance_max_abs_difference": float(np.max(np.abs(matrices["q00"][:n_dist] - official["dists"][:n_dist]))),
                    "lip_close_5e4": bool(np.max(np.abs(lip_features["raw"][:n_lip] - official["lip"][:n_lip])) < 5e-4),
                    "audio_close_5e4": bool(np.max(np.abs(audio_features["raw"][:n_audio] - official["audio"][:n_audio])) < 5e-4),
                    "distance_close_5e4": bool(np.max(np.abs(matrices["q00"][:n_dist] - official["dists"][:n_dist])) < 5e-4),
                    "official_extracted_pcm_sha256": _sha(Path(args.run_dir) / "official_parity_tmp" / "pilot_local_1_primary_q00" / "audio.wav"),
                    "official_mux_command": mux_command,
                })
            face_records[face_id] = {
                "face_id": face_id,
                "score_box": face_meta[face_id]["score_box"],
                "visual": visual_metadata,
                "lip_raw_sha256": ops.array_sha256(lip_features["raw"]),
                "lip_fixed_sha256": ops.array_sha256(lip_features["fixed"]),
                "audio_raw": audio_meta["raw"],
                "audio_fixed": audio_meta["fixed"],
                "audio_raw_sha256": ops.array_sha256(audio_features["raw"]),
                "audio_fixed_sha256": ops.array_sha256(audio_features["fixed"]),
                "common_support_n": int(common_support),
            }
        artifacts[clip_id] = {
            "natural": natural_meta,
            "tts": tts_meta,
            "levels": levels,
            "faces": face_records,
            "elapsed_seconds": float(time.perf_counter() - clip_started),
        }

    return {
        "status": "completed",
        "engineering_smoke_only": True,
        "scientific_cohort": False,
        "run_dir": str(run_dir),
        "method": method,
        "faces": face_meta,
        "clips": artifacts,
        "scores": scores,
        "checks": {
            "all_pcm16_float_identity_exact": all(row["syncnet_pcm_float_identity_exact"] for clip in artifacts.values() for row in (clip["natural"], clip["tts"])),
            "all_direct_decoder_uint8_noop_exact": all(row["uint8_prediction_exact"] for row in parity_rows if "uint8_prediction_exact" in row),
            "no_effect_summary_emitted": True,
            "direct_decoder_parity": parity_rows,
        },
        "timings": {"total_seconds": float(time.perf_counter() - total_started)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--natural-1", required=True)
    parser.add_argument("--natural-2", required=True)
    parser.add_argument("--tts-1", required=True)
    parser.add_argument("--tts-2", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--secondary-face", required=True)
    parser.add_argument("--wav2lip-batch-size", type=int, default=32)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    # The network is initialized once and kept across both clip rows/faces.
    global _SHARED_SYNCNET, _SHARED_SYNCNET_META
    _SHARED_SYNCNET, _SHARED_SYNCNET_META = scoring.load_syncnet(args.syncnet_root, args.syncnet_checkpoint, device="cuda")
    summary = run(args)
    _write_json(Path(args.summary), summary)
    print(json.dumps({"status": summary["status"], "summary": args.summary, "seconds": summary["timings"]["total_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
