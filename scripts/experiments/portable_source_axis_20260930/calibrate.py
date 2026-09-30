"""Build sealed calibration latents/axes and the global q10 SyncNet anchor."""

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
from typing import Any

_AUX_SITE_PACKAGES = os.environ.get("TTS_SOURCE_AXIS_AUX_SITE_PACKAGES", "")
if _AUX_SITE_PACKAGES and Path(_AUX_SITE_PACKAGES).is_dir() and _AUX_SITE_PACKAGES not in sys.path:
    sys.path.append(_AUX_SITE_PACKAGES)

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import ops, scoring, wav2lip


def _sha(path: str | Path) -> str:
    return scoring.sha256_file(path)


def _git_revision(path: str | Path) -> str:
    completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _package_code_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {path.name: _sha(path) for path in sorted(package_dir.glob("*.py"))}


def _read_pcm16(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_rate, pcm = wavfile.read(str(path))
    if sample_rate != 16_000 or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"expected mono PCM16 16 kHz WAV: {path} ({sample_rate},{pcm.shape},{pcm.dtype})")
    waveform = (pcm.astype(np.float64) / 32768.0).astype(np.float32)
    return pcm, waveform, {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "sample_rate_hz": int(sample_rate),
        "channels": 1,
        "samples": int(len(pcm)),
        "duration_s": float(len(pcm) / sample_rate),
        "pcm_dtype": str(pcm.dtype),
        "peak_abs_pcm16": int(np.max(np.abs(pcm.astype(np.int32)))),
    }


def _encode(model, waveform: np.ndarray, w2l_root: Path, device: str, batch_size: int):
    mel, chunks, starts, mel_meta = wav2lip.mel_chunks(waveform, w2l_root)
    z = wav2lip.encode_latents(model, chunks, device=device, batch_size=batch_size)
    return z, chunks, starts, mel, {
        **mel_meta,
        "latent_shape": list(z.shape),
        "latent_dtype": str(z.dtype),
        "latent_sha256": ops.array_sha256(z),
    }


def _nearest_anchor_k(values: list[float], historical_k: int = 3) -> tuple[int, dict[str, Any]]:
    if not values or not np.isfinite(values).all():
        raise ValueError("calibration anchor lag values are empty or nonfinite")
    center = float(np.median(np.asarray(values, dtype=np.float64)))
    lower, upper = int(np.floor(center)), int(np.ceil(center))
    candidates = sorted({lower, upper})
    selected = min(candidates, key=lambda value: (abs(value - center), abs(value - historical_k), value))
    return selected, {
        "values": [float(value) for value in values],
        "median_before_rounding": center,
        "nearest_integer_candidates": candidates,
        "tie_rule": "nearest median; ties prefer k closest to historical k=3, then smaller k",
        "historical_k": int(historical_k),
        "selected_k": int(selected),
        "anchor_column": int(15 + selected),
        "official_syncnet_offset": int(-selected),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "calibration_features").mkdir(parents=True, exist_ok=True)
    input_manifest_path = Path(args.input_manifest).resolve()
    tts_manifest_path = Path(args.tts_pair_manifest).resolve()
    frozen_input_manifest_path = Path(args.frozen_input_manifest).resolve()
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    tts_manifest = json.loads(tts_manifest_path.read_text(encoding="utf-8"))
    frozen_input_manifest = json.loads(frozen_input_manifest_path.read_text(encoding="utf-8"))
    code_hashes = _package_code_hashes()
    if tts_manifest["source_manifest"]["raw_sha256"] != _sha(input_manifest_path):
        raise ValueError("selected input manifest raw hash differs from the frozen TTS pair manifest")
    if tts_manifest["run_frozen_input_manifest"]["raw_sha256"] != _sha(frozen_input_manifest_path):
        raise ValueError("run-frozen source manifest raw hash differs from the TTS pair manifest")
    if frozen_input_manifest.get("manifest_sha256") != tts_manifest["run_frozen_input_manifest"]["embedded_fingerprint"]:
        raise ValueError("run-frozen source embedded fingerprint differs from the TTS pair manifest")
    identity_fields = tts_manifest["input_identity_fields"]
    source_identity = [{field: row.get(field) for field in identity_fields} for row in input_manifest["records"]]
    tts_identity = [row["input_identity"] for row in tts_manifest["records"]]
    identity_bytes = json.dumps(source_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
    if source_identity != tts_identity or identity_sha256 != tts_manifest["input_identity_sha256"]:
        raise ValueError("ordered source/TTS identity array differs from its frozen digest")
    cal_rows = [row for row in input_manifest["records"] if row["split"] == "calibration"]
    tts_rows = {row["id"]: row for row in tts_manifest["records"] if row["split"] == "calibration"}
    if len(cal_rows) != 20 or len(tts_rows) != 20:
        raise ValueError(f"calibration cohort must be exactly 20 rows: source={len(cal_rows)}, TTS={len(tts_rows)}")
    source_fields = ("id", "speaker", "text", "split", "ref_sha256")
    for row in cal_rows:
        tts_row = tts_rows.get(row["id"])
        if tts_row is None or any(row[field] != tts_row["input_identity"][field] for field in source_fields):
            raise ValueError(f"calibration source/TTS identity mismatch: {row['id']}")

    w2l_root = Path(args.wav2lip_root).resolve()
    sync_root = Path(args.syncnet_root).resolve()
    w2l_checkpoint = Path(args.wav2lip_checkpoint).resolve()
    sync_checkpoint = Path(args.syncnet_checkpoint).resolve()
    torch.manual_seed(20260930)
    np.random.seed(20260930)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda"

    frontend = wav2lip.install_librosa_compatibility_shim()
    model, model_meta = wav2lip.load_model(w2l_root, w2l_checkpoint, device=device)
    sync_net, sync_meta = scoring.load_syncnet(sync_root, sync_checkpoint, device=device)
    face_path = Path(args.primary_face).resolve()
    face_frame = cv2.imread(str(face_path))
    if face_frame is None:
        raise RuntimeError(f"cannot read primary face: {face_path}")
    face_meta = wav2lip.detect_static_face(face_frame, w2l_root, device=device)
    face_meta.update({"path": str(face_path), "sha256": _sha(face_path), "width": int(face_frame.shape[1]), "height": int(face_frame.shape[0])})
    face_feature_cache = wav2lip.cache_face_features(
        model, face_frame, face_meta["generation_box_xyxy"],
        range(1, args.wav2lip_batch_size + 1), device=device,
    )

    axis_rows: list[dict[str, Any]] = []
    per_clip: dict[str, Any] = {}
    natural_anchor_rows: list[dict[str, Any]] = []
    for row in cal_rows:
        sample_id = row["id"]
        tts_row = tts_rows[sample_id]
        speaker = row["speaker"]
        natural_path = Path(args.input_root) / "audio" / speaker / f"{sample_id.split('_', 1)[1]}.wav"
        tts_path = Path(args.input_root) / "tts_outputs" / "audio_16k_canonical" / f"{sample_id}.wav"
        natural_pcm, natural_float, natural_meta = _read_pcm16(natural_path)
        tts_pcm, tts_float, tts_meta = _read_pcm16(tts_path)
        if natural_meta["sha256"] != row["ref_sha256"]:
            raise ValueError(f"natural PCM hash mismatch for {sample_id}")
        if tts_meta["sha256"] != tts_row["qwen_canonical_16k"]["sha256"]:
            raise ValueError(f"canonical TTS PCM hash mismatch for {sample_id}")

        fixed_n, norm_n = ops.normalize_rms(natural_float)
        fixed_t, norm_t = ops.normalize_rms(tts_float)
        n_raw_z, n_raw_chunks, n_starts, n_raw_mel, n_raw_meta = _encode(model, natural_float, w2l_root, device, args.wav2lip_batch_size)
        n_fixed_z, n_fixed_chunks, n_fixed_starts, n_fixed_mel, n_fixed_meta = _encode(model, fixed_n, w2l_root, device, args.wav2lip_batch_size)
        t_raw_z, t_raw_chunks, t_starts, t_raw_mel, t_raw_meta = _encode(model, tts_float, w2l_root, device, args.wav2lip_batch_size)
        t_fixed_z, t_fixed_chunks, t_fixed_starts, t_fixed_mel, t_fixed_meta = _encode(model, fixed_t, w2l_root, device, args.wav2lip_batch_size)
        if n_starts != n_fixed_starts or t_starts != t_fixed_starts:
            raise AssertionError("RMS normalization changed Wav2Lip mel timing support")

        if row is cal_rows[0]:
            parity = wav2lip.parity_check_batches(
                model, n_raw_chunks, face_frame, face_meta["generation_box_xyxy"],
                face_feature_cache, batch_size=args.wav2lip_batch_size, device=device,
            )
            if not parity["uint8_prediction_exact"]:
                raise AssertionError(f"shared face-feature cache changed Wav2Lip output: {parity}")
        else:
            parity = None

        plus_pcm, plus_gain_meta = ops.gain_fixed_waveform(fixed_n, 3.0)
        minus_pcm, minus_gain_meta = ops.gain_fixed_waveform(fixed_n, -3.0)
        n_plus_z, n_plus_chunks, n_plus_starts, n_plus_mel, n_plus_meta = _encode(model, plus_pcm, w2l_root, device, args.wav2lip_batch_size)
        n_minus_z, n_minus_chunks, n_minus_starts, n_minus_mel, n_minus_meta = _encode(model, minus_pcm, w2l_root, device, args.wav2lip_batch_size)
        if n_plus_starts != n_fixed_starts or n_minus_starts != n_fixed_starts:
            raise AssertionError("±3 dB level perturbation changed mel timing support")
        mask_n, mask_n_meta = ops.fixed_speech_mask(fixed_n, n_fixed_starts)
        mask_t, mask_t_meta = ops.fixed_speech_mask(fixed_t, t_fixed_starts)

        score_video = run_dir / "calibration_videos" / f"{sample_id}_primary_q10.mkv"
        video_meta = wav2lip.render_video(
            model, n_fixed_z, face_frame, face_meta["generation_box_xyxy"], face_feature_cache,
            score_video, args.ffmpeg, batch_size=args.wav2lip_batch_size, device=device,
        )
        score_crop = run_dir / "calibration_score_crops" / f"{sample_id}_primary_q10.mkv"
        crop_meta = wav2lip.crop_score_video(score_video, face_meta["score_box"], score_crop, args.ffmpeg)
        jpeg_dir = run_dir / "calibration_jpeg" / sample_id
        jpeg_frames, jpeg_meta = scoring.extract_official_jpegs(score_crop, jpeg_dir, args.ffmpeg)
        lip = scoring.lip_embeddings_from_frames(sync_net, jpeg_frames, device=device, batch_size=args.syncnet_batch_size)
        raw_units = natural_pcm.astype(np.float64)
        if not np.array_equal(raw_units, ops.pcm_to_syncnet_integer_units(natural_float)[0]):
            raise AssertionError(f"float PCM did not preserve original PCM16 integer units for {sample_id}")
        audio_window_count = min(len(lip), len(raw_units) // 640 - 5)
        audio, audio_meta = scoring.audio_embeddings_from_syncnet_units(
            sync_net, raw_units, window_count=audio_window_count, device=device, batch_size=args.syncnet_batch_size,
        )
        support = min(len(lip), len(audio))
        dists = scoring.pairwise_distance_matrix(lip[:support], audio[:support], device=device)
        curve = scoring.curve_metrics(dists)
        anchor_column_best = int(curve["optimal_column"])
        clip_k = int(anchor_column_best - 15)
        natural_anchor_rows.append({
            "sample_id": sample_id,
            "speaker_id": speaker,
            "split": row["split"],
            "best_column": anchor_column_best,
            "best_k": clip_k,
            "official_offset": -clip_k,
            "support_n": int(support),
            "score_q10": curve,
            "distance_matrix_sha256": ops.array_sha256(dists),
            "lip_embedding_sha256": ops.array_sha256(lip),
            "audio_embedding_sha256": ops.array_sha256(audio),
            "official_jpeg": jpeg_meta,
            "visual_crop": crop_meta,
            "video": video_meta,
            "audio_frontend": audio_meta,
        })
        np.savez_compressed(
            run_dir / "calibration_features" / f"{sample_id}.npz",
            natural_pcm16=natural_pcm,
            natural_raw_float32=natural_float,
            natural_fixed_float32=fixed_n,
            tts_pcm16=tts_pcm,
            tts_raw_float32=tts_float,
            tts_fixed_float32=fixed_t,
            natural_mel_raw=n_raw_mel,
            natural_mel_fixed=n_fixed_mel,
            tts_mel_raw=t_raw_mel,
            tts_mel_fixed=t_fixed_mel,
            natural_mel_plus3db=n_plus_mel,
            natural_mel_minus3db=n_minus_mel,
            natural_mel_starts=np.asarray(n_fixed_starts, dtype=np.int32),
            tts_mel_starts=np.asarray(t_fixed_starts, dtype=np.int32),
            natural_z_raw=n_raw_z,
            natural_z_fixed=n_fixed_z,
            tts_z_raw=t_raw_z,
            tts_z_fixed=t_fixed_z,
            natural_z_plus3db=n_plus_z,
            natural_z_minus3db=n_minus_z,
            natural_speech_mask_fixed=mask_n,
            tts_speech_mask_fixed=mask_t,
            anchor_lip=lip,
            anchor_audio=audio,
            anchor_dists=dists,
        )
        axis_rows.append({
            "sample_id": sample_id,
            "speaker_id": speaker,
            "split": row["split"],
            "z_natural_fixed": n_fixed_z,
            "z_tts_fixed": t_fixed_z,
            "mask_natural_fixed": mask_n,
            "mask_tts_fixed": mask_t,
            "z_natural_plus3db": n_plus_z,
            "z_natural_minus3db": n_minus_z,
        })
        per_clip[sample_id] = {
            "speaker_id": speaker,
            "natural": natural_meta,
            "tts": tts_meta,
            "tts_provider_native_24k": tts_row["qwen_provider_native_24k"],
            "natural_rms_normalization": norm_n,
            "tts_rms_normalization": norm_t,
            "natural_gain_plus3db": plus_gain_meta,
            "natural_gain_minus3db": minus_gain_meta,
            "natural_frontend": {"raw": n_raw_meta, "fixed": n_fixed_meta, "plus3db": n_plus_meta, "minus3db": n_minus_meta},
            "tts_frontend": {"raw": t_raw_meta, "fixed": t_fixed_meta},
            "shared_face_cache_parity": parity,
            "natural_mask": mask_n_meta,
            "tts_mask": mask_t_meta,
            "anchor": natural_anchor_rows[-1],
        }

    d_source, u_level, axes_meta = ops.calibration_axes(axis_rows)
    speaker_groups: dict[str, list[int]] = {}
    for row in natural_anchor_rows:
        speaker_groups.setdefault(row["speaker_id"], []).append(row["best_k"])
    speaker_medians = {speaker: float(np.median(values)) for speaker, values in sorted(speaker_groups.items())}
    if len(speaker_medians) != 10 or any(len(values) != 2 for values in speaker_groups.values()):
        raise ValueError("global anchor requires 10 calibration speakers with exactly two clips each")
    anchor_k, anchor_meta = _nearest_anchor_k(list(speaker_medians.values()), historical_k=3)
    anchor_meta.update({
        "rule": "q10 = G(FIXED natural), A_RAW natural; clip best k=column-15, median of two clips per speaker, median of 10 speaker medians, nearest integer",
        "speaker_medians": speaker_medians,
        "speaker_count": len(speaker_medians),
        "calibration_clip_count": len(natural_anchor_rows),
        "anchor_column": int(15 + anchor_k),
    })

    axes_npz = run_dir / "calibration_axes.npz"
    np.savez_compressed(axes_npz, d_source=d_source, u_level=u_level)
    summary = {
        "status": "calibration_complete",
        "classification": "sealed calibration; no evaluation effects",
        "run_dir": str(run_dir),
        "inputs": {
            "source_manifest_path": str(input_manifest_path),
            "source_manifest_raw_sha256": _sha(input_manifest_path),
            "source_manifest_embedded_sha256": input_manifest.get("manifest_sha256"),
            "tts_pair_manifest_path": str(tts_manifest_path),
            "tts_pair_manifest_raw_sha256": _sha(tts_manifest_path),
            "tts_pair_manifest_embedded_sha256": tts_manifest.get("manifest_sha256"),
            "run_frozen_input_manifest_path": str(frozen_input_manifest_path),
            "run_frozen_input_manifest_raw_sha256": _sha(frozen_input_manifest_path),
            "run_frozen_input_manifest_embedded_sha256": frozen_input_manifest.get("manifest_sha256"),
            "input_identity_sha256": tts_manifest.get("input_identity_sha256"),
            "input_identity_rows_match": True,
            "calibration_rows": len(cal_rows),
            "calibration_speakers": len(speaker_medians),
        },
        "method": {
            "seed": 20260930,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "wav2lip_repo_revision": _git_revision(w2l_root),
            "syncnet_repo_revision": _git_revision(sync_root),
            "wav2lip_checkpoint": model_meta,
            "syncnet_checkpoint": sync_meta,
            "wav2lip_frontend_compatibility": frontend,
            "library_versions": {
                "numpy": str(np.__version__),
                "opencv": str(cv2.__version__),
                "scipy": str(__import__("scipy").__version__),
                "librosa": str(__import__("librosa").__version__),
                "python_speech_features": sync_meta["python_speech_features_version"],
            },
            "portable_code_sha256": code_hashes,
            "wav2lip_batch_size": args.wav2lip_batch_size,
            "syncnet_batch_size": args.syncnet_batch_size,
            "shared_face_encoder_batch_sizes": list(face_feature_cache),
            "face": face_meta,
            "scoring": "official SyncNet V2 visual JPEG frontend, original PCM16 integer amplitude units; float fixed PCM scores use float64*32768 with no quantization/clipping",
        },
        "axes": {
            "d_sha256": ops.array_sha256(d_source),
            "u_sha256": ops.array_sha256(u_level),
            "d_norm": float(np.linalg.norm(d_source)),
            "u_norm": float(np.linalg.norm(u_level)),
            "d_u_cosine": float((d_source @ u_level) / (np.linalg.norm(d_source) * np.linalg.norm(u_level))),
            "metadata": axes_meta,
            "npz_path": str(axes_npz),
            "npz_sha256": _sha(axes_npz),
        },
        "global_anchor": anchor_meta,
        "clips": per_clip,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output = run_dir / "calibration_summary.json"
    summary["summary_path"] = str(output)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{_sha(output)}  {output.name}\n", encoding="ascii")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--tts-pair-manifest", required=True)
    parser.add_argument("--frozen-input-manifest", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--wav2lip-batch-size", type=int, default=32)
    parser.add_argument("--syncnet-batch-size", type=int, default=20)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    summary = run(args)
    destination = Path(args.summary_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(destination), "elapsed_seconds": summary["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
