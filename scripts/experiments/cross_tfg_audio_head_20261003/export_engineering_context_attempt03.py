#!/usr/bin/env python3
"""Export manifest-bound QC06 context inputs for an engineering-only smoke.

This does not train, score a candidate, modify the old QC run, or write into a
source/model checkout. The face crop follows the pinned Wav2Lip inference
face_detect path; SyncNet frames and track geometry come from QC06's selected
official crop; the scorer PCM16 crop is freshly extracted from that crop AVI.
Exact terminal-status and manifest SHA pins are required at launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


BASE = Path("/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt06/run")
QC06_JOB = Path("/root/tts-research-system-20261001/job_control/cross_tfg_audio_head_20261003_oldpilot_qc_attempt06")
QC06_SOURCE_SHA = "cf4d34832ee93bf5a4d2e4be349dad15469e423888c0a1c41bbbdf53f2a4726d"
QC06_SUPERVISOR_SHA = "e770b7b80b35262211b8f197136bda37e27e173d89539d1295c6c7412924affe"
SOURCE_FACE_SHA = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
ORIGINAL_SOURCE_WAV_SHA = "263b19510c95c32db4874449aadefc9b37c008e4816c18028eaf8fef463828e5"
TRACK_PCM16_N = 91_520
NATURAL_F32_PCM_SHA = "b9a24dd67fd557e012e58339ebe5b1d22655240cf13dd3e31ae7173f3bd0d15a"
NATURAL_F32_FILE_SHA = "d12ced7d44e7d70713a7e201abe47ff60ec190af20e9cd447cba5d325c11caa4"
NATURAL_F32_N = 92_225
FRAME_COUNT = 141
FPS_NUM = 25
FPS_DEN = 1
CROP_SCALE = 0.40


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def video_stream_sha256(path: Path) -> str:
    completed = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
         "-c:v", "copy", "-f", "hash", "-hash", "sha256", "pipe:1"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg video-stream hash failed: " + completed.stderr.decode("utf-8", "replace")[-1000:])
    value = completed.stdout.decode("ascii", "strict").strip()
    if not value.startswith("SHA256="):
        raise RuntimeError("ffmpeg returned no SHA-256 video-stream digest")
    return value.split("=", 1)[1].lower()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def patch_librosa_compatibility() -> dict[str, str]:
    import inspect
    import librosa

    original = librosa.filters.mel
    signature = str(inspect.signature(original))
    if not getattr(original, "_cross_tfg_positional_compat", False):
        def compatible(*args: Any, **kwargs: Any):
            positional = list(args)
            if positional and "sr" not in kwargs:
                kwargs["sr"] = positional.pop(0)
            if positional and "n_fft" not in kwargs:
                kwargs["n_fft"] = positional.pop(0)
            if positional:
                raise TypeError("unexpected positional arguments in librosa.filters.mel compatibility shim")
            return original(**kwargs)
        compatible._cross_tfg_positional_compat = True  # type: ignore[attr-defined]
        librosa.filters.mel = compatible
    return {"librosa_version": str(librosa.__version__), "original_mel_signature": signature,
            "shim": "historical positional sr,n_fft mapped to keyword args; remaining defaults unchanged"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qc-run-root", type=Path, default=BASE)
    parser.add_argument("--qc-terminal-status", type=Path, default=QC06_JOB / "terminal_status_attempt06.json")
    parser.add_argument("--qc-terminal-status-sha256", required=True)
    parser.add_argument("--qc-manifest-sha256", required=True)
    parser.add_argument("--wav2lip-root", type=Path, default=Path("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip"))
    parser.add_argument("--face", type=Path, default=Path("/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png"))
    parser.add_argument("--original-source-wav", type=Path, default=Path("/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/pilot/natural_1.wav"))
    parser.add_argument("--natural-f32-wav", type=Path, default=BASE / "audio/natural_official_load_f32.wav")
    parser.add_argument("--syncnet-root", type=Path, default=Path("/root/autodl-tmp/wav2lip-syncnet/syncnet_python"))
    parser.add_argument("--syncnet-data", type=Path, default=BASE / "rows/identity/syncnet_identity")
    args = parser.parse_args()

    output = args.output_root
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing context export: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": "cross_tfg_differentiable_engineering_context_v1",
        "status": "RUNNING", "engineering_only": True, "candidate_selection_eligible": False,
        "started_unix": started, "output_root": str(output),
    }
    json_write(output / "export_status.json", result)
    try:
        import cv2
        from scipy.io import wavfile

        face_path = args.face.resolve(strict=True)
        original_source_wav = args.original_source_wav.resolve(strict=True)
        natural_wav = args.natural_f32_wav.resolve(strict=True)
        wav2lip_root = args.wav2lip_root.resolve(strict=True)
        syncnet_root = args.syncnet_root.resolve(strict=True)
        data_root = args.syncnet_data.resolve(strict=True)
        qc_run_root = args.qc_run_root.resolve(strict=True)
        qc_manifest_path = qc_run_root / "engineering_qc_manifest.json"
        qc_terminal_path = args.qc_terminal_status.resolve(strict=True)
        qc_terminal_bytes = qc_terminal_path.read_bytes()
        qc_terminal_sha = sha256_bytes(qc_terminal_bytes)
        if qc_terminal_sha != args.qc_terminal_status_sha256.lower():
            raise RuntimeError(f"QC06 terminal-status SHA mismatch: {qc_terminal_sha}")
        qc_terminal = json.loads(qc_terminal_bytes.decode("utf-8"))
        if (qc_terminal.get("schema") != "cross_tfg_oldpilot_qc_attempt06_terminal_v1"
                or qc_terminal.get("status") != "PASS_ENGINEERING_ONLY"
                or qc_terminal.get("child_return_code") != 0
                or qc_terminal.get("manifest_qualification_pass") is not True
                or qc_terminal.get("qc_source_sha256") != QC06_SOURCE_SHA
                or qc_terminal.get("supervisor_source_sha256") != QC06_SUPERVISOR_SHA):
            raise RuntimeError("QC06 terminal status is not an exact-repeat PASS")
        qc_manifest_bytes = qc_manifest_path.read_bytes()
        qc_manifest_sha = sha256_bytes(qc_manifest_bytes)
        if qc_manifest_sha != args.qc_manifest_sha256.lower():
            raise RuntimeError(f"QC06 manifest SHA mismatch: {qc_manifest_sha}")
        qc_manifest = json.loads(qc_manifest_bytes.decode("utf-8"))
        terminal_manifest = qc_terminal.get("qc_manifest", {})
        if (terminal_manifest.get("sha256") != qc_manifest_sha
                or int(terminal_manifest.get("bytes", -1)) != len(qc_manifest_bytes)
                or Path(str(terminal_manifest.get("path", ""))).resolve() != qc_manifest_path.resolve()
                or qc_manifest.get("status") != "PASS_ENGINEERING_ONLY"):
            raise RuntimeError("QC06 terminal receipt does not bind the manifest")
        repeat_qc = qc_manifest.get("identity_repeat_qc", {})
        repeat_checks = (
            "identity_generated_video_elementary_stream_sha256_equal",
            "identity_syncnet_crop_video_sha256_equal",
            "identity_syncnet_face_bbox_arrays_sha256_equal",
            "identity_syncnet_per_query_matrix_sha256_equal",
            "identity_syncnet_tracks_pickle_sha256_equal",
        )
        if repeat_qc.get("status") != "EXACT_MATCH" or any(repeat_qc.get(key) is not True for key in repeat_checks):
            raise RuntimeError("QC06 manifest lacks an exact identity-repeat check")
        qc_rows = qc_manifest.get("rows", [])
        if (len(qc_rows) != 3
                or {str(row.get("method_id")) for row in qc_rows} != {"identity", "rms_fixed", "identity_repeat"}
                or any(row.get("status") != "PASS" for row in qc_rows)):
            raise RuntimeError("QC06 qualified manifest does not contain the three frozen engineering rows")
        identity_row = next(row for row in qc_rows if row.get("method_id") == "identity")
        repeat_row = next(row for row in qc_rows if row.get("method_id") == "identity_repeat")
        identity_tracking = identity_row.get("fresh_syncnet", {}).get("tracking_artifacts", {})
        repeat_tracking = repeat_row.get("fresh_syncnet", {}).get("tracking_artifacts", {})
        crop_records = identity_tracking.get("crop_videos", [])
        repeat_crop_records = repeat_tracking.get("crop_videos", [])
        numeric_tracks = identity_tracking.get("numeric_tracks", [])
        repeat_numeric_tracks = repeat_tracking.get("numeric_tracks", [])
        if (len(crop_records) != 1 or len(repeat_crop_records) != 1
                or len(numeric_tracks) != 1 or len(repeat_numeric_tracks) != 1
                or crop_records[0].get("name") != "00000.avi"
                or repeat_crop_records[0].get("name") != "00000.avi"
                or crop_records[0].get("sha256") != repeat_crop_records[0].get("sha256")
                or identity_tracking.get("tracks_pickle_sha256") != repeat_tracking.get("tracks_pickle_sha256")
                or numeric_tracks[0].get("bbox_sha256") != repeat_numeric_tracks[0].get("bbox_sha256")):
            raise RuntimeError("QC06 identity and repeat track/crop pins do not agree")
        video_path = Path(str(identity_row["generated_video_path"])).resolve(strict=True)
        expected_video_path = (qc_run_root / "rows/identity/generated_video.mp4").resolve()
        if video_path != expected_video_path:
            raise RuntimeError("QC06 identity video path escaped the qualified run root")
        crop_avi = data_root / "pycrop/oldpilot_identity/00000.avi"
        tracks_path = data_root / "pywork/oldpilot_identity/tracks.pckl"
        if not video_path.is_file() or not crop_avi.is_file() or not tracks_path.is_file():
            raise FileNotFoundError("QC06 generated video, selected crop AVI, or tracks.pckl is missing")

        actual_hashes = {
            "original_source_wav": sha256_file(original_source_wav),
            "face": sha256_file(face_path), "natural_f32_wav": sha256_file(natural_wav),
            "source_video": sha256_file(video_path), "crop_avi": sha256_file(crop_avi),
            "tracks_pickle": sha256_file(tracks_path),
            "source_video_elementary_stream": video_stream_sha256(video_path),
            "qc_terminal_status": qc_terminal_sha,
            "qc_manifest": qc_manifest_sha,
            "wav2lip_inference_py": sha256_file(wav2lip_root / "inference.py"),
            "wav2lip_audio_py": sha256_file(wav2lip_root / "audio.py"),
            "wav2lip_hparams_py": sha256_file(wav2lip_root / "hparams.py"),
            "wav2lip_checkpoint": sha256_file(wav2lip_root / "checkpoints/wav2lip_gan.pth"),
            "syncnet_run_pipeline_py": sha256_file(syncnet_root / "run_pipeline.py"),
            "syncnet_instance_py": sha256_file(syncnet_root / "SyncNetInstance.py"),
        }
        expected = {
            "original_source_wav": ORIGINAL_SOURCE_WAV_SHA,
            "face": SOURCE_FACE_SHA, "natural_f32_wav": NATURAL_F32_FILE_SHA,
            "source_video": str(identity_row["generated_video_sha256"]).lower(),
            "source_video_elementary_stream": str(identity_row["generated_video_elementary_stream_sha256"]).lower(),
            "crop_avi": str(crop_records[0]["sha256"]).lower(),
            "tracks_pickle": str(identity_tracking["tracks_pickle_sha256"]).lower(),
        }
        for name, digest in expected.items():
            if actual_hashes[name] != digest:
                raise RuntimeError(f"QC06 {name} SHA mismatch: {actual_hashes[name]} != {digest}")

        sample_rate, natural = wavfile.read(str(natural_wav))
        if sample_rate != 16_000 or natural.dtype != np.float32 or natural.ndim != 1 or natural.size != NATURAL_F32_N:
            raise RuntimeError(f"natural F32 WAV does not match the frozen contract: sr={sample_rate}, dtype={natural.dtype}, shape={natural.shape}")
        natural_pcm_sha = sha256_bytes(np.ascontiguousarray(natural).tobytes())
        if natural_pcm_sha != NATURAL_F32_PCM_SHA:
            raise RuntimeError("natural WAV decoded float32 PCM hash differs from the QC06 mux source")

        # Read the immutable selected track produced by the official SyncNet pipeline.
        import pickle
        with tracks_path.open("rb") as handle:
            tracks = pickle.load(handle)
        if not isinstance(tracks, list) or not tracks:
            raise RuntimeError("tracks.pckl has no selected tracks")
        selected = tracks[0]
        if not isinstance(selected, dict) or "track" not in selected or "proc_track" not in selected:
            raise RuntimeError("track[0] does not expose the official {track,proc_track} contract")
        track = selected["track"]
        proc = selected["proc_track"]
        frames = np.asarray(track["frame"], dtype=np.int64)
        boxes = np.asarray(track["bbox"], dtype=np.float32)
        bbox_sha = sha256_bytes(np.ascontiguousarray(boxes).tobytes())
        bbox_f64_sha = sha256_bytes(np.ascontiguousarray(track["bbox"], dtype="<f8").tobytes())
        if bbox_f64_sha != str(numeric_tracks[0]["bbox_sha256"]).lower():
            raise RuntimeError("selected float64 bbox array differs from the exact QC06 manifest pin")
        if frames.shape != (143,) or int(frames[0]) != 0 or int(frames[-1]) != 142:
            raise RuntimeError(f"selected track frame map differs from QC06: {frames[:2]}..{frames[-2:]}")
        if boxes.shape != (143, 4):
            raise RuntimeError(f"selected track bbox shape mismatch: {boxes.shape}")
        for key in ("x", "y", "s"):
            if len(proc[key]) != len(frames):
                raise RuntimeError(f"proc_track.{key} length mismatch")

        # Reconstruct the pinned static-image face crop using inference.py's own face_detect.
        patch_librosa_compatibility()
        source_img = cv2.imread(str(face_path))
        if source_img is None or source_img.shape[:2] != (512, 512):
            raise RuntimeError(f"unexpected source face image: {None if source_img is None else source_img.shape}")
        os.chdir(output)
        (output / "temp").mkdir(exist_ok=True)
        sys.path.insert(0, str(wav2lip_root))
        sys.argv = [str(wav2lip_root / "inference.py"),
                    "--checkpoint_path", str(wav2lip_root / "checkpoints/wav2lip_gan.pth"),
                    "--face", str(face_path), "--audio", str(natural_wav),
                    "--outfile", str(output / "unused_static_face_probe.mp4"),
                    "--static", "True", "--fps", "25", "--pads", "0", "10", "0", "0"]
        inference = runpy.run_path(str(wav2lip_root / "inference.py"), run_name="cross_tfg_context_face_detect")
        face_items = inference["face_detect"]([source_img])
        if len(face_items) != 1:
            raise RuntimeError("pinned Wav2Lip face_detect did not return exactly one static-image crop")
        face_crop, w2l_coords = face_items[0]
        y1, y2, x1, x2 = [int(v) for v in w2l_coords]
        face96 = cv2.resize(face_crop, (96, 96))
        if face96.dtype != np.uint8 or face96.shape != (96, 96, 3):
            raise RuntimeError("Wav2Lip face crop must be uint8 BGR [96,96,3]")
        face_chw = np.transpose(face96, (2, 0, 1)).copy()
        face_crops = np.repeat(face_chw[None, ...], FRAME_COUNT, axis=0)

        # Decode the actual SyncNet crop AVI. W2L supplied 141 frames; the
        # official 25fps converter/crop supplied 143. The proxy keeps the first
        # 141 source-clock crops and records the two-frame boundary difference.
        cap = cv2.VideoCapture(str(crop_avi))
        decoded: list[np.ndarray] = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            decoded.append(image)
        cap.release()
        if len(decoded) != 143 or any(im.shape != (224, 224, 3) or im.dtype != np.uint8 for im in decoded):
            raise RuntimeError(f"unexpected selected SyncNet crop AVI decode: {len(decoded)} frames")
        syncnet_crops = np.stack([np.transpose(im, (2, 0, 1)) for im in decoded[:FRAME_COUNT]], axis=0).copy()

        # Map the official W2L integer paste rectangle into the official
        # SyncNet crop's 224x224 resized coordinate frame, using run_pipeline's
        # saved median-smoothed track geometry and exact half-open crop bounds.
        patch_boxes: list[list[int]] = []
        for index in range(FRAME_COUNT):
            bs = float(proc["s"][index])
            center_x = float(proc["x"][index])
            center_y = float(proc["y"][index])
            bsi = int(bs * (1.0 + 2.0 * CROP_SCALE))
            y0 = int(center_y + bsi - bs) - bsi
            y_stop = int(center_y + bsi + bs * (1.0 + 2.0 * CROP_SCALE)) - bsi
            x0 = int(center_x + bsi - bs * (1.0 + CROP_SCALE)) - bsi
            x_stop = int(center_x + bsi + bs * (1.0 + CROP_SCALE)) - bsi
            crop_w, crop_h = x_stop - x0, y_stop - y0
            if crop_w <= 0 or crop_h <= 0:
                raise RuntimeError(f"invalid official SyncNet source crop at track frame {index}")
            sx, sy = 224.0 / crop_w, 224.0 / crop_h
            px1 = max(0, min(224, int(np.floor((x1 - x0) * sx))))
            py1 = max(0, min(224, int(np.floor((y1 - y0) * sy))))
            px2 = max(0, min(224, int(np.ceil((x2 - x0) * sx))))
            py2 = max(0, min(224, int(np.ceil((y2 - y0) * sy))))
            if not (0 <= px1 < px2 <= 224 and 0 <= py1 < py2 <= 224):
                raise RuntimeError(f"Wav2Lip patch falls outside SyncNet crop at frame {index}: {(px1, py1, px2, py2)}")
            patch_boxes.append([px1, py1, px2, py2])
        patch_boxes_array = np.asarray(patch_boxes, dtype=np.int32)

        # Recreate SyncNetInstance.evaluate's ffmpeg extraction from the
        # official selected-track AVI and preserve the exact int16 crop.
        extracted_audio = output / "syncnet_track_audio_pcm16.wav"
        ffmpeg_argv = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(crop_avi),
                       "-async", "1", "-ac", "1", "-vn", "-acodec", "pcm_s16le",
                       "-ar", "16000", str(extracted_audio)]
        extraction = subprocess.run(ffmpeg_argv, cwd=str(output), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        (output / "pcm16_extract.stdout.log").write_bytes(extraction.stdout)
        (output / "pcm16_extract.stderr.log").write_bytes(extraction.stderr)
        if extraction.returncode != 0:
            raise RuntimeError(f"ffmpeg scorer-audio extraction failed: {extraction.returncode}")
        pcm_rate, pcm16 = wavfile.read(str(extracted_audio))
        pcm16 = np.ascontiguousarray(pcm16)
        if pcm_rate != 16_000 or pcm16.dtype != np.int16 or pcm16.ndim != 1 or pcm16.size != TRACK_PCM16_N:
            raise RuntimeError(f"unexpected scorer PCM16 crop: sr={pcm_rate}, dtype={pcm16.dtype}, shape={pcm16.shape}")
        pcm_sha = sha256_bytes(pcm16.tobytes())

        # Recompute pinned Wav2Lip chunk starts from the natural source mel.
        audio_module = inference["audio"]
        official_wave = audio_module.load_wav(str(natural_wav), 16_000)
        mel = np.asarray(audio_module.melspectrogram(official_wave))
        mel_starts: list[int] = []
        index = 0
        while True:
            start = int(index * 80.0 / 25.0)
            if start + 16 > mel.shape[1]:
                mel_starts.append(int(mel.shape[1] - 16))
                break
            mel_starts.append(start)
            index += 1
        if len(mel_starts) != FRAME_COUNT:
            raise RuntimeError(f"pinned Wav2Lip chunk count {len(mel_starts)} differs from generated video {FRAME_COUNT}")

        timestamps = (frames[0] + np.arange(FRAME_COUNT, dtype=np.float64)) / FPS_NUM
        timestamp_array = np.ascontiguousarray(timestamps, dtype=np.float64)
        npz_path = output / "context.npz"
        np.savez_compressed(
            npz_path,
            face_crops_bgr_u8=np.ascontiguousarray(face_crops, dtype=np.uint8),
            syncnet_crops_bgr_u8=np.ascontiguousarray(syncnet_crops, dtype=np.uint8),
            patch_boxes_xyxy_i32=np.ascontiguousarray(patch_boxes_array, dtype=np.int32),
            frame_timestamps_seconds_f64=timestamp_array,
            syncnet_pcm16_crop_i16=pcm16,
        )
        npz_file_sha = sha256_file(npz_path)
        natural_int16 = np.rint(np.clip(natural.astype(np.float64)[:TRACK_PCM16_N] * 32768.0,
                                        -32768.0, 32767.0)).astype(np.int16)
        direct_pcm_sha = sha256_bytes(np.ascontiguousarray(natural_int16).tobytes())
        array_sha = {
            "face_crops_bgr_u8": sha256_bytes(face_crops.tobytes()),
            "syncnet_crops_bgr_u8": sha256_bytes(syncnet_crops.tobytes()),
            "patch_boxes_xyxy_i32": sha256_bytes(patch_boxes_array.tobytes()),
            "frame_timestamps_seconds_f64": sha256_bytes(timestamp_array.tobytes()),
            "syncnet_pcm16_crop_i16": pcm_sha,
        }
        result.update({
            "schema": "cross_tfg_oldpilot_differentiable_context_v1",
            "original_source_sha256": actual_hashes["original_source_wav"],
            "primary_face_sha256": actual_hashes["face"],
            "canonical_natural_pcm_sha256": natural_pcm_sha,
            "source_video_sha256": actual_hashes["source_video"],
            "qc_qualification": {
                "attempt": "oldpilot_qc_attempt06",
                "terminal_status_path": str(qc_terminal_path),
                "terminal_status_sha256": qc_terminal_sha,
                "manifest_path": str(qc_manifest_path),
                "manifest_sha256": qc_manifest_sha,
                "identity_repeat_status": repeat_qc["status"],
                "identity_repeat_exact_checks": {key: repeat_qc[key] for key in repeat_checks},
            },
            "face_crops_sha256": array_sha["face_crops_bgr_u8"],
            "syncnet_crops_sha256": array_sha["syncnet_crops_bgr_u8"],
            "npz_file_sha256": npz_file_sha,
            "frame_count": FRAME_COUNT,
            "fps_num": FPS_NUM,
            "fps_den": FPS_DEN,
            "track_start_frame": int(frames[0]),
            "track_pcm16_sample_count": int(pcm16.size),
            "array_sha256": array_sha,
            "pcm16_derivation": {
                "source": "official SyncNet selected-track crop AVI decoded to scorer PCM16",
                "command": ffmpeg_argv,
                "output_path": str(extracted_audio),
                "sample_rate_hz": int(pcm_rate),
                "sample_count": int(pcm16.size),
                "pcm_i16_sha256": pcm_sha,
                "direct_source_slice_sha256": direct_pcm_sha,
                "cause_of_difference": "not assigned; retained as an explicit separate audio lineage",
            },
            "pcm16_direct_float_slice_sha256": direct_pcm_sha,
            "status": "PASS_ENGINEERING_ONLY",
            "context_npz": {"path": str(npz_path), "file_sha256": npz_file_sha,
                            "keys": ["face_crops_bgr_u8", "syncnet_crops_bgr_u8", "patch_boxes_xyxy_i32",
                                     "frame_timestamps_seconds_f64", "syncnet_pcm16_crop_i16"]},
            "context_json": str(output / "context.json"),
            "source_video": {"path": str(video_path), "sha256": actual_hashes["source_video"],
                             "elementary_stream_sha256_from_qc06_manifest": actual_hashes["source_video_elementary_stream"],
                             "decoded_w2l_frame_count": FRAME_COUNT, "decoded_width": 512, "decoded_height": 512},
            "source_face": {"path": str(face_path), "sha256": actual_hashes["face"],
                            "w2l_face_bbox_y1_y2_x1_x2": [y1, y2, x1, x2],
                            "single_crop_sha256": sha256_bytes(np.ascontiguousarray(face_chw).tobytes())},
            "natural_f32": {"path": str(natural_wav), "file_sha256": actual_hashes["natural_f32_wav"],
                            "pcm_f32_sha256": natural_pcm_sha, "sample_rate_hz": 16_000,
                            "sample_count": NATURAL_F32_N},
            "syncnet_track": {"track_index": 0, "frame_indices": [int(frames[0]), int(frames[-1])],
                              "track_start_frame": int(frames[0]), "track_frame_count": int(len(frames)),
                              "fps_num": FPS_NUM, "fps_den": FPS_DEN, "crop_scale": CROP_SCALE,
                              "bbox_shape": list(boxes.shape), "bbox_float32_sha256": bbox_sha,
                              "bbox_float64_sha256_from_qc06_manifest": bbox_f64_sha,
                              "crop_avi_path": str(crop_avi), "crop_avi_sha256": actual_hashes["crop_avi"],
                              "crop_avi_decoded_frame_count": len(decoded), "crop_size_px": [224, 224],
                              "context_frames_used": FRAME_COUNT,
                              "tail_frames_in_official_syncnet_crop_not_rendered_by_w2l": len(decoded) - FRAME_COUNT,
                              "patch_boxes_xyxy_i32_sha256": sha256_bytes(patch_boxes_array.tobytes())},
            "syncnet_scorer_pcm16": {"path": str(extracted_audio), "file_sha256": sha256_file(extracted_audio),
                                     "ffmpeg_argv": ffmpeg_argv, "ffmpeg_returncode": extraction.returncode,
                                     "sample_rate_hz": pcm_rate, "sample_count": int(pcm16.size),
                                     "pcm_i16_sha256": pcm_sha,
                                     "direct_natural_pcm16_slice_same_count_sha256": direct_pcm_sha,
                                     "direct_slice_sample_count": int(natural_int16.size),
                                     "same_pcm_as_direct_source_slice": bool(pcm_sha == direct_pcm_sha),
                                     "interpretation": "official scorer PCM16 is retained as extracted from the selected-track AVI; direct source PCM16 is a distinct diagnostic input; no cause assigned here"},
            "mel_clock": {"official_wav2lip_mel_frames": int(mel.shape[1]), "mel_start_count": len(mel_starts),
                          "mel_starts": mel_starts,
                          "mel_starts_sha256": sha256_bytes(np.asarray(mel_starts, dtype=np.int64).tobytes()),
                          "mel_tail_repetition_policy": "pinned Wav2Lip inference int(i*80/fps), final 16-frame tail"},
            "array_seals": {
                "face_crops_bgr_u8": {"dtype": "uint8", "shape": list(face_crops.shape), "sha256": array_sha["face_crops_bgr_u8"]},
                "syncnet_crops_bgr_u8": {"dtype": "uint8", "shape": list(syncnet_crops.shape), "sha256": array_sha["syncnet_crops_bgr_u8"]},
                "patch_boxes_xyxy_i32": {"dtype": "int32", "shape": list(patch_boxes_array.shape), "sha256": array_sha["patch_boxes_xyxy_i32"]},
                "frame_timestamps_seconds_f64": {"dtype": "float64", "shape": list(timestamp_array.shape), "sha256": array_sha["frame_timestamps_seconds_f64"]},
                "syncnet_pcm16_crop_i16": {"dtype": "int16", "shape": list(pcm16.shape), "sha256": array_sha["syncnet_pcm16_crop_i16"]},
            },
            "source_code_file_sha256": actual_hashes,
            "face_detect_runtime": {"librosa_compatibility": patch_librosa_compatibility(),
                                    "face_detect_source": str(wav2lip_root / "inference.py"),
                                    "coordinate_order": "source inference returns y1,y2,x1,x2; arrays store x1,y1,x2,y2"},
            "finished_unix": time.time(),
        })
        json_write(output / "context.json", result)
        result["context_json_file_sha256"] = sha256_file(output / "context.json")
        json_write(output / "export_status.json", result)
        print(json.dumps({"status": result["status"], "context_npz": str(npz_path),
                          "context_sha256": npz_file_sha, "context_json": str(output / "context.json"),
                          "context_json_sha256": result["context_json_file_sha256"],
                          "track_pcm16_sha256": pcm_sha, "frame_count": FRAME_COUNT}, sort_keys=True))
        return 0
    except BaseException as exc:
        result.update({"status": "FAILED_ENGINEERING_CONTEXT_EXPORT", "finished_unix": time.time(),
                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        json_write(output / "export_status.json", result)
        print(json.dumps({"status": result["status"], "error_type": result["error_type"],
                          "error": result["error"], "export_status": str(output / "export_status.json")}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
