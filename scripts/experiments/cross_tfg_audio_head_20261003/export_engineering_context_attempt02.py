#!/usr/bin/env python3
"""Export immutable oldpilot context inputs for a differentiable engineering smoke.

This does not train, score a candidate, modify the old QC run, or write into a
source/model checkout. The face crop follows the pinned Wav2Lip inference
face_detect path; SyncNet frames and track geometry come from QC05's selected
official crop; the scorer PCM16 crop is freshly extracted from that crop AVI.
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


BASE = Path("/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt05/run")
SOURCE_FACE_SHA = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
ORIGINAL_SOURCE_WAV_SHA = "263b19510c95c32db4874449aadefc9b37c008e4816c18028eaf8fef463828e5"
SOURCE_VIDEO_SHA = "be625feea9e329daff62a59a9bd260a55ff644ae119a66f3abc47c9ac6e9fc04"
SOURCE_VIDEO_ES_SHA = "45fefab2af2960f24d90e93abe75285728e9f9753090a518e7b6cae54e1b9a3e"
CROP_AVI_SHA = "cf536e189524653697a29c609fb8a315744bcdcc6285a80b146e84097b5fb1d8"
TRACK_BBOX_F32_SHA = "ed2d8067a0ab2ba75ab668cee58f56c21379c1f5f3bf1496c962d190d867217c"
TRACK_PCM16_SHA = "91321866746580de045feb8fdcc83f7993aeff9e2ed6282385a64a7aa6be657f"
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
        video_path = BASE / "rows/identity/generated_video.mp4"
        crop_avi = data_root / "pycrop/oldpilot_identity/00000.avi"
        tracks_path = data_root / "pywork/oldpilot_identity/tracks.pckl"
        if not video_path.is_file() or not crop_avi.is_file() or not tracks_path.is_file():
            raise FileNotFoundError("QC05 generated video, selected crop AVI, or tracks.pckl is missing")

        actual_hashes = {
            "original_source_wav": sha256_file(original_source_wav),
            "face": sha256_file(face_path), "natural_f32_wav": sha256_file(natural_wav),
            "source_video": sha256_file(video_path), "crop_avi": sha256_file(crop_avi),
            "tracks_pickle": sha256_file(tracks_path),
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
            "source_video": SOURCE_VIDEO_SHA, "crop_avi": CROP_AVI_SHA,
        }
        for name, digest in expected.items():
            if actual_hashes[name] != digest:
                raise RuntimeError(f"QC05 {name} SHA mismatch: {actual_hashes[name]} != {digest}")

        sample_rate, natural = wavfile.read(str(natural_wav))
        if sample_rate != 16_000 or natural.dtype != np.float32 or natural.ndim != 1 or natural.size != NATURAL_F32_N:
            raise RuntimeError(f"natural F32 WAV does not match the frozen contract: sr={sample_rate}, dtype={natural.dtype}, shape={natural.shape}")
        natural_pcm_sha = sha256_bytes(np.ascontiguousarray(natural).tobytes())
        if natural_pcm_sha != NATURAL_F32_PCM_SHA:
            raise RuntimeError("natural WAV decoded float32 PCM hash differs from the QC05 mux source")

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
        if bbox_sha != TRACK_BBOX_F32_SHA:
            raise RuntimeError(f"selected track bbox SHA mismatch: {bbox_sha}")
        if frames.shape != (143,) or int(frames[0]) != 0 or int(frames[-1]) != 142:
            raise RuntimeError(f"selected track frame map differs from QC05: {frames[:2]}..{frames[-2:]}")
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
        if pcm_sha != TRACK_PCM16_SHA:
            raise RuntimeError(f"official crop AVI PCM16 extraction SHA differs from scorer readback: {pcm_sha}")

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
                             "elementary_stream_sha256_from_qc05": SOURCE_VIDEO_ES_SHA,
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
