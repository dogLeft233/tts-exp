"""Engineering-only cudnn benchmark sensitivity check on the first two CAL rows.

This runner creates no calibration axes and does not inspect evaluation effects.
It executes N/T raw/fixed inputs under benchmark=False and benchmark=True, saves
the official JPEG/SyncNet path, latent and embedding arrays, and records the
flag immediately before each model-forward helper.
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
from typing import Any

_AUX_SITE_PACKAGES = os.environ.get("TTS_SOURCE_AXIS_AUX_SITE_PACKAGES", "")
if _AUX_SITE_PACKAGES and Path(_AUX_SITE_PACKAGES).is_dir() and _AUX_SITE_PACKAGES not in sys.path:
    sys.path.append(_AUX_SITE_PACKAGES)

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import evaluate, ops, scoring, wav2lip


def _sha(path: Path) -> str:
    return scoring.sha256_file(path)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _flags() -> dict[str, Any]:
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _read_pcm16(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_rate, pcm = wavfile.read(str(path))
    if sample_rate != ops.SAMPLE_RATE or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"expected mono 16-kHz PCM16: {path}; got {sample_rate}, {pcm.shape}, {pcm.dtype}")
    waveform = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
    units, units_meta = ops.pcm_to_syncnet_integer_units(waveform)
    if not np.array_equal(units, pcm.astype(np.float64)):
        raise AssertionError(f"PCM16-to-SyncNet float identity failed: {path}")
    return pcm, waveform, {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "sample_rate_hz": int(sample_rate),
        "samples": int(pcm.size),
        "duration_s": float(pcm.size / sample_rate),
        "pcm_dtype": str(pcm.dtype),
        "peak_abs_pcm16": int(np.max(np.abs(pcm.astype(np.int32)))),
        "syncnet_units": units_meta,
    }


def _instrument(
    events: list[dict[str, Any]],
    active: dict[str, str | None],
    decoded_pixels: dict[tuple[str, str], list[np.ndarray]],
) -> list[tuple[Any, str, Any]]:
    originals = []
    for module, name in (
        (wav2lip, "cache_face_features"),
        (wav2lip, "encode_latents"),
        (wav2lip, "decode_latents"),
        (scoring, "lip_embeddings_from_frames"),
        (scoring, "audio_embeddings_from_syncnet_units"),
        (scoring, "pairwise_distance_matrix"),
    ):
        original = getattr(module, name)
        originals.append((module, name, original))

        def wrapped(*args, __name=name, __original=original, **kwargs):
            event = {"operation": __name, "flags_at_call": _flags(), "time_monotonic": time.monotonic()}
            events.append(event)
            result = __original(*args, **kwargs)
            if __name == "decode_latents" and active.get("sample_id") and active.get("condition"):
                decoded_pixels.setdefault((active["sample_id"], active["condition"]), []).append(
                    result.detach().float().cpu().numpy().copy()
                )
            return result

        setattr(module, name, wrapped)
    return originals


def _restore_instrument(originals: list[tuple[Any, str, Any]]) -> None:
    for module, name, original in originals:
        setattr(module, name, original)


def _records(manifest_path: Path, tts_manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tts = json.loads(tts_manifest_path.read_text(encoding="utf-8"))
    cal_rows = [row for row in manifest["records"] if row.get("split") == "calibration"]
    tts_rows = {
        item.get("id", item.get("input_identity", {}).get("id")): item
        for item in tts["records"]
    }
    if len(cal_rows) != 20 or len(tts_rows) < 20:
        raise ValueError(f"expected frozen CAL20 manifests; source={len(cal_rows)}, TTS={len(tts_rows)}")
    fields = ("id", "speaker", "text", "split", "ref_sha256")
    for row in cal_rows:
        item = tts_rows.get(row["id"])
        identity = item.get("input_identity", {}) if item else {}
        if item is None or any(row.get(key) != identity.get(key) for key in fields):
            raise ValueError(f"source/TTS identity mismatch for {row['id']}")
    return cal_rows[:2], {"source": manifest, "tts": tts, "tts_by_id": tts_rows}


def _runtime_code_hashes() -> dict[str, str]:
    paths = {
        "cudnn_flag_engineering.py": Path(__file__).resolve(),
        "evaluate.py": Path(evaluate.__file__).resolve(),
        "ops.py": Path(ops.__file__).resolve(),
        "scoring.py": Path(scoring.__file__).resolve(),
        "wav2lip.py": Path(wav2lip.__file__).resolve(),
    }
    return {name: _sha(path) for name, path in paths.items()}


def _run_flag(args: argparse.Namespace, rows: list[dict[str, Any]], flag: bool, protocol: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve() / ("benchmark_true" if flag else "benchmark_false")
    run_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    torch.manual_seed(20261001)
    np.random.seed(20261001)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    before_model_load = _flags()
    _json(run_dir / "runtime_start.json", {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_cudnn_benchmark": flag,
        "before_model_load_flags": before_model_load,
        "code_sha256": _runtime_code_hashes(),
        "protocol_sha256": protocol["protocol_sha256"],
    })
    device = "cuda"
    frontend = wav2lip.install_librosa_compatibility_shim()
    wav2lip_root = Path(args.wav2lip_root).resolve()
    syncnet_root = Path(args.syncnet_root).resolve()
    model, model_meta = wav2lip.load_model(wav2lip_root, args.wav2lip_checkpoint, device=device)
    flag_after_wav2lip_load = _flags()
    syncnet, syncnet_meta = scoring.load_syncnet(syncnet_root, args.syncnet_checkpoint, device=device)
    flag_after_syncnet_load = _flags()

    face_path = Path(args.primary_face).resolve()
    frame = cv2.imread(str(face_path))
    if frame is None:
        raise RuntimeError(f"cannot read primary face: {face_path}")
    before_detect = _flags()
    face_meta = wav2lip.detect_static_face(frame, wav2lip_root, device=device)
    after_detect = _flags()
    face_meta.update({"path": str(face_path), "sha256": _sha(face_path), "width": int(frame.shape[1]), "height": int(frame.shape[0])})

    # The measured factor is set after the detector and before every forward helper.
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = flag
    after_explicit_override = _flags()
    active: dict[str, str | None] = {"sample_id": None, "condition": None}
    decoded_pixels: dict[tuple[str, str], list[np.ndarray]] = {}
    originals = _instrument(events, active, decoded_pixels)
    rows_out: list[dict[str, Any]] = []
    try:
        face_features: dict[int, list[Any]] = {}
        for source_row in rows:
            sample_id = source_row["id"]
            speaker = source_row["speaker"]
            utterance = sample_id.split("_", 1)[1]
            natural_path = Path(args.input_root).resolve() / "audio" / speaker / f"{utterance}.wav"
            tts_path = Path(args.input_root).resolve() / "tts_outputs" / "audio_16k_canonical" / f"{sample_id}.wav"
            n_pcm, n_float, n_meta = _read_pcm16(natural_path)
            t_pcm, t_float, t_meta = _read_pcm16(tts_path)
            n_fixed, n_norm = ops.normalize_rms(n_float)
            t_fixed, t_norm = ops.normalize_rms(t_float)
            waveforms = {
                "natural_raw": (n_pcm, n_float, n_meta),
                "natural_fixed": (n_pcm, n_fixed, {**n_meta, "normalization": n_norm}),
                "tts_raw": (t_pcm, t_float, t_meta),
                "tts_fixed": (t_pcm, t_fixed, {**t_meta, "normalization": t_norm}),
            }
            for visual_id, (pcm, waveform, input_meta) in waveforms.items():
                active["sample_id"] = sample_id
                active["condition"] = visual_id
                torch.backends.cudnn.benchmark = flag
                mel, chunks, starts, mel_meta = wav2lip.mel_chunks(waveform, wav2lip_root)
                z = wav2lip.encode_latents(model, chunks, device=device, batch_size=args.wav2lip_batch_size)
                if len(z) == 0:
                    raise RuntimeError(f"no Wav2Lip frames for {sample_id}/{visual_id}")
                tail = len(z) % args.wav2lip_batch_size
                counts = {args.wav2lip_batch_size}
                if tail:
                    counts.add(tail)
                missing = counts.difference(face_features)
                if missing:
                    torch.backends.cudnn.benchmark = flag
                    face_features.update(wav2lip.cache_face_features(model, frame, face_meta["generation_box_xyxy"], sorted(missing), device=device))

                sync_units, sync_units_meta = ops.pcm_to_syncnet_integer_units(waveform)
                if visual_id.endswith("_raw"):
                    if not np.array_equal(sync_units, pcm.astype(np.float64)):
                        raise AssertionError(f"raw PCM identity mismatch for {sample_id}/{visual_id}")
                n_windows = min(len(z), len(sync_units) // 640) - 5
                if n_windows < 1:
                    raise ValueError(f"insufficient official common windows: {sample_id}/{visual_id}/{n_windows}")
                torch.backends.cudnn.benchmark = flag
                audio, audio_meta = scoring.audio_embeddings_from_syncnet_units(
                    syncnet, sync_units, window_count=n_windows, device=device, batch_size=args.syncnet_batch_size,
                )

                visual, visual_meta = evaluate._load_or_make_visual(
                    run_dir, sample_id, "primary", visual_id, z, frame, face_meta, model,
                    face_features, syncnet, argparse.Namespace(
                        ffmpeg=args.ffmpeg,
                        wav2lip_batch_size=args.wav2lip_batch_size,
                        syncnet_batch_size=args.syncnet_batch_size,
                    ), device,
                )
                chunks_for_pixels = decoded_pixels.pop((sample_id, visual_id), [])
                if not chunks_for_pixels:
                    raise AssertionError(f"decoder did not produce pixel tensors for {sample_id}/{visual_id}")
                decoder_pixels = np.concatenate(chunks_for_pixels, axis=0).astype(np.float32, copy=False)
                if len(decoder_pixels) != len(z):
                    raise AssertionError(f"decoded pixel frame count mismatch for {sample_id}/{visual_id}: {len(decoder_pixels)} vs {len(z)}")
                decoder_pixels_uint8 = (decoder_pixels.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
                active["sample_id"] = None
                active["condition"] = None
                if len(visual) != len(audio):
                    raise AssertionError(f"A/V windows do not align for {sample_id}/{visual_id}: {len(audio)} vs {len(visual)}")
                torch.backends.cudnn.benchmark = flag
                matrix = scoring.pairwise_distance_matrix(visual, audio, vshift=15, device=device)
                artifact = run_dir / "arrays" / f"{sample_id}__{visual_id}.npz"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    artifact,
                    mel_float32=mel,
                    z_float32=z,
                    syncnet_audio_units_float64=sync_units,
                    audio_embeddings_float32=audio,
                    lip_embeddings_float32=visual,
                    cross_distance_raw_float32=matrix,
                    mel_starts_int32=np.asarray(starts, dtype=np.int32),
                    decoder_pixels_float32=decoder_pixels,
                    decoder_pixels_uint8=decoder_pixels_uint8,
                )
                row_result = {
                    "sample_id": sample_id,
                    "speaker_id": speaker,
                    "split": "calibration",
                    "flag": "benchmark_true" if flag else "benchmark_false",
                    "condition": visual_id,
                    "input": input_meta,
                    "mel": mel_meta,
                    "syncnet_audio": {**sync_units_meta, **audio_meta},
                    "visual": visual_meta,
                    "array_npz": str(artifact),
                    "array_npz_sha256": _sha(artifact),
                    "array_shapes": {
                        "mel": list(mel.shape), "z": list(z.shape), "decoder_pixels": list(decoder_pixels.shape),
                        "A": list(audio.shape), "V": list(visual.shape), "cross_distance": list(matrix.shape),
                    },
                    "common_windows": n_windows,
                    "frames": len(z),
                    "actual_flags": sorted({json.dumps(event["flags_at_call"], sort_keys=True) for event in events if event.get("operation") in {"encode_latents", "cache_face_features", "decode_latents", "audio_embeddings_from_syncnet_units", "lip_embeddings_from_frames", "pairwise_distance_matrix"}}),
                    "artifact_sha256": {
                        "mel": ops.array_sha256(mel), "z": ops.array_sha256(z),
                        "A": ops.array_sha256(audio), "V": ops.array_sha256(visual),
                        "cross_distance": ops.array_sha256(matrix),
                        "decoder_pixels_float32": ops.array_sha256(decoder_pixels),
                        "decoder_pixels_uint8": ops.array_sha256(decoder_pixels_uint8),
                        "video_bgr_pixels": visual_meta["video"]["raw_bgr_pixel_sha256"],
                    },
                }
                rows_out.append(row_result)
                _append_jsonl(run_dir / "rows.jsonl", row_result)
                _json(run_dir / "events.json", {"events": events, "face_detection": {"before": before_detect, "after": after_detect}, "requested_flag": flag})
    finally:
        _restore_instrument(originals)
        _json(run_dir / "events.json", {
            "events": events,
            "face_detection": {"before": before_detect, "after": after_detect},
            "flags_after_explicit_override": after_explicit_override,
            "requested_flag": flag,
        })

    summary = {
        "status": "complete",
        "classification": "engineering-only; first two CAL; no axis fitting and no evaluation effects",
        "requested_cudnn_benchmark": flag,
        "flags": {
            "before_model_load": before_model_load,
            "after_wav2lip_load": flag_after_wav2lip_load,
            "after_syncnet_load": flag_after_syncnet_load,
            "before_static_detection": before_detect,
            "after_static_detection": after_detect,
            "after_explicit_override": after_explicit_override,
            "forward_events": events,
        },
        "runtime": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name(),
            "numpy": str(np.__version__), "opencv": str(cv2.__version__),
            "frontend_compatibility": frontend, "wav2lip_model": model_meta, "syncnet_model": syncnet_meta,
        },
        "face": face_meta,
        "rows": len(rows_out),
        "code_sha256": _runtime_code_hashes(),
    }
    _json(run_dir / "summary.json", summary)
    return {"run_dir": str(run_dir), "summary": summary, "rows": rows_out}


def _compare(false_run: dict[str, Any], true_run: dict[str, Any], output: Path) -> dict[str, Any]:
    false_rows = {(row["sample_id"], row["condition"]): row for row in false_run["rows"]}
    true_rows = {(row["sample_id"], row["condition"]): row for row in true_run["rows"]}
    if set(false_rows) != set(true_rows):
        raise AssertionError("benchmark true/false row identities differ")
    comparisons = []
    for key in sorted(false_rows):
        left = false_rows[key]
        right = true_rows[key]
        left_path = Path(left["array_npz"])
        right_path = Path(right["array_npz"])
        with np.load(left_path, allow_pickle=False) as a, np.load(right_path, allow_pickle=False) as b:
            metrics = {}
            for name in (
                "mel_float32", "z_float32", "syncnet_audio_units_float64",
                "audio_embeddings_float32", "lip_embeddings_float32",
                "cross_distance_raw_float32", "decoder_pixels_float32", "decoder_pixels_uint8",
            ):
                x, y = a[name], b[name]
                delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                metrics[name] = {
                    "shape_equal": x.shape == y.shape,
                    "exact": bool(np.array_equal(x, y)),
                    "max_abs_difference": float(delta.max(initial=0.0)),
                    "differing_elements": int(np.count_nonzero(x != y)),
                }
        comparisons.append({
            "sample_id": key[0], "condition": key[1],
            "false_pixel_sha256": left["artifact_sha256"]["video_bgr_pixels"],
            "true_pixel_sha256": right["artifact_sha256"]["video_bgr_pixels"],
            "pixels_exact": left["artifact_sha256"]["video_bgr_pixels"] == right["artifact_sha256"]["video_bgr_pixels"],
            "arrays": metrics,
        })
    result = {
        "status": "complete",
        "classification": "engineering-only cudnn benchmark sensitivity; first two CAL rows; not an evaluation result",
        "row_count_per_flag": len(false_rows),
        "requested_flags": [False, True],
        "comparisons": comparisons,
    }
    _json(output, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing engineering run directory: {out}")
    manifest_path = Path(args.input_manifest).resolve()
    tts_path = Path(args.tts_pair_manifest).resolve()
    rows, manifests = _records(manifest_path, tts_path)
    code_hashes = _runtime_code_hashes()
    protocol = {
        "schema_version": 1,
        "classification": "engineering-only cudnn benchmark sensitivity; no fitted axes and no evaluation effects",
        "input_manifest": {"path": str(manifest_path), "sha256": _sha(manifest_path)},
        "tts_pair_manifest": {"path": str(tts_path), "sha256": _sha(tts_path)},
        "rows": [{key: row.get(key) for key in ("id", "speaker", "text", "split", "ref_sha256")} for row in rows],
        "conditions": ["natural_raw", "natural_fixed", "tts_raw", "tts_fixed"],
        "faces": ["primary"],
        "benchmark_values": [False, True],
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "frontend": "Wav2Lip mel chunks; SyncNet official JPEG/OpenCV crop and float64 integer-amplitude MFCC path",
        "code_sha256": code_hashes,
        "wav2lip_checkpoint_sha256": _sha(Path(args.wav2lip_checkpoint).resolve()),
        "syncnet_checkpoint_sha256": _sha(Path(args.syncnet_checkpoint).resolve()),
        "primary_face_sha256": _sha(Path(args.primary_face).resolve()),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    serialized = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    protocol["protocol_sha256"] = hashlib.sha256(serialized).hexdigest()
    _json(out / "protocol.json", protocol)
    _json(out / "code_seal.json", {"code_sha256": code_hashes, "protocol_sha256": protocol["protocol_sha256"]})
    _json(out / "attempts.json", {"attempts": [{"attempt": 1, "status": "running", "started_at_utc": protocol["started_at_utc"], "code_sha256": code_hashes}]})
    started = time.perf_counter()
    results = {}
    try:
        for flag in (False, True):
            results[flag] = _run_flag(args, rows, flag, protocol)
        _compare(results[False], results[True], out / "comparison.json")
    except Exception as exc:
        _json(out / "attempts.json", {
            "attempts": [{
                "attempt": 1, "status": "failed", "started_at_utc": protocol["started_at_utc"],
                "elapsed_seconds": time.perf_counter() - started,
                "code_sha256": code_hashes, "error_type": type(exc).__name__, "error": str(exc)[:2000],
            }]
        })
        _json(out / "run_status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]})
        raise
    _json(out / "attempts.json", {
        "attempts": [{
            "attempt": 1, "status": "complete", "started_at_utc": protocol["started_at_utc"],
            "elapsed_seconds": time.perf_counter() - started, "code_sha256": code_hashes,
        }]
    })
    _json(out / "run_status.json", {
        "status": "complete", "classification": protocol["classification"],
        "elapsed_seconds": time.perf_counter() - started,
        "false_run": str(out / "benchmark_false"), "true_run": str(out / "benchmark_true"),
        "comparison": str(out / "comparison.json"),
    })
    return {"status": "complete", "comparison_path": str(out / "comparison.json"), "rows_per_flag": len(results[False]["rows"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--tts-pair-manifest", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wav2lip-batch-size", type=int, default=64)
    parser.add_argument("--syncnet-batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, separators=(",", ":")))
