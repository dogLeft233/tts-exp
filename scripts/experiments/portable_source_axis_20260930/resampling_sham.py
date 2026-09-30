"""Frozen natural 16->24->16 resampling sham, primary face, four cells.

This producer only creates raw audio/video/embedding/distance artifacts. It does
not summarize cohort effects. The four cells are q00=V_N/A_N, q10=V_S/A_N,
q01=V_N/A_S, q11=V_S/A_S, where S is the length-preserved resampling output.
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

CONDITIONS = {
    "q00": ("V_RAW_N", "A_RAW_N"),
    "q10": ("V_RAW_S", "A_RAW_N"),
    "q01": ("V_RAW_N", "A_RAW_S"),
    "q11": ("V_RAW_S", "A_RAW_S"),
}


def _sha(path: Path) -> str:
    return scoring.sha256_file(path)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append(path: Path, value: Any) -> None:
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


def _read_wav(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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


def _runtime_code_hashes() -> dict[str, str]:
    paths = {
        "resampling_sham.py": Path(__file__).resolve(),
        "evaluate.py": Path(evaluate.__file__).resolve(),
        "ops.py": Path(ops.__file__).resolve(),
        "scoring.py": Path(scoring.__file__).resolve(),
        "wav2lip.py": Path(wav2lip.__file__).resolve(),
    }
    return {name: _sha(path) for name, path in paths.items()}


def _make_plan(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = []
    for ordered_index, row in enumerate(source_rows):
        sample = row["id"]
        for condition, (visual_id, audio_id) in CONDITIONS.items():
            plan.append({
                "ordered_eval_index": ordered_index,
                "row_id": f"{sample}__primary__{condition}",
                "sample_id": sample,
                "speaker_id": row["speaker"],
                "split": "evaluation",
                "family": "resampling_sham",
                "condition": condition,
                "face_id": "primary",
                "visual_id": visual_id,
                "audio_id": audio_id,
                "status": "planned",
            })
    return plan


def _load_inputs(
    input_manifest_path: Path,
    resampling_manifest_path: Path,
    resampling_audio_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    resampled = json.loads(resampling_manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in source["records"] if row.get("split") == "evaluation"]
    sham_by_id = {row["id"]: row for row in resampled["records"] if row.get("split") == "evaluation"}
    if len(rows) != 80 or len(sham_by_id) != 80 or len({row["speaker"] for row in rows}) != 40:
        raise ValueError(f"expected evaluation 80 from 40 speakers: source={len(rows)} sham={len(sham_by_id)}")
    for row in rows:
        sham = sham_by_id.get(row["id"])
        if sham is None or sham.get("speaker") != row["speaker"] or sham.get("source_natural16_file_sha256") != row.get("ref_sha256"):
            raise ValueError(f"resampling manifest identity/source WAV mismatch for {row['id']}")
        path = resampling_audio_root / "audio" / f"{row['id']}_length_preserved.wav"
        expected = sham["length_preserved_audio"]["file_sha256"]
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"missing/hash-mismatched length-preserved audio for {row['id']}: {path}")
    return rows, {"source": source, "resampling": resampled, "resampling_by_id": sham_by_id}


def _instrument(events: list[dict[str, Any]]) -> list[tuple[Any, str, Any]]:
    originals = []
    for module, name in (
        (wav2lip, "cache_face_features"),
        (wav2lip, "encode_latents"),
        (wav2lip, "decode_latents"),
        (scoring, "lip_embeddings_from_frames"),
        (scoring, "audio_embeddings_from_syncnet_units"),
        (evaluate, "_cross_distance_matrix"),
    ):
        original = getattr(module, name)
        originals.append((module, name, original))

        def wrapped(*args, __name=name, __original=original, **kwargs):
            # Ensure the frozen flag after detector is true at every forward call.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            events.append({"operation": __name, "flags_at_call": _flags(), "time_monotonic": time.monotonic()})
            return __original(*args, **kwargs)

        setattr(module, name, wrapped)
    return originals


def _restore_instrument(originals: list[tuple[Any, str, Any]]) -> None:
    for module, name, original in originals:
        setattr(module, name, original)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing sham run directory: {out}")
    input_manifest_path = Path(args.input_manifest).resolve()
    resampling_manifest_path = Path(args.resampling_manifest).resolve()
    resampling_audio_root = Path(args.resampling_audio_root).resolve()
    calibration_summary_path = Path(args.calibration_summary).resolve(strict=True)
    calibration = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    anchor = calibration.get("global_anchor", {})
    anchor_k = int(anchor.get("selected_k", 10_000))
    anchor_column = int(anchor.get("anchor_column", -1))
    official_offset = int(anchor.get("official_syncnet_offset", 10_000))
    if (anchor_k, anchor_column, official_offset) != (3, 18, -3):
        raise ValueError(
            "sham requires the frozen CAL anchor k=+3, column=18, official offset=-3; "
            f"got k={anchor_k}, column={anchor_column}, official_offset={official_offset}"
        )
    source_rows, input_records = _load_inputs(input_manifest_path, resampling_manifest_path, resampling_audio_root)
    plan = _make_plan(source_rows)
    plan_bytes = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in plan)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    plan_path = out / "evaluation_rows.jsonl"
    plan_path.write_bytes(plan_bytes)

    protocol = {
        "schema_version": 1,
        "classification": "sealed engineering sham producer; source WAV natural16 vs length-preserved 16->24->16; no cohort effect summary",
        "input_manifest": {"path": str(input_manifest_path), "sha256": _sha(input_manifest_path)},
        "resampling_manifest": {"path": str(resampling_manifest_path), "sha256": _sha(resampling_manifest_path)},
        "resampling_audio_root": str(resampling_audio_root),
        "evaluation_count": 80,
        "speaker_count": 40,
        "face_id": "primary",
        "conditions": CONDITIONS,
        "calibration_summary": {"path": str(calibration_summary_path), "sha256": _sha(calibration_summary_path)},
        "frozen_anchor_k": anchor_k,
        "frozen_anchor_column": anchor_column,
        "frozen_anchor_official_syncnet_offset": official_offset,
        "vshift": 15,
        "supports": ["full", "valid", "guard20", "guard15"],
        "audio_domain": "raw PCM16; SyncNet units are exact float64 PCM*32768, no rounding/clipping",
        "visual_domain": "raw Wav2Lip natural waveform or length-preserved resampling waveform",
        "post_detection_flags": {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True},
        "cross_cells": {
            "q00": "V_RAW_N/A_RAW_N",
            "q10": "V_RAW_S/A_RAW_N",
            "q01": "V_RAW_N/A_RAW_S",
            "q11": "V_RAW_S/A_RAW_S",
        },
        "row_manifest_sha256": plan_sha,
        "planned_rows": len(plan),
        "code_sha256": _runtime_code_hashes(),
        "wav2lip_checkpoint_sha256": _sha(Path(args.wav2lip_checkpoint).resolve()),
        "syncnet_checkpoint_sha256": _sha(Path(args.syncnet_checkpoint).resolve()),
        "primary_face_sha256": _sha(Path(args.primary_face).resolve()),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    protocol_canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    protocol["protocol_sha256"] = hashlib.sha256(protocol_canonical).hexdigest()
    _json(out / "evaluation_protocol.json", protocol)
    _json(out / "code_seal.json", {"code_sha256": protocol["code_sha256"], "protocol_sha256": protocol["protocol_sha256"]})
    _json(out / "attempts.json", {"attempts": [{"attempt": 1, "status": "running", "started_at_utc": protocol["started_at_utc"], "code_sha256": protocol["code_sha256"]}]})
    _json(out / "run_status.json", {"status": "running", "planned_samples": 80, "planned_rows": 320, "completed_samples": 0})

    torch.manual_seed(20261001)
    np.random.seed(20261001)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    frontend = wav2lip.install_librosa_compatibility_shim()
    wav2lip_root = Path(args.wav2lip_root).resolve()
    syncnet_root = Path(args.syncnet_root).resolve()
    model, model_meta = wav2lip.load_model(wav2lip_root, args.wav2lip_checkpoint, device=device)
    syncnet, syncnet_meta = scoring.load_syncnet(syncnet_root, args.syncnet_checkpoint, device=device)
    face_path = Path(args.primary_face).resolve()
    frame = cv2.imread(str(face_path))
    if frame is None:
        raise RuntimeError(f"cannot read primary face: {face_path}")
    flags_before_detection = _flags()
    face_meta = wav2lip.detect_static_face(frame, wav2lip_root, device=device)
    flags_after_detection = _flags()
    # Reassert the declared runtime settings after static face detection.
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    flags_after_override = _flags()
    face_meta.update({"path": str(face_path), "sha256": _sha(face_path), "width": int(frame.shape[1]), "height": int(frame.shape[0])})
    _json(out / "faces_primary.json", {
        "face_id": "primary", "face": face_meta,
        "flags_before_detection": flags_before_detection,
        "flags_after_detection": flags_after_detection,
        "flags_after_explicit_override": flags_after_override,
        "wav2lip_checkpoint_sha256": protocol["wav2lip_checkpoint_sha256"],
        "code_sha256": protocol["code_sha256"],
    })

    events: list[dict[str, Any]] = []
    originals = _instrument(events)
    failed_samples = []
    completed_samples = 0
    try:
        for ordered_index, row in enumerate(source_rows):
            sample_id = row["id"]
            speaker = row["speaker"]
            utterance = sample_id.split("_", 1)[1]
            sample_started = time.perf_counter()
            try:
                natural_path = Path(args.input_root).resolve() / "audio" / speaker / f"{utterance}.wav"
                sham_path = resampling_audio_root / "audio" / f"{sample_id}_length_preserved.wav"
                natural_pcm, natural_float, natural_meta = _read_wav(natural_path)
                sham_pcm, sham_float, sham_meta = _read_wav(sham_path)
                if natural_meta["sha256"] != row.get("ref_sha256"):
                    raise ValueError(f"natural source file hash differs from frozen input manifest for {sample_id}")
                sham_record = input_records["resampling_by_id"][sample_id]
                sham_expected = sham_record["length_preserved_audio"]
                if sham_meta["sha256"] != sham_expected["file_sha256"] or sham_meta["samples"] != sham_expected["decoded_frames"]:
                    raise ValueError(f"length-preserved audio does not match frozen decoded PCM for {sample_id}")
                expected_natural_sha = sham_record["source_natural16_file_sha256"]
                expected_natural_frames = int(sham_record["reference_natural_frames"])
                expected_sham_frames = int(sham_expected["decoded_frames"])
                length_audit = {
                    "natural_source_sha256": natural_meta["sha256"],
                    "sham_source_sha256": sham_meta["sha256"],
                    "manifest_natural_source_sha256": expected_natural_sha,
                    "manifest_sham_source_sha256": sham_expected["file_sha256"],
                    "natural_decoded_frames": int(natural_pcm.size),
                    "sham_decoded_frames": int(sham_pcm.size),
                    "manifest_reference_natural_frames": expected_natural_frames,
                    "manifest_sham_decoded_frames": expected_sham_frames,
                    "equal_length": bool(natural_pcm.size == sham_pcm.size),
                }
                if expected_natural_sha != row.get("ref_sha256"):
                    raise ValueError(f"resampling manifest's raw natural SHA differs from frozen input for {sample_id}: {length_audit}")
                if not (
                    natural_meta["sha256"] == expected_natural_sha
                    and natural_pcm.size == sham_pcm.size
                    and natural_pcm.size == expected_natural_frames
                    and sham_pcm.size == expected_sham_frames
                ):
                    raise ValueError(f"natural/sham decoded lengths differ from each other or frozen manifest for {sample_id}: {length_audit}")
                natural_units, natural_units_meta = ops.pcm_to_syncnet_integer_units(natural_float)
                sham_units, sham_units_meta = ops.pcm_to_syncnet_integer_units(sham_float)
                torch.backends.cudnn.benchmark = False
                natural_mel, natural_chunks, natural_starts, natural_front = wav2lip.mel_chunks(natural_float, wav2lip_root)
                natural_z = wav2lip.encode_latents(model, natural_chunks, device=device, batch_size=args.wav2lip_batch_size)
                torch.backends.cudnn.benchmark = False
                sham_mel, sham_chunks, sham_starts, sham_front = wav2lip.mel_chunks(sham_float, wav2lip_root)
                sham_z = wav2lip.encode_latents(model, sham_chunks, device=device, batch_size=args.wav2lip_batch_size)
                natural_windows = min(len(natural_z), len(natural_units) // 640) - 5
                sham_windows = min(len(sham_z), len(sham_units) // 640) - 5
                common_windows = min(natural_windows, sham_windows)
                if common_windows < 1:
                    raise ValueError(f"no shared SyncNet windows for {sample_id}: N={natural_windows}, S={sham_windows}")
                torch.backends.cudnn.benchmark = False
                natural_mfcc, natural_audio, natural_audio_meta = evaluate._audio_forward(
                    syncnet, natural_units, device=device, batch_size=args.syncnet_batch_size,
                )
                torch.backends.cudnn.benchmark = False
                sham_mfcc, sham_audio, sham_audio_meta = evaluate._audio_forward(
                    syncnet, sham_units, device=device, batch_size=args.syncnet_batch_size,
                )
                needed_mfcc_frames = (common_windows - 1) * 4 + 20
                natural_mfcc = natural_mfcc[:, :needed_mfcc_frames]
                sham_mfcc = sham_mfcc[:, :needed_mfcc_frames]
                natural_audio = natural_audio[:common_windows]
                sham_audio = sham_audio[:common_windows]
                actual_batch_sizes = {args.wav2lip_batch_size}
                for length in (len(natural_z), len(sham_z)):
                    tail = length % args.wav2lip_batch_size
                    if tail:
                        actual_batch_sizes.add(tail)
                torch.backends.cudnn.benchmark = False
                face_cache = wav2lip.cache_face_features(
                    model, frame, face_meta["generation_box_xyxy"], sorted(actual_batch_sizes), device=device,
                )
                visual_n, meta_n = evaluate._load_or_make_visual(
                    out, sample_id, "primary", "V_RAW_N", natural_z, frame, face_meta, model,
                    face_cache, syncnet, argparse.Namespace(
                        ffmpeg=args.ffmpeg, wav2lip_batch_size=args.wav2lip_batch_size,
                        syncnet_batch_size=args.syncnet_batch_size,
                    ), device,
                )
                visual_s, meta_s = evaluate._load_or_make_visual(
                    out, sample_id, "primary", "V_RAW_S", sham_z, frame, face_meta, model,
                    face_cache, syncnet, argparse.Namespace(
                        ffmpeg=args.ffmpeg, wav2lip_batch_size=args.wav2lip_batch_size,
                        syncnet_batch_size=args.syncnet_batch_size,
                    ), device,
                )
                visual_n = visual_n[:common_windows]
                visual_s = visual_s[:common_windows]
                cells = {
                    "q00": (visual_n, natural_audio),
                    "q10": (visual_s, natural_audio),
                    "q01": (visual_n, sham_audio),
                    "q11": (visual_s, sham_audio),
                }
                support = evaluate._support_indices(common_windows)
                context_path = out / "samples" / f"{sample_id}_context.npz"
                context_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    context_path,
                    natural_pcm16=natural_pcm,
                    sham_pcm16=sham_pcm,
                    natural_raw_float32=natural_float,
                    sham_raw_float32=sham_float,
                    natural_mel_raw_float32=natural_mel,
                    sham_mel_raw_float32=sham_mel,
                    natural_mel_starts_int32=np.asarray(natural_starts, dtype=np.int32),
                    sham_mel_starts_int32=np.asarray(sham_starts, dtype=np.int32),
                    natural_z_raw_float32=natural_z,
                    sham_z_raw_float32=sham_z,
                    natural_mfcc_raw_float64=natural_mfcc,
                    sham_mfcc_raw_float64=sham_mfcc,
                    natural_audio_embeddings_raw_float32=natural_audio,
                    sham_audio_embeddings_raw_float32=sham_audio,
                    natural_lip_embeddings_raw_float32=visual_n,
                    sham_lip_embeddings_raw_float32=visual_s,
                    common_support_full_int32=support["full"],
                    common_support_guard20_int32=support["guard20"],
                    common_support_guard15_int32=support["guard15"],
                )
                context_sha = _sha(context_path)
                context_meta = {
                    "sample_id": sample_id, "speaker_id": speaker, "split": "evaluation",
                    "natural_input": natural_meta, "sham_input": sham_meta,
                    "natural_input_manifest_sha256": row["ref_sha256"],
                    "sham_input_manifest_sha256": sham_expected["file_sha256"],
                    "natural_frontend": natural_front, "sham_frontend": sham_front,
                    "length_audit": length_audit,
                    "natural_syncnet_audio": {**natural_units_meta, **natural_audio_meta},
                    "sham_syncnet_audio": {**sham_units_meta, **sham_audio_meta},
                    "natural_visual": meta_n, "sham_visual": meta_s,
                    "common_window_count": common_windows,
                    "actual_flags_at_forwards": sorted({json.dumps(e["flags_at_call"], sort_keys=True) for e in events}),
                    "context_npz": {"path": str(context_path), "sha256": context_sha, "size_bytes": context_path.stat().st_size},
                }
                _json(out / "samples" / f"{sample_id}_context.json", context_meta)

                for condition, (visual, audio) in cells.items():
                    torch.backends.cudnn.benchmark = False
                    cross, to_zero = evaluate._cross_distance_matrix(visual, audio, device)
                    raw_matrix = evaluate._lag_matrix_from_cross(cross, to_zero, common_windows, common_windows, vshift=15)
                    visual_unit = evaluate._row_unit(visual)
                    audio_unit = evaluate._row_unit(audio)
                    torch.backends.cudnn.benchmark = False
                    cross_unit, zero_unit = evaluate._cross_distance_matrix(visual_unit, audio_unit, device)
                    unit_matrix = evaluate._lag_matrix_from_cross(cross_unit, zero_unit, common_windows, common_windows, vshift=15)
                    score_path = out / "scores" / f"{sample_id}__{condition}.npz"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        score_path,
                        cross_distance_raw_float32=cross,
                        cross_distance_to_zero_raw_float32=to_zero,
                        lag_distance_raw_float32=raw_matrix,
                        cross_distance_unit_float32=cross_unit,
                        cross_distance_to_zero_unit_float32=zero_unit,
                        lag_distance_unit_float32=unit_matrix,
                        visual_embeddings_raw_float32=visual,
                        audio_embeddings_raw_float32=audio,
                        visual_embeddings_unit_float32=visual_unit,
                        audio_embeddings_unit_float32=audio_unit,
                        common_support_full_int32=support["full"],
                        common_support_guard20_int32=support["guard20"],
                        common_support_guard15_int32=support["guard15"],
                    )
                    score_sha = _sha(score_path)
                    fixed_anchor_column = anchor_column
                    result = {
                        **plan[ordered_index * 4 + list(CONDITIONS).index(condition)],
                        "status": "complete",
                        "context_npz": str(context_path.relative_to(out)),
                        "context_sha256": context_sha,
                        "visual_npz": str((out / "visuals" / "primary" / f"{sample_id}__{('V_RAW_N' if visual is visual_n else 'V_RAW_S')}.npz").relative_to(out)),
                        "visual_sha256": meta_n["feature_npz_sha256"] if visual is visual_n else meta_s["feature_npz_sha256"],
                        "score_npz": str(score_path.relative_to(out)),
                        "score_sha256": score_sha,
                        "common_window_count": common_windows,
                        "fixed_anchor_k": anchor_k,
                        "fixed_anchor_offset_frames": official_offset,
                        "fixed_anchor_column": fixed_anchor_column,
                        "raw_support_metrics": evaluate._metric_supports(raw_matrix, fixed_anchor_column),
                        "unit_support_metrics": evaluate._metric_supports(unit_matrix, fixed_anchor_column),
                        "flags_at_forward": _flags(),
                    }
                    _append(out / "evaluation_results.jsonl", result)
                completed_samples += 1
                _append(out / "attempts.jsonl", {
                    "attempt": 1, "sample_id": sample_id, "status": "complete",
                    "elapsed_seconds": time.perf_counter() - sample_started,
                    "completed_samples": completed_samples, "planned_samples": 80,
                })
                _json(out / "run_status.json", {
                    "status": "running", "planned_samples": 80, "planned_rows": 320,
                    "completed_samples": completed_samples, "failed_samples": len(failed_samples),
                    "last_sample_id": sample_id,
                })
            except Exception as exc:
                failed = {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "traceback_tail": traceback.format_exc()[-4000:],
                    "elapsed_seconds": time.perf_counter() - sample_started,
                }
                failed_samples.append(failed)
                _append(out / "attempts.jsonl", {"attempt": 1, "status": "failed", **failed})
                for condition, (visual_id, audio_id) in CONDITIONS.items():
                    _append(out / "evaluation_results.jsonl", {
                        **plan[ordered_index * 4 + list(CONDITIONS).index(condition)],
                        "status": "failed", "error_type": failed["error_type"], "error": failed["error"],
                        "traceback_tail": failed["traceback_tail"],
                        "visual_id": visual_id, "audio_id": audio_id,
                    })
                _json(out / "run_status.json", {
                    "status": "running", "planned_samples": 80, "planned_rows": 320,
                    "completed_samples": completed_samples, "failed_samples": len(failed_samples),
                    "last_sample_id": sample_id,
                })
    finally:
        _restore_instrument(originals)

    final = {
        "status": "complete_with_failures" if failed_samples else "complete",
        "classification": protocol["classification"],
        "planned_samples": 80,
        "completed_samples": completed_samples,
        "failed_samples": len(failed_samples),
        "planned_rows": 320,
        "completed_rows": sum(1 for line in (out / "evaluation_results.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line).get("status") == "complete"),
        "failed_rows": sum(1 for line in (out / "evaluation_results.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line).get("status") == "failed"),
        "protocol_sha256": protocol["protocol_sha256"],
        "row_manifest_sha256": plan_sha,
        "code_sha256": protocol["code_sha256"],
        "runtime": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name(),
            "frontend_compatibility": frontend, "wav2lip_model": model_meta, "syncnet_model": syncnet_meta,
        },
        "face_flags": {"before_detection": flags_before_detection, "after_detection": flags_after_detection, "after_override": flags_after_override},
        "forward_events": events,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _json(out / "run_status.json", final)
    _json(out / "all_files_sha256.json", {
        str(path.relative_to(out)): _sha(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"
    })
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--resampling-manifest", required=True)
    parser.add_argument("--resampling-audio-root", required=True)
    parser.add_argument("--calibration-summary", required=True)
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
