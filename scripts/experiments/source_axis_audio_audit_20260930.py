#!/usr/bin/env python3
"""Integrity, round-trip resampling, and calibration-only audio descriptors.

This utility consumes the sealed AISHELL source manifest, Qwen result metadata,
and pair-audio manifest. It never changes those inputs. ``audit-resample``
creates derived PCM and a corrected header/decode audit. ``describe-cal`` reads
only calibration waveforms and writes outcome-blind physical descriptors.

Runtime dependencies: numpy, soundfile; scipy and pyworld are optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "runs/recovered_20260930/local/qwen_source_axis/hf_selected"
DEFAULT_INPUT = DEFAULT_ROOT / "input_manifest.json"
DEFAULT_PAIR = DEFAULT_ROOT / "tts_outputs/tts_pair_manifest.json"
DEFAULT_TTS_META = DEFAULT_ROOT / "tts_outputs/tts_meta.json"
DEFAULT_OUT = DEFAULT_ROOT / "audio_integrity_20260930"
NATIVE_RATE = 24_000
CANONICAL_RATE = 16_000
FRAME_MS = 25
HOP_MS = 10
MEL_BANDS = 40
MEL_POWER_FLOOR = 1e-10
FIXED_TARGET_RMS = 0.0376838172675746
LEVEL_STEP_DB = 3.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(payload: dict[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json(clean)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_path(raw: str, repo_root: Path, override_root: Path | None = None, expected_hash: str | None = None) -> Path:
    # The sealed manifests were authored on Windows and may carry drive-letter
    # paths. Normalize separators so basename overrides also work on Linux.
    candidate = Path(str(raw).replace("\\", "/")).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((repo_root / candidate).resolve())
        candidates.append((Path.cwd() / candidate).resolve())
    if override_root is not None:
        override_root = override_root.resolve()
        candidates.extend((override_root / candidate.name, override_root / candidate.stem / candidate.name))
        candidates.extend(p for p in override_root.rglob(candidate.name) if p.is_file())
    seen: set[str] = set()
    for item in candidates:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not item.is_file():
            continue
        if expected_hash is None or sha256_file(item) == expected_hash:
            return item.resolve()
    raise FileNotFoundError(f"audio path missing or hash mismatch: {candidate.name}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_audio_libs() -> tuple[Any, Any]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("numpy and soundfile are required; run with `uv run --with numpy --with soundfile`") from exc
    return np, sf


def read_wav_header_frames(path: Path) -> dict[str, Any]:
    """Read the RIFF-declared count separately; streamed WAVs may use a sentinel."""
    try:
        with wave.open(str(path), "rb") as src:
            return {
                "declared_frames": int(src.getnframes()),
                "declared_sample_rate_hz": int(src.getframerate()),
                "declared_channels": int(src.getnchannels()),
                "declared_sample_width_bytes": int(src.getsampwidth()),
                "declared_compression": src.getcomptype(),
            }
    except (wave.Error, EOFError, OSError) as exc:
        return {"declared_frames": None, "header_error_type": type(exc).__name__}


def decode_info(path: Path, np: Any, sf: Any) -> tuple[dict[str, Any], Any, Any]:
    info = sf.info(str(path))
    pcm, rate = sf.read(str(path), dtype="int16", always_2d=True)
    if pcm.size == 0 or pcm.shape[0] <= 0:
        raise ValueError(f"decoded empty audio: {path.name}")
    decoded_frames = int(pcm.shape[0])
    hdr = read_wav_header_frames(path)
    mismatch = hdr.get("declared_frames") is not None and int(hdr["declared_frames"]) != decoded_frames
    payload_pcm_hash = hashlib.sha256(pcm.tobytes(order="C")).hexdigest()
    result = {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        **hdr,
        "decoded_frames": decoded_frames,
        "decoded_sample_rate_hz": int(rate),
        "decoded_channels": int(pcm.shape[1]),
        "decoded_subtype": str(info.subtype),
        "decoded_duration_s": round(decoded_frames / int(rate), 9),
        "decoded_pcm16_sha256": payload_pcm_hash,
        "declared_vs_payload_mismatch": bool(mismatch),
        "decoded_payload_bytes": int(pcm.nbytes),
    }
    return result, pcm, pcm.astype(np.float32) / 32768.0


def load_cohort(args: argparse.Namespace, split: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    source = load_json(args.input_manifest.resolve())
    pairs = load_json(args.pair_manifest.resolve())
    tts_meta = load_json(args.tts_meta.resolve())
    if source.get("manifest_type") != "qwen_tts_pair_input_manifest":
        raise ValueError("unexpected input manifest type")
    if source.get("manifest_sha256") and source["manifest_sha256"] != fingerprint(source):
        raise ValueError("input manifest fingerprint mismatch")
    if pairs.get("manifest_type") != "aishell1_qwen_vc_pair_audio_manifest":
        raise ValueError("unexpected pair manifest type")
    if pairs.get("manifest_sha256") and pairs["manifest_sha256"] != fingerprint(pairs):
        raise ValueError("pair manifest fingerprint mismatch")
    if not tts_meta.get("complete") or int(tts_meta.get("samples_ok", -1)) != 100 or int(tts_meta.get("samples_failed", -1)) != 0:
        raise ValueError("TTS batch metadata is not complete 100/100")
    src_rows = source.get("records", [])
    pair_rows = pairs.get("records", [])
    result_rows = tts_meta.get("results", {})
    src_by_id = {str(row["id"]): row for row in src_rows}
    pair_by_id = {str(row["id"]): row for row in pair_rows}
    if len(src_by_id) != 100 or set(src_by_id) != set(pair_by_id) or set(src_by_id) != set(result_rows):
        raise ValueError("source, pair, and TTS manifests do not cover the same 100 IDs")
    src_digest = sha256_file(args.input_manifest.resolve())
    declared_src_sha = pairs.get("source_manifest", {}).get("raw_sha256")
    if declared_src_sha and declared_src_sha != src_digest:
        raise ValueError("pair manifest is bound to a different source manifest file")
    frozen_digest = pairs.get("run_frozen_input_manifest", {}).get("embedded_fingerprint")
    if tts_meta.get("input_manifest_sha256") != frozen_digest:
        raise ValueError("TTS metadata is not bound to the pair manifest frozen input identity")
    ordered_ids = [str(row["id"]) for row in src_rows]
    for sample_id in ordered_ids:
        s, p, t = src_by_id[sample_id], pair_by_id[sample_id], result_rows[sample_id]
        ident = p.get("input_identity", {})
        expected_fields = {
            "id": s.get("id"), "speaker": s.get("speaker"), "text": s.get("text"),
            "split": s.get("split"), "ref_sha256": s.get("ref_sha256"),
        }
        if any(ident.get(k) != v for k, v in expected_fields.items()):
            raise ValueError(f"input identity mismatch for {sample_id}")
        if t.get("status") != "ok" or t.get("provider_audio_sha256") != p["qwen_provider_native_24k"].get("sha256") or t.get("canonical_audio_sha256") != p["qwen_canonical_16k"].get("sha256"):
            raise ValueError(f"TTS file hash binding mismatch for {sample_id}")
    rows = [src_by_id[sample_id] for sample_id in ordered_ids if split is None or src_by_id[sample_id].get("split") == split]
    return source, pairs, tts_meta, rows


def resolve_row_files(row: dict[str, Any], pair_row: dict[str, Any], tts_row: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    root = args.repo_root.resolve()
    return {
        "natural16": resolve_path(str(row["ref_path"]), root, args.natural_root, str(row["ref_sha256"])),
        "tts24": resolve_path(str(tts_row["provider_audio"]), root, args.tts24_root, str(tts_row["provider_audio_sha256"])),
        "tts16": resolve_path(str(tts_row["canonical_16k_audio"]), root, args.tts16_root, str(tts_row["canonical_audio_sha256"])),
    }


def run_ffmpeg(executable: Path, source: Path, target: Path, rate: int) -> list[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "-y", "-v", "error", "-i", str(source), "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", str(target)]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not target.is_file():
        raise RuntimeError(f"ffmpeg failed for {source.name}; rc={result.returncode}")
    return command


def pcm_difference(np: Any, ref: Any, candidate: Any) -> dict[str, Any]:
    a = ref.astype(np.float64).reshape(-1)
    b = candidate.astype(np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n <= 0:
        raise ValueError("empty PCM comparison")
    a, b = a[:n], b[:n]
    d = b - a
    ref_rms = float(np.sqrt(np.mean(a * a)))
    err_rms = float(np.sqrt(np.mean(d * d)))
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    return {
        "compared_samples": n,
        "reference_samples": int(len(ref)),
        "candidate_samples": int(len(candidate)),
        "length_delta_samples": int(len(candidate) - len(ref)),
        "rms_error_pcm_units": round(err_rms, 9),
        "rms_error_dbfs": round(20 * math.log10(max(err_rms / 32768.0, 1e-15)), 6),
        "max_abs_error_pcm_units": int(np.max(np.abs(d))),
        "mean_abs_error_pcm_units": round(float(np.mean(np.abs(d))), 9),
        "snr_db": round(20 * math.log10(max(ref_rms, 1e-15) / max(err_rms, 1e-15)), 6),
        "pearson_r": round(corr, 9) if corr is not None else None,
        "bit_identical_common_length": bool(np.array_equal(a, b)),
    }


def preserve_roundtrip_frame_count(np: Any, roundtrip_pcm: Any, reference_frames: int) -> tuple[Any, int]:
    """Retain the source clock, trimming only the known single-sample excess."""
    length_delta = int(len(roundtrip_pcm) - int(reference_frames))
    if reference_frames <= 0 or length_delta not in (0, 1):
        raise ValueError(f"roundtrip length delta must be 0 or +1 sample, got {length_delta}")
    trim_right = max(0, length_delta)
    return np.asarray(roundtrip_pcm)[:reference_frames], trim_right


def frontend_difference(np: Any, reference: Any, candidate: Any, sr: int) -> dict[str, Any]:
    ref_mel, ref_log, ref_mfcc = feature_matrices(np, reference, sr)
    cand_mel, cand_log, cand_mfcc = feature_matrices(np, candidate, sr)
    frames = min(ref_mel.shape[1], cand_mel.shape[1])
    mel_delta = cand_mel[:, :frames] - ref_mel[:, :frames]
    log_delta = cand_log[:, :frames] - ref_log[:, :frames]
    mfcc_delta = cand_mfcc[:, :frames] - ref_mfcc[:, :frames]
    return {
        "frontend_spec": "40-band mel power and custom C0=mean(ln-mel), C1-C12 orthonormal DCT-II; 25ms Hann/10ms hop/512 FFT",
        "reference_frames": int(ref_mel.shape[1]),
        "candidate_frames": int(cand_mel.shape[1]),
        "compared_frames": int(frames),
        "mel_power_max_abs_difference": round(float(np.max(np.abs(mel_delta))), 12),
        "mel_power_mean_abs_difference": round(float(np.mean(np.abs(mel_delta))), 12),
        "logmel_ln_max_abs_difference": round(float(np.max(np.abs(log_delta))), 12),
        "logmel_ln_mean_abs_difference": round(float(np.mean(np.abs(log_delta))), 12),
        "mfcc_max_abs_difference_by_coefficient": [round(float(v), 12) for v in np.max(np.abs(mfcc_delta), axis=1)],
        "mfcc_mean_abs_difference_by_coefficient": [round(float(v), 12) for v in np.mean(np.abs(mfcc_delta), axis=1)],
        "embedding_difference": {"status": "not_computed_here", "reason": "Wav2Lip audio encoder forward is delegated to the remote experiment runner"},
    }


def audit_resample(args: argparse.Namespace) -> int:
    np, sf = get_audio_libs()
    source, pairs, tts_meta, rows = load_cohort(args)
    if len(rows) != 100:
        raise ValueError("round-trip audit expects the frozen 100-record cohort")
    canonicalization = tts_meta["canonicalization"]
    expected_ffmpeg_sha = str(canonicalization.get("executable_sha256", ""))
    explicit = args.ffmpeg or canonicalization.get("executable_path")
    ffmpeg = Path(shutil.which(str(explicit)) or explicit).expanduser().resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError("the original synthesis ffmpeg binary is unavailable; pass its matching --ffmpeg path")
    ffmpeg_sha = sha256_file(ffmpeg)
    if expected_ffmpeg_sha and ffmpeg_sha != expected_ffmpeg_sha:
        raise ValueError("ffmpeg hash differs from the synthesis canonicalizer")
    version = subprocess.run([str(ffmpeg), "-version"], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()[0]
    outdir = args.outdir.resolve()
    audit_records: list[dict[str, Any]] = []
    roundtrip_records: list[dict[str, Any]] = []
    length_preserved_records: list[dict[str, Any]] = []
    pair_by_id = {str(x["id"]): x for x in pairs["records"]}
    meta_by_id = tts_meta["results"]
    for index, row in enumerate(rows, 1):
        sample_id = str(row["id"])
        pair_row, tts_row = pair_by_id[sample_id], meta_by_id[sample_id]
        files = resolve_row_files(row, pair_row, tts_row, args)
        decoded: dict[str, Any] = {}
        pcm: dict[str, Any] = {}
        for key, path in files.items():
            decoded[key], pcm[key], _ = decode_info(path, np, sf)
        expected_rates = {"natural16": CANONICAL_RATE, "tts24": NATIVE_RATE, "tts16": CANONICAL_RATE}
        for key, rate in expected_rates.items():
            if decoded[key]["decoded_sample_rate_hz"] != rate or decoded[key]["decoded_channels"] != 1 or decoded[key]["decoded_subtype"] != "PCM_16":
                raise ValueError(f"decoded format contract failed for {sample_id}:{key}")
        audit_records.append({
            "id": sample_id,
            "speaker": row["speaker"],
            "split": row["split"],
            "source_identity_sha256": pair_row["input_identity_sha256"],
            "natural16": decoded["natural16"],
            "qwen_provider24_original": decoded["tts24"],
            "qwen_canonical16": decoded["tts16"],
        })
        paths = {
            "natural24_stage": outdir / "roundtrip_24k" / "natural" / f"{sample_id}.wav",
            "natural16_roundtrip": outdir / "roundtrip_16k" / "natural" / f"{sample_id}.wav",
            "tts24_stage": outdir / "roundtrip_24k" / "tts_provider" / f"{sample_id}.wav",
            "tts16_roundtrip": outdir / "roundtrip_16k" / "tts_provider" / f"{sample_id}.wav",
        }
        commands = {
            "natural16_to_24": run_ffmpeg(ffmpeg, files["natural16"], paths["natural24_stage"], NATIVE_RATE),
            "natural24_to_16": run_ffmpeg(ffmpeg, paths["natural24_stage"], paths["natural16_roundtrip"], CANONICAL_RATE),
            "tts24_decode_to_pcm24": run_ffmpeg(ffmpeg, files["tts24"], paths["tts24_stage"], NATIVE_RATE),
            "tts24_to_16": run_ffmpeg(ffmpeg, paths["tts24_stage"], paths["tts16_roundtrip"], CANONICAL_RATE),
        }
        natural24, natural24_pcm, _ = decode_info(paths["natural24_stage"], np, sf)
        natural16_rt, natural16_pcm, _ = decode_info(paths["natural16_roundtrip"], np, sf)
        tts24_stage, tts24_pcm, _ = decode_info(paths["tts24_stage"], np, sf)
        tts16_rt, tts16_pcm, _ = decode_info(paths["tts16_roundtrip"], np, sf)
        reference_frames = int(len(pcm["natural16"]))
        length_delta = int(len(natural16_pcm) - reference_frames)
        try:
            natural16_length_preserved, trim_right = preserve_roundtrip_frame_count(np, natural16_pcm, reference_frames)
        except ValueError as exc:
            raise ValueError(f"natural roundtrip length differs by {length_delta} samples for {sample_id}; expected 0 or +1") from exc
        length_preserved_path = outdir / "roundtrip_16k_length_preserved" / "natural" / f"{sample_id}.wav"
        length_preserved_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(length_preserved_path), natural16_length_preserved, CANONICAL_RATE, subtype="PCM_16", endian="LITTLE")
        length_preserved_info, length_preserved_pcm, _ = decode_info(length_preserved_path, np, sf)
        if len(length_preserved_pcm) != reference_frames:
            raise ValueError(f"length-preserved natural output has wrong frame count for {sample_id}")
        if not np.array_equal(length_preserved_pcm, natural16_length_preserved):
            raise ValueError(f"length-preserved natural PCM changed while writing {sample_id}")
        length_preserved_records.append({
            "id": sample_id,
            "speaker": row["speaker"],
            "split": row["split"],
            "source_identity_sha256": pair_row["input_identity_sha256"],
            "source_natural16_file_sha256": decoded["natural16"]["file_sha256"],
            "source_natural16_pcm16_sha256": decoded["natural16"]["decoded_pcm16_sha256"],
            "uncropped_roundtrip_file_sha256": natural16_rt["file_sha256"],
            "uncropped_roundtrip_pcm16_sha256": natural16_rt["decoded_pcm16_sha256"],
            "uncropped_roundtrip_frames": natural16_rt["decoded_frames"],
            "reference_natural_frames": reference_frames,
            "uncropped_length_delta_samples": length_delta,
            "operation": "right_trim_one_final_sample" if trim_right else "preserve_samples_unchanged",
            "trimmed_right_samples": trim_right,
            "padded_samples": 0,
            "time_warp_or_other_processing": False,
            "length_preserved_audio": length_preserved_info,
            "vs_original_natural16": pcm_difference(np, pcm["natural16"], length_preserved_pcm),
            "frontend_difference_vs_original_natural16": frontend_difference(np, pcm["natural16"].astype(np.float32) / 32768.0, length_preserved_pcm.astype(np.float32) / 32768.0, CANONICAL_RATE),
        })
        roundtrip_records.append({
            "id": sample_id,
            "speaker": row["speaker"],
            "split": row["split"],
            "natural_16k_to_24k_to_16k": {
                "natural24_stage": natural24,
                "natural16_roundtrip": natural16_rt,
                "commands": {"natural16_to_24": commands["natural16_to_24"], "natural24_to_16": commands["natural24_to_16"]},
                "vs_original_natural16": pcm_difference(np, pcm["natural16"], natural16_pcm),
                "frontend_difference_vs_original_natural16": frontend_difference(np, pcm["natural16"].astype(np.float32) / 32768.0, natural16_pcm.astype(np.float32) / 32768.0, CANONICAL_RATE),
                "intermediate_pcm16_wav_written_without_lossy_codec": True,
            },
            "qwen_native24k_to_pcm24k_to_16k": {
                "provider24_original_file_sha256": decoded["tts24"]["file_sha256"],
                "provider24_stage": tts24_stage,
                "canonical16_roundtrip": tts16_rt,
                "commands": {"provider24_to_pcm24": commands["tts24_decode_to_pcm24"], "provider24_to_16": commands["tts24_to_16"]},
                "provider24_original_pcm_matches_stage": bool(decoded["tts24"]["decoded_pcm16_sha256"] == tts24_stage["decoded_pcm16_sha256"]),
                "vs_official_canonical16": pcm_difference(np, pcm["tts16"], tts16_pcm),
                "frontend_difference_vs_official_canonical16": frontend_difference(np, pcm["tts16"].astype(np.float32) / 32768.0, tts16_pcm.astype(np.float32) / 32768.0, CANONICAL_RATE),
            },
        })
        print(f"AUDITED {index}/{len(rows)} {sample_id}", flush=True)
    common = {
        "schema_version": 1,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_manifest_path": str(args.input_manifest.resolve()),
        "source_manifest_file_sha256": sha256_file(args.input_manifest.resolve()),
        "source_manifest_fingerprint": source.get("manifest_sha256"),
        "pair_manifest_file_sha256": sha256_file(args.pair_manifest.resolve()),
        "pair_manifest_fingerprint": pairs.get("manifest_sha256"),
        "tts_meta_file_sha256": sha256_file(args.tts_meta.resolve()),
        "tts_meta_input_fingerprint": tts_meta.get("input_manifest_sha256"),
        "ffmpeg": {
            "path": str(ffmpeg), "sha256": ffmpeg_sha, "version_line": version,
            "settings": ["-y", "-v", "error", "-i", "<input>", "-ar", "<target_rate>", "-ac", "1", "-c:a", "pcm_s16le", "<output>"],
        },
        "cohort_counts": {"total": len(rows), "calibration": sum(x["split"] == "calibration" for x in rows), "evaluation": sum(x["split"] == "evaluation" for x in rows)},
    }
    corr = {**common, "manifest_type": "source_axis_audio_integrity_decode_audit", "records": audit_records}
    corr["manifest_sha256"] = fingerprint(corr)
    write_json(outdir / "audio_integrity_correction_manifest.json", corr)
    resample = {**common, "manifest_type": "source_axis_resampling_robustness", "input_rate_path": "natural 16k -> PCM16 24k -> PCM16 16k; Qwen native 24k -> PCM16 24k intermediate -> PCM16 16k", "records": roundtrip_records}
    resample["manifest_sha256"] = fingerprint(resample)
    write_json(outdir / "roundtrip_resampling_manifest.json", resample)
    length_preserved = {
        **common,
        "manifest_type": "source_axis_natural_resampling_length_preserved_derivatives",
        "status": "DERIVED_FROM_UNCROPPED_RESAMPLING_OUTPUTS; originals remain preserved",
        "parent_roundtrip_resampling_manifest_sha256": resample["manifest_sha256"],
        "rule": "Keep the original 16k frame count. If the uncropped 16->24->16 conversion has exactly one extra final sample, trim only that sample on the right. Otherwise retain all converted samples. Never pad or time-warp.",
        "records": length_preserved_records,
    }
    length_preserved["manifest_sha256"] = fingerprint(length_preserved)
    write_json(outdir / "length_preserved_resampling_manifest.json", length_preserved)
    print(json.dumps({"decoded_records": len(rows), "header_mismatches_raw24": sum(r["qwen_provider24_original"]["declared_vs_payload_mismatch"] for r in audit_records), "natural_length_preserved_records": len(length_preserved_records), "natural_right_trimmed_one_sample": sum(r["trimmed_right_samples"] == 1 for r in length_preserved_records), "outdir": str(outdir), "correction_manifest_sha256": corr["manifest_sha256"], "resampling_manifest_sha256": resample["manifest_sha256"], "length_preserved_manifest_sha256": length_preserved["manifest_sha256"]}, separators=(",", ":")))
    return 0


def make_frames(np: Any, x: Any, sr: int) -> Any:
    frame = int(round(FRAME_MS * sr / 1000))
    hop = int(round(HOP_MS * sr / 1000))
    if x.size == 0:
        return np.zeros((1, frame), dtype=np.float32)
    starts = np.arange(0, len(x), hop)
    padded = np.pad(x, (0, max(0, int(starts[-1] + frame - len(x)))))
    return np.stack([padded[start:start + frame] for start in starts]).astype(np.float32, copy=False)


def mel_filterbank(np: Any, sr: int, nfft: int, bands: int) -> Any:
    def hz_to_mel(freq: Any) -> Any:
        return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)

    def mel_to_hz(mel: Any) -> Any:
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), bands + 2))
    bins = np.floor((nfft + 1) * edges / sr).astype(int)
    filters = np.zeros((bands, nfft // 2 + 1), dtype=np.float64)
    for m in range(1, bands + 1):
        left, center, right = bins[m - 1:m + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, filters.shape[1])):
            filters[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, filters.shape[1])):
            filters[m - 1, k] = (right - k) / max(1, right - center)
    return filters


def feature_matrices(np: Any, audio: Any, sr: int) -> tuple[Any, Any, Any]:
    x = audio.reshape(-1).astype(np.float32, copy=False)
    frames = make_frames(np, x, sr)
    nfft = 512
    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window[None, :], n=nfft, axis=1)) ** 2
    mel = mel_filterbank(np, sr, nfft, MEL_BANDS) @ spectrum.T
    logmel = np.log(np.maximum(mel, MEL_POWER_FLOOR))
    # C0 is defined as mean natural-log mel energy, so a gain g has exact
    # theoretical delta 2*ln(g); C1-C12 use zero-mean DCT-II bases.
    n_mels = logmel.shape[0]
    n_mfcc = 13
    k = np.arange(n_mfcc)[:, None]
    n = np.arange(n_mels)[None, :]
    basis = np.cos(np.pi / n_mels * (n + 0.5) * k)
    basis[0] = 1.0 / n_mels
    if n_mfcc > 1:
        basis[1:] *= math.sqrt(2 / n_mels)
    mfcc = basis @ logmel
    return mel, logmel, mfcc


def basic_descriptors(np: Any, audio: Any, sr: int) -> dict[str, Any]:
    x = audio.reshape(-1).astype(np.float32, copy=False)
    frames = make_frames(np, x, sr)
    rms_full = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(frame_rms, 1e-12))
    active = frame_rms >= 10 ** (-45.0 / 20.0)
    active_db = db[active] if np.any(active) else db
    mel, logmel, mfcc = feature_matrices(np, x, sr)
    mel_floor_fraction = float(np.mean(mel <= MEL_POWER_FLOOR))
    nfft = 512
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, n=nfft, axis=1)) ** 2, n=nfft, axis=1)
    lag_min = max(1, int(sr / 500))
    lag_max = min(nfft - 1, int(sr / 65))
    segment = ac[:, lag_min:lag_max + 1]
    best_rel = np.argmax(segment, axis=1)
    best_idx = best_rel + lag_min
    periodicity = segment[np.arange(len(segment)), best_rel] / np.maximum(ac[:, 0], 1e-12)
    voiced = periodicity >= 0.30
    f0 = sr / np.maximum(best_idx.astype(np.float64), 1)
    voiced_f0 = f0[voiced]
    return {
        "decoded_samples": int(x.size),
        "duration_s": round(x.size / sr, 9),
        "sample_rate_hz": int(sr),
        "peak_abs": round(peak, 8),
        "rms": round(rms_full, 8),
        "rms_dbfs": round(20 * math.log10(max(rms_full, 1e-12)), 6),
        "clipped_sample_fraction_abs_ge_0_999": round(float(np.mean(np.abs(x) >= 0.999)), 9),
        "peak_over_full_scale": round(max(0.0, peak - 1.0), 9),
        "frame_ms": FRAME_MS,
        "hop_ms": HOP_MS,
        "frame_count": int(len(frames)),
        "effective_speech_frame_threshold_dbfs": -45.0,
        "effective_speech_frame_fraction": round(float(np.mean(active)), 9),
        "silent_frame_fraction": round(float(np.mean(~active)), 9),
        "active_frame_rms_dbfs_p10_p50_p90": [round(float(v), 6) for v in np.percentile(active_db, [10, 50, 90])],
        "active_frame_dynamic_range_p90_minus_p10_db": round(float(np.percentile(active_db, 90) - np.percentile(active_db, 10)), 6),
        "logmel_definition": "natural log mel power; 40 triangular bands; floor=1e-10; 25ms Hann, 10ms hop, nfft=512",
        "logmel_mean_ln_mean_p10_p50_p90": [round(float(v), 9) for v in [np.mean(logmel), np.percentile(logmel, 10), np.percentile(logmel, 50), np.percentile(logmel, 90)]],
        "logmel_floor_power": MEL_POWER_FLOOR,
        "logmel_floor_bin_fraction": round(mel_floor_fraction, 9),
        "logmel_40band_mean_ln_by_band": [round(float(v), 9) for v in np.mean(logmel, axis=1)],
        "mfcc13_definition": "C0=mean(log-mel); C1-C12=orthonormal DCT-II coefficients",
        "mfcc13_mean": [round(float(v), 9) for v in np.mean(mfcc, axis=1)],
        "mfcc13_std": [round(float(v), 9) for v in np.std(mfcc, axis=1)],
        "autocorr_f0_fallback_hz_p10_p50_p90": [round(float(v), 4) for v in np.percentile(voiced_f0, [10, 50, 90])] if voiced_f0.size else None,
        "autocorr_voiced_fraction_periodicity_ge_0_30": round(float(np.mean(voiced)), 9),
        "autocorr_periodicity_median": round(float(np.median(periodicity)), 9),
    }


def world_descriptors(np: Any, audio: Any, sr: int) -> dict[str, Any]:
    try:
        import pyworld
    except ImportError:
        return {"status": "unavailable", "reason": "pyworld_not_installed"}
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    f0, time_axis = pyworld.dio(x, sr, f0_floor=65.0, f0_ceil=500.0, frame_period=5.0)
    f0 = pyworld.stonemask(x, f0, time_axis, sr)
    sp = pyworld.cheaptrick(x, f0, time_axis, sr)
    ap = pyworld.d4c(x, f0, time_axis, sr)
    voiced = np.isfinite(f0) & (f0 > 0)
    vf0 = f0[voiced]
    vt = time_axis[voiced]
    f0_slope_hz_s = None
    f0_slope_semitones_s = None
    f0_cv = None
    if vf0.size:
        f0_cv = float(np.std(vf0) / np.mean(vf0)) if float(np.mean(vf0)) > 0.0 else None
    if vf0.size >= 2 and float(np.sum((vt - np.mean(vt)) ** 2)) > 0:
        centered_t = vt - np.mean(vt)
        f0_slope_hz_s = float(np.sum(centered_t * (vf0 - np.mean(vf0))) / np.sum(centered_t ** 2))
        semitones = 12.0 * np.log2(vf0 / 440.0)
        f0_slope_semitones_s = float(np.sum(centered_t * (semitones - np.mean(semitones))) / np.sum(centered_t ** 2))
    spectral_power = np.maximum(sp, 1e-20)
    normalized_sp = spectral_power / np.maximum(np.sum(spectral_power, axis=1, keepdims=True), 1e-20)
    normalized_sp_db = 10.0 * np.log10(np.maximum(normalized_sp, 1e-20))
    fft_size = 2 * (sp.shape[1] - 1)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sr)
    bands = ((0.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0), (4000.0, sr / 2.0 + 1.0))
    normalized_sp_by_band: dict[str, float | None] = {}
    normalized_sp_log10_mean_by_band: dict[str, float | None] = {}
    ap_summary_by_band: dict[str, dict[str, float | None]] = {}
    for low, high in bands:
        band_name = f"{int(low)}_{int(min(high, sr / 2.0))}Hz"
        selected = (frequencies >= low) & (frequencies < high)
        if not np.any(selected):
            normalized_sp_by_band[band_name] = None
            normalized_sp_log10_mean_by_band[band_name] = None
            ap_summary_by_band[band_name] = {"mean": None, "p10": None, "p50": None, "p90": None}
            continue
        band_energy = np.sum(normalized_sp[:, selected], axis=1)
        normalized_sp_by_band[band_name] = round(float(np.mean(band_energy)), 12)
        normalized_sp_log10_mean_by_band[band_name] = round(float(np.mean(normalized_sp_db[:, selected])), 9)
        band_ap = ap[:, selected].reshape(-1)
        ap_summary_by_band[band_name] = {
            "mean": round(float(np.mean(band_ap)), 9),
            "p10": round(float(np.percentile(band_ap, 10)), 9),
            "p50": round(float(np.percentile(band_ap, 50)), 9),
            "p90": round(float(np.percentile(band_ap, 90)), 9),
        }
    adjacent_spectral_delta = np.diff(normalized_sp_db, axis=0)
    synthesized = np.asarray(pyworld.synthesize(f0, sp, ap, sr, frame_period=5.0), dtype=np.float64).reshape(-1)
    compared_samples = min(len(x), len(synthesized))
    original_common = x[:compared_samples]
    synthesized_common = synthesized[:compared_samples]
    roundtrip_error = synthesized_common - original_common
    original_rms = float(np.sqrt(np.mean(original_common ** 2))) if compared_samples else 0.0
    error_rms = float(np.sqrt(np.mean(roundtrip_error ** 2))) if compared_samples else 0.0
    correlation = float(np.corrcoef(original_common, synthesized_common)[0, 1]) if compared_samples > 1 and np.std(original_common) and np.std(synthesized_common) else None
    return {
        "status": "computed",
        "algorithm": "WORLD DIO + StoneMask + CheapTrick + D4C",
        "frame_period_ms": 5.0,
        "f0_floor_hz": 65.0,
        "f0_ceil_hz": 500.0,
        "frame_count": int(len(f0)),
        "f0_voiced_fraction": round(float(np.mean(voiced)), 9),
        "f0_hz_p10_p50_p90": [round(float(v), 4) for v in np.percentile(vf0, [10, 50, 90])] if vf0.size else None,
        "f0_hz_median": round(float(np.median(vf0)), 4) if vf0.size else None,
        "f0_hz_std": round(float(np.std(vf0)), 4) if vf0.size else None,
        "f0_hz_iqr": round(float(np.percentile(vf0, 75) - np.percentile(vf0, 25)), 4) if vf0.size else None,
        "f0_coefficient_of_variation": round(f0_cv, 9) if f0_cv is not None else None,
        "f0_linear_slope_hz_per_s": round(f0_slope_hz_s, 6) if f0_slope_hz_s is not None else None,
        "f0_linear_slope_semitones_per_s": round(f0_slope_semitones_s, 6) if f0_slope_semitones_s is not None else None,
        "f0_slope_fit_domain": "voiced DIO/StoneMask frames only; least-squares against WORLD frame time",
        "spectral_envelope_log10_mean_p10_p50_p90": [round(float(v), 9) for v in np.percentile(np.log10(np.maximum(sp, 1e-12)), [10, 50, 90])],
        "cheaptrick_sp_normalization": "divide each frame by its summed nonnegative frequency-bin power; report normalized band energy and log10 shape",
        "cheaptrick_sp_normalized_mean_energy_by_band": normalized_sp_by_band,
        "cheaptrick_sp_normalized_log10_mean_by_band": normalized_sp_log10_mean_by_band,
        "cheaptrick_sp_adjacent_frame_local_change_db": {
            "mean_absolute": round(float(np.mean(np.abs(adjacent_spectral_delta))), 9) if adjacent_spectral_delta.size else None,
            "p90_absolute": round(float(np.percentile(np.abs(adjacent_spectral_delta), 90)), 9) if adjacent_spectral_delta.size else None,
            "mean_absolute_by_band": {
                f"{int(low)}_{int(min(high, sr / 2.0))}Hz": (
                    round(float(np.mean(np.abs(adjacent_spectral_delta[:, (frequencies >= low) & (frequencies < high)]))), 9)
                    if adjacent_spectral_delta.size and np.any((frequencies >= low) & (frequencies < high)) else None
                ) for low, high in bands
            },
        },
        "aperiodicity_mean_p10_p50_p90": [round(float(v), 9) for v in np.percentile(ap, [10, 50, 90])],
        "aperiodicity_frame_mean_db_like": round(float(np.mean(10 * np.log10(np.maximum(ap, 1e-12)))), 9),
        "d4c_aperiodicity_mean_p10_p50_p90_by_band": ap_summary_by_band,
        "world_roundtrip_synthesis": {
            "input_samples": int(len(x)),
            "synthesized_samples": int(len(synthesized)),
            "length_delta_samples": int(len(synthesized) - len(x)),
            "compared_prefix_samples": int(compared_samples),
            "alignment_or_time_warp_applied": False,
            "input_rms": round(original_rms, 12),
            "error_rms": round(error_rms, 12),
            "error_rms_to_input_rms": round(error_rms / max(original_rms, 1e-15), 9),
            "snr_db": round(20 * math.log10(max(original_rms, 1e-15) / max(error_rms, 1e-15)), 6),
            "max_abs_error": round(float(np.max(np.abs(roundtrip_error))), 12) if compared_samples else None,
            "pearson_r": round(correlation, 9) if correlation is not None else None,
        },
        "sp_shape": list(sp.shape),
        "ap_shape": list(ap.shape),
    }


def world_runtime_metadata() -> dict[str, Any]:
    try:
        import pyworld
    except ImportError:
        return {"available": False, "reason": "pyworld_not_installed"}
    import numpy as np
    extension_path = getattr(getattr(pyworld, "pyworld", None), "__file__", None)
    return {
        "available": True,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pyworld_version": str(getattr(pyworld, "__version__", "unknown")),
        "pyworld_package_path": str(Path(pyworld.__file__).resolve()),
        "pyworld_extension_path": str(Path(extension_path).resolve()) if extension_path else None,
        "pyworld_extension_sha256": sha256_file(Path(extension_path)) if extension_path and Path(extension_path).is_file() else None,
    }


def mfcc_gain_check(np: Any, raw_audio: Any, transformed_audio: Any, sr: int, gain: float) -> dict[str, Any]:
    raw_mel, raw_log, raw_mfcc = feature_matrices(np, raw_audio, sr)
    new_mel, new_log, new_mfcc = feature_matrices(np, transformed_audio, sr)
    common_frames = min(raw_log.shape[1], new_log.shape[1])
    raw_mel, new_mel = raw_mel[:, :common_frames], new_mel[:, :common_frames]
    raw_mfcc, new_mfcc = raw_mfcc[:, :common_frames], new_mfcc[:, :common_frames]
    floor_bins = (raw_mel <= MEL_POWER_FLOOR) | (new_mel <= MEL_POWER_FLOOR)
    floor_frames = np.any(floor_bins, axis=0)
    clean_frames = ~floor_frames
    expected_c0 = 2.0 * math.log(max(gain, 1e-300))
    c0_delta = new_mfcc[0] - raw_mfcc[0]
    c1_12_delta = new_mfcc[1:] - raw_mfcc[1:]
    return {
        "gain": round(gain, 12),
        "gain_db": round(20 * math.log10(max(gain, 1e-300)), 9),
        "expected_c0_delta_2ln_gain": round(expected_c0, 12),
        "measured_c0_delta_mean": round(float(np.mean(c0_delta)), 12),
        "max_abs_c0_error_nonfloor_frames": round(float(np.max(np.abs(c0_delta[clean_frames] - expected_c0))), 12) if np.any(clean_frames) else None,
        "max_abs_c1_c12_delta_nonfloor_frames": round(float(np.max(np.abs(c1_12_delta[:, clean_frames]))), 12) if np.any(clean_frames) else None,
        "mel_bins_at_floor_either_side": int(np.count_nonzero(floor_bins)),
        "mel_bin_total": int(floor_bins.size),
        "mel_floor_exception_fraction": round(float(np.mean(floor_bins)), 12),
        "frames_with_any_floor_exception": int(np.count_nonzero(floor_frames)),
        "frames_without_floor_exception": int(np.count_nonzero(clean_frames)),
        "raw_logmel_ln_mean": round(float(np.mean(raw_log)), 12),
        "transformed_logmel_ln_mean": round(float(np.mean(new_log)), 12),
    }


def describe_cal(args: argparse.Namespace) -> int:
    np, sf = get_audio_libs()
    source, pairs, tts_meta, rows = load_cohort(args, "calibration")
    if len(rows) != 20:
        raise ValueError("calibration descriptor stage requires exactly 20 frozen calibration records")
    pair_by_id = {str(x["id"]): x for x in pairs["records"]}
    meta_by_id = tts_meta["results"]
    # Target is fixed in the frozen OpenSpec. Compute the calibration-N geometric
    # mean only as a descriptive comparator; it does not alter this target.
    cal_natural_rms = []
    for row in rows:
        pair_row = pair_by_id[str(row["id"])]
        tts_row = meta_by_id[str(row["id"])]
        files = resolve_row_files(row, pair_row, tts_row, args)
        _, _, audio = decode_info(files["natural16"], np, sf)
        cal_natural_rms.append(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))))
    cal_natural_geomean = float(np.exp(np.mean(np.log(np.maximum(cal_natural_rms, 1e-15)))))
    target_rms = FIXED_TARGET_RMS
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sample_id = str(row["id"])
        pair_row, tts_row = pair_by_id[sample_id], meta_by_id[sample_id]
        files = resolve_row_files(row, pair_row, tts_row, args)
        sides = {}
        for condition, key in (("natural", "natural16"), ("tts", "tts16")):
            decoded, _, audio = decode_info(files[key], np, sf)
            if decoded["decoded_sample_rate_hz"] != CANONICAL_RATE or decoded["decoded_channels"] != 1:
                raise ValueError(f"calibration descriptor audio format mismatch for {sample_id}/{condition}")
            raw_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            if raw_rms <= 1e-8:
                raise ValueError(f"silence gate failed for calibration {sample_id}/{condition}")
            fixed_gain = target_rms / raw_rms
            side_variants: dict[str, Any] = {}
            variant_waves = {
                "raw": audio.astype(np.float64),
                "fixed": audio.astype(np.float64) * fixed_gain,
                "fixed_minus3dB": audio.astype(np.float64) * fixed_gain * (10.0 ** (-LEVEL_STEP_DB / 20.0)),
                "fixed_plus3dB": audio.astype(np.float64) * fixed_gain * (10.0 ** (LEVEL_STEP_DB / 20.0)),
            }
            for variant, candidate in variant_waves.items():
                gain_to_raw = 1.0 if variant == "raw" else float(np.sqrt(np.mean(candidate * candidate)) / raw_rms) if variant == "fixed" else fixed_gain * (10.0 ** ((-LEVEL_STEP_DB if variant == "fixed_minus3dB" else LEVEL_STEP_DB) / 20.0))
                if variant == "raw":
                    analyzed = audio.astype(np.float32, copy=False)
                    output_path = files[key]
                    output_hash = decoded["file_sha256"]
                else:
                    derived = args.outdir.resolve() / "audio" / "calibration" / sample_id / condition / f"{variant}.wav"
                    derived.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(derived), candidate.astype(np.float32), CANONICAL_RATE, subtype="FLOAT")
                    analyzed, variant_rate = sf.read(str(derived), dtype="float32", always_2d=True)
                    if variant_rate != CANONICAL_RATE or analyzed.shape[1] != 1:
                        raise ValueError(f"derived fixed waveform has invalid format: {sample_id}/{condition}/{variant}")
                    analyzed = analyzed[:, 0]
                    output_path = derived.resolve()
                    output_hash = sha256_file(output_path)
                descriptors = basic_descriptors(np, analyzed, CANONICAL_RATE)
                side_variants[variant] = {
                    "audio_path": str(output_path),
                    "audio_file_sha256": output_hash,
                    "input_audio_sha256": decoded["file_sha256"],
                    "decoded_source_pcm16_sha256": decoded["decoded_pcm16_sha256"],
                    "gain_to_raw": round(gain_to_raw, 12),
                    "gain_db_to_raw": round(20 * math.log10(max(gain_to_raw, 1e-300)), 9),
                    "rms": descriptors["rms"],
                    "peak_abs": descriptors["peak_abs"],
                    "peak_over_full_scale": descriptors["peak_over_full_scale"],
                    "duration_s": descriptors["duration_s"],
                    "effective_speech_fraction": descriptors["effective_speech_frame_fraction"],
                    "active_frame_dynamic_range_p90_minus_p10_db": descriptors["active_frame_dynamic_range_p90_minus_p10_db"],
                    "logmel_mean_ln": descriptors["logmel_mean_ln_mean_p10_p50_p90"][0],
                    "logmel_floor_bin_fraction": descriptors["logmel_floor_bin_fraction"],
                    "mfcc13_mean": descriptors["mfcc13_mean"],
                    "mfcc13_std": descriptors["mfcc13_std"],
                    "mfcc_gain_identity_vs_raw": mfcc_gain_check(np, audio, analyzed, CANONICAL_RATE, gain_to_raw),
                    "full_descriptors": descriptors,
                    "world": world_descriptors(np, analyzed, CANONICAL_RATE),
                }
            sides[condition] = {
                "source_path": decoded["path"],
                "source_file_sha256": decoded["file_sha256"],
                "decoded_pcm16_sha256": decoded["decoded_pcm16_sha256"],
                "header_audit": {k: decoded.get(k) for k in ("declared_frames", "decoded_frames", "declared_vs_payload_mismatch")},
                "raw_rms": round(raw_rms, 12),
                "fixed_gain": round(fixed_gain, 12),
                "fixed_gain_db": round(20 * math.log10(fixed_gain), 9),
                "variants": side_variants,
            }
        records.append({
            "id": sample_id,
            "speaker": row["speaker"],
            "split": "calibration",
            "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
            "source_identity_sha256": pair_row["input_identity_sha256"],
            "natural": sides["natural"],
            "tts": sides["tts"],
        })
        print(f"DESCRIBED CAL {index}/{len(rows)} {sample_id}", flush=True)
    names = ["rms_dbfs", "peak_abs", "silent_frame_fraction", "active_frame_dynamic_range_p90_minus_p10_db", "autocorr_voiced_fraction_periodicity_ge_0_30"]
    paired_summary = {}
    for variant in ("raw", "fixed", "fixed_minus3dB", "fixed_plus3dB"):
        for name in names:
            diffs = [r["tts"]["variants"][variant]["full_descriptors"][name] - r["natural"]["variants"][variant]["full_descriptors"][name] for r in records]
            paired_summary[f"{variant}/{name}"] = {
                "n": len(diffs),
                "mean_tts_minus_natural": round(float(np.mean(diffs)), 9),
                "median_tts_minus_natural": round(float(np.median(diffs)), 9),
                "p10_p90_tts_minus_natural": [round(float(v), 9) for v in np.percentile(diffs, [10, 90])],
                "positive_pairs": int(sum(d > 0 for d in diffs)),
            }
    world_metric_getters = {
        "voiced_fraction": lambda w: w.get("f0_voiced_fraction"),
        "f0_median_hz": lambda w: w.get("f0_hz_median"),
        "f0_std_hz": lambda w: w.get("f0_hz_std"),
        "f0_iqr_hz": lambda w: w.get("f0_hz_iqr"),
        "f0_cv": lambda w: w.get("f0_coefficient_of_variation"),
        "f0_slope_hz_per_s": lambda w: w.get("f0_linear_slope_hz_per_s"),
        "f0_slope_semitones_per_s": lambda w: w.get("f0_linear_slope_semitones_per_s"),
        "d4c_ap_mean_0_500hz": lambda w: w.get("d4c_aperiodicity_mean_p10_p50_p90_by_band", {}).get("0_500Hz", {}).get("mean"),
        "d4c_ap_mean_500_1000hz": lambda w: w.get("d4c_aperiodicity_mean_p10_p50_p90_by_band", {}).get("500_1000Hz", {}).get("mean"),
        "d4c_ap_mean_1000_2000hz": lambda w: w.get("d4c_aperiodicity_mean_p10_p50_p90_by_band", {}).get("1000_2000Hz", {}).get("mean"),
        "d4c_ap_mean_2000_4000hz": lambda w: w.get("d4c_aperiodicity_mean_p10_p50_p90_by_band", {}).get("2000_4000Hz", {}).get("mean"),
        "d4c_ap_mean_4000_8000hz": lambda w: w.get("d4c_aperiodicity_mean_p10_p50_p90_by_band", {}).get("4000_8000Hz", {}).get("mean"),
        "normalized_sp_local_change_db": lambda w: w.get("cheaptrick_sp_adjacent_frame_local_change_db", {}).get("mean_absolute"),
        "world_roundtrip_relative_error": lambda w: w.get("world_roundtrip_synthesis", {}).get("error_rms_to_input_rms"),
        "world_roundtrip_snr_db": lambda w: w.get("world_roundtrip_synthesis", {}).get("snr_db"),
    }
    world_paired_summary: dict[str, Any] = {}
    for variant in ("raw", "fixed", "fixed_minus3dB", "fixed_plus3dB"):
        for metric, getter in world_metric_getters.items():
            diffs = []
            for row in records:
                natural_value = getter(row["natural"]["variants"][variant]["world"])
                tts_value = getter(row["tts"]["variants"][variant]["world"])
                if natural_value is not None and tts_value is not None:
                    diffs.append(float(tts_value) - float(natural_value))
            world_paired_summary[f"{variant}/{metric}"] = {
                "n": len(diffs),
                "mean_tts_minus_natural": round(float(np.mean(diffs)), 9) if diffs else None,
                "median_tts_minus_natural": round(float(np.median(diffs)), 9) if diffs else None,
                "p10_p90_tts_minus_natural": [round(float(v), 9) for v in np.percentile(diffs, [10, 90])] if diffs else None,
                "positive_pairs": int(sum(d > 0 for d in diffs)),
            }
    mfcc_checks = [r[arm]["variants"][variant]["mfcc_gain_identity_vs_raw"] for r in records for arm in ("natural", "tts") for variant in ("fixed", "fixed_minus3dB", "fixed_plus3dB")]
    script_spec = REPO_ROOT / "openspec/changes/confirm-tts-source-mechanism-20260930/spec.md"
    payload = {
        "schema_version": 1,
        "manifest_type": "source_axis_calibration_physical_audio_descriptors",
        "status": "CAL_ONLY_OUTCOME_BLIND",
        "selection_split": "calibration",
        "evaluation_waveforms_read": False,
        "source_manifest_sha256": sha256_file(args.input_manifest.resolve()),
        "source_manifest_fingerprint": source.get("manifest_sha256"),
        "pair_manifest_sha256": sha256_file(args.pair_manifest.resolve()),
        "pair_manifest_fingerprint": pairs.get("manifest_sha256"),
        "tts_meta_sha256": sha256_file(args.tts_meta.resolve()),
        "tts_meta_input_fingerprint": tts_meta.get("input_manifest_sha256"),
        "producer": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "command_argv": [sys.executable, *sys.argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "fixed_target": {"target_rms": target_rms, "source": str(script_spec.resolve()), "source_sha256": sha256_file(script_spec), "selection": "fixed in frozen OpenSpec; not re-estimated", "calibration_natural_rms_geometric_mean_descriptive_only": round(cal_natural_geomean, 12), "operation": "float64 waveform multiplied by target_RMS/current_RMS; save FLOAT32; no clipping or quantization"},
        "method": {"frame_ms": FRAME_MS, "hop_ms": HOP_MS, "effective_speech_mask_rms_dbfs": -45.0, "mel_bands": MEL_BANDS, "logmel_floor_power": MEL_POWER_FLOOR, "mfcc_definition": "C0=mean natural-log mel; C1-C12=orthonormal DCT-II; expected C0 delta=2ln(gain), C1-C12 invariant except mel floor", "fixed_level_offsets_db": [-LEVEL_STEP_DB, LEVEL_STEP_DB], "world": "WORLD DIO/StoneMask/CheapTrick/D4C; F0 slope/spread, voiced fraction, frequency-band AP, per-frame sum-normalized SP shape/local change, no-alignment roundtrip error"},
        "world_runtime": world_runtime_metadata(),
        "mfcc_gain_identity_summary": {
            "max_abs_c0_error_nonfloor_frames": max((x["max_abs_c0_error_nonfloor_frames"] for x in mfcc_checks if x["max_abs_c0_error_nonfloor_frames"] is not None), default=None),
            "max_abs_c1_c12_delta_nonfloor_frames": max((x["max_abs_c1_c12_delta_nonfloor_frames"] for x in mfcc_checks if x["max_abs_c1_c12_delta_nonfloor_frames"] is not None), default=None),
            "total_mel_floor_exception_bins": sum(x["mel_bins_at_floor_either_side"] for x in mfcc_checks),
            "comparisons": len(mfcc_checks),
        },
        "paired_summary_tts_minus_natural": paired_summary,
        "paired_world_summary_tts_minus_natural": world_paired_summary,
        "records": records,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    out_path = args.outdir.resolve() / "calibration_physical_descriptors.json"
    write_json(out_path, payload)
    print(json.dumps({"rows": len(records), "evaluation_waveforms_read": False, "manifest_sha256": payload["manifest_sha256"], "out": str(out_path), "summary": paired_summary}, separators=(",", ":")))
    return 0


def base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--pair-manifest", type=Path, default=DEFAULT_PAIR)
    p.add_argument("--tts-meta", type=Path, default=DEFAULT_TTS_META)
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--natural-root", type=Path)
    p.add_argument("--tts24-root", type=Path)
    p.add_argument("--tts16-root", type=Path)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    audit = sub.add_parser("audit-resample", parents=[base_parser()], help="full decode audit and 16k->24k->16k controls")
    audit.add_argument("--ffmpeg", default=None, help="path to the exact ffmpeg binary used for TTS canonicalization")
    audit.set_defaults(func=audit_resample)
    describe = sub.add_parser("describe-cal", parents=[base_parser()], help="paired, outcome-blind physical features for calibration only")
    describe.set_defaults(func=describe_cal)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
