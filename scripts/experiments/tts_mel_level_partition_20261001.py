"""CPU producer for the frozen Wav2Lip mel interior/boundary partition.

This module only decodes canonical PCM, computes the repository Wav2Lip mel
frontend, builds the exact four-cell arrays, and seals CPU diagnostics. It does
not render videos, score SyncNet, or inspect model effects.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import importlib
import json
import math
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


TARGET_RMS = 0.0376838172675746
SPEC_RELATIVE = "openspec/changes/confirm-mel-level-partition-20261001/spec.md"
CONDITIONS = ("q00", "q10", "q01", "q11")
IDENTITY_FIELDS_FALLBACK = ("id", "speaker", "text", "split", "ref_sha256")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _embedded_fingerprint(value: dict[str, Any]) -> str | None:
    embedded = value.get("manifest_sha256")
    if not isinstance(embedded, str):
        return None
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _verify_input_identity(source: dict[str, Any], tts: dict[str, Any]) -> str:
    fields = tuple(tts.get("input_identity_fields") or IDENTITY_FIELDS_FALLBACK)
    if fields != IDENTITY_FIELDS_FALLBACK:
        raise ValueError(f"unexpected frozen input identity fields: {fields}")
    source_rows = source.get("records")
    tts_rows = tts.get("records")
    if not isinstance(source_rows, list) or not isinstance(tts_rows, list) or len(source_rows) != 100 or len(tts_rows) != 100:
        raise ValueError("frozen source and TTS manifests must each contain exactly 100 rows")
    source_identity = [{field: row.get(field) for field in fields} for row in source_rows]
    tts_identity = [row.get("input_identity") for row in tts_rows]
    if source_identity != tts_identity:
        raise ValueError("ordered source/TTS identity rows differ")
    digest = hashlib.sha256(_canonical_json(source_identity)).hexdigest()
    if digest != tts.get("input_identity_sha256"):
        raise ValueError("ordered source/TTS identity digest differs from the frozen TTS manifest")
    source_counts = {split: sum(row.get("split") == split for row in source_rows) for split in ("calibration", "evaluation")}
    tts_counts = {split: sum(row.get("split") == split for row in tts_rows) for split in ("calibration", "evaluation")}
    if source_counts != {"calibration": 20, "evaluation": 80} or tts_counts != source_counts:
        raise ValueError(f"frozen cohort split counts differ: source={source_counts}, tts={tts_counts}")
    return digest


def _wav_path_for_source(input_root: Path, row: dict[str, Any]) -> Path:
    utterance = Path(str(row["ref_path"]).replace("\\", "/")).name
    return input_root / "audio" / row["speaker"] / utterance


def _wav_path_for_tts(input_root: Path, row: dict[str, Any]) -> Path:
    return input_root / "tts_outputs" / "audio_16k_canonical" / f"{row['id']}.wav"


def _read_pcm16(path: Path, expected_sha: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"input WAV hash differs from frozen row: {path}")
    rate, pcm = wavfile.read(str(path))
    if rate != 16_000 or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"expected canonical mono PCM16 16 kHz WAV at {path}: {(rate, pcm.shape, pcm.dtype)}")
    waveform = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"decoded waveform is empty or nonfinite: {path}")
    return pcm, waveform, {
        "path": str(path.resolve()),
        "file_sha256": actual_sha,
        "sample_rate_hz": 16_000,
        "channels": 1,
        "samples": int(waveform.size),
        "duration_s": float(waveform.size / 16_000),
        "pcm_dtype": "int16",
        "waveform_dtype": "float32",
        "waveform_array_sha256": array_sha256(waveform),
        "peak_abs_pcm16": int(np.max(np.abs(pcm.astype(np.int32)))),
        "quantization": "canonical PCM16 mapped by float64 / 32768 then one float32 cast",
    }


def _fixed_waveform(waveform: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(waveform)
    if source.dtype != np.float32 or source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("RMS normalization requires finite mono float32 source PCM")
    source64 = source.astype(np.float64, copy=False)
    rms_before = float(np.sqrt(np.mean(np.square(source64), dtype=np.float64)))
    if not math.isfinite(rms_before) or rms_before <= 0:
        raise ValueError("cannot RMS-normalize a silent or nonfinite waveform")
    gain = np.float64(TARGET_RMS) / np.float64(rms_before)
    fixed = np.multiply(source64, gain, dtype=np.float64).astype(np.float32)
    rms_after = float(np.sqrt(np.mean(np.square(fixed.astype(np.float64)), dtype=np.float64)))
    peak = float(np.max(np.abs(fixed)))
    return fixed, {
        "target_rms": TARGET_RMS,
        "input_rms": rms_before,
        "gain": float(gain),
        "gain_db": float(20.0 * np.log10(gain)),
        "realized_rms_float32": rms_after,
        "peak_abs_float32": peak,
        "peak_over_one_no_clipping_applied": bool(peak > 1.0),
        "clipping_applied": False,
        "output_dtype": "float32",
        "fixed_waveform_array_sha256": array_sha256(fixed),
    }


def _import_wav2lip_audio(wav2lip_root: Path, portable_source_root: Path):
    root = str(wav2lip_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    package_parent = str(portable_source_root.resolve())
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    import librosa

    original = librosa.filters.mel
    mel_signature = str(inspect.signature(original))
    helper = importlib.import_module("portable_source_axis_20260930.wav2lip")
    ops = importlib.import_module("portable_source_axis_20260930.ops")
    shim_meta = helper.install_librosa_compatibility_shim()
    import audio
    import hparams as wav2lip_hparams

    hp = getattr(audio, "hp", getattr(wav2lip_hparams, "hparams", wav2lip_hparams))
    data = getattr(hp, "data", None)
    if isinstance(hp, dict):
        hp_values = hp
    elif isinstance(data, dict):
        hp_values = data
    else:
        hp_values = vars(hp) if hasattr(hp, "__dict__") else {}
    relevant_names = (
        "sample_rate", "n_fft", "num_mels", "hop_size", "win_size", "fmin", "fmax",
        "preemphasize", "preemphasis", "signal_normalization", "allow_clipping_in_normalization",
        "symmetric_mels", "max_abs_value", "min_level_db", "ref_level_db",
    )
    actual_hparams = {
        str(name): value for name, value in hp_values.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    actual_hparams.update({name: hp_values[name] for name in relevant_names if name in hp_values})
    if not actual_hparams:
        actual_hparams = {name: getattr(hp, name) for name in relevant_names if hasattr(hp, name)}

    return audio, {
        "librosa_version": str(librosa.__version__),
        "librosa_filters_mel_signature_before_shim": mel_signature,
        "shim": shim_meta.get("shim"),
        "shim_source_path": str(Path(helper.__file__).resolve()),
        "shim_source_sha256": sha256_file(Path(helper.__file__).resolve()),
        "chunk_ops_source_path": str(Path(ops.__file__).resolve()),
        "chunk_ops_source_sha256": sha256_file(Path(ops.__file__).resolve()),
        "actual_wav2lip_hparams": actual_hparams,
    }


def _chunk_starts(mel_frames: int) -> list[int]:
    ops = importlib.import_module("portable_source_axis_20260930.ops")
    return [int(value) for value in ops.mel_chunk_starts(int(mel_frames))]


def _mel_chunks(mel: np.ndarray, starts: list[int]) -> np.ndarray:
    return np.stack([mel[:, start : start + 16] for start in starts], axis=0).astype(np.float32, copy=False)


def _mel_summary(m0: np.ndarray, m1: np.ndarray, u: np.ndarray, q: dict[str, np.ndarray], gain: float) -> dict[str, Any]:
    lo = np.float32(-4.0)
    hi = np.float32(4.0)
    inward_lo = np.nextafter(lo, np.float32(0.0), dtype=np.float32)
    inward_hi = np.nextafter(hi, np.float32(0.0), dtype=np.float32)
    d = m1.astype(np.float64) - m0.astype(np.float64)
    q10d = q["q10"].astype(np.float64) - m0.astype(np.float64)
    q01d = q["q01"].astype(np.float64) - m0.astype(np.float64)
    closure = d - (q10d + q01d)
    pred = float(1.6 * np.log10(np.float64(gain)))
    interior = d[u]
    pred_err = interior - pred

    def count(condition: np.ndarray) -> int:
        return int(np.count_nonzero(condition))

    return {
        "shape": list(m0.shape),
        "dtype": "float32",
        "total_bins": int(m0.size),
        "U_interior_bins": int(np.count_nonzero(u)),
        "S_complement_bins": int(np.count_nonzero(~u)),
        "raw_lower_lock": count((m0 == lo) & (m1 == lo)),
        "raw_lower_unlock": count((m0 == lo) & (m1 > lo)),
        "fixed_lower_entry": count((m0 > lo) & (m1 == lo)),
        "raw_upper_lock": count((m0 == hi) & (m1 == hi)),
        "raw_upper_unlock": count((m0 == hi) & (m1 < hi)),
        "fixed_upper_entry": count((m0 < hi) & (m1 == hi)),
        "upper_cross_raw_interior_to_fixed_upper": count((m0 < hi) & (m1 == hi)),
        "upper_cross_raw_upper_to_fixed_interior": count((m0 == hi) & (m1 < hi)),
        "inward_one_ulp": {
            "raw_lower": count(m0 == inward_lo),
            "raw_upper": count(m0 == inward_hi),
            "fixed_lower": count(m1 == inward_lo),
            "fixed_upper": count(m1 == inward_hi),
        },
        "delta": {
            "l1_sum": float(np.abs(d).sum(dtype=np.float64)),
            "l2": float(np.linalg.norm(d.ravel())),
            "linf": float(np.max(np.abs(d), initial=0.0)),
            "interior_l2": float(np.linalg.norm(interior)),
            "boundary_l2": float(np.linalg.norm(d[~u])),
        },
        "interior_waveform_gain_prediction": {
            "gain": float(gain),
            "predicted_mel_offset": pred,
            "count": int(interior.size),
            "mean_error": float(pred_err.mean()) if pred_err.size else None,
            "rmse_error": float(np.sqrt(np.mean(np.square(pred_err)))) if pred_err.size else None,
            "max_abs_error": float(np.max(np.abs(pred_err), initial=0.0)),
        },
        "four_cell": {
            "q00_equals_M0": bool(np.array_equal(q["q00"], m0)),
            "q11_equals_M1": bool(np.array_equal(q["q11"], m1)),
            "q10_equals_where_U_M1_else_M0": bool(np.array_equal(q["q10"], np.where(u, m1, m0))),
            "q01_equals_where_U_M0_else_M1": bool(np.array_equal(q["q01"], np.where(u, m0, m1))),
            "float64_closure_max_abs": float(np.max(np.abs(closure), initial=0.0)),
            "float64_closure_exact": bool(np.array_equal(closure, np.zeros_like(closure))),
        },
    }


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("scipy", "librosa", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-as-distribution"
    return versions


def _source_files(wav2lip_root: Path) -> dict[str, dict[str, str]]:
    result = {}
    for name in ("audio.py", "hparams.py"):
        path = wav2lip_root / name
        if not path.is_file():
            raise FileNotFoundError(f"required Wav2Lip source file is missing: {path}")
        result[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def _select_paths(input_root: Path, source_row: dict[str, Any], tts_row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    specs = []
    if split == "calibration":
        natural_path = _wav_path_for_source(input_root, source_row)
        natural_expected = str(source_row["ref_sha256"])
        tts_contract = tts_row.get("qwen_canonical_16k")
        if not isinstance(tts_contract, dict):
            raise ValueError(f"canonical TTS audio metadata missing for {source_row['id']}")
        tts_path = _wav_path_for_tts(input_root, tts_row)
        specs.append({"source_condition": "natural", "path": natural_path, "expected_sha256": natural_expected})
        specs.append({"source_condition": "tts", "path": tts_path, "expected_sha256": str(tts_contract["sha256"])})
    elif split == "evaluation":
        natural_path = _wav_path_for_source(input_root, source_row)
        specs.append({"source_condition": "natural", "path": natural_path, "expected_sha256": str(source_row["ref_sha256"])})
    else:
        raise ValueError(f"unsupported split: {split}")
    return specs


def _build_plan(args: argparse.Namespace, source: dict[str, Any], tts: dict[str, Any], identity_sha: str,
                source_path: Path, tts_path: Path, spec_path: Path, wav2lip_root: Path,
                portable_source_root: Path,
                frontend_meta: dict[str, Any]) -> dict[str, Any]:
    split = args.split
    source_rows = [row for row in source["records"] if row["split"] == split]
    if len(source_rows) != (20 if split == "calibration" else 80):
        raise ValueError(f"{split} expected row count mismatch: {len(source_rows)}")
    tts_rows = {row["id"]: row for row in tts["records"]}
    rows = []
    for index, source_row in enumerate(source_rows):
        tts_row = tts_rows.get(source_row["id"])
        if tts_row is None or tts_row.get("input_identity") != {key: source_row.get(key) for key in IDENTITY_FIELDS_FALLBACK}:
            raise ValueError(f"frozen pair row does not match source row: {source_row['id']}")
        for arm in _select_paths(Path(args.input_root).resolve(), source_row, tts_row, split):
            path = Path(arm["path"]).resolve()
            observed_sha = sha256_file(path) if path.is_file() else None
            input_status = "preflight_pass" if observed_sha == arm["expected_sha256"] else "preflight_failed"
            rows.append({
                "ordered_index": index,
                "id": source_row["id"],
                "speaker": source_row["speaker"],
                "split": split,
                "source_condition": arm["source_condition"],
                "text": source_row["text"],
                "ref_sha256": source_row["ref_sha256"],
                "wav_path": str(path),
                "expected_wav_sha256": arm["expected_sha256"],
                "observed_wav_sha256": observed_sha,
                "status": input_status,
                "preflight_failure": None if input_status == "preflight_pass" else ("missing_input" if observed_sha is None else "input_sha256_mismatch"),
                "sample_rate_hz": 16_000,
                "channels": 1,
                "condition_ids": list(CONDITIONS),
            })
    expected_inputs = 40 if split == "calibration" else 80
    if len(rows) != expected_inputs:
        raise ValueError(f"planned source-arm row count mismatch: {len(rows)} != {expected_inputs}")
    code_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "experiment_id": "confirm-mel-level-partition-20261001",
        "status": "frozen_pre_effects",
        "created_at_utc": _utc_now(),
        "split": split,
        "scope": "calibration20_both_sources" if split == "calibration" else "evaluation80_natural_only",
        "planned_audio_rows": len(rows),
        "planned_generation_rows": len(rows) * 4,
        "target_rms": TARGET_RMS,
        "conditions": list(CONDITIONS),
        "partition_definition": {
            "U": "(-4 < M0) & (M0 < 4) & (-4 < M1) & (M1 < 4)",
            "S": "logical complement of U",
            "q00": "exact M0 RAW at U and S",
            "q10": "where(U, M1, M0)",
            "q01": "where(U, M0, M1)",
            "q11": "exact M1 FIXED at U and S",
            "no_epsilon_or_float32_boundary_reassignment": True,
        },
        "spec": {"path": str(spec_path.resolve()), "sha256": sha256_file(spec_path)},
        "input_manifest": {
            "path": str(source_path.resolve()), "raw_sha256": sha256_file(source_path),
            "embedded_manifest_sha256": source.get("manifest_sha256"),
            "canonical_payload_sha256_recomputed": _embedded_fingerprint(source),
            "canonical_payload_matches_embedded_fingerprint": _embedded_fingerprint(source) == source.get("manifest_sha256"),
        },
        "tts_pair_manifest": {
            "path": str(tts_path.resolve()), "raw_sha256": sha256_file(tts_path),
            "embedded_manifest_sha256": tts.get("manifest_sha256"),
            "canonical_payload_sha256_recomputed": _embedded_fingerprint(tts),
            "canonical_payload_matches_embedded_fingerprint": _embedded_fingerprint(tts) == tts.get("manifest_sha256"),
        },
        "input_identity_sha256": identity_sha,
        "input_identity_serialization": "ordered JSON array of objects with sorted keys and compact separators, UTF-8",
        "wav2lip_root": str(wav2lip_root.resolve()),
        "portable_source_root": str(portable_source_root.resolve()),
        "wav2lip_source": _source_files(wav2lip_root),
        "frontend": {
            **frontend_meta,
            "audio_module_sha256": sha256_file(wav2lip_root / "audio.py"),
            "hparams_module_sha256": sha256_file(wav2lip_root / "hparams.py"),
            "mel_frontend": "actual Wav2Lip audio.melspectrogram; compatibility shim positional sr,n_fft mapped to keyword arguments; librosa defaults unchanged",
            "chunk_rule": "for i=0.., start=int(i*80/25), append mel[:,start:start+16], when start+16>mel_frames append mel[:,-16:] and stop",
        },
        "runtime": _package_versions(),
        "producer_code": {"path": str(code_path), "sha256": sha256_file(code_path)},
        "rows": rows,
        "model_forward_performed": False,
        "effects_read": False,
    }


def _prepare_artifact(row: dict[str, Any], audio: Any, outdir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pcm, waveform, raw_meta = _read_pcm16(Path(row["wav_path"]), row["expected_wav_sha256"])
    fixed, fixed_meta = _fixed_waveform(waveform)
    m0 = np.asarray(audio.melspectrogram(waveform), dtype=np.float32)
    m1 = np.asarray(audio.melspectrogram(fixed), dtype=np.float32)
    if m0.ndim != 2 or m0.shape[0] != 80 or m0.shape != m1.shape or m0.shape[1] < 16:
        raise ValueError(f"frontend shape mismatch: {m0.shape}/{m1.shape}")
    if not np.isfinite(m0).all() or not np.isfinite(m1).all() or float(m0.min()) < -4.0 or float(m0.max()) > 4.0 or float(m1.min()) < -4.0 or float(m1.max()) > 4.0:
        raise ValueError("endpoint mel contains nonfinite/out-of-bound values")
    u = ((m0 > np.float32(-4.0)) & (m0 < np.float32(4.0)) & (m1 > np.float32(-4.0)) & (m1 < np.float32(4.0)))
    s = ~u
    q = {
        "q00": m0.copy(),
        "q10": np.where(u, m1, m0).astype(np.float32, copy=False),
        "q01": np.where(u, m0, m1).astype(np.float32, copy=False),
        "q11": m1.copy(),
    }
    starts = _chunk_starts(m0.shape[1])
    chunks = {name: _mel_chunks(value, starts) for name, value in q.items()}
    summary = _mel_summary(m0, m1, u, q, fixed_meta["gain"])
    if not summary["four_cell"]["float64_closure_exact"]:
        raise AssertionError("partition closure is not exact")
    destination = outdir / "arrays" / row["id"] / row["source_condition"]
    destination.mkdir(parents=True, exist_ok=True)
    fixed_path = destination / "fixed_pcm_float32.npy"
    np.save(fixed_path, fixed, allow_pickle=False)
    arrays_path = destination / "mel_partition.npz"
    array_payload: dict[str, np.ndarray] = {"M0": m0, "M1": m1, "U": u, "S": s, "mel_starts": np.asarray(starts, dtype=np.int32)}
    array_payload.update(q)
    array_payload.update({f"chunks_{key}": value for key, value in chunks.items()})
    np.savez_compressed(arrays_path, **array_payload)
    arrays_meta = {key: {"shape": list(value.shape), "dtype": str(value.dtype), "array_sha256": array_sha256(value)} for key, value in array_payload.items()}
    record = {
        "id": row["id"],
        "speaker": row["speaker"],
        "split": row["split"],
        "source_condition": row["source_condition"],
        "status": "prepared",
        "source_wav_path": raw_meta["path"],
        "source_wav_sha256": raw_meta["file_sha256"],
        "source_waveform_array_sha256": raw_meta["waveform_array_sha256"],
        "sample_rate_hz": raw_meta["sample_rate_hz"],
        "sample_count": raw_meta["samples"],
        "source_duration_s": raw_meta["duration_s"],
        "fixed_pcm_path": str(fixed_path.resolve()),
        "fixed_pcm_file_sha256": sha256_file(fixed_path),
        "fixed_pcm_array_sha256": fixed_meta["fixed_waveform_array_sha256"],
        "fixed_pcm_dtype": "float32",
        "gain": fixed_meta,
        "mel_path": str(arrays_path.resolve()),
        "mel_file_sha256": sha256_file(arrays_path),
        "arrays": arrays_meta,
        "chunk_count": len(starts),
        "mel_shape": list(m0.shape),
        "U_count": int(np.count_nonzero(u)),
        "S_count": int(np.count_nonzero(s)),
        "calibration_qa": summary if row["split"] == "calibration" else None,
    }
    return record, summary


def prepare(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir).resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"output directory must not exist or must be empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    source_path, tts_path = Path(args.input_manifest).resolve(), Path(args.tts_pair_manifest).resolve()
    spec_path, wav2lip_root = Path(args.spec).resolve(), Path(args.wav2lip_root).resolve()
    source, tts = _json(source_path), _json(tts_path)
    identity_sha = _verify_input_identity(source, tts)
    portable_source_root = Path(args.portable_source_root).resolve()
    audio, frontend_meta = _import_wav2lip_audio(wav2lip_root, portable_source_root)
    plan = _build_plan(args, source, tts, identity_sha, source_path, tts_path, spec_path, wav2lip_root, portable_source_root, frontend_meta)
    rows_jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in plan["rows"])
    plan["rows_sha256"] = hashlib.sha256(rows_jsonl.encode("utf-8")).hexdigest()
    payload_sha = hashlib.sha256(_canonical_json(plan)).hexdigest()
    plan["payload_sha256"] = payload_sha
    _atomic_json(outdir / "plan.json", plan)
    (outdir / "rows.jsonl").write_text(rows_jsonl, encoding="utf-8", newline="\n")
    if sha256_file(outdir / "rows.jsonl") != plan["rows_sha256"]:
        raise AssertionError("row plan byte hash does not match the sealed plan")
    shutil.copyfile(Path(__file__).resolve(), outdir / "producer_snapshot.py")

    artifact_rows = []
    qa_rows = []
    for row in plan["rows"]:
        if row["status"] != "preflight_pass":
            failed = {
                "id": row["id"], "speaker": row["speaker"], "split": row["split"],
                "source_condition": row["source_condition"], "status": "preflight_failed",
                "preflight_failure": row["preflight_failure"], "source_wav_path": row["wav_path"],
                "expected_source_wav_sha256": row["expected_wav_sha256"],
                "observed_source_wav_sha256": row["observed_wav_sha256"],
                "mel_path": None, "mel_file_sha256": None, "arrays": {},
                "planned_conditions": list(CONDITIONS),
            }
            artifact_rows.append(failed)
            if row["split"] == "calibration":
                qa_rows.append({"id": row["id"], "speaker": row["speaker"], "source_condition": row["source_condition"], "status": "preflight_failed", "failure": row["preflight_failure"]})
            print(f"preflight_failed {row['split']} {row['id']} {row['source_condition']} reason={row['preflight_failure']}", flush=True)
            continue
        try:
            record, summary = _prepare_artifact(row, audio, outdir)
        except Exception as exc:
            record = {
                "id": row["id"], "speaker": row["speaker"], "split": row["split"],
                "source_condition": row["source_condition"], "status": "producer_failed",
                "failure_type": type(exc).__name__, "failure_message": str(exc),
                "source_wav_path": row["wav_path"], "expected_source_wav_sha256": row["expected_wav_sha256"],
                "observed_source_wav_sha256": row["observed_wav_sha256"],
                "planned_conditions": list(CONDITIONS), "mel_path": None, "arrays": {},
            }
            artifact_rows.append(record)
            if row["split"] == "calibration":
                qa_rows.append({"id": row["id"], "speaker": row["speaker"], "source_condition": row["source_condition"], "status": "producer_failed", "failure_type": type(exc).__name__, "failure_message": str(exc)})
            print(f"producer_failed {row['split']} {row['id']} {row['source_condition']} type={type(exc).__name__} detail={exc}", flush=True)
            continue
        artifact_rows.append(record)
        if row["split"] == "calibration":
            qa_rows.append({"id": row["id"], "speaker": row["speaker"], "source_condition": row["source_condition"], "status": "prepared", **summary})
        print(f"prepared {row['split']} {row['id']} {row['source_condition']} frames={record['mel_shape'][1]} chunks={record['chunk_count']}", flush=True)

    result = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "status": "complete_cpu_preflight" if all(row.get("status") == "prepared" for row in artifact_rows) else "complete_with_row_failures",
        "created_at_utc": _utc_now(),
        "plan_path": str((outdir / "plan.json").resolve()),
        "plan_raw_sha256": sha256_file(outdir / "plan.json"),
        "rows_path": str((outdir / "rows.jsonl").resolve()),
        "rows_raw_sha256": sha256_file(outdir / "rows.jsonl"),
        "producer_snapshot_sha256": sha256_file(outdir / "producer_snapshot.py"),
        "input_identity_sha256": identity_sha,
        "split": plan["split"],
        "audio_rows": len(artifact_rows),
        "generation_rows": len(artifact_rows) * 4,
        "artifacts": artifact_rows,
        "model_forward_performed": False,
        "effects_read": False,
    }
    manifest_payload_sha = hashlib.sha256(_canonical_json(result)).hexdigest()
    result["payload_sha256"] = manifest_payload_sha
    _atomic_json(outdir / "run_manifest.json", result)
    cal_qa_status: str | None = None
    if plan["split"] == "calibration":
        qa = {
            "schema_version": 1,
            "status": "PASS" if len(qa_rows) == 40 and all(row.get("status", "prepared") == "prepared" for row in qa_rows) else "INCOMPLETE",
            "rule": "CAL20 CPU frontend/partition QA only; no model forward or effect inspection",
            "split": "calibration",
            "input_rows": len(qa_rows),
            "prepared_input_rows": sum(row.get("status", "prepared") == "prepared" for row in qa_rows),
            "source_rows": 20,
            "directions": ["natural", "tts"],
            "required_generation_rows": 160,
            "planned_generation_rows": len(qa_rows) * 4,
            "prepared_generation_rows": sum(row.get("status", "prepared") == "prepared" for row in qa_rows) * 4,
            "all_finite_in_range": all(row.get("status", "prepared") == "prepared" for row in qa_rows),
            "exact_endpoint_and_partition_copies": all(item.get("four_cell", {}).get("q00_equals_M0", False) and item.get("four_cell", {}).get("q11_equals_M1", False) and item.get("four_cell", {}).get("q10_equals_where_U_M1_else_M0", False) and item.get("four_cell", {}).get("q01_equals_where_U_M0_else_M1", False) for item in qa_rows if item.get("status", "prepared") == "prepared"),
            "float64_closure_exact": all(item.get("four_cell", {}).get("float64_closure_exact", False) for item in qa_rows if item.get("status", "prepared") == "prepared"),
            "rows": qa_rows,
            "run_manifest_sha256": sha256_file(outdir / "run_manifest.json"),
            "spec_sha256": plan["spec"]["sha256"],
            "effects_read": False,
        }
        _atomic_json(outdir / "calibration_qa.json", qa)
        cal_qa_status = str(qa["status"])
    ledger = {}
    for path in sorted(p for p in outdir.rglob("*") if p.is_file() and p.name != "all_files_sha256.json"):
        ledger[str(path.relative_to(outdir)).replace("\\", "/")] = sha256_file(path)
    _atomic_json(outdir / "all_files_sha256.json", ledger)
    if plan["split"] == "calibration" and cal_qa_status != "PASS":
        raise RuntimeError("CAL input/partition QA is incomplete; sealed intended rows and failures for review")
    print(json.dumps({"status": result["status"], "split": plan["split"], "audio_rows": len(artifact_rows), "planned_generation_rows": len(artifact_rows) * 4, "prepared_generation_rows": sum(row.get("status") == "prepared" for row in artifact_rows) * 4, "run_manifest_sha256": sha256_file(outdir / "run_manifest.json"), "calibration_qa_sha256": sha256_file(outdir / "calibration_qa.json") if (outdir / "calibration_qa.json").exists() else None}, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="create exact mel endpoints, partitions, chunks, and preflight seals")
    p.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    p.add_argument("--input-manifest", required=True, type=Path)
    p.add_argument("--tts-pair-manifest", required=True, type=Path)
    p.add_argument("--input-root", required=True, type=Path, help="directory containing audio/ and tts_outputs/")
    p.add_argument("--wav2lip-root", required=True, type=Path, help="official Wav2Lip checkout used for audio.melspectrogram")
    p.add_argument("--portable-source-root", required=True, type=Path, help="parent of the portable_source_axis_20260930 package used for the frozen librosa shim")
    p.add_argument("--spec", default=SPEC_RELATIVE, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.set_defaults(func=prepare)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
