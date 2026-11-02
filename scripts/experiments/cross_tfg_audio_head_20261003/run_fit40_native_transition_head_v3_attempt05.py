"""Fit the shared V3 transition head on the sealed forty-row fit roster.

This runner is source-only until a separate reviewed launch packet supplies the
actual completed fit40 RMS geometry manifest.  It deliberately reuses the
qualified V2 model/audio-cache loader and V4 native render/objective functions,
but not the V4 eight-row runtime factory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
import uuid as uuidlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_SCHEMA = "cross_tfg_fit40_native_transition_head_v3_training_runner_attempt02"
EXPECTED_PARENT_PLAN_SHA256 = "e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06"
EXPECTED_PARENT_PAYLOAD_SHA256 = "7200d93801ca0749b472937645d1e955e5b39c730f6a2d1a5e24d3f380f6dce9"
EXPECTED_SOURCE_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_V2_RUNTIME_SHA256 = "d086c61cdb993bffa5f9ff7a55bf4b66b1b9127028f8b80eb3b5f1be74464d7b"
EXPECTED_V4_RUNTIME_SHA256 = "dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458"
EXPECTED_V3_HEAD_SHA256 = "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87"
EXPECTED_V3_V4_ADAPTER_SHA256 = "4f4f6f496bd15e073268694673b8df5678404e672aafa625ff9a6abc135248aa"
EXPECTED_AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
EXPECTED_HEAD_CONFIG_SHA256 = "0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5"
EXPECTED_FIT_SPEAKERS = [f"S{number:04d}" for number in range(52, 72)]
EXPECTED_PAIRS = [(speaker, canonical_j) for speaker in EXPECTED_FIT_SPEAKERS for canonical_j in (0, 15)]
TARGET_RMS = 0.0376838172675746
ZERO_GATE_MAX_ABS = 1e-5
OPTIMIZER_STEPS = 120
FIT_SEED = 20261003
OBJECTIVE_WEIGHTS = {
    "relative_Cbest_improvement": 1.0,
    "far_lag_contrast_improvement": 0.25,
    "D0_nonworsening_guard": 2.0,
    "Canchor_nonworsening_guard": 1.0,
    "visual_embedding_log_norm_ratio_squared": 0.1,
    "waveform_relative_mse_to_rms_reference": 0.02,
    "mask_total_variation": 0.002,
}


class FitError(RuntimeError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return _sha(_canonical_bytes(value))


def _read_json(path: Path, expected_sha: str | None = None) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise FitError(f"expected a non-symlink JSON path: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FitError(f"expected a regular JSON file: {resolved}")
    raw = resolved.read_bytes()
    if expected_sha is not None and _sha(raw) != expected_sha:
        raise FitError(f"JSON raw SHA-256 mismatch: {resolved}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise FitError(f"expected a JSON object: {resolved}")
    return value, raw


def _pin(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise FitError(f"expected a non-symlink file path: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FitError(f"expected a regular file: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha_file(resolved)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to replace existing output: {path}")
    os.replace(temporary, path)
    return raw


def _write_status(out: Path, payload: dict[str, Any]) -> None:
    path = out / "status.json"
    temporary = path.with_name("status.json.tmp")
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    with path.open("ab") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_child(root: Path, relative: str) -> Path:
    candidate_rel = Path(relative)
    if candidate_rel.is_absolute() or ".." in candidate_rel.parts:
        raise FitError(f"unsafe relative path: {relative!r}")
    root_real = root.resolve(strict=True)
    if not root.is_dir():
        raise FitError(f"source root is not a directory: {root}")
    current = root_real
    for part in candidate_rel.parts:
        current = current / part
        if current.is_symlink():
            raise FitError(f"source member contains a symlink path component: {current}")
    candidate = (root_real / candidate_rel).resolve(strict=True)
    try:
        candidate.relative_to(root_real)
    except ValueError as exc:
        raise FitError(f"path escapes its declared root: {relative!r}") from exc
    if not candidate.is_file():
        raise FitError(f"path is not a regular file: {candidate}")
    return candidate


def _expected_source_pairs() -> list[tuple[str, int]]:
    return list(EXPECTED_PAIRS)


def _build_fit_rows(source_manifest: dict[str, Any], source_root: Path, geometry_rows: dict[str, Any]):
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    source_rows = source_manifest.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 40:
        raise FitError("fit40 source manifest must contain exactly 40 endpoint rows")
    pairs = [(str(row.get("speaker_id")), int(row.get("canonical_j", -1))) for row in source_rows]
    if pairs != _expected_source_pairs():
        raise FitError("source manifest is not in the frozen S0052–S0071 j=0/j=15 order")
    if len({str(row.get("row_id")) for row in source_rows}) != 40:
        raise FitError("fit40 source manifest row IDs are not unique")
    prepared = []
    for src in source_rows:
        row_id = str(src["row_id"])
        geom = geometry_rows.get(row_id)
        if not isinstance(geom, dict):
            raise FitError(f"native geometry manifest is missing a source row: {row_id}")
        wav_path = _safe_child(source_root, str(src["transfer_relative_path"]))
        file_pin = _pin(wav_path)
        if file_pin["bytes"] != int(src["transfer_file_bytes"]) or file_pin["sha256"] != str(src["transfer_file_sha256"]).lower():
            raise FitError(f"fit40 WAV file differs from source manifest: {row_id}")
        audio, info = read_canonical_wav(wav_path)
        expected_n = int(src["sample_count"])
        if (
            audio.dtype.name != "float32" or audio.ndim != 1 or not audio.size
            or info.file_sha256 != file_pin["sha256"]
            or info.pcm_float32_sha256 != str(src["canonical_float32_pcm_sha256"]).lower()
            or info.frames != expected_n or info.sample_rate != 16_000
            or info.source_channels != 1 or str(src.get("source_encoding")) != "PCM_16"
        ):
            raise FitError(f"fit40 WAV canonical decode differs from source manifest: {row_id}")
        pcm16 = __import__("numpy").rint(
            __import__("numpy").clip(audio.astype(__import__("numpy").float64) * 32768.0, -32768.0, 32767.0)
        ).astype("<i2")
        pcm16_sha = _sha(pcm16.tobytes())
        if pcm16_sha != str(src["source_pcm_payload_sha256"]).lower():
            raise FitError(f"fit40 direct PCM16 payload differs from frozen member: {row_id}")
        full = geom.get("full_context_fit")
        source_input = geom.get("source_input")
        if not isinstance(full, dict) or not isinstance(source_input, dict):
            raise FitError(f"fit40 geometry row lacks audio clock/context bindings: {row_id}")
        mfcc_frames = 1 if expected_n <= 400 else 1 + math.ceil((expected_n - 400) / 160)
        valid_windows = max(0, (mfcc_frames - 20) // 4 + 1)
        embedding_count = max(0, expected_n // 640 - 5)
        if (
            str(geom.get("sample_id")) != str(src["sample_id"])
            or str(geom.get("speaker_id")) != str(src["speaker_id"])
            or str(geom.get("split")) != "fit"
            or str(geom.get("method_id")) != "rms_fixed"
            or int(source_input.get("file_bytes", -1)) != file_pin["bytes"]
            or str(Path(str(source_input.get("path", ""))).resolve(strict=True)) != str(wav_path)
            or int(source_input.get("sample_count", -1)) != expected_n
            or str(source_input.get("file_sha256", "")).lower() != file_pin["sha256"]
            or str(source_input.get("canonical_float32_pcm_sha256", "")).lower()
            != str(src["canonical_float32_pcm_sha256"]).lower()
            or str(source_input.get("source_pcm16_sha256", "")).lower() != pcm16_sha
        ):
            raise FitError(f"fit40 native geometry/source row identity mismatch: {row_id}")
        if int(full.get("source_sample_count", -1)) != expected_n:
            raise FitError(f"fit40 native geometry sample count mismatch: {row_id}")
        if (
            int(full.get("psf_mfcc_frame_count", -1)) != mfcc_frames
            or int(full.get("all_valid_mfcc_window_count", -1)) != valid_windows
            or int(full.get("scoring_audio_embedding_count", -1)) != embedding_count
            or int(full.get("valid_mfcc_windows_excluded_after_official_query_end", -1)) != valid_windows - embedding_count
            or int(full.get("video_frame_count", -1)) != 32
            or int(full.get("visual_query_count", -1)) != 27
            or full.get("lag_offsets_frames") != list(range(-15, 16))
            or full.get("audio_query_rule")
            != "q0=floor((N_audio_embeddings-27)/2); 27 center queries and all 31 offsets -15..+15 must remain unpadded and in bounds"
            or str(source_input.get("source_encoding")) != "PCM_16"
            or int(source_input.get("sample_rate_hz", -1)) != 16_000
        ):
            raise FitError(f"fit40 full-context audio/query clock metadata is not the frozen 32/27/31 rule: {row_id}")
        natural = {
            "path": str(wav_path),
            "file_sha256": file_pin["sha256"],
            "pcm_sha256": str(src["canonical_float32_pcm_sha256"]).lower(),
            "direct_source_pcm16_sha256": pcm16_sha,
            "sample_count": expected_n,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "source_encoding": "original PCM16 source WAV decoded to canonical float32 for Wav2Lip; exact source PCM16 payload retained for SyncNet MFCC and mux",
        }
        fit_row = {
            "row_id": row_id,
            "sample_id": str(src["sample_id"]),
            "speaker_id": str(src["speaker_id"]),
            "split": "fit",
            "historical_exposed": True,
            "canonical_j": int(src["canonical_j"]),
            "fold_index": int(src["fold_index"]),
            "natural_pcm": natural,
            "full_context_fit": full,
            "source_manifest_row": src,
        }
        prepared.append({"row": fit_row, "source": src, "audio": audio, "pcm16": pcm16, "geometry": geom})
    if len(prepared) != 40:
        raise FitError("fit40 endpoint denominator changed")
    return prepared


def _clone_audio_cache(geometry_manifest: dict[str, Any], out_cache: Path) -> dict[str, Any]:
    cache_ref = geometry_manifest.get("audio_cache")
    if not isinstance(cache_ref, dict):
        raise FitError("geometry manifest has no pinned audio cache manifest")
    source_manifest_path = Path(str(cache_ref.get("path", "")))
    if source_manifest_path.is_symlink():
        raise FitError("producer audio cache manifest path is a symlink")
    source_manifest_path = source_manifest_path.resolve(strict=True)
    if not source_manifest_path.is_file():
        raise FitError("producer audio cache manifest is not a regular file")
    source_manifest_raw = source_manifest_path.read_bytes()
    if (
        len(source_manifest_raw) != int(cache_ref.get("bytes", -1))
        or _sha(source_manifest_raw) != str(cache_ref.get("sha256", "")).lower()
    ):
        raise FitError("producer audio cache manifest raw pin mismatch")
    source_cache = json.loads(source_manifest_raw)
    if source_cache.get("row_count") != 40 or source_cache.get("schema") != "cross_tfg_syncnet_full_audio_embedding_cache_manifest_v1":
        raise FitError("producer audio cache is not the exact 40-row V2 cache")
    cache_payload = {key: value for key, value in source_cache.items() if key != "manifest_sha256"}
    if source_cache.get("manifest_sha256") != _canonical_sha(cache_payload):
        raise FitError("producer audio cache manifest canonical payload digest mismatch")
    records = source_cache.get("rows")
    if not isinstance(records, list) or len(records) != 40:
        raise FitError("producer audio cache row manifest denominator mismatch")
    cache_row_ids = [str(item.get("row_id")) for item in records if isinstance(item, dict)]
    if len(cache_row_ids) != 40 or len(set(cache_row_ids)) != 40 or set(cache_row_ids) != {str(row["row_id"]) for row in geometry_manifest["rows"]}:
        raise FitError("producer audio cache rows do not match geometry rows")
    out_cache.mkdir(parents=True, exist_ok=False)
    copied = []
    cache_root = source_manifest_path.parent.resolve(strict=True)
    for record in records:
        for field, hash_field in (("npz_path", "npz_sha256"), ("metadata_path", "metadata_sha256")):
            source_input_path = Path(str(record.get(field, "")))
            if source_input_path.is_symlink():
                raise FitError(f"producer cache member is a symlink: {source_input_path}")
            source = source_input_path.resolve(strict=True)
            try:
                source.relative_to(cache_root)
            except ValueError as exc:
                raise FitError(f"producer cache member escapes cache root: {source}") from exc
            if not source.is_file() or _sha_file(source) != str(record.get(hash_field, "")).lower():
                raise FitError(f"producer cache member pin mismatch: {source}")
            target = out_cache / source.name
            if target.exists():
                raise FitError(f"duplicate cache member filename: {target.name}")
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            if target.stat().st_size != source.stat().st_size or _sha_file(target) != _sha_file(source):
                raise FitError(f"fresh V2 cache clone differs from producer bytes: {target.name}")
        copied.append({"row_id": str(record["row_id"]), "npz_sha256": str(record["npz_sha256"]), "metadata_sha256": str(record["metadata_sha256"])})
    return {
        "source_cache_manifest": {"path": str(source_manifest_path), "bytes": len(source_manifest_raw), "sha256": _sha(source_manifest_raw)},
        "producer_cache_rows": copied,
        "fresh_cache_root": str(out_cache.resolve()),
        "copy_policy": "hardlink when same filesystem; otherwise byte-verified copy; producer cache and manifest are never modified",
    }


def _float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().item()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite training diagnostic")
    return result


def _tensor_to_f32_np(value, torch):
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return array.astype("<f4", copy=False)


def _canonical_gpu_uuid(value: Any) -> str:
    """Normalize Torch/NVML spellings while requiring identical UUID bits."""
    if isinstance(value, uuidlib.UUID):
        text = value.hex
    elif isinstance(value, bytes):
        try:
            text = value.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise FitError("GPU UUID bytes are not ASCII") from exc
    elif isinstance(value, str):
        text = value.strip()
    elif type(value).__module__ == "torch._C" and type(value).__name__ == "_CUuuid":
        text = str(value).strip()
    else:
        raise FitError("GPU UUID value has an unsupported representation")
    if text.startswith("GPU-"):
        text = text[4:]
    compact = text.replace("-", "")
    if len(text) == 32:
        valid_shape = all(ch in "0123456789abcdefABCDEF" for ch in text)
    elif len(text) == 36:
        valid_shape = (
            text[8] == "-" and text[13] == "-" and text[18] == "-" and text[23] == "-"
            and all(ch in "0123456789abcdefABCDEF" for index, ch in enumerate(text) if index not in {8, 13, 18, 23})
        )
    else:
        valid_shape = False
    if not valid_shape or len(compact) != 32:
        raise FitError("GPU UUID is not a canonical 128-bit UUID string")
    try:
        parsed = uuidlib.UUID(text)
    except (ValueError, AttributeError) as exc:
        raise FitError("GPU UUID cannot be parsed") from exc
    if parsed.hex != compact.lower():
        raise FitError("GPU UUID canonicalization changed UUID bits")
    return "GPU-" + str(parsed)


def _required_torch_gpu_uuid(properties) -> str:
    value = getattr(properties, "uuid", None)
    if value is None:
        raise FitError("the qualified Torch CUDA device properties do not expose a GPU UUID")
    return _canonical_gpu_uuid(value)


def _curve_visual(adapter, v4, context, geometry_record, waveform):
    rendered = v4._render_native_video(adapter, context, waveform, geometry_record)
    curve = adapter.sync_distance_curve(rendered, context=context)
    visual = adapter.last_visual_embeddings
    audio_center = adapter.last_audio_center_embeddings
    if visual is None or audio_center is None:
        raise FitError("native V4 scorer did not expose visual/audio embeddings")
    if tuple(curve.shape) != (31,) or tuple(visual.shape) != (27, 1024):
        raise FitError("native V4 scorer returned unexpected 31-lag/27-query dimensions")
    if not adapter.torch.isfinite(curve).all().item() or not adapter.torch.isfinite(visual).all().item():
        raise FloatingPointError("native V4 scorer returned non-finite features")
    return curve, visual, audio_center


def _row_geometry_record(native_geometries: dict[str, Any], row_id: str) -> dict[str, Any]:
    entry = native_geometries[row_id]
    return {
        "geometry": entry["geometry"],
        "prediction_source_clock_frames": entry["prediction_source_clock_frames"],
    }


def _verify_baseline_wav(row: dict[str, Any], geometry_row: dict[str, Any], natural_audio, baselines_module):
    baseline_record = geometry_row.get("rms_fixed_baseline")
    if not isinstance(baseline_record, dict):
        raise FitError(f"native row lacks its RMS baseline binding: {row['row_id']}")
    if (
        baseline_record.get("method_id") != "rms_fixed"
        or baseline_record.get("peak_ceiling") is not None
        or baseline_record.get("peak_limited") is not False
        or baseline_record.get("audio_time_warped") is not False
        or abs(float(baseline_record.get("target_rms", -1.0)) - TARGET_RMS) > 1e-15
    ):
        raise FitError(f"RMS baseline policy differs from the frozen unbounded target: {row['row_id']}")
    wav_pin = baseline_record.get("rms_input_float32_wav")
    if not isinstance(wav_pin, dict) or set(wav_pin) != {"path", "bytes", "sha256"}:
        raise FitError(f"RMS baseline waveform pin is malformed: {row['row_id']}")
    wav_path = Path(str(wav_pin["path"]))
    if wav_path.is_symlink():
        raise FitError(f"RMS baseline waveform path is a symlink: {row['row_id']}")
    wav_path = wav_path.resolve(strict=True)
    if not wav_path.is_file():
        raise FitError(f"RMS baseline waveform is not a regular file: {row['row_id']}")
    wav_pin_actual = _pin(wav_path)
    if wav_pin_actual["bytes"] != int(wav_pin["bytes"]) or wav_pin_actual["sha256"] != str(wav_pin["sha256"]).lower():
        raise FitError(f"RMS baseline WAV raw pin mismatch: {row['row_id']}")
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    saved_audio, info = read_canonical_wav(wav_path)
    recomputed, rms_info = baselines_module.apply_global_rms(
        natural_audio, TARGET_RMS, peak_ceiling=None
    )
    pcm_sha = _sha(recomputed.astype("<f4", copy=False).tobytes())
    if (
        saved_audio.dtype != recomputed.dtype or saved_audio.shape != recomputed.shape
        or not __import__("numpy").array_equal(saved_audio, recomputed)
        or info.pcm_float32_sha256 != pcm_sha
        or pcm_sha != str(baseline_record.get("output_float32_pcm_sha256", "")).lower()
        or bool(rms_info.peak_limited)
    ):
        raise FitError(f"saved geometry RMS waveform differs from exact unbounded canonical RMS: {row['row_id']}")
    return recomputed, {
        "path": str(wav_path),
        "file_bytes": wav_pin_actual["bytes"],
        "file_sha256": wav_pin_actual["sha256"],
        "pcm_float32_sha256": pcm_sha,
        "input_rms": float(rms_info.input_rms),
        "requested_gain": float(rms_info.requested_gain),
        "applied_gain": float(rms_info.applied_gain),
        "output_rms": float(rms_info.output_rms),
        "peak_limited": bool(rms_info.peak_limited),
    }


def _make_v3_parameter(head, torch) -> dict[str, Any]:
    values = _tensor_to_f32_np(head.band_contrasts_db, torch).astype("<f4", copy=False)
    return {
        "name": "band_contrasts_db",
        "shape": [3, 3],
        "dtype": "float32",
        "temporal_components": ["static", "positive_flux", "negative_flux"],
        "non_reference_band_order": [0, 1, 2],
        "values": values.tolist(),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--expected-parent-plan-sha256", required=True)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--expected-geometry-manifest-sha256", required=True)
    parser.add_argument("--head-config", required=True, type=Path)
    parser.add_argument("--expected-head-config-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, help="explicit cuda or cuda:N; CPU fallback is forbidden")
    return parser.parse_args(argv)


def run(args) -> dict[str, Any]:
    if args.output_dir.is_symlink():
        raise FitError("fit40 output directory may not be a symlink")
    out = args.output_dir.resolve(strict=False)
    allowed_root = Path("/dev/shm/cross_tfg_audio_head_20261003").resolve(strict=False)
    try:
        out.relative_to(allowed_root)
    except ValueError as exc:
        raise FitError("fit40 output must stay in the dedicated /dev/shm task namespace") from exc
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise FitError(f"fit40 output directory must be new or empty: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)
    status = {
        "schema": EXPECTED_SCHEMA,
        "status": "RUNNING_INPUT_BINDING",
        "stage": "input_binding",
        "started_unix_seconds": time.time(),
        "completed_optimizer_steps": 0,
        "planned_optimizer_steps": OPTIMIZER_STEPS,
    }
    _write_status(out, status)
    active_stage = "input_binding"
    completed_optimizer_steps = 0
    try:
        if args.expected_parent_plan_sha256.lower() != EXPECTED_PARENT_PLAN_SHA256:
            raise FitError("caller parent-plan SHA differs from the sealed exact V2 plan")
        if args.expected_source_manifest_sha256.lower() != EXPECTED_SOURCE_MANIFEST_SHA256:
            raise FitError("caller source-manifest SHA differs from the sealed ADR013 fit40 roster")
        if args.expected_head_config_sha256.lower() != EXPECTED_HEAD_CONFIG_SHA256:
            raise FitError("caller head-constructor config SHA differs from the qualified V3 config")
        if (
            len(args.expected_geometry_manifest_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in args.expected_geometry_manifest_sha256.lower())
        ):
            raise FitError("geometry raw SHA pin must be supplied after actual producer completion")
        if args.device != "cuda" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise FitError("an explicit CUDA device is required; no CPU fallback")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise FitError("CUBLAS_WORKSPACE_CONFIG must be set before PyTorch import")
        for env_name in ("CROSS_TFG_PSF_ROOT", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED"):
            if not os.environ.get(env_name):
                raise FitError(f"required pinned runtime environment is absent: {env_name}")
        plan, plan_raw = _read_json(args.parent_plan, EXPECTED_PARENT_PLAN_SHA256)
        plan_payload = {key: value for key, value in plan.items() if key != "plan_payload_sha256"}
        if plan.get("plan_payload_sha256") != EXPECTED_PARENT_PAYLOAD_SHA256 or _canonical_sha(plan_payload) != EXPECTED_PARENT_PAYLOAD_SHA256:
            raise FitError("exact V2 parent plan canonical payload digest mismatch")
        source_manifest, source_raw = _read_json(args.source_manifest, EXPECTED_SOURCE_MANIFEST_SHA256)
        if source_manifest.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1":
            raise FitError("fit40 source manifest schema is not the frozen ADR013 roster")
        if source_manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
            raise FitError("fit40 source manifest is not a source-only materialization")
        scope = source_manifest.get("scope", {})
        if any(scope.get(key) is not False for key in ("candidate_transform_executed", "confirm_audio_read", "dev_audio_read", "model_forward", "render", "score")):
            raise FitError("fit40 source manifest scope includes a treatment/dev/confirmation effect")
        geometry, geometry_raw = _read_json(args.geometry_manifest, args.expected_geometry_manifest_sha256.lower())
        if geometry.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1" or geometry.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY":
            raise FitError("native RMS geometry manifest is not a completed fit-only artifact")
        geometry_payload = {key: value for key, value in geometry.items() if key != "payload_sha256"}
        if geometry.get("payload_sha256") != _canonical_sha(geometry_payload):
            raise FitError("native geometry canonical payload digest mismatch")
        if int(geometry.get("fit_rows", -1)) != 40 or geometry.get("method_id") != "rms_fixed":
            raise FitError("native geometry must contain exactly the 40 fit RMS controls")
        if (int(geometry.get("source_clock_fps_num", -1)), int(geometry.get("source_clock_fps_den", -1))) != (25, 1):
            raise FitError("fit40 geometry changed the frozen 25/1 source frame clock")
        if geometry.get("audio_clock_policy") != "unchanged_original_natural_pcm_mfcc_queries":
            raise FitError("fit40 geometry may not move the original natural-audio query clock")
        if geometry.get("baseline_control", {}).get("peak_ceiling") is not None or float(geometry.get("baseline_control", {}).get("target_rms", -1.0)) != TARGET_RMS:
            raise FitError("fit40 geometry baseline is not the exact unbounded fixed-RMS control")
        if geometry.get("source_pins", {}).get("source_manifest", {}).get("sha256") != EXPECTED_SOURCE_MANIFEST_SHA256:
            raise FitError("geometry manifest does not bind the exact fit40 source manifest")
        if geometry.get("source_pins", {}).get("v2_runtime_plan", {}).get("sha256") != EXPECTED_PARENT_PLAN_SHA256:
            raise FitError("geometry manifest does not bind the exact original V2 parent plan")
        geometry_rows = geometry.get("rows")
        if not isinstance(geometry_rows, list) or len(geometry_rows) != 40:
            raise FitError("geometry row denominator is not exactly 40")
        geometry_by_id = {str(row.get("row_id")): row for row in geometry_rows if isinstance(row, dict)}
        if len(geometry_by_id) != 40:
            raise FitError("geometry row IDs are not unique")
        geometry_order = [str(row["row_id"]) for row in geometry_rows]
        prepared = _build_fit_rows(source_manifest, args.source_root, geometry_by_id)
        row_ids = [item["row"]["row_id"] for item in prepared]
        if row_ids != geometry_order or set(row_ids) != set(geometry_by_id):
            raise FitError("geometry rows must exactly match the frozen source roster and order")
        head_config, head_config_raw = _read_json(args.head_config, EXPECTED_HEAD_CONFIG_SHA256)
        constructor = head_config.get("head_constructor")
        if not isinstance(constructor, dict) or constructor.get("target_rms") != TARGET_RMS or constructor.get("peak_ceiling") is not None:
            raise FitError("V3 constructor must use the exact target RMS and no peak cap")
        if head_config.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_config_v1":
            raise FitError("V3 constructor config schema differs from its qualified frozen source")
        if int(constructor.get("sample_rate_hz", -1)) != 16_000 or int(constructor.get("video_fps_num", -1)) != 25 or int(constructor.get("video_fps_den", -1)) != 1:
            raise FitError("V3 constructor changed the frozen sample/video clocks")

        active_stage = "audio_cache_clone"
        cache_root = out / "input_cache_clone"
        cache_clone_record = _clone_audio_cache(geometry, cache_root)
        v2_plan = copy.deepcopy(plan)
        v2_plan["runtime"]["full_context"]["audio_embedding_cache_root"] = str(cache_root)
        v2_plan["runtime"]["full_context"]["source_sample_count_total"] = sum(int(item["row"]["natural_pcm"]["sample_count"]) for item in prepared)
        v2_plan["fit_config"]["selected_rows"] = [str(item["row"]["row_id"]) for item in prepared]
        v2_plan["fit_config"]["source_sample_count_total"] = v2_plan["runtime"]["full_context"]["source_sample_count_total"]
        v2_runtime_cfg = copy.deepcopy(v2_plan["runtime"])
        v2_runtime_cfg["fit_v4"] = {
            "schema": "cross_tfg_wav2lip_fullcontext_fit_runtime_v4_attempt01",
            "native_geometry_manifest_path": str(args.geometry_manifest.resolve(strict=True)),
            "native_geometry_manifest_sha256": _sha(geometry_raw),
        }

        active_stage = "runtime_model_load"
        import torch
        import torch.nn.functional as functional
        import numpy as np
        from scripts.audio_enhancement import baselines
        from scripts.audio_enhancement import shared_recipient_transition_head_v3 as v3_head
        from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v2_attempt02 as v2
        from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v4_attempt01 as v4
        from scripts.audio_enhancement import v3_v4_native_runtime_adapter_attempt03 as v3_v4_adapter

        module_pins = {
            "v2_runtime": {"path": str(Path(v2.__file__).resolve()), "bytes": Path(v2.__file__).stat().st_size, "sha256": _sha_file(Path(v2.__file__).resolve())},
            "v4_runtime": {"path": str(Path(v4.__file__).resolve()), "bytes": Path(v4.__file__).stat().st_size, "sha256": _sha_file(Path(v4.__file__).resolve())},
            "v3_head": {"path": str(Path(v3_head.__file__).resolve()), "bytes": Path(v3_head.__file__).stat().st_size, "sha256": _sha_file(Path(v3_head.__file__).resolve())},
            "v3_v4_adapter": {"path": str(Path(v3_v4_adapter.__file__).resolve()), "bytes": Path(v3_v4_adapter.__file__).stat().st_size, "sha256": _sha_file(Path(v3_v4_adapter.__file__).resolve())},
            "audio_io": {"path": str(REPO_ROOT / "scripts/audio_enhancement/audio_io.py"), "bytes": (REPO_ROOT / "scripts/audio_enhancement/audio_io.py").stat().st_size, "sha256": _sha_file(REPO_ROOT / "scripts/audio_enhancement/audio_io.py")},
            "baselines": {"path": str(Path(baselines.__file__).resolve()), "bytes": Path(baselines.__file__).stat().st_size, "sha256": _sha_file(Path(baselines.__file__).resolve())},
            "enhance_v3_cli": {"path": str(REPO_ROOT / "scripts/audio_enhancement/enhance_v3_attempt02.py"), "bytes": (REPO_ROOT / "scripts/audio_enhancement/enhance_v3_attempt02.py").stat().st_size, "sha256": _sha_file(REPO_ROOT / "scripts/audio_enhancement/enhance_v3_attempt02.py")},
            "fit40_geometry_producer": {"path": str(REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt04.py"), "bytes": (REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt04.py").stat().st_size, "sha256": _sha_file(REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt04.py")},
        }
        expected_module_pins = {
            "v2_runtime": EXPECTED_V2_RUNTIME_SHA256,
            "v4_runtime": EXPECTED_V4_RUNTIME_SHA256,
            "v3_head": EXPECTED_V3_HEAD_SHA256,
            "v3_v4_adapter": EXPECTED_V3_V4_ADAPTER_SHA256,
            "audio_io": EXPECTED_AUDIO_IO_SHA256,
            "baselines": EXPECTED_BASELINES_SHA256,
            "enhance_v3_cli": "7a560332ce86a82ffede4254b015cabf5f10176e6f601bfc172a331f7bb885e6",
            "fit40_geometry_producer": "0ecfd1f3be25f6c56d76fa6fecd1ab5dd9b1868e03b5394f355939e74dc14df7",
        }
        for name, expected in expected_module_pins.items():
            if module_pins[name]["sha256"] != expected:
                raise FitError(f"pinned training dependency changed: {name}")
        producer_pin = geometry.get("source_pins", {}).get("geometry_producer")
        local_producer_pin = module_pins["fit40_geometry_producer"]
        if (
            not isinstance(producer_pin, dict)
            or int(producer_pin.get("bytes", -1)) != local_producer_pin["bytes"]
            or str(producer_pin.get("sha256", "")).lower() != local_producer_pin["sha256"]
        ):
            raise FitError("geometry manifest does not bind the exact locally pinned producer source")
        if _sha(head_config_raw) != EXPECTED_HEAD_CONFIG_SHA256:
            raise FitError("head config changed during its initial verification")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(FIT_SEED)
        np.random.seed(FIT_SEED)
        torch.manual_seed(FIT_SEED)
        torch.cuda.manual_seed_all(FIT_SEED)
        if not torch.cuda.is_available():
            raise FitError("frozen fit runtime requires CUDA; CPU fallback is forbidden")
        device = torch.device(args.device)
        if device.type != "cuda" or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise FitError(f"requested CUDA device is not available: {args.device}")
        current_device = torch.cuda.current_device()
        if device.index is not None and current_device != device.index:
            torch.cuda.set_device(device)
            current_device = torch.cuda.current_device()
        device = torch.device("cuda", current_device)
        if list(sys.version_info[:3]) != [3, 10, 12]:
            raise FitError(f"runtime Python differs from the pinned environment: {list(sys.version_info[:3])}")
        if str(torch.__version__) != "2.5.1+cu124" or str(np.__version__) != "1.26.4":
            raise FitError("runtime Torch/NumPy versions differ from the pinned CUDA environment")
        if torch.cuda.is_initialized() is False:
            # model load below is expected to initialize the selected device.
            pass
        runtime = v2.create_runtime(plan=v2_plan, fit_rows=[item["row"] for item in prepared], device=str(device), seed=FIT_SEED)
        runtime_pins = runtime["source_pins"]
        if not isinstance(runtime_pins, dict):
            raise FitError("V2 runtime did not return its qualified source pins")
        adapter = runtime["adapter"]
        for model in (adapter.wav2lip_model, adapter.syncnet_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        if not torch.cuda.is_initialized():
            raise FitError("model runtime did not initialize the explicitly selected CUDA device")
        gpu_props = torch.cuda.get_device_properties(current_device)
        gpu_uuid = _required_torch_gpu_uuid(gpu_props)
        gpu_identity = {
            "uuid": gpu_uuid,
            "cuda_device_index": int(current_device),
            "device_name": str(gpu_props.name),
            "total_memory_bytes": int(gpu_props.total_memory),
            "torch_cuda_version": str(torch.version.cuda),
            "runtime_device": str(device),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "python_version": list(sys.version_info[:3]),
            "torch_num_threads": int(torch.get_num_threads()),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        }
        producer_gpu_identity = geometry.get("runtime_gpu_identity")
        if (
            not isinstance(producer_gpu_identity, dict)
            or producer_gpu_identity.get("schema") != "cross_tfg_runtime_gpu_identity_v1"
        ):
            raise FitError("fit40 RMS geometry lacks the declared GPU/runtime identity receipt")
        for row in geometry_rows:
            row_identity = row.get("runtime_gpu_identity")
            if not isinstance(row_identity, dict) or _canonical_bytes(row_identity) != _canonical_bytes(producer_gpu_identity):
                raise FitError(f"fit40 geometry row and manifest GPU/runtime identities differ: {row.get('row_id')}")
        full_context_cfg = v2_plan["runtime"]["full_context"]
        expected_batch_policy = {
            "video_frame_batch_size": int(full_context_cfg["video_frame_batch_size"]),
            "audio_embedding_batch_size": int(full_context_cfg["audio_embedding_batch_size"]),
            "syncnet_batch_size": int(full_context_cfg["syncnet_batch_size"]),
        }
        if (
            int(adapter.frame_batch_size) != expected_batch_policy["video_frame_batch_size"]
            or int(adapter.syncnet_batch_size) != expected_batch_policy["syncnet_batch_size"]
            or int(runtime_pins.get("full_context_audio_embedding_cache", {}).get("batch_size", -1))
            != expected_batch_policy["audio_embedding_batch_size"]
        ):
            raise FitError("loaded native renderer/scorer batch settings differ from the pinned V2 runtime plan")
        producer_profile = producer_gpu_identity.get("runtime_profile")
        if not isinstance(producer_profile, dict):
            raise FitError("fit40 producer GPU identity lacks its qualified runtime profile")
        expected_gpu_identity = {
            "uuid": gpu_uuid,
            "device_index": int(current_device),
            "device_name": str(gpu_props.name),
            "memory_total_bytes": int(gpu_props.total_memory),
            "memory_total_mib": int(gpu_props.total_memory) // (1024 * 1024),
            "device_literal": args.device,
            "torch_device_name": str(gpu_props.name),
            "torch_total_memory_bytes": int(gpu_props.total_memory),
            "torch_compute_capability": [int(gpu_props.major), int(gpu_props.minor)],
        }
        for key, expected in expected_gpu_identity.items():
            actual = producer_gpu_identity.get(key)
            if key == "uuid" and actual is not None:
                actual = _canonical_gpu_uuid(actual)
            if _canonical_bytes(actual) != _canonical_bytes(expected):
                raise FitError(f"fit40 geometry producer and optimizer GPU/runtime identity differ: {key}")
        expected_runtime_profile = {
            "python_executable_literal": str(sys.executable),
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "seed": FIT_SEED,
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "batch_policy": {
                "wav2lip_native_frame_batch_size": expected_batch_policy["video_frame_batch_size"],
                "syncnet_audio_embedding_batch_size": expected_batch_policy["audio_embedding_batch_size"],
                "syncnet_visual_embedding_batch_size": expected_batch_policy["syncnet_batch_size"],
            },
            "frame_rate": {"numerator": 25, "denominator": 1},
            "source_sample_rate_hz": 16_000,
            "wav2lip_git_head": str(plan["runtime"]["wav2lip"]["git_head"]),
            "wav2lip_checkpoint_sha256": str(plan["runtime"]["wav2lip"]["checkpoint_sha256"]),
            "syncnet_git_head": str(plan["runtime"]["syncnet"]["git_head"]),
            "syncnet_checkpoint_sha256": str(plan["runtime"]["syncnet"]["checkpoint_sha256"]),
            "fixed_portrait_sha256": str(geometry.get("source_portrait", {}).get("sha256", "")).lower(),
        }
        for key, expected in expected_runtime_profile.items():
            if _canonical_bytes(producer_profile.get(key)) != _canonical_bytes(expected):
                raise FitError(f"fit40 geometry producer and optimizer runtime profile differ: {key}")
        if runtime_pins.get("v2_runtime", {}).get("sha256") != EXPECTED_V2_RUNTIME_SHA256:
            raise FitError("V2 runtime self-pin differs from the frozen source")
        producer_model_pins = geometry.get("source_pins", {}).get("runtime_models_and_frontends")
        if not isinstance(producer_model_pins, dict):
            raise FitError("native RMS geometry lacks the pinned model/frontend source record")
        required_model_pin_keys = (
            "wav2lip_git_head", "wav2lip_sources", "wav2lip_checkpoint",
            "syncnet_git_head", "syncnet_sources", "syncnet_checkpoint",
            "official_frontend", "mel_spec", "python_speech_features",
            "torch_version", "numpy_version", "runtime_factory_path", "runtime_factory_sha256",
        )
        for key in required_model_pin_keys:
            if key not in producer_model_pins or key not in runtime_pins:
                raise FitError(f"producer/current renderer model identity lacks required source pin: {key}")
            if _canonical_bytes(producer_model_pins[key]) != _canonical_bytes(runtime_pins[key]):
                raise FitError(f"fit40 RMS geometry was generated with a different model/frontend input: {key}")
        geometry_runtime_cfg = v2_plan["runtime"].setdefault("fit_v4", v2_runtime_cfg["fit_v4"])
        manifest, native_geometries, native_geometry_records = v4._load_native_geometry_manifest(
            v2_plan, [item["row"] for item in prepared], geometry_runtime_cfg
        )
        geometry_record_by_row = {str(item["row_id"]): item for item in native_geometry_records}
        if set(native_geometries) != set(row_ids) or set(geometry_record_by_row) != set(row_ids):
            raise FitError("V4 native geometry loader did not return the exact 40 fit rows")

        # Use the loader only for qualified frozen model/context/audio components.
        # Its V2 train head and optimizer are not stepped by this V3 runner.
        del runtime["optimizer"]
        del runtime["enhancement_stack"]
        base_context_for_row = runtime["context_for_row"]

        active_stage = "baseline_native_scoring"
        _write_status(out, {
            **status, "status": "RUNNING_RMS_BASELINE_NATIVE_SCORING", "stage": active_stage,
            "started_unix_seconds": status["started_unix_seconds"], "completed_optimizer_steps": 0,
            "planned_optimizer_steps": OPTIMIZER_STEPS, "gpu_identity": gpu_identity,
        })
        contexts: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        baseline_waveforms: dict[str, Any] = {}
        baseline_curves: dict[str, Any] = {}
        baseline_visuals: dict[str, Any] = {}
        baseline_records = []
        baseline_curves_np = []
        baseline_visuals_np = []
        baseline_queries_np = []
        baseline_jsonl = out / "rms_baselines.jsonl"
        if baseline_jsonl.exists():
            raise FitError("baseline ledger already exists; this runner never resumes a partial attempt")
        with torch.no_grad():
            for index, item in enumerate(prepared):
                row = item["row"]
                row_id = str(row["row_id"])
                context = base_context_for_row(row)
                cache_source_row = next(entry for entry in manifest["rows"] if str(entry["row_id"]) == row_id)
                expected_cache = cache_source_row["natural_audio_embedding_cache"]
                actual_cache = context.cache_record
                if (
                    str(actual_cache.get("npz_sha256")) != str(expected_cache.get("npz_sha256"))
                    or str(actual_cache.get("metadata_sha256")) != str(expected_cache.get("metadata_sha256"))
                ):
                    raise FitError(f"V2 loader audio cache differs from the geometry producer cache: {row_id}")
                rms_audio_np, rms_info = _verify_baseline_wav(row, cache_source_row, item["audio"], baselines)
                rms_tensor = torch.from_numpy(np.ascontiguousarray(rms_audio_np)).to(device=device, dtype=torch.float32)
                geometry_record = _row_geometry_record(native_geometries, row_id)
                curve, visual, audio_center = _curve_visual(adapter, v4, context, geometry_record, rms_tensor)
                curve_np = _tensor_to_f32_np(curve, torch)
                visual_np = _tensor_to_f32_np(visual, torch)
                if curve_np.shape != (31,) or visual_np.shape != (27, 1024):
                    raise FitError(f"RMS native baseline dimensions differ from frozen scoring support: {row_id}")
                metrics = {key: _float(value) for key, value in v4._curve_metrics(torch, curve).items()}
                vnorm, _, cos0 = v4._embedding_metrics(torch, functional, visual, audio_center)
                metrics["mean_visual_embedding_l2"] = _float(vnorm.mean())
                metrics["mean_zero_lag_unit_cosine"] = _float(cos0.mean())
                baseline_waveforms[row_id] = rms_tensor
                baseline_curves[row_id] = curve.detach()
                baseline_visuals[row_id] = visual.detach()
                contexts[row_id] = context
                sources[row_id] = torch.from_numpy(np.ascontiguousarray(item["audio"])).to(device=device, dtype=torch.float32)
                record = {
                    "sequence_index": index,
                    "row_id": row_id,
                    "sample_id": row["sample_id"],
                    "speaker_id": row["speaker_id"],
                    "canonical_j": row["canonical_j"],
                    "source_pcm_file_sha256": row["natural_pcm"]["file_sha256"],
                    "source_pcm_float32_sha256": row["natural_pcm"]["pcm_sha256"],
                    "source_pcm16_sha256": row["natural_pcm"]["direct_source_pcm16_sha256"],
                    "source_sample_count": row["natural_pcm"]["sample_count"],
                    "audio_query_start_frame": int(context.query_start_frame),
                    "audio_embedding_count": int(context.audio_embedding_count),
                    "geometry_record_sha256": _canonical_sha(geometry_record_by_row[row_id]),
                    "producer_rms_waveform": rms_info,
                    "baseline_metrics": metrics,
                    "curve31_float32": curve_np.tolist(),
                }
                baseline_records.append(record)
                baseline_curves_np.append(curve_np)
                baseline_visuals_np.append(visual_np)
                baseline_queries_np.append(np.arange(context.query_start_frame, context.query_start_frame + 27, dtype=np.int32))
                _append_jsonl(baseline_jsonl, record)
                _write_status(out, {
                    **status, "status": "RUNNING_RMS_BASELINE_NATIVE_SCORING", "stage": active_stage,
                    "completed_baseline_rows": index + 1, "expected_baseline_rows": 40,
                    "completed_optimizer_steps": 0, "planned_optimizer_steps": OPTIMIZER_STEPS,
                    "gpu_identity": gpu_identity,
                })
        if len(baseline_records) != 40:
            raise FitError("native RMS baseline scoring did not complete all 40 rows")

        baseline_arrays_path = out / "rms_baseline_score_arrays.npz"
        with baseline_arrays_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                row_ids=np.asarray(row_ids, dtype="U96"),
                rms_curves_float32=np.stack(baseline_curves_np).astype("<f4", copy=False),
                rms_visual_embeddings_float32=np.stack(baseline_visuals_np).astype("<f4", copy=False),
                query_frame_indices_int32=np.stack(baseline_queries_np).astype("<i4", copy=False),
            )
            handle.flush()
            os.fsync(handle.fileno())

        active_stage = "zero_head_same_gpu_gate"
        _write_status(out, {
            **status, "status": "RUNNING_ZERO_HEAD_SAME_GPU_GATE", "stage": active_stage,
            "completed_baseline_rows": 40, "expected_baseline_rows": 40,
            "completed_zero_gate_rows": 0, "expected_zero_gate_rows": 40,
            "completed_optimizer_steps": 0, "planned_optimizer_steps": OPTIMIZER_STEPS,
            "gpu_identity": gpu_identity,
        })
        head = v3_head.SharedRecipientTransitionHeadV3(**constructor).to(device=device)
        head.eval()
        with torch.no_grad():
            if not torch.equal(head.band_contrasts_db, torch.zeros_like(head.band_contrasts_db)):
                raise FitError("V3 head must begin at exact all-zero theta")
        gate_records = []
        gate_curves_np = []
        gate_visuals_np = []
        zero_gate_jsonl = out / "zero_head_gate.jsonl"
        if zero_gate_jsonl.exists():
            raise FitError("zero-gate ledger already exists; this runner never resumes a partial attempt")
        for index, item in enumerate(prepared):
            row = item["row"]
            row_id = str(row["row_id"])
            source_tensor = sources[row_id]
            baseline_tensor = baseline_waveforms[row_id]
            context = contexts[row_id]
            v3_result = head(source_tensor, baseline_tensor)
            output_np = _tensor_to_f32_np(v3_result["output_pcm_float32"], torch)
            baseline_np = _tensor_to_f32_np(baseline_tensor, torch)
            waveform_exact = bool(np.array_equal(output_np, baseline_np))
            waveform_max = float(np.max(np.abs(output_np.astype(np.float64) - baseline_np.astype(np.float64))))
            if not waveform_exact:
                raise FitError(f"zero-head waveform is not bit-identical to the RMS baseline: {row_id}")
            with torch.no_grad():
                curve, visual, audio_center = _curve_visual(
                    adapter, v4, context, _row_geometry_record(native_geometries, row_id), v3_result["output_pcm_float32"]
                )
            curve_diff = torch.max(torch.abs(curve - baseline_curves[row_id]))
            visual_diff = torch.max(torch.abs(visual - baseline_visuals[row_id]))
            curve_max = _float(curve_diff)
            visual_max = _float(visual_diff)
            candidate_metrics = {key: _float(value) for key, value in v4._curve_metrics(torch, curve).items()}
            baseline_metrics = {key: _float(value) for key, value in v4._curve_metrics(torch, baseline_curves[row_id]).items()}
            max_mask = _float(v3_result["postprojection_max_abs_mask_db"])
            checks = {
                "waveform_bit_equal_to_same_device_rms": waveform_exact,
                "waveform_max_abs": waveform_max,
                "visual_embedding_max_abs": visual_max,
                "curve31_max_abs": curve_max,
                "projected_mask_max_abs_db": max_mask,
                "same_cuda_device": str(next(adapter.wav2lip_model.parameters()).device) == str(device),
                "syncnet_on_same_cuda_device": str(next(adapter.syncnet_model.parameters()).device) == str(device),
                "visual_embedding_within_1e_5": visual_max <= ZERO_GATE_MAX_ABS,
                "curve31_within_1e_5": curve_max <= ZERO_GATE_MAX_ABS,
                "zero_mask": max_mask == 0.0,
            }
            passed = all(checks[key] for key in (
                "waveform_bit_equal_to_same_device_rms", "same_cuda_device", "syncnet_on_same_cuda_device",
                "visual_embedding_within_1e_5", "curve31_within_1e_5", "zero_mask",
            ))
            record = {
                "sequence_index": index,
                "row_id": row_id,
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "canonical_j": row["canonical_j"],
                "checks": checks,
                "baseline_metrics": baseline_metrics,
                "zero_head_metrics": candidate_metrics,
                "curve31_float32": _tensor_to_f32_np(curve, torch).tolist(),
                "passed": passed,
            }
            gate_records.append(record)
            gate_curves_np.append(_tensor_to_f32_np(curve, torch))
            gate_visuals_np.append(_tensor_to_f32_np(visual, torch))
            _append_jsonl(zero_gate_jsonl, record)
            _write_status(out, {
                **status, "status": "RUNNING_ZERO_HEAD_SAME_GPU_GATE", "stage": active_stage,
                "completed_baseline_rows": 40, "expected_baseline_rows": 40,
                "completed_zero_gate_rows": index + 1, "expected_zero_gate_rows": 40,
                "zero_gate_passed_so_far": sum(1 for value in gate_records if value["passed"]),
                "completed_optimizer_steps": 0, "planned_optimizer_steps": OPTIMIZER_STEPS,
                "gpu_identity": gpu_identity,
            })
            if not passed:
                raise FitError(f"same-GPU zero-head RMS gate failed for {row_id}; no V3 optimizer was created or stepped")
        if len(gate_records) != 40 or not all(row["passed"] for row in gate_records):
            raise FitError("same-GPU zero-head gate did not pass all 40 planned rows")
        zero_arrays_path = out / "zero_head_score_arrays.npz"
        with zero_arrays_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                row_ids=np.asarray(row_ids, dtype="U96"),
                zero_head_curves_float32=np.stack(gate_curves_np).astype("<f4", copy=False),
                zero_head_visual_embeddings_float32=np.stack(gate_visuals_np).astype("<f4", copy=False),
                rms_curves_float32=np.stack(baseline_curves_np).astype("<f4", copy=False),
                rms_visual_embeddings_float32=np.stack(baseline_visuals_np).astype("<f4", copy=False),
            )
            handle.flush()
            os.fsync(handle.fileno())

        # V2 creates an unused legacy optimizer while loading its frozen runtime;
        # that object was discarded above and is never stepped. The V3 optimizer
        # is created only after the complete all-40 same-GPU identity/RMS gate.
        active_stage = "optimizer_step"
        optimizer = torch.optim.Adam(head.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)
        step_jsonl = out / "training_steps.jsonl"
        if step_jsonl.exists():
            raise FitError("training step ledger already exists; this runner never resumes or selects a prior step")
        step_records = []
        train_curves_np = []
        train_visuals_np = []
        train_metrics = []
        step_index = 0
        start_train = time.time()
        for pass_index in range(3):
            for row_index, item in enumerate(prepared):
                step_index += 1
                row = item["row"]
                row_id = str(row["row_id"])
                source_tensor = sources[row_id]
                baseline_tensor = baseline_waveforms[row_id]
                context = contexts[row_id]
                theta_before = _make_v3_parameter(head, torch)["values"]
                optimizer.zero_grad(set_to_none=True)
                v3_result = head(source_tensor, baseline_tensor)
                objective = v3_v4_adapter.v4_objective_for_v3_candidate(
                    torch=torch,
                    functional=functional,
                    v4_runtime=v4,
                    v4_adapter=adapter,
                    context=context,
                    geometry_record=_row_geometry_record(native_geometries, row_id),
                    v3_result=v3_result,
                    rms_curve=baseline_curves[row_id],
                    rms_visual_embeddings=baseline_visuals[row_id],
                    objective_weights=OBJECTIVE_WEIGHTS,
                )
                loss = objective["loss"]
                if loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise FloatingPointError(f"non-finite/non-scalar V4 objective at step {step_index}")
                loss.backward()
                grad = head.band_contrasts_db.grad
                if grad is None or grad.shape != (3, 3) or not torch.isfinite(grad).all().item():
                    raise FloatingPointError(f"V3 nine-parameter gradient is absent or non-finite at step {step_index}")
                grad_l2 = _float(torch.linalg.vector_norm(grad))
                grad_max = _float(grad.abs().max())
                metrics = {name: _float(value) for name, value in objective["metrics"].items()}
                output_np = _tensor_to_f32_np(v3_result["output_pcm_float32"], torch)
                curve_np = _tensor_to_f32_np(objective["score"]["curve_float32"], torch)
                visual_np = _tensor_to_f32_np(objective["score"]["visual_embeddings_float32"], torch)
                waveform_sha = _sha(output_np.tobytes())
                output_peak_abs = _float(v3_result["normalization"]["final_output_peak"])
                peak_ceiling = v3_result["normalization"]["declared_peak_ceiling"]
                peak_ceiling_excess = v3_result["normalization"]["final_output_peak_ceiling_excess"]
                optimizer.step()
                completed_optimizer_steps = step_index
                if not torch.isfinite(head.band_contrasts_db).all().item():
                    raise FloatingPointError(f"non-finite V3 parameter update at step {step_index}")
                theta_after = _make_v3_parameter(head, torch)["values"]
                record = {
                    "step": step_index,
                    "pass_index_zero_based": pass_index,
                    "row_index_zero_based": row_index,
                    "row_id": row_id,
                    "sample_id": row["sample_id"],
                    "speaker_id": row["speaker_id"],
                    "canonical_j": row["canonical_j"],
                    "source_pcm_float32_sha256": row["natural_pcm"]["pcm_sha256"],
                    "optimizer": {"class": "Adam", "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8},
                    "loss": _float(loss),
                    "objective_components": metrics,
                    "gradient_l2": grad_l2,
                    "gradient_max_abs": grad_max,
                    "gradient_nonzero_scalar_count": int(torch.count_nonzero(grad).item()),
                    "theta_before_update": theta_before,
                    "theta_after_update": theta_after,
                    "candidate_waveform_float32_sha256": waveform_sha,
                    "candidate_output_peak_abs": output_peak_abs,
                    "configured_peak_ceiling": peak_ceiling,
                    "peak_limiter_activated": peak_ceiling_excess is not None and _float(peak_ceiling_excess) > 0.0,
                    "curve31_float32": curve_np.tolist(),
                    "elapsed_seconds": time.time() - start_train,
                }
                step_records.append(record)
                train_curves_np.append(curve_np)
                train_visuals_np.append(visual_np)
                train_metrics.append(metrics)
                _append_jsonl(step_jsonl, record)
                _write_status(out, {
                    **status, "status": "RUNNING_FIT120", "stage": "optimizer_step",
                    "completed_baseline_rows": 40, "expected_baseline_rows": 40,
                    "completed_zero_gate_rows": 40, "expected_zero_gate_rows": 40,
                    "completed_optimizer_steps": step_index, "planned_optimizer_steps": OPTIMIZER_STEPS,
                    "last_row_id": row_id, "last_loss": record["loss"], "last_gradient_l2": grad_l2,
                    "gpu_identity": gpu_identity,
                })
        if step_index != OPTIMIZER_STEPS or len(step_records) != OPTIMIZER_STEPS:
            raise FitError("training did not complete exactly the 120 predeclared optimizer steps")

        training_arrays_path = out / "training_score_arrays.npz"
        with training_arrays_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                row_ids=np.asarray([str(record["row_id"]) for record in step_records], dtype="U96"),
                step_indices_int32=np.arange(1, OPTIMIZER_STEPS + 1, dtype="<i4"),
                candidate_curves_float32=np.stack(train_curves_np).astype("<f4", copy=False),
                candidate_visual_embeddings_float32=np.stack(train_visuals_np).astype("<f4", copy=False),
            )
            handle.flush()
            os.fsync(handle.fileno())

        implementation_path = REPO_ROOT / "scripts/audio_enhancement/shared_recipient_transition_head_v3.py"
        implementation_pin = {"path": "scripts/audio_enhancement/shared_recipient_transition_head_v3.py", "bytes": implementation_path.stat().st_size, "sha256": _sha_file(implementation_path)}
        if implementation_pin["sha256"] != EXPECTED_V3_HEAD_SHA256:
            raise FitError("V3 implementation changed before checkpoint export")
        parameter = _make_v3_parameter(head, torch)
        constructor_sha = _canonical_sha(constructor)
        runtime_pins_path = out / "runtime_source_pins.json"
        runtime_payload = {
            "schema": "cross_tfg_fit40_native_transition_runtime_pins_v1",
            "runner": _pin(Path(__file__).resolve()),
            "parent_plan": {"path": str(args.parent_plan.resolve(strict=True)), "bytes": len(plan_raw), "sha256": _sha(plan_raw), "payload_sha256": EXPECTED_PARENT_PAYLOAD_SHA256},
            "fit_source_manifest": {"path": str(args.source_manifest.resolve(strict=True)), "bytes": len(source_raw), "sha256": _sha(source_raw)},
            "native_geometry_manifest": {"path": str(args.geometry_manifest.resolve(strict=True)), "bytes": len(geometry_raw), "sha256": _sha(geometry_raw), "payload_sha256": geometry["payload_sha256"]},
            "head_config": {"path": str(args.head_config.resolve(strict=True)), "bytes": len(head_config_raw), "sha256": _sha(head_config_raw)},
            "training_modules": module_pins,
            "qualified_v2_runtime_source_pins": runtime_pins,
            "head_constructor_sha256": constructor_sha,
            "peak_policy": {
                "configured_peak_ceiling": constructor.get("peak_ceiling"),
                "training_behavior": "unbounded fixed-RMS baseline; no .98 peak limiter; peak values are recorded without clipping",
                "candidate_peak_max_abs_over_120_steps": max(float(record["candidate_output_peak_abs"]) for record in step_records),
            },
            "fit_rows": [
                {"row_id": item["row"]["row_id"], "sample_id": item["row"]["sample_id"], "speaker_id": item["row"]["speaker_id"], "canonical_j": item["row"]["canonical_j"], "natural_pcm": item["row"]["natural_pcm"]}
                for item in prepared
            ],
            "geometry_loader_records": native_geometry_records,
            "audio_cache_clone": cache_clone_record,
            "gpu_identity": gpu_identity,
            "optimizer": {"class": "Adam", "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8},
            "training_order": "exact source-manifest order repeated in three complete passes; no shuffle, early stopping, dev, or confirmation",
        }
        runtime_raw = _atomic_json(runtime_pins_path, runtime_payload)
        checkpoint = {
            "schema": "cross_tfg_v3_parameter_checkpoint_v1",
            "head_schema": "cross_tfg_shared_recipient_transition_head_v3",
            "implementation": implementation_pin,
            "constructor_sha256": constructor_sha,
            "parameter": parameter,
            "training_artifact_bindings": {
                "training_steps_jsonl_sha256": _sha_file(step_jsonl),
                "training_score_arrays_sha256": _sha_file(training_arrays_path),
                "runtime_source_pins_sha256": _sha(runtime_raw),
                "completed_optimizer_steps": OPTIMIZER_STEPS,
                "fit_source_manifest_sha256": _sha(source_raw),
                "native_geometry_manifest_sha256": _sha(geometry_raw),
                "parent_plan_raw_sha256": _sha(plan_raw),
            },
            "selection_status": "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED",
        }
        checkpoint_path = out / "shared_recipient_transition_head_v3_checkpoint.json"
        checkpoint_raw = _atomic_json(checkpoint_path, checkpoint)
        training_record = {
            "schema": "cross_tfg_v3_candidate_training_record_v1",
            "status": "COMPLETE",
            "scope": "fit only: forty canonical historical utterance endpoints across twenty exposed speakers; no development or confirmation data",
            "completed_optimizer_steps": OPTIMIZER_STEPS,
            "planned_optimizer_steps": OPTIMIZER_STEPS,
            "pass_count": 3,
            "rows_per_pass": 40,
            "fit_speaker_count": 20,
            "fit_row_count": 40,
            "fit_seed": FIT_SEED,
            "optimizer": {"class": "Adam", "learning_rate": 0.1, "betas": [0.9, 0.999], "eps": 1e-8},
            "objective": {
                "direction": "maximize native RMS-relative reward; minimize loss=-reward+regularizers",
                "weights": OBJECTIVE_WEIGHTS,
                "terms": ["delta_Cbest", "delta_far_lag_contrast_abs4plus", "D0_guard_violation", "Canchor_guard_violation", "visual_embedding_log_norm_ratio_squared", "waveform_relative_mse_to_rms_reference", "mask_total_variation_ln_gain"],
                "mask_tv_units": "V3 projected log-magnitude mask dB multiplied by ln(10)/20 before frozen V4 natural-log mask-TV reducer",
                "lag_support": "27 original-clock queries and all 31 offsets -15..+15, no padding",
            },
            "zero_head_gate": {
                "status": "PASS",
                "row_count": 40,
                "same_cuda_device_as_rms_baseline": True,
                "waveform_requirement": "bit-identical to exact unbounded fixed-RMS baseline on the same CUDA device",
                "visual_embedding_max_abs_tolerance": ZERO_GATE_MAX_ABS,
                "curve31_max_abs_tolerance": ZERO_GATE_MAX_ABS,
                "peak_policy": "unbounded; no peak limiter is applied to baseline or zero head",
                "passed_rows": [record["row_id"] for record in gate_records],
            },
            "checkpoint_sha256": _sha(checkpoint_raw),
            "head_implementation_sha256": implementation_pin["sha256"],
            "constructor_sha256": constructor_sha,
            "fit_source_manifest_sha256": _sha(source_raw),
            "native_geometry_manifest_sha256": _sha(geometry_raw),
            "parent_plan_raw_sha256": _sha(plan_raw),
            "training_steps_jsonl_sha256": _sha_file(step_jsonl),
            "training_score_arrays_sha256": _sha_file(training_arrays_path),
            "runtime_source_pins_sha256": _sha(runtime_raw),
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "selection_record_created": False,
            "selection_status": "NOT_SELECTED",
        }
        training_record_raw = _atomic_json(out / "training_record.json", training_record)
        completion = {
            "schema": EXPECTED_SCHEMA,
            "status": "COMPLETE_FIT40_V3_120_STEPS",
            "completed_optimizer_steps": OPTIMIZER_STEPS,
            "fit_rows": 40,
            "fit_speakers": 20,
            "zero_gate_rows_passed": 40,
            "checkpoint": _pin(checkpoint_path),
            "training_record": _pin(out / "training_record.json"),
            "runtime_source_pins": _pin(runtime_pins_path),
            "baseline_arrays": _pin(baseline_arrays_path),
            "zero_gate_arrays": _pin(zero_arrays_path),
            "training_score_arrays": _pin(training_arrays_path),
            "training_steps": _pin(step_jsonl),
            "training_record_raw_sha256": _sha(training_record_raw),
            "gpu_identity": gpu_identity,
            "selection_record_created": False,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "elapsed_seconds": time.time() - status["started_unix_seconds"],
        }
        _atomic_json(out / "completion.json", completion)
        _write_status(out, {**completion, "stage": "complete"})
        print(json.dumps(completion, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return completion
    except Exception as exc:
        failure = {
            "schema": EXPECTED_SCHEMA,
            "status": "FAILED_FIT40_V3_TRAINING",
            "stage": active_stage,
            "completed_optimizer_steps": completed_optimizer_steps,
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=25),
            "output_root": str(out),
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "optimizer_updates_are_not_resumable": True,
            "timestamp_unix_seconds": time.time(),
        }
        try:
            if not (out / "failure.json").exists():
                _atomic_json(out / "failure.json", failure)
            _write_status(out, {"schema": EXPECTED_SCHEMA, "status": "FAILED", "stage": active_stage, "completed_optimizer_steps": completed_optimizer_steps, "error": str(exc)})
        finally:
            raise


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        print(json.dumps({"schema": EXPECTED_SCHEMA, "status": "FAILED", "error": str(exc)}, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
