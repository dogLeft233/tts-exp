#!/usr/bin/env python3
"""Audio-only descriptive decomposition of the frozen V4 spectral log mask.

This tool has two modes. ``build-input`` binds the already frozen development
audio stage and historical fit audio manifests into a compact remote input
index. ``run`` replays the frozen transform on CPU, verifies the staged V4
waveforms byte-for-byte at the canonical PCM level, and writes per-row
frequency-mean (mu) and time-varying residual (delta) diagnostics. It does not
render video, score SyncNet, train, select, or read confirmation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

STAGE_LOCAL_RAW = "85be3063e9d324e6d2b519a99ec37eb781f0ba2314db47f0e014a4d7e83b9cf8"
STAGE_PAYLOAD = "f19a5d83ae8ddab682694bdad80c51b842d4b5c0a7fdea38ba7a43eb1eddc560"
FIT_LOCAL_RAW = "279baeca4b05c21fbc57bd9fbc8a625572f4b99799fd9548e904f17cf4416cc0"
FIT_PAYLOAD = "9a9777f1459aa383bba1eeb6adac85e8b6fa4e0ef7d5274f2ac30fe5c1f303c1"
PLAN_LOCAL_RAW = "70701ac4a83e5ea2abd737708aba50b5c5ed2a0d51c18bad38ae84b4480ef28c"
PLAN_PAYLOAD = "a00c3e45373bf182df66a709321b88bf89fd8a4041a0378de3a16e13d5005934"
CHECKPOINT_SHA = "3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4"
METHOD_CONFIG_SHA = "e1fb7fa010dbc6ea22e8362230f144f00dcea09bfd34500e0c2cdeed1fb08cd7"
METHOD_ID = "spectral_head_fit_v4_native_attempt01"
EXPECTED_PROJECT_ROOT = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/stageb_v4_dev_attempt02/code"
EXPECTED_STAGE_MANIFEST = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/stageb_v4_dev_attempt02/audio_only_stage_attempt01/output/audio_only_candidate_manifest.json"
EXPECTED_PLAN_REMOTE = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/stageb_v4_dev_attempt02/plan/plan.json"
EXPECTED_CHECKPOINT_REMOTE = "/dev/shm/cross_tfg_audio_head_20261003/fit_v4_native_fullcontext_attempt01/output/shared_audio_head.pt"
SOURCE_LABEL_TO_RELATIVE = {
    "audio_enhancement_heads.py": "scripts/audio_enhancement/heads.py",
    "audio_enhancement_pipeline.py": "scripts/audio_enhancement/pipeline.py",
    "run_cross_frontend_dev_attempt03.py": "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt03.py",
    "run_cross_frontend_fit_profiles_attempt02.py": "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_fit_profiles_attempt02.py",
    "run_stagea_dev_attempt03.py": "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py",
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _payload_hash(obj: dict[str, Any], hash_key: str) -> str:
    return _sha_bytes(_canonical({k: v for k, v in obj.items() if k != hash_key}))


def _load_pinned_json(path: Path, raw_sha: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or _sha_file(path) != raw_sha:
        raise RuntimeError(f"{label} raw SHA/path mismatch: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return obj


def build_input(args: argparse.Namespace) -> dict[str, Any]:
    stage_path = Path(args.stage_manifest).resolve(strict=True)
    fit_path = Path(args.fit_manifest).resolve(strict=True)
    plan_path = Path(args.plan).resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    stage = _load_pinned_json(stage_path, STAGE_LOCAL_RAW, "V4 audio-stage manifest")
    fit = _load_pinned_json(fit_path, FIT_LOCAL_RAW, "historical fit-audio manifest")
    plan = _load_pinned_json(plan_path, PLAN_LOCAL_RAW, "V4 frozen plan")
    if stage.get("manifest_sha256") != STAGE_PAYLOAD or _payload_hash(stage, "manifest_sha256") != STAGE_PAYLOAD:
        raise RuntimeError("V4 audio-stage payload SHA mismatch")
    if fit.get("manifest_sha256") != FIT_PAYLOAD or _payload_hash(fit, "manifest_sha256") != FIT_PAYLOAD:
        raise RuntimeError("historical fit-audio payload SHA mismatch")
    if plan.get("plan_sha256") != PLAN_PAYLOAD or _payload_hash(plan, "plan_sha256") != PLAN_PAYLOAD:
        raise RuntimeError("V4 plan payload SHA mismatch")
    if _sha_file(checkpoint_path) != CHECKPOINT_SHA:
        raise RuntimeError("frozen V4 checkpoint SHA mismatch")
    if stage.get("status") != "PASS" or stage.get("confirmation_audio_read") is not False:
        raise RuntimeError("stage manifest is not a frozen development-only PASS")
    if stage.get("method_id") != METHOD_ID or stage.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise RuntimeError("stage method/checkpoint binding differs")
    if stage.get("source_plan_raw_sha256") != PLAN_LOCAL_RAW or stage.get("method_config_sha256") != METHOD_CONFIG_SHA:
        raise RuntimeError("stage plan/config pins differ")
    method_list = plan.get("methods")
    methods = [m for m in method_list if isinstance(m, dict) and m.get("method_id") == METHOD_ID] if isinstance(method_list, list) else []
    if len(methods) != 1:
        raise RuntimeError("frozen plan does not contain exactly one V4 method")
    method = methods[0]
    if method.get("config_sha256") != METHOD_CONFIG_SHA or method.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise RuntimeError("frozen method configuration/checkpoint pins differ")
    config = method.get("config", {})
    if config.get("transform_source_sha256") != stage.get("transform_source_sha256"):
        raise RuntimeError("frozen method source map differs from stage source map")
    if _sha_file(checkpoint_path) != config.get("checkpoint_sha256"):
        raise RuntimeError("frozen method checkpoint bytes differ")

    dev_rows = stage.get("rows")
    if not isinstance(dev_rows, list) or len(dev_rows) != 20:
        raise RuntimeError("audio stage must contain exactly 20 dev rows")
    dev_inputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in dev_rows:
        sid = str(row.get("sample_id", ""))
        if sid in seen or row.get("split") != "dev" or row.get("status") != "PASS":
            raise RuntimeError(f"invalid/duplicate development row: {sid}")
        seen.add(sid)
        natural = row.get("natural_pcm", {})
        candidate = row.get("candidate_pcm", {})
        if (natural.get("sample_rate_hz") != 16000 or candidate.get("sample_rate_hz") != 16000
                or natural.get("sample_count") != candidate.get("sample_count")
                or natural.get("pcm_float32_sha256") is None or candidate.get("pcm_float32_sha256") is None):
            raise RuntimeError(f"invalid natural/candidate PCM contract: {sid}")
        dev_inputs.append({
            "sample_id": sid, "speaker_id": str(row["speaker_id"]), "split": "dev",
            "natural": {"path": natural["path"], "file_sha256": natural["file_sha256"],
                        "pcm_float32_sha256": natural["pcm_float32_sha256"], "sample_count": int(natural["sample_count"]),
                        "sample_rate_hz": int(natural["sample_rate_hz"]), "channels": 1},
            "candidate": {"path": candidate["path"], "file_sha256": candidate["file_sha256"],
                          "pcm_float32_sha256": candidate["pcm_float32_sha256"], "sample_count": int(candidate["sample_count"]),
                          "sample_rate_hz": int(candidate["sample_rate_hz"]), "channels": 1},
            "planned_row_ids": list(row["planned_row_ids"]),
        })
    if len({r["speaker_id"] for r in dev_inputs}) != 10:
        raise RuntimeError("development cohort must retain its frozen 10-speaker denominator")

    fit_rows = fit.get("rows")
    if not isinstance(fit_rows, list) or len(fit_rows) != 8:
        raise RuntimeError("historical fit audio manifest must contain eight rows")
    fit_inputs: list[dict[str, Any]] = []
    fit_ids: set[str] = set()
    for row in fit_rows:
        sid = row["row_id"].split("__", 1)[0]
        if sid in fit_ids or not row["row_id"].endswith("__center_aligned_32f"):
            raise RuntimeError(f"invalid/duplicate historical fit input: {row.get('row_id')}")
        fit_ids.add(sid)
        if row.get("sample_rate_hz") != 16000 or row.get("channels") != 1:
            raise RuntimeError(f"historical fit audio rate/channel mismatch: {sid}")
        fit_inputs.append({
            "sample_id": sid, "speaker_id": sid.split("_", 1)[0], "split": "historical_fit",
            "row_id": row["row_id"], "path": row["remote_path"], "file_sha256": row["file_sha256"],
            "pcm_float32_sha256": row["pcm_float32_sha256"], "direct_source_pcm16_sha256": row["direct_source_pcm16_sha256"],
            "sample_count": int(row["sample_count"]), "sample_rate_hz": 16000, "channels": 1,
            "source_archive_sha256": row["source_archive_sha256"], "source_archive_member": row["source_archive_member"],
        })
    if len(fit_ids) != 8:
        raise RuntimeError("historical fit roster changed")

    result: dict[str, Any] = {
        "schema": "cross-tfg-v4-logmask-decomposition-input-v1",
        "status": "FROZEN_INPUTS",
        "scope": {"development_rows": 20, "development_speakers": 10, "historical_fit_rows": 8,
                  "confirmation_audio_read": False, "video_rendered": False, "syncnet_called": False,
                  "asr_called": False, "training_performed": False, "selection_performed": False},
        "pins": {
            "stage_manifest_path": EXPECTED_STAGE_MANIFEST, "stage_manifest_raw_sha256": STAGE_LOCAL_RAW,
            "stage_manifest_payload_sha256": STAGE_PAYLOAD,
            "fit_manifest_local_path": str(fit_path), "fit_manifest_raw_sha256": FIT_LOCAL_RAW,
            "fit_manifest_payload_sha256": FIT_PAYLOAD,
            "plan_path": EXPECTED_PLAN_REMOTE, "plan_raw_sha256": PLAN_LOCAL_RAW, "plan_payload_sha256": PLAN_PAYLOAD,
            "checkpoint_path": EXPECTED_CHECKPOINT_REMOTE, "checkpoint_sha256": CHECKPOINT_SHA,
            "method_id": METHOD_ID, "method_config_sha256": METHOD_CONFIG_SHA,
            "project_root": EXPECTED_PROJECT_ROOT,
            "runner_source_sha256": _sha_file(Path(__file__).resolve()),
            "transform_source_sha256": stage["transform_source_sha256"],
            "sample_rate_hz": 16000, "n_fft": 512, "hop_length": 128,
            "torch_version": "2.5.1+cu124", "numpy_version": "1.26.4", "device": "cpu", "torch_num_threads": 1,
        },
        "method_config": config,
        "development_rows": dev_inputs,
        "historical_fit_rows": fit_inputs,
    }
    result["input_manifest_sha256"] = _payload_hash(result, "input_manifest_sha256")
    out = Path(args.output)
    if out.exists():
        raise RuntimeError(f"input manifest output must be new: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    receipt = {"schema": "cross-tfg-v4-logmask-decomposition-input-build-v1", "status": "PASS",
               "input_manifest_path": str(out.resolve()), "input_manifest_raw_sha256": _sha_file(out),
               "input_manifest_payload_sha256": result["input_manifest_sha256"],
               "dev_rows": len(dev_inputs), "dev_speakers": len({r['speaker_id'] for r in dev_inputs}),
               "historical_fit_rows": len(fit_inputs), "confirmation_audio_read": False}
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return receipt


def _pearson(a: Any, b: Any) -> tuple[float | None, int]:
    import numpy as np
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None, int(x.size)
    value = float(np.corrcoef(x, y)[0, 1])
    return (value if math.isfinite(value) else None), int(x.size)


def _slope_db_octave(power: Any, freqs: Any):
    import numpy as np
    logp = 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), 1e-20))
    freq = np.asarray(freqs, dtype=np.float64)
    out = np.full(logp.shape, np.nan, dtype=np.float32)
    logf = np.log2(np.maximum(freq, 1e-12))
    for k in range(freq.size):
        if freq[k] < 150.0 or freq[k] > 7800.0:
            continue
        lo, hi = max(1, k - 4), min(freq.size, k + 5)
        x = logf[lo:hi]
        xc = x - x.mean()
        den = float(np.dot(xc, xc))
        if den <= 0.0:
            continue
        y = logp[lo:hi, :]
        yc = y - y.mean(axis=0, keepdims=True)
        out[k, :] = ((xc[:, None] * yc).sum(axis=0) / den).astype(np.float32)
    return out


def _activity_mask(wave: Any, frame_count: int, hop: int, sr: int = 16000):
    import numpy as np
    block = int(round(0.040 * sr))
    x = np.asarray(wave, dtype=np.float32)
    n_full = x.size // block
    if n_full == 0:
        return np.full(frame_count, -1, dtype=np.int8), {"full_blocks": 0, "threshold": None, "active_blocks": 0}
    chunks = x[:n_full * block].reshape(n_full, block).astype(np.float64)
    rms = np.sqrt(np.mean(chunks * chunks, axis=1))
    p95 = float(np.percentile(rms, 95))
    threshold = max(1e-4, 0.01 * p95)
    active_blocks = rms >= threshold
    centers = np.arange(frame_count, dtype=np.int64) * int(hop)
    idx = centers // block
    out = np.full(frame_count, -1, dtype=np.int8)
    valid = idx < n_full
    out[valid] = active_blocks[idx[valid]].astype(np.int8)
    return out, {"block_ms": 40, "full_blocks": int(n_full), "threshold_rms": threshold,
                 "threshold_rule": "max(1e-4, 0.01 * p95(full_40ms_RMS))",
                 "active_blocks": int(active_blocks.sum()), "classified_frames": int(valid.sum()),
                 "interpretation": "energy-active/quiet proxy, not transcript or phoneme VAD"}


def _mask_stats(mask_ln: Any, slope: Any, activity: Any, freqs: Any) -> dict[str, Any]:
    import numpy as np
    ln_to_db = 20.0 / math.log(10.0)
    mdb = np.asarray(mask_ln, dtype=np.float64) * ln_to_db
    mu = mdb.mean(axis=1)
    delta = mdb - mu[:, None]
    global_db = float(mu.mean())
    centered_mu = mu - global_db
    static_dc_energy = global_db * global_db
    static_shape_energy = float(np.mean(centered_mu * centered_mu))
    dynamic_energy = float(np.mean(delta * delta))
    total_energy = static_dc_energy + static_shape_energy + dynamic_energy
    valid_t = activity >= 0
    active_t = activity == 1
    quiet_t = activity == 0
    freq = np.asarray(freqs, dtype=np.float64)
    slope_arr = np.asarray(slope, dtype=np.float64)
    freq_valid = (freq >= 150.0) & (freq <= 7800.0)
    result: dict[str, Any] = {
        "status": "PASS", "mask_shape_frequency_by_frame": list(mask_ln.shape),
        "frequency_bins": int(mask_ln.shape[0]), "mask_frames": int(mask_ln.shape[1]),
        "mean_mask_db": float(mdb.mean()), "mask_db_min": float(mdb.min()), "mask_db_max": float(mdb.max()),
        "mask_db_std": float(mdb.std()), "percent_tf_abs_over_1db": float(np.mean(np.abs(mdb) > 1.0) * 100.0),
        "percent_tf_positive": float(np.mean(mdb > 0.0) * 100.0), "percent_tf_negative": float(np.mean(mdb < 0.0) * 100.0),
        "mu_time_mean_frequency_profile_db": mu.tolist(),
        "max_abs_delta_time_mean_db": float(np.max(np.abs(delta.mean(axis=1)))),
        "max_abs_mu_plus_delta_reconstruction_error_db": float(np.max(np.abs((mu[:, None] + delta) - mdb))),
        "decomposition": {
            "global_offset_db": global_db,
            "centered_static_profile_rms_db": float(np.sqrt(static_shape_energy)),
            "dynamic_delta_rms_db": float(np.sqrt(dynamic_energy)),
            "static_dc_energy_share_percent": float(100.0 * static_dc_energy / total_energy) if total_energy else 0.0,
            "static_shape_energy_share_percent": float(100.0 * static_shape_energy / total_energy) if total_energy else 0.0,
            "dynamic_energy_share_percent": float(100.0 * dynamic_energy / total_energy) if total_energy else 0.0,
            "identity": "M(f,t)=global_offset + centered_mu(f) + delta(f,t); delta has zero time mean per frequency by construction",
        },
        "activity_proxy": {"classified_frames": int(valid_t.sum()), "active_frames": int(active_t.sum()),
                           "quiet_frames": int(quiet_t.sum()), "unclassified_frames": int((~valid_t).sum())},
        "band_stats": {},
    }
    for name, lo, hi in (("0_300", 0, 300), ("300_1000", 300, 1000), ("1000_3000", 1000, 3000),
                         ("3000_6000", 3000, 6000), ("6000_8000", 6000, 8000.0001)):
        fmask = (freq >= lo) & (freq < hi)
        if not fmask.any():
            result["band_stats"][name] = {"n_bins": 0, "estimable": False}
            continue
        band = mdb[fmask, :]
        result["band_stats"][name] = {
            "n_bins": int(fmask.sum()), "mu_mean_db": float(mu[fmask].mean()),
            "mu_std_db": float(mu[fmask].std()), "delta_rms_db": float(np.sqrt(np.mean(delta[fmask, :] ** 2))),
            "active_mean_mask_db": float(band[:, active_t].mean()) if active_t.any() else None,
            "quiet_mean_mask_db": float(band[:, quiet_t].mean()) if quiet_t.any() else None,
            "active_abs_over_1db_percent": float(np.mean(np.abs(band[:, active_t]) > 1.0) * 100.0) if active_t.any() else None,
        }
    active_slope = slope_arr[:, active_t & np.isfinite(slope_arr).any(axis=0)].astype(np.float64)
    quiet_slope = slope_arr[:, quiet_t & np.isfinite(slope_arr).any(axis=0)].astype(np.float64)
    if active_slope.shape[1]:
        # Some frequency rows can remain entirely non-finite (for example, if all
        # local power values are at the numerical floor). Avoid nanmean's warning
        # and retain the valid-bin denominator explicitly for the descriptive output.
        active_valid = np.isfinite(active_slope)
        active_valid_count = active_valid.sum(axis=1)
        active_sum = np.where(active_valid, active_slope, 0.0).sum(axis=1, dtype=np.float64)
        mean_active_slope = np.full(active_slope.shape[0], np.nan, dtype=np.float64)
        np.divide(active_sum, active_valid_count, out=mean_active_slope, where=active_valid_count > 0)
        corr, n = _pearson(mu[freq_valid], mean_active_slope[freq_valid])
        result["slope_association"] = {"mu_vs_mean_active_natural_slope_db_per_octave": corr,
                                       "mu_vs_mean_active_slope_frequency_bins": n,
                                       "active_mean_slope_valid_frequency_bins": int(np.sum(freq_valid & (active_valid_count > 0)))}
        cell_active = active_t[None, :] & np.isfinite(slope_arr) & freq_valid[:, None]
        corr, n = _pearson(delta[cell_active], slope_arr[cell_active])
        result["slope_association"]["delta_vs_active_natural_slope_db_per_octave"] = corr
        result["slope_association"]["delta_vs_active_tf_cells"] = n
    else:
        result["slope_association"] = {"mu_vs_mean_active_natural_slope_db_per_octave": None,
                                        "mu_vs_mean_active_slope_frequency_bins": 0,
                                        "active_mean_slope_valid_frequency_bins": 0,
                                        "delta_vs_active_natural_slope_db_per_octave": None,
                                        "delta_vs_active_tf_cells": 0}
    cell_quiet = quiet_t[None, :] & np.isfinite(slope_arr) & freq_valid[:, None]
    corr, n = _pearson(delta[cell_quiet], slope_arr[cell_quiet])
    result["slope_association"]["delta_vs_quiet_natural_slope_db_per_octave"] = corr
    result["slope_association"]["delta_vs_quiet_tf_cells"] = n
    return result


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    manifest_path = Path(args.input_manifest).resolve(strict=True)
    manifest_raw = _sha_file(manifest_path)
    if manifest_raw != args.expected_input_manifest_raw_sha256:
        raise RuntimeError("input manifest raw SHA mismatch")
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("schema") != "cross-tfg-v4-logmask-decomposition-input-v1" or input_manifest.get("input_manifest_sha256") != _payload_hash(input_manifest, "input_manifest_sha256"):
        raise RuntimeError("input manifest schema/payload hash mismatch")
    pins = input_manifest["pins"]
    runner_path = Path(__file__).resolve(strict=True)
    runner_sha = _sha_file(runner_path)
    if runner_sha != pins.get("runner_source_sha256"):
        raise RuntimeError("diagnostic runner source SHA mismatch")
    project_root = Path(pins["project_root"]).resolve(strict=True)
    source_pins = pins.get("transform_source_sha256", {})
    for label, relative in SOURCE_LABEL_TO_RELATIVE.items():
        source_path = project_root / relative
        expected_source_sha = source_pins.get(label)
        if source_path.is_symlink() or not source_path.is_file() or _sha_file(source_path) != expected_source_sha:
            raise RuntimeError(f"frozen transform source mismatch: {label}")
    if _sha_file(Path(pins["stage_manifest_path"])) != pins["stage_manifest_raw_sha256"]:
        raise RuntimeError("remote stage manifest raw pin mismatch")
    stage = json.loads(Path(pins["stage_manifest_path"]).read_text(encoding="utf-8"))
    if stage.get("manifest_sha256") != pins["stage_manifest_payload_sha256"] or _payload_hash(stage, "manifest_sha256") != pins["stage_manifest_payload_sha256"]:
        raise RuntimeError("remote stage manifest payload pin mismatch")
    plan_path = Path(pins["plan_path"]).resolve(strict=True)
    if _sha_file(plan_path) != pins["plan_raw_sha256"]:
        raise RuntimeError("remote V4 plan raw pin mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != pins["plan_payload_sha256"] or _payload_hash(plan, "plan_sha256") != pins["plan_payload_sha256"]:
        raise RuntimeError("remote V4 plan payload pin mismatch")
    checkpoint = Path(pins["checkpoint_path"]).resolve(strict=True)
    if _sha_file(checkpoint) != pins["checkpoint_sha256"]:
        raise RuntimeError("remote checkpoint raw pin mismatch")
    torch.set_num_threads(1)
    if torch.__version__ != pins["torch_version"] or np.__version__ != pins["numpy_version"] or torch.cuda.is_initialized():
        raise RuntimeError("runtime mismatch or CUDA initialized; diagnostic must be CPU-only")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.audio_enhancement.audio_io import pcm_float32_sha256, read_canonical_wav
    from scripts.audio_enhancement.heads import EnhancementStack
    from scripts.experiments.cross_tfg_audio_head_20261003 import run_stagea_dev_attempt03 as stagea

    method = next(m for m in plan["methods"] if m.get("method_id") == pins["method_id"])
    if method.get("config_sha256") != pins["method_config_sha256"] or method.get("config") != input_manifest.get("method_config"):
        raise RuntimeError("method config differs from the locally frozen input")
    if method.get("config", {}).get("use_local_envelope") is not False:
        raise RuntimeError("unexpected local envelope configuration")
    head, checkpoint_sha = stagea._load_spectral_head(method)
    if checkpoint_sha != pins["checkpoint_sha256"]:
        raise RuntimeError("Stage-A head loader returned an unexpected checkpoint")
    stack = EnhancementStack(head).cpu().eval()
    config = method["config"]
    stack_kwargs = {
        "use_global_rms": bool(config.get("use_global_rms", False)),
        "use_local_envelope": bool(config.get("use_local_envelope", False)),
        "use_spectral_head": bool(config.get("use_spectral_head", True)),
        "local_window_ms": float(config.get("local_window_ms", 32.0)),
        "local_ratio": float(config.get("local_ratio", 0.8)),
        "local_min_gain_db": float(config.get("local_min_gain_db", -6.0)),
        "local_max_gain_db": float(config.get("local_max_gain_db", 6.0)),
        "post_rms_match": str(config.get("post_rms_match", "none")),
        "target_rms": float(config.get("target_rms", 0.0376838172675746)),
        "rms_stage_peak_ceiling": config.get("rms_stage_peak_ceiling"),
        "peak_ceiling": config.get("peak_ceiling"),
        "return_diagnostics": True,
    }
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise RuntimeError(f"output path must be new: {out}")
    out.mkdir(parents=True)
    array_values: dict[str, Any] = {}
    row_records: list[dict[str, Any]] = []
    all_rows = [("dev", r) for r in input_manifest["development_rows"]] + [("historical_fit", r) for r in input_manifest["historical_fit_rows"]]
    sample_rate, n_fft, hop = 16000, 512, 128
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate).astype(np.float32)
    for split, row in all_rows:
        sid = row["sample_id"]
        prefix = "dev" if split == "dev" else "fit"
        natural_ref = row["natural"] if split == "dev" else row
        natural_path = Path(natural_ref["path"]).resolve(strict=True)
        if natural_path.is_symlink() or not natural_path.is_file() or _sha_file(natural_path) != natural_ref["file_sha256"]:
            raise RuntimeError(f"{split} natural source file SHA mismatch: {sid}")
        natural, info = read_canonical_wav(natural_path)
        if (info.pcm_float32_sha256 != natural_ref["pcm_float32_sha256"] or info.frames != natural_ref["sample_count"]
                or info.sample_rate != 16000 or info.source_channels != 1):
            raise RuntimeError(f"{split} canonical source PCM metadata mismatch: {sid}")
        x = torch.from_numpy(np.ascontiguousarray(natural, dtype=np.float32))
        with torch.inference_mode():
            y, mask_t, torch_diag = stack.forward_with_mask(x, **stack_kwargs)
        y_np = np.ascontiguousarray(y.detach().cpu().numpy(), dtype=np.float32)
        if y_np.shape != natural.shape or not np.isfinite(y_np).all():
            raise RuntimeError(f"transform replay contract failed: {split}/{sid}")
        replay_info = {"exact_stage_candidate_pcm": None, "candidate_pcm_sha256": None, "candidate_file_sha256": None}
        if split == "dev":
            candidate_ref = row["candidate"]
            candidate_path = Path(candidate_ref["path"]).resolve(strict=True)
            if candidate_path.is_symlink() or not candidate_path.is_file() or _sha_file(candidate_path) != candidate_ref["file_sha256"]:
                raise RuntimeError(f"dev candidate file SHA mismatch: {sid}")
            candidate, candidate_info = read_canonical_wav(candidate_path)
            if (candidate_info.pcm_float32_sha256 != candidate_ref["pcm_float32_sha256"]
                    or candidate_info.frames != candidate_ref["sample_count"] or candidate_info.sample_rate != 16000
                    or candidate_info.source_channels != 1):
                raise RuntimeError(f"dev candidate PCM metadata mismatch: {sid}")
            if not np.array_equal(y_np, candidate):
                raise RuntimeError(f"frozen V4 transform replay differs from staged candidate samples: {sid}")
            replay_info = {"exact_stage_candidate_pcm": True, "candidate_pcm_sha256": candidate_info.pcm_float32_sha256,
                           "candidate_file_sha256": candidate_info.file_sha256}
        mask = np.ascontiguousarray(mask_t.detach().cpu().numpy()[0], dtype=np.float32)
        if mask.shape[0] != n_fft // 2 + 1 or not np.isfinite(mask).all():
            raise RuntimeError(f"invalid V4 log mask: {split}/{sid}")
        # Natural-source spectrum slope is a descriptive covariate; scaling by global RMS
        # leaves the dB/octave slope unchanged. It is not a fitted/selected feature.
        stft = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=torch.hann_window(n_fft, periodic=True, dtype=x.dtype),
                          center=True, pad_mode="constant", normalized=False,
                          onesided=True, return_complex=True)
        power = stft.abs().square().detach().cpu().numpy().astype(np.float32)
        slope = _slope_db_octave(power, freqs)
        activity, activity_info = _activity_mask(natural, mask.shape[1], hop, sample_rate)
        stats = _mask_stats(mask, slope, activity, freqs)
        centers = np.arange(mask.shape[1], dtype=np.int64) * hop
        safe_id = sid.replace("/", "_").replace("\\", "_")
        key = f"{prefix}__{safe_id}__"
        mask64 = mask.astype(np.float64)
        mu_ln = mask64.mean(axis=1, dtype=np.float64)
        delta_ln = np.ascontiguousarray(mask64 - mu_ln[:, None], dtype=np.float64)
        array_values[key + "mask_ln"] = mask
        array_values[key + "mu_ln"] = mu_ln
        array_values[key + "delta_ln"] = delta_ln
        array_values[key + "natural_slope_db_octave"] = slope
        array_values[key + "energy_activity"] = activity
        array_values[key + "frame_center_sample"] = centers
        array_values[key + "frequencies_hz"] = freqs
        row_records.append({
            "sample_id": sid, "speaker_id": row["speaker_id"], "split": split, "status": "PASS",
            "source_audio": {"path": str(natural_path), "file_sha256": info.file_sha256,
                             "pcm_float32_sha256": info.pcm_float32_sha256, "source_encoding": info.source_encoding,
                             "source_sample_rate_hz": info.source_sample_rate, "sample_count": info.frames},
            "candidate_replay": replay_info,
            "mask_arrays_prefix": key,
            "activity_rule": activity_info,
            "mask_summary": stats,
        })

    npz_path = out / "v4_logmask_arrays.npz"
    np.savez_compressed(npz_path, **array_values)
    cohort_summary: dict[str, Any] = {}
    for split in ("dev", "historical_fit"):
        subset = [r for r in row_records if r["split"] == split and r["status"] == "PASS"]
        scalar_fields = (
            ("global_offset_db", lambda r: r["mask_summary"]["decomposition"]["global_offset_db"]),
            ("centered_static_profile_rms_db", lambda r: r["mask_summary"]["decomposition"]["centered_static_profile_rms_db"]),
            ("dynamic_delta_rms_db", lambda r: r["mask_summary"]["decomposition"]["dynamic_delta_rms_db"]),
            ("static_dc_energy_share_percent", lambda r: r["mask_summary"]["decomposition"]["static_dc_energy_share_percent"]),
            ("static_shape_energy_share_percent", lambda r: r["mask_summary"]["decomposition"]["static_shape_energy_share_percent"]),
            ("dynamic_energy_share_percent", lambda r: r["mask_summary"]["decomposition"]["dynamic_energy_share_percent"]),
            ("mu_vs_active_slope_correlation", lambda r: r["mask_summary"]["slope_association"]["mu_vs_mean_active_natural_slope_db_per_octave"]),
            ("delta_vs_active_slope_correlation", lambda r: r["mask_summary"]["slope_association"]["delta_vs_active_natural_slope_db_per_octave"]),
        )
        cohort_summary[split] = {"planned_rows": 20 if split == "dev" else 8, "completed_rows": len(subset),
                                 "speakers": len({r["speaker_id"] for r in subset}), "descriptive_only": True,
                                 "means": {name: (float(np.mean(vals)) if vals else None) for name, fn in scalar_fields
                                           if (vals := [fn(r) for r in subset if fn(r) is not None]) is not None}}
    summary: dict[str, Any] = {
        "schema": "cross-tfg-v4-logmask-decomposition-result-v1",
        "status": "PASS" if len(row_records) == 28 else "INCOMPLETE",
        "scope": {"planned_development_rows": 20, "completed_development_rows": sum(r["split"] == "dev" for r in row_records),
                  "planned_historical_fit_rows": 8, "completed_historical_fit_rows": sum(r["split"] == "historical_fit" for r in row_records),
                  "confirmation_audio_read": False, "video_rendered": False, "syncnet_called": False,
                  "asr_called": False, "training_performed": False, "selection_performed": False,
                  "interpretation": "descriptive only; historical fit and previously exposed development rows are separate cohorts"},
        "pins": {**pins, "input_manifest_raw_sha256": manifest_raw,
                 "input_manifest_payload_sha256": input_manifest["input_manifest_sha256"],
                 "runner_source_path": str(runner_path), "runner_source_sha256": runner_sha,
                 "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                 "torch_threads": torch.get_num_threads(), "torch_device": "cpu"},
        "methodology": {
            "mask_definition": "M(f,t) is the frozen SpectralResidualHead.forward_with_mask log mask in natural-log amplitude units; report dB=20/ln(10)*M.",
            "decomposition": "mu(f)=mean_t M(f,t); delta(f,t)=M(f,t)-mu(f). mu includes the global mask offset; static shape is mu minus its frequency mean. No transform or coefficients are changed.",
            "decomposition_reduction": "Frozen log mask M is preserved as float32; mu and delta are formed in float64 for descriptive reduction only.",
            "audio_input": "the exact original natural PCM at 16 kHz mono; V4 method configuration replays its global RMS, mask, post-RMS match, and caps as frozen.",
            "slope": "natural-source 512-point Hann centered STFT; local least-squares dB power slope over +/-4 frequency bins, reported in dB/octave for 150-7800 Hz. Descriptive association only.",
            "speech_activity": "40 ms full-block natural RMS proxy at the original sample clock; threshold max(1e-4, 0.01*p95); not transcript/phoneme VAD.",
            "limitations": "one frozen head and one V4 transform; correlations are descriptive and not causal; no hypothesis tests or confidence intervals."},
        "cohort_summary": cohort_summary,
        "rows": row_records,
        "arrays": {"path": str(npz_path.resolve()), "bytes": npz_path.stat().st_size, "raw_sha256": _sha_file(npz_path),
                   "keys": sorted(array_values), "frequency_axis_hz_sha256": _sha_bytes(freqs.tobytes())},
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "summary_path": str(summary_path),
                      "summary_raw_sha256": _sha_file(summary_path), "array_path": str(npz_path),
                      "array_raw_sha256": _sha_file(npz_path), "dev_rows": cohort_summary["dev"]["completed_rows"],
                      "fit_rows": cohort_summary["historical_fit"]["completed_rows"],
                      "confirmation_audio_read": False, "video_rendered": False, "syncnet_called": False},
                     sort_keys=True, separators=(",", ":")))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    b = subs.add_parser("build-input")
    b.add_argument("--stage-manifest", required=True)
    b.add_argument("--fit-manifest", required=True)
    b.add_argument("--plan", required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--output", required=True)
    r = subs.add_parser("run")
    r.add_argument("--input-manifest", required=True)
    r.add_argument("--expected-input-manifest-raw-sha256", required=True)
    r.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "build-input":
        build_input(args)
    else:
        run_diagnostic(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
