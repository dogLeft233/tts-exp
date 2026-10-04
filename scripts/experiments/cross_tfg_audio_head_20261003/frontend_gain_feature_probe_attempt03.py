#!/usr/bin/env python3
"""Fit-only, CPU frontend-input gain probe for the frozen W2L/Muse/Ditto plan.

This reads the eight frozen historical source WAVs, applies four fixed gain
conditions, and records frontend input features/statistics. It does not load a
generator, audio encoder, ONNX Runtime, SyncNet, video, or development/confirm
audio. Ditto is limited to the two independently read ONNX input affine
constants and the transformed raw waveform; it makes no claim about encoder
gain invariance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
TRANSFER_MANIFEST_SHA256 = "be772cbfa4b394991d889f8d5da2cbac3f07d5f8a66979d7278486bc068c8528"
W2L_HELPER_SHA256 = "3480087daecf8dc295f4a3b20703f393bada2690e1f5ce7839ba1dc4fbdec8ef"
W2L_AUDIO_SHA256 = "71d5e927cac3d0b7103e7671a43b82b41b6a56398a092bbfa2e96d3041249c4c"
W2L_HPARAMS_SHA256 = "4a4c31dc51ebd8140da18ac24841e582b2472389f7c65e7f0a6b4447fc0c71cf"
WHISPER_PREPROCESSOR_SHA256 = "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"
MUSE_AUDIO_PROCESSOR_SHA256 = "b7b62f3daa99678576138bf581a6394a10d346571a2530a760d1299ffd622b81"
DITTO_CONSTANTS_SHA256 = "873d525f1b1b4669594b9fbfb50fa3421f5c537da70cf967e5b3ee88ac74d449"
DITTO_ROOT_QC_SHA256 = "e8ad71c17000254427205317c5e026a082b62327f182077a3289070b00a97edf"
DITTO_ROOT_RECEIPT_SHA256 = "21555b138c29567dfaca357b361ebdb0fa4a75978edca6c84f8efaf13b4ef781"
DITTO_ONNX_SHA256 = "04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697"
DITTO_SUBTRACT = 2.586069967946969e-05
DITTO_DIVIDE = 0.20523720979690552
TARGET_RMS = 0.0376838172675746
SAMPLE_RATE = 16_000
CONDITIONS = ("natural", "global_minus_6db", "global_plus_6db", "fixed_rms_0p0376838173")
RUNTIME_HELPER_SHA256 = "9e0bf54fe830e07fdeac7f63871b3e589f9b34eeebd411dbc72852476649920a"
RUNTIME_DISCOVERY_SHA256 = "f6122c492b2a0dcc210eb0b804fa0d42b4dd9a780e47e7e52814fc7f60ececff"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_runtime_contract_helper(path: Path):
    if path.is_symlink() or not path.is_file() or sha256_file(path) != RUNTIME_HELPER_SHA256:
        raise ValueError("measured Transformers runtime contract helper SHA mismatch")
    spec = importlib.util.spec_from_file_location("frontend_gain_transformers_runtime_contract_attempt02", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import measured Transformers runtime contract helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_f32(values: np.ndarray) -> str:
    raw = np.ascontiguousarray(np.asarray(values, dtype="<f4")).tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def canonical_json_sha(obj: dict[str, Any], field: str) -> str:
    value = dict(obj)
    value.pop(field, None)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("feature tensor is empty or non-finite")
    return {
        "count": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "rms": float(np.sqrt(np.mean(x * x))),
    }


def corr(x: np.ndarray, y: np.ndarray) -> float | None:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return None if den <= 0 else float(np.dot(a, b) / den)


def load_frontend_helper(path: Path, expected_sha: str, wav2lip_root: Path):
    if sha256_file(path) != expected_sha:
        raise ValueError("Wav2Lip frontend helper SHA mismatch")
    spec = importlib.util.spec_from_file_location("cross_tfg_wav2lip_frontend_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen Wav2Lip frontend helper")
    module = importlib.util.module_from_spec(spec)
    # The helper defines dataclasses at module scope. Register it before
    # execution so dataclasses can resolve the module namespace reliably.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audio_module, mel_spec, meta = module.import_official_wav2lip_audio(wav2lip_root)
    if meta["audio_sha256"] != W2L_AUDIO_SHA256 or meta["hparams_sha256"] != W2L_HPARAMS_SHA256:
        raise ValueError("official Wav2Lip source pins differ from the frozen fit plan")
    return audio_module, mel_spec, meta


def normalize_feature_shape(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    x = np.asarray(value, dtype=np.float32)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError(f"expected a 2D [feature,time] tensor, got {x.shape}")
    # Transformers returns [feature,time]; fail closed instead of silently
    # transposing an unexpected implementation.
    return np.ascontiguousarray(x)


def conditions(x: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("source RMS must be positive and finite")
    gains = {
        "natural": 1.0,
        "global_minus_6db": 10.0 ** (-6.0 / 20.0),
        "global_plus_6db": 10.0 ** (6.0 / 20.0),
        "fixed_rms_0p0376838173": TARGET_RMS / rms,
    }
    arrays = {
        name: np.ascontiguousarray(x * np.float32(gain), dtype=np.float32)
        for name, gain in gains.items()
    }
    meta = {
        name: {
            "gain_linear": float(gains[name]),
            "gain_db": float(20.0 * math.log10(gains[name])),
            "rms": float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))),
            "peak_abs": float(np.max(np.abs(arr))),
            "fraction_abs_gt_1": float(np.mean(np.abs(arr) > 1.0)),
            "hard_clipping_applied": False,
            "sample_count": int(arr.size),
        }
        for name, arr in arrays.items()
    }
    meta["fixed_rms_0p0376838173"]["target_rms"] = TARGET_RMS
    return arrays, meta


def feature_curve_metrics(reference: np.ndarray, current: np.ndarray, valid_frames: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)
    if ref.shape != cur.shape:
        raise ValueError(f"gain condition changed feature geometry {ref.shape} != {cur.shape}")
    n = min(ref.shape[1], cur.shape[1], max(1, int(valid_frames)))
    ref = ref[:, :n].astype(np.float64)
    cur = cur[:, :n].astype(np.float64)
    ref_common = ref.mean(axis=0)
    cur_common = cur.mean(axis=0)
    ref_centered = ref - ref_common[None, :]
    cur_centered = cur - cur_common[None, :]
    dot = np.sum(ref_centered * cur_centered, axis=0)
    den = np.sqrt(np.sum(ref_centered * ref_centered, axis=0) * np.sum(cur_centered * cur_centered, axis=0))
    frame_cos = np.divide(dot, den, out=np.full_like(dot, np.nan), where=den > 0)
    centered_diff = cur_centered - ref_centered
    common_delta = cur_common - ref_common
    result = {
        "valid_feature_frames": n,
        "all_tensor_frames": int(current.shape[1]),
        "common_mode_delta_mean": float(common_delta.mean()),
        "common_mode_delta_std": float(common_delta.std()),
        "common_mode_delta_min": float(common_delta.min()),
        "common_mode_delta_max": float(common_delta.max()),
        "common_mode_correlation_with_natural": corr(ref_common, cur_common),
        "centered_relative_rms_change": float(np.sqrt(np.mean(centered_diff ** 2))),
        "centered_frame_cosine_mean": float(np.nanmean(frame_cos)) if np.isfinite(frame_cos).any() else None,
        "centered_frame_cosine_min": float(np.nanmin(frame_cos)) if np.isfinite(frame_cos).any() else None,
    }
    curves = {
        "natural_common_mode": ref_common.astype(np.float32),
        "condition_common_mode": cur_common.astype(np.float32),
        "common_mode_delta": common_delta.astype(np.float32),
        "centered_frame_cosine": frame_cos.astype(np.float32),
        "centered_relative_rms_by_frame": np.sqrt(np.mean(centered_diff ** 2, axis=0)).astype(np.float32),
    }
    return result, curves


def tensor_summary(frontend: str, feature: np.ndarray, valid_frames: int) -> dict[str, Any]:
    x = np.asarray(feature, dtype=np.float32)
    n = min(x.shape[1], max(1, int(valid_frames)))
    valid, full = x[:, :n], x
    out: dict[str, Any] = {
        "shape_feature_time": [int(v) for v in x.shape],
        "valid_waveform_supported_frames": int(n),
        "valid_feature_stats": finite_stats(valid),
        "full_tensor_stats": finite_stats(full),
    }
    if frontend == "wav2lip_mel":
        out["floor_or_clip"] = {
            "configured_clip_min": -4.0,
            "configured_clip_max": 4.0,
            "fraction_exact_min_waveform_supported": float(np.mean(valid <= -4.0)),
            "fraction_exact_min_full_tensor": float(np.mean(full <= -4.0)),
            "fraction_exact_max_waveform_supported": float(np.mean(valid >= 4.0)),
            "fraction_exact_max_full_tensor": float(np.mean(full >= 4.0)),
            "fraction_within_0p01_of_min_waveform_supported": float(np.mean(valid <= -3.99)),
            "fraction_within_0p01_of_max_waveform_supported": float(np.mean(valid >= 3.99)),
        }
    elif frontend == "muse_whisper_logmel":
        # Pinned Transformers maps clamp(log-mel, max-8) to (x+4)/4,
        # leaving a per-utterance feature floor exactly 2 units below max.
        floor_level = float(np.max(valid) - 2.0)
        out["floor_or_clip"] = {
            "relative_logmel_floor_feature_value": floor_level,
            "fraction_at_relative_logmel_floor_waveform_supported": float(np.mean(np.isclose(valid, floor_level, rtol=0.0, atol=2e-6))),
            "fraction_at_relative_logmel_floor_full_padded_tensor": float(np.mean(np.isclose(full, floor_level, rtol=0.0, atol=2e-6))),
            "full_tensor_includes_30s_whisper_padding": bool(full.shape[1] > valid.shape[1]),
            "relative_floor_definition": "(max(log_power)-8 + 4)/4 = max(feature)-2; per input segment",
            "do_normalize_explicitly_passed": False,
        }
    return out


def load_ditto_constants(path: Path, root_qc_path: Path) -> dict[str, float]:
    if sha256_file(path) != DITTO_CONSTANTS_SHA256:
        raise ValueError("Ditto ONNX constants audit raw SHA mismatch")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_METADATA_ONLY_CONSTANTS_READ":
        raise ValueError("Ditto ONNX constants audit did not pass")
    if audit.get("model", {}).get("sha256_bound_from_qualified_report") != DITTO_ONNX_SHA256:
        raise ValueError("Ditto constants audit ONNX binding mismatch")
    if audit.get("input_path_nodes", {}).get("subtract", {}).get("index") != 411 or audit.get("input_path_nodes", {}).get("divide", {}).get("index") != 413:
        raise ValueError("Ditto raw input affine node path differs from qualified audit")
    if not root_qc_path.is_file() or root_qc_path.is_symlink() or sha256_file(root_qc_path) != DITTO_ROOT_QC_SHA256:
        raise ValueError("independent Ditto constant numeric-QC receipt raw SHA mismatch")
    root_qc = json.loads(root_qc_path.read_text(encoding="utf-8"))
    if (
        root_qc.get("status") != "PASS_ROOT_DOWNLOADED_CONSTANTS_NUMERIC_AND_BINDING_QC"
        or root_qc.get("receipt_sha256") != DITTO_ROOT_RECEIPT_SHA256
        or root_qc.get("audit_sha256") != DITTO_CONSTANTS_SHA256
        or root_qc.get("full_model_sha256") != DITTO_ONNX_SHA256
        or root_qc.get("subtract") != DITTO_SUBTRACT
        or root_qc.get("divide") != DITTO_DIVIDE
        or root_qc.get("inference_performed") is not False
        or root_qc.get("audio_read") is not False
    ):
        raise ValueError("independent Ditto constants numeric-QC receipt mismatch")
    values = audit.get("input_constants", {})
    parsed: dict[str, float] = {}
    for role, expected in (("subtract", DITTO_SUBTRACT), ("divide", DITTO_DIVIDE)):
        item = values.get(role, {})
        decoded = item.get("decoded_values", [])
        if not item.get("constant_exact") or item.get("decoded_value_count") != 1 or len(decoded) != 1:
            raise ValueError(f"Ditto {role} constant is not one exact scalar")
        actual = float(decoded[0])
        if not math.isfinite(actual) or actual != expected:
            raise ValueError(f"Ditto {role} scalar differs from independent root numeric readback")
        parsed[role] = actual
    return parsed


def ditto_frame_rms(values: np.ndarray) -> np.ndarray:
    n = int(values.size)
    frame_count = (n + 639) // 640
    padded = np.zeros(frame_count * 640, dtype=np.float32)
    padded[:n] = values
    frames = padded.reshape(frame_count, 640).astype(np.float64)
    return np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--transfer-manifest", type=Path, required=True)
    ap.add_argument("--wav2lip-root", type=Path, required=True)
    ap.add_argument("--wav2lip-helper", type=Path, required=True)
    ap.add_argument("--muse-model-workspace", type=Path, required=True)
    ap.add_argument("--muse-audio-processor", type=Path, required=True)
    ap.add_argument("--ditto-constants", type=Path, required=True)
    ap.add_argument("--ditto-root-qc", type=Path, required=True)
    ap.add_argument("--runtime-helper", type=Path, required=True)
    ap.add_argument("--runtime-discovery", type=Path, required=True)
    ap.add_argument("--preflight-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if sha256_file(args.plan) != PLAN_SHA256 or sha256_file(args.transfer_manifest) != TRANSFER_MANIFEST_SHA256:
        raise ValueError("frozen plan or original fit-audio transfer manifest raw SHA mismatch")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.transfer_manifest.read_text(encoding="utf-8"))
    if canonical_json_sha(manifest, "manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("transfer manifest canonical payload checksum mismatch")
    if any(manifest.get(k) is not False for k in ("development_audio_opened", "confirmation_audio_opened", "effects_or_scores_read")):
        raise ValueError("fit-audio transfer manifest scope flags are not closed")
    fit_rows = [r for r in plan.get("rows", []) if r.get("split") == "fit" and r.get("model_id") == "wav2lip_gan_bac9a81e" and r.get("method_id") == "identity"]
    plan_by_sample = {r.get("sample_id"): r for r in fit_rows}
    fit_ids = set(plan_by_sample)
    rows = manifest.get("rows", [])
    if len(fit_rows) != 8 or len(rows) != 8 or len(fit_ids) != 8 or {r.get("source_manifest_row_id") for r in rows} != fit_ids:
        raise ValueError("expected exactly the eight frozen historical fit utterances")
    model_by_id = {m.get("model_id"): m for m in plan.get("models", [])}
    w2l_plan = model_by_id.get("wav2lip_gan_bac9a81e", {})
    muse_plan = model_by_id.get("musetalk_fixed_portrait_attempt16", {})
    ditto_plan = model_by_id.get("ditto_hubert_fixed_portrait_attempt18", {})
    if str(args.wav2lip_root) != w2l_plan.get("generator", {}).get("repo"):
        raise ValueError("Wav2Lip root differs from the frozen fit model entry")
    if str(args.muse_model_workspace) != muse_plan.get("generator", {}).get("model_workspace"):
        raise ValueError("Muse workspace differs from the frozen fit model entry")
    expected_muse_processor = Path(muse_plan.get("generator", {}).get("model_source_root", "")) / "musetalk" / "utils" / "audio_processor.py"
    if args.muse_audio_processor != expected_muse_processor:
        raise ValueError("Muse audio-processor path differs from the frozen fit model source root")
    whisper_asset = next((a for a in muse_plan.get("model_assets", {}).get("whisper", []) if a.get("relative_path") == "preprocessor_config.json"), None)
    if whisper_asset is None or whisper_asset.get("sha256") != WHISPER_PREPROCESSOR_SHA256:
        raise ValueError("Muse Whisper preprocessor asset pin differs from the frozen model plan")
    if not ditto_plan or ditto_plan.get("model_id") != "ditto_hubert_fixed_portrait_attempt18":
        raise ValueError("frozen Ditto model entry is unavailable")
    if args.out.exists():
        raise FileExistsError("fresh output directory already exists")
    if args.out.is_symlink():
        raise ValueError("output path may not be a symlink")

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("CPU-only probe requires CUDA_VISIBLE_DEVICES to be empty")
    if args.runtime_discovery.is_symlink() or not args.runtime_discovery.is_file() or sha256_file(args.runtime_discovery) != RUNTIME_DISCOVERY_SHA256:
        raise ValueError("actual Muse Transformers runtime discovery evidence mismatch")
    runtime_contract_helper = load_runtime_contract_helper(args.runtime_helper)
    runtime_contract = runtime_contract_helper.verify_runtime_contract(args.runtime_discovery)
    if args.preflight_report.is_symlink() or not args.preflight_report.is_file():
        raise ValueError("preflight report is absent or unsafe")
    preflight_raw_sha256 = sha256_file(args.preflight_report)
    preflight_report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    if (
        preflight_report.get("status") != "PREFLIGHT_PASS"
        or preflight_report.get("fit_audio_rows") != 8
        or preflight_report.get("plan_raw_sha256") != PLAN_SHA256
        or preflight_report.get("transfer_manifest_raw_sha256") != TRANSFER_MANIFEST_SHA256
        or preflight_report.get("runner_source_sha256") != sha256_file(Path(__file__))
        or preflight_report.get("runtime_discovery_raw_sha256") != RUNTIME_DISCOVERY_SHA256
        or preflight_report.get("runtime_contract_helper_sha256") != RUNTIME_HELPER_SHA256
        or preflight_report.get("runtime_contract") != runtime_contract
        or preflight_report.get("model_or_encoder_forward") is not False
        or preflight_report.get("onnx_runtime_loaded") is not False
        or preflight_report.get("development_or_confirmation_audio") is not False
        or preflight_report.get("cuda_initialized") is not False
    ):
        raise ValueError("preflight report does not bind this exact runtime and frozen probe")
    if sys.version_info[:3] != (3, 10, 12):
        raise ValueError("frontend probe Python version differs from the frozen runtime")
    import librosa
    librosa_version = str(librosa.__version__)
    numpy_version = str(np.__version__)
    if librosa_version != "0.10.2.post1" or numpy_version != "1.26.4":
        raise ValueError("frontend probe NumPy/librosa versions differ from frozen runtime")
    runtime_versions = {
        "python": sys.version.split()[0],
        "numpy": numpy_version,
        "librosa": librosa_version,
    }
    runtime_versions["transformers"] = runtime_contract["transformers_distribution_version"]

    if not args.muse_audio_processor.is_file() or args.muse_audio_processor.is_symlink() or sha256_file(args.muse_audio_processor) != MUSE_AUDIO_PROCESSOR_SHA256:
        raise ValueError("pinned Muse audio processor source SHA mismatch")

    audio_module, mel_spec, w2l_meta = load_frontend_helper(args.wav2lip_helper, W2L_HELPER_SHA256, args.wav2lip_root)
    whisper_config = args.muse_model_workspace / "models" / "whisper" / "preprocessor_config.json"
    if not whisper_config.is_file() or whisper_config.is_symlink() or sha256_file(whisper_config) != WHISPER_PREPROCESSOR_SHA256:
        raise ValueError("pinned Muse Whisper preprocessor config unavailable at frozen workspace path")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoFeatureExtractor
    extractor = AutoFeatureExtractor.from_pretrained(str(whisper_config.parent), local_files_only=True)
    if type(extractor).__name__ != "WhisperFeatureExtractor":
        raise ValueError("pinned Muse asset did not resolve to WhisperFeatureExtractor")
    ditto = load_ditto_constants(args.ditto_constants, args.ditto_root_qc)
    if "onnxruntime" in sys.modules:
        raise RuntimeError("ONNX Runtime was imported in a frontend-only probe")
    torch_module = sys.modules.get("torch")
    if torch_module is None or torch_module.__version__ != "2.5.1+cu124":
        raise ValueError("frontend probe Torch version differs from frozen runtime")
    runtime_versions["torch"] = torch_module.__version__
    if bool(torch_module.cuda.is_initialized()):
        raise RuntimeError("frontend-only probe unexpectedly initialized CUDA")

    args.out.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema": "CROSS_TFG_FIT8_FRONTEND_GAIN_FEATURE_PROBE_V2",
        "status": "RUNNING",
        "scope": {
            "split": "historical_fit_only",
            "source_clip_count": 8,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "effects_scores_models_or_videos_read": False,
            "generator_model_forward": False,
            "audio_encoder_forward": False,
            "onnx_runtime_loaded": False,
            "feature_only": True,
            "waveforms_written_or_rerendered": False,
            "clip_or_peak_ceiling_applied": False,
            "interpretation_limit": "frontend-input response only; no learned encoder or video invariance claim",
        },
        "source_pins": {
            "plan_raw_sha256": PLAN_SHA256,
            "transfer_manifest_raw_sha256": TRANSFER_MANIFEST_SHA256,
            "wav2lip_helper_sha256": W2L_HELPER_SHA256,
            "wav2lip_audio_sha256": w2l_meta["audio_sha256"],
            "wav2lip_hparams_sha256": w2l_meta["hparams_sha256"],
            "whisper_preprocessor_sha256": WHISPER_PREPROCESSOR_SHA256,
            "muse_audio_processor_sha256": MUSE_AUDIO_PROCESSOR_SHA256,
            "ditto_onnx_sha256": DITTO_ONNX_SHA256,
            "ditto_constants_audit_raw_sha256": DITTO_CONSTANTS_SHA256,
            "ditto_constants_root_numeric_qc_sha256": DITTO_ROOT_QC_SHA256,
            "ditto_constants_root_numeric_receipt_sha256": DITTO_ROOT_RECEIPT_SHA256,
            "runtime_discovery_json_sha256": RUNTIME_DISCOVERY_SHA256,
            "runtime_discovery_response_raw_sha256": runtime_contract["runtime_discovery_response_raw_sha256"],
            "runtime_contract_helper_sha256": RUNTIME_HELPER_SHA256,
            "preflight_report_raw_sha256": preflight_raw_sha256,
            "transformers_module_sha256": runtime_contract["transformers_init_sha256"],
            "whisper_feature_module_sha256": runtime_contract["whisper_feature_module_sha256"],
            "whisper_call_source_sha256": runtime_contract["whisper_call_source_sha256"],
            "ditto_constants": ditto,
        },
        "frontend_contracts": {
            "wav2lip": {"operation": "official audio.melspectrogram on loaded float32 samples", "mel_spec": vars(mel_spec), "pre_mel_peak_rescale": None},
            "musetalk": {"operation": "pinned Transformers 4.39.2 WhisperFeatureExtractor call as Muse AudioProcessor uses it", "explicit_do_normalize_argument": False, "feature_extractor": type(extractor).__name__, "input_sample_rate_hz": SAMPLE_RATE, "call_source_sha256": runtime_contract["whisper_call_source_sha256"], "do_normalize_default": runtime_contract["whisper_call_defaults"]["do_normalize"], "sampling_rate_default": runtime_contract["whisper_call_defaults"]["sampling_rate"]},
            "ditto": {"operation": "first ONNX raw-input affine only: (x-subtract)/divide", "formula": "Sub_411(raw_audio, Constant_410=2.586069967946969e-05) then Div_413(result, Constant_412=0.20523720979690552)", "frame_envelope": "contiguous source-clock 640-sample RMS bins at 25 Hz; not HuBERT embeddings"},
        },
        "runtime_versions": runtime_versions,
        "runtime_contract": runtime_contract,
        "preflight_receipt": {"path": str(args.preflight_report), "raw_sha256": preflight_raw_sha256},
        "condition_definition": {
            "natural": "original canonical float32 samples",
            "global_minus_6db": "multiply by 10^(-6/20), no clipping",
            "global_plus_6db": "multiply by 10^(+6/20), no clipping",
            "fixed_rms_0p0376838173": "global scalar to RMS 0.0376838172675746, no peak ceiling",
        },
        "rows": [],
        "problems": [],
    }
    curves: dict[str, np.ndarray] = {}

    seen_samples: set[str] = set()
    for row in sorted(rows, key=lambda r: r["source_manifest_row_id"]):
        sample_id = row["source_manifest_row_id"]
        plan_row = plan_by_sample[sample_id]
        natural = plan_row.get("natural_pcm", {})
        if sample_id in seen_samples:
            raise ValueError(f"duplicate source audio in transfer manifest: {sample_id}")
        seen_samples.add(sample_id)
        if (
            plan_row.get("seed") != 20261003
            or natural.get("path") != row.get("remote_path")
            or natural.get("file_sha256") != row.get("file_sha256")
            or natural.get("pcm_sha256") != row.get("pcm_float32_sha256")
            or natural.get("sample_count") != row.get("sample_count")
            or natural.get("sample_rate_hz") != row.get("sample_rate_hz")
            or natural.get("channels") != row.get("channels")
            or natural.get("subtype") != "PCM_16"
            or row.get("sample_rate_hz") != SAMPLE_RATE
            or row.get("channels") != 1
            or row.get("sample_count", 0) >= 30 * SAMPLE_RATE
        ):
            raise ValueError(f"source PCM binding or single-segment Muse support differs from frozen inputs: {sample_id}")
        path = Path(row["remote_path"])
        if not path.is_file() or path.is_symlink() or sha256_file(path) != row["file_sha256"]:
            raise ValueError(f"source WAV byte pin mismatch: {sample_id}")
        # The source files are canonical mono 16 kHz PCM_16. Do not resample;
        # let any unexpected source rate fail closed below.
        audio, sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if sr != SAMPLE_RATE or audio.size != row["sample_count"] or not np.isfinite(audio).all():
            raise ValueError(f"canonical sample support mismatch: {sample_id}")
        if sha256_f32(audio) != row["pcm_float32_sha256"]:
            raise ValueError(f"decoded float32 PCM hash mismatch: {sample_id}")
        variants, variant_meta = conditions(audio)
        clip_record: dict[str, Any] = {
            "sample_id": sample_id,
            "speaker_id": sample_id.split("_", 1)[0],
            "source_wav_path": str(path),
            "source_wav_sha256": row["file_sha256"],
            "source_pcm_float32_sha256": row["pcm_float32_sha256"],
            "source_pcm16_sha256": row["direct_source_pcm16_sha256"],
            "source_archive_path": row.get("source_archive_path"),
            "source_archive_sha256": row.get("source_archive_sha256"),
            "source_archive_member": row.get("source_archive_member"),
            "sample_count": int(audio.size),
            "sample_rate_hz": int(sr),
            "source_waveform": finite_stats(audio),
            "conditions": variant_meta,
            "frontends": {},
        }

        w2l_features: dict[str, np.ndarray] = {}
        muse_features: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            w = variants[condition]
            mel = normalize_feature_shape(audio_module.melspectrogram(w))
            w2l_features[condition] = mel
            mel_valid = min(mel.shape[1], int(math.floor(audio.size / mel_spec.hop_length)) + 1)
            clip_record["frontends"].setdefault("wav2lip_mel", {})[condition] = tensor_summary("wav2lip_mel", mel, mel_valid)

            # Keep the pinned Muse call contract: do not pass do_normalize or
            # any gain, padding, or truncation override.
            whisper_batch = extractor(w, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            whisper = normalize_feature_shape(whisper_batch.input_features)
            muse_features[condition] = whisper
            hop = int(getattr(extractor, "hop_length", 160))
            whisper_valid = min(whisper.shape[1], int(math.floor(audio.size / hop)) + 1)
            clip_record["frontends"].setdefault("muse_whisper_logmel", {})[condition] = tensor_summary("muse_whisper_logmel", whisper, whisper_valid)

            sub, div = np.float32(ditto["subtract"]), np.float32(ditto["divide"])
            affine = np.ascontiguousarray((w - sub) / div, dtype=np.float32)
            if not np.isfinite(affine).all():
                raise ValueError("Ditto raw affine produced non-finite values")
            frame_rms = ditto_frame_rms(affine)
            raw_frame_rms = ditto_frame_rms(w)
            clip_record["frontends"].setdefault("ditto_hubert_input_affine_only", {})[condition] = {
                "formula": "(x - 2.586069967946969e-05) / 0.20523720979690552",
                "affine_waveform_stats": finite_stats(affine),
                "fraction_exact_zero": float(np.mean(affine == 0.0)),
                "fraction_abs_lt_1e-7": float(np.mean(np.abs(affine) < 1e-7)),
                "fraction_abs_gt_1": float(np.mean(np.abs(affine) > 1.0)),
                "clipping_operation_at_input_affine": False,
                "frame_rms_25hz_stats": finite_stats(frame_rms),
                "raw_frame_rms_25hz_stats": finite_stats(raw_frame_rms),
            }
            key = f"{row['source_manifest_row_id']}|ditto|{condition}"
            curves[key + "|frame_rms"] = frame_rms
            curves[key + "|raw_frame_rms"] = raw_frame_rms

        for frontend_name, tensor_set, valid_fn in (
            ("wav2lip_mel", w2l_features, lambda z: int(math.floor(audio.size / mel_spec.hop_length)) + 1),
            ("muse_whisper_logmel", muse_features, lambda z: int(math.floor(audio.size / int(getattr(extractor, "hop_length", 160)))) + 1),
        ):
            reference = tensor_set["natural"]
            for condition in CONDITIONS:
                current = tensor_set[condition]
                valid = min(current.shape[1], valid_fn(current))
                if condition == "natural":
                    metrics, trace = feature_curve_metrics(reference, reference, valid)
                else:
                    metrics, trace = feature_curve_metrics(reference, current, valid)
                clip_record["frontends"][frontend_name][condition]["relative_to_natural"] = metrics
                for curve_name, curve in trace.items():
                    key = f"{row['source_manifest_row_id']}|{frontend_name}|{condition}|{curve_name}"
                    curves[key] = curve

        ditto_base = variants["natural"]
        ditto_affine_base = (ditto_base - np.float32(ditto["subtract"])) / np.float32(ditto["divide"])
        baseline_frame = ditto_frame_rms(ditto_affine_base).astype(np.float64)
        for condition in CONDITIONS:
            item = clip_record["frontends"]["ditto_hubert_input_affine_only"][condition]
            trace = curves[f"{row['source_manifest_row_id']}|ditto|{condition}|frame_rms"].astype(np.float64)
            item["frame_rms_correlation_with_natural"] = corr(baseline_frame, trace)
            item["frame_rms_relative_change"] = float(np.sqrt(np.mean((trace - baseline_frame) ** 2)))
        summary["rows"].append(clip_record)

    summary["status"] = "PASS_FRONTEND_INPUT_FEATURES_ONLY"
    summary["row_count"] = len(summary["rows"])
    summary["curve_count"] = len(curves)
    curve_path = args.out / "time_curves.npz"
    np.savez_compressed(curve_path, **curves)
    summary_raw = json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    summary_path = args.out / "summary.json"
    summary_path.write_bytes(summary_raw)
    with (args.out / "per_frontend_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("sample_id", "speaker_id", "frontend", "condition", "feature_shape", "valid_frames", "min", "max", "mean", "std", "floor_or_clip_fraction_waveform_supported", "floor_or_clip_fraction_full_tensor", "opposite_clip_fraction_waveform_supported", "relative_common_delta_mean", "centered_relative_rms_change", "centered_frame_cosine_mean"))
        for row in summary["rows"]:
            for frontend, conditions_map in row["frontends"].items():
                for condition, data in conditions_map.items():
                    if frontend.startswith("ditto"):
                        stats = data["affine_waveform_stats"]
                        writer.writerow((row["sample_id"], row["speaker_id"], frontend, condition, "waveform", row["sample_count"], stats["min"], stats["max"], stats["mean"], stats["std"], "", "", "", "", "", ""))
                    else:
                        stats = data["valid_feature_stats"]
                        floor = data["floor_or_clip"]
                        rel = data["relative_to_natural"]
                        if frontend == "wav2lip_mel":
                            floor_fraction = floor["fraction_exact_min_waveform_supported"]
                            full_fraction = floor["fraction_exact_min_full_tensor"]
                            opposite_fraction = floor["fraction_exact_max_waveform_supported"]
                        else:
                            floor_fraction = floor["fraction_at_relative_logmel_floor_waveform_supported"]
                            full_fraction = floor["fraction_at_relative_logmel_floor_full_padded_tensor"]
                            opposite_fraction = ""
                        writer.writerow((row["sample_id"], row["speaker_id"], frontend, condition, "x".join(map(str, data["shape_feature_time"])), data["valid_waveform_supported_frames"], stats["min"], stats["max"], stats["mean"], stats["std"], floor_fraction, full_fraction, opposite_fraction, rel["common_mode_delta_mean"], rel["centered_relative_rms_change"], rel["centered_frame_cosine_mean"]))
    outputs = [summary_path, curve_path, args.out / "per_frontend_summary.csv"]
    receipt = {
        "schema": "CROSS_TFG_FIT8_FRONTEND_GAIN_FEATURE_PROBE_RECEIPT_V2",
        "status": summary["status"],
        "summary": {"path": str(summary_path), "bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "curves_npz": {"path": str(curve_path), "bytes": curve_path.stat().st_size, "sha256": sha256_file(curve_path)},
        "csv": {"path": str(outputs[2]), "bytes": outputs[2].stat().st_size, "sha256": sha256_file(outputs[2])},
        "input_audio_count": len(summary["rows"]),
        "model_or_encoder_forward": False,
        "video_render": False,
        "onnx_runtime_loaded": False,
        "development_or_confirmation_audio": False,
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
