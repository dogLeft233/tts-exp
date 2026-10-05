"""One-row CPU engineering qualification of V3 through the frozen V4 path.

This is not a training entrypoint. It performs no optimizer step, no dev or
confirmation access, no CUDA work, and saves no source/candidate waveform.
Model loading and the 31-lag forward are attempted on CPU only; CPU support is
not claimed until a real execution completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross_tfg_v3_v4_native_cpu_engineering_qualification_attempt07"
PREPARATION_ONLY = True
EXPECTED_PIN_KEYS = (
    "adapter_source_sha256",
    "v3_head_source_sha256",
    "v4_runtime_source_sha256",
    "v2_runtime_source_sha256",
    "fit_runtime_source_sha256",
    "wav2lip_frontend_source_sha256",
    "native_crop_source_sha256",
    "heads_source_sha256",
    "training_source_sha256",
    "fit_cli_v4_source_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: Any, np: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json_pinned(spec: Mapping[str, Any], label: str) -> tuple[dict[str, Any], Path]:
    path = Path(str(spec["path"])).resolve(strict=True)
    if not path.is_file() or _sha256_file(path) != str(spec["sha256"]).lower():
        raise ValueError(f"{label} raw file pin mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must decode to an object")
    return payload, path


def _verify_source_closure(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for relative, expected in config["source_files"].items():
        path = (repo_root / relative).resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(f"pinned source is not a regular file: {relative}")
        actual = _sha256_file(path)
        if actual != str(expected).lower():
            raise ValueError(f"source hash mismatch: {relative}")
        checked[relative] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return checked


def _as_float(value: Any, torch: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite scalar in V3/V4 CPU qualification")
    return result


def _numpy_array(value: Any, np: Any) -> Any:
    return np.ascontiguousarray(value.detach().to(device="cpu").numpy())


def _make_smooth_probe_weights(probe: Mapping[str, Any], count: int, torch: Any) -> tuple[Any, Any]:
    sample_rate = int(probe["sample_rate_hz"])
    t = torch.arange(count, dtype=torch.float32, device="cpu") / float(sample_rate)
    objective = probe["objective"]
    linear = torch.zeros_like(t)
    for term in objective["linear_terms"]:
        linear = linear + float(term["coefficient"]) * torch.sin(
            2.0 * math.pi * float(term["frequency_hz"]) * t + float(term["phase_radians"])
        )
    quadratic = torch.full_like(t, float(objective["quadratic_weight_offset"]))
    for term in objective["quadratic_weight_terms"]:
        quadratic = quadratic + float(term["coefficient"]) * torch.cos(
            2.0 * math.pi * float(term["frequency_hz"]) * t + float(term["phase_radians"])
        )
    return linear.contiguous(), quadratic.contiguous()


def _smooth_waveform_only_loss(
    head: Any,
    source: Any,
    baseline: Any,
    linear_weight: Any,
    quadratic_weight: Any,
    loss_scale: float,
    torch: Any,
) -> Any:
    result = head(source, baseline)
    residual = result["output_pcm_float32"] - baseline
    return float(loss_scale) * (
        (residual * linear_weight).mean() + (residual.square() * quadratic_weight).mean()
    )


def _cpu_cuda_state(torch: Any) -> bool:
    # Read the private lazy-init flag only; do not call CUDA availability APIs.
    return bool(getattr(torch.cuda, "_initialized", False))


def _rss_high_water_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except Exception:
        return None


def _run(config_path: Path, config_sha_expected: str, output_dir: Path) -> dict[str, Any]:
    start_time = time.monotonic()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"fresh output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    config_raw_sha = _sha256_file(config_path)
    if config_raw_sha != config_sha_expected.lower():
        raise ValueError("qualification config raw SHA mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("qualification config schema mismatch")
    repo_root = Path(str(config["repo_root"])).resolve(strict=True)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty before Python starts")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for the bounded CPU run")
    if sys.executable != config["runtime"]["python_executable_literal"]:
        raise RuntimeError("CPU qualification interpreter path differs from the pinned literal")

    source_checked = _verify_source_closure(repo_root, config)
    for key in EXPECTED_PIN_KEYS:
        if key not in config["runtime_source_identity"]:
            raise ValueError(f"missing runtime-source identity pin: {key}")
    for pin_key, source_path_key in (
        ("adapter_source_sha256", "adapter_source_relative"),
        ("v3_head_source_sha256", "v3_head_source_relative"),
        ("v4_runtime_source_sha256", "v4_runtime_source_relative"),
        ("v2_runtime_source_sha256", "v2_runtime_source_relative"),
        ("fit_runtime_source_sha256", "fit_runtime_source_relative"),
        ("wav2lip_frontend_source_sha256", "wav2lip_frontend_source_relative"),
        ("native_crop_source_sha256", "native_crop_source_relative"),
        ("heads_source_sha256", "heads_source_relative"),
        ("training_source_sha256", "training_source_relative"),
        ("fit_cli_v4_source_sha256", "fit_cli_v4_source_relative"),
    ):
        relative = config["runtime_source_identity"][source_path_key]
        if source_checked[relative]["sha256"] != str(config["runtime_source_identity"][pin_key]).lower():
            raise ValueError(f"runtime source identity pin disagrees with source closure: {relative}")

    import numpy as np
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if str(torch.__version__) != config["runtime"]["torch_version"]:
        raise RuntimeError("Torch version differs from the pinned CPU runtime")
    if str(np.__version__) != config["runtime"]["numpy_version"]:
        raise RuntimeError("NumPy version differs from the pinned CPU runtime")
    if int(torch.get_num_threads()) != 1:
        raise RuntimeError("Torch intra-op thread count is not one")
    if _cpu_cuda_state(torch):
        raise RuntimeError("CUDA was initialized before CPU qualification")
    device = torch.device("cpu")

    plan_overlay, plan_path = _load_json_pinned(config["inputs"]["v4_plan"], "frozen V4 parent-overlay plan")
    from scripts.audio_enhancement import fit_cli_v4_attempt01 as fit_cli_v4
    plan = fit_cli_v4._load_effective_plan(plan_overlay)
    parent_ref = plan_overlay.get("parent_plan_ref", {})
    parent_pin = config["inputs"]["v2_parent_plan"]
    if (
        parent_ref.get("path") != parent_pin["path"]
        or parent_ref.get("raw_sha256") != parent_pin["sha256"]
        or parent_ref.get("payload_sha256") != parent_pin["payload_sha256"]
    ):
        raise ValueError("V4 overlay parent ref differs from the exact frozen V2 plan pin")
    if plan.get("plan_payload_sha256") != config["inputs"]["v4_plan"]["effective_plan_payload_sha256"]:
        raise ValueError("merged effective plan payload differs from the frozen V4 overlay pin")
    fit_rows = plan.get("rows")
    if not isinstance(fit_rows, list) or len(fit_rows) != 8:
        raise ValueError("V4 plan must carry the exact eight historical fit rows")
    if any(row.get("split") != "fit" or not row.get("historical_exposed") for row in fit_rows):
        raise ValueError("V4 CPU qualification may read only the frozen exposed fit rows")
    row_id = str(config["inputs"]["selected_row_id"])
    row_matches = [row for row in fit_rows if str(row.get("row_id")) == row_id]
    if len(row_matches) != 1:
        raise ValueError("selected fit row is not unique in the frozen V4 plan")
    row = row_matches[0]
    if row.get("sample_id") != config["inputs"]["sample_id"] or row.get("speaker_id") != config["inputs"]["speaker_id"]:
        raise ValueError("selected fit row identity differs from the sealed config")
    if row.get("natural_pcm", {}).get("path") != config["inputs"]["source_wav"]["path"]:
        raise ValueError("selected fit row PCM path differs from the sealed config")
    if row.get("natural_pcm", {}).get("file_sha256") != config["inputs"]["source_wav"]["sha256"]:
        raise ValueError("selected fit row PCM file hash differs from the sealed config")
    if int(row.get("natural_pcm", {}).get("sample_count", -1)) != int(config["inputs"]["source_wav"]["sample_count"]):
        raise ValueError("selected fit row sample count differs from the sealed config")

    fit_v4_runtime = plan.get("runtime", {}).get("fit_v4")
    if not isinstance(fit_v4_runtime, dict):
        raise ValueError("V4 plan lacks the frozen native runtime configuration")
    if fit_v4_runtime.get("native_geometry_manifest_path") != config["inputs"]["geometry_manifest"]["path"]:
        raise ValueError("geometry manifest path differs from V4 plan")
    if fit_v4_runtime.get("native_geometry_manifest_sha256") != config["inputs"]["geometry_manifest"]["sha256"]:
        raise ValueError("geometry manifest SHA differs from V4 plan")
    v4_cfg = plan.get("fit_config", {}).get("v4_objective")
    if not isinstance(v4_cfg, dict):
        raise ValueError("V4 plan lacks its frozen objective block")
    expected_weights = config["objective"]["weights"]
    frozen_weights = {
        "relative_Cbest_improvement": 1.0,
        "far_lag_contrast_improvement": 0.25,
        "D0_nonworsening_guard": 2.0,
        "Canchor_nonworsening_guard": 1.0,
        "visual_embedding_log_norm_ratio_squared": 0.1,
        "waveform_relative_mse_to_rms_reference": 0.02,
        "mask_total_variation": 0.002,
    }
    if expected_weights != frozen_weights:
        raise ValueError("qualification config does not carry the exact frozen V4 objective weights")
    if v4_cfg.get("objective_weights") != frozen_weights:
        raise ValueError("V4 objective weights differ from this qualification config")
    if v4_cfg.get("one_sided_guard_slack") != {"D0": 0.0, "Canchor": 0.0}:
        raise ValueError("V4 D0/Canchor guard slack is not the frozen zero value")
    if v4_cfg.get("rms_reference", {}).get("target_rms") != config["rms_policy"]["target_rms"]:
        raise ValueError("V4 RMS target differs from the V3 qualification config")
    if v4_cfg.get("rms_reference", {}).get("peak_ceiling") != config["rms_policy"]["peak_ceiling"]:
        raise ValueError("V4 final peak policy differs from the V3 qualification config")
    if v4_cfg.get("rms_reference", {}).get("rms_stage_peak_ceiling") != config["rms_policy"]["rms_stage_peak_ceiling"]:
        raise ValueError("V4 RMS-stage peak policy differs from the V3 qualification config")
    if v4_cfg.get("rms_reference", {}).get("post_rms_match") != config["rms_policy"]["post_rms_match"]:
        raise ValueError("V4 post-RMS matching policy differs from the V3 qualification config")
    if v4_cfg.get("rms_reference", {}).get("use_local_envelope") is not False or v4_cfg.get("rms_reference", {}).get("use_spectral_head") is not False:
        raise ValueError("V4 canonical RMS reference unexpectedly includes another transform")

    # Read source and frozen cached features only. Never call the cache builder,
    # which can create or replace files under the original fit cache root.
    from scripts.audio_enhancement import wav2lip_fit_runtime as fit_runtime
    from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v2_attempt02 as v2_runtime
    from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v4_attempt01 as v4_runtime
    from scripts.audio_enhancement import shared_recipient_transition_head_v3 as head_module
    from scripts.audio_enhancement import v3_v4_native_runtime_adapter_attempt03 as adapter_module

    runtime_cfg = plan["runtime"]
    if os.environ.get("CROSS_TFG_PSF_ROOT") != runtime_cfg["execution"]["python_speech_features_root"]:
        raise RuntimeError("python_speech_features overlay root differs from frozen plan")
    source_audio_path = Path(str(config["inputs"]["source_wav"]["path"])).resolve(strict=True)
    if _sha256_file(source_audio_path) != config["inputs"]["source_wav"]["sha256"]:
        raise ValueError("source WAV raw file SHA mismatch")
    audio, audio_info = fit_runtime.read_canonical_wav(source_audio_path)
    if (
        audio.dtype != np.float32
        or audio.ndim != 1
        or audio_info.frames != int(config["inputs"]["source_wav"]["sample_count"])
        or audio_info.sample_rate != 16_000
        or audio_info.source_channels != 1
        or audio_info.pcm_float32_sha256 != config["inputs"]["source_wav"]["pcm_float32_sha256"]
    ):
        raise ValueError("canonical source PCM identity/type/count mismatch")
    source = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device=device)

    context_arrays, context_metadata, context_file_pins = v2_runtime._load_context_seed(runtime_cfg)
    if context_file_pins["npz_sha256"] != config["inputs"]["context_npz"]["sha256"]:
        raise ValueError("fixed context NPZ raw SHA differs from config")
    if context_file_pins["metadata_sha256"] != config["inputs"]["context_metadata"]["sha256"]:
        raise ValueError("fixed context metadata raw SHA differs from config")

    audio_cache_npz = Path(str(config["inputs"]["audio_embedding_cache_npz"]["path"])).resolve(strict=True)
    audio_cache_meta_path = Path(str(config["inputs"]["audio_embedding_cache_metadata"]["path"])).resolve(strict=True)
    if _sha256_file(audio_cache_npz) != config["inputs"]["audio_embedding_cache_npz"]["sha256"]:
        raise ValueError("full-audio embedding cache NPZ raw SHA mismatch")
    audio_cache_meta, _ = _load_json_pinned(config["inputs"]["audio_embedding_cache_metadata"], "full-audio cache metadata")
    if audio_cache_meta.get("npz_sha256") != config["inputs"]["audio_embedding_cache_npz"]["sha256"]:
        raise ValueError("full-audio embedding cache metadata does not bind pinned NPZ")
    with np.load(audio_cache_npz, allow_pickle=False) as archive:
        if set(archive.files) != {"mfcc_frames_by_coeff", "audio_embeddings", "audio_window_start_mfcc"}:
            raise ValueError("full-audio embedding cache has an unexpected NPZ inventory")
        cache_mfcc = np.ascontiguousarray(archive["mfcc_frames_by_coeff"])
        audio_embeddings_np = np.ascontiguousarray(archive["audio_embeddings"])
        audio_window_starts = np.ascontiguousarray(archive["audio_window_start_mfcc"])
    for name, array in (
        ("mfcc_frames_by_coeff", cache_mfcc),
        ("audio_embeddings", audio_embeddings_np),
        ("audio_window_start_mfcc", audio_window_starts),
    ):
        record = audio_cache_meta.get("arrays", {}).get(name)
        if not isinstance(record, dict) or list(array.shape) != record.get("shape") or str(array.dtype) != record.get("dtype"):
            raise ValueError(f"full-audio cache array shape/dtype mismatch: {name}")
        if _array_sha256(array, np) != record.get("array_sha256"):
            raise ValueError(f"full-audio cache array hash mismatch: {name}")
    expected_embedding_count = int(row["full_context_fit"]["scoring_audio_embedding_count"])
    if audio_embeddings_np.shape != (expected_embedding_count, 1024) or audio_embeddings_np.dtype != np.float32:
        raise ValueError("cached audio embeddings differ from the official full-clip query count")
    query_start = int(audio_cache_meta["query_start_frame"])
    expected_query_start = (expected_embedding_count - 27) // 2
    if query_start != expected_query_start or not (query_start - 15 >= 0 and query_start + 26 + 15 < expected_embedding_count):
        raise ValueError("cached 27-query center lacks complete unpadded 31-lag support")
    audio_embeddings = torch.from_numpy(audio_embeddings_np).to(device=device, dtype=torch.float32)

    loaded_models = fit_runtime._load_models(runtime_cfg, device)
    wav2lip_model, syncnet_model, mel_frontend, audio_module, mfcc_function, model_source_pins = loaded_models
    del audio_module, mfcc_function
    if _cpu_cuda_state(torch):
        raise RuntimeError("lower-level model loader initialized CUDA during CPU qualification")

    # The lower-level loader is used to avoid constructing the training factory,
    # so restore the pinned V2 runtime checks explicitly before any model forward.
    import librosa
    execution = runtime_cfg.get("execution", {})
    expected_librosa_version = str(execution.get("librosa_version", ""))
    actual_librosa_version = str(librosa.__version__)
    if not expected_librosa_version or actual_librosa_version != expected_librosa_version:
        raise RuntimeError("librosa module version differs from the frozen V2 plan")
    psf_record = model_source_pins.get("python_speech_features", {})
    psf_source_files = psf_record.get("source_files", [])
    psf_module_path = Path(str(psf_record.get("module_path", ""))).resolve(strict=True)
    psf_package_root = psf_module_path.parent
    actual_psf_tree_sha = v2_runtime._python_source_tree_sha(psf_source_files, psf_package_root)
    expected_psf_tree_sha = str(execution.get("python_speech_features_tree_sha256", "")).lower()
    if not expected_psf_tree_sha or actual_psf_tree_sha != expected_psf_tree_sha:
        raise RuntimeError("python_speech_features source tree differs from the frozen V2 plan")
    for model in (wav2lip_model, syncnet_model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    # Only compute the pinned official W2L conditioning mel to bind the existing
    # context's selected source-clock frame starts; this is not a model forward.
    with torch.no_grad():
        full_mel = mel_frontend(source)
    all_mel_starts = v2_runtime.official_chunk_starts(int(full_mel.shape[-1]), fps_num=25, fps_den=1)
    source_frames = list(range(query_start, query_start + 32))
    if source_frames[-1] >= len(all_mel_starts):
        raise ValueError("full-audio W2L mel clock does not cover the selected 32-frame window")
    if source_frames[-1] >= int(context_arrays["face_crops_bgr_u8"].shape[0]):
        raise ValueError("fixed portrait context does not cover the selected 32 source-clock frames")
    from scripts.audio_enhancement.wav2lip_fullcontext_fit_runtime_v2_attempt02 import FullContextFitContext

    context = FullContextFitContext(
        face_crops_bgr=np.ascontiguousarray(context_arrays["face_crops_bgr_u8"][source_frames]),
        syncnet_crops_bgr=np.ascontiguousarray(context_arrays["syncnet_crops_bgr_u8"][source_frames]),
        patch_boxes_xyxy=np.ascontiguousarray(context_arrays["patch_boxes_xyxy_i32"][source_frames]),
        mel_starts=tuple(int(value) for value in all_mel_starts[query_start : query_start + 32]),
        source_video_frame_indices=tuple(source_frames),
        frame_timestamps_seconds=tuple(float(index) / 25.0 for index in source_frames),
        audio_embeddings=audio_embeddings,
        audio_embedding_count=int(audio_embeddings.shape[0]),
        query_start_frame=query_start,
        query_count=27,
        expected_sample_count=int(audio.size),
        source_video_sha256=str(context_metadata["source_video_sha256"]),
        cache_record={
            "row_id": row_id,
            "npz_sha256": config["inputs"]["audio_embedding_cache_npz"]["sha256"],
            "metadata_sha256": config["inputs"]["audio_embedding_cache_metadata"]["sha256"],
        },
    )
    context.validate()

    geometry_manifest, geometry_by_row, geometry_records = v4_runtime._load_native_geometry_manifest(
        plan, fit_rows, fit_v4_runtime
    )
    if _sha256_file(Path(config["inputs"]["geometry_manifest"]["path"])) != config["inputs"]["geometry_manifest"]["sha256"]:
        raise ValueError("native geometry manifest raw SHA mismatch after V4 load")
    geometry_record = geometry_by_row[row_id]
    sync_adapter = v2_runtime.FullContextSyncAdapter(
        wav2lip_model=wav2lip_model,
        syncnet_model=syncnet_model,
        mel_frontend=mel_frontend,
        frame_batch_size=1,
        syncnet_batch_size=1,
    )

    baseline_npz = Path(str(config["inputs"]["v4_rms_cache_npz"]["path"])).resolve(strict=True)
    baseline_meta, baseline_meta_path = _load_json_pinned(
        config["inputs"]["v4_rms_cache_metadata"], "V4 RMS baseline metadata"
    )
    if _sha256_file(baseline_npz) != config["inputs"]["v4_rms_cache_npz"]["sha256"]:
        raise ValueError("V4 canonical RMS cache NPZ raw SHA mismatch")
    if baseline_meta.get("schema") != v4_runtime.BASELINE_CACHE_SCHEMA:
        raise ValueError("V4 canonical RMS cache schema mismatch")
    baseline_payload = {key: value for key, value in baseline_meta.items() if key != "metadata_payload_sha256"}
    if v4_runtime._canonical_sha(baseline_payload) != baseline_meta.get("metadata_payload_sha256"):
        raise ValueError("V4 RMS baseline metadata payload seal mismatch")
    if baseline_meta.get("npz_file_sha256") != config["inputs"]["v4_rms_cache_npz"]["sha256"]:
        raise ValueError("V4 baseline metadata does not bind the pinned NPZ")
    with np.load(baseline_npz, allow_pickle=False) as archive:
        expected_names = {
            "rms_reference_audio_float32",
            "rms_visual_embedding_float32",
            "rms_distance_curve_float32",
            "query_frame_index_int32",
        }
        if set(archive.files) != expected_names:
            raise ValueError("V4 RMS baseline cache array inventory mismatch")
        baseline_arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for name, array in baseline_arrays.items():
        record = baseline_meta.get("arrays", {}).get(name)
        if not isinstance(record, dict) or list(array.shape) != record.get("shape") or str(array.dtype) != record.get("dtype"):
            raise ValueError(f"V4 RMS baseline array metadata mismatch: {name}")
        if _array_sha256(array, np) != record.get("array_sha256"):
            raise ValueError(f"V4 RMS baseline array payload SHA mismatch: {name}")
    baseline_audio_np = baseline_arrays["rms_reference_audio_float32"]
    baseline_visual_np = baseline_arrays["rms_visual_embedding_float32"]
    baseline_curve_np = baseline_arrays["rms_distance_curve_float32"]
    query_map_np = baseline_arrays["query_frame_index_int32"]
    if baseline_audio_np.shape != audio.shape or baseline_audio_np.dtype != np.float32:
        raise ValueError("V4 RMS waveform cache differs from source waveform shape/dtype")
    if baseline_visual_np.shape != (27, 1024) or baseline_visual_np.dtype != np.float32:
        raise ValueError("V4 RMS visual embedding cache differs from the 27 x 1024 contract")
    if baseline_curve_np.shape != (31,) or baseline_curve_np.dtype != np.float32:
        raise ValueError("V4 RMS curve cache differs from the 31-lag contract")
    expected_query_indices = np.arange(query_start, query_start + 27, dtype=np.int32)
    if not np.array_equal(query_map_np, expected_query_indices):
        raise ValueError("V4 RMS baseline queries differ from the unchanged original source clock")
    baseline = torch.from_numpy(baseline_audio_np.copy()).to(device=device, dtype=torch.float32)
    baseline_curve = torch.from_numpy(baseline_curve_np.copy()).to(device=device, dtype=torch.float32)
    baseline_visual = torch.from_numpy(baseline_visual_np.copy()).to(device=device, dtype=torch.float32)

    cpu_cfg, cpu_cfg_path = _load_json_pinned(config["inputs"]["v3_cpu_qualification_config"], "V3 CPU config")
    head_kwargs = dict(cpu_cfg["head_constructor"])
    if head_kwargs.get("peak_ceiling") != config["rms_policy"]["v3_config_peak_ceiling_before_v4_mapping"]:
        raise ValueError("V3 CPU constructor source config changed from its sealed value")
    head_kwargs["peak_ceiling"] = float(config["rms_policy"]["peak_ceiling"])
    if not math.isclose(float(head_kwargs["target_rms"]), float(config["rms_policy"]["target_rms"]), rel_tol=0.0, abs_tol=0.0):
        raise ValueError("V3 head and V4 baseline RMS targets differ")
    head = head_module.SharedRecipientTransitionHeadV3(**head_kwargs).to(device=device)
    head.eval()
    zero_theta = torch.zeros((3, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        head.band_contrasts_db.copy_(zero_theta)
    with torch.no_grad():
        zero_result = head(source, baseline)
    zero_audio_np = np.ascontiguousarray(zero_result["output_pcm_float32"].detach().cpu().numpy(), dtype=np.float32)
    zero_audio_equal = np.array_equal(zero_audio_np, baseline_audio_np)
    zero_audio_max_abs = float(np.max(np.abs(zero_audio_np.astype(np.float64) - baseline_audio_np.astype(np.float64))))
    if not zero_audio_equal:
        raise RuntimeError("zero-parameter V3 output is not byte-identical to the canonical V4 RMS waveform")
    if _cpu_cuda_state(torch):
        raise RuntimeError("CUDA initialized before the V4 baseline parity render")

    # Score direct canonical-RMS PCM and zero-parameter V3 output separately
    # with the same CPU scorer, batch sizes, context, and native geometry.
    # This same-device reproduction gate is distinct from the preserved strict
    # comparison against the previously sealed CUDA-generated cache.
    with torch.no_grad():
        cpu_rms_score = adapter_module.score_with_v4_native_path(
            v4_runtime=v4_runtime,
            v4_adapter=sync_adapter,
            context=context,
            geometry_record=geometry_record,
            enhanced_audio=baseline,
        )
        zero_score = adapter_module.score_with_v4_native_path(
            v4_runtime=v4_runtime,
            v4_adapter=sync_adapter,
            context=context,
            geometry_record=geometry_record,
            enhanced_audio=zero_result["output_pcm_float32"].detach(),
        )
    cpu_rms_curve_np = np.ascontiguousarray(
        cpu_rms_score["curve_float32"].detach().cpu().numpy(), dtype=np.float32
    )
    cpu_rms_visual_np = np.ascontiguousarray(
        cpu_rms_score["visual_embeddings_float32"].detach().cpu().numpy(), dtype=np.float32
    )
    zero_curve_np = np.ascontiguousarray(zero_score["curve_float32"].detach().cpu().numpy(), dtype=np.float32)
    zero_visual_np = np.ascontiguousarray(zero_score["visual_embeddings_float32"].detach().cpu().numpy(), dtype=np.float32)
    same_cpu_curve_abs = np.abs(cpu_rms_curve_np.astype(np.float64) - zero_curve_np.astype(np.float64))
    same_cpu_visual_abs = np.abs(cpu_rms_visual_np.astype(np.float64) - zero_visual_np.astype(np.float64))
    same_cpu_curve_max_abs = float(same_cpu_curve_abs.max())
    same_cpu_visual_max_abs = float(same_cpu_visual_abs.max())
    same_cpu_curve_parity = same_cpu_curve_max_abs <= float(config["parity_tolerances"]["curve_max_abs"])
    same_cpu_visual_parity = same_cpu_visual_max_abs <= float(
        config["parity_tolerances"]["visual_embedding_max_abs"]
    )
    curve_abs = np.abs(zero_curve_np.astype(np.float64) - baseline_curve_np.astype(np.float64))
    visual_abs = np.abs(zero_visual_np.astype(np.float64) - baseline_visual_np.astype(np.float64))
    curve_max_abs = float(curve_abs.max())
    visual_max_abs = float(visual_abs.max())
    curve_parity = curve_max_abs <= float(config["parity_tolerances"]["curve_max_abs"])
    visual_parity = visual_max_abs <= float(config["parity_tolerances"]["visual_embedding_max_abs"])
    if _cpu_cuda_state(torch):
        raise RuntimeError("CUDA initialized during the CPU-only zero-init parity render")

    probe = cpu_cfg["synthetic_probe"]
    theta_np = np.asarray(probe["diagnostic_parameter_matrix_db"], dtype=np.float32)
    if theta_np.shape != (3, 3) or not np.isfinite(theta_np).all() or not np.any(theta_np):
        raise ValueError("sealed nine-parameter finite-gradient probe matrix is invalid")
    with torch.no_grad():
        head.band_contrasts_db.copy_(torch.from_numpy(theta_np).to(device=device))
    candidate_result = head(source, baseline)
    objective = adapter_module.v4_objective_for_v3_candidate(
        torch=torch,
        functional=torch.nn.functional,
        v4_runtime=v4_runtime,
        v4_adapter=sync_adapter,
        context=context,
        geometry_record=geometry_record,
        v3_result=candidate_result,
        rms_curve=baseline_curve,
        rms_visual_embeddings=baseline_visual,
        objective_weights=v4_cfg["objective_weights"],
    )
    candidate_curve = objective["score"]["curve_float32"]
    v4_loss = objective["loss"]
    parameter = head.band_contrasts_db
    v4_parameter_gradient = torch.autograd.grad(v4_loss, parameter, retain_graph=True, allow_unused=True)[0]
    waveform_gradient = torch.autograd.grad(
        v4_loss,
        candidate_result["output_pcm_float32"],
        retain_graph=True,
        allow_unused=True,
    )[0]
    if v4_parameter_gradient is None or waveform_gradient is None:
        raise RuntimeError("frozen V4 objective did not preserve gradients through V3 parameters and waveform")
    grad_np = np.ascontiguousarray(v4_parameter_gradient.detach().cpu().numpy(), dtype=np.float32)
    waveform_grad_np = np.ascontiguousarray(waveform_gradient.detach().cpu().numpy(), dtype=np.float32)
    v4_grad_finite = bool(np.isfinite(grad_np).all() and np.isfinite(waveform_grad_np).all())
    if not v4_grad_finite:
        raise FloatingPointError("non-finite V4 objective gradient through V3 head")

    # Keep the frozen SyncNet-derived reward separate from waveform and mask
    # regularizers. A finite full-loss gradient can exist even if the score path
    # is disconnected, so both pure-reward derivatives have their own gate.
    pure_reward = objective["metrics"]["reward"]
    reward_parameter_gradient = torch.autograd.grad(
        pure_reward,
        parameter,
        retain_graph=True,
        allow_unused=True,
    )[0]
    reward_waveform_gradient = torch.autograd.grad(
        pure_reward,
        candidate_result["output_pcm_float32"],
        retain_graph=True,
        allow_unused=True,
    )[0]
    if reward_parameter_gradient is None or reward_waveform_gradient is None:
        raise RuntimeError("pure frozen V4 reward did not preserve gradients through V3 parameters and waveform")
    reward_grad_np = np.ascontiguousarray(reward_parameter_gradient.detach().cpu().numpy(), dtype=np.float32)
    reward_waveform_grad_np = np.ascontiguousarray(
        reward_waveform_gradient.detach().cpu().numpy(), dtype=np.float32
    )
    pure_reward_grad_finite = bool(
        np.isfinite(reward_grad_np).all() and np.isfinite(reward_waveform_grad_np).all()
    )
    pure_reward_parameter_l2 = float(np.linalg.norm(reward_grad_np.astype(np.float64)))
    pure_reward_waveform_l2 = float(np.linalg.norm(reward_waveform_grad_np.astype(np.float64)))
    pure_reward_grad_nonzero = bool(pure_reward_parameter_l2 > 0.0 and pure_reward_waveform_l2 > 0.0)
    if not pure_reward_grad_finite:
        raise FloatingPointError("non-finite pure frozen V4 reward gradient")
    if not pure_reward_grad_nonzero:
        raise RuntimeError("pure frozen V4 reward gradient must have nonzero norm for both parameters and waveform")

    candidate_curve_np = np.ascontiguousarray(candidate_curve.detach().cpu().numpy(), dtype=np.float32)
    candidate_visual_np = np.ascontiguousarray(
        objective["score"]["visual_embeddings_float32"].detach().cpu().numpy(), dtype=np.float32
    )

    # Independently exercise the constructor's smooth waveform-only objective.
    # This FD never evaluates W2L/SyncNet and is not a derivative check for the
    # native renderer's hard STE path.
    linear_weight, quadratic_weight = _make_smooth_probe_weights(probe, int(source.numel()), torch)
    probe_loss_scale = float(probe["objective"]["loss_scale"])
    smooth_loss = _smooth_waveform_only_loss(
        head, source, baseline, linear_weight, quadratic_weight, probe_loss_scale, torch
    )
    smooth_gradient = torch.autograd.grad(smooth_loss, parameter, retain_graph=False, allow_unused=False)[0]
    smooth_grad_np = smooth_gradient.detach().cpu().numpy().astype(np.float64)
    fd_cfg = probe["finite_difference"]
    step_db = float(fd_cfg["coordinate_step_db"])
    abs_tol = float(fd_cfg["absolute_tolerance"])
    rel_tol = float(fd_cfg["relative_tolerance"])
    base_theta = theta_np.astype(np.float32, copy=True)
    coordinate_rows = []

    def smooth_scalar_for_theta(values: Any) -> float:
        with torch.no_grad():
            head.band_contrasts_db.copy_(torch.as_tensor(values, dtype=torch.float32, device=device))
            value = _smooth_waveform_only_loss(
                head, source, baseline, linear_weight, quadratic_weight, probe_loss_scale, torch
            )
        return float(value.detach().item())

    for flat_index in range(9):
        plus = base_theta.reshape(-1).copy()
        minus = base_theta.reshape(-1).copy()
        plus[flat_index] += step_db
        minus[flat_index] -= step_db
        fd_value = (
            smooth_scalar_for_theta(plus.reshape(3, 3))
            - smooth_scalar_for_theta(minus.reshape(3, 3))
        ) / (2.0 * step_db)
        auto_value = float(smooth_grad_np.reshape(-1)[flat_index])
        error = abs(fd_value - auto_value)
        allowed = abs_tol + rel_tol * max(abs(fd_value), abs(auto_value))
        coordinate_rows.append({
            "parameter_flat_index": flat_index,
            "autograd_smooth_waveform_only": auto_value,
            "central_fd_smooth_waveform_only": fd_value,
            "absolute_error": error,
            "allowed_error": allowed,
            "agreement": bool(error <= allowed),
            "scope": "smooth waveform-only; not V4 hard-STE score differentiation",
        })
    direction = np.asarray(fd_cfg["direction_matrix"], dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction.shape != (3, 3) or not math.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError("sealed smooth waveform FD direction is invalid")
    direction /= direction_norm
    directional_step = float(fd_cfg["directional_step_db"])
    directional_plus = base_theta.astype(np.float64) + directional_step * direction
    directional_minus = base_theta.astype(np.float64) - directional_step * direction
    directional_fd = (
        smooth_scalar_for_theta(directional_plus.astype(np.float32))
        - smooth_scalar_for_theta(directional_minus.astype(np.float32))
    ) / (2.0 * directional_step)
    directional_auto = float(np.sum(smooth_grad_np * direction))
    directional_error = abs(directional_fd - directional_auto)
    directional_allowed = abs_tol + rel_tol * max(abs(directional_fd), abs(directional_auto))
    with torch.no_grad():
        head.band_contrasts_db.copy_(torch.from_numpy(theta_np).to(device=device))

    # Nine forward directional derivatives at theta=0, without SyncNet calls.
    # Each JVP returns one length-N column; this avoids constructing the output
    # Jacobian by N reverse passes. The N x 9 matrix is transient only.
    def zero_output_for_theta(theta: Any) -> Any:
        return torch.func.functional_call(
            head,
            {"band_contrasts_db": theta},
            (source, baseline),
        )["output_pcm_float32"]

    zero_parameter = torch.zeros((3, 3), dtype=torch.float32, device=device, requires_grad=True)
    jvp_columns = []
    for flat_index in range(9):
        tangent = torch.zeros_like(zero_parameter)
        tangent.reshape(-1)[flat_index] = 1.0
        _, jvp_column = torch.autograd.functional.jvp(
            zero_output_for_theta,
            zero_parameter,
            tangent,
            create_graph=False,
            strict=False,
        )
        if tuple(jvp_column.shape) != tuple(source.shape):
            raise RuntimeError("zero-parameter waveform JVP returned an unexpected shape")
        jvp_columns.append(jvp_column.reshape(-1).detach())
    jacobian_matrix = torch.stack(jvp_columns, dim=1).to(torch.float64)
    gram = jacobian_matrix.transpose(0, 1) @ jacobian_matrix
    singular_values = torch.linalg.svdvals(jacobian_matrix)
    singular_np = singular_values.detach().cpu().numpy().astype(np.float64)
    gram_np = _numpy_array(gram, np).astype(np.float64, copy=False)
    gram_eigenvalues_np = np.linalg.eigvalsh(gram_np)
    relative_rank_tolerance = float(config["jacobian"]["relative_rank_tolerance"])
    absolute_rank_tolerance = float(config["jacobian"]["absolute_rank_tolerance"])
    effective_rank_threshold = max(
        absolute_rank_tolerance,
        float(singular_np.max()) * relative_rank_tolerance if singular_np.size else 0.0,
    )
    rank = int(np.sum(singular_np > effective_rank_threshold))
    jacobian_full_effective_rank = rank == 9
    condition = (
        float(singular_np[0] / singular_np[rank - 1])
        if rank > 0 and singular_np[rank - 1] > 0.0
        else None
    )
    if not np.isfinite(gram_np).all() or not np.isfinite(gram_eigenvalues_np).all() or not np.isfinite(singular_np).all():
        raise FloatingPointError("zero-parameter waveform-Jacobian Gram/SVD is non-finite")
    del jvp_columns, jacobian_matrix

    candidate_metrics = {
        key: _as_float(value, torch)
        for key, value in objective["metrics"].items()
    }
    candidate_mask = _numpy_array(candidate_result["projected_log_magnitude_mask_db"], np).astype(np.float32, copy=False)
    raw_mask = _numpy_array(candidate_result["raw_log_magnitude_mask_db"], np).astype(np.float32, copy=False)
    positive_flux = _numpy_array(candidate_result["positive_relative_flux"], np).astype(np.float32, copy=False)
    negative_flux = _numpy_array(candidate_result["negative_relative_flux"], np).astype(np.float32, copy=False)
    valid_transition = _numpy_array(candidate_result["valid_transition_mask"], np).astype(np.uint8, copy=False)
    positive_gate = _numpy_array(candidate_result["positive_flux_gate"], np).astype(np.float32, copy=False)
    negative_gate = _numpy_array(candidate_result["negative_flux_gate"], np).astype(np.float32, copy=False)
    frame_power = candidate_result["physical_frame_power"]
    arrays = {
        "rms_curve_gpu_f32": baseline_curve_np,
        "rms_curve_cpu_direct_f32": cpu_rms_curve_np,
        "zero_v3_curve_cpu_f32": zero_curve_np,
        "candidate_curve_cpu_f32": candidate_curve_np,
        "rms_visual_gpu_f32": baseline_visual_np,
        "rms_visual_cpu_direct_f32": cpu_rms_visual_np,
        "zero_v3_visual_cpu_f32": zero_visual_np,
        "candidate_visual_cpu_f32": candidate_visual_np,
        "v4_loss_gradient_band_contrasts_f32": grad_np,
        "v4_loss_waveform_gradient_f32": waveform_grad_np,
        "v4_pure_reward_gradient_band_contrasts_f32": reward_grad_np,
        "v4_pure_reward_waveform_gradient_f32": reward_waveform_grad_np,
        "smooth_waveform_loss_gradient_f32": smooth_grad_np.astype(np.float32),
        "zero_waveform_jacobian_gram_f64": gram_np,
        "zero_waveform_jacobian_eigenvalues_f64": gram_eigenvalues_np,
        "zero_waveform_jacobian_singular_values_f64": singular_np,
        "raw_log_magnitude_mask_db_f32": raw_mask,
        "projected_log_magnitude_mask_db_f32": candidate_mask,
        "positive_relative_flux_f32": positive_flux,
        "negative_relative_flux_f32": negative_flux,
        "valid_transition_mask_u8": valid_transition,
        "positive_flux_gate_f32": positive_gate,
        "negative_flux_gate_f32": negative_gate,
        "source_frame_power_f32": _numpy_array(frame_power["source_original_pcm"], np).astype(np.float32, copy=False),
        "candidate_pre_rms_frame_power_f32": _numpy_array(frame_power["candidate_before_rms"], np).astype(np.float32, copy=False),
        "rms_baseline_frame_power_f32": _numpy_array(frame_power["canonical_rms_baseline"], np).astype(np.float32, copy=False),
        "candidate_output_frame_power_f32": _numpy_array(frame_power["output_pcm"], np).astype(np.float32, copy=False),
        "source_clock_frame_centers_i64": _numpy_array(candidate_result["video_frame_centers_source_samples"], np).astype(np.int64, copy=False),
        "stft_frame_centers_i64": _numpy_array(candidate_result["stft_frame_centers_source_samples"], np).astype(np.int64, copy=False),
    }
    arrays_meta = {
        key: {
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
            "array_sha256": _array_sha256(value, np),
        }
        for key, value in arrays.items()
    }
    npz_path = output_dir / "v3_v4_native_cpu_diagnostics.npz"
    with npz_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    npz_sha = _sha256_file(npz_path)
    npz_bytes = npz_path.stat().st_size
    max_output = int(config["runtime"]["max_output_bytes"])
    if npz_bytes > max_output:
        raise RuntimeError("compact diagnostic output exceeded its declared byte bound")

    zero_metrics = {
        key: _as_float(value, torch)
        for key, value in v4_runtime._curve_metrics(torch, torch.from_numpy(zero_curve_np)).items()
    }
    smooth_fd_all = all(item["agreement"] for item in coordinate_rows) and bool(directional_error <= directional_allowed)
    same_cpu_parity_all = bool(zero_audio_equal and same_cpu_curve_parity and same_cpu_visual_parity)
    parity_all = bool(zero_audio_equal and same_cpu_parity_all and curve_parity and visual_parity)
    elapsed_seconds = time.monotonic() - start_time
    rss_high_water_bytes = _rss_high_water_bytes()
    runtime_budget_ok = (
        elapsed_seconds <= float(config["runtime"]["wall_budget_seconds"])
        and rss_high_water_bytes is not None
        and rss_high_water_bytes <= int(config["runtime"]["rss_budget_bytes"])
    )
    terminal_status = (
        "COMPLETE_PASS"
        if parity_all and v4_grad_finite and pure_reward_grad_finite and pure_reward_grad_nonzero and smooth_fd_all and jacobian_full_effective_rank and runtime_budget_ok
        else "COMPLETE_WITH_CONTROL_FAILURE"
    )
    summary = {
        "schema": SCHEMA,
        "status": terminal_status,
        "scope": "one historical exposed fit utterance; no dev/confirmation; CPU only; no optimizer; no waveform persisted",
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "device": "cpu",
            "torch_num_threads": int(torch.get_num_threads()),
            "cuda_initialized_after_run": _cpu_cuda_state(torch),
            "frame_batch_size": 1,
            "syncnet_batch_size": 1,
            "elapsed_seconds": elapsed_seconds,
            "rss_high_water_bytes": rss_high_water_bytes,
            "resource_budget_within_limits": runtime_budget_ok,
            "declared_wall_budget_seconds": int(config["runtime"]["wall_budget_seconds"]),
            "declared_rss_budget_bytes": int(config["runtime"]["rss_budget_bytes"]),
        },
        "source_pins": {
            "worker": {"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size, "sha256": _sha256_file(Path(__file__).resolve())},
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size, "sha256": config_raw_sha},
            "repo_source_closure": source_checked,
            "plan": {"path": str(plan_path), "bytes": plan_path.stat().st_size, "sha256": config["inputs"]["v4_plan"]["sha256"]},
            "v3_cpu_config": {"path": str(cpu_cfg_path), "bytes": cpu_cfg_path.stat().st_size, "sha256": config["inputs"]["v3_cpu_qualification_config"]["sha256"]},
            "source_wav": {"path": str(source_audio_path), "bytes": source_audio_path.stat().st_size, "sha256": config["inputs"]["source_wav"]["sha256"], "pcm_float32_sha256": audio_info.pcm_float32_sha256},
            "context": context_file_pins,
            "audio_embedding_cache": {"npz_path": str(audio_cache_npz), "npz_sha256": config["inputs"]["audio_embedding_cache_npz"]["sha256"], "metadata_path": str(audio_cache_meta_path), "metadata_sha256": config["inputs"]["audio_embedding_cache_metadata"]["sha256"]},
            "v4_baseline_cache": {"npz_path": str(baseline_npz), "npz_sha256": config["inputs"]["v4_rms_cache_npz"]["sha256"], "metadata_path": str(baseline_meta_path), "metadata_sha256": config["inputs"]["v4_rms_cache_metadata"]["sha256"]},
            "geometry_manifest_sha256": config["inputs"]["geometry_manifest"]["sha256"],
            "loaded_model_source_pins": model_source_pins,
            "lower_level_runtime_checks": {
                "librosa_expected_version": expected_librosa_version,
                "librosa_actual_module_version": actual_librosa_version,
                "librosa_version_match": actual_librosa_version == expected_librosa_version,
                "python_speech_features_module_path": str(psf_module_path),
                "python_speech_features_package_root": str(psf_package_root),
                "python_speech_features_expected_tree_sha256": expected_psf_tree_sha,
                "python_speech_features_actual_tree_sha256": actual_psf_tree_sha,
                "python_speech_features_tree_match": actual_psf_tree_sha == expected_psf_tree_sha,
            },
            "effective_plan_payload_sha256": plan.get("plan_payload_sha256"),
            "parent_v2_plan_raw_sha256": parent_ref.get("raw_sha256"),
            "geometry_record": next(record for record in geometry_records if record["row_id"] == row_id),
        },
        "row": {
            "row_id": row_id,
            "sample_id": config["inputs"]["sample_id"],
            "speaker_id": config["inputs"]["speaker_id"],
            "sample_count": int(source.numel()),
            "query_start": query_start,
            "query_count": 27,
            "lag_frames": list(range(-15, 16)),
            "source_query_clock_unchanged": True,
        },
        "zero_v3_vs_canonical_rms": {
            "waveform_byte_equal": zero_audio_equal,
            "waveform_max_abs": zero_audio_max_abs,
            "same_cpu_direct_rms_vs_zero_v3_curve_max_abs": same_cpu_curve_max_abs,
            "same_cpu_direct_rms_vs_zero_v3_curve_within_tolerance": same_cpu_curve_parity,
            "same_cpu_direct_rms_vs_zero_v3_visual_max_abs": same_cpu_visual_max_abs,
            "same_cpu_direct_rms_vs_zero_v3_visual_within_tolerance": same_cpu_visual_parity,
            "same_cpu_parity_all_27_queries_and_31_lags": same_cpu_parity_all,
            "same_cpu_device": "cpu",
            "same_cpu_frame_batch_size": int(config["runtime"]["frame_batch_size"]),
            "same_cpu_syncnet_batch_size": int(config["runtime"]["syncnet_batch_size"]),
            "curve_max_abs_vs_sealed_v4_rms_cache": curve_max_abs,
            "curve_tolerance": float(config["parity_tolerances"]["curve_max_abs"]),
            "curve_all_31_lags_within_tolerance": curve_parity,
            "visual_embedding_max_abs_vs_sealed_v4_rms_cache": visual_max_abs,
            "visual_embedding_tolerance": float(config["parity_tolerances"]["visual_embedding_max_abs"]),
            "all_27_visual_queries_within_tolerance": visual_parity,
            "zero_curve_metrics": zero_metrics,
        },
        "nonzero_v3_probe": {
            "theta_source": "exact diagnostic_parameter_matrix_db from pinned V3 engineering qualification config; no fit/selection",
            "theta_db": theta_np.tolist(),
            "objective": "frozen V4 Cbest + 0.25 far-lag contrast with one-sided D0/Canchor guards and existing norm/waveform/TV regularizers",
            "loss_metrics": candidate_metrics,
            "v4_loss": _as_float(v4_loss, torch),
            "nine_parameter_v4_gradient_finite": v4_grad_finite,
            "nine_parameter_v4_gradient_nonzero_count": int(np.count_nonzero(grad_np)),
            "nine_parameter_v4_gradient_l2": float(np.linalg.norm(grad_np.astype(np.float64))),
            "candidate_waveform_gradient_finite": bool(np.isfinite(waveform_grad_np).all()),
            "candidate_waveform_gradient_l2": float(np.linalg.norm(waveform_grad_np.astype(np.float64))),
            "pure_reward_scope": "V4 SyncNet-derived reward only; excludes waveform, visual-norm, and mask-TV regularizers",
            "pure_reward_value": _as_float(pure_reward, torch),
            "pure_reward_parameter_gradient_finite": pure_reward_grad_finite,
            "pure_reward_parameter_gradient_nonzero_count_descriptive": int(np.count_nonzero(reward_grad_np)),
            "pure_reward_parameter_gradient_l2": pure_reward_parameter_l2,
            "pure_reward_waveform_gradient_finite": bool(np.isfinite(reward_waveform_grad_np).all()),
            "pure_reward_waveform_gradient_nonzero_count_descriptive": int(np.count_nonzero(reward_waveform_grad_np)),
            "pure_reward_waveform_gradient_l2": pure_reward_waveform_l2,
            "pure_reward_gradient_nonzero_norm_gate": pure_reward_grad_nonzero,
            "maximum_absolute_projected_mask_db": float(np.max(np.abs(candidate_mask.astype(np.float64)))),
            "candidate_output_sample_count": int(candidate_result["output_pcm_float32"].numel()),
            "candidate_output_rms": _as_float(candidate_result["normalization"]["final_output_rms"], torch),
            "candidate_output_peak": _as_float(candidate_result["normalization"]["final_output_peak"], torch),
        },
        "smooth_waveform_only_finite_difference": {
            "scope": "head waveform objective only; explicitly not a finite-difference validation of V4 native score or hard STE",
            "coordinate_step_db": step_db,
            "absolute_tolerance": abs_tol,
            "relative_tolerance": rel_tol,
            "coordinate_rows": coordinate_rows,
            "all_nine_coordinates_agree": smooth_fd_all,
            "directional_step_db": directional_step,
            "directional_autograd": directional_auto,
            "directional_finite_difference": directional_fd,
            "directional_absolute_error": directional_error,
            "directional_allowed_error": directional_allowed,
            "directional_agrees": bool(directional_error <= directional_allowed),
        },
        "zero_parameter_waveform_jacobian": {
            "definition": "d canonical-anchor-normalized output PCM / d nine V3 band_contrasts_db entries at theta=0; nine forward JVP columns, not output-sized reverse Jacobian passes",
            "column_method": "torch.autograd.functional.jvp once per canonical basis direction",
            "singular_values": singular_np.tolist(),
            "absolute_rank_tolerance": absolute_rank_tolerance,
            "relative_rank_tolerance": relative_rank_tolerance,
            "effective_rank_threshold": effective_rank_threshold,
            "absolute_threshold_rank": int(np.sum(singular_np > absolute_rank_tolerance)),
            "combined_threshold_rank": rank,
            "all_nine_effective_directions_gate": jacobian_full_effective_rank,
            "gram_eigenvalues": gram_eigenvalues_np.tolist(),
            "condition_number_over_retained_rank": condition,
            "gram_shape": [9, 9],
            "not_a_SyncNet_or_candidate_efficiency_metric": True,
        },
        "diagnostic_arrays": {
            "path": str(npz_path),
            "bytes": npz_bytes,
            "sha256": npz_sha,
            "arrays": arrays_meta,
            "saved_input_or_output_waveform": False,
        },
    }
    summary_path = output_dir / "qualification_summary.json"
    _atomic_json(summary_path, summary)
    receipt = {
        "schema": "cross_tfg_v3_v4_native_cpu_engineering_qualification_receipt_v1",
        "status": terminal_status,
        "summary": {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": _sha256_file(summary_path),
        },
        "diagnostic_arrays": {"path": str(npz_path), "bytes": npz_bytes, "sha256": npz_sha},
        "source_worker_sha256": _sha256_file(Path(__file__).resolve()),
        "config_sha256": config_raw_sha,
        "no_source_or_candidate_waveform_persisted": True,
    }
    _atomic_json(output_dir / "qualification_receipt.json", receipt)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started_at = time.monotonic()
    config_path = Path(args.config).resolve(strict=True)
    output_dir = Path(args.output_dir)
    try:
        result = _run(config_path, args.config_sha256, output_dir)
        print(json.dumps({
            "schema": SCHEMA,
            "status": result["status"],
            "row_id": result["row"]["row_id"],
            "zero_curve_max_abs": result["zero_v3_vs_canonical_rms"]["curve_max_abs_vs_sealed_v4_rms_cache"],
            "v4_gradient_finite": result["nonzero_v3_probe"]["nine_parameter_v4_gradient_finite"],
            "diagnostic_npz_sha256": result["diagnostic_arrays"]["sha256"],
            "pure_reward_parameter_gradient_l2": result["nonzero_v3_probe"]["pure_reward_parameter_gradient_l2"],
            "pure_reward_waveform_gradient_l2": result["nonzero_v3_probe"]["pure_reward_waveform_gradient_l2"],
            "zero_waveform_jacobian_full_rank_gate": result["zero_parameter_waveform_jacobian"]["all_nine_effective_directions_gate"],
        }, sort_keys=True, separators=(",", ":")))
        return 0 if result["status"] == "COMPLETE_PASS" else 2
    except Exception as exc:
        output_dir = output_dir.resolve()
        if output_dir.exists() and output_dir.is_dir() and not (output_dir / "qualification_summary.json").exists():
            failure = {
                "schema": SCHEMA,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started_at,
                "no_optimizer": True,
                "source_or_candidate_waveform_persisted": False,
            }
            try:
                _atomic_json(output_dir / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps({"schema": SCHEMA, "status": "FAILED", "error_type": type(exc).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
