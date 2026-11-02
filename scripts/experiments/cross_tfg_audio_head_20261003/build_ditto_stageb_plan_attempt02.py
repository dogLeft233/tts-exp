#!/usr/bin/env python3
"""Derive a fresh Ditto-only 240-row dev plan from the frozen shared roster.

The plan keeps the shared original audio, development clips, four predeclared
methods, and three render seeds, while requiring 240 new Ditto renders. It has
no confirmation rows or confirmation seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_PLAN_DEFAULT = Path("runs/cross_tfg_audio_head_20261003/joint_v3_analysis_attempt03/readback_attempt01/joint_package/plan.json")
PORTRAIT_REPORT_DEFAULT = Path(".codex/scratch/cross_tfg_model_deploy_attempt18/portrait_readback_attempt01/ditto_portrait_render_attempt03.json")
OUT_DEFAULT = Path("runs/cross_tfg_audio_head_20261003/ditto_stageb_attempt02/frozen_plan")
RUN_ID = "cross_tfg_audio_head_20261003_ditto_stageb_attempt02"
MODEL_ID = "ditto_hubert_fixed_portrait_attempt18"
AUDIO_FRONTEND = "ditto_hubert_onnx_cpu_torch_cuda_static_portrait_attempt18"
METHOD_IDS = ("identity", "rms_fixed", "rms_localcompress", "spectral_head_fit_v3_zero_lag_unitcos")
SEEDS = (20261003, 20261004, 20261005)
EXPECTED_SOURCE_PLAN_PAYLOAD = "950e43c736e79d9abab7cf2b28c9913c08027470296bee994b0858e716f5867b"
EXPECTED_PORTRAIT_REPORT_SHA256 = "866fc656d69caef7203bc2a3d3c61e12046353c4c3515722fd8ad73c9d16170c"
EXPECTED_RENDERER_SHA256 = "1415ee28206ba6afe70e32fb728b29fb3b49bb638bdc21d3c2a7e89c66131df0"
EXPECTED_FFMPEG_SHA256 = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
EXPECTED_SCORE_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
EXPECTED_IMPORT_REPORT_SHA256 = "2ba259045dd79c797f7c0e34403f600ab1d0accc40fa03ca2aa820c8c80f7806"
EXPECTED_RUNTIME_EXTENSION_SHA256 = "ce1524957cf0ef6d6f8065630c3caebb654485576867a2a91450779277451509"
EXPECTED_CONFIG_SHA256 = "8e4c9ecb8dbb5aa4d22cb3f9b2c6ceabc62499b04af5fe074826d5b1044c4146"
EXPECTED_FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
STAGEA_CORE_SHA256 = "30c894ea31541cdde76a5a915bf635a36838a1cdb11f761c334c8986e71a01bc"
SYNCNET_EVALUATOR_SHA256 = "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b"
PCM16_LINEAGE_SHA256 = "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0"
REMOTE_PROJECT = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003"
SOURCE_ROOT = REMOTE_PROJECT + "/model_sources_attempt02/ditto-talkinghead"
MODEL_ROOT = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03/models/ditto_pytorch"
CONFIG_PATH = MODEL_ROOT + "/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
WAV2LIP_PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
SYNCNET_ROOT = "/root/autodl-tmp/wav2lip-syncnet/syncnet_python"
SYNCNET_PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python"
SYNCNET_CHECKPOINT = SYNCNET_ROOT + "/data/syncnet_v2.model"
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
DITTO_MODEL_IDENTITY = "ditto"
IMPORT_REPORT_PATH = REMOTE_PROJECT + "/job_control/model_deploy_attempt13/ditto_streamsdk_import_probe_attempt12/ditto_streamsdk_import_probe_attempt12.json"
RUNTIME_EXTENSION_PATH = REMOTE_PROJECT + "/model_deploy_attempt13/code/import_probe_native_cache_attempt12/pyximport/lib.linux-x86_64-3.10/core/utils/blend/blend.cpython-310-x86_64-linux-gnu.so"
HEADER_REPORT_PATH = REMOTE_PROJECT + "/model_deploy_attempt13/code/python310_headers_attempt14/python310_headers_attempt14.json"
SETUPTOOLS_STAGE_PATH = REMOTE_PROJECT + "/model_deploy_attempt13/code/setuptools_overlay_attempt12/setuptools_stage_receipt.json"
QUALIFICATION_REPORT_PATH = REMOTE_PROJECT + "/job_control/model_deploy_attempt18/ditto_portrait_render_attempt03.json"
RENDERER_REMOTE = REMOTE_PROJECT + "/model_deploy_attempt18/code/render_silent_video_attempt07.py"
FACE_REMOTE = "/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png"
OVERLAYS = (
    REMOTE_PROJECT + "/model_deploy_attempt13/code/overlays/setuptools_overlay_attempt12",
    REMOTE_PROJECT + "/model_deploy_attempt13/code/overlays/cython_overlay_attempt12",
    REMOTE_PROJECT + "/model_deploy_attempt13/code/overlays/filetype_overlay",
    REMOTE_PROJECT + "/model_deploy_attempt10/code/future_overlay_attempt02",
    REMOTE_PROJECT + "/model_deploy_attempt09/code/overlay_python310_attempt02",
)
HEADERS_ROOT = REMOTE_PROJECT + "/model_deploy_attempt13/code/python310_headers_attempt14/overlay"
CFLAGS = " ".join((
    "-I" + HEADERS_ROOT + "/usr/include",
    "-I" + HEADERS_ROOT + "/usr/include/python3.10",
    "-I" + HEADERS_ROOT + "/usr/include/x86_64-linux-gnu/python3.10",
))
DITTO_RENDERER_PROFILE = "ditto_hubert_onnx_cpu_renderer_torch_cuda"
DITTO_STAGEA_PROFILE = "ditto_hubert_onnx_cpu_torch_cuda_fixed_portrait"
DITTO_SOURCE_COMMIT = "c3e47eee2e626500017a0556b470d6d4182f85e8"
DITTO_ADAPTER_SCHEMA = "cross_tfg_ditto_stagea_adapter_receipt_v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def contrast(contrast_id: str, method_a: str, method_b: str, role: str) -> dict[str, str]:
    return {"contrast_id": contrast_id, "method_a": method_a, "method_b": method_b, "role": role}


def build(source_path: Path, portrait_report_path: Path, output_dir: Path,
          shim_path: Path, combined_runner_path: Path) -> dict[str, Any]:
    source_raw = source_path.read_bytes()
    source = json.loads(source_raw.decode("utf-8"))
    if source.get("status") != "FROZEN" or source.get("plan_sha256") != sha_bytes(canonical({k: v for k, v in source.items() if k != "plan_sha256"})):
        raise ValueError("source joint plan is not correctly sealed")
    if source["plan_sha256"] != EXPECTED_SOURCE_PLAN_PAYLOAD or source.get("confirm_speaker_ids") or source.get("generation_seed_family", {}).get("confirm"):
        raise ValueError("source plan differs from the exact reviewed dev-only joint plan")

    portrait_raw = portrait_report_path.read_bytes()
    portrait = json.loads(portrait_raw.decode("utf-8"))
    if sha_bytes(portrait_raw) != EXPECTED_PORTRAIT_REPORT_SHA256:
        raise ValueError("actual Ditto18 portrait report raw SHA differs from the local readback seal")
    if portrait.get("status") != "PASS_ENGINEERING_ONLY" or portrait.get("profile") != DITTO_RENDERER_PROFILE:
        raise ValueError("Ditto portrait report did not pass the exact declared runtime profile")
    if portrait.get("source_and_weights", {}).get("source_commit") != DITTO_SOURCE_COMMIT:
        raise ValueError("Ditto report source commit differs from the source-pinned runtime")
    if portrait.get("source_and_weights", {}).get("weights", {}).get("ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", {}).get("sha256") != EXPECTED_CONFIG_SHA256:
        raise ValueError("Ditto model configuration hash differs from qualified portrait report")
    if portrait.get("runtime", {}).get("ditto_streamsdk_import_report", {}).get("raw_sha256") != EXPECTED_IMPORT_REPORT_SHA256:
        raise ValueError("Ditto full StreamSDK import report is not the qualified source")
    if portrait.get("runtime", {}).get("cython_cache", {}).get("blend_extension_sha256_after") != EXPECTED_RUNTIME_EXTENSION_SHA256:
        raise ValueError("Ditto native runtime extension does not match the qualified import")
    if portrait.get("audio", {}).get("file_sha256") is None or portrait.get("face", {}).get("file_sha256") != EXPECTED_FACE_SHA256:
        raise ValueError("Ditto portrait source face/audio binding is incomplete")
    if portrait.get("silent_video", {}).get("probe", {}).get("fps") != "25/1":
        raise ValueError("qualified portrait renderer did not report 25/1 fps")

    source_files = portrait["source_and_weights"]["source_files"]
    weight_files = portrait["source_and_weights"]["weights"]
    assets = {"source_files": source_files, "weights": weight_files}
    for relative, expected in {
        "environment.yaml": "8dbadc3c01eb1bd7629deeb74ff06ab2e0ad4660274e8333929ab60c12dcdc92",
        "inference.py": "a7df5b5a006913c8c216d8e442d669726a8c60255b5b6960a8de69bea54a60b3",
        "stream_pipeline_offline.py": "9ddd7546b44be71fbbcda22de2cd85ced00560d3564c5c4ee1f2ae71d840bb8d",
    }.items():
        if source_files.get(relative, {}).get("sha256") != expected:
            raise ValueError(f"Ditto source report differs at {relative}")
    if len(weight_files) != 12 or any(len(str(item.get("sha256", ""))) != 64 or int(item.get("bytes", 0)) <= 0 for item in weight_files.values()):
        raise ValueError("Ditto portrait report did not bind all 12 config/model assets")
    assets_sha = sha_bytes(canonical(assets))

    score_model_sha = source["models"][0]["score_model_sha256"]
    if score_model_sha != EXPECTED_SCORE_MODEL_SHA256:
        raise ValueError("frozen SyncNet scoring checkpoint differs from its qualified pin")
    source_rows = [dict(row) for row in source["rows"] if row["method_id"] in METHOD_IDS]
    if len(source_rows) != 240 or {int(row["seed"]) for row in source_rows} != set(SEEDS):
        raise ValueError("the frozen joint roster is not 20 clips x 3 seeds x the selected methods")
    if len({(row["sample_id"], row["seed"]) for row in source_rows}) != 60:
        raise ValueError("expected 60 paired clip/seed contexts")
    methods_by_id = {str(item["method_id"]): item for item in source["methods"]}
    if set(METHOD_IDS) - set(methods_by_id):
        raise ValueError("source plan does not contain all four declared methods")

    rows: list[dict[str, Any]] = []
    for original in source_rows:
        row = dict(original)
        row["model_id"] = MODEL_ID
        row["audio_frontend"] = AUDIO_FRONTEND
        row["model_role"] = "candidate"
        row["row_id"] = (f"dev_{row['speaker_id']}_{row['sample_id']}_{MODEL_ID}_"
                         f"{row['face_id']}_{row['seed']}_{row['method_id']}")
        if row.get("source_face", {}).get("file_sha256") != EXPECTED_FACE_SHA256:
            raise ValueError(f"row {row['row_id']} does not use the frozen static primary portrait")
        settings = dict(row.get("generation_settings", {}))
        settings.update({
            "face_mode": "static_image",
            "fps_num": 25, "fps_den": 1,
            "generator": "Ditto StreamSDK v0.4 HuBERT native inference; ONNX auxiliary sessions CPU, Torch renderer CUDA",
            "input_audio": "enhanced canonical 16k mono float32; condition only; Stage-A remux/score uses original natural PCM",
            "seed": int(row["seed"]),
        })
        settings.pop("pre_mel_peak_normalization", None)
        row["generation_settings"] = settings
        rows.append(row)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("derived Ditto plan has duplicate row IDs")

    code_pins = {
        "stagea_core_sha256": STAGEA_CORE_SHA256,
        "combined_frontend_runner_sha256": sha_file(combined_runner_path),
        "shim_sha256": sha_file(shim_path),
        "ditto_renderer_sha256": EXPECTED_RENDERER_SHA256,
        "ditto_import_report_sha256": EXPECTED_IMPORT_REPORT_SHA256,
        "runtime_extension_sha256": EXPECTED_RUNTIME_EXTENSION_SHA256,
        "model_assets_sha256": assets_sha,
        "source_commit": DITTO_SOURCE_COMMIT,
    }
    preprocessing = {
        "audio_frontend": AUDIO_FRONTEND,
        "source_commit": DITTO_SOURCE_COMMIT,
        "source_files": source_files,
        "model_assets": weight_files,
        "model_assets_sha256": assets_sha,
        "configuration_path": CONFIG_PATH,
        "configuration_sha256": EXPECTED_CONFIG_SHA256,
        "profile": DITTO_RENDERER_PROFILE,
        "fps_num": 25, "fps_den": 1,
        "source_clock": "25/1 frame timestamps start at zero and follow source_clock_frame_index/25; no time warp; keep every fresh renderer output frame",
        "face_mode": "fixed primary static portrait; model-native face preparation, no moving source video",
        "torch_profile": "pinned Torch 2.5.1+cu124 renderer on RTX 4080 SUPER",
        "onnxruntime_profile": "four auxiliary ONNX sessions explicitly use CPUExecutionProvider",
        "native_blend_extension_sha256": EXPECTED_RUNTIME_EXTENSION_SHA256,
        "full_import_only_report_sha256": EXPECTED_IMPORT_REPORT_SHA256,
        "native_portrait_report_raw_sha256": EXPECTED_PORTRAIT_REPORT_SHA256,
    }
    preprocessing_sha = sha_bytes(canonical(preprocessing))
    model_assets_sha = sha_bytes(canonical(assets))
    shim_remote = REMOTE_PROJECT + "/stageb_ditto_attempt02/code/ditto_five_arg_shim_attempt02.py"
    frontend_runner_remote = REMOTE_PROJECT + "/stageb_ditto_attempt02/code/run_cross_frontend_dev_attempt03.py"
    generator = {
        "contract": "cross_tfg_silent_video_cli_v1",
        "adapter_path": shim_remote,
        "adapter_sha256": code_pins["shim_sha256"],
        "receipt_required": True,
        "receipt_suffix": ".adapter_receipt.json",
        "receipt_schema": DITTO_ADAPTER_SCHEMA,
        "runner_sha256": STAGEA_CORE_SHA256,
        "child_script_sha256": code_pins["shim_sha256"],
        "renderer_path": RENDERER_REMOTE,
        "renderer_sha256": EXPECTED_RENDERER_SHA256,
        "python": WAV2LIP_PYTHON,
        "cwd": SOURCE_ROOT,
        "fps_num": 25, "fps_den": 1,
        "cuda_visible_devices": "0",
        "timeout_seconds": 1800,
        "env": {
            "PYTHONPATH": ":".join((*OVERLAYS, SOURCE_ROOT)),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CFLAGS": CFLAGS,
        },
        "profile": DITTO_STAGEA_PROFILE,
        "ditto_source_root": SOURCE_ROOT,
        "ditto_weights_root": MODEL_ROOT,
        "ditto_config_path": CONFIG_PATH,
        "model_assets_sha256": model_assets_sha,
        "qualification_report_path": QUALIFICATION_REPORT_PATH,
        "qualification_report_sha256": EXPECTED_PORTRAIT_REPORT_SHA256,
        "import_report_path": IMPORT_REPORT_PATH,
        "import_report_sha256": EXPECTED_IMPORT_REPORT_SHA256,
        "runtime_extension_path": RUNTIME_EXTENSION_PATH,
        "runtime_extension_sha256": EXPECTED_RUNTIME_EXTENSION_SHA256,
        "header_report_path": REMOTE_PROJECT + "/model_deploy_attempt13/code/python310_headers_attempt14/python310_headers_attempt14.json",
        "header_report_sha256": "b997b9e98ed61be480788442237bf1008f4a5cef24e74dd55bdb71ea55b1f0cd",
        "header_tree_sha256": "e9dcde923150324ed29f3e8104d18f9ca61f61b7b32b07a6a1874e47c15a2a6e",
        "setuptools_stage_sha256": "2ecaf424eb3a2ccdacb903fab9ac80134d21417ed8d27e6d322d1be51e1d7746",
        "model_overlay_roots": list(OVERLAYS),
    }

    contrasts = [
        contrast("rms_fixed_minus_identity", "rms_fixed", "identity", "descriptive_control"),
        contrast("rms_localcompress_minus_identity", "rms_localcompress", "identity", "descriptive_control"),
        contrast("rms_localcompress_minus_rms_fixed", "rms_localcompress", "rms_fixed", "descriptive_control"),
        contrast("spectral_head_fit_v3_zero_lag_unitcos_minus_identity", "spectral_head_fit_v3_zero_lag_unitcos", "identity", "main"),
        contrast("spectral_head_fit_v3_zero_lag_unitcos_minus_rms_fixed", "spectral_head_fit_v3_zero_lag_unitcos", "rms_fixed", "main"),
        contrast("spectral_head_fit_v3_zero_lag_unitcos_minus_rms_localcompress", "spectral_head_fit_v3_zero_lag_unitcos", "rms_localcompress", "diagnostic"),
    ]
    endpoints = [
        {"endpoint_id": f"dev_{item['contrast_id']}_{support}_Cbest", "split": "dev",
         "model_scope": "model", "model_id": MODEL_ID, "contrast_id": item["contrast_id"],
         "support_id": support, "metric": "Cbest", "audio_estimand": "native_roi_original_pcm16",
         "roi_estimand": "native_official", "exploratory": True}
        for item in contrasts for support in ("full", "guard20")
    ]
    first = next(row for row in rows if row["method_id"] == "identity" and row["seed"] == SEEDS[0])
    repeat_config_sha = sha_bytes(canonical({
        "model_id": MODEL_ID, "shim_sha256": code_pins["shim_sha256"],
        "renderer_sha256": EXPECTED_RENDERER_SHA256, "method_id": "identity",
        "sample_id": first["sample_id"], "face_id": first["face_id"], "seed": SEEDS[0],
        "dev_seed_ids": list(SEEDS),
        "comparison_fields": ["decoded_rgb_frames", "timestamp_map", "query_map", "distance_matrix", "Cbest"],
    }))
    repeat_qc = [{
        "split": "dev", "model_id": MODEL_ID, "input_sample_id": first["sample_id"],
        "face_id": first["face_id"], "seed": SEEDS[0], "method_id": "identity",
        "reference_row_id": first["row_id"], "deterministic": False,
        "followup_protocol": {"same_seed_repeat": True, "dev_seed_ids": list(SEEDS),
                              "config_sha256": repeat_config_sha,
                              "comparison_fields": ["decoded_rgb_frames", "timestamp_map", "query_map", "distance_matrix", "Cbest"]},
    }]

    source_raw_sha = sha_bytes(source_raw)
    source_payload = source["plan_sha256"]
    plan: dict[str, Any] = {
        "schema_version": source["schema_version"],
        "run_id": RUN_ID,
        "status": "FROZEN",
        "dataset": source["dataset"],
        "dataset_revision": source["dataset_revision"],
        "split_manifest_sha256": source["split_manifest_sha256"],
        "fit_speaker_ids": source["fit_speaker_ids"],
        "dev_speaker_ids": source["dev_speaker_ids"],
        "confirm_speaker_ids": [],
        "historical_exposed_speaker_ids": source["historical_exposed_speaker_ids"],
        "historical_exposed_sample_ids": source["historical_exposed_sample_ids"],
        "historical_exposure_source": source["historical_exposure_source"],
        "generation_seed_family": {"fit": [], "dev": list(SEEDS), "confirm": []},
        "methods": [methods_by_id[method_id] for method_id in METHOD_IDS],
        "models": [{
            "model_id": MODEL_ID,
            "audio_frontend": AUDIO_FRONTEND,
            "role": "candidate",
            "git_head": DITTO_SOURCE_COMMIT,
            "model_sha256": model_assets_sha,
            "checkpoint_sha256": model_assets_sha,
            "preprocessing_sha256": preprocessing_sha,
            "generator_audio_preprocessing_sha256": source_files["inference.py"]["sha256"],
            "score_model_sha256": score_model_sha,
            "fps_num": 25, "fps_den": 1,
            "code_sha256": sha_bytes(canonical(code_pins)),
            "source_sha256": source_files,
            "generator": generator,
            "model_assets": assets,
            "source_commit": DITTO_SOURCE_COMMIT,
            "profile": preprocessing,
            "qualification_report": {"path": QUALIFICATION_REPORT_PATH, "raw_sha256": EXPECTED_PORTRAIT_REPORT_SHA256},
        }],
        "supports": source["supports"],
        "protocol": {
            **source["protocol"],
            "generator": {
                "model_id": MODEL_ID, "audio_frontend": AUDIO_FRONTEND,
                "face_id": "primary_face_static_portrait_v1",
                "face_input": "same pinned primary static portrait for every dev row/method/seed",
                "renderer_settings": {"fps_num": 25, "fps_den": 1, "static_image": True,
                                      "onnx_providers": ["CPUExecutionProvider"], "torch_cuda": True},
                "source_commit": DITTO_SOURCE_COMMIT,
                "source_sha256": source_files,
                "model_assets_sha256": model_assets_sha,
                "renderer_sha256": EXPECTED_RENDERER_SHA256,
                "adapter_sha256": code_pins["shim_sha256"],
                "stagea_core_sha256": STAGEA_CORE_SHA256,
                "combined_frontend_runner_sha256": code_pins["combined_frontend_runner_sha256"],
                "generator_profile": preprocessing,
                "source_video_role": "shared primary static portrait; no model-side time warp or generated-source reuse",
            },
        },
        "analysis": {
            "exploratory_dev_only": True,
            "aggregation": {"seed_within_sample": "equal_weight", "samples_within_speaker": "equal_weight", "speakers": "equal_weight"},
            "bootstrap": {"unit": "speaker_equal_weight", "bit_generator": "PCG64", "seed": 2026100301,
                          "draw_count": 10000, "confidence_level": 0.95},
            "contrasts": contrasts,
            "summary_endpoints": endpoints,
            "simultaneous_family_ids": [], "simultaneous_family_status": "NOT_APPLICABLE",
            "cross_frontend_reference": {"wav2lip_run_id": source["run_id"],
                                          "wav2lip_plan_raw_sha256": source_raw_sha,
                                          "same_dev_inputs_and_methods": True,
                                          "fresh_ditto_render_rows": 240,
                                          "source_video_results_reused": False},
        },
        "contrasts": contrasts,
        "generation_repeat_qc": repeat_qc,
        "source_rows_manifest": source["source_rows_manifest"],
        "cross_plan_lineage": source.get("cross_plan_lineage", {}),
        "execution": {
            "stage": "fresh_cross_frontend_stagea_dev",
            "scope": "fresh Ditto fixed-primary-portrait development run, four shared waveform methods, 20 dev utterances x 3 seeds, no confirmation rows/audio/effects",
            "planned_rows": 240, "generation_repeat_qc_rows": len(repeat_qc), "fresh_model_rows": 240,
            "source_joint_plan_raw_sha256": source_raw_sha,
            "source_joint_plan_payload_sha256": source_payload,
            "combined_runner_sha256": code_pins["combined_frontend_runner_sha256"],
            "stagea_core_sha256": STAGEA_CORE_SHA256,
            "adapter_sha256": code_pins["shim_sha256"],
            "renderer_sha256": EXPECTED_RENDERER_SHA256,
            "score_model_sha256": score_model_sha,
            "primary_audio_lineage_code_sha256": PCM16_LINEAGE_SHA256,
            "official_syncnet_evaluator_wrapper_sha256": SYNCNET_EVALUATOR_SHA256,
            "reuse_policy": "reuse frozen source audio, method configs, and fit-v3 checkpoint; every Ditto render, native tracking, and score is fresh",
            "source_model_profile_status": "PASS_ENGINEERING_ONLY single static-primary-portrait render; HuBERT ONNX auxiliary sessions CPU and Torch renderer CUDA",
            "qualification_report_raw_sha256": EXPECTED_PORTRAIT_REPORT_SHA256,
        },
        "config_sha256": sha_bytes(canonical({
            "source_joint_plan_payload_sha256": source_payload,
            "model_id": MODEL_ID, "audio_frontend": AUDIO_FRONTEND,
            "methods": list(METHOD_IDS), "generation_seed_family": list(SEEDS),
            "model_source_commit": DITTO_SOURCE_COMMIT,
            "model_assets_sha256": model_assets_sha,
            "source_code_pins": code_pins,
            "preprocessing_sha256": preprocessing_sha,
        })),
        "git_revision": "a66c915023ad72493f2d79e415ab849b6ec1d675",
        "source_reference_plan_raw_sha256": source_raw_sha,
        "source_reference_plan_payload_sha256": source_payload,
        "rows": rows,
    }
    plan["plan_sha256"] = sha_bytes(canonical(plan))
    raw_plan = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "plan.json").write_bytes(raw_plan)
    (output_dir / "planned_rows.jsonl").write_bytes(rows_raw)
    packet = {
        "schema": "cross_tfg_ditto_stageb_packet_v1", "status": "FROZEN_LOCAL_PENDING_ROOT_REVIEW",
        "run_id": RUN_ID, "source_plan_path": str(source_path.resolve()),
        "source_plan_raw_sha256": source_raw_sha, "source_plan_payload_sha256": source_payload,
        "portrait_report_path": str(portrait_report_path.resolve()),
        "portrait_report_raw_sha256": EXPECTED_PORTRAIT_REPORT_SHA256,
        "plan_path": str((output_dir / "plan.json").resolve()),
        "plan_raw_sha256": sha_bytes(raw_plan), "plan_payload_sha256": plan["plan_sha256"],
        "planned_rows_path": str((output_dir / "planned_rows.jsonl").resolve()),
        "planned_rows_raw_sha256": sha_bytes(rows_raw), "planned_rows": len(rows),
        "rows_by_method": {method_id: sum(row["method_id"] == method_id for row in rows) for method_id in METHOD_IDS},
        "rows_by_seed": {str(seed): sum(int(row["seed"]) == seed for row in rows) for seed in SEEDS},
        "unique_samples": len({row["sample_id"] for row in rows}),
        "unique_speakers": len({row["speaker_id"] for row in rows}),
        "confirm_rows": 0, "fresh_video_rows": len(rows), "generation_repeat_qc_rows": len(repeat_qc),
        "shim_path": str(shim_path.resolve()), "shim_sha256": code_pins["shim_sha256"],
        "combined_frontend_runner_path": str(combined_runner_path.resolve()),
        "combined_frontend_runner_sha256": code_pins["combined_frontend_runner_sha256"],
        "source_commit": DITTO_SOURCE_COMMIT, "model_assets_sha256": assets_sha,
        "preprocessing_sha256": preprocessing_sha,
        "warning": "portrait render qualifies only the declared fixed-portrait runtime; this dev plan is exploratory and has no confirmation data",
    }
    write_json(output_dir / "packet_manifest.json", packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, default=SOURCE_PLAN_DEFAULT)
    parser.add_argument("--portrait-report", type=Path, default=PORTRAIT_REPORT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--shim", type=Path, default=Path(__file__).with_name("ditto_five_arg_shim_attempt02.py"))
    parser.add_argument("--frontend-runner", type=Path, default=Path(__file__).with_name("run_cross_frontend_dev_attempt03.py"))
    args = parser.parse_args()
    packet = build(args.source_plan, args.portrait_report, args.output_dir, args.shim, args.frontend_runner)
    print(json.dumps(packet, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
