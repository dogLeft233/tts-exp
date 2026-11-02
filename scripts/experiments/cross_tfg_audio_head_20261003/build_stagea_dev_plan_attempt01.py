#!/usr/bin/env python3
"""Freeze the Wav2Lip-only Stage-A development plan before its score run.

The script reads only the independently audited twenty-row dev PCM manifest.
It never opens confirmation audio or score artifacts. Output is immutable: an
existing output directory is an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEV_MANIFEST_SHA256 = "a553c7dff561c724f5ae653161703b9da498b8ef71a7182425cd1640993c6ad8"
FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
W2L_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
W2L_HEAD = "bac9a81e63ecc153202353372e5724b83d9e6322"
SYNCNET_HEAD = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
SEEDS = [20261003, 20261004, 20261005]
RMS_TARGET = 0.0376838172675746
MODEL_ID = "wav2lip_gan_bac9a81e"
FACE_ID = "primary_face_static_portrait_v1"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sealed_hash(value: Any, field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return _sha_bytes(_canonical(body))


def _ref(path: str, file_sha: str, pcm_sha: str, samples: int) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": file_sha,
        "pcm_sha256": pcm_sha,
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_count": int(samples),
        "subtype": "IEEE_FLOAT_32_LE",
    }


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()


def build_plan(
    manifest_path: Path,
    protocol_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    remote_audio_root: str,
    remote_face_path: str,
    remote_portrait_video_path: str,
    portrait_video_sha256: str,
    ffmpeg_sha256: str,
    pcm16_lineage_code_sha256: str,
    generator_child_sha256: str,
    syncnet_evaluator_sha256: str,
    stagea_runner_sha256: str,
) -> dict[str, Any]:
    raw_manifest = manifest_path.read_bytes()
    manifest_sha = _sha_bytes(raw_manifest)
    if manifest_sha != DEV_MANIFEST_SHA256:
        raise ValueError(f"dev manifest raw SHA mismatch: {manifest_sha}")
    manifest = json.loads(raw_manifest.decode("utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("split") != "dev":
        raise ValueError("input must be the completed frozen dev PCM manifest")
    if manifest.get("confirm_audio_read") is not False or manifest.get("effects_read") is not False:
        raise ValueError("dev manifest does not certify the no-confirm/no-effects boundary")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("dev_speaker_ids") != manifest.get("dev_speaker_ids"):
        raise ValueError("dev roster differs between protocol and PCM manifest")
    if protocol.get("confirm_speaker_ids") is None or len(protocol["confirm_speaker_ids"]) != 40:
        raise ValueError("expected the separately frozen forty-speaker confirmation roster")
    for label, digest in (
        ("portrait reference video", portrait_video_sha256),
        ("FFmpeg", ffmpeg_sha256),
        ("PCM16 lineage source", pcm16_lineage_code_sha256),
        ("Wav2Lip child source", generator_child_sha256),
        ("SyncNet evaluator source", syncnet_evaluator_sha256),
        ("Stage-A runner source", stagea_runner_sha256),
    ):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"{label} SHA must be a 64-character hexadecimal digest")

    method_configs: list[dict[str, Any]] = [
        {"method_id": "identity", "kind": "identity", "config": {"method": "identity", "sample_rate_hz": 16000}},
        {
            "method_id": "rms_fixed",
            "kind": "rms_fixed",
            "target_rms": RMS_TARGET,
            "target_rms_tolerance": 1e-7,
            "config": {"method": "rms", "target_rms": RMS_TARGET, "sample_rate_hz": 16000, "peak_ceiling": None},
        },
        {
            "method_id": "rms_localcompress",
            "kind": "bounded_local_compression_candidate",
            "config": {
                "method": "rms_then_localcompress",
                "sample_rate_hz": 16000,
                "pre_rms_target": RMS_TARGET,
                "local_window_ms": 32.0,
                "local_ratio": 0.8,
                "local_min_gain_db": -6.0,
                "local_max_gain_db": 6.0,
                "envelope_floor": 1e-5,
                "post_rms_target": RMS_TARGET,
            },
        },
    ]
    methods = []
    for item in method_configs:
        methods.append({
            "method_id": item["method_id"],
            "kind": item["kind"],
            "config": item["config"],
            "config_sha256": _sha_bytes(_canonical(item["config"])),
            "checkpoint_sha256": None,
            **({"target_rms": item["target_rms"], "target_rms_tolerance": item["target_rms_tolerance"]}
               if item["kind"] == "rms_fixed" else {}),
        })

    w2l_sources = {
        "audio.py": "71d5e927cac3d0b7103e7671a43b82b41b6a56398a092bbfa2e96d3041249c4c",
        "hparams.py": "4a4c31dc51ebd8140da18ac24841e582b2472389f7c65e7f0a6b4447fc0c71cf",
        "inference.py": "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6",
        "models/wav2lip.py": "a02b54288f8200332ee4f0ea0c8013bca542b94d8149e514ac25a742a7a70e66",
    }
    syncnet_sources = {
        "SyncNetInstance.py": "bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6",
        "SyncNetModel.py": "41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22",
        "run_pipeline.py": "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf",
        "run_syncnet.py": "c78fe9f75665aa9e7a2bb523e4e7e3734a857137ff70b29c88a7b4893c422ded",
    }
    model_code_sha = _sha_bytes(_canonical({"git_head": W2L_HEAD, "source_sha256": w2l_sources}))
    w2l_preprocess_sha = _sha_bytes(_canonical({
        "git_head": W2L_HEAD,
        "audio_sha256": w2l_sources["audio.py"],
        "hparams_sha256": w2l_sources["hparams.py"],
        "official_load": "librosa.load(path,sr=16000); mono default; no pre-mel peak normalization",
        "mel_chunk": "80 mel bins,16 steps; floor(frame*80/25),repeat final tail",
    }))
    syncnet_preprocess_sha = _sha_bytes(_canonical({
        "git_head": SYNCNET_HEAD,
        "source_sha256": syncnet_sources,
        "scorer_audio": "canonical natural float32 WAV full-source FFmpeg s16le decode then exact selected-track source-clock sample slice; fresh PCM16/crop-video Matroska passed to SyncNetInstance.evaluate",
        "pcm16_lineage_code_sha256": pcm16_lineage_code_sha256,
        "syncnet_evaluator_sha256": syncnet_evaluator_sha256,
        "crop": {"facedet_scale": 0.25, "crop_scale": 0.40, "min_track": 5, "frame_rate": 25, "num_failed_det": 25, "min_face_size": 100},
    }))

    face = {
        "path": remote_face_path,
        "file_sha256": FACE_SHA256,
        "role": "shared_static_portrait_input",
    }
    source_video = {
        "path": remote_portrait_video_path,
        "file_sha256": portrait_video_sha256,
        "fps_num": 25,
        "fps_den": 1,
        "role": "static_portrait_reference_video",
    }
    model = {
        "model_id": MODEL_ID,
        "audio_frontend": "wav2lip_official_16k_float32_mel",
        "role": "candidate",
        "model_sha256": W2L_CHECKPOINT_SHA256,
        "code_sha256": model_code_sha,
        "score_model_sha256": SYNCNET_CHECKPOINT_SHA256,
        "preprocessing_sha256": syncnet_preprocess_sha,
        "generator_audio_preprocessing_sha256": w2l_preprocess_sha,
        "fps_num": 25,
        "fps_den": 1,
        "git_head": W2L_HEAD,
        "source_sha256": w2l_sources,
        "generator": {
            "implementation": "official Wav2Lip inference.py child; seeded Python/NumPy/Torch; static primary_face",
            "python": "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python",
            "repo": "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip",
            "checkpoint": "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/checkpoints/wav2lip_gan.pth",
            "checkpoint_sha256": W2L_CHECKPOINT_SHA256,
            "face_path": remote_face_path,
            "fps_num": 25,
            "fps_den": 1,
            "child_script_sha256": generator_child_sha256,
            "runner_sha256": stagea_runner_sha256,
        },
    }

    rows = []
    sample_rows = manifest.get("rows")
    if not isinstance(sample_rows, list) or len(sample_rows) != 20:
        raise ValueError("the frozen dev PCM manifest must contain exactly twenty rows")
    for sample in sample_rows:
        sid, sample_id = sample["speaker_id"], sample["sample_id"]
        if sid not in protocol["dev_speaker_ids"]:
            raise ValueError(f"unfrozen dev speaker encountered: {sid}")
        audio_path = f"{remote_audio_root.rstrip('/')}/{sid}/{sample_id}.wav"
        natural_ref = _ref(audio_path, sample["natural_pcm"]["file_sha256"], sample["natural_pcm"]["pcm_sha256"], sample["natural_pcm"]["sample_count"])
        for seed in SEEDS:
            for method in methods:
                mid = method["method_id"]
                rows.append({
                    "row_id": f"dev_{sid}_{sample_id}_{MODEL_ID}_{FACE_ID}_{seed}_{mid}",
                    "split": "dev",
                    "sample_id": sample_id,
                    "speaker_id": sid,
                    "model_id": MODEL_ID,
                    "audio_frontend": model["audio_frontend"],
                    "model_role": model["role"],
                    "face_id": FACE_ID,
                    "seed": seed,
                    "method_id": mid,
                    "method_config_sha256": method["config_sha256"],
                    "checkpoint_sha256": None,
                    "historical_exposed": bool(sample.get("historical_exposed", False)),
                    "natural_pcm": natural_ref,
                    "score_audio": dict(natural_ref),
                    "source_face": face,
        "source_video": source_video,
        "generation_settings": {
                        "generator": "official_wav2lip_inference.py",
                        "input_audio": "enhanced_generator_condition_only",
                        "face_mode": "static_image",
                        "seed": seed,
                        "fps_num": 25,
                        "fps_den": 1,
                        "pre_mel_peak_normalization": False,
                    },
                })

    first_identity = next(row for row in rows if row["method_id"] == "identity" and row["seed"] == SEEDS[0])
    repeat_protocol = {
        "dev_seed_ids": SEEDS,
        "same_seed_repeat": True,
        "comparison_fields": ["decoded_rgb_frames", "timestamp_map", "query_map", "distance_matrix", "Cbest"],
    }
    contrasts = [
        {"contrast_id": "rms_fixed_minus_identity", "method_a": "rms_fixed", "method_b": "identity"},
        {"contrast_id": "rms_localcompress_minus_identity", "method_a": "rms_localcompress", "method_b": "identity"},
        {"contrast_id": "rms_localcompress_minus_rms_fixed", "method_a": "rms_localcompress", "method_b": "rms_fixed"},
    ]
    endpoints = []
    for contrast in contrasts:
        for support_id in ("full", "guard20"):
            endpoints.append({
                "endpoint_id": f"dev_{contrast['contrast_id']}_{support_id}_Cbest",
                "split": "dev",
                "contrast_id": contrast["contrast_id"],
                "support_id": support_id,
                "metric": "Cbest",
                "model_scope": "model",
                "model_id": MODEL_ID,
                "roi_estimand": "native_official",
                "audio_estimand": "native_roi_original_pcm16",
                "exploratory": True,
            })

    plan: dict[str, Any] = {
        "schema_version": "cross-tfg-audio-eval-v1",
        "run_id": "cross_tfg_audio_head_20261003_stagea_wav2lip_dev_attempt01",
        "status": "FROZEN",
        "git_revision": "local-uncommitted-stagea-executor; see stageA source SHA in result manifest",
        "split_manifest_sha256": manifest_sha,
        "config_sha256": _sha_file(config_path),
        "fit_speaker_ids": list(protocol["fit_speaker_ids"]),
        "dev_speaker_ids": list(protocol["dev_speaker_ids"]),
        "confirm_speaker_ids": [],
        "historical_exposed_speaker_ids": list(protocol["historically_exposed_speaker_ids"]),
        "historical_exposed_sample_ids": [],
        "historical_exposure_source": {
            "protocol_path": str(protocol_path),
            "protocol_raw_sha256": _sha_file(protocol_path),
            "source_roster": "data-protocol.json historical exposure list",
        },
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["dataset_revision"],
        "source_rows_manifest": {
            "raw_sha256": manifest_sha,
            "payload_sha256": manifest.get("payload_sha256", "f5796ebc3c280a14d077323101fc8bcea313037e55f64134002b472f0cafa3bd"),
            "independent_audit": "276 checks, 0 failures; audit receipt SHA 958c821b3f3907241d72588933e9ee9f14c837f2570f73d9630c8c303e0272d9",
        },
        "generation_seed_family": {"fit": [], "dev": SEEDS, "confirm": []},
        "models": [model],
        "methods": methods,
        "rows": rows,
        "generation_repeat_qc": [{
            "model_id": MODEL_ID,
            "split": "dev",
            "method_id": "identity",
            "deterministic": False,
            "reference_row_id": first_identity["row_id"],
            "input_sample_id": first_identity["sample_id"],
            "face_id": FACE_ID,
            "seed": SEEDS[0],
            "followup_protocol": {
                **repeat_protocol,
                "config_sha256": _sha_bytes(_canonical(repeat_protocol)),
            },
        }],
        "supports": [
            {"support_id": "full", "trim_each_end": 0, "minimum_queries": 1, "primary": True},
            {"support_id": "guard20", "trim_each_end": 20, "minimum_queries": 41, "primary": False},
        ],
        "protocol": {
            "audio": {"sample_rate_hz": 16000, "channels": 1, "dtype": "float32_le", "resampling": "AISHELL source already mono/16k; no resampling"},
            "syncnet": {
                "score_model_id": "syncnet_v2",
                "score_model_sha256": SYNCNET_CHECKPOINT_SHA256,
                "source_sha256": syncnet_sources,
                "fps_num": 25,
                "fps_den": 1,
                "lag_offsets_frames": list(range(-15, 16)),
                "zero_lag_column": 15,
                "anchor_column": 15,
                "best_lag_definition": "lag_offsets_frames[best_column]",
                "official_offset_definition": "negative_best_lag_frames",
                "metric_atol": 2e-6,
                "metric_rtol": 2e-6,
                "score_audio_estimand": "native_roi_original_pcm16",
                "primary_audio_input": {
                    "mode": "canonical_natural_pcm16_crop_window",
                    "source": "natural_pcm",
                    "codec": "pcm_s16le",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "operation": "ffmpeg_decode_full_source_then_slice_sample_window",
                    "frame_clock": "original_render_source_pts",
                    "ffmpeg_sha256": ffmpeg_sha256.lower(),
                    "code_sha256": pcm16_lineage_code_sha256.lower(),
                    "argv_template": [
                        "-v", "error", "-i", "{natural_pcm_path}", "-map", "0:a:0", "-vn",
                        "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
                    ],
                },
                "score_reduction": {
                    "backend": "pytorch_cpu",
                    "torch_version": "2.5.1+cu124",
                    "num_threads": 1,
                    "matrix_layout": "query_by_lag_float32",
                    "stacked_layout": "lag_by_query_contiguous_float32",
                    "curve_operation": "torch.mean(torch.stack(query_distance_vectors, dim=1), dim=1)",
                    "median_operation": "torch.median(curve)",
                    "minimum_operation": "torch.min(curve, dim=0)",
                    "contrast_dtype": "float32",
                    "median_semantics": "lower_middle_order_statistic",
                },
            },
            "roi": {
                "mode": "native_official",
                "estimand": "native_roi_original_pcm16",
                "facedet_scale": 0.25,
                "crop_scale": 0.40,
                "min_track": 5,
                "frame_rate": 25,
                "num_failed_det": 25,
                "min_face_size": 100,
                "selected_track_rule": "longest_contiguous_track_then_lowest_original_track_index",
                "query_frame_map": "crop-local evaluator query row i maps through selected_track_crop.frame_map to source_clock_frame_index on the nominal original 25-fps clock; render_frame_index may be null for nominal clock rows outside decoded render",
            },
            "mux": {"container": "matroska", "audio_codec": "pcm_f32le", "audio_start_time_s": 0.0, "audio_start_time_tolerance_s": 0.02},
            "generator": {
                "model_id": MODEL_ID,
                "face_id": FACE_ID,
                "official_wav2lip_git_head": W2L_HEAD,
                "wav2lip_checkpoint_sha256": W2L_CHECKPOINT_SHA256,
                "audio_preprocessing_sha256": w2l_preprocess_sha,
                "score_audio_preprocessing_sha256": syncnet_preprocess_sha,
                "child_script_sha256": generator_child_sha256.lower(),
                "stagea_runner_sha256": stagea_runner_sha256.lower(),
                "face_input": "same pinned primary static portrait for all dev rows/methods/seeds",
                "renderer_settings": {"fps_num": 25, "fps_den": 1, "static_image": True, "pre_mel_peak_normalization": False},
                "source_video_role": "shared static portrait reference; Wav2Lip consumes source_face image directly",
            },
        },
        "contrasts": contrasts,
        "analysis": {
            "contrasts": contrasts,
            "summary_endpoints": endpoints,
            "simultaneous_family_ids": [],
            "simultaneous_family_status": "NOT_APPLICABLE",
            "bootstrap": {"bit_generator": "PCG64", "seed": 2026100301, "draw_count": 10000, "confidence_level": 0.95, "unit": "speaker_equal_weight"},
            "aggregation": {"seed_within_sample": "equal_weight", "samples_within_speaker": "equal_weight", "speakers": "equal_weight", "exploratory_dev_only": True},
        },
        "execution": {
            "stage": "A",
            "scope": "dev-only; no confirmation audio/effects read",
            "planned_rows": 180,
            "primary_score": "native_roi_original_pcm16",
            "native_mp3_score": "diagnostic only",
            "candidate_selection": "development_only_after_all_predeclared_rows_and_consumer_audit",
            "stagea_runner_sha256": stagea_runner_sha256.lower(),
            "primary_pcm16_lineage_code_sha256": pcm16_lineage_code_sha256.lower(),
            "official_syncnet_evaluator_wrapper_sha256": syncnet_evaluator_sha256.lower(),
        },
    }
    plan["plan_sha256"] = _sealed_hash(plan, "plan_sha256")
    if len(rows) != 180:
        raise ValueError(f"unexpected Stage-A row count: {len(rows)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty plan output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_exclusive(output_dir / "plan.json", (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    rows_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
    _write_exclusive(output_dir / "planned_rows.jsonl", rows_bytes)
    seal = {
        "schema": "cross_tfg_stagea_plan_build_v1",
        "run_id": plan["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "plan_raw_sha256": _sha_file(output_dir / "plan.json"),
        "planned_rows_sha256": _sha_file(output_dir / "planned_rows.jsonl"),
        "planned_rows": len(rows),
        "method_counts": {method["method_id"]: sum(row["method_id"] == method["method_id"] for row in rows) for method in methods},
        "sample_count": len(sample_rows),
        "seed_family": SEEDS,
        "confirm_speaker_ids_in_plan": len(plan["confirm_speaker_ids"]),
        "confirm_audio_read": False,
        "scores_read": False,
        "portrait_video_sha256": portrait_video_sha256,
    }
    seal["manifest_sha256"] = _sealed_hash(seal, "manifest_sha256")
    _write_exclusive(output_dir / "plan_manifest.json", (json.dumps(seal, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return seal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-audio-root", required=True)
    parser.add_argument("--remote-face-path", required=True)
    parser.add_argument("--remote-portrait-video-path", required=True)
    parser.add_argument("--portrait-video-sha256", required=True)
    parser.add_argument("--ffmpeg-sha256", required=True)
    parser.add_argument("--pcm16-lineage-code-sha256", required=True)
    parser.add_argument("--generator-child-sha256", required=True)
    parser.add_argument("--syncnet-evaluator-sha256", required=True)
    parser.add_argument("--stagea-runner-sha256", required=True)
    args = parser.parse_args()
    result = build_plan(
        args.dev_manifest.resolve(strict=True),
        args.protocol.resolve(strict=True),
        args.config.resolve(strict=True),
        args.output_dir.resolve(),
        remote_audio_root=args.remote_audio_root,
        remote_face_path=args.remote_face_path,
        remote_portrait_video_path=args.remote_portrait_video_path,
        portrait_video_sha256=args.portrait_video_sha256.lower(),
        ffmpeg_sha256=args.ffmpeg_sha256.lower(),
        pcm16_lineage_code_sha256=args.pcm16_lineage_code_sha256.lower(),
        generator_child_sha256=args.generator_child_sha256.lower(),
        syncnet_evaluator_sha256=args.syncnet_evaluator_sha256.lower(),
        stagea_runner_sha256=args.stagea_runner_sha256.lower(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
